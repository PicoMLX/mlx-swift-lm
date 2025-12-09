// Copyright © 2024 Apple Inc.
//
// Batched text generation inspired by the Python implementation at
// `mlx_lm/generate.py` (BatchGenerator, Batch, BatchStats, batch_generate).

import Foundation
import MLX
import Tokenizers

/// Statistics gathered for batched generation.
public struct BatchGenerateStats: Sendable {

    public var promptTokenCount: Int
    public var promptTime: TimeInterval
    public var generationTokenCount: Int
    public var generationTime: TimeInterval
    public var peakMemoryGB: Double

    public init(
        promptTokenCount: Int = 0,
        promptTime: TimeInterval = 0,
        generationTokenCount: Int = 0,
        generationTime: TimeInterval = 0,
        peakMemoryGB: Double = 0
    ) {
        self.promptTokenCount = promptTokenCount
        self.promptTime = promptTime
        self.generationTokenCount = generationTokenCount
        self.generationTime = generationTime
        self.peakMemoryGB = peakMemoryGB
    }

    public var promptTokensPerSecond: Double {
        guard promptTime > 0 else { return 0 }
        return Double(promptTokenCount) / promptTime
    }

    public var generationTokensPerSecond: Double {
        guard generationTime > 0 else { return 0 }
        return Double(generationTokenCount) / generationTime
    }
}

/// Configuration parameters for batched generation.
public struct BatchGenerateParameters: Sendable {

    /// Default maximum tokens to generate per prompt.
    public var defaultMaxTokens: Int

    /// Maximum number of completions to keep active simultaneously.
    public var completionBatchSize: Int

    /// Number of prompts to prefill at a time.
    public var prefillBatchSize: Int

    /// Number of prompt tokens to process per prefill step.
    public var prefillStepSize: Int

    /// Underlying single-sequence generation parameters (sampling, quantization, etc.).
    public var generation: GenerateParameters

    /// Whether to return log probabilities for each generated token.
    public var returnLogProbs: Bool

    public init(
        maxTokens: Int = 128,
        completionBatchSize: Int = 32,
        prefillBatchSize: Int = 8,
        prefillStepSize: Int = 2_048,
        generation: GenerateParameters = .init(),
        returnLogProbs: Bool = false
    ) {
        self.defaultMaxTokens = maxTokens
        self.completionBatchSize = completionBatchSize
        self.prefillBatchSize = prefillBatchSize
        self.prefillStepSize = prefillStepSize

        var generationParameters = generation
        if generationParameters.maxTokens == nil {
            generationParameters.maxTokens = maxTokens
        }
        self.generation = generationParameters
        self.returnLogProbs = returnLogProbs
    }
}

/// Result payload returned from `batchGenerate`.
public struct BatchGenerateResult: Sendable {

    public struct Sequence: Sendable {
        public let uid: Int
        public let tokens: [Int]
        public let text: String
        public let finishReason: FinishReason
        public let logProbs: MLXArray?

        public init(
            uid: Int,
            tokens: [Int],
            text: String,
            finishReason: FinishReason,
            logProbs: MLXArray? = nil
        ) {
            self.uid = uid
            self.tokens = tokens
            self.text = text
            self.finishReason = finishReason
            self.logProbs = logProbs
        }
    }

    public enum FinishReason: String, Sendable {
        case stop
        case length
        case unspecified
    }

    public let sequences: [Sequence]
    public let stats: BatchGenerateStats

    public init(sequences: [Sequence], stats: BatchGenerateStats) {
        self.sequences = sequences
        self.stats = stats
    }
}

fileprivate struct LogitProcessorBox: Sendable {
    private var processor: any LogitProcessor

    init(_ processor: any LogitProcessor) {
        self.processor = processor
    }

    mutating func prompt(_ prompt: MLXArray) {
        processor.prompt(prompt)
    }

    mutating func process(logits: MLXArray) -> MLXArray {
        processor.process(logits: logits)
    }

    mutating func didSample(token: MLXArray) {
        processor.didSample(token: token)
    }
}

fileprivate struct PendingPrompt {
    let uid: Int
    let tokens: [Int]
    let maxTokens: Int
    let sampler: LogitSampler
    var processor: LogitProcessorBox?

    var length: Int { tokens.count }
}

fileprivate final class ActiveBatch {
    var uids: [Int]
    var tokens: MLXArray
    /// Optional cache of selected logits [B, vocab] from the previous step.
    /// Present only when returnLogProbs == true to avoid needless bandwidth.
    var prevSelectedLogits: MLXArray?
    var maxTokens: [Int]
    var numTokens: [Int]
    var cache: [KVCache]
    var samplers: [LogitSampler]
    var processors: [LogitProcessorBox?]

    var count: Int { uids.count }

    init(
        uids: [Int],
        tokens: MLXArray,
        prevSelectedLogits: MLXArray?,
        maxTokens: [Int],
        numTokens: [Int],
        cache: [KVCache],
        samplers: [LogitSampler],
        processors: [LogitProcessorBox?]
    ) {
        self.uids = uids
        self.tokens = tokens
        self.prevSelectedLogits = prevSelectedLogits
        self.maxTokens = maxTokens
        self.numTokens = numTokens
        self.cache = cache
        self.samplers = samplers
        self.processors = processors
    }

    func filter(keeping indices: [Int]) {
        let keepIndices = MLXArray(indices.map(Int32.init))
        uids = indices.map { uids[$0] }
        maxTokens = indices.map { maxTokens[$0] }
        numTokens = indices.map { numTokens[$0] }
        samplers = indices.map { samplers[$0] }
        processors = indices.map { processors[$0] }

        tokens = tokens[keepIndices]
        if let lp = prevSelectedLogits {
            prevSelectedLogits = lp[keepIndices, .ellipsis]
        }
        
        for i in cache.indices {
            switch cache[i] {
            case let batch as BatchKVCache:
                batch.filter(batchIndices: keepIndices)
            case let arrays as ArraysCache:
                arrays.filter(batchIndices: keepIndices)
            case let rotating as BatchRotatingKVCache:
                rotating.filter(batchIndices: keepIndices)
            case let list as CacheList:
                list.filter(batchIndices: keepIndices)
            default:
                fatalError("\(type(of: cache[i])) does not support batched filtering")
            }
        }
    }

    func extend(_ other: ActiveBatch) {
        uids.append(contentsOf: other.uids)
        maxTokens.append(contentsOf: other.maxTokens)
        numTokens.append(contentsOf: other.numTokens)
        samplers.append(contentsOf: other.samplers)
        processors.append(contentsOf: other.processors)

        tokens = MLX.concatenated([tokens, other.tokens])
        switch (prevSelectedLogits, other.prevSelectedLogits) {
        case let (lhs?, rhs?):
            prevSelectedLogits = MLX.concatenated([lhs, rhs])
        case (nil, nil):
            prevSelectedLogits = nil
        default:
            fatalError("ActiveBatch.extend: returnLogProbs presence mismatch between batches")
        }

        for i in cache.indices {
            switch (cache[i], other.cache[i]) {
            case (let lhs as BatchKVCache, let rhs as BatchKVCache):
                lhs.extend(other: rhs)
            case (let lhs as ArraysCache, let rhs as ArraysCache):
                lhs.extend(other: rhs)
            case (let lhs as BatchRotatingKVCache, let rhs as BatchRotatingKVCache):
                lhs.extend(other: rhs)
            case (let lhs as CacheList, let rhs as CacheList):
                lhs.extend(other: rhs)
            default:
                fatalError("\(type(of: cache[i])) does not support batched extension")
            }
        }
    }
}

/// Iterator that mirrors the lifecycle of Python's `BatchGenerator`.
public struct BatchTokenIterator: Sequence, IteratorProtocol {

    public struct Response: Sendable {
        public let uid: Int
        public let token: Int
        public let logProbs: MLXArray?
        public let finishReason: BatchGenerateResult.FinishReason?
        public init(
            uid: Int,
            token: Int,
            logProbs: MLXArray?,
            finishReason: BatchGenerateResult.FinishReason?
        ) {
            self.uid = uid
            self.token = token
            self.logProbs = logProbs
            self.finishReason = finishReason
        }
    }

    private var model: any LanguageModel
    private let parameters: BatchGenerateParameters
    private let defaultSampler: LogitSampler
    private let kvBits: Int?
    private let kvGroupSize: Int
    private let quantizedKVStart: Int
    private let stopTokens: Set<Int>
    private let unknownTokenId: Int?
    private let returnLogProbs: Bool

    private var unprocessedPrompts: [PendingPrompt] = []
    private var activeBatch: ActiveBatch?
    private var uidCounter: Int = 0

    private var promptTokenCount = 0
    private var promptTime: TimeInterval = 0
    private var generationTokenCount = 0
    private var generationTime: TimeInterval = 0
    
    // MARK: - Observability Hooks
    
    /// Called once per prefill chunk before processing it.
    /// `promptCount` is the number of prompts in this chunk.
    /// `tokenCount` is the total number of tokens across all prompts in this chunk.
    public var onPrefillStart: (@Sendable (_ promptCount: Int, _ tokenCount: Int) -> Void)?
    
    /// Called once per prefill chunk after processing it.
    /// `elapsed` is wall-clock seconds spent in `processPrompts(chunk)`.
    /// `tokenCount` is the total number of tokens prefilled in this chunk.
    public var onPrefillComplete: (@Sendable (_ elapsed: TimeInterval, _ tokenCount: Int) -> Void)?
    
    /// Number of prompts waiting to be prefilled.
    public var pendingPrefillCount: Int { unprocessedPrompts.count }
    
    /// Number of sequences currently in the active decode batch.
    public var activeBatchCount: Int { activeBatch?.count ?? 0 }

    public init(
        model: any LanguageModel,
        parameters: BatchGenerateParameters,
        stopTokens: Set<Int>,
        unknownTokenId: Int?
    ) {
        self.model = model
        self.parameters = parameters
        self.defaultSampler = parameters.generation.sampler()
        self.kvBits = parameters.generation.kvBits
        self.kvGroupSize = parameters.generation.kvGroupSize
        self.quantizedKVStart = parameters.generation.quantizedKVStart
        self.stopTokens = stopTokens
        self.unknownTokenId = unknownTokenId
        self.returnLogProbs = parameters.returnLogProbs
    }

    @discardableResult
    public mutating func insert(
        prompts: [[Int]],
        maxTokens: [Int]? = nil,
        samplers: [LogitSampler]? = nil,
        processors: [LogitProcessor?]? = nil
    ) -> [Int] {
        guard maxTokens == nil || maxTokens!.count == prompts.count else {
            fatalError("maxTokens.count must match prompts.count")
        }

        if let samplers, samplers.count != prompts.count {
            fatalError("samplers.count must match prompts.count")
        }

        if let processors, processors.count != prompts.count {
            fatalError("processors.count must match prompts.count")
        }

        let resolvedMaxTokens: [Int] = maxTokens ?? Array(
            repeating: parameters.defaultMaxTokens, count: prompts.count)

        var assigned = [Int]()
        for (index, element) in prompts.enumerated() {
            let tokens = element
            let limit = resolvedMaxTokens[index]
            let uid = uidCounter
            uidCounter += 1
            let sampler = samplers?[index] ?? defaultSampler
            var processorBox: LogitProcessorBox?
            if let processor = processors?[index] ?? parameters.generation.processor() {
                processorBox = LogitProcessorBox(processor)
            }
            unprocessedPrompts.append(
                .init(uid: uid, tokens: tokens, maxTokens: limit, sampler: sampler, processor: processorBox)
            )
            assigned.append(uid)
        }

        unprocessedPrompts.sort { $0.length < $1.length }
        return assigned
    }

    public mutating func remove(uids: some Sequence<Int>) {
        let removal = Set(uids)
        guard !removal.isEmpty else { return }

        unprocessedPrompts.removeAll { removal.contains($0.uid) }

        if let current = activeBatch {
            let keep = current.uids.enumerated().compactMap { index, uid in
                removal.contains(uid) ? nil : index
            }

            if keep.count != current.uids.count {
                if keep.isEmpty {
                    activeBatch = nil
                } else {
                    current.filter(keeping: keep)
                    activeBatch = current
                }
            }
        }
    }

    public mutating func adoptExisting(
        caches: [KVCache],
        nextTokens: MLXArray,
        sampler: LogitSampler,
        processor: (any LogitProcessor)?,
        maxTokens: Int,
        generatedTokens: Int,
        promptTokenCount promptCount: Int
    ) -> Int {
        let uid = uidCounter
        uidCounter += 1

        let convertedCaches = caches.map { convertSoloCache($0, leftPadding: [0]) }

        let processors: [LogitProcessorBox?]
        if let processor {
            processors = [LogitProcessorBox(processor)]
        } else {
            processors = [nil]
        }

        activeBatch = ActiveBatch(
            uids: [uid],
            tokens: nextTokens,
            prevSelectedLogits: nil,
            maxTokens: [maxTokens],
            numTokens: [generatedTokens],
            cache: convertedCaches,
            samplers: [sampler],
            processors: processors
        )

        promptTokenCount += promptCount
        generationTokenCount += generatedTokens

        return uid
    }

    public mutating func next() -> [Response]? {
        if activeBatch == nil && unprocessedPrompts.isEmpty {
            return nil
        }

        var batch = activeBatch
        var promptProcessing = false
        var tic = Date.timeIntervalSinceReferenceDate
        let prefillBatchSize = Swift.max(parameters.prefillBatchSize, 1)

        var numActive = batch?.count ?? 0
        var numToAdd = Swift.max(parameters.completionBatchSize - numActive, 0)

        while numToAdd >= prefillBatchSize {
            let chunk = Array(unprocessedPrompts.prefix(prefillBatchSize))

            if chunk.isEmpty {
                if numActive > 0 {
                    break
                } else {
                    activeBatch = nil
                    return []
                }
            }

            if let existing = batch, !promptProcessing {
                if let lp = existing.prevSelectedLogits {
                    eval(existing.tokens, lp)
                } else {
                    eval(existing.tokens)
                }
                let now = Date.timeIntervalSinceReferenceDate
                generationTime += now - tic
                tic = now
            }

            // Calculate token count for this chunk
            let chunkTokenCount = chunk.reduce(0) { $0 + $1.tokens.count }
            
            // Prefill callback - notify before starting
            onPrefillStart?(chunk.count, chunkTokenCount)
            let prefillStart = Date.timeIntervalSinceReferenceDate
            
            let newBatch = processPrompts(chunk)
            
            // Prefill callback - notify after completion
            let prefillElapsed = Date.timeIntervalSinceReferenceDate - prefillStart
            onPrefillComplete?(prefillElapsed, chunkTokenCount)
            
            unprocessedPrompts.removeFirst(chunk.count)
            promptProcessing = true

            if let current = batch {
                current.extend(newBatch)
            } else {
                batch = newBatch
            }

            numActive = batch?.count ?? 0
            numToAdd = Swift.max(parameters.completionBatchSize - numActive, 0)
        }

        guard let currentBatch = batch ?? activeBatch else {
            return []
        }

        // Store references to previous iteration's results
        let previousTokens = currentBatch.tokens
        let previousSelected = currentBatch.prevSelectedLogits // may be nil

        // Compute next iteration's tokens asynchronously
        // Mutate cache in-place to avoid copy-on-write overhead
        var processors = currentBatch.processors
        let (nextTokens, nextSelected) = step(
            inputTokens: previousTokens[0..., .newAxis],
            cache: &currentBatch.cache,
            samplers: currentBatch.samplers,
            processors: &processors)
        currentBatch.processors = processors
        if let s = nextSelected { asyncEval(nextTokens, s) } else { asyncEval(nextTokens) }

        // Materialize tokens to CPU - this synchronizes with GPU
        // Match Python: y.tolist() converts entire array at once
        let tokenArray = previousTokens.asArray(Int.self)

        // Measure time AFTER materialization (matches Python: y.tolist() then perf_counter())
        let toc = Date.timeIntervalSinceReferenceDate

        // Python: if prompt_processing: prompt_time += toc - tic else: generation_time += toc - tic
        if promptProcessing {
            promptTime += toc - tic
        } else {
            generationTime += toc - tic
        }

        var responses: [Response] = []
        var keepIndices: [Int] = []
        var endIndices: [Int] = []
        
        responses.reserveCapacity(tokenArray.count)
        keepIndices.reserveCapacity(tokenArray.count)
        endIndices.reserveCapacity(4)

        for idx in 0 ..< tokenArray.count {
            var finish: BatchGenerateResult.FinishReason? = nil
            let token = tokenArray[idx]

            currentBatch.numTokens[idx] += 1

            if stopTokens.contains(token) {
                finish = .stop
                endIndices.append(idx)
            } else if currentBatch.numTokens[idx] >= currentBatch.maxTokens[idx] {
                finish = .length
                endIndices.append(idx)
            } else {
                keepIndices.append(idx)
            }

            let logProbRow: MLXArray?
            if returnLogProbs, let sel = previousSelected {
                logProbRow = normalizeLogProbsRow(sel[idx])
            } else {
                logProbRow = nil
            }
            responses.append(Response(
                uid: currentBatch.uids[idx],
                token: token,
                logProbs: logProbRow,
                finishReason: finish))
        }

        currentBatch.tokens = nextTokens
        currentBatch.prevSelectedLogits = nextSelected

        generationTokenCount += responses.count

        if !endIndices.isEmpty {
            if keepIndices.isEmpty {
                activeBatch = nil
            } else {
                currentBatch.filter(keeping: keepIndices)
                activeBatch = currentBatch
            }
        } else {
            activeBatch = currentBatch
        }

        return responses
    }

    public func stats() -> BatchGenerateStats {
        BatchGenerateStats(
            promptTokenCount: promptTokenCount,
            promptTime: promptTime,
            generationTokenCount: generationTokenCount,
            generationTime: generationTime,
            peakMemoryGB: Double(Memory.peakMemory) / 1_000_000_000.0
        )
    }

    private mutating func processPrompts(_ prompts: [PendingPrompt]) -> ActiveBatch {
        let tokens = prompts.map(\.tokens)
        let lengths = tokens.map(\.count)

        promptTokenCount += lengths.reduce(0, +)

        let (padded, leftPadding) = leftPadPrompts(tokens)

        var cache = makeBatchCache(
            model: model, parameters: parameters, leftPadding: leftPadding)

        var processors = prompts.map(\.processor)
        for index in processors.indices {
            if var processor = processors[index] {
                processor.prompt(MLXArray(tokens[index]))
                processors[index] = processor
            }
        }

        var remaining = padded
        while remaining.dim(1) > 1 {
            let nToProcess = Swift.min(parameters.prefillStepSize, remaining.dim(1) - 1)
            let slice = remaining[0..., ..<nToProcess]
            _ = model(slice, cache: cache)
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: kvBits,
                kvGroupSize: kvGroupSize,
                quantizedKVStart: quantizedKVStart)
            let states = cache.flatMap(\.state)
            if !states.isEmpty {
                eval(states)
            }
            let length = remaining.dim(1)
            remaining = remaining[0..., nToProcess..<length]
            Memory.clearCache()
        }

        let samplers = prompts.map(\.sampler)

        let (y, maybeSelected) = step(
            inputTokens: remaining,
            cache: &cache,
            samplers: samplers,
            processors: &processors)
        if let s = maybeSelected { asyncEval(y, s) } else { asyncEval(y) }

        return ActiveBatch(
            uids: prompts.map(\.uid),
            tokens: y,
            prevSelectedLogits: maybeSelected,
            maxTokens: prompts.map(\.maxTokens),
            numTokens: Array(repeating: 0, count: prompts.count),
            cache: cache,
            samplers: samplers,
            processors: processors)
    }

    private mutating func step(
        inputTokens: MLXArray,
        cache: inout [KVCache],
        samplers: [LogitSampler],
        processors: inout [LogitProcessorBox?]
    ) -> (MLXArray, MLXArray?) {
        let logits = model(inputTokens, cache: cache)
        let selected = logits[0..., -1, 0...]

        var sampledTokens = [Int32]()
        sampledTokens.reserveCapacity(samplers.count)

        for index in 0..<samplers.count {
            let sampler = samplers[index]
            var logitsRow = selected[index, .ellipsis]

            if var processor = processors[index] {
                logitsRow = processor.process(logits: logitsRow)
                processors[index] = processor
            }

            let sampledRow = sampler.sample(logits: logitsRow)

            if var processor = processors[index] {
                processor.didSample(token: sampledRow)
                processors[index] = processor
            }

            sampledTokens.append(sampledRow.item(Int32.self))
        }

        let sampled = MLXArray(sampledTokens)
        if returnLogProbs {
            return (sampled, selected)   // carry [B, vocab] only when needed
        } else {
            return (sampled, nil)
        }
    }

    private func normalizeLogProbsRow(_ row: MLXArray) -> MLXArray {
        let logSum = logSumExp(row, axis: row.ndim - 1, keepDims: true)
        return row - logSum
    }

    private func convertSoloCache(_ cache: KVCache, leftPadding: [Int]) -> KVCache {
        switch cache {
        case let simple as KVCacheSimple:
            let batch = BatchKVCache(leftPadding: leftPadding)
            let state = simple.state
            if state.count == 2 {
                _ = batch.update(keys: state[0], values: state[1])
            }
            return batch
        case let rotating as RotatingKVCache:
            guard let maxSize = rotating.maxSize else {
                fatalError("RotatingKVCache promotion requires a maxSize")
            }
            let batch = BatchRotatingKVCache(maxSize: maxSize, leftPadding: leftPadding)
            let state = rotating.state
            if state.count == 2 {
                _ = batch.update(keys: state[0], values: state[1])
            }
            batch.metaState = rotating.metaState
            return batch
        case let arrays as ArraysCache:
            let converted = ArraysCache(size: arrays.state.count, leftPadding: leftPadding)
            converted.state = arrays.state
            return converted
        case let list as CacheList:
            let convertedChildren = (0 ..< list.count).map { index in
                convertSoloCache(list[index], leftPadding: leftPadding)
            }
            return CacheList(convertedChildren)
        default:
            return cache
        }
    }
}

// MARK: - Sendable Conformance

/// BatchTokenIterator is marked @unchecked Sendable because:
/// - It holds non-Sendable types (model, caches)
/// - But we guarantee it's only mutated within DeviceEngine.runForward calls
/// - DeviceEngine serializes all access globally, preventing concurrent mutation
extension BatchTokenIterator: @unchecked Sendable {}

/// High-level API mirroring the Python `batch_generate` helper.
@discardableResult
public func batchGenerate(
    prompts: [String],
    maxTokens: [Int]? = nil,
    parameters: BatchGenerateParameters,
    context: ModelContext
) throws -> BatchGenerateResult {
    let tokenized: [[Int]] = prompts.map { context.tokenizer.encode(text: $0) }

    let defaultMaxTokens = maxTokens ?? Array(
        repeating: parameters.defaultMaxTokens, count: tokenized.count)
    guard defaultMaxTokens.count == tokenized.count else {
        fatalError("maxTokens.count must match prompts.count")
    }

    var stopTokens = Set<Int>()
    if let eos = context.tokenizer.eosTokenId {
        stopTokens.insert(eos)
    }
    for token in context.configuration.extraEOSTokens {
        if let id = context.tokenizer.convertTokenToId(token) {
            stopTokens.insert(id)
        }
    }

    var iterator = BatchTokenIterator(
        model: context.model,
        parameters: parameters,
        stopTokens: stopTokens,
        unknownTokenId: context.tokenizer.unknownTokenId)

    let uids = iterator.insert(prompts: tokenized, maxTokens: defaultMaxTokens)

    var generatedTokens = [Int: [Int]]()
    var logProbHistory: [Int: [MLXArray]]? = parameters.returnLogProbs ? [Int: [MLXArray]]() : nil
    var finishReasons = [Int: BatchGenerateResult.FinishReason]()

    while let responses = iterator.next(), !responses.isEmpty {
        for response in responses {
            if response.finishReason != .stop {
                generatedTokens[response.uid, default: []].append(response.token)
            }
            // Do NOT eval per token; keep device-resident to avoid sync stalls.
            if let lp = response.logProbs {
                if var dict = logProbHistory {
                    dict[response.uid, default: []].append(lp)
                    logProbHistory = dict
                }
            }
            if let finish = response.finishReason {
                finishReasons[response.uid] = finish
            }
        }
    }

    Stream().synchronize()

    let stats = iterator.stats()
    var sequences: [BatchGenerateResult.Sequence] = []
    sequences.reserveCapacity(uids.count)

    for uid in uids {
        let tokens = generatedTokens[uid] ?? []
        let text = context.tokenizer.decode(tokens: tokens)
        let finish = finishReasons[uid] ?? .unspecified
        let logProbRows = logProbHistory?[uid] ?? []
        let logProbs = logProbRows.isEmpty ? nil : MLX.stacked(logProbRows, axis: 0)

        sequences.append(
            .init(uid: uid, tokens: tokens, text: text, finishReason: finish, logProbs: logProbs))
    }

    return BatchGenerateResult(sequences: sequences, stats: stats)
}

// MARK: - Helpers

private func leftPadPrompts(_ prompts: [[Int]]) -> (MLXArray, [Int]) {
    guard let maxLength = prompts.map(\.count).max() else {
        return (MLXArray.zeros([0, 0], type: Int32.self), [])
    }

    var padded = [Int]()
    padded.reserveCapacity(prompts.count * maxLength)

    var leftPadding = [Int]()
    leftPadding.reserveCapacity(prompts.count)

    for prompt in prompts {
        let pad = maxLength - prompt.count
        leftPadding.append(pad)
        padded.append(contentsOf: Array(repeating: 0, count: pad))
        padded.append(contentsOf: prompt)
    }

    let array = MLXArray(padded, [prompts.count, maxLength])
    return (array, leftPadding)
}

private func makeBatchCache(
    model: any LanguageModel,
    parameters: BatchGenerateParameters,
    leftPadding: [Int]
) -> [KVCache] {
    let caches = model.newCache(parameters: parameters.generation)
    return caches.map {
        convertCache($0, parameters: parameters, leftPadding: leftPadding)
    }
}

private func convertCache(
    _ cache: KVCache,
    parameters: BatchGenerateParameters,
    leftPadding: [Int]
) -> KVCache {
    switch cache {
    case is BatchKVCache:
        return cache
    case is KVCacheSimple:
        return BatchKVCache(leftPadding: leftPadding)
    case let rotating as RotatingKVCache:
        guard rotating.keepTokens == 0 else {
            fatalError("RotatingKVCache with keep tokens is not supported in batched generation")
        }
        guard let window = rotating.maxSize else {
            fatalError("RotatingKVCache expected to have a max size for batching")
        }
        return BatchRotatingKVCache(maxSize: window, leftPadding: leftPadding)
    case let arrays as ArraysCache:
        arrays.setLeftPadding(leftPadding)
        return arrays
    case let list as CacheList:
        let converted = (0 ..< list.count).map {
            convertCache(list[$0], parameters: parameters, leftPadding: leftPadding)
        }
        return CacheList(converted)
    default:
        fatalError("\(type(of: cache)) does not yet support batching")
    }
}
