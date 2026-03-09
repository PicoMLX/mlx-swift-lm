// Copyright © 2025 Apple Inc.

import Foundation
import MLX

struct BatchExecutionOptions: Sendable, Equatable {
    let prefillStepSize: Int
    let maxKVSize: Int?
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int

    init(parameters: GenerateParameters) {
        prefillStepSize = parameters.prefillStepSize
        maxKVSize = parameters.maxKVSize
        kvBits = parameters.kvBits
        kvGroupSize = parameters.kvGroupSize
        quantizedKVStart = parameters.quantizedKVStart
    }
}

struct BatchIteratorConfiguration: Sendable {
    let completionBatchSize: Int
    let prefillBatchSize: Int
    let generation: GenerateParameters

    var batchOptions: BatchExecutionOptions {
        BatchExecutionOptions(parameters: generation)
    }
}

struct BatchLogitProcessorBox: Sendable {
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

private struct PendingPrompt {
    let uid: Int
    let tokens: [Int]
    let maxTokens: Int
    let sampler: any LogitSampler
    var processor: BatchLogitProcessorBox?

    var length: Int { tokens.count }
}

private final class ActiveBatch {
    var uids: [Int]
    var tokens: MLXArray
    var maxTokens: [Int]
    var numTokens: [Int]
    var cache: [KVCache]
    var samplers: [any LogitSampler]
    var processors: [BatchLogitProcessorBox?]

    var count: Int { uids.count }

    init(
        uids: [Int],
        tokens: MLXArray,
        maxTokens: [Int],
        numTokens: [Int],
        cache: [KVCache],
        samplers: [any LogitSampler],
        processors: [BatchLogitProcessorBox?]
    ) {
        self.uids = uids
        self.tokens = tokens
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

        for index in cache.indices {
            switch cache[index] {
            case let batch as BatchKVCache:
                batch.filter(batchIndices: keepIndices)
            case let rotating as BatchRotatingKVCache:
                rotating.filter(batchIndices: keepIndices)
            case let arrays as ArraysCache:
                arrays.filter(batchIndices: keepIndices)
            case let list as CacheList:
                list.filter(batchIndices: keepIndices)
            default:
                fatalError("\(type(of: cache[index])) does not support batched filtering")
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

        for index in cache.indices {
            switch (cache[index], other.cache[index]) {
            case (let lhs as BatchKVCache, let rhs as BatchKVCache):
                lhs.extend(other: rhs)
            case (let lhs as BatchRotatingKVCache, let rhs as BatchRotatingKVCache):
                lhs.extend(other: rhs)
            case (let lhs as ArraysCache, let rhs as ArraysCache):
                lhs.extend(other: rhs)
            case (let lhs as CacheList, let rhs as CacheList):
                lhs.extend(other: rhs)
            default:
                fatalError("\(type(of: cache[index])) does not support batched extension")
            }
        }
    }
}

struct BatchTokenIterator {
    struct Response: Sendable {
        let uid: Int
        let token: Int
        let stopReason: GenerateStopReason?
    }

    private var model: any LanguageModel
    private let configuration: BatchIteratorConfiguration
    private let stopTokens: Set<Int>
    private let unknownTokenId: Int?

    private var unprocessedPrompts: [PendingPrompt] = []
    private var activeBatch: ActiveBatch?

    var batchOptions: BatchExecutionOptions { configuration.batchOptions }
    var batchSize: Int { activeBatch?.count ?? 0 }
    var hasWork: Bool { activeBatch != nil || !unprocessedPrompts.isEmpty }

    init(
        model: any LanguageModel,
        configuration: BatchIteratorConfiguration,
        stopTokens: Set<Int>,
        unknownTokenId: Int?
    ) {
        self.model = model
        self.configuration = configuration
        self.stopTokens = stopTokens
        self.unknownTokenId = unknownTokenId
    }

    mutating func insert(
        uids: [Int],
        prompts: [[Int]],
        maxTokens: [Int],
        samplers: [any LogitSampler],
        processors: [BatchLogitProcessorBox?]
    ) {
        precondition(uids.count == prompts.count, "uids.count must match prompts.count")
        precondition(maxTokens.count == prompts.count, "maxTokens.count must match prompts.count")
        precondition(samplers.count == prompts.count, "samplers.count must match prompts.count")
        precondition(processors.count == prompts.count, "processors.count must match prompts.count")

        for index in prompts.indices {
            unprocessedPrompts.append(
                PendingPrompt(
                    uid: uids[index],
                    tokens: prompts[index],
                    maxTokens: maxTokens[index],
                    sampler: samplers[index],
                    processor: processors[index]
                )
            )
        }

        unprocessedPrompts.sort { $0.length < $1.length }
    }

    mutating func remove(uids: some Sequence<Int>) {
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

    mutating func next() -> [Response]? {
        if activeBatch == nil && unprocessedPrompts.isEmpty {
            return nil
        }

        var batch = activeBatch

        var numActive = batch?.count ?? 0
        var numToAdd = max(configuration.completionBatchSize - numActive, 0)
        let maxChunkSize = max(configuration.prefillBatchSize, 1)

        while numToAdd > 0 && !unprocessedPrompts.isEmpty {
            let chunkSize = min(numToAdd, maxChunkSize)
            let chunk = Array(unprocessedPrompts.prefix(chunkSize))
            let newBatch = processPrompts(chunk)
            unprocessedPrompts.removeFirst(chunk.count)

            if let current = batch {
                current.extend(newBatch)
            } else {
                batch = newBatch
            }

            numActive = batch?.count ?? 0
            numToAdd = max(configuration.completionBatchSize - numActive, 0)
        }

        guard let currentBatch = batch ?? activeBatch else {
            return nil
        }

        let emittedTokens = currentBatch.tokens
        var processors = currentBatch.processors
        let nextTokens = step(
            inputTokens: emittedTokens[0..., .newAxis],
            cache: &currentBatch.cache,
            samplers: currentBatch.samplers,
            processors: &processors
        )
        currentBatch.processors = processors
        asyncEval(nextTokens)

        let emittedTokenArray = emittedTokens.asArray(Int.self)

        var responses: [Response] = []
        var keepIndices: [Int] = []
        var finishedAny = false

        responses.reserveCapacity(emittedTokenArray.count)
        keepIndices.reserveCapacity(emittedTokenArray.count)

        for index in emittedTokenArray.indices {
            let token = emittedTokenArray[index]
            currentBatch.numTokens[index] += 1

            let hitStopToken = token == unknownTokenId || stopTokens.contains(token)
            let hitLengthLimit = currentBatch.numTokens[index] >= currentBatch.maxTokens[index]

            let stopReason: GenerateStopReason?
            if hitStopToken {
                stopReason = .stop
                finishedAny = true
            } else if hitLengthLimit {
                stopReason = .length
                finishedAny = true
            } else {
                stopReason = nil
                keepIndices.append(index)
            }

            responses.append(.init(uid: currentBatch.uids[index], token: token, stopReason: stopReason))
        }

        currentBatch.tokens = nextTokens

        if finishedAny {
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

    private mutating func processPrompts(_ prompts: [PendingPrompt]) -> ActiveBatch {
        let tokenLists = prompts.map(\.tokens)
        let (padded, leftPadding) = leftPadPrompts(tokenLists)

        var cache = makeBatchCache(
            model: model,
            parameters: configuration.generation,
            leftPadding: leftPadding
        )

        var processors = prompts.map(\.processor)
        for index in processors.indices {
            if var processor = processors[index] {
                processor.prompt(MLXArray(tokenLists[index].map(Int32.init)))
                processors[index] = processor
            }
        }

        var remaining = padded
        while remaining.dim(1) > 1 {
            let nToProcess = min(configuration.generation.prefillStepSize, remaining.dim(1) - 1)
            let slice = remaining[0..., ..<nToProcess]
            _ = model(slice, cache: cache)
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: configuration.generation.kvBits,
                kvGroupSize: configuration.generation.kvGroupSize,
                quantizedKVStart: configuration.generation.quantizedKVStart
            )

            let states = cache.flatMap(\.state)
            if !states.isEmpty {
                eval(states)
            }

            let length = remaining.dim(1)
            remaining = remaining[0..., nToProcess..<length]
            Memory.clearCache()
        }

        let firstTokens = step(
            inputTokens: remaining,
            cache: &cache,
            samplers: prompts.map(\.sampler),
            processors: &processors
        )
        asyncEval(firstTokens)

        return ActiveBatch(
            uids: prompts.map(\.uid),
            tokens: firstTokens,
            maxTokens: prompts.map(\.maxTokens),
            numTokens: Array(repeating: 0, count: prompts.count),
            cache: cache,
            samplers: prompts.map(\.sampler),
            processors: processors
        )
    }

    private mutating func step(
        inputTokens: MLXArray,
        cache: inout [KVCache],
        samplers: [any LogitSampler],
        processors: inout [BatchLogitProcessorBox?]
    ) -> MLXArray {
        let logits = model(inputTokens, cache: cache)
        let selected = logits[0..., -1, 0...]

        var sampledTokens: [Int32] = []
        sampledTokens.reserveCapacity(samplers.count)

        for index in 0 ..< samplers.count {
            var row = selected[index, .ellipsis]

            if var processor = processors[index] {
                row = processor.process(logits: row)
                processors[index] = processor
            }

            let sampled = samplers[index].sample(logits: row)

            if var processor = processors[index] {
                processor.didSample(token: sampled)
                processors[index] = processor
            }

            sampledTokens.append(sampled.item(Int32.self))
        }

        return MLXArray(sampledTokens)
    }
}

extension BatchTokenIterator: @unchecked Sendable {}

private func leftPadPrompts(_ prompts: [[Int]]) -> (MLXArray, [Int]) {
    guard let maxLength = prompts.map(\.count).max() else {
        return (MLXArray.zeros([0, 0], type: Int32.self), [])
    }

    var padded: [Int] = []
    padded.reserveCapacity(prompts.count * maxLength)

    var leftPadding: [Int] = []
    leftPadding.reserveCapacity(prompts.count)

    for prompt in prompts {
        let pad = maxLength - prompt.count
        leftPadding.append(pad)
        padded.append(contentsOf: Array(repeating: 0, count: pad))
        padded.append(contentsOf: prompt)
    }

    return (MLXArray(padded, [prompts.count, maxLength]), leftPadding)
}

private func makeBatchCache(
    model: any LanguageModel,
    parameters: GenerateParameters,
    leftPadding: [Int]
) -> [KVCache] {
    model.newCache(parameters: parameters).map {
        convertCache($0, leftPadding: leftPadding)
    }
}

private func convertCache(_ cache: KVCache, leftPadding: [Int]) -> KVCache {
    switch cache {
    case is BatchKVCache:
        return cache
    case is KVCacheSimple:
        return BatchKVCache(leftPadding: leftPadding)
    case let rotating as RotatingKVCache:
        guard rotating.keepTokens == 0 else {
            fatalError("RotatingKVCache with keep tokens is not supported in batched generation")
        }
        guard let maxSize = rotating.maxSize else {
            fatalError("RotatingKVCache expected to have a max size for batching")
        }
        return BatchRotatingKVCache(maxSize: maxSize, leftPadding: leftPadding)
    case let arrays as ArraysCache:
        arrays.setLeftPadding(leftPadding)
        return arrays
    case let list as CacheList:
        let converted = (0 ..< list.count).map {
            convertCache(list[$0], leftPadding: leftPadding)
        }
        return CacheList(converted)
    default:
        fatalError("\(type(of: cache)) does not yet support batching")
    }
}
