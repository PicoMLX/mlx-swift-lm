// Copyright © 2025 Apple Inc.

import Foundation
import MLX

struct BatchExecutionOptions: Sendable, Equatable, Hashable {
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
    let promptCache: [KVCache]?
    let maxTokens: Int
    let sampler: any LogitSampler
    var processor: BatchLogitProcessorBox?

    var length: Int { tokens.count }
}

private struct PrefilledPrompt {
    let uid: Int
    let firstToken: Int32
    let maxTokens: Int
    let cache: [KVCache]
    let sampler: any LogitSampler
    var processor: BatchLogitProcessorBox?
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
        promptCaches: [[KVCache]?],
        maxTokens: [Int],
        samplers: [any LogitSampler],
        processors: [BatchLogitProcessorBox?]
    ) {
        precondition(uids.count == prompts.count, "uids.count must match prompts.count")
        precondition(promptCaches.count == prompts.count, "promptCaches.count must match prompts.count")
        precondition(maxTokens.count == prompts.count, "maxTokens.count must match prompts.count")
        precondition(samplers.count == prompts.count, "samplers.count must match prompts.count")
        precondition(processors.count == prompts.count, "processors.count must match prompts.count")

        for index in prompts.indices {
            unprocessedPrompts.append(
                PendingPrompt(
                    uid: uids[index],
                    tokens: prompts[index],
                    promptCache: promptCaches[index],
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
        let cachedPrompts = prompts.filter { $0.promptCache != nil }
        let freshPrompts = prompts.filter { $0.promptCache == nil }

        var batches: [ActiveBatch] = []
        if !freshPrompts.isEmpty {
            batches.append(processFreshPrompts(freshPrompts))
        }
        if !cachedPrompts.isEmpty {
            batches.append(processCachedPrompts(cachedPrompts))
        }

        guard let batch = batches.first else {
            fatalError("processPrompts requires at least one prompt")
        }
        for nextBatch in batches.dropFirst() {
            batch.extend(nextBatch)
        }
        return batch
    }

    private mutating func processFreshPrompts(_ prompts: [PendingPrompt]) -> ActiveBatch {
        let tokenLists = prompts.map(\.tokens)
        let (padded, leftPadding) = leftPadPrompts(tokenLists)

        var cache = makeBatchCache(
            model: model,
            parameters: configuration.generation,
            leftPadding: leftPadding
        )

        var processors = prompts.map(\.processor)
        for index in processors.indices {
            prepareProcessor(&processors[index], with: tokenLists[index])
        }

        let firstTokens = prefillAndSampleFirstTokens(
            inputTokens: padded,
            cache: &cache,
            samplers: prompts.map(\.sampler),
            processors: &processors
        )

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

    private mutating func processCachedPrompts(_ prompts: [PendingPrompt]) -> ActiveBatch {
        let prefills = prompts.map { prefillSingle($0) }
        let mergedCache = mergePerSequenceCaches(prefills.map(\.cache))
        let firstTokens = MLXArray(prefills.map(\.firstToken))
        asyncEval(firstTokens)

        return ActiveBatch(
            uids: prefills.map(\.uid),
            tokens: firstTokens,
            maxTokens: prefills.map(\.maxTokens),
            numTokens: Array(repeating: 0, count: prefills.count),
            cache: mergedCache,
            samplers: prefills.map(\.sampler),
            processors: prefills.map(\.processor)
        )
    }

    private mutating func prefillSingle(_ prompt: PendingPrompt) -> PrefilledPrompt {
        var cache = prompt.promptCache ?? model.newCache(parameters: configuration.generation)
        var processor = prompt.processor
        let input = LMInput(tokens: MLXArray(prompt.tokens.map(Int32.init)))

        prepareProcessor(&processor, with: prompt.tokens)

        let firstToken: Int32
        do {
            switch try model.prepare(
                input,
                cache: cache,
                windowSize: configuration.generation.prefillStepSize
            ) {
            case .tokens(let tokens):
                var processors = [processor]
                firstToken = prefillAndSampleFirstTokens(
                    inputTokens: tokens.tokens,
                    cache: &cache,
                    samplers: [prompt.sampler],
                    processors: &processors
                ).item(Int32.self)
                processor = processors[0]

            case .logits(let result):
                firstToken = sample(
                    logits: result.logits,
                    sampler: prompt.sampler,
                    processor: &processor
                )
            }
        } catch {
            fatalError("Failed to prefill cached prompt for batching: \(error)")
        }

        return PrefilledPrompt(
            uid: prompt.uid,
            firstToken: firstToken,
            maxTokens: prompt.maxTokens,
            cache: cache,
            sampler: prompt.sampler,
            processor: processor
        )
    }

    private func prepareProcessor(_ processor: inout BatchLogitProcessorBox?, with tokens: [Int]) {
        guard var localProcessor = processor else { return }
        localProcessor.prompt(MLXArray(tokens.map(Int32.init)))
        processor = localProcessor
    }

    private mutating func prefillAndSampleFirstTokens(
        inputTokens: MLXArray,
        cache: inout [KVCache],
        samplers: [any LogitSampler],
        processors: inout [BatchLogitProcessorBox?]
    ) -> MLXArray {
        var remaining = inputTokens
        if remaining.ndim == 1 {
            remaining = remaining[.newAxis]
        }

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
            samplers: samplers,
            processors: &processors
        )
        asyncEval(firstTokens)
        return firstTokens
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

    private mutating func sample(
        logits: MLXArray,
        sampler: any LogitSampler,
        processor: inout BatchLogitProcessorBox?
    ) -> Int32 {
        var row = logits[0..., -1, 0...]

        if var localProcessor = processor {
            row = localProcessor.process(logits: row)
            processor = localProcessor
        }

        let sampled = sampler.sample(logits: row)

        if var localProcessor = processor {
            localProcessor.didSample(token: sampled)
            processor = localProcessor
        }

        return sampled.item(Int32.self)
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

private func mergePerSequenceCaches(_ perSequenceCaches: [[KVCache]]) -> [KVCache] {
    guard let first = perSequenceCaches.first else { return [] }
    let layerCount = first.count
    return (0 ..< layerCount).map { layerIndex in
        mergeLayerCaches(perSequenceCaches.map { $0[layerIndex] })
    }
}

private func mergeLayerCaches(_ caches: [KVCache]) -> KVCache {
    guard let first = caches.first else {
        fatalError("Cannot merge an empty cache list")
    }

    switch first {
    case is KVCacheSimple:
        let layerCaches = caches.map {
            guard let cache = $0 as? KVCacheSimple else {
                fatalError("Mismatched cache types while merging KVCacheSimple batch")
            }
            return cache
        }
        return BatchKVCache.fromSingle(perSequenceCaches: layerCaches.map { [$0] }).first!

    case let rotating as RotatingKVCache:
        let layerCaches = caches.map {
            guard let cache = $0 as? RotatingKVCache else {
                fatalError("Mismatched cache types while merging RotatingKVCache batch")
            }
            return cache
        }
        return mergeRotatingCaches(layerCaches, maxSize: rotating.maxSize ?? 0)

    case is MambaCache:
        return mergeArraysCaches(caches, constructor: { MambaCache() })

    case is ArraysCache:
        let stateCount = (first as? ArraysCache)?.state.count ?? 0
        return mergeArraysCaches(caches, constructor: { ArraysCache(size: stateCount) })

    case is CacheList:
        let lists = caches.map {
            guard let list = $0 as? CacheList else {
                fatalError("Mismatched cache types while merging CacheList batch")
            }
            return list
        }
        let merged = (0 ..< lists[0].count).map { index in
            mergeLayerCaches(lists.map { $0[index] })
        }
        return CacheList(merged)

    default:
        fatalError("\(type(of: first)) does not support batched merging")
    }
}

private func mergeArraysCaches(
    _ caches: [KVCache],
    constructor: () -> ArraysCache
) -> ArraysCache {
    let merged = constructor()
    let arrays = caches.map {
        guard let arrays = $0 as? ArraysCache else {
            fatalError("Mismatched array cache types while merging batch")
        }
        return arrays
    }

    let stateCount = arrays.first?.state.count ?? 0
    for index in 0 ..< stateCount {
        let values = arrays.compactMap { $0[index] }
        guard values.count == arrays.count else {
            continue
        }
        merged[index] = MLX.concatenated(values, axis: 0)
    }
    return merged
}

private func mergeRotatingCaches(
    _ caches: [RotatingKVCache],
    maxSize: Int
) -> BatchRotatingKVCache {
    let merged = BatchRotatingKVCache(maxSize: maxSize, leftPadding: Array(repeating: 0, count: caches.count))

    guard let firstState = caches.first?.state, firstState.count >= 2 else {
        return merged
    }

    let keys = MLX.concatenated(caches.map {
        guard $0.state.count >= 2 else {
            return MLXArray.zeros(firstState[0].shape, dtype: firstState[0].dtype)
        }
        return $0.state[0]
    }, axis: 0)
    let values = MLX.concatenated(caches.map {
        guard $0.state.count >= 2 else {
            return MLXArray.zeros(firstState[1].shape, dtype: firstState[1].dtype)
        }
        return $0.state[1]
    }, axis: 0)

    let offsets = MLXArray(caches.map { $0.offset })
    merged.state = [keys, values, offsets, MLXArray(Array(repeating: 0, count: caches.count))]

    let maxOffset = caches.map(\.offset).max() ?? 0
    let indices = caches.map { cache -> Int in
        let meta = cache.metaState
        return meta.count >= 5 ? (Int(meta[4]) ?? min(cache.offset, maxSize)) : min(cache.offset, maxSize)
    }
    let rotated = caches.contains { $0.offset >= maxSize && ($0.metaState.count >= 5 ? (Int($0.metaState[4]) ?? maxSize) : maxSize) < maxSize }
    merged.metaState = [String(maxSize), String(maxOffset), String(indices.max() ?? min(maxOffset, maxSize)), rotated ? "1" : "0"]
    return merged
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
