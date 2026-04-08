// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private func appendPromptCacheSyntheticKV(
    to caches: [KVCache]?,
    inputTokens: MLXArray,
    defaultHeads: Int = 2,
    defaultHeadDim: Int = 4
) {
    guard let caches else { return }

    let batchSize = inputTokens.dim(0)
    let seqLen = inputTokens.dim(1)

    for (layerIndex, cache) in caches.enumerated() {
        let state = cache.innerState()
        let existingKeys = state.first
        let existingValues = state.count > 1 ? state[1] : nil

        let heads = existingKeys?.dim(1) ?? defaultHeads
        let keyDim = existingKeys?.dim(3) ?? defaultHeadDim
        let valueDim = existingValues?.dim(3) ?? keyDim

        let baseValue = Float(layerIndex + 1)
        let keys = MLXArray.ones([batchSize, heads, seqLen, keyDim]) * baseValue
        let values = MLXArray.ones([batchSize, heads, seqLen, valueDim]) * (baseValue + 1)
        _ = cache.update(keys: keys, values: values)
    }
}

private final class PromptCacheModel: Module, LanguageModel, @unchecked Sendable {
    let vocabSize: Int
    let numLayers: Int

    var callCount = 0
    var totalTokensProcessed = 0

    init(vocabSize: Int = 32, numLayers: Int = 2) {
        self.vocabSize = vocabSize
        self.numLayers = numLayers
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        callCount += 1
        totalTokensProcessed += input.tokens.dim(0) * input.tokens.dim(1)
        appendPromptCacheSyntheticKV(to: cache, inputTokens: input.tokens)

        let tokens = input.tokens
        let batchSize = tokens.dim(0)
        let steps = tokens.dim(1)
        var logitsFlat = [Float]()
        logitsFlat.reserveCapacity(batchSize * steps * vocabSize)

        for batch in 0 ..< batchSize {
            for step in 0 ..< steps {
                let token = Int(tokens[batch, step].item(Int32.self))
                let predictedToken = (token + 1) % vocabSize
                var row = [Float](repeating: -100, count: vocabSize)
                row[predictedToken] = 0
                logitsFlat.append(contentsOf: row)
            }
        }

        return LMOutput(logits: MLXArray(logitsFlat, [batchSize, steps, vocabSize]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< numLayers).map { _ in KVCacheSimple() }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    func resetCounters() {
        callCount = 0
        totalTokensProcessed = 0
    }
}

private final class RotatingPromptCacheModel: Module, LanguageModel, @unchecked Sendable {
    let vocabSize: Int
    let numLayers: Int
    let maxKVSize: Int

    init(vocabSize: Int = 32, numLayers: Int = 2, maxKVSize: Int = 64) {
        self.vocabSize = vocabSize
        self.numLayers = numLayers
        self.maxKVSize = maxKVSize
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        appendPromptCacheSyntheticKV(to: cache, inputTokens: input.tokens)

        let tokens = input.tokens
        let batchSize = tokens.dim(0)
        let steps = tokens.dim(1)
        var logitsFlat = [Float]()
        logitsFlat.reserveCapacity(batchSize * steps * vocabSize)

        for batch in 0 ..< batchSize {
            for step in 0 ..< steps {
                let token = Int(tokens[batch, step].item(Int32.self))
                let predictedToken = (token + 1) % vocabSize
                var row = [Float](repeating: -100, count: vocabSize)
                row[predictedToken] = 0
                logitsFlat.append(contentsOf: row)
            }
        }

        return LMOutput(logits: MLXArray(logitsFlat, [batchSize, steps, vocabSize]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< numLayers).map { _ in
            RotatingKVCache(maxSize: maxKVSize, keep: 4)
        }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

private final class MixedLayerPromptCacheModel: Module, LanguageModel, @unchecked Sendable {
    let vocabSize: Int
    let maxKVSize: Int

    init(vocabSize: Int = 32, maxKVSize: Int = 64) {
        self.vocabSize = vocabSize
        self.maxKVSize = maxKVSize
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        appendPromptCacheSyntheticKV(to: cache, inputTokens: input.tokens)

        let tokens = input.tokens
        let batchSize = tokens.dim(0)
        let steps = tokens.dim(1)
        var logitsFlat = [Float]()
        logitsFlat.reserveCapacity(batchSize * steps * vocabSize)

        for batch in 0 ..< batchSize {
            for step in 0 ..< steps {
                let token = Int(tokens[batch, step].item(Int32.self))
                let predictedToken = (token + 1) % vocabSize
                var row = [Float](repeating: -100, count: vocabSize)
                row[predictedToken] = 0
                logitsFlat.append(contentsOf: row)
            }
        }

        return LMOutput(logits: MLXArray(logitsFlat, [batchSize, steps, vocabSize]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [
            KVCacheSimple(),
            RotatingKVCache(maxSize: maxKVSize, keep: 4),
        ]
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

private func makeSimpleCache(
    seqLen: Int,
    heads: Int = 2,
    headDim: Int = 4,
    value: Float = 1.0
) -> KVCacheSimple {
    let cache = KVCacheSimple()
    if seqLen > 0 {
        let keys = MLXArray.ones([1, heads, seqLen, headDim]) * value
        let values = MLXArray.ones([1, heads, seqLen, headDim]) * (value + 1)
        _ = cache.update(keys: keys, values: values)
    }
    return cache
}

private func makeSimplePromptCache(
    layers: Int = 2,
    seqLen: Int,
    value: Float = 1.0
) -> [KVCache] {
    (0 ..< layers).map { layer in
        makeSimpleCache(seqLen: seqLen, value: value + Float(layer))
    }
}

private func makeRotatingCache(
    seqLen: Int,
    maxSize: Int,
    heads: Int = 2,
    headDim: Int = 4,
    value: Float = 1.0
) -> RotatingKVCache {
    let cache = RotatingKVCache(maxSize: maxSize, keep: 4)
    if seqLen > 0 {
        let keys = MLXArray.ones([1, heads, seqLen, headDim]) * value
        let values = MLXArray.ones([1, heads, seqLen, headDim]) * (value + 1)
        _ = cache.update(keys: keys, values: values)
    }
    return cache
}

private func makeRotatingPromptCache(
    layers: Int = 2,
    seqLen: Int,
    maxSize: Int,
    value: Float = 1.0
) -> [KVCache] {
    (0 ..< layers).map { layer in
        makeRotatingCache(seqLen: seqLen, maxSize: maxSize, value: value + Float(layer))
    }
}

private func makeMixedLayerPromptCache(
    seqLen: Int,
    maxSize: Int,
    value: Float = 1.0
) -> [KVCache] {
    [
        makeSimpleCache(seqLen: seqLen, value: value),
        makeRotatingCache(seqLen: seqLen, maxSize: maxSize, value: value + 1),
    ]
}

@Suite(
    "Prompt Cache",
    .enabled(if: MLXMetalGuard.isAvailable, MLXMetalGuard.unavailableComment),
    .serialized
)
struct PromptCacheTests {

    @Test("Empty cache returns nil and the full remainder")
    func emptyCacheReturnsNil() {
        let cache = LRUPromptCache(maxSize: 10)
        let (result, remainder) = cache.fetchNearestCache(model: "model", tokens: [1, 2, 3])

        #expect(result == nil)
        #expect(remainder == [1, 2, 3])
    }

    @Test("Nearest-cache lookup prefers the longest prefix and trims longer entries")
    func nearestCacheLookupHandlesShorterAndLongerMatches() throws {
        let cache = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "model",
            tokens: [1, 2],
            promptCache: makeSimplePromptCache(seqLen: 2)
        )
        cache.insertCache(
            model: "model",
            tokens: [1, 2, 3, 4],
            promptCache: makeSimplePromptCache(seqLen: 4, value: 2)
        )

        let (shorterMatch, shorterRemainder) = cache.fetchNearestCache(
            model: "model",
            tokens: [1, 2, 3, 4, 5]
        )
        let shorterLayer = try #require(shorterMatch?.first)
        #expect(shorterRemainder == [5])
        #expect(shorterLayer.offset == 4)

        let (longerMatch, longerRemainder) = cache.fetchNearestCache(
            model: "model",
            tokens: [1, 2, 3]
        )
        let longerLayer = try #require(longerMatch?.first)
        #expect(longerRemainder.isEmpty)
        #expect(longerLayer.offset == 3)
    }

    @Test("LRU eviction honors recency")
    func lruEvictionHonorsRecency() {
        let cache = LRUPromptCache(maxSize: 2)
        cache.insertCache(model: "model", tokens: [1], promptCache: makeSimplePromptCache(seqLen: 1))
        cache.insertCache(model: "model", tokens: [2], promptCache: makeSimplePromptCache(seqLen: 1))

        _ = cache.fetchNearestCache(model: "model", tokens: [1])
        cache.insertCache(model: "model", tokens: [3], promptCache: makeSimplePromptCache(seqLen: 1))

        #expect(cache.fetchNearestCache(model: "model", tokens: [1]).0 != nil)
        #expect(cache.fetchNearestCache(model: "model", tokens: [2]).0 == nil)
        #expect(cache.fetchNearestCache(model: "model", tokens: [3]).0 != nil)
    }

    @Test("Fetch returns deep copies and model names stay isolated")
    func fetchReturnsDeepCopiesAndHonorsModelIsolation() throws {
        let cache = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "model-a",
            tokens: [1, 2, 3],
            promptCache: makeSimplePromptCache(seqLen: 3)
        )

        let (copyA, _) = cache.fetchNearestCache(model: "model-a", tokens: [1, 2, 3])
        let (copyB, _) = cache.fetchNearestCache(model: "model-a", tokens: [1, 2, 3])
        let (otherModel, remainder) = cache.fetchNearestCache(model: "model-b", tokens: [1, 2, 3])

        let first = try #require(copyA?.first)
        let second = try #require(copyB?.first)
        first.trim(1)

        #expect(first.offset != second.offset)
        #expect(otherModel == nil)
        #expect(remainder == [1, 2, 3])
    }

    @Test("Cached prefixes reduce prefill token work")
    func cachedPrefixesReducePrefillWork() {
        let model = PromptCacheModel(vocabSize: 32, numLayers: 2)
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        let fullIterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )
        _ = fullIterator.insert(prompts: [prompt], maxTokens: [1])
        _ = fullIterator.next()
        let fullWork = model.totalTokensProcessed

        model.resetCounters()

        let cachedIterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )
        _ = cachedIterator.insert(
            prompts: [prompt],
            maxTokens: [1],
            cachedKVStates: [makeSimplePromptCache(seqLen: 8)]
        )
        _ = cachedIterator.next()
        let cachedWork = model.totalTokensProcessed

        #expect(cachedWork < fullWork)
    }

    @Test("Exact-hit rotating caches survive cached batch generation")
    func exactHitRotatingCachesSurviveCachedBatchGeneration() {
        let model = RotatingPromptCacheModel(vocabSize: 32, numLayers: 2, maxKVSize: 64)
        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let prompt = [1, 2, 3, 4, 5]
        let uids = iterator.insert(
            prompts: [prompt],
            maxTokens: [2],
            cachedKVStates: [makeRotatingPromptCache(seqLen: 5, maxSize: 64)]
        )

        var tokensPerUID = [Int: [Int]]()
        while let responses = iterator.next(), !responses.isEmpty {
            for response in responses {
                tokensPerUID[response.uid, default: []].append(response.token)
            }
        }

        #expect(tokensPerUID[uids[0]]?.count == 2)
    }

    @Test("Mixed-layer partial hits preserve per-layer batch cache types")
    func mixedLayerPartialHitsPreservePerLayerCacheTypes() throws {
        let model = MixedLayerPromptCacheModel(vocabSize: 32, maxKVSize: 64)
        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        _ = iterator.insert(
            prompts: [[1, 2, 3, 4, 5, 6, 7, 8]],
            maxTokens: [3],
            cachedKVStates: [makeMixedLayerPromptCache(seqLen: 5, maxSize: 64)]
        )

        let firstResponses = try #require(iterator.next())
        #expect(firstResponses.isEmpty == false)

        let batchCache = try #require(iterator.activeBatch?.cache)
        #expect(batchCache.count == 2)
        #expect(batchCache[0] is BatchKVCache)
        #expect(batchCache[1] is BatchRotatingKVCache)
    }

    @Test("Mixed-depth cached batches still produce tokens for every sequence")
    func mixedDepthCachedBatchesStillProduceTokensForEverySequence() {
        let model = PromptCacheModel(vocabSize: 32, numLayers: 2)
        let iterator = BatchTokenIterator(
            model: model,
            defaultSampler: ArgMaxSampler(),
            completionBatchSize: 32,
            prefillBatchSize: 8
        )

        let promptA = [1, 2, 3, 4, 5, 6, 7, 8]
        let promptB = [11, 12, 13, 14, 15, 16, 17, 18]
        let uids = iterator.insert(
            prompts: [promptA, promptB],
            maxTokens: [3, 3],
            cachedKVStates: [
                makeSimplePromptCache(seqLen: 5, value: 1),
                makeSimplePromptCache(seqLen: 3, value: 2),
            ]
        )

        var tokensPerUID = [Int: [Int]]()
        while let responses = iterator.next(), !responses.isEmpty {
            for response in responses {
                tokensPerUID[response.uid, default: []].append(response.token)
            }
        }

        #expect(tokensPerUID[uids[0]]?.count == 3)
        #expect(tokensPerUID[uids[1]]?.count == 3)
    }
}
