// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite(
    "Batch Token Iterator",
    .enabled(if: MLXMetalGuard.isAvailable, MLXMetalGuard.unavailableComment),
    .serialized
)
struct BatchTokenIteratorTests {

    @Test("Insert allocates unique ids and sorts pending prompts by length")
    func insertAllocatesUniqueUIDsAndSortsPendingByLength() {
        let iterator = BatchTokenIterator(model: TrackingBatchLanguageModel())

        let uids = iterator.insert(
            prompts: [[1, 2, 3, 4], [9, 10], [5, 6, 7]],
            maxTokens: [1, 1, 1]
        )

        #expect(Set(uids).count == 3)
        #expect(iterator.pendingPrompts.map { $0.tokens.count } == [2, 3, 4])
    }

    @Test("Segmented prefill does less work than naive max-length padding")
    func segmentedPrefillUsesLessWorkThanNaivePadding() {
        let model = TrackingBatchLanguageModel()
        let prompts = [[1, 2], [3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
        let iterator = BatchTokenIterator(
            model: model,
            completionBatchSize: 8,
            prefillBatchSize: 8,
            prefillStepSize: 64
        )

        _ = iterator.insert(prompts: prompts, maxTokens: [1, 1, 1])
        while let responses = iterator.next(), !responses.isEmpty {}

        let totalWork = model.inputShapes.reduce(0) { partial, shape in
            partial + shape[0] * shape[1]
        }
        let naiveWork = prompts.count * (prompts.map(\.count).max() ?? 0)

        #expect(totalWork < naiveWork)
        #expect(model.inputShapes.contains(where: { $0[0] < prompts.count && $0[1] > 0 }))
    }

    @Test("Per-request samplers remain isolated")
    func perRequestSamplersRemainIsolated() {
        let iterator = BatchTokenIterator(
            model: TrackingBatchLanguageModel(),
            completionBatchSize: 4,
            prefillBatchSize: 4
        )

        let uids = iterator.insert(
            prompts: [[1, 2], [3, 4]],
            maxTokens: [2, 2],
            samplers: [FixedTokenSampler(token: 7), FixedTokenSampler(token: 9)]
        )

        var tokensPerUID = [Int: [Int]]()
        while let responses = iterator.next(), !responses.isEmpty {
            for response in responses {
                tokensPerUID[response.uid, default: []].append(response.token)
            }
        }

        #expect(tokensPerUID[uids[0]] == [7, 7])
        #expect(tokensPerUID[uids[1]] == [9, 9])
    }

    @Test("Per-request processors see only their own prompt and samples")
    func perRequestProcessorsRemainIsolated() {
        let processorA = RecordingProcessor()
        let processorB = RecordingProcessor()
        let iterator = BatchTokenIterator(
            model: TrackingBatchLanguageModel(),
            completionBatchSize: 4,
            prefillBatchSize: 4
        )

        _ = iterator.insert(
            prompts: [[1, 2, 3], [8, 9]],
            maxTokens: [2, 2],
            processors: [processorA, processorB]
        )

        while let responses = iterator.next(), !responses.isEmpty {}

        #expect(processorA.promptTokens == [[1, 2, 3]])
        #expect(processorB.promptTokens == [[8, 9]])
        #expect(processorA.sampledTokens.count == 2)
        #expect(processorB.sampledTokens.count == 2)
    }

    @Test("Cache clear hook runs at the configured interval")
    func cacheClearHookRunsAtConfiguredInterval() {
        let counter = LockedCounter()
        let iterator = BatchTokenIterator(
            model: TrackingBatchLanguageModel(),
            completionBatchSize: 4,
            prefillBatchSize: 4,
            cacheClearInterval: 2
        )
        iterator.clearCache = {
            counter.increment()
        }

        _ = iterator.insert(prompts: [[1, 2, 3]], maxTokens: [5])
        while let responses = iterator.next(), !responses.isEmpty {}

        #expect(counter.read() == 2)
    }

    @Test("cacheMaxKVSize creates rotating batch caches and extracted rotating caches")
    func cacheMaxKVSizeControlsBatchAndExtractedCacheTypes() throws {
        let iterator = BatchTokenIterator(
            model: ParameterAwareCacheMockModel(),
            completionBatchSize: 4,
            prefillBatchSize: 4,
            cacheMaxKVSize: 5
        )

        _ = iterator.insert(prompts: [[1, 2, 3, 4, 5, 6, 7, 8]], maxTokens: [3])

        let firstStep = try #require(iterator.next())
        #expect(firstStep.isEmpty == false)

        let batchCache = try #require(iterator.activeBatch?.cache.first)
        let rotatingBatch = try #require(batchCache as? BatchRotatingKVCache)
        #expect(rotatingBatch.maxSize == 5)
        #expect(rotatingBatch.keep == 4)

        var finalCache: [KVCache]?
        while let responses = iterator.next(), !responses.isEmpty {
            for response in responses where response.finishReason != nil {
                finalCache = response.finalCache
            }
        }

        let extracted = try #require(finalCache?.first as? RotatingKVCache)
        let keys = try #require(extracted.state.first)
        #expect(extracted.maxSize == 5)
        #expect(keys.dim(2) <= 5)
    }

    @Test("Mixed cache templates preserve rotating layers in the active batch")
    func mixedLayerTemplatesPreserveRotatingCacheTypes() throws {
        let iterator = BatchTokenIterator(
            model: MixedLayerBatchLanguageModel(),
            completionBatchSize: 4,
            prefillBatchSize: 4
        )

        _ = iterator.insert(prompts: [[1, 2, 3]], maxTokens: [2])
        let firstStep = try #require(iterator.next())
        #expect(firstStep.isEmpty == false)

        let batchCache = try #require(iterator.activeBatch?.cache)
        #expect(batchCache.count == 3)
        #expect(batchCache[0] is BatchKVCache)
        #expect(batchCache[1] is BatchRotatingKVCache)
        #expect(batchCache[2] is BatchKVCache)

        let rotatingLayer = try #require(batchCache[1] as? BatchRotatingKVCache)
        #expect(rotatingLayer.maxSize == 64)
        #expect(rotatingLayer.keep == 4)
    }
}

private func appendSyntheticKV(
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

private final class TrackingBatchLanguageModel: Module, LanguageModel, @unchecked Sendable {
    let vocabSize: Int
    let numLayers: Int

    var callCount = 0
    var inputShapes = [[Int]]()

    init(vocabSize: Int = 32, numLayers: Int = 1) {
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
        inputShapes.append(input.tokens.shape)
        appendSyntheticKV(to: cache, inputTokens: input.tokens)

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
}

private final class MixedLayerBatchLanguageModel: Module, LanguageModel, @unchecked Sendable {
    let vocabSize: Int
    let slidingWindowMaxSize: Int
    let slidingWindowKeep: Int

    init(vocabSize: Int = 32, slidingWindowMaxSize: Int = 64, slidingWindowKeep: Int = 4) {
        self.vocabSize = vocabSize
        self.slidingWindowMaxSize = slidingWindowMaxSize
        self.slidingWindowKeep = slidingWindowKeep
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        appendSyntheticKV(to: cache, inputTokens: input.tokens)

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
            RotatingKVCache(maxSize: slidingWindowMaxSize, keep: slidingWindowKeep),
            KVCacheSimple(),
        ]
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

private final class ParameterAwareCacheMockModel: Module, LanguageModel, @unchecked Sendable {
    let vocabSize: Int

    init(vocabSize: Int = 32) {
        self.vocabSize = vocabSize
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        appendSyntheticKV(to: cache, inputTokens: input.tokens)

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
        if let maxKVSize = parameters?.maxKVSize {
            return [RotatingKVCache(maxSize: maxKVSize, keep: 4)]
        } else {
            return [KVCacheSimple()]
        }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

private final class LockedCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    func increment() {
        lock.lock()
        value += 1
        lock.unlock()
    }

    func read() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

private struct FixedTokenSampler: LogitSampler {
    let token: Int

    func sample(logits: MLXArray) -> MLXArray {
        MLXArray(Int32(token))
    }
}

private final class RecordingProcessor: @unchecked Sendable, LogitProcessor {
    private let lock = NSLock()
    private var _promptTokens = [[Int]]()
    private var _sampledTokens = [Int]()

    func prompt(_ prompt: MLXArray) {
        lock.lock()
        _promptTokens.append(prompt.asArray(Int32.self).map(Int.init))
        lock.unlock()
    }

    func process(logits: MLXArray) -> MLXArray {
        logits
    }

    func didSample(token: MLXArray) {
        lock.lock()
        _sampledTokens.append(token.item(Int.self))
        lock.unlock()
    }

    var promptTokens: [[Int]] {
        lock.lock()
        defer { lock.unlock() }
        return _promptTokens
    }

    var sampledTokens: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return _sampledTokens
    }
}
