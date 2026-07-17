// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

// MARK: - Stop sequence matcher

@Suite("Stop sequence matcher")
struct StopSequenceMatcherTests {

    @Test("Matches multi-token stops and transitions between states")
    func matchesMultiTokenStopsAndTransitions() {
        let machine = StopSequenceMatcher(
            states: [
                "normal": [(sequence: [4, 5], next: "afterMarker")],
                "afterMarker": [(sequence: [6], next: nil)],
            ]
        )

        var state = machine.makeState()

        var result = machine.match(state, 4)
        #expect(result.matchedSequence == nil)
        #expect(result.currentState == "normal")
        state = result.next

        result = machine.match(state, 5)
        #expect(result.matchedSequence == [4, 5])
        #expect(result.currentState == "afterMarker")
        state = result.next

        result = machine.match(state, 6)
        #expect(result.matchedSequence == [6])
        #expect(result.currentState == nil)
    }

    @Test("Matches overlapping stop sequences")
    func matchesOverlappingStopSequence() {
        let machine = StopSequenceMatcher(states: ["normal": [(sequence: [1, 2], next: nil)]])
        var state = machine.makeState()

        var result = machine.match(state, 1)
        #expect(result.matchedSequence == nil)
        state = result.next

        result = machine.match(state, 1)
        #expect(result.matchedSequence == nil)
        state = result.next

        result = machine.match(state, 2)
        #expect(result.matchedSequence == [1, 2])
        #expect(result.currentState == nil)
    }

    @Test("A shorter stop completing inside a longer partial match still fires")
    func shorterStopInsideLongerPartialMatch() {
        // Stops [1,2,3] and [2]: after tokens 1,2 the matcher holds the
        // viable prefix [1,2] of the longer stop -- but [2] has completed
        // and must fire (it would otherwise be shadowed forever).
        let machine = StopSequenceMatcher(
            states: [
                "normal": [
                    (sequence: [1, 2, 3], next: nil),
                    (sequence: [2], next: nil),
                ]
            ]
        )
        var state = machine.makeState()

        var result = machine.match(state, 1)
        #expect(result.matchedSequence == nil)
        state = result.next

        result = machine.match(state, 2)
        #expect(result.matchedSequence == [2])
        #expect(result.currentState == nil)
    }

    @Test("Longer stop still wins when no shorter stop completes")
    func longerStopUnaffectedBySuffixScan() {
        // Stops [1,2] and [2,3]: feeding 1,2 must fire [1,2] (a completed
        // longer match beats a live shorter prefix), and the suffix scan
        // must not misfire [2,3] on its non-terminal prefix.
        let machine = StopSequenceMatcher(
            states: [
                "normal": [
                    (sequence: [1, 2], next: nil),
                    (sequence: [2, 3], next: nil),
                ]
            ]
        )
        var state = machine.makeState()

        var result = machine.match(state, 1)
        #expect(result.matchedSequence == nil)
        state = result.next

        result = machine.match(state, 2)
        #expect(result.matchedSequence == [1, 2])
        #expect(result.currentState == nil)
    }
}

// MARK: - Row samplers

@Suite(.serialized)
struct RowSamplerTests {

    @Test("Top-K of one always selects the best token")
    func topKOneAlwaysSelectsBestToken() {
        let sampler = makeRowSampler(temperature: 1, topP: 1, topK: 1, seed: 7)
        let logprobs = MLXArray([0.1 as Float, 3.0 as Float, 2.0 as Float])[
            .newAxis, .ellipsis
        ]

        for _ in 0 ..< 5 {
            #expect(sampler(logprobs).item(Int.self) == 1)
        }
    }

    @Test("Greedy sampler picks the arg-max token")
    func greedySamplerPicksArgMax() {
        let logprobs = MLXArray([0.1 as Float, 3.0 as Float, 2.0 as Float])[
            .newAxis, .ellipsis
        ]
        #expect(greedySampler(logprobs).item(Int.self) == 1)
    }
}

// MARK: - Engine: cache topology validation

@Suite(.serialized)
struct BatchGenerationEngineTopologyTests {

    @Test("Accepts supported cache topologies")
    func acceptsSupportedCacheTopologies() throws {
        _ = try BatchGenerationEngine(
            model: CacheTopologyLanguageModel { _ in [KVCacheSimple()] }
        )

        _ = try BatchGenerationEngine(
            model: CacheTopologyLanguageModel { _ in [RotatingKVCache(maxSize: 8, keep: 0)] }
        )

        // SSM topologies (ArraysCache/MambaCache, alone or inside a
        // CacheList) are factory-rejected until the in-repo models all
        // thread createSSMMask; see makeBatchedCacheFactory and the
        // "Rejects unsupported cache topologies" test below.
    }

    @Test("Rejects unsupported cache topologies")
    func rejectsUnsupportedCacheTopologies() {
        assertEngineRejectsCache(
            QuantizedKVCache(),
            expectedType: "QuantizedKVCache",
            expectedPath: "layer"
        )
        assertEngineRejectsCache(
            ChunkedKVCache(),
            expectedType: "ChunkedKVCache",
            expectedPath: "layer"
        )
        // keep-prefix rotation cannot be combined with per-row left padding
        // (see makeBatchedCacheFactory); these topologies stay single-stream.
        assertEngineRejectsCache(
            RotatingKVCache(maxSize: 8, keep: 4),
            expectedType: "RotatingKVCache",
            expectedPath: "layer"
        )
        // SSM caches are factory-rejected, so the composite rejects at its
        // Mamba child (the first child hit by the recursion).
        assertEngineRejectsCache(
            CacheList(MambaCache(), QuantizedKVCache()),
            expectedType: "MambaCache",
            expectedPathContains: "children"
        )
    }

    @Test("Passes cache parameters through to the model")
    func passesCacheParametersToModel() throws {
        let model = CacheTopologyLanguageModel { _ in [KVCacheSimple()] }

        _ = try BatchGenerationEngine(
            model: model,
            cacheParameters: GenerateParameters(maxKVSize: 17)
        )

        #expect(model.receivedParameters?.maxKVSize == 17)
    }
}

// MARK: - Engine: input validation

@Suite(.serialized)
struct BatchGenerationEngineValidationTests {

    @Test("Rejects nonpositive configuration values")
    func rejectsInvalidConfiguration() {
        #expect(throws: BatchGenerationEngineError.self) {
            _ = try BatchGenerationEngine(
                model: IncrementingLanguageModel(),
                defaultMaxTokens: 0
            )
        }
    }

    @Test("Rejects empty prompt rows")
    func rejectsEmptyPromptRows() throws {
        let engine = try BatchGenerationEngine(model: IncrementingLanguageModel())
        #expect(throws: BatchGenerationEngineError.self) {
            _ = try engine.insert(prompts: [[1, 2], []])
        }
    }

    @Test("Rejects mismatched per-row option counts")
    func rejectsMismatchedOptionCounts() throws {
        let engine = try BatchGenerationEngine(model: IncrementingLanguageModel())
        #expect(throws: BatchGenerationEngineError.self) {
            _ = try engine.insert(prompts: [[1], [2]], maxTokens: [4])
        }
    }

    @Test("Rejects nonpositive max tokens")
    func rejectsNonPositiveMaxTokens() throws {
        let engine = try BatchGenerationEngine(model: IncrementingLanguageModel())
        #expect(throws: BatchGenerationEngineError.self) {
            _ = try engine.insert(prompts: [[1]], maxTokens: [0])
        }
    }
}

// MARK: - Engine: pull loop

@Suite(.serialized)
struct BatchGenerationEnginePullLoopTests {

    @Test("Admits queued rows and reports finish reasons")
    func admitsQueuedRowsAndReportsFinishReasons() throws {
        let engine = try BatchGenerationEngine(
            model: IncrementingLanguageModel(),
            eosTokens: [[5]],
            defaultMaxTokens: 4,
            prefillBatchSize: 1,
            completionBatchSize: 2
        )

        let uids = try engine.insert(prompts: [[1, 2], [8]], maxTokens: [4, 2])
        #expect(uids == [0, 1])

        var tokensByUID: [Int: [Int]] = [:]
        var finishReasonByUID: [Int: GenerateStopReason] = [:]
        var steps = 0

        while engine.hasWork {
            steps += 1
            #expect(steps < 10)

            for response in engine.next() {
                tokensByUID[response.uid, default: []].append(response.token)
                if let finishReason = response.finishReason {
                    finishReasonByUID[response.uid] = finishReason
                }
            }
        }

        #expect(tokensByUID[0] == [3, 4, 5])
        #expect(tokensByUID[1] == [9, 10])
        #expect(finishReasonByUID[0] == .stop)
        #expect(finishReasonByUID[1] == .length)
        #expect(engine.promptTokensProcessed == 3)
        #expect(engine.hasWork == false)
    }

    @Test("Cancel removes a queued request")
    func cancelRemovesQueuedRequest() throws {
        let engine = try BatchGenerationEngine(
            model: IncrementingLanguageModel(),
            defaultMaxTokens: 3,
            prefillBatchSize: 1,
            completionBatchSize: 1
        )

        let uids = try engine.insert(prompts: [[1], [8]], maxTokens: [3, 3])
        #expect(engine.cancel(uid: uids[1]))
        #expect(engine.cancel(uid: 999) == false)

        var seenUIDs = Set<Int>()
        var steps = 0
        while engine.hasWork {
            steps += 1
            #expect(steps < 10)
            for response in engine.next() {
                seenUIDs.insert(response.uid)
            }
        }

        #expect(seenUIDs == [uids[0]])
    }

    @Test("Cancel removes an active request mid-decode")
    func cancelRemovesActiveRequest() throws {
        let engine = try BatchGenerationEngine(
            model: IncrementingLanguageModel(),
            defaultMaxTokens: 4,
            prefillBatchSize: 2,
            completionBatchSize: 2
        )

        let uids = try engine.insert(prompts: [[1], [8]], maxTokens: [4, 4])
        let firstStep = engine.next()
        #expect(Set(firstStep.map(\.uid)) == Set(uids))

        #expect(engine.cancel(uid: uids[0]))

        var laterUIDs = Set<Int>()
        var steps = 0
        while engine.hasWork {
            steps += 1
            #expect(steps < 10)
            for response in engine.next() {
                laterUIDs.insert(response.uid)
            }
        }

        #expect(laterUIDs.contains(uids[0]) == false)
        #expect(laterUIDs.contains(uids[1]))
        #expect(engine.hasWork == false)
        #expect(engine.next().isEmpty)
    }

    @Test("Capturing final caches yields one cache per finished row")
    func capturingFinalCachesYieldsCachePerFinishedRow() throws {
        let engine = try BatchGenerationEngine(
            model: IncrementingLanguageModel(),
            eosTokens: [[5]],
            defaultMaxTokens: 4,
            prefillBatchSize: 1,
            completionBatchSize: 2
        )

        _ = try engine.insert(prompts: [[1, 2], [8]], maxTokens: [4, 2])

        var finishedUIDs = Set<Int>()
        var steps = 0
        while engine.hasWork {
            steps += 1
            #expect(steps < 10)
            let (responses, finishedCaches) = engine.next(capturingFinalCaches: true)
            // Every finished response must come with a captured cache, and
            // every captured cache must correspond to a finished response.
            let finishedResponses = responses.filter { $0.finishReason != nil }
            #expect(finishedCaches.count == finishedResponses.count)
            for cache in finishedCaches {
                finishedUIDs.insert(cache.uid)
                #expect(cache.finalCache.isEmpty == false)
            }
        }

        #expect(finishedUIDs == [0, 1])
    }

    @Test("Penalty processors receive raw logits, not logprobs")
    func processorsReceiveRawLogits() throws {
        // The single `TokenIterator` applies `LogitProcessor.process` to raw
        // logits before logSoftmax; `RepetitionContext` branches on the logit
        // sign, so feeding it logprobs (all <= 0) penalizes repeated
        // positive-logit tokens the wrong way. The fixture model emits a +4
        // peak: if the processor sees a max > 1 it received raw logits, while
        // logprobs would cap every value at <= 0.
        let model = PositiveLogitsLanguageModel()
        let capture = LogitCapture()
        let factories = try makeBatchedCacheFactories(
            for: model.newCache(parameters: nil))
        let caches = factories.map { $0([0]) }

        let batch = DecodeBatch(
            model: model,
            uids: [0],
            seedTokens: MLXArray([Int32(1)]),
            promptCache: caches,
            tokens: [[1]],
            maxTokens: [4],
            processors: [CapturingProcessor(capture: capture)]
        )
        _ = batch.next()

        #expect(!capture.maxima.isEmpty)
        #expect(capture.maxima.allSatisfy { $0 > 1 })
    }
}

// MARK: - Test models & helpers

private func assertEngineRejectsCache(
    _ cache: any KVCache,
    expectedType: String,
    expectedPath: String? = nil,
    expectedPathContains: String? = nil
) {
    let model = CacheTopologyLanguageModel { _ in [cache] }

    do {
        _ = try BatchGenerationEngine(model: model)
        Issue.record("Expected BatchedCacheError.unsupportedCacheTopology, but no error was thrown")
    } catch let BatchedCacheError.unsupportedCacheTopology(_, path, cacheType, reason) {
        if let expectedPath {
            #expect(path == expectedPath)
        }
        if let expectedPathContains {
            #expect(path.contains(expectedPathContains))
        }
        #expect(cacheType == expectedType)
        #expect(reason.isEmpty == false)
    } catch {
        Issue.record("Expected BatchedCacheError.unsupportedCacheTopology, got \(error)")
    }
}

/// A model that produces deterministic "increment the input token" logits and
/// updates whatever caches it is given. Used to exercise the engine pull loop
/// without any real weights.
private final class IncrementingLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize = 16
    var kvHeads: [Int] { [1] }

    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let batchSize = inputs.dim(0)
        let sequenceLength = inputs.dim(1)

        if let cache {
            let keys = MLXArray.ones([batchSize, 1, sequenceLength, 1], dtype: .float32)
            let values = keys * 2
            for layerCache in cache {
                _ = layerCache.update(keys: keys, values: values)
            }
        }

        let inputTokens = inputs.asArray(UInt32.self).map { Int($0) }
        var logits = Array(
            repeating: Float(-1_000),
            count: batchSize * sequenceLength * vocabularySize
        )

        for batchIndex in 0 ..< batchSize {
            for tokenIndex in 0 ..< sequenceLength {
                let inputIndex = batchIndex * sequenceLength + tokenIndex
                let nextToken = (inputTokens[inputIndex] + 1) % vocabularySize
                let logitIndex =
                    (batchIndex * sequenceLength + tokenIndex) * vocabularySize + nextToken
                logits[logitIndex] = 0
            }
        }

        return MLXArray(logits).reshaped([batchSize, sequenceLength, vocabularySize])
    }
}

/// Like ``IncrementingLanguageModel`` but with a +4 peak logit (and -10
/// elsewhere), so tests can distinguish raw logits from logprobs (which are
/// always <= 0) at the `LogitProcessor` boundary.
private final class PositiveLogitsLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize = 16
    var kvHeads: [Int] { [1] }

    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let batchSize = inputs.dim(0)
        let sequenceLength = inputs.dim(1)

        if let cache {
            let keys = MLXArray.ones([batchSize, 1, sequenceLength, 1], dtype: .float32)
            let values = keys * 2
            for layerCache in cache {
                _ = layerCache.update(keys: keys, values: values)
            }
        }

        let inputTokens = inputs.asArray(UInt32.self).map { Int($0) }
        var logits = Array(
            repeating: Float(-10),
            count: batchSize * sequenceLength * vocabularySize
        )

        for batchIndex in 0 ..< batchSize {
            for tokenIndex in 0 ..< sequenceLength {
                let inputIndex = batchIndex * sequenceLength + tokenIndex
                let nextToken = (inputTokens[inputIndex] + 1) % vocabularySize
                let logitIndex =
                    (batchIndex * sequenceLength + tokenIndex) * vocabularySize + nextToken
                logits[logitIndex] = 4
            }
        }

        return MLXArray(logits).reshaped([batchSize, sequenceLength, vocabularySize])
    }
}

/// Reference sink for ``CapturingProcessor`` (the processor itself must be a
/// value type per ``LogitProcessor``'s mutating requirements).
private final class LogitCapture {
    var maxima: [Float] = []
}

/// Records the maximum value of every logit array it is asked to process,
/// passing the logits through unchanged.
private struct CapturingProcessor: LogitProcessor {
    let capture: LogitCapture
    mutating func prompt(_ prompt: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray {
        capture.maxima.append(logits.max().item(Float.self))
        return logits
    }
    mutating func didSample(token: MLXArray) {}
}

private final class CacheTopologyLanguageModel: Module, LanguageModel {
    private let cacheFactory: (GenerateParameters?) -> [any KVCache]
    private(set) var receivedParameters: GenerateParameters?

    init(_ cacheFactory: @escaping (GenerateParameters?) -> [any KVCache]) {
        self.cacheFactory = cacheFactory
    }

    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        fatalError("CacheTopologyLanguageModel is only used for cache topology tests")
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        receivedParameters = parameters
        return cacheFactory(parameters)
    }
}
