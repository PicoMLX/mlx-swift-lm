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

// MARK: - Decode batch: penalty ordering (single-stream parity)

@Suite(.serialized)
struct DecodeBatchPenaltyOrderingTests {

    /// A `LogitProcessor` spy that records the maximum value of the logits it
    /// is handed in `process(logits:)`. Used to verify the batched decode path
    /// applies penalties to the **raw** logits (which can be positive) rather
    /// than to already-normalized log-probs (which are <= 0), matching the
    /// single-stream `TokenIterator.convertToToken` ordering.
    final class MaxRecordingProcessor: LogitProcessor {
        let box: MaxBox
        init(box: MaxBox) { self.box = box }
        func prompt(_ prompt: MLXArray) {}
        func process(logits: MLXArray) -> MLXArray {
            box.observed = logits.max().item(Float.self)
            return logits
        }
        func didSample(token: MLXArray) {}
    }

    final class MaxBox { var observed: Float? = nil }

    /// A model whose final-position logits put a large **positive** value on
    /// one token, so raw logits and log-probs are clearly distinguishable
    /// (raw max > 0; log-prob max <= 0).
    final class PositiveLogitModel: Module, LanguageModel, KVCacheDimensionProvider {
        let vocabularySize = 8
        var kvHeads: [Int] { [1] }
        let peak: Float

        init(peak: Float = 12.0) { self.peak = peak }

        func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
            -> PrepareResult
        {
            .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
            let batchSize = inputs.dim(0)
            let sequenceLength = inputs.dim(1)
            var logits = Array(
                repeating: Float(0),
                count: batchSize * sequenceLength * vocabularySize
            )
            // Put `peak` on token index 1 at every position.
            for b in 0 ..< batchSize {
                for t in 0 ..< sequenceLength {
                    let base = (b * sequenceLength + t) * vocabularySize
                    logits[base + 1] = peak
                }
            }
            return MLXArray(logits).reshaped([batchSize, sequenceLength, vocabularySize])
        }
    }

    @Test("Penalty processor sees raw logits, not log-probs")
    func processorSeesRawLogits() {
        let box = MaxBox()
        let model = PositiveLogitModel(peak: 12.0)
        // A non-greedy fallback forces the sampler path (where processors run).
        let batch = DecodeBatch(
            model: model,
            uids: [0],
            seedTokens: MLXArray([UInt32(1)]),
            promptCache: [BatchKVCache(leftPadding: [0])],
            tokens: [[1]],
            maxTokens: [4],
            processors: [MaxRecordingProcessor(box: box)]
        )
        _ = batch.next()

        // The processor must have observed the raw logit peak (> 0), proving it
        // ran before log-softmax. If penalties were applied to log-probs the
        // observed max would be <= 0.
        let observed = try! #require(box.observed)
        #expect(observed > 1.0)
    }
}

// MARK: - Decode batch: matcher replay on adoption

@Suite(.serialized)
struct DecodeBatchMatcherReplayTests {

    @Test("Replaying generated history completes a partial stop sequence")
    func replayCompletesPartialStop() {
        // Stop on the two-token sequence [8, 9]. The migrated row already
        // produced `8` (the seed) while running single; the next batched token
        // is `9` (IncrementingLanguageModel: 8 -> 9), which must complete the
        // stop only because the prior `8` was replayed into the matcher.
        let machine = StopSequenceMatcher(
            states: ["normal": [(sequence: [8, 9], next: nil)]]
        )
        let batch = DecodeBatch(
            model: IncrementingLanguageModel(),
            uids: [0],
            seedTokens: MLXArray([UInt32(8)]),
            promptCache: [BatchKVCache(leftPadding: [0])],
            tokens: [[8]],
            maxTokens: [10],
            stateMachines: [machine],
            numTokens: [1],
            replayMatcherTokens: [[8]]
        )

        let responses = batch.next()
        #expect(responses.count == 1)
        #expect(responses[0].token == 9)
        #expect(responses[0].finishReason == .stop)
        #expect(responses[0].matchedSequence == [8, 9])
    }

    @Test("Without replay the same partial stop is missed")
    func withoutReplayPartialStopMissed() {
        let machine = StopSequenceMatcher(
            states: ["normal": [(sequence: [8, 9], next: nil)]]
        )
        let batch = DecodeBatch(
            model: IncrementingLanguageModel(),
            uids: [0],
            seedTokens: MLXArray([UInt32(8)]),
            promptCache: [BatchKVCache(leftPadding: [0])],
            tokens: [[8]],
            maxTokens: [10],
            stateMachines: [machine],
            numTokens: [1]
        )

        let responses = batch.next()
        #expect(responses.count == 1)
        #expect(responses[0].token == 9)
        // The matcher only saw `9`, never the preceding `8`, so no stop.
        #expect(responses[0].finishReason == nil)
    }
}

// MARK: - Decode batch: fallback sampler preservation on extend

@Suite(.serialized)
struct DecodeBatchExtendFallbackTests {

    @Test("Extending preserves the other batch's fallback for nil-sampler rows")
    func extendPreservesOtherFallback() {
        // `other` has a nil per-row sampler but a distinctive fallback that
        // always returns token 5, regardless of the logits. The left batch's
        // fallback is greedy. After `extend`, the appended row must still pick
        // token 5 (its own fallback), not resolve through the greedy fallback.
        let constantFive: RowSampler = { logits in
            let rows = logits.dim(0)
            return MLXArray(Array(repeating: Int32(5), count: rows))
        }

        let left = DecodeBatch(
            model: IncrementingLanguageModel(),
            uids: [0],
            seedTokens: MLXArray([UInt32(1)]),
            promptCache: [BatchKVCache(leftPadding: [0])],
            tokens: [[1]],
            maxTokens: [10],
            fallbackSampler: greedySampler,
            fallbackIsGreedy: true
        )

        let right = DecodeBatch(
            model: IncrementingLanguageModel(),
            uids: [1],
            seedTokens: MLXArray([UInt32(1)]),
            promptCache: [BatchKVCache(leftPadding: [0])],
            tokens: [[1]],
            maxTokens: [10],
            fallbackSampler: constantFive,
            fallbackIsGreedy: false
        )

        left.extend(right)

        let responses = left.next()
        let byUID = Dictionary(uniqueKeysWithValues: responses.map { ($0.uid, $0.token) })
        // Left row resolves through greedy (IncrementingLanguageModel: 1 -> 2).
        #expect(byUID[0] == 2)
        // Right row must keep its own fallback (constant 5), not greedy.
        #expect(byUID[1] == 5)
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
            model: CacheTopologyLanguageModel { _ in [ArraysCache(size: 3)] }
        )

        _ = try BatchGenerationEngine(
            model: CacheTopologyLanguageModel { _ in [MambaCache()] }
        )

        _ = try BatchGenerationEngine(
            model: CacheTopologyLanguageModel { _ in [RotatingKVCache(maxSize: 8, keep: 0)] }
        )

        _ = try BatchGenerationEngine(
            model: CacheTopologyLanguageModel { _ in [CacheList(MambaCache(), KVCacheSimple())] }
        )
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
        assertEngineRejectsCache(
            CacheList(MambaCache(), QuantizedKVCache()),
            expectedType: "QuantizedKVCache",
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
