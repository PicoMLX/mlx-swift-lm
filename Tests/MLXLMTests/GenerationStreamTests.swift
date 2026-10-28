// Copyright © 2026 Apple Inc.

import Cmlx
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// `synchronizeGeneration()` waits for exactly the work an iterator submitted with
/// `asyncEval`, on whichever streams it runs.
///
/// The work under test runs on a private CPU stream for a few hundred milliseconds, so
/// it is still running when a test checks it right after the iterator submits it.
@Suite(.serialized)
struct GenerationStreamTests {
    @Test("Waiting covers a step's cache work when its token was already available")
    func readyTokenWithPendingCacheWork() throws {
        let side = MLX.Stream(.cpu)
        let cache = StateCache()
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1)])),
            model: FixedLogitsModel(cache: cache),
            processor: nil, sampler: ClosureSampler { _ in MLXArray([Int32(3)]) },
            maxTokens: 1)
        cache.makeState = { slowWork(on: side) }

        #expect(iterator.next() == 3)
        let pending = try #require(cache.submitted.last)
        // The token limit is reached, so this submits nothing new.
        #expect(iterator.next() == nil)
        iterator.synchronizeGeneration()

        #expect(isAvailable(pending))
    }

    @Test("Waiting covers a token sampled on another stream")
    func tokenOnAnotherStream() throws {
        let side = MLX.Stream(.cpu)
        let sampled = SampledTokens()
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1)])),
            model: FixedLogitsModel(cache: StateCache()),
            processor: nil,
            sampler: ClosureSampler { logits in
                let scaled = multiply(logits, slowWork(on: side), stream: .stream(side))
                return sampled.record(argMax(scaled, axis: -1, stream: .stream(side)))
            },
            maxTokens: 2)

        #expect(iterator.next() == 3)
        let pending = try #require(sampled.tokens.last)
        iterator.synchronizeGeneration()

        #expect(isAvailable(pending))
    }

    @Test("Waiting covers earlier steps' work, not only the last step's")
    func earlierStepsStayCovered() throws {
        let side = MLX.Stream(.cpu)
        let cache = StateCache()
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1)])),
            model: FixedLogitsModel(cache: cache),
            processor: nil, sampler: ClosureSampler { _ in MLXArray([Int32(3)]) },
            maxTokens: 2)

        cache.makeState = { slowWork(on: side) }
        #expect(iterator.next() == 3)
        let earlier = try #require(cache.submitted.last)
        cache.makeState = { MLXArray(Int32(0)) }
        #expect(iterator.next() == 3)
        iterator.synchronizeGeneration()

        #expect(isAvailable(earlier))
    }

    @Test("Waiting excludes work someone else queues later on a shared stream")
    func laterUnrelatedWorkExcluded() throws {
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1)])),
            model: FixedLogitsModel(cache: StateCache()),
            processor: nil, sampler: ArgMaxSampler(), maxTokens: 1)
        #expect(iterator.next() == 3)

        let unrelated = slowWork(on: MLX.Stream.cpu)
        asyncEval(unrelated)
        iterator.synchronizeGeneration()

        #expect(!isAvailable(unrelated))
        eval(unrelated)
    }

    @Test("Waiting at the end reads no cache state and runs no forward")
    func waitingSubmitsNoNewWork() throws {
        let cache = StateCache()
        let model = FixedLogitsModel(cache: cache)
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1)])), model: model,
            processor: nil, sampler: ArgMaxSampler(), maxTokens: 1)
        #expect(iterator.next() == 3)
        let stateReads = cache.submitted.count
        let forwards = model.forwardCount

        iterator.synchronizeGeneration()

        #expect(cache.submitted.count == stateReads)
        #expect(model.forwardCount == forwards)
    }

    @Test("A speculative iterator waits for the draft token its prefill submitted")
    func speculativeDraftTokenCovered() throws {
        let side = MLX.Stream(.cpu)
        let draftModel = FixedLogitsModel(cache: StateCache())
        draftModel.makeLogits = {
            multiply(fixedLogits(), slowWork(on: side), stream: .stream(side))
        }
        let iterator = try SpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1)])),
            mainModel: FixedLogitsModel(cache: StateCache()), draftModel: draftModel,
            parameters: GenerateParameters(maxTokens: 0, temperature: 0), numDraftTokens: 2)
        let pending = iterator.draftY.tokens

        iterator.synchronizeGeneration()

        #expect(isAvailable(pending))
    }
}

/// About a few hundred milliseconds of CPU work on `stream`, evaluating to 1.
private func slowWork(on stream: MLX.Stream) -> MLXArray {
    let device = StreamOrDevice.stream(stream)
    let identity = MLX.eye(2048, stream: device)
    var x = MLXArray.ones([2048, 2048], stream: device)
    for _ in 0 ..< 25 {
        x = matmul(x, identity, stream: device)
    }
    return x.max(stream: device)
}

private func isAvailable(_ array: MLXArray) -> Bool {
    var available = false
    precondition(_mlx_array_is_available(&available, array.ctx) == 0)
    return available
}

private func fixedLogits() -> MLXArray {
    MLXArray([Float(0), 1, 2, 3]).reshaped([1, 1, 4])
}

/// Returns `makeLogits()` for prefill and every step, and uses `cache` as its cache.
private final class FixedLogitsModel: Module, LanguageModel {
    let cache: StateCache
    var makeLogits: () -> MLXArray = fixedLogits
    private(set) var forwardCount = 0

    init(cache: StateCache) {
        self.cache = cache
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [cache] }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .logits(.init(logits: makeLogits()))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forwardCount += 1
        return makeLogits()
    }
}

/// A cache whose state, which `TokenIterator.next()` submits with each token, comes from
/// `makeState`. `submitted` holds every state array it handed out.
private final class StateCache: BaseKVCache {
    var makeState: () -> MLXArray = { MLXArray(Int32(0)) }
    private(set) var submitted: [MLXArray] = []

    /// Speculative decoding refuses a draft cache it can't trim.
    override var isTrimmable: Bool { true }

    override var state: [MLXArray] {
        get {
            let state = makeState()
            submitted.append(state)
            return [state]
        }
        set {}
    }
}

private struct ClosureSampler: LogitSampler {
    let pick: (MLXArray) -> MLXArray

    func sample(logits: MLXArray) -> MLXArray { pick(logits) }
}

private final class SampledTokens {
    private(set) var tokens: [MLXArray] = []

    func record(_ token: MLXArray) -> MLXArray {
        tokens.append(token)
        return token
    }
}
