// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing
import os

@testable import MLXLMCommon

// Behavioral tests for prefix-cache admission on the batched path
// (`EngineDriver.admitViaPrefixFetch`): a request whose prompt prefix is in
// the LRU prompt cache must decode identical tokens to a cold request while
// prefilling only the remainder.

@Suite(.serialized)
struct PrefixFetchTests {

    @Test("Batched admission reuses a cached prompt prefix")
    func batchedAdmissionReusesPrefix() async throws {
        let prompt = [1, 2, 3, 4, 5]
        let prefix = [1, 2, 3]

        // Cold reference run: no prompt cache; the engine prefills the whole
        // prompt EXCEPT its final token (PrefillBatch seeds the last token
        // into the decode batch), so the recorder sees a width-(N-1) pass.
        let coldRecorder = WidthRecorder()
        let cold = try await runBatched(
            model: WidthRecordingLanguageModel(recorder: coldRecorder),
            prompt: prompt, promptCache: nil)
        #expect(coldRecorder.widths.contains(prompt.count - 1))

        // Warm run: seed the cache with the prefix's KV, then submit the full
        // prompt. Only the remainder may be prefilled: the widest forward
        // pass is the 1-token remainder chunk / decode steps, never the full
        // prompt (and never even the full remainder, since its last token
        // seeds the decode).
        let warmRecorder = WidthRecorder()
        let warmModel = WidthRecordingLanguageModel(recorder: warmRecorder)
        let promptCache = LRUPromptCache()
        let seedCaches = warmModel.newCache(parameters: nil)
        _ = warmModel.callAsFunction(
            MLXArray(prefix.map { UInt32($0) }).reshaped([1, prefix.count]),
            cache: seedCaches)
        eval(seedCaches.flatMap { $0.state })
        promptCache.insertCache(
            model: "test/model", tokens: prefix, promptCache: seedCaches)
        warmRecorder.reset()

        let warm = try await runBatched(
            model: warmModel, prompt: prompt, promptCache: promptCache)

        #expect(warm.tokens == cold.tokens)
        #expect(warm.info?.stopReason == .length)
        #expect(warm.info?.promptTokenCount == prompt.count)
        #expect(warmRecorder.maxWidth < prompt.count)
        #expect(warmRecorder.maxWidth <= prompt.count - prefix.count)
    }

    @Test("Exact-hit admission re-feeds the trimmed final token")
    func exactHitAdmissionRefeedsFinalToken() async throws {
        let prompt = [1, 2, 3, 4]

        let cold = try await runBatched(
            model: WidthRecordingLanguageModel(recorder: WidthRecorder()),
            prompt: prompt, promptCache: nil)

        // Cache an entry covering the ENTIRE prompt. Admission must trim the
        // final token's KV and re-feed it as the seed, not skip decoding.
        let warmRecorder = WidthRecorder()
        let warmModel = WidthRecordingLanguageModel(recorder: warmRecorder)
        let promptCache = LRUPromptCache()
        let seedCaches = warmModel.newCache(parameters: nil)
        _ = warmModel.callAsFunction(
            MLXArray(prompt.map { UInt32($0) }).reshaped([1, prompt.count]),
            cache: seedCaches)
        eval(seedCaches.flatMap { $0.state })
        promptCache.insertCache(
            model: "test/model", tokens: prompt, promptCache: seedCaches)
        warmRecorder.reset()

        let warm = try await runBatched(
            model: warmModel, prompt: prompt, promptCache: promptCache)

        #expect(warm.tokens == cold.tokens)
        #expect(warmRecorder.maxWidth <= 1)
    }

    // MARK: - Harness

    private struct RunResult {
        let tokens: [Int]
        let info: GenerateCompletionInfo?
    }

    /// Submit one request through an `EngineDriver`, drain it, and collect
    /// the raw token stream.
    private func runBatched(
        model: WidthRecordingLanguageModel,
        prompt: [Int],
        promptCache: LRUPromptCache?
    ) async throws -> RunResult {
        let engine = try BatchGenerationEngine(
            model: model,
            eosTokens: [[15]],
            defaultMaxTokens: 3,
            prefillBatchSize: 1,
            completionBatchSize: 2
        )
        let driver = EngineDriver(engine: SendableBox(engine), promptCache: nil)

        let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()
        let handler = SchedulerTokenHandler.rawToken(
            continuation: continuation, includeStopToken: false)

        let request = SchedulerRequest(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })),
            parameters: GenerateParameters(),
            inputTokens: prompt,
            modelName: "test/model",
            promptCache: promptCache,
            promptCacheSalt: 0,
            wiredMemoryTicket: nil
        )
        let uid = await driver.submit(
            request,
            handler: handler,
            cancelToken: 0,
            maxTokens: 3,
            sampler: nil,
            stateMachine: nil
        )
        #expect(uid != nil)
        await driver.drain()

        var tokens = [Int]()
        var info: GenerateCompletionInfo?
        for await generation in stream {
            switch generation {
            case .token(let token): tokens.append(token)
            case .info(let completionInfo): info = completionInfo
            }
        }
        return RunResult(tokens: tokens, info: info)
    }
}

// MARK: - Fixture

/// Checked-`Sendable` width log. Lives OUTSIDE the model's non-`Sendable`
/// region so the test can keep reading it after the model is transferred
/// into the driver actor (Sendable references are excluded from region
/// transfer).
private final class WidthRecorder: Sendable {
    private let storage = OSAllocatedUnfairLock(initialState: [Int]())
    func record(_ width: Int) { storage.withLock { $0.append(width) } }
    func reset() { storage.withLock { $0.removeAll() } }
    var widths: [Int] { storage.withLock { $0 } }
    var maxWidth: Int { widths.max() ?? 0 }
}

/// Records the sequence width of every forward pass. Emits a +4 logit at
/// `(token + 1) % vocab` so decoding is deterministic: a full-prompt prefill
/// shows up as a width-N call, a cache-hit admission never exceeds the
/// remainder width.
private final class WidthRecordingLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize = 16
    var kvHeads: [Int] { [1] }
    let recorder: WidthRecorder

    init(recorder: WidthRecorder) {
        self.recorder = recorder
        super.init()
    }

    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let batchSize = inputs.dim(0)
        let sequenceLength = inputs.dim(1)
        recorder.record(sequenceLength)

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
