// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@preconcurrency @testable import MLXLMCommon
import MLXNN
import Testing

// These suites exercise the PR4 scheduler/driver pieces that do NOT require a
// Metal device: the token-handler fan-out, the de-`@unchecked`'d UpgradeFlag
// handshake, the single→batch cache-migration dispatch, and the request/value
// types. End-to-end auto-upgrade with real decoding is covered by the macOS CI
// lane (it needs a GPU); there is intentionally no MLXMetalGuard here.

// MARK: - SchedulerTokenHandler

@Suite("Scheduler Token Handler")
struct SchedulerTokenHandlerTests {

    @Test("Raw token handler emits tokens, stop token, and info")
    func rawTokenHandlerEmitsTokensAndInfo() async throws {
        let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()
        let handler = SchedulerTokenHandler.rawToken(
            continuation: continuation,
            includeStopToken: true
        )

        #expect(handler.processToken(4))
        #expect(handler.processStopToken(8))
        handler.yieldInfo(
            GenerateCompletionInfo(
                promptTokenCount: 2,
                generationTokenCount: 2,
                promptTime: 0.1,
                generationTime: 0.2,
                stopReason: .stop
            )
        )
        handler.finish()

        let collected = await collectTokenGenerations(stream)
        let info = try #require(collected.info)
        #expect(collected.tokens == [4, 8])
        #expect(info.generationTokenCount == 2)
        #expect(info.stopReason == .stop)
    }

    @Test("Raw token handler omits stop token when not requested")
    func rawTokenHandlerOmitsStopToken() async throws {
        let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()
        let handler = SchedulerTokenHandler.rawToken(
            continuation: continuation,
            includeStopToken: false
        )

        #expect(handler.processToken(4))
        #expect(handler.processStopToken(8))
        handler.finish()

        let collected = await collectTokenGenerations(stream)
        #expect(collected.tokens == [4])
    }

    @Test("Text handler emits decoded chunks and completion info")
    func textHandlerEmitsChunksAndInfo() async throws {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let handler = SchedulerTokenHandler.text(
            continuation: continuation,
            tokenizer: TestTokenizer(vocabularySize: 20),
            toolCallFormat: .json
        )

        #expect(handler.processToken(1))
        handler.processEndOfSequence()
        handler.yieldInfo(
            GenerateCompletionInfo(
                promptTokenCount: 1,
                generationTokenCount: 1,
                promptTime: 0.05,
                generationTime: 0.05,
                stopReason: .length
            )
        )
        handler.finish()

        let collected = await collectGenerations(stream)
        let info = try #require(collected.info)
        #expect(info.stopReason == .length)
    }
}

// MARK: - UpgradeFlag handshake

@Suite("UpgradeFlag handshake")
struct UpgradeFlagTests {

    @Test("Deposit before request resolves the awaiting scheduler")
    func depositResolvesContinuation() async throws {
        let flag = InferenceScheduler.UpgradeFlag()

        // Scheduler side awaits the snapshot; single side deposits it.
        async let awaited: SendableBox<InferenceScheduler.LiveIteratorState>? =
            withCheckedContinuation { continuation in
                flag.requestUpgrade(continuation: continuation)
            }

        // Spin until the request flag is observed, then deposit.
        while !flag.upgradeRequested {
            await Task.yield()
        }
        #expect(flag.upgradeRequested)

        let snapshot = InferenceScheduler.LiveIteratorState(
            cache: [],
            currentToken: 7,
            tokenCount: 3,
            maxTokens: 10,
            parameters: GenerateParameters(),
            generatedTokenIds: [1, 2, 3]
        )
        flag.depositLiveState(SendableBox(snapshot))

        let result = try #require(await awaited).consume()
        #expect(result.currentToken == 7)
        #expect(result.tokenCount == 3)
        #expect(result.generatedTokenIds == [1, 2, 3])
    }

    @Test("Task finishing before request resolves the scheduler with nil")
    func taskFinishedResolvesNil() async throws {
        let flag = InferenceScheduler.UpgradeFlag()

        // The single task finished first.
        flag.markTaskFinished()

        // A later upgrade request must resume immediately with nil so the
        // scheduler does not hang.
        let result: SendableBox<InferenceScheduler.LiveIteratorState>? =
            await withCheckedContinuation { continuation in
                flag.requestUpgrade(continuation: continuation)
            }
        #expect(result == nil)
    }

    @Test("markTaskFinished resumes a pending request with nil")
    func markTaskFinishedResumesPending() async throws {
        let flag = InferenceScheduler.UpgradeFlag()

        async let awaited: SendableBox<InferenceScheduler.LiveIteratorState>? =
            withCheckedContinuation { continuation in
                flag.requestUpgrade(continuation: continuation)
            }

        while !flag.upgradeRequested {
            await Task.yield()
        }
        // The task exits its loop without depositing state.
        flag.markTaskFinished()

        let result = await awaited
        #expect(result == nil)
    }
}

// MARK: - Cache migration dispatch

@Suite("Single→batch cache migration")
struct CacheMigrationTests {

    @Test("Simple cache migrates to a batched cache")
    func simpleMigrates() {
        let migrated = InferenceScheduler.migrateCaches([KVCacheSimple()])
        let caches = try? #require(migrated)
        #expect(caches?.count == 1)
        #expect(caches?.first is BatchKVCache)
    }

    @Test("Rotating cache (keep=0 and keep>0) migrates")
    func rotatingMigrates() {
        let zero = InferenceScheduler.migrateCaches([RotatingKVCache(maxSize: 8, keep: 0)])
        #expect(zero?.first is BatchRotatingKVCache)

        let keep = InferenceScheduler.migrateCaches([RotatingKVCache(maxSize: 8, keep: 4)])
        #expect(keep?.first is BatchRotatingKVCache)
    }

    @Test("Unsupported topologies fall back to nil (fresh request)")
    func unsupportedReturnsNil() {
        // Quantized/chunked caches have no fromSingle migration.
        #expect(InferenceScheduler.migrateCaches([QuantizedKVCache()]) == nil)
        #expect(InferenceScheduler.migrateCaches([ChunkedKVCache()]) == nil)
    }

    @Test("A mixed list with one unsupported layer fails the whole migration")
    func mixedListFailsClosed() {
        let mixed: [KVCache] = [KVCacheSimple(), QuantizedKVCache()]
        #expect(InferenceScheduler.migrateCaches(mixed) == nil)
    }
}

// MARK: - Configuration & value types

@Suite("Scheduler configuration & request values")
struct SchedulerValueTests {

    @Test("Configuration defaults are sensible")
    func configurationDefaults() {
        let config = InferenceScheduler.Configuration()
        #expect(config.completionBatchSize == 32)
        #expect(config.prefillBatchSize == 8)
        #expect(config.prefillStepSize == 2048)
        #expect(config.maxBatchSize == nil)
    }

    @Test("RowSampler bridge is greedy at temperature 0")
    func rowSamplerGreedyAtZeroTemp() {
        var params = GenerateParameters()
        params.temperature = 0
        // A greedy sampler over a one-hot row must pick that row's argmax.
        let sampler = InferenceScheduler.rowSampler(for: params)
        let logits = MLXArray([Float(-1), 5, -1, -1]).reshaped([1, 4])
        let picked = sampler(logits)
        #expect(picked.asArray(Int32.self).first == 1)
    }

    @Test("GenerationRequest carries the full LMInput and defaults")
    func generationRequestDefaults() {
        let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))
        let request = GenerationRequest(input: input)
        #expect(request.input.text.tokens.size == 3)
        #expect(request.promptCacheSalt == 0)
        #expect(request.promptCache == nil)
        #expect(request.wiredMemoryTicket == nil)
    }

    @Test("BatchedGenerationError values compare by case")
    func batchedErrorEquality() {
        #expect(BatchedGenerationError.batchTooSmall == .batchTooSmall)
        let incompatible = BatchedGenerationError.incompatibleRequests([1, 2])
        #expect(incompatible == .incompatibleRequests([1, 2]))
        #expect(BatchedGenerationError.batchTooSmall != .schedulerBusy)
    }
}

// MARK: - Helpers

private struct CollectedGeneration {
    let chunks: [String]
    let info: GenerateCompletionInfo?
}

private struct CollectedTokenGeneration {
    let tokens: [Int]
    let info: GenerateCompletionInfo?
}

private func collectGenerations(_ stream: AsyncStream<Generation>) async -> CollectedGeneration {
    var chunks = [String]()
    var info: GenerateCompletionInfo?
    for await generation in stream {
        switch generation {
        case .chunk(let chunk): chunks.append(chunk)
        case .info(let completionInfo): info = completionInfo
        case .toolCall: break
        }
    }
    return CollectedGeneration(chunks: chunks, info: info)
}

private func collectTokenGenerations(_ stream: AsyncStream<TokenGeneration>) async
    -> CollectedTokenGeneration
{
    var tokens = [Int]()
    var info: GenerateCompletionInfo?
    for await generation in stream {
        switch generation {
        case .token(let token): tokens.append(token)
        case .info(let completionInfo): info = completionInfo
        }
    }
    return CollectedTokenGeneration(tokens: tokens, info: info)
}
