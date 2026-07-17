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

        #expect(handler.processToken(4) == .more)
        #expect(handler.processStopToken(8) == .more)
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

        #expect(handler.processToken(4) == .more)
        #expect(handler.processStopToken(8) == .more)
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

        #expect(handler.processToken(1) == .more)
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

    @Test("Text handler reports a semantic stop when a stop string completes")
    func textHandlerStopsOnStopString() async throws {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let handler = SchedulerTokenHandler.text(
            continuation: continuation,
            tokenizer: TestTokenizer(vocabulary: [1: "alpha", 2: "STOP", 3: "omega"]),
            stopStrings: ["STOP"],
            toolCallFormat: .json
        )

        let first = handler.processToken(1)
        _ = handler.processToken(2)
        // Whether the detokenizer released "STOP" on its own token or held it
        // until more text arrived, the filter has latched by now: parity with
        // TextToolTokenLoopHandler, which stops the loop at the match.
        let afterStop = handler.processToken(3)
        #expect(first == .more)
        #expect(afterStop == .stop)

        handler.processEndOfSequence()
        handler.finish()

        let text = await collectGenerations(stream).chunks.joined()
        #expect(text.contains("alpha"))
        #expect(!text.contains("STOP"))
        #expect(!text.contains("omega"))
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
            generatedTokenIds: [1, 2, 3],
            firstTokenAt: nil
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

    @Test("An empty topology (cacheless model) does not migrate")
    func emptyTopologyReturnsNil() {
        // BatchGenerationEngine.init rejects models with no per-layer caches;
        // a vacuous [] here would approve an upgrade the engine then refuses,
        // truncating the running request mid-generation.
        #expect(InferenceScheduler.migrateCaches([]) == nil)
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

    @Test("RowSampler bridge is nil (engine greedy fallback) at temperature 0")
    func rowSamplerGreedyAtZeroTemp() {
        var params = GenerateParameters()
        params.temperature = 0
        // Greedy parameters yield no per-row sampler so the engine's argMax
        // fast path stays reachable; the engine's greedy fallback covers the
        // row.
        #expect(InferenceScheduler.rowSampler(for: params) == nil)

        // Non-zero temperature yields a real sampler.
        params.temperature = 0.7
        let sampler = InferenceScheduler.rowSampler(for: params)
        #expect(sampler != nil)
        // A strongly peaked row should still pick the dominant token.
        let logits = MLXArray([Float(-100), 100, -100, -100]).reshaped([1, 4])
        if let sampler {
            let picked = sampler(logits)
            #expect(picked.asArray(Int32.self).first == 1)
        }
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
        #expect(BatchedGenerationError.batchTooSmall != .schedulerUnavailable)
    }
}

// MARK: - Seeded sampler PRNG continuity

@Suite("SamplerKeyHolder PRNG continuity")
struct SamplerKeyHolderTests {

    @Test("advance(by: n) matches a fresh holder that consumed n draws")
    func advanceMatchesConsumedDraws() throws {
        // The upgrade path fast-forwards a fresh holder past the draws the
        // single iterator already consumed; draw n+1 must be bit-identical
        // to the chain a live holder would continue.
        let consumed = SamplerKeyHolder(seed: 42)
        for _ in 0 ..< 5 {
            _ = consumed.next()
        }
        let advanced = SamplerKeyHolder(seed: 42)
        advanced.advance(by: 5)

        let expected = try #require(consumed.next())
        let actual = try #require(advanced.next())
        #expect(expected.asArray(UInt32.self) == actual.asArray(UInt32.self))
    }

    @Test("advance(by: 0) leaves the chain at its first draw")
    func advanceZeroIsIdentity() throws {
        let fresh = SamplerKeyHolder(seed: 7)
        let advanced = SamplerKeyHolder(seed: 7)
        advanced.advance(by: 0)

        let expected = try #require(fresh.next())
        let actual = try #require(advanced.next())
        #expect(expected.asArray(UInt32.self) == actual.asArray(UInt32.self))
    }

    @Test("Unseeded holder stays keyless through advance")
    func unseededAdvanceStaysNil() {
        let holder = SamplerKeyHolder(seed: nil)
        holder.advance(by: 3)
        // nil key = "use MLX's global PRNG"; advancing must not invent one.
        #expect(holder.next() == nil)
    }
}

// MARK: - Attach / detach lifecycle

@Suite("Scheduler attach lifecycle")
struct SchedulerAttachLifecycleTests {

    /// Stand-in for a `ModelContainer`: attach/detach key purely on
    /// `ObjectIdentifier`, so any class instance works as an owner.
    private final class Owner {}

    private func makeContext(name: String) -> ModelContext {
        ModelContext(
            configuration: ModelConfiguration(id: name),
            model: UnusedLanguageModel(),
            processor: StandInUserInputProcessor(),
            tokenizer: TestTokenizer(vocabularySize: 16)
        )
    }

    @Test("Detach releases the owner; a new owner attaches and replaces the context")
    func detachThenReattachReplacesContext() async throws {
        let scheduler = InferenceScheduler()
        // Keep both owners alive for the whole test so their identities can
        // never alias through address reuse.
        let a = Owner()
        let b = Owner()
        let ownerA = ObjectIdentifier(a)
        let ownerB = ObjectIdentifier(b)

        let attachedA = await scheduler.attach(
            owner: ownerA,
            context: SendableBox(makeContext(name: "test/model-a")),
            promptCache: nil
        )
        #expect(attachedA)
        #expect(await scheduler.currentOwner == ownerA)
        #expect(await scheduler.attachedModelName == "test/model-a")

        // A different container cannot steal the binding while A owns it.
        let stolen = await scheduler.attach(
            owner: ownerB,
            context: SendableBox(makeContext(name: "test/model-b")),
            promptCache: nil
        )
        #expect(stolen == false)

        // Detach from the wrong owner is a no-op.
        await scheduler.detach(owner: ownerB)
        #expect(await scheduler.currentOwner == ownerA)

        // Detach from the bound owner releases the binding; the context is
        // retained only for in-flight work.
        await scheduler.detach(owner: ownerA)
        #expect(await scheduler.currentOwner == nil)

        // A new owner now attaches — and REPLACES the retained context, so
        // its requests never run on the dead container's model.
        let attachedB = await scheduler.attach(
            owner: ownerB,
            context: SendableBox(makeContext(name: "test/model-b")),
            promptCache: nil
        )
        #expect(attachedB)
        #expect(await scheduler.currentOwner == ownerB)
        #expect(await scheduler.attachedModelName == "test/model-b")
    }

    @Test("Same-owner re-attach keeps the first context")
    func sameOwnerReattachIsIdempotent() async throws {
        let scheduler = InferenceScheduler()
        let a = Owner()
        let ownerA = ObjectIdentifier(a)

        #expect(
            await scheduler.attach(
                owner: ownerA,
                context: SendableBox(makeContext(name: "test/model-a")),
                promptCache: nil
            ))
        #expect(
            await scheduler.attach(
                owner: ownerA,
                context: SendableBox(makeContext(name: "test/model-a2")),
                promptCache: nil
            ))
        // The first-bound context stays (the second box is not consumed).
        #expect(await scheduler.attachedModelName == "test/model-a")
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

/// A structurally-conforming `LanguageModel` for lifecycle tests that never
/// run a decode. `callAsFunction` traps to make accidental use obvious
/// (mirrors `NoopLanguageModel` in the ModelContainer batching tests).
private final class UnusedLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize = 16
    var kvHeads: [Int] { [1] }

    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        fatalError("UnusedLanguageModel must not be run by lifecycle tests")
    }
}
