// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing
import os

@preconcurrency @testable import MLXLMCommon

// End-to-end auto-upgrade tests through the PUBLIC front door: a
// `ModelContainer` with an `InferenceScheduler` over a deterministic
// fixture model. Two concurrent `generate` calls force the scheduler's
// `.single → .upgrading → .batched` handshake (live-state deposit, cache
// migration, engine adoption), and the tests assert the upgraded request's
// full token stream is IDENTICAL to a solo baseline — pinning the handoff
// accounting (no dropped, duplicated, or reordered tokens), the completion
// info, and the return to idle. Nothing else in the suite drives two
// concurrent requests through `InferenceScheduler.route`/`upgrade`.
//
// Determinism: the fixture model's next token is a pure function of the
// previous token (increment mod vocab), so any token-accounting slip at the
// upgrade boundary shows up as a text mismatch against the solo baseline. The
// interleaving is made deterministic by an armed gate in the model: the
// fixture BLOCKS on a chosen forward call until the test releases it, so the
// first request is provably still mid-decode when the second is submitted
// (no sleep-based timing races); a decode-step gate in the model plus an
// instrumented `newCache` (the scheduler's `canUpgrade` topology probe) tell
// the test exactly when the second request has reached the upgrade path.
//
// Pattern after PrefixFetchTests (instrumented model + real driver), but one
// level up: the public ModelContainer/InferenceScheduler surface.
@Suite("Scheduler upgrade parity", .serialized)
struct SchedulerUpgradeParityTests {

    private let promptA = [1, 2, 3]
    private let promptB = [9, 10]

    // MARK: - Auto-upgrade token parity + completion infos

    @Test("Auto-upgrade preserves both requests' token streams and completion infos")
    func upgradeTokenParity() async throws {
        let maxA = 24
        let maxB = 12

        // Solo baselines on fresh containers (pure single-stream path).
        let soloA = try await runSolo(prompt: promptA, maxTokens: maxA)
        let soloB = try await runSolo(prompt: promptB, maxTokens: maxB)

        // The fixture is deterministic, so the baselines are known exactly:
        // each generated token is (previous + 1) mod 16.
        #expect(soloA.text == expectedText(prompt: promptA, count: maxA))
        #expect(soloB.text == expectedText(prompt: promptB, count: maxB))
        #expect(soloA.info?.stopReason == .length)
        #expect(soloA.info?.promptTokenCount == promptA.count)
        #expect(soloA.info?.generationTokenCount == maxA)
        #expect(soloB.info?.stopReason == .length)
        #expect(soloB.info?.generationTokenCount == maxB)

        // Concurrent run: A first; once a few of A's tokens have arrived and
        // A is provably paused mid-decode, submit B to force the upgrade.
        let harness = makeHarness(stepDelay: 0.001)
        harness.probe.armPause(afterForwardCalls: 8)
        defer { harness.probe.release() }

        let streamA = try await harness.container.generate(
            input: LMInput(tokens: MLXArray(promptA.map { Int32($0) })),
            parameters: GenerateParameters(maxTokens: maxA, temperature: 0)
        )

        var collectedA = CollectedUpgradeStream()
        var iteratorA = streamA.makeAsyncIterator()
        while collectedA.chunks.count < 3, let generation = await iteratorA.next() {
            collectedA.append(generation)
        }
        #expect(collectedA.chunks.count == 3, "first request never produced tokens")

        let container = harness.container
        let promptB = self.promptB
        let taskB = Task {
            try await container.generate(
                input: LMInput(tokens: MLXArray(promptB.map { Int32($0) })),
                parameters: GenerateParameters(maxTokens: maxB, temperature: 0)
            )
        }

        // B's routing probes the model's cache topology (`canUpgrade` calls
        // `newCache`) with no suspension before the upgrade handshake is
        // requested — call #1 was A's own TokenIterator, so a second call
        // proves B reached the scheduler's upgrade path. Only then let the
        // paused single task proceed and deposit its live state.
        let routed = await waitUntil { harness.probe.newCacheCallCount >= 2 }
        #expect(routed, "second request never reached the scheduler's upgrade path")
        try? await Task.sleep(nanoseconds: 20_000_000)
        harness.probe.release()

        let streamB = try await taskB.value
        let collectedB = await collect(streamB)
        while let generation = await iteratorA.next() {
            collectedA.append(generation)
        }

        // Upgrade witness: some decode step ran with BOTH rows in one batch,
        // i.e. the handshake completed and B joined A's migrated batch (had A
        // finished before B arrived, the batch would never exceed width 1).
        #expect(harness.probe.maxBatchWidth == 2, "requests never decoded in one batch")

        // Token parity across the .single → .upgrading → .batched handshake:
        // the upgraded request's stream must be indistinguishable from its
        // solo run, and the joiner's from its own.
        #expect(collectedA.text == soloA.text)
        #expect(collectedB.text == soloB.text)

        // Completion infos survive the handoff with correct accounting.
        let infoA = try #require(collectedA.info)
        #expect(infoA.promptTokenCount == promptA.count)
        #expect(infoA.generationTokenCount == maxA)
        #expect(infoA.stopReason == .length)
        #expect(infoA.promptTime >= 0)
        #expect(infoA.promptTime.isFinite)
        #expect(infoA.tokensPerSecond.isFinite, "tokens/sec must not be +inf")

        let infoB = try #require(collectedB.info)
        #expect(infoB.promptTokenCount == promptB.count)
        #expect(infoB.generationTokenCount == maxB)
        #expect(infoB.stopReason == .length)
        #expect(infoB.promptTime >= 0)
        #expect(infoB.promptTime.isFinite)
        #expect(infoB.tokensPerSecond.isFinite, "tokens/sec must not be +inf")

        // Both rows finished: the scheduler must be idle again.
        let idle = await waitUntil { await harness.scheduler.activeRequestCount == 0 }
        #expect(idle, "scheduler did not return to idle after the batch drained")
    }

    // MARK: - Joiner cancellation

    @Test("Dropping the joiner's stream leaves the upgraded request intact and returns to idle")
    func cancelledJoinerLeavesFirstRequestIntact() async throws {
        let maxA = 24
        let maxB = 20

        let soloA = try await runSolo(prompt: promptA, maxTokens: maxA)
        #expect(soloA.info?.generationTokenCount == maxA)

        let harness = makeHarness(stepDelay: 0.002)
        harness.probe.armPause(afterForwardCalls: 8)
        defer { harness.probe.release() }

        let streamA = try await harness.container.generate(
            input: LMInput(tokens: MLXArray(promptA.map { Int32($0) })),
            parameters: GenerateParameters(maxTokens: maxA, temperature: 0)
        )

        var collectedA = CollectedUpgradeStream()
        var iteratorA = streamA.makeAsyncIterator()
        while collectedA.chunks.count < 3, let generation = await iteratorA.next() {
            collectedA.append(generation)
        }
        #expect(collectedA.chunks.count == 3)

        let container = harness.container
        let promptB = self.promptB
        let taskB = Task {
            try await container.generate(
                input: LMInput(tokens: MLXArray(promptB.map { Int32($0) })),
                parameters: GenerateParameters(maxTokens: maxB, temperature: 0)
            )
        }

        let routed = await waitUntil { harness.probe.newCacheCallCount >= 2 }
        #expect(routed, "second request never reached the scheduler's upgrade path")
        try? await Task.sleep(nanoseconds: 20_000_000)
        harness.probe.release()

        // Consume exactly one of B's chunks, then drop the stream — the
        // consumer-cancellation path (`onTermination` → scheduler cancel →
        // driver row removal). The stream is deliberately never bound in this
        // scope so no reference outlives the helper.
        let gotFirstB = await consumeFirstChunkAndDrop(try await taskB.value)
        #expect(gotFirstB, "joiner never produced a token before cancellation")

        // The upgraded request must be completely unaffected by the
        // mid-batch cancellation of its batch-mate.
        while let generation = await iteratorA.next() {
            collectedA.append(generation)
        }
        #expect(harness.probe.maxBatchWidth == 2, "requests never decoded in one batch")
        #expect(collectedA.text == soloA.text)

        let infoA = try #require(collectedA.info)
        #expect(infoA.generationTokenCount == maxA)
        #expect(infoA.stopReason == .length)

        // The cancelled row must not strand the scheduler: back to idle.
        let idle = await waitUntil { await harness.scheduler.activeRequestCount == 0 }
        #expect(idle, "scheduler did not return to idle after cancellation")
    }

    // MARK: - Harness

    private struct Harness {
        let container: ModelContainer
        let scheduler: InferenceScheduler
        let probe: DecodeProbe
    }

    private func makeHarness(stepDelay: TimeInterval = 0) -> Harness {
        let probe = DecodeProbe()
        let model = GatedIncrementingLanguageModel(probe: probe, stepDelay: stepDelay)
        let vocabulary = Dictionary(
            uniqueKeysWithValues: (0 ..< model.vocabularySize).map { ($0, "t\($0)") })
        let context = ModelContext(
            configuration: ModelConfiguration(id: "test/incrementing"),
            model: model,
            processor: StandInUserInputProcessor(),
            tokenizer: TestTokenizer(vocabulary: vocabulary)
        )
        let scheduler = InferenceScheduler()
        let container = ModelContainer(context: context, scheduler: scheduler)
        return Harness(container: container, scheduler: scheduler, probe: probe)
    }

    /// Run a single request alone on a fresh container (never upgraded) and
    /// collect its full stream.
    private func runSolo(prompt: [Int], maxTokens: Int) async throws -> CollectedUpgradeStream {
        let harness = makeHarness()
        let stream = try await harness.container.generate(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })),
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0)
        )
        return await collect(stream)
    }

    /// The exact text the fixture must produce for `prompt`: `count` tokens of
    /// (previous + 1) mod 16, decoded by the deterministic test vocabulary.
    private func expectedText(prompt: [Int], count: Int) -> String {
        var tokens: [Int] = []
        var current = prompt.last ?? 0
        for _ in 0 ..< count {
            current = (current + 1) % 16
            tokens.append(current)
        }
        return tokens.map { "t\($0)" }.joined(separator: " ")
    }

    private func collect(_ stream: AsyncStream<Generation>) async -> CollectedUpgradeStream {
        var collected = CollectedUpgradeStream()
        for await generation in stream {
            collected.append(generation)
        }
        return collected
    }

    /// Iterate a stream until its first `.chunk`, then return — dropping the
    /// only reference so the scheduler sees a consumer cancellation.
    private func consumeFirstChunkAndDrop(_ stream: AsyncStream<Generation>) async -> Bool {
        for await generation in stream {
            if case .chunk = generation {
                return true
            }
        }
        return false
    }

    /// Poll `condition` until it holds or `timeout` elapses.
    private func waitUntil(
        timeout: TimeInterval = 20,
        _ condition: () async -> Bool
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if await condition() { return true }
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        return await condition()
    }
}

// MARK: - Collected stream

private struct CollectedUpgradeStream {
    var chunks: [String] = []
    var info: GenerateCompletionInfo?

    /// The full decoded output. Chunk boundaries differ between the single
    /// and batched paths, so parity is asserted on the joined text (the test
    /// vocabulary is injective: equal text ⇔ equal token sequence).
    var text: String { chunks.joined() }

    mutating func append(_ generation: Generation) {
        switch generation {
        case .chunk(let chunk): chunks.append(chunk)
        case .info(let completionInfo): info = completionInfo
        case .toolCall: break
        }
    }
}

// MARK: - Instrumented fixture

/// Checked-`Sendable` probe shared between the test and the fixture model.
/// Lives OUTSIDE the model's non-`Sendable` region so the test can keep
/// reading it after the model is transferred into the container/scheduler
/// (same move as PrefixFetchTests' WidthRecorder).
///
/// Also carries the deterministic interleaving gate: `armPause` makes the
/// model BLOCK on the given forward call until `release()` (or a generous
/// safety timeout), so a test can hold the first request mid-decode while it
/// submits the second — no sleep-based timing races.
private final class DecodeProbe: Sendable {
    private struct State {
        var forwardCalls = 0
        var maxBatchWidth = 0
        var newCacheCalls = 0
        var pauseAfterForwardCalls: Int?
        var released = false
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    /// The widest batch dimension seen by any forward call.
    var maxBatchWidth: Int { state.withLock { $0.maxBatchWidth } }

    /// Number of `newCache` calls (TokenIterator construction, the
    /// scheduler's `canUpgrade` topology probe, engine factories).
    var newCacheCallCount: Int { state.withLock { $0.newCacheCalls } }

    func armPause(afterForwardCalls calls: Int) {
        state.withLock {
            $0.pauseAfterForwardCalls = calls
            $0.released = false
        }
    }

    func release() {
        state.withLock { $0.released = true }
    }

    func recordNewCache() {
        state.withLock { $0.newCacheCalls += 1 }
    }

    /// Called at the top of every model forward. Records the batch width and,
    /// on the exact armed call, blocks the decode thread until `release()`
    /// (bounded by a safety timeout so a failing test cannot hang CI).
    func onForward(batchWidth: Int) {
        let shouldBlock = state.withLock { s -> Bool in
            s.forwardCalls += 1
            s.maxBatchWidth = max(s.maxBatchWidth, batchWidth)
            guard let pauseAt = s.pauseAfterForwardCalls else { return false }
            return s.forwardCalls == pauseAt && !s.released
        }
        guard shouldBlock else { return }

        let deadline = Date().addingTimeInterval(30)
        while Date() < deadline {
            if state.withLock({ $0.released }) { return }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }
}

/// Deterministic "increment the input token" model (the BatchEngineTests
/// fixture shape) with the probe wired into `newCache` and every forward
/// pass, plus an optional per-step delay that keeps decode slow enough for
/// the tests' interleaving to be observable. Greedy decoding of any history
/// is a pure function of the previous token, so token parity across the
/// upgrade handshake is exactly checkable.
private final class GatedIncrementingLanguageModel: Module, LanguageModel,
    KVCacheDimensionProvider
{
    let vocabularySize = 16
    var kvHeads: [Int] { [1] }

    let probe: DecodeProbe
    let stepDelay: TimeInterval

    init(probe: DecodeProbe, stepDelay: TimeInterval) {
        self.probe = probe
        self.stepDelay = stepDelay
        super.init()
    }

    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        probe.recordNewCache()
        return [KVCacheSimple()]
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        probe.onForward(batchWidth: inputs.dim(0))
        if stepDelay > 0 {
            Thread.sleep(forTimeInterval: stepDelay)
        }

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
