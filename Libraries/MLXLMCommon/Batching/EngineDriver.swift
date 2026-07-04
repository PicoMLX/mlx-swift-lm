// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The single serial executor that owns the non-thread-safe
/// ``BatchGenerationEngine`` and (optionally) the cross-request prompt cache.
///
/// ## Why this exists
///
/// ``BatchGenerationEngine``, ``DecodeBatch``, and the per-layer
/// ``BatchedCache`` instances are **not** `Sendable` and **not** thread-safe.
/// They must be touched by exactly one executor. `EngineDriver` is an `actor`,
/// so every method below runs with mutual exclusion: `submit`, `cancel`,
/// `adopt`, and the decode loop never overlap. The decode loop yields
/// (`await Task.yield()`) **between** engine steps, which lets queued actor
/// messages (a new `submit`, a `cancel`) run *between* steps — never during
/// one. Engine/cache mutation is therefore serialized with zero locks and zero
/// `@unchecked Sendable` (the Swift-native equivalent of the GCD `engineQueue`
/// used by server-shaped forks, but without their `NSLock`).
///
/// ## Token fan-out
///
/// Only `Sendable` data leaves the actor: each decode step produces
/// `BatchStepResult` values (`uid`/`token: Int`/`finishReason:
/// GenerateStopReason`), which are pushed to per-request
/// ``SchedulerTokenHandler`` closures (themselves `Sendable`). The
/// non-`Sendable` ``FinishedRowCache`` produced for prompt-cache write-back is
/// consumed **here on the driver**, never across a boundary.
///
/// ``InferenceScheduler`` holds policy/state and sends commands to this driver.
actor EngineDriver {

    /// Per-request bookkeeping the driver needs to route tokens and finalize a
    /// finished row. The handler is `Sendable`; the rest are value types.
    private struct RequestRecord {
        let handler: SchedulerTokenHandler
        /// Scheduler-owned stable id, used to cancel this row by token (e.g. on
        /// stream cancellation) without the caller knowing the engine UID.
        let cancelToken: Int
        let promptTokenCount: Int
        let inputTokens: [Int]
        let modelName: String
        let promptCacheSalt: UInt64
        /// The request's own prompt-cache instance (GenerationRequest
        /// documents `promptCache` as the cache to use). Preferred over the
        /// driver-level cache for this row's write-back; the auto-routed
        /// ModelContainer path sets both to the same instance.
        let promptCache: (any PromptCaching)?
        let submitTime: TimeInterval
        /// Whether to write this request's final KV state back to the prompt
        /// cache when it finishes.
        let writeBackToPromptCache: Bool
        /// Whether the engine's `allTokens` for this row includes the prompt
        /// (true for freshly-prefilled rows, where `PrefillBatch` appends the
        /// prompt; false for adopted/upgraded rows, whose history is seeded with
        /// generated-only tokens). Controls how `generationTokenCount` is
        /// derived in the completion info.
        let tokensIncludePrompt: Bool
        /// When the row's first token reached its handler. Splits the
        /// completion info's timing into admission-to-first-token
        /// (reported as `promptTime`: queue wait + prefill) and pure decode
        /// time; for adopted rows the earlier single-stream tokens predate
        /// the driver, so this marks the first *batched* token.
        var firstTokenAt: TimeInterval?
        /// Tokens delivered to this row's handler so far. Used as the
        /// generated-token count when a stop-string match finishes the row
        /// mid-stream (no engine `allTokens` is available on that path).
        var deliveredTokenCount: Int = 0
    }

    /// The number of generated tokens for a finished row, excluding prompt
    /// tokens when the engine's history includes them and excluding a final
    /// stop token that was appended to the history but suppressed from the
    /// stream (the single-stream loop checks EOS before counting, so it never
    /// counts the stop token — this keeps the batched path consistent).
    private func generationCount(
        _ record: RequestRecord, allTokens: [Int]?, suppressedStopToken: Bool
    ) -> Int {
        var total = allTokens?.count ?? 0
        if record.tokensIncludePrompt {
            total -= record.promptTokenCount
        }
        if suppressedStopToken {
            total -= 1
        }
        return max(0, total)
    }

    /// Whether a finishing response's stop token is appended to the row history
    /// but never delivered to the consumer, so it must not be counted as a
    /// generated token. Only stop-reason finishes suppress a token, and only
    /// when the handler does not surface it (decoded text mode, or raw-token
    /// mode with `includeStopToken: false`).
    private func suppressesStopToken(
        _ record: RequestRecord, reason: GenerateStopReason
    ) -> Bool {
        guard reason == .stop else { return false }
        switch record.handler.mode {
        case .decoded:
            return true
        case .rawTokens(let includeStopToken):
            return !includeStopToken
        }
    }

    /// The sole owner of the engine. Never escapes the actor.
    private let engine: BatchGenerationEngine

    /// Optional cross-request prompt cache. Owned here so `FinishedRowCache`
    /// write-back happens on this executor (the cache itself is `Sendable`).
    private let promptCache: (any PromptCaching)?

    /// uid → routing/finalization record for every admitted request.
    private var records: [Int: RequestRecord] = [:]

    /// Requests accepted but not yet admitted to the engine because the batch
    /// is at `maxBatchSize`. Drained in FIFO order as rows finish.
    private var waiting: [PendingSubmission] = []

    /// Runtime admission cap. `nil` means unbounded (the default). Clamped to
    /// `>= 1` on set so admission can never deadlock.
    private var maxBatchSize: Int?

    /// Whether a decode loop is currently running, so `submit`/`adopt` does not
    /// start a second one (the loop already drains all work).
    private var draining = false

    /// The most recently supplied wired-memory ticket from an admitted request.
    /// The decode loop runs each step under this limit when set, so callers that
    /// rely on `wiredMemoryTicket` to bound generation still get a memory policy
    /// on the batched path. (A batch may serve several requests; the latest
    /// non-nil ticket governs — tickets are a coordination mechanism, not
    /// per-row state.)
    private var activeWiredMemoryTicket: WiredMemoryTicket?

    /// A request waiting for batch capacity. Carries everything `submit` needs
    /// to admit it later.
    private struct PendingSubmission {
        let request: SchedulerRequest
        let handler: SchedulerTokenHandler
        let cancelToken: Int
        let maxTokens: Int
        let sampler: RowSampler?
        let processorSource: RowProcessorSource?
        let stateMachine: StopSequenceMatcher?
    }

    /// The engine arrives in a `SendableBox` because it is constructed from
    /// the scheduler actor's region (it references the shared model); the box
    /// is the repo's established hand-off for exactly-once transfer of
    /// non-`Sendable` values across isolation.
    init(engine: SendableBox<BatchGenerationEngine>, promptCache: (any PromptCaching)? = nil) {
        self.engine = engine.consume()
        self.promptCache = promptCache
    }

    // MARK: - Stats

    /// Number of requests currently decoding in the engine.
    var activeCount: Int { engine.activeCount }

    /// Number of requests queued inside the engine awaiting prefill.
    var engineQueuedCount: Int { engine.queuedCount }

    /// Number of requests held by the driver awaiting batch capacity.
    var waitingCount: Int { waiting.count }

    /// The current admission cap (`nil` = unbounded).
    var currentMaxBatchSize: Int? { maxBatchSize }

    /// Whether the driver currently has any in-flight or queued work.
    var hasWork: Bool { engine.hasWork || !waiting.isEmpty }

    // MARK: - Admission control

    /// Note a wired-memory ticket to govern the decode loop. Used by the
    /// scheduler's upgrade path to carry the migrated single request's ticket
    /// onto the batched loop (the row bypasses `admit`, which is where freshly
    /// admitted requests' tickets are captured).
    func noteWiredMemoryTicket(_ ticket: WiredMemoryTicket?) {
        if let ticket {
            activeWiredMemoryTicket = ticket
        }
    }

    /// Set the runtime admission cap. Modeled on Layr's `setMaxNumSeqs(_:)`:
    /// values `< 1` are clamped to `1` so a zero/negative cap cannot deadlock
    /// admission. Lowering the cap below the live row count does not evict
    /// running rows; it just stops new admissions until rows finish. Raising it
    /// immediately admits any waiting requests it now has room for.
    func setMaxBatchSize(_ size: Int?) {
        if let size {
            maxBatchSize = max(1, size)
        } else {
            maxBatchSize = nil
        }
        admitWaitingIfPossible()
    }

    /// Free admission slots given the current cap and live row count.
    /// Returns `Int.max` when unbounded.
    private var freeSlots: Int {
        guard let cap = maxBatchSize else { return Int.max }
        return max(0, cap - engine.activeCount - engine.queuedCount)
    }

    // MARK: - Submit

    /// Admit a request to the engine, registering `handler` for token fan-out.
    /// If the batch is at capacity the request is queued inside the driver and
    /// admitted later as rows finish. Returns the engine UID once admitted, or
    /// `nil` if the request is currently waiting for capacity (it will be
    /// admitted automatically once a slot frees, or dropped on ``close()``).
    ///
    /// The request is transferred with `sending` (region-based isolation): it
    /// carries non-`Sendable` `MLXArray`/`KVCache` state that is handed off to
    /// the driver and not touched by the caller afterward.
    @discardableResult
    func submit(
        _ request: sending SchedulerRequest,
        handler: SchedulerTokenHandler,
        cancelToken: Int,
        maxTokens: Int,
        sampler: RowSampler?,
        processorSource: RowProcessorSource? = nil,
        stateMachine: StopSequenceMatcher?
    ) -> Int? {
        if freeSlots <= 0 {
            waiting.append(
                PendingSubmission(
                    request: request,
                    handler: handler,
                    cancelToken: cancelToken,
                    maxTokens: maxTokens,
                    sampler: sampler,
                    processorSource: processorSource,
                    stateMachine: stateMachine
                ))
            return nil
        }
        return admit(
            request,
            handler: handler,
            cancelToken: cancelToken,
            maxTokens: maxTokens,
            sampler: sampler,
            processorSource: processorSource,
            stateMachine: stateMachine
        )
    }

    /// Immediately insert a request into the engine (capacity already checked).
    private func admit(
        _ request: SchedulerRequest,
        handler: SchedulerTokenHandler,
        cancelToken: Int,
        maxTokens: Int,
        sampler: RowSampler?,
        processorSource: RowProcessorSource?,
        stateMachine: StopSequenceMatcher?
    ) -> Int? {
        // Parity with the single path for a zero token budget: TokenIterator
        // produces no tokens and generateTask reports `.length`. The engine
        // rejects nonpositive limits, and the generic catch below would close
        // the stream with no completion info at all.
        guard maxTokens > 0 else {
            handler.yieldInfo(
                GenerateCompletionInfo(
                    promptTokenCount: request.inputTokens.count,
                    generationTokenCount: 0,
                    promptTime: 0,
                    generationTime: 0,
                    stopReason: .length
                ))
            handler.finish()
            return nil
        }
        let uids: [Int]
        do {
            uids = try engine.insert(
                prompts: [request.inputTokens],
                maxTokens: [maxTokens],
                samplers: [sampler],
                processorSources: [processorSource],
                stateMachines: stateMachine.map { [$0] }
            )
        } catch {
            // Malformed request (e.g. empty prompt): close the stream cleanly.
            handler.finish()
            return nil
        }
        guard let uid = uids.first else {
            handler.finish()
            return nil
        }
        if let ticket = request.wiredMemoryTicket {
            activeWiredMemoryTicket = ticket
        }
        records[uid] = RequestRecord(
            handler: handler,
            cancelToken: cancelToken,
            promptTokenCount: request.inputTokens.count,
            inputTokens: request.inputTokens,
            modelName: request.modelName,
            promptCacheSalt: request.promptCacheSalt,
            promptCache: request.promptCache,
            submitTime: Date.timeIntervalSinceReferenceDate,
            writeBackToPromptCache: request.promptCache != nil,
            tokensIncludePrompt: true
        )
        return uid
    }

    // MARK: - Adopt (single → batch upgrade bridge)

    /// Routing/finalization info for a migrated (adopted) request, supplied by
    /// the scheduler's upgrade path. Mirrors ``RequestRecord`` but is the
    /// public hand-off shape.
    struct AdoptedRecord {
        let handler: SchedulerTokenHandler
        /// Scheduler-owned stable id, so an adopted (upgraded) row can be
        /// cancelled by token if its consumer drops the stream.
        let cancelToken: Int
        let promptTokenCount: Int
        let inputTokens: [Int]
        let modelName: String
        let promptCacheSalt: UInt64
        let promptCache: (any PromptCaching)?
        let submitTime: TimeInterval
        let writeBackToPromptCache: Bool
    }

    /// Adopt a migrated single request: build a ``DecodeBatch`` from its
    /// already-converted per-layer caches and splice it in as the running set,
    /// registering its routing record so its tokens fan out to the original
    /// request's handler. This happens entirely on the driver — the
    /// non-`Sendable` ``DecodeBatch`` never leaves the actor (it references the
    /// engine's model), so nothing is round-tripped across an isolation
    /// boundary.
    ///
    /// `caches` is transferred with `sending` (region isolation): the migrated
    /// single request's per-layer batched caches, handed off once. The
    /// constructor runs one priming decode step on the engine's executor.
    ///
    /// The adopted row's engine UID is allocated by the engine itself (from its
    /// own `uidCounter`, the single source of truth for row identity) and
    /// returned by `makeAdoptedBatch`, so a later `engine.insert` can never
    /// reuse it and overwrite this record. The scheduler-owned id used to cancel
    /// the row by token still travels on ``AdoptedRecord/cancelToken``.
    func adoptMigrated(
        seedToken: Int,
        caches: sending [any BatchedCache],
        sampler: RowSampler?,
        processorSource: RowProcessorSource? = nil,
        stateMachine: StopSequenceMatcher?,
        maxTokens: Int,
        numTokens: Int,
        tokens: [Int],
        record: AdoptedRecord
    ) {
        let (batch, uids) = engine.makeAdoptedBatch(
            seedTokens: MLXArray([UInt32(seedToken)]),
            caches: caches,
            samplers: sampler.map { [$0] },
            processorSources: [processorSource],
            stateMachines: stateMachine.map { [$0] },
            maxTokens: [maxTokens],
            numTokens: [numTokens],
            tokens: [tokens]
        )
        guard let uid = uids.first else {
            // Unreachable today (makeAdoptedBatch allocates one uid per token
            // row unconditionally), but if it ever fires the stream must be
            // closed rather than left hanging for a caller awaiting tokens.
            record.handler.finish()
            return
        }
        records[uid] = RequestRecord(
            handler: record.handler,
            cancelToken: record.cancelToken,
            promptTokenCount: record.promptTokenCount,
            inputTokens: record.inputTokens,
            modelName: record.modelName,
            promptCacheSalt: record.promptCacheSalt,
            promptCache: record.promptCache,
            submitTime: record.submitTime,
            writeBackToPromptCache: record.writeBackToPromptCache,
            // Adopted rows carry the FULL history (prompt + generated): the
            // scheduler passes prompt-inclusive `tokens` so the row's penalty
            // processor is seeded with the same context window the single
            // iterator had (which seeded from the prompt before decoding).
            // `makeAdoptedBatch` drops the trailing seed and priming
            // re-appends it, so `allTokens` equals prompt + generated.
            tokensIncludePrompt: true,
            // Seed with the tokens already emitted on the single path so a
            // later stop-string finish reports the full generated count.
            deliveredTokenCount: numTokens
        )
        engine.adoptActiveBatch(batch)
    }

    // MARK: - Cancel

    /// Cancel a request by engine UID. Removes it from the engine (queued or
    /// active), finishes its stream, and ends its bookkeeping. Returns whether
    /// anything was removed.
    @discardableResult
    func cancel(uid: Int) -> Bool {
        let removed = engine.cancel(uid: uid)
        if let record = records.removeValue(forKey: uid) {
            record.handler.finish()
        }
        // A finished/cancelled row freed a slot; pull in any waiters.
        admitWaitingIfPossible()
        return removed
    }

    /// Cancel a request by its scheduler-owned token, wherever it currently
    /// lives: waiting for batch capacity (drop from the queue and close its
    /// stream) or admitted to the engine (route through ``cancel(uid:)``). Used
    /// when a consumer drops the stream before, or after, the request reaches
    /// the engine. Returns whether anything was removed.
    @discardableResult
    func cancel(token: Int) -> Bool {
        if let index = waiting.firstIndex(where: { $0.cancelToken == token }) {
            let pending = waiting.remove(at: index)
            pending.handler.finish()
            // A waiter never occupied an engine slot, so there is nothing to
            // re-admit here.
            return true
        }
        if let uid = records.first(where: { $0.value.cancelToken == token })?.key {
            return cancel(uid: uid)
        }
        return false
    }

    // MARK: - Decode loop

    /// Run the engine to completion, fanning each step's tokens out to the
    /// registered handlers and writing finished rows back to the prompt cache.
    ///
    /// Each step optionally runs under a wired-memory limit (the
    /// the wired-memory `drain` fold-in). `onResult` is invoked once per
    /// engine step with that step's responses (used by the scheduler for
    /// state-machine transitions, e.g. detecting the batch has emptied).
    ///
    /// Between steps the loop `await Task.yield()`s. Because the engine is
    /// actor-isolated, queued `submit`/`cancel` actor-messages run at that
    /// yield point — **between** steps, never during one — so engine mutation
    /// stays serial with no locks. (`withWiredLimit` takes a synchronous
    /// closure and cannot itself contain the yield, so the wired limit is
    /// applied per step rather than around the whole loop; the per-step cost is
    /// negligible next to a decode step.)
    ///
    /// Idempotent: if a loop is already running this returns immediately, since
    /// the running loop already drains all queued/admitted work.
    /// Returns `true` if this call owned the loop (ran it to completion),
    /// `false` if a loop was already running — callers use this to run
    /// loop-completion logic exactly once per real drain.
    @discardableResult
    func drain(
        wiredMemoryTicket: WiredMemoryTicket? = nil,
        onResult: (@Sendable ([BatchStepResult]) -> Void)? = nil
    ) async -> Bool {
        if draining { return false }
        draining = true
        defer { draining = false }

        while true {
            admitWaitingIfPossible()
            if !engine.hasWork { break }

            // Re-read per step (not captured once before the loop): a
            // ticketed request can join an already-running drain via
            // `submit`, and its memory policy must govern the steps that
            // decode it. An explicit override still wins for the whole loop.
            if let ticket = wiredMemoryTicket ?? activeWiredMemoryTicket {
                await stepOnceWired(ticket, onResult: onResult)
            } else {
                stepOnce(onResult: onResult)
            }

            // Cooperative yield: lets queued actor messages (submit/cancel)
            // run between steps. Engine mutation stays serial.
            await Task.yield()
        }
        // Drop the captured ticket once the batch is fully drained so a
        // long-idle driver doesn't let a stale request's memory policy
        // govern an unrelated future drain.
        activeWiredMemoryTicket = nil
        return true
    }

    /// Runs one engine step under the ticket's wired-memory limit.
    ///
    /// `nonisolated` on purpose: `withWiredLimit`'s body is a `sending`
    /// parameter, and a closure formed inside an actor-isolated method
    /// inherits the actor's isolation (making it unsendable) even when it only
    /// `await`s. Formed here, the closure captures only `Sendable` values
    /// (the actor reference and the handler) and re-enters the actor for the
    /// step itself; the wired limit is a process-global GPU setting, so it
    /// spans the hop.
    nonisolated private func stepOnceWired(
        _ ticket: WiredMemoryTicket,
        onResult: (@Sendable ([BatchStepResult]) -> Void)?
    ) async {
        await WiredMemoryTicket.withWiredLimit(ticket) {
            await self.stepOnce(onResult: onResult)
        }
    }

    /// One engine step with token fan-out and prompt-cache write-back.
    private func stepOnce(onResult: (@Sendable ([BatchStepResult]) -> Void)?) {
        // Capture final caches when ANY destination cache exists -- the
        // driver-level one or a per-record (per-request) instance.
        let captureCaches =
            promptCache != nil || records.values.contains { $0.promptCache != nil }
        let (responses, finishedCaches) = engine.next(capturingFinalCaches: captureCaches)

        // Write finished rows back BEFORE delivery, while their records (which
        // carry the model name / salt) are still present.
        for finished in finishedCaches {
            writeBack(finished)
        }
        for response in responses {
            deliver(response)
        }
        onResult?(responses)
    }

    /// Fan one row's result out to its handler. On a finishing response the
    /// stream is flushed (tool calls), an `.info` event is emitted, and the
    /// handler is dropped.
    private func deliver(_ response: BatchStepResult) {
        if records[response.uid]?.firstTokenAt == nil {
            records[response.uid]?.firstTokenAt = Date.timeIntervalSinceReferenceDate
        }
        records[response.uid]?.deliveredTokenCount += 1
        guard let record = records[response.uid] else { return }
        let handler = record.handler

        if let reason = response.finishReason {
            // Final token. For raw-token consumers a stop token may need to be
            // surfaced before the stream closes; text consumers ignore it.
            if reason == .stop {
                _ = handler.processStopToken(response.token)
            } else {
                _ = handler.processToken(response.token)
            }
            handler.processEndOfSequence()

            let now = Date.timeIntervalSinceReferenceDate
            let info = GenerateCompletionInfo(
                promptTokenCount: record.promptTokenCount,
                generationTokenCount: generationCount(
                    record,
                    allTokens: response.allTokens,
                    suppressedStopToken: suppressesStopToken(record, reason: reason)
                ),
                promptTime: max(0, (record.firstTokenAt ?? now) - record.submitTime),
                generationTime: max(0, now - (record.firstTokenAt ?? now)),
                stopReason: reason
            )
            handler.yieldInfo(info)
            handler.finish()
            records.removeValue(forKey: response.uid)
        } else {
            switch handler.processToken(response.token) {
            case .more:
                break
            case .stop:
                // A configured stop string completed in this row's decoded
                // text. Finish the stream like a natural stop and drop the
                // row from the engine.
                finishOnSemanticStop(response.uid)
            case .cancelled:
                // Consumer cancelled mid-stream. Route through `cancel(uid:)` so
                // the engine drop, stream finish, record removal, and waiting-
                // queue admission all happen in one place.
                cancel(uid: response.uid)
            }
        }
    }

    /// Finish a row whose handler reported a semantic stop (`.stop` from
    /// `processToken`: a configured stop string completed in the decoded
    /// text). Mirrors the natural-finish branch of `deliver` — EOS flush,
    /// `.info` with `.stop`, stream close — then removes the row from the
    /// engine and admits any waiters into the freed slot.
    private func finishOnSemanticStop(_ uid: Int) {
        if let record = records.removeValue(forKey: uid) {
            let handler = record.handler
            handler.processEndOfSequence()
            let now = Date.timeIntervalSinceReferenceDate
            let info = GenerateCompletionInfo(
                promptTokenCount: record.promptTokenCount,
                // No engine `allTokens` on this path (the row is still live
                // from the engine's perspective); the per-record delivery
                // counter is exactly the generated-token count.
                generationTokenCount: record.deliveredTokenCount,
                promptTime: max(0, (record.firstTokenAt ?? now) - record.submitTime),
                generationTime: max(0, now - (record.firstTokenAt ?? now)),
                stopReason: .stop
            )
            handler.yieldInfo(info)
            handler.finish()
        }
        _ = engine.cancel(uid: uid)
        admitWaitingIfPossible()
    }

    /// Write a finished row's final KV state back to the prompt cache. Runs on
    /// the driver actor (it owns the cache reference); ``FinishedRowCache`` is
    /// non-`Sendable` and never crosses a boundary.
    private func writeBack(_ finished: FinishedRowCache) {
        guard let record = records[finished.uid],
            record.writeBackToPromptCache,
            // The request's own cache instance wins; the driver-level cache
            // (from attach) is the fallback for records without one.
            let cache = record.promptCache ?? promptCache
        else { return }

        // Force the extracted caches to the device before storing them.
        let arrays = finished.finalCache.flatMap { $0.state }
        if !arrays.isEmpty {
            eval(arrays)
        }

        // The prompt-cache key must be the FULL token history the stored KV
        // state actually covers (prompt + generated). Freshly-prefilled rows
        // already carry the prompt in `allTokens`. Adopted (upgraded) rows do
        // not: their history was seeded generated-only, while the migrated
        // caches still hold the original prompt's KV. Keying such a row by the
        // generated-only suffix would file prompt-bearing KV under an unrelated
        // key, so a later lookup for that suffix could reuse positions/content
        // from a different prompt. Reconstruct the full key from the prompt.
        let keyTokens: [Int]
        if record.tokensIncludePrompt {
            keyTokens = finished.allTokens
        } else {
            keyTokens = record.inputTokens + finished.allTokens
        }
        cache.insertCache(
            model: record.modelName,
            tokens: keyTokens,
            promptCache: finished.finalCache,
            checkpoint: false,
            salt: record.promptCacheSalt
        )
    }

    // MARK: - Waiting-queue admission

    /// Admit as many waiting requests as there are free slots, in FIFO order.
    private func admitWaitingIfPossible() {
        guard !waiting.isEmpty else { return }
        while !waiting.isEmpty, freeSlots > 0 {
            let pending = waiting.removeFirst()
            _ = admit(
                pending.request,
                handler: pending.handler,
                cancelToken: pending.cancelToken,
                maxTokens: pending.maxTokens,
                sampler: pending.sampler,
                processorSource: pending.processorSource,
                stateMachine: pending.stateMachine
            )
        }
    }

    // MARK: - Shutdown

    /// Cancel everything and release the engine.
    func close() {
        for (_, record) in records {
            record.handler.finish()
        }
        records.removeAll()
        for pending in waiting {
            pending.handler.finish()
        }
        waiting.removeAll()
        engine.close()
    }
}
