// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The single serial executor that owns the non-thread-safe
/// ``BatchInferenceEngine`` and (optionally) the cross-request prompt cache.
///
/// ## Why this exists
///
/// ``BatchInferenceEngine``, ``DecodeBatch``, and the per-layer
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
        let promptTokenCount: Int
        let inputTokens: [Int]
        let modelName: String
        let promptCacheSalt: UInt64
        let submitTime: TimeInterval
        /// Whether to write this request's final KV state back to the prompt
        /// cache when it finishes.
        let writeBackToPromptCache: Bool
    }

    /// The sole owner of the engine. Never escapes the actor.
    private let engine: BatchInferenceEngine

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

    /// A request waiting for batch capacity. Carries everything `submit` needs
    /// to admit it later.
    private struct PendingSubmission {
        let request: SchedulerRequest
        let handler: SchedulerTokenHandler
        let maxTokens: Int
        let sampler: RowSampler?
        let stateMachine: StopSequenceMatcher?
    }

    init(engine: BatchInferenceEngine, promptCache: (any PromptCaching)? = nil) {
        self.engine = engine
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
        maxTokens: Int,
        sampler: RowSampler?,
        stateMachine: StopSequenceMatcher?
    ) -> Int? {
        if freeSlots <= 0 {
            waiting.append(
                PendingSubmission(
                    request: request,
                    handler: handler,
                    maxTokens: maxTokens,
                    sampler: sampler,
                    stateMachine: stateMachine
                ))
            return nil
        }
        return admit(
            request,
            handler: handler,
            maxTokens: maxTokens,
            sampler: sampler,
            stateMachine: stateMachine
        )
    }

    /// Immediately insert a request into the engine (capacity already checked).
    private func admit(
        _ request: SchedulerRequest,
        handler: SchedulerTokenHandler,
        maxTokens: Int,
        sampler: RowSampler?,
        stateMachine: StopSequenceMatcher?
    ) -> Int? {
        let uids: [Int]
        do {
            uids = try engine.insert(
                prompts: [request.inputTokens],
                maxTokens: [maxTokens],
                samplers: [sampler],
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
        records[uid] = RequestRecord(
            handler: handler,
            promptTokenCount: request.inputTokens.count,
            inputTokens: request.inputTokens,
            modelName: request.modelName,
            promptCacheSalt: request.promptCacheSalt,
            submitTime: Date.timeIntervalSinceReferenceDate,
            writeBackToPromptCache: request.promptCache != nil
        )
        return uid
    }

    // MARK: - Adopt (single → batch upgrade bridge)

    /// Splice an already-decoding ``DecodeBatch`` (built from a migrated single
    /// request's live caches) into the engine as the running set, and register
    /// its per-uid routing records so its tokens fan out to the original
    /// request's handler.
    ///
    /// `batch` is transferred with `sending`: it carries non-`Sendable` decode
    /// state handed off from the upgrade path, which provably stops touching it
    /// after the deposit.
    func adopt(
        _ batch: sending DecodeBatch,
        records newRecords: [Int: AdoptedRecord]
    ) {
        for (uid, rec) in newRecords {
            records[uid] = RequestRecord(
                handler: rec.handler,
                promptTokenCount: rec.promptTokenCount,
                inputTokens: rec.inputTokens,
                modelName: rec.modelName,
                promptCacheSalt: rec.promptCacheSalt,
                submitTime: rec.submitTime,
                writeBackToPromptCache: rec.writeBackToPromptCache
            )
        }
        engine.adoptActiveBatch(batch)
    }

    /// Routing/finalization info for a migrated (adopted) request, supplied by
    /// the scheduler's upgrade path. Mirrors ``RequestRecord`` but is the
    /// public hand-off shape.
    struct AdoptedRecord {
        let handler: SchedulerTokenHandler
        let promptTokenCount: Int
        let inputTokens: [Int]
        let modelName: String
        let promptCacheSalt: UInt64
        let submitTime: TimeInterval
        let writeBackToPromptCache: Bool
    }

    /// Build a ``DecodeBatch`` from migrated single-request decode state. Runs
    /// on the driver so the priming decode step the constructor performs touches
    /// the engine's model from the engine's executor.
    ///
    /// Caches are transferred with `sending` (region isolation): they are the
    /// migrated single request's per-layer batched caches, handed off once.
    func makeAdoptedBatch(
        uid: Int,
        seedToken: Int,
        caches: sending [any BatchedCache],
        sampler: RowSampler?,
        stateMachine: StopSequenceMatcher?,
        maxTokens: Int,
        numTokens: Int,
        tokens: [Int]
    ) -> sending DecodeBatch {
        engine.makeAdoptedBatch(
            uids: [uid],
            seedTokens: MLXArray([UInt32(seedToken)]),
            caches: caches,
            samplers: sampler.map { [$0] },
            stateMachines: stateMachine.map { [$0] },
            maxTokens: [maxTokens],
            numTokens: [numTokens],
            tokens: [tokens]
        )
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

    // MARK: - Decode loop

    /// Run the engine to completion, fanning each step's tokens out to the
    /// registered handlers and writing finished rows back to the prompt cache.
    ///
    /// Each step optionally runs under a wired-memory limit (the
    /// ``layr/wiredmemory`` `drain` fold-in). `onResult` is invoked once per
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
    func drain(
        wiredMemoryTicket: WiredMemoryTicket? = nil,
        onResult: (@Sendable ([BatchStepResult]) -> Void)? = nil
    ) async {
        if draining { return }
        draining = true
        defer { draining = false }

        while true {
            admitWaitingIfPossible()
            if !engine.hasWork { break }

            if let ticket = wiredMemoryTicket {
                await WiredMemoryTicket.withWiredLimit(ticket) {
                    self.stepOnce(onResult: onResult)
                }
            } else {
                stepOnce(onResult: onResult)
            }

            // Cooperative yield: lets queued actor messages (submit/cancel)
            // run between steps. Engine mutation stays serial.
            await Task.yield()
        }
    }

    /// One engine step with token fan-out and prompt-cache write-back.
    private func stepOnce(onResult: (@Sendable ([BatchStepResult]) -> Void)?) {
        let captureCaches = promptCache != nil
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
            let generatedCount = response.allTokens?.count ?? 0
            let info = GenerateCompletionInfo(
                promptTokenCount: record.promptTokenCount,
                generationTokenCount: generatedCount,
                promptTime: 0,
                generationTime: max(0, now - record.submitTime),
                stopReason: reason
            )
            handler.yieldInfo(info)
            handler.finish()
            records.removeValue(forKey: response.uid)
        } else {
            if handler.processToken(response.token) == false {
                // Consumer cancelled mid-stream: drop the row from the engine.
                engine.cancel(uid: response.uid)
                handler.finish()
                records.removeValue(forKey: response.uid)
            }
        }
    }

    /// Write a finished row's final KV state back to the prompt cache. Runs on
    /// the driver actor (it owns the cache reference); ``FinishedRowCache`` is
    /// non-`Sendable` and never crosses a boundary.
    private func writeBack(_ finished: FinishedRowCache) {
        guard let cache = promptCache,
            let record = records[finished.uid],
            record.writeBackToPromptCache
        else { return }

        // Force the extracted caches to the device before storing them.
        let arrays = finished.finalCache.flatMap { $0.state }
        if !arrays.isEmpty {
            eval(arrays)
        }
        cache.insertCache(
            model: record.modelName,
            tokens: finished.allTokens,
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
                maxTokens: pending.maxTokens,
                sampler: pending.sampler,
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
