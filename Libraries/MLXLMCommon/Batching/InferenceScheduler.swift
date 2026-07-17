// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import os

// MARK: - InferenceScheduler

/// Thin actor that routes inference requests between a zero-overhead
/// single-stream path and the continuous-batching ``EngineDriver``, with an
/// automatic single→batch upgrade when a second concurrent request arrives.
///
/// ## Design
///
/// The scheduler owns **policy and state only**; it owns no engine state
/// directly. All decode/stop/sampling/cache math lives in the PR2 engine
/// (driven by ``EngineDriver``) and the PR1 caches. State machine:
///
/// ```
/// idle -> single -> (2nd submit) -> upgrading -> batched -> (all rows finish) -> idle
/// ```
///
/// - **`.single`** — the first request runs on a plain ``TokenIterator`` with
///   zero batch overhead (the common LAN/desktop/phone case).
/// - **Auto-upgrade** — when a second request arrives, the scheduler asks the
///   running single task to deposit its live decode state (caches, current
///   token, sampler) via a three-phase ``UpgradeFlag`` + `CheckedContinuation`
///   handshake, migrates those caches into a batch with PR1's `fromSingle`,
///   and hands the already-decoding batch to the ``EngineDriver`` — **without
///   re-prefilling** the first prompt. A third request during the window runs
///   independently and joins the batch afterward.
/// - **`.batched`** — all requests run through the ``EngineDriver``; as rows
///   finish the engine shrinks the batch, and a final empty batch returns the
///   scheduler to `.idle`.
///
/// ## Concurrency (zero `@unchecked Sendable`)
///
/// The cross-actor upgrade snapshot (``LiveIteratorState``) is a plain
/// non-`Sendable` struct transferred **once** via `sending`
/// (`CheckedContinuation.resume(returning:)` is `sending` per SE-0430); the
/// single task provably stops touching it after the deposit. ``UpgradeFlag``
/// holds its shared flag + continuation in an `OSAllocatedUnfairLock<State>`,
/// so it is checked `Sendable`. The engine is never touched here — only through
/// the ``EngineDriver`` actor.
public actor InferenceScheduler {

    /// Configuration controlling how the scheduler constructs the batch engine.
    public struct Configuration: Sendable {
        /// Maximum number of rows the engine decodes concurrently.
        public var completionBatchSize: Int
        /// Maximum number of prompts prefilled together per admission.
        public var prefillBatchSize: Int
        /// Prompt prefill chunk size. Engine-level BY DESIGN (matching
        /// upstream mlx_lm's batch generator): batched prefill chunks the
        /// whole ragged sub-batch uniformly, so the per-request
        /// `GenerateParameters.prefillStepSize` honored by the single path
        /// does not apply on the batched path.
        public var prefillStepSize: Int
        /// Optional runtime admission cap (`nil` = `completionBatchSize`).
        public var maxBatchSize: Int?

        public init(
            completionBatchSize: Int = 32,
            prefillBatchSize: Int = 8,
            prefillStepSize: Int = 2048,
            maxBatchSize: Int? = nil
        ) {
            self.completionBatchSize = completionBatchSize
            self.prefillBatchSize = prefillBatchSize
            self.prefillStepSize = prefillStepSize
            self.maxBatchSize = maxBatchSize
        }
    }

    private let configuration: Configuration

    /// The model/tokenizer/config the scheduler drives. Set once via
    /// ``attach(context:)`` before the first request (``ModelContainer`` does
    /// this when it installs the scheduler).
    private var context: ModelContext?

    /// The driver that solely owns the batch engine, created lazily on the
    /// first request that needs batching.
    private var driver: EngineDriver?

    /// Bumped every time ``refreshContext(owner:context:)`` replaces the
    /// context. A running single request captures the epoch it started under;
    /// the upgrade gate requires equality so old-model KV is never migrated
    /// into an engine built from a newer context (real corruption otherwise:
    /// K/V computed by the old weights decoded by the new model).
    private var contextEpoch = 0

    /// Set by ``refreshContext(owner:context:)`` when the context was replaced
    /// while a batch was still draining: the driver (which caches the model
    /// and EOS set) is torn down at the next idle transition instead of
    /// mid-flight.
    private var driverStale = false

    /// Optional cross-request prompt cache (shared with the driver).
    private var promptCache: (any PromptCaching)?

    /// Identity of the `ModelContainer` this scheduler is attached to.
    private var attachedOwner: ObjectIdentifier?

    /// Bumped whenever new work is handed to the driver; lets
    /// `batchLoopFinished` detect a submission that raced with its
    /// `driver.hasWork` read instead of idling under a live batch.
    private var batchAdmissionEpoch = 0

    /// Current routing state.
    private var state: SchedulerState = .idle

    /// Monotonic request id for diagnostics.
    private var nextRequestID: Int = 0

    /// Requests that arrived while a non-upgradeable single request was running
    /// (its cache topology cannot be migrated single→batch). They are held —
    /// not started — until the running single request completes, then drained
    /// in FIFO order so the first request is never interrupted/truncated. The
    /// elements are non-`Sendable` (`ScheduledRequest`), but they are stored and
    /// later re-routed entirely within this actor's isolation, so no isolation
    /// boundary is crossed.
    private var queuedRequests: [ScheduledRequest] = []

    public init(configuration: Configuration = .init()) {
        self.configuration = configuration
    }

    /// Bind the model context the scheduler drives. Idempotent for the same
    /// owner (the context is captured on its first attach); a new owner
    /// attaching after a ``detach(owner:)`` replaces the retained context.
    /// Called by ``ModelContainer`` when the scheduler is first used.
    ///
    /// The context is transferred inside a `SendableBox` (the repo's existing
    /// non-`Sendable` hand-off box, also used by `generateTask`): ``ModelContext``
    /// carries non-`Sendable` model/processor references. The scheduler shares
    /// the (already-evaluated) model weights with the container, the same way
    /// ``ModelContainer/generate(input:parameters:wiredMemoryTicket:)`` permits
    /// concurrent access to evaluated weights. No NEW `@unchecked Sendable` is
    /// introduced — `SendableBox` is pre-existing infrastructure.
    /// Returns `false` (without consuming the box) if the scheduler is
    /// already bound to a different owner — a scheduler drives exactly one
    /// container's model; silently reusing the first context would generate
    /// wrong outputs for the second container.
    func attach(
        owner: ObjectIdentifier,
        context box: SendableBox<ModelContext>,
        promptCache: (any PromptCaching)?
    ) -> Bool {
        if let attachedOwner, attachedOwner != owner {
            return false
        }
        let newOwner = attachedOwner == nil
        attachedOwner = owner
        if self.context == nil {
            self.context = box.consume()
        } else if newOwner {
            // A NEW owner is attaching after a `detach(owner:)`: the retained
            // context (and cache) belong to the departed container — the old
            // keep-first behavior would silently generate every request with
            // the previous container's model. Replace them and invalidate
            // epoch-gated state so nothing built from the old context (a
            // driver, or an in-flight single's upgrade) can be reused.
            self.context = box.consume()
            self.promptCache = promptCache
            contextEpoch += 1
            if driver != nil {
                driverStale = true
            }
        }
        if self.promptCache == nil {
            self.promptCache = promptCache
        }
        return true
    }

    /// Release the binding for `owner` (called from ``ModelContainer``'s
    /// `deinit`). Without this, the scheduler stays `schedulerAlreadyAttached`
    /// forever after its container dies — and worse, an `ObjectIdentifier`
    /// recycled onto a NEW container could pass the `attachedOwner` equality
    /// check and silently serve requests with the dead container's retained
    /// context. The context itself is kept for any still-running requests,
    /// but the epoch is bumped and the driver marked stale so nothing built
    /// from it survives past the next attach/idle transition.
    func detach(owner: ObjectIdentifier) {
        guard attachedOwner == owner else { return }
        attachedOwner = nil
        contextEpoch += 1
        if driver != nil {
            driverStale = true
        }
    }

    /// The container this scheduler is bound to, if any.
    var currentOwner: ObjectIdentifier? { attachedOwner }

    /// Whether the scheduler already has a bound context.
    var isAttached: Bool { context != nil }

    /// Internal test probe: the configuration name of the bound context
    /// (`nil` when no context is bound).
    var attachedModelName: String? { context?.configuration.name }

    /// Replace the bound context after ``ModelContainer/update(_:)`` swaps a
    /// context field (model/tokenizer/processor). Without this, scheduler-
    /// routed generations would keep the attach-time copy forever while the
    /// non-scheduler path re-reads the container's context per call.
    ///
    /// Owner-gated: a refresh from a container that is not the bound owner is
    /// ignored. The driver caches the model and EOS set, so it is torn down —
    /// immediately when no batch is running, otherwise deferred to
    /// ``batchLoopFinished`` via `driverStale` (in-flight batched requests
    /// finish on the model they started with, matching the single path, where
    /// an in-flight iterator keeps its model reference).
    /// Note: requests that JOIN a still-draining batch after a refresh also
    /// run on the old driver's model (one batch, one model) -- coherent but
    /// stale, same continuity rule as the in-flight rows -- and a single
    /// request started before the refresh can no longer upgrade (epoch gate).
    func refreshContext(owner: ObjectIdentifier, context box: SendableBox<ModelContext>) async {
        guard attachedOwner == owner else { return }
        self.context = box.consume()
        contextEpoch += 1
        if let driver, await driver.hasWork {
            driverStale = true
        } else {
            driver = nil
            driverStale = false
        }
    }

    // MARK: - Stats

    /// Current number of actively decoding requests (best effort).
    public var activeRequestCount: Int {
        get async {
            switch state {
            case .idle: return 0
            case .single, .pendingUpgrade: return 1
            case .upgrading: return 1
            case .batched:
                guard let driver else { return 0 }
                return await driver.activeCount
            }
        }
    }

    // MARK: - Admission cap

    /// Adjust the runtime admission cap. Forwarded to the ``EngineDriver``
    /// (clamp `< 1 → 1`, queue-when-full). Stored for drivers created later.
    public func setMaxBatchSize(_ size: Int?) async {
        if let driver {
            await driver.setMaxBatchSize(size)
        }
        // Remember for a driver created on a later upgrade.
        pendingMaxBatchSize = size.map { max(1, $0) } ?? nil
        hasPendingMaxBatchSize = true
    }

    private var pendingMaxBatchSize: Int?
    private var hasPendingMaxBatchSize = false

    // MARK: - State machine

    enum SchedulerState {
        case idle
        case single(SingleRequest)
        case pendingUpgrade(SingleRequest)
        case upgrading
        case batched
    }
}

// MARK: - Single request state

extension InferenceScheduler {

    /// Bookkeeping for a request running on the single-stream path.
    struct SingleRequest {
        let requestID: Int
        let task: Task<Void, Never>
        let handler: SchedulerTokenHandler
        let upgradeFlag: UpgradeFlag
        let parameters: GenerateParameters
        let inputTokens: [Int]
        let modelName: String
        let promptCacheSalt: UInt64
        let writeBackToPromptCache: Bool
        /// The effective cache instance for this request (request-supplied,
        /// falling back to the attach-time container cache); carried so the
        /// upgrade path can hand it to the driver's adopted record.
        let promptCache: (any PromptCaching)?
        let submitTime: TimeInterval
        let wiredMemoryTicket: WiredMemoryTicket?
        /// Whether this request's input can be reproduced on the batch
        /// engine (no explicit attention mask, single flat token row). A
        /// masked single request must not upgrade: its prefill positions
        /// came from the mask, which the engine's flat rows cannot carry.
        let inputIsBatchable: Bool
        /// The scheduler context epoch this request started under; upgrades
        /// require it to still be current (see `contextEpoch`).
        let contextEpoch: Int
    }
}

// MARK: - LiveIteratorState (the `sending` upgrade snapshot)

extension InferenceScheduler {

    /// Snapshot of a live ``TokenIterator``'s decode state, deposited once by
    /// the single task during an upgrade and consumed by the scheduler to seed
    /// the batch.
    ///
    /// This is a **plain non-`Sendable` struct** (it holds non-`Sendable`
    /// `KVCache`, `LMInput.Text`, `LogitSampler`, `LogitProcessor?`). It is
    /// transferred across the actor boundary exactly once with `sending`
    /// (region-based isolation): the single task deposits it via
    /// `CheckedContinuation.resume(returning:)` (which takes `sending` per
    /// SE-0430) and **never reads the captured state again** afterward, so no
    /// `@unchecked Sendable` is needed.
    struct LiveIteratorState {
        /// Per-layer KV caches with the latest decode state.
        let cache: [KVCache]
        /// The current decode token to seed the next step.
        let currentToken: Int
        /// Tokens generated so far on the single path.
        let tokenCount: Int
        /// Maximum tokens allowed for this request.
        let maxTokens: Int?
        /// Generation parameters (for reconstructing a row sampler).
        let parameters: GenerateParameters
        /// All token IDs produced on the single path (for write-back keys).
        let generatedTokenIds: [Int]
        /// When the single path produced its first token, so an adopted row's
        /// completion info keeps the true prompt/decode timing split.
        let firstTokenAt: TimeInterval?
    }
}

// MARK: - UpgradeFlag (checked-Sendable handshake primitive)

extension InferenceScheduler {

    /// Three-phase handshake primitive shared between the scheduler and a
    /// running single task.
    ///
    /// 1. The scheduler calls ``requestUpgrade(continuation:)``, depositing a
    ///    `CheckedContinuation` and setting `upgradeRequested`.
    /// 2. The single task, after each token, observes `upgradeRequested`,
    ///    deposits its ``LiveIteratorState`` via ``depositLiveState(_:)`` (which
    ///    resumes the continuation with a `sending` value), and returns without
    ///    touching that state again.
    /// 3. If the task instead finishes first, ``markTaskFinished()`` resumes any
    ///    pending continuation with `nil` so the scheduler falls back to
    ///    starting the second request fresh.
    ///
    /// All mutable state lives in an `OSAllocatedUnfairLock<State>`, so the type
    /// is **checked** `Sendable` — no `@unchecked`. The `uncheckedState:`
    /// initializer only relaxes the *State*'s Sendability (it transports a
    /// `CheckedContinuation` over a non-`Sendable` payload); the conformance
    /// itself is checked.
    final class UpgradeFlag: Sendable {
        typealias Cont = CheckedContinuation<SendableBox<LiveIteratorState>?, Never>

        private struct State {
            var upgradeRequested = false
            var taskFinished = false
            var continuation: Cont?
        }

        private let state = OSAllocatedUnfairLock<State>(uncheckedState: State())

        /// Whether the scheduler has asked this task to upgrade.
        var upgradeRequested: Bool {
            state.withLockUnchecked { $0.upgradeRequested }
        }

        /// Scheduler side: deposit the continuation and request the upgrade. If
        /// the task already finished, resume immediately with `nil`.
        func requestUpgrade(continuation: Cont) {
            let resumeNil = state.withLockUnchecked { s -> Bool in
                if s.taskFinished {
                    return true
                }
                s.continuation = continuation
                s.upgradeRequested = true
                return false
            }
            if resumeNil {
                continuation.resume(returning: nil)
            }
        }

        /// Single-task side: deposit live state and resume the scheduler.
        ///
        /// The state rides a `SendableBox` (exactly-once transfer); the caller
        /// stops touching it after this returns.
        func depositLiveState(_ liveState: SendableBox<LiveIteratorState>) {
            let cont = state.withLockUnchecked { s -> Cont? in
                let c = s.continuation
                s.continuation = nil
                return c
            }
            cont?.resume(returning: liveState)
        }

        /// Single-task side: mark the task finished. Resumes any pending
        /// continuation with `nil` so the scheduler does not hang.
        func markTaskFinished() {
            let cont = state.withLockUnchecked { s -> Cont? in
                s.taskFinished = true
                let c = s.continuation
                s.continuation = nil
                return c
            }
            cont?.resume(returning: nil)
        }
    }
}

// MARK: - Cache migration (single → batch)

extension InferenceScheduler {

    /// Convert a single request's per-layer `[KVCache]` into the batched
    /// `[any BatchedCache]` the engine decodes over, using PR1's `fromSingle`
    /// breadth. Returns `nil` if any layer's cache type cannot be migrated
    /// (e.g. quantized/chunked), in which case the caller starts the second
    /// request fresh rather than risking a bad migration.
    static func migrateCaches(_ caches: [KVCache]) -> [any BatchedCache]? {
        var batched: [any BatchedCache] = []
        batched.reserveCapacity(caches.count)
        for cache in caches {
            // Exact-type match: ChunkedKVCache subclasses KVCacheSimple, so a
            // plain `as?` would wrongly accept it.
            if Swift.type(of: cache) == KVCacheSimple.self, let simple = cache as? KVCacheSimple {
                batched.append(BatchKVCache.fromSingle(simple))
            } else if let rotating = cache as? RotatingKVCache {
                batched.append(BatchRotatingKVCache.fromSingle(rotating))
            } else {
                // SSM/composite/quantized/chunked single→batch migration is not
                // yet supported by fromSingle; fall back to a fresh second
                // request. (PR1 may later extend fromSingle to SSM/composite.)
                return nil
            }
        }
        return batched
    }

    /// Whether a single request running with `parameters` could be migrated
    /// into a batch, decided from the model's cache **topology** alone — before
    /// any upgrade handshake. This lets the scheduler avoid initiating an
    /// upgrade it cannot complete (which would otherwise force the first,
    /// already-running request to stop mid-generation).
    ///
    /// `model.newCache(parameters:)` is a pure, side-effect-free factory that
    /// returns the exact per-layer cache types the running ``TokenIterator``
    /// built for this request (the iterator uses the same call with the same
    /// `parameters`), so probing fresh caches faithfully predicts whether the
    /// live caches will migrate — without touching the running task's state.
    ///
    /// KV quantization (`kvBits != nil`) is treated as non-upgradeable even
    /// though the *fresh* caches start as `KVCacheSimple`: the iterator's decode
    /// loop dynamically rewrites them to `QuantizedKVCache` after
    /// `quantizedKVStart` tokens (`maybeQuantizeKVCache`), which `migrateCaches`
    /// rejects. Gating on `kvBits` here keeps the first request on its single
    /// iterator instead of discovering the rejection only after the handshake.
    private func canUpgrade(parameters: GenerateParameters) -> Bool {
        guard let context else { return false }
        guard Self.parametersAreBatchable(parameters) else { return false }
        return Self.migrateCaches(context.model.newCache(parameters: parameters)) != nil
    }

    /// Whether a request with these parameters may run on the shared batch
    /// engine at all. The engine's per-layer cache factories are built once
    /// from `newCache(parameters: nil)`, so per-request parameters that change
    /// the cache *topology* cannot be honored there:
    ///
    /// - `kvBits` / `kvScheme` — the single path rewrites caches to
    ///   `QuantizedKVCache` mid-decode (`maybeQuantizeKVCache`; `kvScheme`
    ///   triggers it even with `kvBits == nil`); the batched engine has no
    ///   equivalent, so the request would silently lose its quantization —
    ///   and a running `kvScheme` single would fail cache migration at the
    ///   upgrade handshake and be truncated early.
    /// - `maxKVSize` — `newCache` returns `RotatingKVCache(keep: 4)` for every
    ///   layer; migrating or admitting such rows next to the engine's
    ///   `BatchKVCache` rows would trip `extendBatched`'s topology
    ///   precondition (and the factory rejects keep > 0 regardless). The
    ///   request would otherwise silently lose its KV-memory cap.
    ///
    /// Non-batchable requests run on their own single iterator instead
    /// (correctness over batching).
    static func parametersAreBatchable(_ parameters: GenerateParameters) -> Bool {
        parameters.kvBits == nil && parameters.kvScheme == nil
            && parameters.maxKVSize == nil
    }

    /// Whether a request's INPUT can run on the batch engine. The engine
    /// consumes flat token rows; an explicit attention mask (a left-padded or
    /// pre-batched `LMInput`) or a multi-row tensor would be silently dropped
    /// by that flattening, prefilling different tokens/positions than the
    /// single path (which honors the mask via `LMInput.Text.sequenceLengths`).
    /// Such requests stay on the single path.
    static func inputIsBatchable(_ input: LMInput) -> Bool {
        guard input.text.mask == nil else { return false }
        let tokens = input.text.tokens
        return tokens.ndim == 1 || (tokens.ndim == 2 && tokens.dim(0) == 1)
    }
}

// MARK: - Public submit API

extension InferenceScheduler {

    /// Submit a request, returning a per-request `AsyncStream<Generation>`.
    ///
    /// Routing:
    /// - `.idle` → start the request on the single-stream path.
    /// - `.single` → request the running single task to upgrade; on a
    ///   successful handshake migrate it + this request into the batch,
    ///   otherwise (task already finishing) start this request fresh on the
    ///   single path or, if the driver exists, on the batch.
    /// - `.upgrading`/`.batched` → admit directly to the ``EngineDriver``.
    ///
    /// `request` is transferred with `sending` (it carries a non-`Sendable`
    /// `LMInput`).
    public func submit(_ request: sending GenerationRequest) async throws
        -> AsyncStream<Generation>
    {
        try await submitOne(request)
    }

    /// Submit several requests at once (the manual ``ModelContainer/generateBatched(_:)``
    /// path), returning one stream per request in order. The array is
    /// transferred with `sending` and owned by the actor; each request is
    /// routed internally with no further boundary crossing.
    public func submitBatch(_ requests: sending [GenerationRequest]) async throws
        -> [AsyncStream<Generation>]
    {
        // Box each element before routing: the region checker cannot transfer
        // one element of an actor-held array while the rest stays reachable,
        // so each request rides a `SendableBox` and is consumed (yielding a
        // disconnected value) right at the `sending` boundary.
        var boxes: [SendableBox<GenerationRequest>] = []
        boxes.reserveCapacity(requests.count)
        var remaining = Array(requests.reversed())
        while !remaining.isEmpty {
            boxes.append(SendableBox(remaining.removeLast()))
        }
        var streams: [AsyncStream<Generation>] = []
        streams.reserveCapacity(boxes.count)
        for box in boxes {
            streams.append(try await submitOne(box.consume()))
        }
        return streams
    }

    /// Build the per-request stream + handler and route it.
    private func submitOne(_ request: sending GenerationRequest) async throws
        -> AsyncStream<Generation>
    {
        guard let context else {
            throw BatchedGenerationError.schedulerUnavailable
        }

        var request = request
        // A container-level cache applies to every scheduler request unless
        // the request carries its own instance -- generateBatched callers
        // (whose GenerationRequests default promptCache to nil) previously
        // lost the attached cache silently.
        if request.promptCache == nil {
            request.promptCache = self.promptCache
        }
        // Topology-aware cache keying: fold the per-layer cache topology
        // (concrete type + window size) into the salt before any fetch or
        // write-back uses it, so entries created under different cache
        // configurations (e.g. two maxKVSize values -> different rotating
        // windows) can never collide under the same model/salt. Only done for
        // cache-opted-in requests; the salt is unused otherwise.
        if request.promptCache != nil {
            request.promptCacheSalt = LRUPromptCache.topologySalt(
                base: request.promptCacheSalt,
                caches: context.model.newCache(parameters: request.parameters)
            )
        }

        let (stream, continuation) = AsyncStream.makeStream(of: Generation.self)
        let handler = SchedulerTokenHandler.text(
            continuation: continuation,
            tokenizer: context.tokenizer,
            // Parity with generateTask's TextToolTokenLoopHandler: configured
            // stop strings end generation semantically on both the single and
            // batched paths (the handler returns `.stop` to its driver).
            stopStrings: context.configuration.effectiveStopStrings,
            toolCallFormat: context.configuration.toolCallFormat ?? .json
        )

        let requestID = nextRequestID
        nextRequestID += 1

        // Wire stream cancellation once, generically, before routing: a consumer
        // that drops the stream — while held in the scheduler's queue, while
        // waiting for batch capacity, during prefill, or mid-decode — cancels the
        // request wherever it currently lives (single task, scheduler queue, or
        // driver). Mirrors the historical `generateTask` `onTermination`
        // cancellation, which the scheduler's own stream previously lacked except
        // on the single path.
        handler.onCancellation { [weak self] in
            Task { await self?.cancel(requestID: requestID) }
        }

        let scheduled = ScheduledRequest(
            requestID: requestID,
            request: request,
            handler: handler,
            inputTokens: request.input.text.tokens.asArray(Int.self),
            modelName: context.configuration.name
        )

        try await route(scheduled)
        return stream
    }

    /// Cancel a request by its scheduler-owned id, wherever it currently lives.
    /// Invoked when a consumer drops its stream (the `onCancellation` wiring in
    /// ``submitOne(_:)``):
    ///
    /// - On the single path (`.single`/`.pendingUpgrade`) for this request:
    ///   cancel its task; the task closes the stream and reports back through
    ///   ``singleTaskFinished(requestID:generatedTokenIds:)``, which returns the
    ///   scheduler to `.idle` and drains any queued requests.
    /// - Held in the scheduler's queue (waiting behind a non-upgradeable single
    ///   request): drop it and close its stream.
    /// - Otherwise: forward to the driver, which removes it whether it is still
    ///   waiting for batch capacity or already decoding in the engine.
    private func cancel(requestID: Int) async {
        let runningSingle: SingleRequest?
        switch state {
        case .single(let s), .pendingUpgrade(let s):
            runningSingle = s.requestID == requestID ? s : nil
        default:
            runningSingle = nil
        }
        if let runningSingle {
            runningSingle.task.cancel()
            return
        }

        if let index = queuedRequests.firstIndex(where: { $0.requestID == requestID }) {
            let dropped = queuedRequests.remove(at: index)
            dropped.handler.finish()
            return
        }

        await driver?.cancel(token: requestID)
    }

    /// Internal carrier bundling a request with its derived routing data.
    /// Not `Sendable`: holds the non-`Sendable` `GenerationRequest`. The model
    /// context is read from the actor's stored ``context`` rather than carried
    /// here, so this value can be passed to `sending` routing methods.
    struct ScheduledRequest {
        /// Scheduler-owned stable id, allocated when the stream is created. Used
        /// as the single-task id, the queue identity, and the driver's cancel
        /// token, so a single ``cancel(requestID:)`` can stop the request
        /// wherever it currently lives.
        let requestID: Int
        var request: GenerationRequest
        let handler: SchedulerTokenHandler
        let inputTokens: [Int]
        let modelName: String
    }

    private func route(_ scheduled: sending ScheduledRequest) async throws {
        switch state {
        case .idle:
            // The iterator is built synchronously inside `startSingle`; a failing
            // prefill throws out here → `submitOne`/`submit` →
            // `ModelContainer.generate`, mirroring the historical single path's
            // synchronous init-error semantics. (Drain re-routes — where the
            // stream is already live — close the stream instead; see
            // `drainQueuedRequests`.)
            try startSingle(scheduled)

        case .single(let single):
            if !Self.parametersAreBatchable(scheduled.request.parameters)
                || !Self.inputIsBatchable(scheduled.request.input)
            {
                // The NEW request's parameters (kvBits / maxKVSize) or masked
                // input cannot run on the shared engine; hold it behind the
                // running single request and run it on its own iterator once
                // the scheduler is idle again (drained by singleTaskFinished /
                // batchLoopFinished).
                queuedRequests.append(scheduled)
                state = .pendingUpgrade(single)
            } else if single.inputIsBatchable,
                single.contextEpoch == contextEpoch,
                canUpgrade(parameters: single.parameters)
            {
                // The running request's caches can migrate single→batch; perform
                // the handshake and join both into the batch.
                await upgrade(from: single, joining: scheduled)
            } else {
                // The running request's cache topology (SSM/composite/chunked, or
                // KV-quantized) cannot migrate. Never interrupt/truncate the
                // first request: keep it on its single iterator and hold this
                // request until it finishes (then it is drained — see
                // `singleTaskFinished`). Enter `.pendingUpgrade` to mark "single
                // still running, requests queued".
                queuedRequests.append(scheduled)
                state = .pendingUpgrade(single)
            }

        case .pendingUpgrade:
            // A non-upgradeable single request is still running with requests
            // already queued behind it; keep queuing so it stays uninterrupted.
            // (State stays `.pendingUpgrade`.)
            queuedRequests.append(scheduled)

        case .upgrading:
            // An upgrade handshake is mid-flight; admit this request to the
            // (about-to-be) batched engine. It joins the batch on the next step.
            await admitToBatch(scheduled)

        case .batched:
            await admitToBatch(scheduled)
        }
    }
}

// MARK: - Single-stream path

extension InferenceScheduler {

    /// Non-`Sendable` bundle of everything the single task needs that cannot
    /// cross into the detached `Task` directly. Transferred via ``SendableBox``
    /// (the repo's existing hand-off box; no new `@unchecked`). The
    /// ``TokenIterator`` is built on the actor (so its prefill error surfaces to
    /// the caller) and carried here so the task owns it after the single
    /// transfer. (Sendable inputs like `parameters` are passed to `runSingle`
    /// directly, not boxed.)
    private struct SingleTaskInputs {
        let configuration: ModelConfiguration
        let tokenizer: Tokenizer
        let iterator: any TokenIteratorProtocol
    }

    /// Start a request on the zero-overhead single-stream path. The task drives
    /// a ``TokenIterator``, streams decoded text to the handler, and checks the
    /// upgrade flag after each token so it can hand off its live state.
    ///
    /// The ``TokenIterator`` is constructed **here, on the actor**, before this
    /// returns — its prefill runs the model's `prepare`, which can throw (bad
    /// input / model configuration). Building it synchronously (rather than
    /// inside the detached task) lets that error propagate to the original
    /// `submit`/``ModelContainer/generate`` caller, mirroring the historical
    /// non-scheduler ``generate(input:cache:parameters:context:wiredMemoryTicket:tools:)``
    /// which also constructs the iterator before handing back its stream. The
    /// non-`Sendable` iterator is then transferred into the detached task via
    /// the repo's existing ``SendableBox`` (no new `@unchecked`).
    private func startSingle(_ scheduled: sending ScheduledRequest) throws {
        guard let context else {
            scheduled.handler.finish()
            return
        }

        // Reuse the stream's stable id (allocated in `submitOne`) so the generic
        // `cancel(requestID:)` resolves to this task.
        let requestID = scheduled.requestID

        let upgradeFlag = UpgradeFlag()
        let handler = scheduled.handler
        let parameters = scheduled.request.parameters
        let inputTokens = scheduled.inputTokens
        let modelName = scheduled.modelName
        let salt = scheduled.request.promptCacheSalt
        let writeBack = scheduled.request.promptCache != nil
        let submitTime = Date.timeIntervalSinceReferenceDate
        let promptTokenCount = inputTokens.count
        let ticket = scheduled.request.wiredMemoryTicket
        // Captured before the input is transferred into the iterator/task:
        // gates the single->batch upgrade, since a masked input's prefill
        // positions cannot be reproduced by the engine's flat token rows.
        let batchableInput = Self.inputIsBatchable(scheduled.request.input)

        // Cross-request prompt-cache reuse: fetch the nearest cached prefix and
        // prefill only the remainder. The fetched cache is a deep copy owned by
        // this request, so the stored entry is never aliased. (Batched-path
        // fetch — mixed-depth prefill in the engine — is not wired yet; engine
        // rows still prefill fully and only write back.)
        // Honor the request's own cache instance first (GenerationRequest
        // documents `promptCache` as the cache to use); the attach-time
        // container cache is the fallback. The auto-routed ModelContainer
        // path sets both to the same instance.
        let effectivePromptCache = scheduled.request.promptCache ?? self.promptCache
        // One eligibility decision for BOTH fetch and write-back: a request
        // that cannot fetch (e.g. masked input, whose cache-hit rewrite would
        // drop the mask) must not write back either -- its KV would be keyed
        // by flattened tokens and pollute the cache for unmasked requests.
        let cacheEligible = LRUPromptCache.canUsePromptCache(
            input: scheduled.request.input, parameters: parameters, model: context.model)

        var iteratorCache: [KVCache]? = nil
        var iteratorInput = scheduled.request.input
        if let promptCache = effectivePromptCache, writeBack, cacheEligible {
            let fetch = promptCache.fetchNearestCacheResult(
                model: modelName, tokens: inputTokens, salt: salt)
            if let fetched = fetch.cache {
                var remainder = fetch.remainder
                if remainder.isEmpty, canTrimPromptCache(fetched), let last = inputTokens.last {
                    // Exact hit: the iterator still needs at least one token to
                    // evaluate, so trim the final token's KV and re-feed it.
                    trimPromptCache(fetched, numTokens: 1)
                    remainder = [last]
                }
                if !remainder.isEmpty {
                    iteratorCache = fetched
                    iteratorInput = LMInput(tokens: MLXArray(remainder.map { Int32($0) }))
                }
            }
        }

        // Construct the iterator on the actor so a failing prefill/`prepare`
        // throws to the caller (the stream has not been handed back yet on the
        // primary `idle` route). A thrown error leaves `state`/the handler
        // untouched — no task is launched and the request is simply not started.
        let iterator = try TokenIterator(
            input: iteratorInput,
            model: context.model,
            cache: iteratorCache,
            parameters: parameters
        )

        // Box the non-`Sendable` config/tokenizer and the already-built iterator
        // so the detached `@Sendable` task can consume them; the iterator never
        // crosses the boundary except inside the box (transferred once, then
        // owned solely by the task). `parameters` is `Sendable` and captured
        // directly.
        let boxed = SendableBox(
            SingleTaskInputs(
                configuration: context.configuration,
                tokenizer: context.tokenizer,
                iterator: iterator
            ))

        // The (Sendable) prompt cache rides into the task for write-back on
        // natural completion; nil when this request did not opt in.
        let cacheForWriteBack = (writeBack && cacheEligible) ? effectivePromptCache : nil

        let task = Task<Void, Never> { [weak self] in
            let inputs = boxed.consume()
            await Self.runSingle(
                parameters: parameters,
                configuration: inputs.configuration,
                tokenizer: inputs.tokenizer,
                iterator: inputs.iterator,
                handler: handler,
                upgradeFlag: upgradeFlag,
                promptTokenCount: promptTokenCount,
                submitTime: submitTime,
                wiredMemoryTicket: ticket,
                promptCache: cacheForWriteBack,
                modelName: modelName,
                promptCacheSalt: salt,
                inputTokens: inputTokens,
                onFinished: { [weak self] tokenIds in
                    await self?.singleTaskFinished(
                        requestID: requestID, generatedTokenIds: tokenIds)
                }
            )
        }

        // Stream cancellation is wired generically in `submitOne` (routing to
        // `cancel(requestID:)`, which cancels this task when this request is the
        // running single one).

        let single = SingleRequest(
            requestID: requestID,
            task: task,
            handler: handler,
            upgradeFlag: upgradeFlag,
            parameters: parameters,
            inputTokens: inputTokens,
            modelName: modelName,
            promptCacheSalt: salt,
            writeBackToPromptCache: writeBack,
            promptCache: cacheForWriteBack,
            submitTime: submitTime,
            wiredMemoryTicket: ticket,
            inputIsBatchable: batchableInput,
            contextEpoch: contextEpoch
        )
        state = .single(single)
    }

    /// The single-request decode loop. Runs detached from the actor and owns the
    /// non-`Sendable` `TokenIterator` (built on the actor by ``startSingle(_:)``
    /// and transferred in once) so the iterator never crosses an isolation
    /// boundary afterward. On an upgrade request it deposits a `sending`
    /// ``LiveIteratorState`` and exits without finishing the stream (the batch
    /// loop takes over). Otherwise it finishes the stream itself.
    /// How the synchronous single decode loop exited.
    private enum SingleLoopOutcome {
        /// An upgrade was requested: live state was captured (to deposit after
        /// the wired-memory scope closes).
        case deposit(LiveIteratorState)
        /// The loop completed normally with the given stop reason.
        case finished(GenerateStopReason)
    }

    private static func runSingle(
        parameters: GenerateParameters,
        configuration: ModelConfiguration,
        tokenizer: Tokenizer,
        iterator: consuming any TokenIteratorProtocol,
        handler: SchedulerTokenHandler,
        upgradeFlag: UpgradeFlag,
        promptTokenCount: Int,
        submitTime: TimeInterval,
        wiredMemoryTicket: WiredMemoryTicket?,
        promptCache: (any PromptCaching)?,
        modelName: String,
        promptCacheSalt: UInt64,
        inputTokens: [Int],
        onFinished: @Sendable @escaping ([Int]) async -> Void
    ) async {
        // The iterator was constructed on the actor (its prefill `prepare` ran
        // there, surfacing any init error to the caller before the stream was
        // returned); the task only drives it.

        // Same stop set the synchronous single-stream loop uses.
        let stopTokenIds = Self.stopTokenIds(
            configuration: configuration, tokenizer: tokenizer)
        let unknownTokenId = tokenizer.unknownTokenId

        var it = iterator
        var generated: [Int] = []
        // When the first token came back from the iterator; splits the
        // completion info into promptTime (queue + prefill) and
        // generationTime (decode), matching EngineDriver.deliver.
        var firstTokenAt: TimeInterval? = nil
        // The stop token that ended the loop, if any. Its forward pass already
        // ran (the step that returned it wrote its KV), so the prompt-cache
        // write-back key must include it even though it is never emitted.
        var finalStopToken: Int? = nil

        // The synchronous decode loop, run under the wired-memory limit when a
        // ticket is supplied (parity with `generateTask`). It captures live
        // state on upgrade instead of returning, so the deposit happens after
        // the limit scope closes.
        func decodeLoop() -> SingleLoopOutcome {
            var stopReason: GenerateStopReason? = nil
            var cancelled = false
            while true {
                if Task.isCancelled {
                    cancelled = true
                    break
                }
                // Upgrade hand-off point: capture live state and stop. After
                // this the task never touches `it`/its caches again
                // (region-isolation invariant).
                if upgradeFlag.upgradeRequested {
                    if let concrete = it as? TokenIterator {
                        return .deposit(
                            LiveIteratorState(
                                cache: concrete.cache,
                                currentToken: concrete.y.tokens.item(Int.self),
                                tokenCount: concrete.tokenCount,
                                maxTokens: concrete.maxTokens,
                                parameters: parameters,
                                generatedTokenIds: generated,
                                firstTokenAt: firstTokenAt
                            ))
                    }
                    // A non-plain iterator (e.g. a future speculative one on
                    // this path) has no batch-migratable snapshot: decline
                    // the handshake so the scheduler falls back, and keep
                    // decoding single. Unreachable today -- `startSingle`
                    // only constructs plain `TokenIterator`s -- but keeps
                    // the loop honest once other iterators arrive here.
                    upgradeFlag.markTaskFinished()
                }

                guard let token = it.next() else {
                    break
                }
                if firstTokenAt == nil {
                    firstTokenAt = Date.timeIntervalSinceReferenceDate
                }

                // Stop-token detection mirrors `runSynchronousGenerationLoop`:
                // the stop token is NOT emitted to the consumer.
                if token == unknownTokenId || stopTokenIds.contains(token) {
                    stopReason = .stop
                    finalStopToken = token
                    break
                }

                generated.append(token)

                switch handler.processToken(token) {
                case .more:
                    break
                case .stop:
                    // A configured stop string completed in the decoded text;
                    // finish exactly like a stop token (the filter already
                    // emitted the text ahead of the stop and suppressed the
                    // rest).
                    stopReason = .stop
                case .cancelled:
                    cancelled = true
                }
                if stopReason != nil || cancelled {
                    break
                }
            }

            // Resolve the final stop reason, matching the synchronous loop.
            if cancelled {
                return .finished(.cancelled)
            } else if let stopReason {
                return .finished(stopReason)
            } else if let maxTokens = it.maxTokens, it.tokenCount >= maxTokens {
                return .finished(.length)
            } else {
                return .finished(.cancelled)
            }
        }

        let outcome: SingleLoopOutcome
        if let ticket = wiredMemoryTicket {
            var captured: SingleLoopOutcome = .finished(.cancelled)
            await WiredMemoryTicket.withWiredLimit(ticket) {
                captured = decodeLoop()
            }
            outcome = captured
        } else {
            outcome = decodeLoop()
        }

        if case .deposit(let live) = outcome {
            // Hand the live state to the scheduler and exit, leaving the stream
            // open for the batch loop to continue.
            upgradeFlag.depositLiveState(SendableBox(live))
            return
        }
        guard case .finished(let resolvedReason) = outcome else { return }

        // If an upgrade was requested in the same instant we exited, hand off
        // nil so the scheduler does not hang waiting on the continuation.
        upgradeFlag.markTaskFinished()

        handler.processEndOfSequence()
        let now = Date.timeIntervalSinceReferenceDate
        // Split at the first token, exactly like EngineDriver's completion
        // info: promptTime covers queueing + prefill, generationTime covers
        // decode only. The previous `now - submitTime` counted prefill in
        // BOTH fields, deflating reported generation tokens/sec.
        let firstToken = firstTokenAt ?? now
        let info = GenerateCompletionInfo(
            promptTokenCount: promptTokenCount,
            generationTokenCount: generated.count,
            promptTime: max(0, firstToken - submitTime),
            generationTime: max(0, now - firstToken),
            stopReason: resolvedReason
        )
        handler.yieldInfo(info)
        handler.finish()

        // Write the finished request's KV state back to the prompt cache (the
        // single-path counterpart of EngineDriver.writeBack). The iterator's
        // caches cover the full prompt plus every token whose forward pass
        // ran: all generated tokens, and the stop token when the loop broke on
        // EOS. `insertCache` deep-copies, so the stored entry never aliases
        // the iterator's live caches; incompatible caches (e.g. quantized via
        // kvBits mid-run) are rejected inside `insertCache`.
        if let promptCache {
            var coveredTokens = inputTokens + generated
            if let finalStopToken {
                coveredTokens.append(finalStopToken)
            }
            promptCache.insertCache(
                model: modelName,
                tokens: coveredTokens,
                promptCache: it.cache,
                salt: promptCacheSalt
            )
        }

        await onFinished(generated)
    }

    /// The EOS/stop token id set for the single path, matching the in-module
    /// `buildStopTokenIds` used by the synchronous generation loop.
    private static func stopTokenIds(
        configuration: ModelConfiguration,
        tokenizer: Tokenizer
    ) -> Set<Int> {
        var ids = configuration.eosTokenIds
        if let eos = tokenizer.eosTokenId {
            ids.insert(eos)
        }
        for token in configuration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                ids.insert(id)
            }
        }
        // The single-stream loop stops on the unknown token as well; include
        // it here so batched rows (and the upgrade-handoff EOS triage) share
        // the same stop semantics.
        if let unknown = tokenizer.unknownTokenId {
            ids.insert(unknown)
        }
        return ids
    }
}

// MARK: - Batched path

extension InferenceScheduler {

    /// Lazily create the ``EngineDriver`` (and its ``BatchGenerationEngine``)
    /// for the bound context. Reuses an existing driver unless it is stale
    /// AND idle, in which case it is torn down and rebuilt from the current
    /// context. Returns `nil` if the model's cache topology cannot be batched
    /// (caller falls back to single).
    private func ensureDriver() async -> EngineDriver? {
        if let existing = driver {
            guard driverStale else { return existing }
            // The context was replaced (or the owner detached) while this
            // driver existed. While it still drains, keep returning it so
            // in-flight rows — and rows joining that batch — finish on the
            // model they started with (documented continuity limitation, see
            // `refreshContext`). Once idle, tear it down here instead of
            // waiting for a `batchLoopFinished` that may never run again.
            if await existing.hasWork {
                return existing
            }
            // `hasWork` suspended; a concurrent `refreshContext` may already
            // have torn the driver down or a concurrent `ensureDriver` may
            // have replaced it. Only tear down the instance we checked.
            if self.driver === existing {
                self.driver = nil
                driverStale = false
            } else if let replacement = self.driver {
                return replacement
            }
        }
        guard let context else { return nil }

        let engine: BatchGenerationEngine
        do {
            engine = try BatchGenerationEngine(
                model: context.model,
                eosTokens: Self.eosSequences(
                    configuration: context.configuration, tokenizer: context.tokenizer),
                prefillStepSize: configuration.prefillStepSize,
                prefillBatchSize: configuration.prefillBatchSize,
                completionBatchSize: configuration.completionBatchSize
            )
        } catch {
            return nil
        }

        let driver = EngineDriver(engine: SendableBox(engine), promptCache: promptCache)
        // Assign BEFORE the awaits below: actor reentrancy could otherwise let
        // a concurrent admission observe `driver == nil` while this call is
        // suspended and construct a second engine over the same model.
        self.driver = driver
        if hasPendingMaxBatchSize {
            await driver.setMaxBatchSize(pendingMaxBatchSize)
        } else if let cap = configuration.maxBatchSize {
            await driver.setMaxBatchSize(cap)
        }
        // Re-check after the await(s): a concurrent `refreshContext` may have
        // nilled or replaced `self.driver` while this call was suspended.
        // Never resurrect the discarded instance — return whatever is current
        // (possibly nil; callers already handle that).
        guard self.driver === driver else { return self.driver }
        return driver
    }

    /// EOS token sequences for the batched engine's `StopSequenceMatcher`
    /// (one single-token stop per EOS id).
    private static func eosSequences(
        configuration: ModelConfiguration,
        tokenizer: Tokenizer
    ) -> [[Int]] {
        stopTokenIds(configuration: configuration, tokenizer: tokenizer).map { [$0] }
    }

    /// The default per-row stop matcher (model EOS), matching what
    /// `BatchGenerationEngine.insert` installs for freshly-admitted rows. Used
    /// for the adopted (upgraded) row so it keeps stopping on EOS instead of
    /// running to `maxTokens` (`DecodeBatch.init` does not fall back to the
    /// engine default the way `insert` does).
    private func defaultStopMatcher() -> StopSequenceMatcher? {
        guard let context else { return nil }
        let sequences = Self.eosSequences(
            configuration: context.configuration, tokenizer: context.tokenizer)
        guard !sequences.isEmpty else { return nil }
        return StopSequenceMatcher(
            states: ["normal": sequences.map { (sequence: $0, next: nil) }],
            initial: "normal"
        )
    }

    /// Admit a request directly to the batch engine and ensure the driver's
    /// decode loop is running. Used for 3rd+ concurrent requests and for the
    /// manual ``generateBatched(_:)`` path.
    private func admitToBatch(_ scheduled: sending ScheduledRequest) async {
        // Topology-changing parameters (kvBits / maxKVSize) and masked inputs
        // (see `inputIsBatchable`) cannot run on the shared engine. Hold such
        // a request until the scheduler is idle again, then run it on its own
        // single iterator — never silently drop its cache configuration or
        // attention mask.
        guard Self.parametersAreBatchable(scheduled.request.parameters),
            Self.inputIsBatchable(scheduled.request.input)
        else {
            if case .idle = state {
                startSingleIfIdle(scheduled)
            } else {
                queuedRequests.append(scheduled)
            }
            return
        }

        // `ensureDriver` suspends; resolve it before consuming `scheduled` so
        // the non-Sendable request is not held across the await unnecessarily.
        guard let driver = await ensureDriver() else {
            // Cannot batch this model — run the request on the single path
            // instead (correctness over batching).
            startSingleIfIdle(scheduled)
            return
        }
        state = .batched

        // Pull the Sendable routing fields out first, then build the request as
        // the sole non-Sendable value handed to the driver via `sending`.
        let handler = scheduled.handler
        let cancelToken = scheduled.requestID
        let sampler = Self.rowSampler(for: scheduled.request.parameters)
        let processorSource = Self.rowProcessorSource(for: scheduled.request.parameters)
        let maxTokens = scheduled.request.parameters.maxTokens ?? defaultMaxTokens
        let req = SchedulerRequest(
            input: scheduled.request.input,
            parameters: scheduled.request.parameters,
            inputTokens: scheduled.inputTokens,
            modelName: scheduled.modelName,
            promptCache: scheduled.request.promptCache,
            promptCacheSalt: scheduled.request.promptCacheSalt,
            wiredMemoryTicket: scheduled.request.wiredMemoryTicket
        )

        batchAdmissionEpoch += 1
        await driver.submit(
            req,
            handler: handler,
            cancelToken: cancelToken,
            maxTokens: maxTokens,
            sampler: sampler,
            processorSource: processorSource,
            stateMachine: nil
        )
        // Actor mailboxes are not FIFO: a `batchLoopFinished` whose
        // `driver.hasWork` read was served BEFORE the submit above landed can
        // have passed its admission-epoch guard and idled the scheduler while
        // this row is now live. Self-heal on the submit side: re-assert
        // `.batched` when a stale loop-finish idled it — but never clobber a
        // `.single`/`.pendingUpgrade` a concurrent path installed meanwhile
        // (the state machine still tracks that request; the driver fans this
        // row's tokens out regardless). `startBatchLoop` below restarts the
        // loop if needed; `drain` is ownership-idempotent
        // (`if draining { return false }`), so a redundant start is a no-op.
        if case .idle = state {
            state = .batched
        }
        startBatchLoop(driver)
    }

    /// Run a non-batchable request on its own single iterator. Two concurrent
    /// single tasks are safe (the single path shares no engine state; each owns
    /// its caches and the model weights are already evaluated), so this runs the
    /// request single regardless of state. The stream was already handed back to
    /// the caller, so a failing iterator prefill cannot be thrown out here — it
    /// is surfaced by closing the stream (the request simply yields nothing),
    /// matching the pre-existing best-effort handling on this path.
    private func startSingleIfIdle(_ scheduled: sending ScheduledRequest) {
        let handler = scheduled.handler
        do {
            try startSingle(scheduled)
        } catch {
            handler.finish()
            // `startSingle` throws before installing any state; make sure a
            // stale `.upgrading` marker from the calling path cannot strand
            // the scheduler (subsequent requests would route to the batch
            // forever even with nothing running).
            if case .upgrading = state { state = .idle }
        }
    }

    /// Build a `RowSampler` from generation parameters: temperature / top-p /
    /// min-p / top-k, matching the single path's `TopPSampler` filter order.
    ///
    /// Returns `nil` for greedy parameters (`temperature <= 0`): the engine's
    /// default fallback is already greedy, and any non-nil per-row sampler
    /// disables `DecodeBatch`'s whole-batch `argMax` fast path.
    static func rowSampler(for parameters: GenerateParameters) -> RowSampler? {
        guard parameters.temperature > 0 else { return nil }
        return makeRowSampler(
            temperature: parameters.temperature,
            topP: parameters.topP,
            topK: parameters.topK,
            minP: parameters.minP,
            // Honor the request's seed (GenerateParameters.seed, upstream
            // #377) so batched sampling is reproducible per request. The
            // per-row key sequence differs from the single path's sampler, so
            // a seeded request is deterministic on each path but not bitwise
            // identical across single vs. batched.
            seed: parameters.seed
        )
    }

    /// Build the per-row penalty-processor source (repetition / presence /
    /// frequency) from generation parameters. The source is `Sendable`; the
    /// engine materializes the stateful processor on its own executor, and it
    /// yields `nil` (no per-step cost) when no penalties are configured.
    static func rowProcessorSource(for parameters: GenerateParameters) -> RowProcessorSource {
        makeRowProcessorSource(parameters)
    }

    /// Per-request token budget when `maxTokens` is unset. The single path
    /// treats nil as unlimited (`TokenIterator` never length-stops), so the
    /// batched path must not invent a smaller limit — otherwise a request's
    /// output length would depend on whether another request happened to
    /// arrive concurrently.
    private var defaultMaxTokens: Int { Int.max }

    /// Start (or no-op if already running) the driver's decode loop as a
    /// detached task. The loop ends when the engine empties; on completion the
    /// scheduler is returned to `.idle` if no work remains.
    ///
    /// No explicit wired-memory ticket is passed: the driver applies the ticket
    /// it captured from admitted requests (see `EngineDriver.drain`).
    private func startBatchLoop(_ driver: EngineDriver) {
        Task { [weak self] in
            // `drain` returns false when a loop is already running; only the
            // call that actually owned the loop runs the completion callback,
            // so no-op drains cannot spawn stale `batchLoopFinished` races.
            if await driver.drain(onResult: nil) {
                await self?.batchLoopFinished()
            }
        }
    }

    /// Called when the driver's decode loop finishes. Returns to `.idle` if no
    /// work remains so the next lone request starts fresh on the single path.
    private func batchLoopFinished() async {
        guard let driver else {
            // The driver was torn down while this loop-finish was in flight
            // (context refresh). Same never-clobber rule as below: only a
            // still-`.batched` state may be idled — a `.single`/`.upgrading`
            // installed concurrently is tracking live work.
            if case .batched = state {
                state = .idle
            }
            await drainQueuedRequests()
            return
        }
        let epochAtExit = batchAdmissionEpoch
        if await driver.hasWork {
            // More work arrived after the loop exited; restart it.
            startBatchLoop(driver)
            return
        }
        // A submission that raced with the `hasWork` read above bumped the
        // epoch while this call was suspended; restart the loop for it
        // instead of idling underneath it.
        guard epochAtExit == batchAdmissionEpoch else {
            startBatchLoop(driver)
            return
        }
        // The context was replaced while this batch drained; drop the driver
        // now that it is confirmed idle so the next batch is built from the
        // fresh context (see `refreshContext`). This runs BEFORE the
        // `.batched` guard below: when a concurrent path already moved the
        // state machine on (early return), the stale driver must still be
        // torn down here or it would survive — idle — indefinitely and serve
        // a later batch with the old model.
        if driverStale {
            self.driver = nil
            driverStale = false
        }
        // Only transition out of `.batched`: a `.single`/`.upgrading` state
        // installed by a concurrent path after the batch emptied must not be
        // clobbered back to `.idle` (its request would become untracked).
        guard case .batched = state else { return }
        state = .idle
        // Requests held because their parameters could not run on the
        // shared engine (see `parametersAreBatchable`) are drained once
        // the batch empties, mirroring `singleTaskFinished`.
        await drainQueuedRequests()
    }
}

// MARK: - Single → batch upgrade (3-phase handshake)

extension InferenceScheduler {

    /// Upgrade the running single request `single` into a batch that also
    /// includes the newly-arrived `joining` request.
    ///
    /// Phase 1: set `state = .upgrading` (so a 3rd request during the window
    /// runs/admits independently), then await the single task's live decode
    /// state via the ``UpgradeFlag`` `CheckedContinuation` handshake. Phase 2:
    /// migrate the live caches with PR1 `fromSingle`, build an adopted
    /// ``DecodeBatch``, and hand it to the ``EngineDriver``. Phase 3: admit
    /// `joining`, reuse the first request's original handler (stream
    /// continuity), and start the batch decode loop.
    ///
    /// If the handshake yields `nil` (the single task already finished) or the
    /// caches cannot be migrated, both requests fall back: the single task is
    /// done, so `joining` simply starts on the (now idle) single or batch path.
    private func upgrade(
        from single: SingleRequest,
        joining: sending ScheduledRequest
    ) async {
        // Capture the epoch BEFORE the handshake suspension. `route` verified
        // `single.contextEpoch == contextEpoch` at dispatch time, but the
        // `withCheckedContinuation` below suspends this actor: a
        // `refreshContext` (public `ModelContainer.update`) landing in that
        // window swaps the model and bumps the epoch, and nothing on the
        // resume path re-validated. Splicing the deposited caches into a
        // driver built from the NEW context would decode old-model KV with
        // new-model weights — real corruption — so the epoch is re-checked
        // right after the deposit is consumed.
        let epochAtHandshake = contextEpoch

        // Phase 1: enter the upgrade window and request the snapshot.
        state = .upgrading

        let liveBox: SendableBox<LiveIteratorState>? = await withCheckedContinuation {
            continuation in
            single.upgradeFlag.requestUpgrade(continuation: continuation)
        }

        guard let liveBox else {
            // The single task finished before it could deposit state. Its
            // stream is already (or about to be) closed by its own loop; start
            // the joining request fresh. It is (almost always) alone now, so
            // prefer the zero-overhead single path; only join the engine when
            // a third request already put live work in the driver (a single
            // beside an active batch would decode concurrently with it).
            if let driver, await driver.hasWork {
                await admitToBatch(joining)
            } else {
                startSingleAfterUpgradeFallback(joining)
            }
            return
        }

        let live = liveBox.consume()

        // The single task has stopped and deposited `live`; we now own it.
        // Pull out every value-typed (Sendable) field up front so that, after
        // the cache is migrated below, `live` (and its non-Sendable caches) is
        // provably dead — the migrated caches can then transfer to the driver
        // via `sending` with no aliasing back into `live`.
        let liveCurrentToken = live.currentToken
        let liveTokenCount = live.tokenCount
        let liveMaxTokens = live.maxTokens
        let liveParameters = live.parameters
        let liveGeneratedTokenIds = live.generatedTokenIds
        let liveFirstTokenAt = live.firstTokenAt

        // `liveCurrentToken` is the token the single `TokenIterator` already
        // sampled but has not yet returned/delivered. It must be EOS-checked and
        // delivered here; the adopted batch then seeds from it (the batch's
        // priming step consumes the seed without re-emitting it, so delivering
        // it manually avoids dropping exactly one token at the hand-off).
        let stopIds: Set<Int>
        if let context {
            stopIds = Self.stopTokenIds(
                configuration: context.configuration, tokenizer: context.tokenizer)
        } else {
            stopIds = []
        }
        let liveTokenIsStop = stopIds.contains(liveCurrentToken)

        // Use the engine's own default budget when unset, matching admitToBatch.
        // `liveCurrentToken` is the (liveTokenCount + 1)-th token: it was sampled
        // by the iterator's pump but only DELIVERED if the iterator would have
        // returned it on its next `next()`. `TokenIterator.next()` returns nil
        // once `tokenCount >= maxTokens`, so the live token is real and
        // deliverable iff `liveTokenCount < budgetLimit`, i.e.
        // `remainingBudget >= 1`. When `remainingBudget == 0` the request has
        // already emitted its full `maxTokens`; `liveCurrentToken` is a purely
        // speculative precompute and must NOT be delivered or counted (doing so
        // would emit/count one token past the caller's maximum).
        //
        // The request still needs the batch only if at least one more token
        // fits after delivering the live one, i.e. `remainingBudget >= 2`. When
        // `remainingBudget <= 1` it is finalized here (delivering the live token
        // only if `remainingBudget == 1`), never adopted — adopting would let
        // `DecodeBatch.next()` (which increments before its length check) emit a
        // token past the maximum.
        let budgetLimit = liveMaxTokens ?? defaultMaxTokens
        let remainingBudget = budgetLimit - liveTokenCount
        // The final token to flush when finalizing without adoption: nil for a
        // stop token (never emitted) or when the budget is already exhausted
        // (the live token is speculative); otherwise the real undelivered token.
        let liveFinalToken = (liveTokenIsStop || remainingBudget <= 0) ? nil : liveCurrentToken

        // Re-validate the epoch AFTER the handshake suspension: if the context
        // was swapped while the single task was depositing, the caches in
        // `live` belong to the OLD model and must never be adopted into an
        // engine built from the new context. Finalize the original request
        // cleanly on the tokens it already produced (never truncate silently)
        // and start the joining request fresh on the current context. No
        // prompt-cache write-back: see `writeBackUpgradedSingle`'s epoch gate.
        guard contextEpoch == epochAtHandshake else {
            finishUpgradedSingle(
                handler: single.handler,
                lastToken: liveFinalToken,
                promptTokenCount: single.inputTokens.count,
                generatedCount: liveGeneratedTokenIds.count,
                submitTime: single.submitTime,
                reason: liveTokenIsStop ? .stop : .length
            )
            startSingleAfterUpgradeFallback(joining)
            return
        }

        guard let driver = await ensureDriver() else {
            // Cannot batch this model: deliver the last token, finalize the
            // original stream cleanly, then run the joining request fresh.
            writeBackUpgradedSingle(single, cache: live.cache, generated: liveGeneratedTokenIds)
            finishUpgradedSingle(
                handler: single.handler,
                lastToken: liveFinalToken,
                promptTokenCount: single.inputTokens.count,
                generatedCount: liveGeneratedTokenIds.count,
                submitTime: single.submitTime,
                reason: liveTokenIsStop ? .stop : .length
            )
            startSingleAfterUpgradeFallback(joining)
            return
        }

        if liveTokenIsStop || remainingBudget <= 1 {
            // The original is already done: EOS reached, or delivering the live
            // token reaches `budgetLimit` (no room for the batch to add another
            // without overshooting). Finalize it cleanly (delivering the final
            // non-stop token only when one is actually undelivered) and run the
            // joining request through the batch.
            writeBackUpgradedSingle(single, cache: live.cache, generated: liveGeneratedTokenIds)
            finishUpgradedSingle(
                handler: single.handler,
                lastToken: liveFinalToken,
                promptTokenCount: single.inputTokens.count,
                generatedCount: liveGeneratedTokenIds.count,
                submitTime: single.submitTime,
                reason: liveTokenIsStop ? .stop : .length
            )
            // The joining request is alone unless a third request already put
            // live work in the driver; prefer the zero-overhead single path.
            if await driver.hasWork {
                await admitToBatch(joining)
            } else {
                startSingleAfterUpgradeFallback(joining)
            }
            return
        }

        // Deliver the live token BEFORE cache migration (the batch will seed
        // from it but not re-emit it), so a stop-string completion can still
        // finalize with the un-migrated caches in hand.
        switch single.handler.processToken(liveCurrentToken) {
        case .stop:
            // The handoff token completed a configured stop string: finish
            // like a semantic stop instead of adopting (adoption would decode
            // one extra token before the latched filter caught it, over-
            // counting the generation). The token was already delivered; its
            // forward pass never ran, so the write-back key excludes it.
            writeBackUpgradedSingle(single, cache: live.cache, generated: liveGeneratedTokenIds)
            finishUpgradedSingle(
                handler: single.handler,
                lastToken: nil,
                promptTokenCount: single.inputTokens.count,
                generatedCount: liveGeneratedTokenIds.count + 1,
                submitTime: single.submitTime,
                reason: .stop
            )
            if await driver.hasWork {
                await admitToBatch(joining)
            } else {
                startSingleAfterUpgradeFallback(joining)
            }
            return
        case .more, .cancelled:
            // `.cancelled` is routed by the generic stream-cancellation
            // wiring; adoption proceeds and the driver drops the row.
            break
        }

        // Migrate the live caches LAST (consumes `live.cache`). After this point
        // `live` must not be read.
        guard let batchedCaches = Self.migrateCaches(live.cache) else {
            // Migration unsupported for this cache topology. The original's KV
            // cannot be carried into the batch, so finalize it cleanly
            // (delivering its last token + completion info — never truncate
            // silently) and run the joining request fresh.
            // TODO(PR4 auto-upgrade): extend fromSingle to SSM/composite so this
            // fallback is unreachable for those models.
            finishUpgradedSingle(
                handler: single.handler,
                lastToken: liveCurrentToken,
                promptTokenCount: single.inputTokens.count,
                generatedCount: liveGeneratedTokenIds.count,
                submitTime: single.submitTime,
                reason: .length
            )
            // Same solo-request preference as above.
            if await driver.hasWork {
                await admitToBatch(joining)
            } else {
                startSingleAfterUpgradeFallback(joining)
            }
            return
        }

        // Phase 2: build the adopted decode batch from migrated state. The
        // adopted row's engine UID is allocated by the engine (the single
        // source of truth for row identity) inside `adoptMigrated`, never from
        // the scheduler's `nextRequestID`, so a later `insert` cannot reuse it
        // and overwrite the row's record. The scheduler-owned request id rides
        // along as `cancelToken` for stream-cancellation routing.
        let sampler = Self.rowSampler(for: liveParameters)
        let processorSource = Self.rowProcessorSource(for: liveParameters)
        let stopMatcher = defaultStopMatcher()
        let firstRecord = EngineDriver.AdoptedRecord(
            handler: single.handler,
            // The adopted row keeps the original request's stable id, so a later
            // stream cancellation routes through `cancel(requestID:)` to it.
            cancelToken: single.requestID,
            promptTokenCount: single.inputTokens.count,
            inputTokens: single.inputTokens,
            modelName: single.modelName,
            promptCacheSalt: single.promptCacheSalt,
            promptCache: single.promptCache,
            submitTime: single.submitTime,
            writeBackToPromptCache: single.writeBackToPromptCache,
            firstTokenAt: liveFirstTokenAt
        )
        // Build + splice the migrated batch entirely on the driver (the
        // DecodeBatch never crosses an isolation boundary). The live token was
        // already produced, so the adopted row's budget counts it (numTokens =
        // liveTokenCount + 1) and it keeps the model's EOS stop matcher.
        //
        // `tokens` is the FULL history -- prompt, generated tokens, and the
        // seed (`liveCurrentToken`). The prompt prefix matters: the adopted
        // row's penalty processor is seeded from this history, and the single
        // iterator it replaces was seeded with the prompt before decoding, so
        // omitting it would let post-handoff rows repeat prompt tokens still
        // inside the penalty context window. `makeAdoptedBatch` drops the
        // trailing seed and the batch's priming `step()` re-appends exactly
        // that token, so the post-priming history equals the true sequence
        // with no duplicate (parity with `PrefillBatch.generate`).
        await driver.adoptMigrated(
            seedToken: liveCurrentToken,
            caches: batchedCaches,
            sampler: sampler,
            processorSource: processorSource,
            stateMachine: stopMatcher,
            maxTokens: budgetLimit,
            numTokens: liveTokenCount + 1,
            tokens: single.inputTokens + liveGeneratedTokenIds + [liveCurrentToken],
            record: firstRecord
        )
        // Carry the migrated request's wired-memory ticket onto the batch loop
        // (the adopted row bypasses `admit`, which captures fresh tickets).
        await driver.noteWiredMemoryTicket(single.wiredMemoryTicket)

        // Phase 3: admit the joining request and start the batch loop. Pull the
        // Sendable fields out first so the request is the sole non-Sendable
        // value handed to the driver via `sending`.
        state = .batched

        let joinHandler = joining.handler
        let joinCancelToken = joining.requestID
        let joinSampler = Self.rowSampler(for: joining.request.parameters)
        let joinProcessorSource = Self.rowProcessorSource(for: joining.request.parameters)
        let joinMax = joining.request.parameters.maxTokens ?? defaultMaxTokens
        let req = SchedulerRequest(
            input: joining.request.input,
            parameters: joining.request.parameters,
            inputTokens: joining.inputTokens,
            modelName: joining.modelName,
            promptCache: joining.request.promptCache,
            promptCacheSalt: joining.request.promptCacheSalt,
            wiredMemoryTicket: joining.request.wiredMemoryTicket
        )
        batchAdmissionEpoch += 1
        await driver.submit(
            req,
            handler: joinHandler,
            cancelToken: joinCancelToken,
            maxTokens: joinMax,
            sampler: joinSampler,
            processorSource: joinProcessorSource,
            stateMachine: nil
        )

        // Same submit-side self-heal as `admitToBatch`: a stale
        // `batchLoopFinished` served between `.batched` above and the submit
        // landing may have idled the scheduler under the live batch.
        if case .idle = state {
            state = .batched
        }
        startBatchLoop(driver)
    }

    /// Write an upgrade-finalized single's KV back to its prompt cache -- the
    /// counterpart of `runSingle`'s natural-completion `insertCache`, which
    /// the upgrade deposit skipped (`runSingle` returns before it). Keyed
    /// WITHOUT the handoff token: its forward pass never ran, so the caches
    /// do not cover it. Incompatible topologies are rejected inside
    /// `insertCache`.
    private func writeBackUpgradedSingle(
        _ single: SingleRequest, cache: [KVCache], generated: [Int]
    ) {
        guard single.writeBackToPromptCache, let promptCache = single.promptCache else { return }
        // Epoch gate: after a `refreshContext`, the KV in `cache` was computed
        // by the OLD model, but `ModelContainer.update` can swap weights
        // without renaming the configuration — the captured `modelName` (the
        // cache key) cannot be trusted to isolate the entry, so writing it
        // back could serve old-model KV to new-model requests. Skip instead.
        // (This also covers finalize paths reached via `ensureDriver`, which
        // suspends and can observe a refresh mid-flight.)
        guard single.contextEpoch == contextEpoch else { return }
        promptCache.insertCache(
            model: single.modelName,
            tokens: single.inputTokens + generated,
            promptCache: cache,
            salt: single.promptCacheSalt
        )
    }

    /// Finalize an upgraded single request's stream when it will not actually
    /// join the batch (EOS reached at the hand-off, no budget, or unsupported
    /// cache migration). Delivers a final non-stop token (if any), flushes
    /// pending tool calls, emits completion info, and closes the stream — so the
    /// original request is never silently truncated.
    private func finishUpgradedSingle(
        handler: SchedulerTokenHandler,
        lastToken: Int?,
        promptTokenCount: Int,
        generatedCount: Int,
        submitTime: TimeInterval,
        reason: GenerateStopReason
    ) {
        if let lastToken {
            _ = handler.processToken(lastToken)
        }
        handler.processEndOfSequence()
        let now = Date.timeIntervalSinceReferenceDate
        let info = GenerateCompletionInfo(
            promptTokenCount: promptTokenCount,
            generationTokenCount: generatedCount + (lastToken == nil ? 0 : 1),
            promptTime: 0,
            generationTime: max(0, now - submitTime),
            stopReason: reason
        )
        handler.yieldInfo(info)
        handler.finish()
    }

    /// Fallback when an upgrade cannot batch the model: run the joining request
    /// on a fresh single iterator after the upgrade window. Two concurrent
    /// single tasks are safe (no shared engine state). The joining stream is
    /// already live, so a failing iterator prefill is surfaced by closing the
    /// stream rather than thrown.
    private func startSingleAfterUpgradeFallback(_ scheduled: sending ScheduledRequest) {
        // Reset to idle so `startSingle` installs cleanly.
        state = .idle
        let handler = scheduled.handler
        do {
            try startSingle(scheduled)
        } catch {
            handler.finish()
        }
    }

    /// Called from the single task when it finishes naturally (not via
    /// upgrade). If the scheduler is still in `.single`/`.pendingUpgrade` for
    /// this request, return to `.idle` and drain any requests that were queued
    /// behind a non-upgradeable single request.
    private func singleTaskFinished(requestID: Int, generatedTokenIds: [Int]) async {
        let activeID: Int?
        switch state {
        case .single(let s): activeID = s.requestID
        case .pendingUpgrade(let s): activeID = s.requestID
        default: activeID = nil
        }
        guard activeID == requestID else {
            // The request was upgraded or superseded; nothing to do.
            return
        }
        // (Prompt-cache write-back happens inside `runSingle` on the task that
        // owns the iterator's caches, before this callback fires.)
        state = .idle
        await drainQueuedRequests()
    }

    /// Re-route requests that were held while a non-upgradeable single request
    /// ran. Drained in FIFO order: the first starts on the now-idle single path;
    /// subsequent ones land on `.single` and (for the same non-upgradeable
    /// model) re-queue behind it, naturally serializing them one-at-a-time so no
    /// running request is ever interrupted.
    private func drainQueuedRequests() async {
        guard !queuedRequests.isEmpty else { return }
        // Move the held requests out of the actor's storage, then re-route them.
        // Same boxing dance as `submitBatch`: elements of an actor-held array
        // cannot be sent while the array stays reachable, so each request is
        // boxed and consumed at the `sending` boundary.
        var boxes: [SendableBox<ScheduledRequest>] = []
        boxes.reserveCapacity(queuedRequests.count)
        var pending = Array(queuedRequests.reversed())
        queuedRequests.removeAll()
        while !pending.isEmpty {
            boxes.append(SendableBox(pending.removeLast()))
        }
        for box in boxes {
            let next = box.consume()
            // These streams are already live (returned to their callers when
            // queued), so an iterator-prefill failure during re-route cannot be
            // thrown out — close the stream so the request ends instead of
            // hanging open. Other routing errors are likewise terminal here.
            let handler = next.handler
            do {
                try await route(next)
            } catch {
                handler.finish()
            }
        }
    }
}
