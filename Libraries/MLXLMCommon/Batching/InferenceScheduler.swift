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
        /// Prompt prefill chunk size.
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

    /// Optional cross-request prompt cache (shared with the driver).
    private var promptCache: (any PromptCaching)?

    /// Current routing state.
    private var state: SchedulerState = .idle

    /// Monotonic request id for diagnostics.
    private var nextRequestID: Int = 0

    public init(configuration: Configuration = .init()) {
        self.configuration = configuration
    }

    /// Bind the model context the scheduler drives. Idempotent; the context is
    /// captured on first attach. Called by ``ModelContainer`` when the
    /// scheduler is first used.
    ///
    /// The context is transferred inside a `SendableBox` (the repo's existing
    /// non-`Sendable` hand-off box, also used by `generateTask`): ``ModelContext``
    /// carries non-`Sendable` model/processor references. The scheduler shares
    /// the (already-evaluated) model weights with the container, the same way
    /// ``ModelContainer/generate(input:parameters:wiredMemoryTicket:)`` permits
    /// concurrent access to evaluated weights. No NEW `@unchecked Sendable` is
    /// introduced — `SendableBox` is pre-existing infrastructure.
    func attach(context box: SendableBox<ModelContext>, promptCache: (any PromptCaching)?) {
        if self.context == nil {
            self.context = box.consume()
        }
        if self.promptCache == nil {
            self.promptCache = promptCache
        }
    }

    /// Whether the scheduler already has a bound context.
    var isAttached: Bool { context != nil }

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
        let submitTime: TimeInterval
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
        private struct State {
            var upgradeRequested = false
            var taskFinished = false
            var continuation: CheckedContinuation<LiveIteratorState?, Never>?
        }

        private let state = OSAllocatedUnfairLock<State>(uncheckedState: State())

        /// Whether the scheduler has asked this task to upgrade.
        var upgradeRequested: Bool {
            state.withLockUnchecked { $0.upgradeRequested }
        }

        /// Scheduler side: deposit the continuation and request the upgrade. If
        /// the task already finished, resume immediately with `nil`.
        func requestUpgrade(
            continuation: CheckedContinuation<LiveIteratorState?, Never>
        ) {
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
        /// `state` is transferred with `sending`; the caller stops touching it
        /// after this returns.
        func depositLiveState(_ liveState: sending LiveIteratorState) {
            let cont = state.withLockUnchecked { s -> CheckedContinuation<
                LiveIteratorState?, Never
            >? in
                let c = s.continuation
                s.continuation = nil
                return c
            }
            cont?.resume(returning: liveState)
        }

        /// Single-task side: mark the task finished. Resumes any pending
        /// continuation with `nil` so the scheduler does not hang.
        func markTaskFinished() {
            let cont = state.withLockUnchecked { s -> CheckedContinuation<
                LiveIteratorState?, Never
            >? in
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
        var remaining = requests
        var streams: [AsyncStream<Generation>] = []
        streams.reserveCapacity(remaining.count)
        while !remaining.isEmpty {
            // Within the actor the element is actor-isolated; moving it out and
            // routing it does not cross an isolation boundary.
            let request = remaining.removeFirst()
            streams.append(try await submitOne(request))
        }
        return streams
    }

    /// Build the per-request stream + handler and route it.
    private func submitOne(_ request: sending GenerationRequest) async throws
        -> AsyncStream<Generation>
    {
        guard let context else {
            throw BatchedGenerationError.schedulerBusy
        }

        let (stream, continuation) = AsyncStream.makeStream(of: Generation.self)
        let handler = SchedulerTokenHandler.text(
            continuation: continuation,
            tokenizer: context.tokenizer,
            toolCallFormat: context.configuration.toolCallFormat ?? .json
        )

        let scheduled = ScheduledRequest(
            request: request,
            handler: handler,
            inputTokens: request.input.text.tokens.asArray(Int.self),
            modelName: context.configuration.name
        )

        try await route(scheduled)
        return stream
    }

    /// Internal carrier bundling a request with its derived routing data.
    /// Not `Sendable`: holds the non-`Sendable` `GenerationRequest`. The model
    /// context is read from the actor's stored ``context`` rather than carried
    /// here, so this value can be passed to `sending` routing methods.
    struct ScheduledRequest {
        var request: GenerationRequest
        let handler: SchedulerTokenHandler
        let inputTokens: [Int]
        let modelName: String
    }

    private func route(_ scheduled: sending ScheduledRequest) async throws {
        switch state {
        case .idle:
            startSingle(scheduled)

        case .single(let single):
            await upgrade(from: single, joining: scheduled)

        case .pendingUpgrade, .upgrading:
            // An upgrade is mid-flight; admit this request to the (about-to-be
            // or already) batched engine. It joins the batch on the next step.
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
    /// (the repo's existing hand-off box; no new `@unchecked`).
    private struct SingleTaskInputs {
        let request: GenerationRequest
        let model: any LanguageModel
        let configuration: ModelConfiguration
        let tokenizer: Tokenizer
    }

    /// Start a request on the zero-overhead single-stream path. The task drives
    /// a ``TokenIterator``, streams decoded text to the handler, and checks the
    /// upgrade flag after each token so it can hand off its live state.
    private func startSingle(_ scheduled: sending ScheduledRequest) {
        guard let context else {
            scheduled.handler.finish()
            return
        }

        let requestID = nextRequestID
        nextRequestID += 1

        let upgradeFlag = UpgradeFlag()
        let handler = scheduled.handler
        let parameters = scheduled.request.parameters
        let inputTokens = scheduled.inputTokens
        let modelName = scheduled.modelName
        let salt = scheduled.request.promptCacheSalt
        let writeBack = scheduled.request.promptCache != nil
        let submitTime = Date.timeIntervalSinceReferenceDate
        let promptTokenCount = inputTokens.count

        // Box the non-`Sendable` inputs (model/tokenizer/request) so the
        // detached `@Sendable` task can consume them; the iterator is built and
        // owned inside the task.
        let boxed = SendableBox(
            SingleTaskInputs(
                request: scheduled.request,
                model: context.model,
                configuration: context.configuration,
                tokenizer: context.tokenizer
            ))

        let task = Task<Void, Never> { [weak self] in
            let inputs = boxed.consume()
            await Self.runSingle(
                request: inputs.request,
                model: inputs.model,
                configuration: inputs.configuration,
                tokenizer: inputs.tokenizer,
                handler: handler,
                upgradeFlag: upgradeFlag,
                promptTokenCount: promptTokenCount,
                submitTime: submitTime,
                onFinished: { tokenIds in
                    await self?.singleTaskFinished(
                        requestID: requestID, generatedTokenIds: tokenIds)
                }
            )
        }

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
            submitTime: submitTime
        )
        state = .single(single)
    }

    /// The single-request decode loop. Runs detached from the actor; builds and
    /// owns the `TokenIterator` so the non-`Sendable` iterator never crosses an
    /// isolation boundary. On an upgrade request it deposits a `sending`
    /// ``LiveIteratorState`` and exits without finishing the stream (the batch
    /// loop takes over). Otherwise it finishes the stream itself.
    private static func runSingle(
        request: sending GenerationRequest,
        model: any LanguageModel,
        configuration: ModelConfiguration,
        tokenizer: Tokenizer,
        handler: SchedulerTokenHandler,
        upgradeFlag: UpgradeFlag,
        promptTokenCount: Int,
        submitTime: TimeInterval,
        onFinished: @Sendable @escaping ([Int]) async -> Void
    ) async {
        let parameters = request.parameters
        let iterator: TokenIterator
        do {
            iterator = try TokenIterator(
                input: request.input, model: model, parameters: parameters)
        } catch {
            upgradeFlag.markTaskFinished()
            handler.finish()
            await onFinished([])
            return
        }

        // Same stop set the synchronous single-stream loop uses.
        let stopTokenIds = Self.stopTokenIds(
            configuration: configuration, tokenizer: tokenizer)
        let unknownTokenId = tokenizer.unknownTokenId

        var it = iterator
        var generated: [Int] = []
        var stopReason: GenerateStopReason? = nil
        var cancelled = false

        while true {
            if Task.isCancelled {
                cancelled = true
                break
            }
            // Upgrade hand-off point: deposit live state and exit, leaving the
            // stream open for the batch loop to continue. After this the task
            // never touches `it`/its caches again (region-isolation invariant).
            if upgradeFlag.upgradeRequested {
                let live = LiveIteratorState(
                    cache: it.cache,
                    currentToken: it.y.tokens.item(Int.self),
                    tokenCount: it.tokenCount,
                    maxTokens: it.maxTokens,
                    parameters: parameters,
                    generatedTokenIds: generated
                )
                upgradeFlag.depositLiveState(live)
                return
            }

            guard let token = it.next() else {
                break
            }

            // Stop-token detection mirrors `runSynchronousGenerationLoop`: the
            // stop token is NOT emitted to the consumer, generation just ends.
            if token == unknownTokenId || stopTokenIds.contains(token) {
                stopReason = .stop
                break
            }

            generated.append(token)

            if handler.processToken(token) == false {
                cancelled = true
                break
            }
        }

        // Resolve the stop reason the same way the synchronous loop does.
        if stopReason == nil {
            if let maxTokens = it.maxTokens, it.tokenCount >= maxTokens {
                stopReason = .length
            } else if cancelled {
                stopReason = .cancelled
            } else {
                stopReason = .cancelled
            }
        }

        // If an upgrade was requested in the same instant we exited, hand off
        // nil so the scheduler does not hang waiting on the continuation.
        upgradeFlag.markTaskFinished()

        handler.processEndOfSequence()
        let now = Date.timeIntervalSinceReferenceDate
        let info = GenerateCompletionInfo(
            promptTokenCount: promptTokenCount,
            generationTokenCount: generated.count,
            promptTime: it.promptPrefillTime,
            generationTime: max(0, now - submitTime),
            stopReason: cancelled ? .cancelled : (stopReason ?? .stop)
        )
        handler.yieldInfo(info)
        handler.finish()
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
        return ids
    }
}

// MARK: - Batched path

extension InferenceScheduler {

    /// Lazily create the ``EngineDriver`` (and its ``BatchInferenceEngine``)
    /// for the bound context. Reuses an existing driver. Returns `nil` if the
    /// model's cache topology cannot be batched (caller falls back to single).
    private func ensureDriver() async -> EngineDriver? {
        if let driver { return driver }
        guard let context else { return nil }

        let engine: BatchInferenceEngine
        do {
            engine = try BatchInferenceEngine(
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

        let driver = EngineDriver(engine: engine, promptCache: promptCache)
        if hasPendingMaxBatchSize {
            await driver.setMaxBatchSize(pendingMaxBatchSize)
        } else if let cap = configuration.maxBatchSize {
            await driver.setMaxBatchSize(cap)
        }
        self.driver = driver
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

    /// Admit a request directly to the batch engine and ensure the driver's
    /// decode loop is running. Used for 3rd+ concurrent requests and for the
    /// manual ``generateBatched(_:)`` path.
    private func admitToBatch(_ scheduled: sending ScheduledRequest) async {
        guard let driver = await ensureDriver() else {
            // Cannot batch this model — run the request on the single path
            // instead (correctness over batching).
            startSingleIfIdle(scheduled)
            return
        }
        state = .batched

        let req = SchedulerRequest(
            input: scheduled.request.input,
            parameters: scheduled.request.parameters,
            inputTokens: scheduled.inputTokens,
            modelName: scheduled.modelName,
            promptCache: scheduled.request.promptCache,
            promptCacheSalt: scheduled.request.promptCacheSalt
        )
        let sampler = Self.rowSampler(for: scheduled.request.parameters)
        let maxTokens = scheduled.request.parameters.maxTokens ?? defaultMaxTokens

        await driver.submit(
            req,
            handler: scheduled.handler,
            maxTokens: maxTokens,
            sampler: sampler,
            stateMachine: nil
        )
        startBatchLoop(driver)
    }

    /// If the model cannot be batched and we are idle, run on the single path;
    /// otherwise just finish the stream (a non-batchable model cannot join an
    /// existing batch).
    private func startSingleIfIdle(_ scheduled: sending ScheduledRequest) {
        if case .idle = state {
            startSingle(scheduled)
        } else {
            // Best effort: a non-batchable model with concurrent requests is
            // unusual; stream this one on its own single iterator after the
            // current one frees. For now, run it single regardless — the
            // single path does not share engine state, so two concurrent
            // single tasks are safe (each owns its own caches/model calls are
            // already-evaluated weights).
            startSingle(scheduled)
        }
    }

    /// Build a `RowSampler` from generation parameters. Note: penalties
    /// (`LogitProcessor`: repetition/presence/frequency) are NOT represented by
    /// `RowSampler` and are dropped on the batched path; only
    /// temperature/top-p/top-k carry over. See the upgrade TODO.
    static func rowSampler(for parameters: GenerateParameters) -> RowSampler {
        makeRowSampler(
            temperature: parameters.temperature,
            topP: parameters.topP,
            topK: parameters.topK,
            seed: nil
        )
    }

    /// Default per-request token budget when `maxTokens` is unset, matching the
    /// engine's own default.
    private var defaultMaxTokens: Int { 128 }

    /// Start (or no-op if already running) the driver's decode loop as a
    /// detached task. The loop ends when the engine empties; on completion the
    /// scheduler is returned to `.idle` if no work remains.
    private func startBatchLoop(_ driver: EngineDriver) {
        let ticket: WiredMemoryTicket? = nil
        Task { [weak self] in
            await driver.drain(
                wiredMemoryTicket: ticket,
                onResult: nil
            )
            await self?.batchLoopFinished()
        }
    }

    /// Called when the driver's decode loop finishes. Returns to `.idle` if no
    /// work remains so the next lone request starts fresh on the single path.
    private func batchLoopFinished() async {
        guard let driver else {
            state = .idle
            return
        }
        if await driver.hasWork {
            // More work arrived after the loop exited; restart it.
            startBatchLoop(driver)
        } else {
            state = .idle
        }
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
        // Phase 1: enter the upgrade window and request the snapshot.
        state = .upgrading

        let live: LiveIteratorState? = await withCheckedContinuation { continuation in
            single.upgradeFlag.requestUpgrade(continuation: continuation)
        }

        guard let live else {
            // The single task finished before it could deposit state. Its
            // stream is already (or about to be) closed by its own loop; start
            // the joining request fresh.
            await admitToBatch(joining)
            return
        }

        // The single task has stopped and deposited `live`; we now own it.
        guard let driver = await ensureDriver() else {
            // Cannot batch this model: deliver the joining request on a fresh
            // single path. (The original request's tokens were already
            // streamed; it cannot continue without batching — close it.)
            single.handler.finish()
            startSingleAfterUpgradeFallback(joining)
            return
        }

        guard let batchedCaches = Self.migrateCaches(live.cache) else {
            // Migration unsupported for this cache topology. Close the original
            // (its KV cannot be carried) and start the joining request fresh.
            // TODO(PR4 auto-upgrade): extend fromSingle to SSM/composite so this
            // fallback is unreachable for those models.
            single.handler.finish()
            await admitToBatch(joining)
            return
        }

        // Phase 2: build the adopted decode batch from migrated state.
        let firstUID = nextRequestID
        nextRequestID += 1

        let remainingBudget = (live.maxTokens ?? Int.max) - live.tokenCount
        guard remainingBudget > 0 else {
            // Zero remaining budget: the original is effectively done. Finish
            // it and run the joining request.
            single.handler.finish()
            await admitToBatch(joining)
            return
        }

        let sampler = Self.rowSampler(for: live.parameters)
        let adopted = await driver.makeAdoptedBatch(
            uid: firstUID,
            seedToken: live.currentToken,
            caches: batchedCaches,
            sampler: sampler,
            stateMachine: nil,
            maxTokens: remainingBudget,
            numTokens: 0,
            tokens: live.generatedTokenIds
        )

        let firstRecord = EngineDriver.AdoptedRecord(
            handler: single.handler,
            promptTokenCount: single.inputTokens.count,
            inputTokens: single.inputTokens,
            modelName: single.modelName,
            promptCacheSalt: single.promptCacheSalt,
            submitTime: single.submitTime,
            writeBackToPromptCache: single.writeBackToPromptCache
        )
        await driver.adopt(adopted, records: [firstUID: firstRecord])

        // Phase 3: admit the joining request and start the batch loop.
        state = .batched

        let req = SchedulerRequest(
            input: joining.request.input,
            parameters: joining.request.parameters,
            inputTokens: joining.inputTokens,
            modelName: joining.modelName,
            promptCache: joining.request.promptCache,
            promptCacheSalt: joining.request.promptCacheSalt
        )
        let joinSampler = Self.rowSampler(for: joining.request.parameters)
        let joinMax = joining.request.parameters.maxTokens ?? defaultMaxTokens
        await driver.submit(
            req,
            handler: joining.handler,
            maxTokens: joinMax,
            sampler: joinSampler,
            stateMachine: nil
        )

        startBatchLoop(driver)
    }

    /// Fallback when an upgrade cannot batch the model: run the joining request
    /// on a fresh single iterator after the upgrade window. Two concurrent
    /// single tasks are safe (no shared engine state).
    private func startSingleAfterUpgradeFallback(_ scheduled: sending ScheduledRequest) {
        // Reset to idle so `startSingle` installs cleanly.
        state = .idle
        startSingle(scheduled)
    }

    /// Called from the single task when it finishes naturally (not via
    /// upgrade). If the scheduler is still in `.single`/`.pendingUpgrade` for
    /// this request, return to `.idle`.
    private func singleTaskFinished(requestID: Int, generatedTokenIds: [Int]) async {
        let activeID: Int?
        switch state {
        case .single(let s): activeID = s.requestID
        case .pendingUpgrade(let s): activeID = s.requestID
        default: activeID = nil
        }
        if activeID == requestID {
            // Write the finished single request's KV state back to the prompt
            // cache, if enabled. (Single-path write-back is a future hookup;
            // the single task does not currently surface its final caches here.
            // TODO(PR4): wire single-path prompt-cache write-back.)
            state = .idle
        }
        // Otherwise the request was upgraded or superseded; nothing to do.
    }
}
