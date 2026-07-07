// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Validation errors raised by ``BatchGenerationEngine`` for malformed
/// configuration or requests. Unsupported cache topologies surface as
/// ``BatchedCacheError`` from the PR1 cache factory.
public enum BatchGenerationEngineError: Error, CustomStringConvertible, Equatable {
    case invalidConfiguration(field: String, value: Int)
    case emptyPrompt(rowIndex: Int)
    case requestArrayLengthMismatch(field: String, expected: Int, got: Int)
    case nonPositiveMaxTokens(rowIndex: Int, value: Int)

    public var description: String {
        switch self {
        case .invalidConfiguration(let field, let value):
            return "\(field) must be greater than zero; got \(value)"
        case .emptyPrompt(let rowIndex):
            return "prompts[\(rowIndex)] must not be empty"
        case .requestArrayLengthMismatch(let field, let expected, let got):
            return "\(field).count must equal prompts.count; expected \(expected), got \(got)"
        case .nonPositiveMaxTokens(let rowIndex, let value):
            return "maxTokens[\(rowIndex)] must be greater than zero; got \(value)"
        }
    }
}

/// Continuous-batching engine (the low-level, opt-in batched inference API).
///
///   1. `insert(prompts:)` queues new requests and returns their UIDs.
///   2. `next()` runs one engine step: drains the queue into prefill,
///      runs one decode step, and emits per-row responses. Finished rows
///      are filtered out and their slots become available for new
///      admissions on the next call.
///   3. `close()` releases resources.
///
/// `BatchGenerationEngine` is stateful and intentionally **not** `Sendable`.
/// Drive each instance from one execution context at a time. For concurrent
/// request admission, cancellation, or streaming, wrap the engine in an actor
/// or scheduler that serializes access to `insert`, `cancel`, `next`,
/// `adoptActiveBatch`, and `close` (this is what PR4's `EngineDriver` does).
public final class BatchGenerationEngine {

    public let model: any LanguageModel
    public let prefillStepSize: Int
    public let prefillBatchSize: Int
    public let completionBatchSize: Int
    public let defaultMaxTokens: Int

    public let defaultEosTokens: [[Int]]
    public let defaultSampler: RowSampler

    /// Whether ``defaultSampler`` is the deterministic ``greedySampler``.
    /// Forwarded as `fallbackIsGreedy` to every batch the engine builds so
    /// the greedy `argMax` fast path is only taken when the fallback truly is
    /// greedy. (The engine currently always uses ``greedySampler`` here.)
    private let defaultSamplerIsGreedy: Bool
    public let defaultStateMachine: StopSequenceMatcher

    private var uidCounter: Int = 0
    private var unprocessed: [QueuedRequest] = []
    private var generationBatch: DecodeBatch?

    /// Per-layer batched-cache allocators, validated once at construction so
    /// per-admission allocation is a cheap closure call that cannot fail.
    /// Owned by the PR1 cache layer (``makeBatchedCacheFactories(for:)``);
    /// the engine no longer carries its own factory.
    private let cacheFactories: [BatchedCacheFactory]

    public private(set) var promptTokensProcessed: Int = 0
    public private(set) var generatedTokens: Int = 0

    public init(
        model: any LanguageModel,
        eosTokens: [[Int]] = [],
        defaultMaxTokens: Int = 128,
        prefillStepSize: Int = 2048,
        prefillBatchSize: Int = 8,
        completionBatchSize: Int = 32,
        cacheParameters: GenerateParameters? = nil
    ) throws {
        try Self.validateConfiguration(
            defaultMaxTokens: defaultMaxTokens,
            prefillStepSize: prefillStepSize,
            prefillBatchSize: prefillBatchSize,
            completionBatchSize: completionBatchSize
        )

        self.model = model
        self.prefillStepSize = prefillStepSize
        self.prefillBatchSize = prefillBatchSize
        // Honor the caller's decode cap exactly: admission already bounds each
        // prefill sub-batch by the remaining completion capacity, so a
        // completionBatchSize below prefillBatchSize simply admits smaller
        // sub-batches (the old silent max() raised the documented decode
        // maximum instead).
        self.completionBatchSize = completionBatchSize
        self.defaultMaxTokens = defaultMaxTokens
        self.defaultEosTokens = eosTokens
        self.defaultSampler = greedySampler
        self.defaultSamplerIsGreedy = true
        // The engine never rewrites caches mid-decode, so runtime KV
        // quantization parameters cannot be honored here (the single path
        // quantizes via maybeQuantizeKVCache after each step). Models whose
        // newCache builds QuantizedKVCache directly are fine -- the factory
        // routes them -- but for default-newCache models these fields would
        // be silently dropped. Reject loudly, mirroring the scheduler's
        // parametersAreBatchable gate.
        let probe = model.newCache(parameters: cacheParameters)
        if cacheParameters?.kvBits != nil || cacheParameters?.kvScheme != nil {
            guard probe.contains(where: { $0 is QuantizedKVCache }) else {
                throw BatchGenerationEngineError.invalidConfiguration(
                    field: "cacheParameters.kvBits/kvScheme", value: cacheParameters?.kvBits ?? -1)
            }
        }
        // A model with no per-layer caches has nothing to batch over, and the
        // batched forward passes would hand it a non-nil empty cache array
        // (which models index into positionally, e.g. DeepseekV3's
        // `cache?[i]`). The single path passes nil for empty caches; here it
        // is a topology we cannot serve.
        guard !probe.isEmpty else {
            throw BatchGenerationEngineError.invalidConfiguration(
                field: "model.newCache (no per-layer caches)", value: 0)
        }
        self.cacheFactories = try makeBatchedCacheFactories(for: probe)

        if eosTokens.isEmpty {
            self.defaultStateMachine = StopSequenceMatcher()
        } else {
            self.defaultStateMachine = StopSequenceMatcher(
                states: ["normal": eosTokens.map { (sequence: $0, next: nil) }],
                initial: "normal"
            )
        }
    }

    /// Append a batch of prompts. Returns the assigned UIDs in input order.
    ///
    /// Throws ``BatchGenerationEngineError`` for empty prompt rows, mismatched
    /// per-row option counts, or nonpositive max-token limits.
    ///
    /// - Parameters:
    ///   - samplers: Per-row ``RowSampler``. Build with `makeRowSampler`,
    ///     which already encodes temperature / top-p / **min-p** / top-k, so
    ///     min-p is carried here (there is no separate `minP` engine
    ///     parameter).
    ///   - processorSources: Per-row ``RowProcessorSource`` carrying the
    ///     repetition / presence / frequency penalties. `RowProcessorSource`
    ///     is `Sendable`; the engine invokes it on its own executor to build
    ///     the stateful (non-`Sendable`) ``LogitProcessor`` for the row.
    ///     Build with `makeRowProcessorSource(parameters)`.
    @discardableResult
    public func insert(
        prompts: [[Int]],
        maxTokens: [Int]? = nil,
        samplers: [RowSampler?]? = nil,
        processorSources: [RowProcessorSource?]? = nil,
        stateMachines: [StopSequenceMatcher]? = nil
    ) throws -> [Int] {
        try Self.validateRequest(
            prompts: prompts,
            maxTokens: maxTokens,
            samplers: samplers,
            processorSources: processorSources,
            stateMachines: stateMachines
        )

        var assignedUids: [Int] = []
        assignedUids.reserveCapacity(prompts.count)

        for i in 0 ..< prompts.count {
            let uid = uidCounter
            uidCounter += 1
            assignedUids.append(uid)
            unprocessed.append(
                QueuedRequest(
                    uid: uid,
                    tokens: prompts[i],
                    maxTokens: maxTokens?[i] ?? defaultMaxTokens,
                    sampler: samplers?[i] ?? nil,
                    processorSource: processorSources?[i] ?? nil,
                    stateMachine: stateMachines?[i] ?? defaultStateMachine
                ))
        }
        return assignedUids
    }

    private static func validateConfiguration(
        defaultMaxTokens: Int,
        prefillStepSize: Int,
        prefillBatchSize: Int,
        completionBatchSize: Int
    ) throws {
        for (field, value) in [
            ("defaultMaxTokens", defaultMaxTokens),
            ("prefillStepSize", prefillStepSize),
            ("prefillBatchSize", prefillBatchSize),
            ("completionBatchSize", completionBatchSize),
        ] where value <= 0 {
            throw BatchGenerationEngineError.invalidConfiguration(field: field, value: value)
        }
    }

    private static func validateRequest(
        prompts: [[Int]],
        maxTokens: [Int]?,
        samplers: [RowSampler?]?,
        processorSources: [RowProcessorSource?]?,
        stateMachines: [StopSequenceMatcher]?
    ) throws {
        if let rowIndex = prompts.firstIndex(where: { $0.isEmpty }) {
            throw BatchGenerationEngineError.emptyPrompt(rowIndex: rowIndex)
        }

        try validateRequestArrayLength("maxTokens", maxTokens?.count, prompts.count)
        try validateRequestArrayLength("samplers", samplers?.count, prompts.count)
        try validateRequestArrayLength(
            "processorSources", processorSources?.count, prompts.count)
        try validateRequestArrayLength("stateMachines", stateMachines?.count, prompts.count)

        if let maxTokens {
            for (rowIndex, value) in maxTokens.enumerated() where value <= 0 {
                throw BatchGenerationEngineError.nonPositiveMaxTokens(
                    rowIndex: rowIndex,
                    value: value
                )
            }
        }
    }

    private static func validateRequestArrayLength(
        _ field: String,
        _ count: Int?,
        _ expected: Int
    ) throws {
        if let count, count != expected {
            throw BatchGenerationEngineError.requestArrayLengthMismatch(
                field: field,
                expected: expected,
                got: count
            )
        }
    }

    /// Run one engine step. Returns per-uid responses for the active rows;
    /// rows that finished on this step have a non-nil `finishReason`.
    ///
    /// Looped to drain the queue:
    /// `while engine.hasWork { for r in engine.next() { ... } }`
    public func next() -> [BatchStepResult] {
        next(capturingFinalCaches: false).responses
    }

    /// Run one engine step, optionally capturing the final per-row KV caches
    /// of rows that finished on this step. The returned ``FinishedRowCache``
    /// values are **not** `Sendable` (they hold mutable caches); PR4's driver
    /// actor evaluates and writes them back to the prompt cache on its own
    /// executor. Callers that do not need write-back use `next()`.
    public func next(capturingFinalCaches: Bool) -> (
        responses: [BatchStepResult], finishedCaches: [FinishedRowCache]
    ) {
        admitFromQueue()

        if let gen = generationBatch, !gen.isEmpty {
            let (responses, finishedCaches) = gen.next(capturingFinalCaches: capturingFinalCaches)
            generatedTokens += responses.count
            if gen.isEmpty {
                generationBatch = nil
            }
            return (responses, finishedCaches)
        }

        if unprocessed.isEmpty {
            return ([], [])
        }

        admitFromQueue(forceTransition: true)
        return next(capturingFinalCaches: capturingFinalCaches)
    }

    public var hasWork: Bool {
        !unprocessed.isEmpty
            || (generationBatch?.isEmpty == false)
    }

    public var queuedCount: Int { unprocessed.count }
    public var activeCount: Int { generationBatch?.batchSize ?? 0 }

    /// Remove a queued or active request from the engine.
    @discardableResult
    public func cancel(uid: Int) -> Bool {
        if let queuedIndex = unprocessed.firstIndex(where: { $0.uid == uid }) {
            unprocessed.remove(at: queuedIndex)
            return true
        }

        if let active = generationBatch, let row = active.uids.firstIndex(of: uid) {
            let keep = active.uids.indices.filter { $0 != row }
            active.filter(keep: keep)
            if active.isEmpty {
                generationBatch = nil
            }
            return true
        }

        return false
    }

    public func close() {
        unprocessed.removeAll()
        generationBatch = nil
    }

    // MARK: - PR4 bridge seam (single → batch upgrade)

    /// Splice a pre-filled, already-decoding ``DecodeBatch`` in as part of the
    /// running set. This mirrors upstream mlx-lm's `setActiveBatch` idea,
    /// used by PR4's single→batch upgrade to migrate an in-flight request
    /// (with its already-populated KV cache) into the batch **without
    /// re-prefilling** the prompt.
    ///
    /// If a batch is already running, `batch`'s rows are appended via
    /// `DecodeBatch.extend`; otherwise `batch` becomes the running set.
    ///
    /// The engine touches the spliced batch only from the executor that calls
    /// this method, preserving the single-executor invariant.
    public func adoptActiveBatch(_ batch: DecodeBatch) {
        guard !batch.isEmpty else { return }
        if let existing = generationBatch {
            existing.extend(batch)
        } else {
            generationBatch = batch
        }
    }

    /// Build a ``DecodeBatch`` from already-migrated decode state (caches that
    /// have been converted via PR1's `BatchKVCache/BatchRotatingKVCache.fromSingle`,
    /// the current seed token(s), and per-row bookkeeping) so it can be handed
    /// to ``adoptActiveBatch(_:)``.
    ///
    /// `maxTokens`/`numTokens` carry the migrated request's original limit and
    /// the count it already produced while running single, so its remaining
    /// budget is honored. `tokens` is the full per-row token history (including
    /// the current/seed token) so the final response reports the complete
    /// sequence.
    ///
    /// Constructing the batch runs one priming decode step (the double-buffer)
    /// just like a freshly-prefilled batch. That priming `step()` treats
    /// `seedTokens` as the current token and appends it to the batch's
    /// `tokens`. To avoid double-counting the seed, we hand `DecodeBatch` the
    /// history with the already-counted current token dropped from each row;
    /// after priming, the batch's `tokens` (and thus `allTokens` / any
    /// prompt-cache write-back keyed by it) equals the true history with no
    /// duplicate. This mirrors `PrefillBatch.generate`, which seeds the last
    /// prompt token and passes only the preceding prefix to `DecodeBatch`.
    ///
    /// The adopted rows' engine UIDs are allocated here, from the engine's own
    /// `uidCounter` — the single source of truth for row identity — and
    /// returned alongside the batch, so a later `insert` can never reuse one
    /// and collide with an adopted row's bookkeeping in the caller.
    ///
    /// - Parameters:
    ///   - samplers: Per-row ``RowSampler`` (includes min-p via
    ///     `makeRowSampler`), or `nil` for the greedy fallback.
    ///   - processorSources: Per-row ``RowProcessorSource`` carrying the
    ///     penalties; built into the row's ``LogitProcessor`` on this (engine)
    ///     executor. The migrated row's full `tokens` history seeds the
    ///     processor's penalty context so penalties continue seamlessly across
    ///     the single→batch upgrade.
    public func makeAdoptedBatch(
        seedTokens: MLXArray,
        caches: [any BatchedCache],
        samplers: [RowSampler?]? = nil,
        processorSources: [RowProcessorSource?]? = nil,
        stateMachines: [StopSequenceMatcher]? = nil,
        maxTokens: [Int],
        numTokens: [Int]? = nil,
        tokens: [[Int]]
    ) -> (batch: DecodeBatch, uids: [Int]) {
        precondition(
            !tokens.contains(where: { $0.isEmpty }),
            "makeAdoptedBatch requires non-empty token histories; each row's "
                + "last element is the current/seed token re-appended by priming"
        )
        if let processorSources {
            precondition(
                processorSources.count == tokens.count,
                "makeAdoptedBatch: processorSources/tokens count mismatch")
        }
        var uids: [Int] = []
        uids.reserveCapacity(tokens.count)
        for _ in 0 ..< tokens.count {
            uids.append(uidCounter)
            uidCounter += 1
        }
        let priorTokens = tokens.map { Array($0.dropLast()) }
        let batch = DecodeBatch(
            model: model,
            uids: uids,
            seedTokens: seedTokens,
            promptCache: caches,
            tokens: priorTokens,
            maxTokens: maxTokens,
            samplers: samplers,
            fallbackSampler: defaultSampler,
            fallbackIsGreedy: defaultSamplerIsGreedy,
            processors: processorSources.map { sources in
                sources.map { $0?() }
            },
            // Default to the engine's EOS matcher, matching rows admitted
            // through `insert` -- a nil parameter would otherwise install a
            // never-matching matcher and adopted rows would ignore EOS.
            stateMachines: stateMachines
                ?? Array(repeating: defaultStateMachine, count: tokens.count),
            numTokens: numTokens
        )
        return (batch, uids)
    }

    // MARK: - Admission

    /// Admit up to `min(prefillBatchSize, free completion slots)` queued
    /// requests, prefill them as a sub-batch, and merge them into the
    /// running ``DecodeBatch``.
    private func admitFromQueue(forceTransition: Bool = false) {
        let activeRunning = generationBatch?.batchSize ?? 0
        var capacity = max(0, completionBatchSize - activeRunning)

        if generationBatch == nil {
            capacity = max(capacity, 1)
        }
        if capacity == 0 || unprocessed.isEmpty { return }

        let admitCount = min(capacity, prefillBatchSize, unprocessed.count)
        let batchSlice = Array(unprocessed.prefix(admitCount))
        unprocessed.removeFirst(admitCount)

        let promptCache = makeBatchedCache(batchSize: batchSlice.count)
        // Build each row's penalty processor on this (engine) executor from its
        // `Sendable` source. The stateful `LogitProcessor` is held only inside
        // the actor-confined prefill/decode batches and never crosses an actor
        // boundary -- mirroring how `RowSampler` closures are constructed.
        let prompt = PrefillBatch(
            model: model,
            uids: batchSlice.map { $0.uid },
            promptCache: promptCache,
            tokens: Array(repeating: [], count: batchSlice.count),
            maxTokens: batchSlice.map { $0.maxTokens },
            prefillStepSize: prefillStepSize,
            samplers: batchSlice.map { $0.sampler },
            fallbackSampler: defaultSampler,
            fallbackIsGreedy: defaultSamplerIsGreedy,
            processors: batchSlice.map { $0.processorSource?() },
            stateMachines: batchSlice.map { $0.stateMachine }
        )

        let admittedGen = prompt.generate(
            lastTokensOf: batchSlice.map { $0.tokens }
        )
        promptTokensProcessed += batchSlice.reduce(into: 0) { $0 += $1.tokens.count }

        if let existing = generationBatch {
            existing.extend(admittedGen)
        } else if !admittedGen.isEmpty || forceTransition {
            generationBatch = admittedGen
        }
    }

    /// Allocate one batched cache per layer using the topology validated at
    /// init time by the PR1 factory.
    private func makeBatchedCache(batchSize B: Int) -> [any BatchedCache] {
        let zeroLeftPadding = Array(repeating: 0, count: B)
        return cacheFactories.map { $0(zeroLeftPadding) }
    }

    private struct QueuedRequest {
        let uid: Int
        let tokens: [Int]
        let maxTokens: Int
        let sampler: RowSampler?
        /// `Sendable` factory for this row's penalty processor; the processor
        /// itself is built on the engine executor at admission time.
        let processorSource: RowProcessorSource?
        let stateMachine: StopSequenceMatcher
    }
}
