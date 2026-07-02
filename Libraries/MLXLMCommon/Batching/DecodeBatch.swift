// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Picks one token per row from a `[B, vocab]` logits tensor.
public typealias RowSampler = @Sendable (MLXArray) -> MLXArray

/// Deterministic greedy sampler.
@Sendable public func greedySampler(_ logprobs: MLXArray) -> MLXArray {
    argMax(logprobs, axis: -1)
}

/// Per-row response from a single decode step.
///
/// This is the `Sendable` value type used to cross concurrency domains
/// (e.g. PR4's `EngineDriver` actor → an `AsyncStream.Continuation`). It
/// carries only value data; the non-`Sendable` final KV caches are surfaced
/// separately via ``FinishedRowCache`` from ``DecodeBatch/next(capturingFinalCaches:)``.
public struct BatchStepResult: Sendable {
    public let uid: Int
    public let token: Int

    /// `.stop`, `.length`, or nil if the row is still generating. Stop
    /// detection happens exactly once, here in the engine.
    public let finishReason: GenerateStopReason?

    /// The matched stop sequence if a multi-token stop completed on this token.
    public let matchedSequence: [Int]?

    /// State name after this token's transition (nil = terminated).
    public let currentState: String?

    /// All produced tokens for this row. Set only on the final response.
    public let allTokens: [Int]?
}

/// Handoff for prompt-cache write-back paths (the PR4 bridge seam for
/// per-row final caches). This intentionally does **not** conform to
/// `Sendable` because `finalCache` carries mutable (non-`Sendable`) `KVCache`
/// references. PR4's driver actor evaluates and writes these back to the
/// prompt cache on its own executor; they must never cross an actor boundary
/// via this struct (transfer with `sending` or write back in place instead).
public struct FinishedRowCache {
    public let uid: Int
    public let allTokens: [Int]
    public let finalCache: [any KVCache]

    public init(uid: Int, allTokens: [Int], finalCache: [any KVCache]) {
        self.uid = uid
        self.allTokens = allTokens
        self.finalCache = finalCache
    }
}

/// Decode-phase batch over a shared `[any BatchedCache]` (one per layer).
/// Each layer's cache is the appropriate batched type for that layer:
/// `BatchKVCache` for full attention, `ArraysCache`/`MambaCache` for SSM.
/// Construct after prefill has populated the caches; call `next()` to
/// drive generation one step at a time.
///
/// This is mutable decode engine state owned by ``BatchGenerationEngine`` and
/// is intentionally **not** `Sendable`. Drive it from a single executor.
/// ``BatchStepResult`` is the `Sendable` value type for crossing concurrency
/// domains.
public final class DecodeBatch {

    public let model: any LanguageModel
    public private(set) var uids: [Int]
    public private(set) var promptCache: [any BatchedCache]
    public private(set) var tokens: [[Int]]
    public private(set) var maxTokens: [Int]

    public private(set) var samplers: [RowSampler?]
    public let fallbackSampler: RowSampler

    /// Whether `fallbackSampler` is the deterministic ``greedySampler``. The
    /// greedy `argMax` fast path in `step()` is gated on this: when a row has
    /// no explicit sampler it falls back to `fallbackSampler`, so the fast
    /// path may only be taken when that fallback is itself greedy (otherwise
    /// rows that intended to sample would silently get `argMax`).
    ///
    /// `extend(_:)` conservatively ANDs the two batches' flags so a merged
    /// batch never takes the fast path if either side's fallback may sample.
    public private(set) var fallbackIsGreedy: Bool

    /// Per-row ``LogitProcessor`` (repetition / presence / frequency
    /// penalties), mirroring the single path's `GenerateParameters.processor()`
    /// applied per `TokenIterator`. `nil` for a row with no penalties.
    ///
    /// These are **built** (non-`Sendable`, stateful) processors. `DecodeBatch`
    /// is itself not `Sendable` and is driven from a single executor, so it may
    /// hold them directly; the engine builds them on its executor from the
    /// `Sendable` ``RowProcessorSource`` before constructing the batch.
    private var processors: [LogitProcessor?]
    public private(set) var stateMachines: [StopSequenceMatcher]

    /// Tokens queued for the next model call. At construction this is the
    /// final prompt token for each row. After priming, and after every
    /// decode step, it holds the sampled token that should be returned on
    /// the next `next()` call. `[B]`.
    private var nextTokens: MLXArray
    private var numTokens: [Int]
    private var matcherStates: [StopSequenceMatcherState]

    /// - Parameters:
    ///   - fallbackIsGreedy: Whether `fallbackSampler` is ``greedySampler``.
    ///     Gates the greedy `argMax` fast path; pass `false` whenever the
    ///     fallback may sample so rows without an explicit sampler are not
    ///     silently forced to `argMax`.
    ///   - processors: Per-row ``LogitProcessor`` (penalties), or `nil` to
    ///     run all rows penalty-free. Built by the engine on its executor
    ///     from the `Sendable` ``RowProcessorSource``; a row that has a
    ///     processor is routed through the sampler path (never the greedy
    ///     `argMax` fast path, which penalties would invalidate).
    ///   - numTokens: Per-row count of tokens already produced before this
    ///     batch took over. Defaults to all-zeros (a fresh decode). Used by
    ///     the single→batch upgrade so a migrated request's `maxTokens`
    ///     budget reflects the tokens it already emitted while single.
    public init(
        model: any LanguageModel,
        uids: [Int],
        seedTokens: MLXArray,
        promptCache: [any BatchedCache],
        tokens: [[Int]],
        maxTokens: [Int],
        samplers: [RowSampler?]? = nil,
        fallbackSampler: @escaping RowSampler = greedySampler,
        fallbackIsGreedy: Bool = true,
        processors: [LogitProcessor?]? = nil,
        stateMachines: [StopSequenceMatcher]? = nil,
        numTokens: [Int]? = nil
    ) {
        precondition(uids.count == tokens.count, "uids/tokens count mismatch")
        precondition(uids.count == maxTokens.count, "uids/max_tokens count mismatch")
        if let numTokens {
            precondition(uids.count == numTokens.count, "uids/numTokens count mismatch")
        }
        if let samplers {
            precondition(uids.count == samplers.count, "uids/samplers count mismatch")
        }
        if let processors {
            precondition(uids.count == processors.count, "uids/processors count mismatch")
        }
        if let stateMachines {
            precondition(uids.count == stateMachines.count, "uids/stateMachines count mismatch")
        }
        self.model = model
        self.uids = uids
        self.promptCache = promptCache
        self.tokens = tokens
        self.maxTokens = maxTokens
        self.samplers = samplers ?? Array(repeating: nil, count: uids.count)
        self.fallbackSampler = fallbackSampler
        self.fallbackIsGreedy = fallbackIsGreedy
        self.processors = processors ?? Array(repeating: nil, count: uids.count)
        let machines = stateMachines ?? Array(repeating: StopSequenceMatcher(), count: uids.count)
        self.stateMachines = machines
        self.matcherStates = machines.map { $0.makeState() }
        self.numTokens = numTokens ?? Array(repeating: 0, count: uids.count)
        self.nextTokens = seedTokens

        // Seed each row's penalty processor with its prompt prefix, mirroring
        // the single path's `processor?.prompt(input.text.tokens)` before any
        // step. `tokens[i]` here is the prompt history *excluding* the seed
        // (the seed is the row's final prompt token in `seedTokens`); the seed
        // is fed to the processor as the first `didSample` inside the priming
        // `step()` below, so by the time `process()` runs to predict the first
        // generated token the ring holds the full prompt -- exactly the single
        // path's invariant.
        for i in 0 ..< self.processors.count
        where self.processors[i] != nil && !self.tokens[i].isEmpty {
            self.processors[i]?.prompt(MLXArray(self.tokens[i].map { Int32($0) }))
        }

        // Match upstream mlx_lm.GenerationBatch: immediately run one
        // decode step in the constructor so the first call to `next()`
        // returns an already-computed token while scheduling the following
        // token. This double-buffer keeps the GPU queue ahead of the CPU
        // token extraction path.
        if !uids.isEmpty {
            _ = step()
        }
    }

    /// Run one decode step. Finished rows (length / stop) are filtered out
    /// of the active set after this call; their final responses appear with
    /// non-nil `finishReason`.
    public func next() -> [BatchStepResult] {
        next(capturingFinalCaches: false).responses
    }

    /// Internal variant for scheduler-level prompt-cache write-back. When
    /// enabled, finished rows are extracted before the active batch is
    /// filtered so row indices still refer to the pre-filtered cache.
    func next(capturingFinalCaches: Bool) -> (
        responses: [BatchStepResult], finishedCaches: [FinishedRowCache]
    ) {
        if uids.isEmpty { return ([], []) }

        let stepTokens = step()

        var keep: [Int] = []
        var responses: [BatchStepResult] = []
        var finishedCaches: [FinishedRowCache] = []
        responses.reserveCapacity(uids.count)

        for i in 0 ..< uids.count {
            numTokens[i] += 1

            var finishReason: GenerateStopReason? = nil
            if numTokens[i] >= maxTokens[i] {
                finishReason = .length
            }

            let machine = stateMachines[i]
            let (nextState, matchedSequence, currentState) =
                machine.match(matcherStates[i], stepTokens[i])
            matcherStates[i] = nextState
            if matchedSequence != nil, currentState == nil {
                finishReason = .stop
            }

            if finishReason != nil {
                let allTokens = tokens[i]
                if capturingFinalCaches {
                    finishedCaches.append(
                        FinishedRowCache(
                            uid: uids[i],
                            allTokens: allTokens,
                            finalCache: promptCache.map { $0.extractBatched(i) }
                        ))
                }
                responses.append(
                    BatchStepResult(
                        uid: uids[i],
                        token: stepTokens[i],
                        finishReason: finishReason,
                        matchedSequence: matchedSequence,
                        currentState: currentState,
                        allTokens: allTokens
                    ))
            } else {
                keep.append(i)
                responses.append(
                    BatchStepResult(
                        uid: uids[i],
                        token: stepTokens[i],
                        finishReason: nil,
                        matchedSequence: matchedSequence,
                        currentState: currentState,
                        allTokens: nil
                    ))
            }
        }

        if keep.count < uids.count {
            filter(keep: keep)
        }

        return (responses, finishedCaches)
    }

    /// In-place keep only the rows at the given indices.
    public func filter(keep: [Int]) {
        let keepArr = MLXArray(keep.map { Int32($0) })

        if keep.isEmpty {
            promptCache.removeAll()
        } else {
            for cache in promptCache {
                cache.filterBatched(batchIndices: keepArr)
            }
        }

        uids = keep.map { uids[$0] }
        tokens = keep.map { tokens[$0] }
        samplers = keep.map { samplers[$0] }
        processors = keep.map { processors[$0] }
        maxTokens = keep.map { maxTokens[$0] }
        stateMachines = keep.map { stateMachines[$0] }
        matcherStates = keep.map { matcherStates[$0] }
        numTokens = keep.map { numTokens[$0] }
        // Unconditional so an emptied batch does not retain stale rows in
        // `nextTokens` while every sibling array is empty (`take` with zero
        // indices yields the matching zero-row array).
        nextTokens = take(nextTokens, keepArr, axis: 0)
    }

    /// In-place: append `other`'s rows to this batch. Per-layer caches are
    /// concatenated via `BatchedCache.extendBatched`.
    public func extend(_ other: DecodeBatch) {
        precondition(
            promptCache.count == other.promptCache.count,
            "Cannot extend with a batch that has a different layer count"
        )
        for (a, b) in zip(promptCache, other.promptCache) {
            a.extendBatched(b)
        }
        uids.append(contentsOf: other.uids)
        tokens.append(contentsOf: other.tokens)
        if other.fallbackIsGreedy != fallbackIsGreedy {
            // The merged batch keeps `self`'s fallback, so materialize
            // `other`'s (different) fallback into its nil sampler slots --
            // otherwise those rows would silently switch sampling behavior.
            samplers.append(contentsOf: other.samplers.map { $0 ?? other.fallbackSampler })
        } else {
            samplers.append(contentsOf: other.samplers)
        }
        processors.append(contentsOf: other.processors)
        maxTokens.append(contentsOf: other.maxTokens)
        stateMachines.append(contentsOf: other.stateMachines)
        matcherStates.append(contentsOf: other.matcherStates)
        numTokens.append(contentsOf: other.numTokens)
        // Conservatively disable the greedy fast path if either side's
        // fallback may sample, so merged rows that relied on a sampling
        // fallback are not forced to `argMax`.
        fallbackIsGreedy = fallbackIsGreedy && other.fallbackIsGreedy
        nextTokens = concatenated([nextTokens, other.nextTokens], axis: 0)
    }

    public var isEmpty: Bool { uids.isEmpty }
    public var batchSize: Int { uids.count }

    /// One forward pass + per-row sample, double-buffered like upstream
    /// `mlx_lm.generate.GenerationBatch._step`.
    ///
    /// `nextTokens` is treated as the *current* token batch to return from
    /// this call. We immediately feed it back through the model, sample the
    /// following token batch, and `asyncEval` that future batch before
    /// synchronously materializing the current tokens for CPU-side stop
    /// detection / response dispatch.
    private func step() -> [Int] {
        let currentTokens = nextTokens
        let inputs = currentTokens[0..., .newAxis]

        let logits = model.callAsFunction(inputs, cache: promptCache.map { $0 as any KVCache })

        // [B, 1, vocab] -> [B, vocab]
        let stepLogits = logits[.ellipsis, -1, 0...]

        // The greedy `argMax` fast path is valid only when *every* row resolves
        // to plain greedy: no explicit per-row sampler, no per-row penalty
        // processor (penalties change the arg-max), and a known-greedy fallback
        // (a non-greedy `fallbackSampler` would otherwise be silently dropped
        // for rows without an explicit sampler).
        let anySampler = samplers.contains { $0 != nil }
        let anyProcessor = processors.contains { $0 != nil }
        let useSamplerPath = anySampler || anyProcessor || !fallbackIsGreedy

        let sampledTokens: MLXArray
        if useSamplerPath {
            // Promote bfloat16 logits to float32 before building logprobs, so
            // the batched sampler path matches the single-request `TopPSampler`
            // (Evaluate.swift), which casts bfloat16 -> float32 before
            // `logSoftmax`. Computing logprobs directly from bfloat16 would
            // shift the top-P/top-K thresholds and the categorical draw, so a
            // request could sample differently batched vs. single. The greedy
            // `argMax` fast path below is order-preserving and needs no cast.
            let sampleLogits =
                stepLogits.dtype == .bfloat16 ? stepLogits.asType(.float32) : stepLogits
            // Shared logprobs for penalty-free rows. Rows with a processor
            // compute their own from the *processed* logits below; MLX is
            // lazy, so this node costs nothing if no row slices it.
            let logprobs = sampleLogits - logSumExp(sampleLogits, axis: -1, keepDims: true)
            var samples: [MLXArray] = []
            samples.reserveCapacity(uids.count)
            for i in 0 ..< uids.count {
                // Drive this row's penalty processor incrementally, exactly as
                // the single `TokenIterator` does: feed it the current input
                // token (`currentTokens[i]`, which was the previous step's
                // sampled token), then `process()` the row's logit slice before
                // the sampler runs. With the prompt prefix loaded in the
                // constructor, the ring holds the full history through the
                // current token at `process()` time -- matching the single
                // path's `process -> sample -> didSample` invariant. These are
                // GPU-only mask writes (see `TokenRing`), so they don't force a
                // CPU sync and preserve `asyncEval` pipelining.
                let rowLogprobs: MLXArray
                if processors[i] != nil {
                    processors[i]?.didSample(token: currentTokens[i ..< (i + 1)])
                    // Penalties must see RAW logits: the single path applies
                    // `process()` before `logSoftmax`, and `RepetitionContext`
                    // branches on the logit sign (divide positive, multiply
                    // negative) -- on logprobs (all <= 0) every repeated token
                    // would be scaled the wrong way. Normalize per row after
                    // processing so the sampler still receives logprobs.
                    var rowLogits = sampleLogits[i ..< (i + 1), 0...]
                    rowLogits = processors[i]?.process(logits: rowLogits) ?? rowLogits
                    rowLogprobs = rowLogits - logSumExp(rowLogits, axis: -1, keepDims: true)
                } else {
                    rowLogprobs = logprobs[i ..< (i + 1), 0...]
                }
                let sampler = samplers[i] ?? fallbackSampler
                samples.append(sampler(rowLogprobs))
            }
            sampledTokens = concatenated(samples, axis: 0)
        } else {
            // Greedy fast path. Avoid the full-vocabulary logSumExp when
            // all rows are greedy: argMax(logits) == argMax(logprobs),
            // and Swift does not currently expose logprobs downstream.
            // This removes one expensive reduction kernel per decode step.
            sampledTokens = argMax(stepLogits, axis: -1)
        }

        // Start computing the next token before forcing the current token
        // values back to the CPU. This overlaps GPU work with the CPU
        // extraction / response-building path.
        nextTokens = sampledTokens
        asyncEval(sampledTokens)

        eval(currentTokens)
        // `currentTokens` may be `uint32` (greedy `argMax`) or `int32`
        // (`MLXRandom.categorical` on the sampler path), so normalize to
        // `int32` before extracting to avoid a dtype-mismatch crash.
        let stepTokens = currentTokens.asType(.int32).asArray(Int32.self).map { Int($0) }

        for (i, t) in stepTokens.enumerated() {
            tokens[i].append(t)
        }

        return stepTokens
    }
}
