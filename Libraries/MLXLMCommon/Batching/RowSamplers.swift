// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import os

// Per-row samplers for the batched inference engine. Mirrors the
// single-request `TopPSampler` in `Evaluate.swift` (which itself follows
// `mlx_lm.sample_utils`): top-P (nucleus) truncation, top-K truncation, and
// optional seeded categorical sampling, with temperature applied at the
// categorical draw. Each constructed sampler is `@Sendable` so it can be
// stored alongside an admitted batch row and crossed into the engine's
// executor.
//
// `MLXRandom.categorical`/`key`/`split` resolve transitively through the
// umbrella `import MLX` (as `Evaluate.swift` does with `categorical` /
// `withRandomState`), so no explicit `import MLXRandom` is required.

/// Builds a fresh per-row ``LogitProcessor`` (repetition / presence /
/// frequency penalties) for the batched inference engine.
///
/// `LogitProcessor` is **not** `Sendable` and carries mutable per-row state
/// (the penalty ring buffers), so a *built* processor must never cross an
/// actor boundary. Instead the engine's public APIs accept this `@Sendable`
/// factory: the scheduler/caller (potentially on another actor) constructs it
/// from the `Sendable` ``GenerateParameters``, and the engine invokes it on
/// its own executor to materialize the per-row processor just before building
/// the batch -- exactly how `makeRowSampler` captures `Sendable` config and
/// returns a `@Sendable` closure.
///
/// Returns `nil` when a row has no penalties configured.
public typealias RowProcessorSource = @Sendable () -> LogitProcessor?

/// Build a ``RowProcessorSource`` from ``GenerateParameters``.
///
/// The returned factory captures only the `Sendable` parameters and defers
/// `parameters.processor()` (which allocates the penalty ring buffers) until
/// the engine calls it on its executor. Returns a source that yields `nil`
/// when no repetition/presence/frequency penalty is configured, mirroring
/// `GenerateParameters.processor()`.
public func makeRowProcessorSource(_ parameters: GenerateParameters) -> RowProcessorSource {
    { parameters.processor() }
}

/// Build a `RowSampler` from OpenAI-style request parameters.
///
/// Behavior, matching the single-request `TopPSampler` in `Evaluate.swift`
/// (which mirrors `mlx_lm.sample_utils.make_sampler`):
///   - `temperature == 0`: return `greedySampler` (no RNG, deterministic).
///     ONLY exact zero is greedy — `GenerateParameters.sampler()` treats a
///     negative temperature as a real divisor (sampling the inverted
///     distribution), so the batched path must too.
///   - otherwise: optionally mask the unscaled logprobs to top-P
///     then min-P then top-K, scale by `1/temperature`, and sample
///     categorically. The order (filter first, temperature at the draw)
///     means one batched row reproduces single-stream sampling for the
///     same params/seed.
///
/// `seed` makes the resulting stream deterministic. Each call to the
/// returned sampler advances the per-sampler PRNG key, so successive
/// decode steps produce different draws even with the same input
/// distribution. Two requests sharing the same seed but called in
/// different orders/contexts will produce **different** sequences --
/// this matches `mlx_lm` behavior and OpenAI's documented "best-effort"
/// determinism.
///
/// - Parameters:
///   - temperature: 0 = greedy, otherwise the logits divisor.
///   - topP: nucleus probability mass (1.0 = disabled).
///   - topK: number of top tokens to keep (0 = disabled).
///   - minP: min-P threshold relative to the most likely token (0 = disabled).
///   - seed: optional RNG seed; nil uses MLX's global PRNG.
///   - advancedBy: number of draws the request already consumed elsewhere
///     (the single→batch upgrade path); the seeded key chain is
///     fast-forwarded past them so the next draw CONTINUES the sequence
///     instead of bit-identically replaying it. 0 (a fresh request) is a
///     no-op.
public func makeRowSampler(
    temperature: Float = 0.0,
    topP: Float = 1.0,
    topK: Int = 0,
    minP: Float = 0.0,
    seed: UInt64? = nil,
    advancedBy priorDraws: Int = 0
) -> RowSampler {
    if temperature == 0 {
        return greedySampler
    }

    let keyHolder = SamplerKeyHolder(seed: seed)
    keyHolder.advance(by: priorDraws)
    let temp = temperature
    let p = topP
    let k = topK
    let mp = minP

    return { @Sendable logprobs in
        // Match the single-request `TopPSampler` exactly (Evaluate.swift):
        // filter on the *unscaled* logprobs in top-P -> min-P -> top-K order,
        // then apply temperature at the categorical draw. The input is already
        // `logSoftmax(logits)` (see `DecodeBatch.step`), mirroring
        // `TopPSampler.sample`'s `logprobs = logSoftmax(logits)`. Applying
        // temperature before these filters would reshape the softmax mass and
        // change the nucleus / min-P boundary, so one batched row would
        // diverge from single-stream sampling for the same params/seed.
        var lp = logprobs
        if p > 0, p < 1 { lp = applyTopP(lp, p: p) }
        if mp > 0 { lp = applyMinP(lp, minP: mp) }
        if k > 0 { lp = applyTopK(lp, k: k) }
        let key = keyHolder.next()
        return MLXRandom.categorical(lp * (1.0 / temp), axis: -1, key: key)
    }
}

// MARK: - Top-K

/// Mask all but the top-K logits along the last axis to `-inf`, keeping
/// exactly `k` tokens. If `k >= vocab` the input is returned unchanged.
///
/// Mirrors the single-request `TopKSampler.applyTopK` in `Evaluate.swift`
/// (`apply_top_k` from `mlx_lm.sample_utils`): an O(V) `argPartition` on the
/// negated logprobs lands the top-`k` token indices in `[0, k)`, and the
/// remaining `[k, V)` indices are masked to `-inf`. Unlike a threshold
/// comparison (`logprobs >= cutoff`), this keeps exactly `k` tokens even when
/// several tokens tie at the cutoff logprob (e.g. uniform logits with
/// `topK: 1` keeps a single token instead of the whole vocabulary), matching
/// single-stream top-K semantics.
@usableFromInline
func applyTopK(_ logprobs: MLXArray, k: Int) -> MLXArray {
    let vocab = logprobs.shape.last ?? 0
    if k <= 0 || k >= vocab { return logprobs }

    let negInf = MLXArray(-Float.infinity)
    // Indices at [k, V) after partitioning are the tokens to mask out.
    let maskIndices = MLX.argPartition(-logprobs, kth: k - 1, axis: -1)[.ellipsis, k...]
    return putAlong(logprobs, maskIndices, values: negInf, axis: -1)
}

// MARK: - Min-P

/// Keep tokens whose probability is at least `minP * maxProb`, masking the
/// rest to `-inf`. Operates on log-probabilities, so the threshold in
/// log-space is `maxLogprob + log(minP)`.
///
/// Mirrors the single-request `TopPSampler.applyMinP` in `Evaluate.swift`
/// (`apply_min_p` from `mlx_lm.sample_utils`): the input is already
/// `logSoftmax(logits)`, so comparing `logprobs >= maxLogprob + log(minP)`
/// is equivalent to `prob >= minP * maxProb`. Applied between top-P and
/// top-K to match the single path's `top_p -> min_p -> top_k` filter chain.
@usableFromInline
func applyMinP(_ logprobs: MLXArray, minP: Float) -> MLXArray {
    // threshold in log-space: log(maxProb * minP) = maxLogprob + log(minP).
    // Compute log(minP) via MLX (as `Evaluate.swift` does) to avoid scalar
    // Float/Double ambiguity and to match the single path bit-for-bit.
    let maxLogprob = logprobs.max(axis: -1, keepDims: true)
    let threshold = maxLogprob + log(MLXArray(minP))
    let negInf = MLXArray(-Float.infinity)
    return which(logprobs .>= threshold, logprobs, negInf)
}

// MARK: - Top-P (nucleus)

/// Keep the smallest set of tokens whose cumulative softmax mass is at
/// least `p`. Tokens outside that nucleus are masked to `-inf`. The
/// "first token over threshold" stays in (mlx_lm semantics) so a
/// degenerate distribution still yields one valid pick.
@usableFromInline
func applyTopP(_ logprobs: MLXArray, p: Float) -> MLXArray {
    let sortedIdxAsc = argSort(logprobs, axis: -1)
    let sortedLogits = takeAlong(logprobs, sortedIdxAsc, axis: -1)
    let sortedProbs = softmax(sortedLogits, axis: -1)
    let cumProbs = sortedProbs.cumsum(axis: -1)

    // Keep tokens whose suffix-mass (tail starting here) >= 1 - p.
    // i.e. mask sorted positions where cumProbs <= 1 - p, but keep the
    // boundary token (first one above threshold).
    let keepThreshold = MLXArray(1.0 - p)
    let keepMask = cumProbs .> keepThreshold
    let negInf = MLXArray(-Float.infinity)
    let maskedSorted = which(keepMask, sortedLogits, negInf)

    // Scatter back to original token order.
    let inverseIdx = argSort(sortedIdxAsc, axis: -1)
    return takeAlong(maskedSorted, inverseIdx, axis: -1)
}

// MARK: - Per-sampler PRNG state

/// Holds a per-sampler PRNG key and advances it on every draw.
///
/// `RowSampler` is `@Sendable`, so the captured key holder must be
/// `Sendable` too. The mutable `MLXArray?` key is guarded by an
/// `OSAllocatedUnfairLock`, making this type **checked** `Sendable` with no
/// type-level `@unchecked Sendable` escape hatch (`OSAllocatedUnfairLock`
/// is itself unconditionally `Sendable`).
///
/// The `uncheckedState:` initializer and `withLockUnchecked` accessor are
/// required because `MLXArray` is not `Sendable`; they relax the *State*
/// Sendability requirement of the lock, not the type's `Sendable`
/// conformance. In practice the engine invokes a row's sampler from a single
/// executor, so the lock is uncontended; it exists only to satisfy the
/// `Sendable` checker without an `@unchecked` type conformance.
final class SamplerKeyHolder: Sendable {
    private let key: OSAllocatedUnfairLock<MLXArray?>

    init(seed: UInt64?) {
        self.key = OSAllocatedUnfairLock(uncheckedState: seed.map { MLXRandom.key($0) })
    }

    func next() -> MLXArray? {
        key.withLockUnchecked { current in
            guard let value = current else { return nil }
            // Role order matches mlx-swift's RandomState.next() (used by the
            // single path's seeded TopPSampler): split[0] becomes the next
            // PRNG state, split[1] is the per-draw subkey. Reversing them
            // consumes a disjoint key chain and breaks seeded single-vs-
            // batched parity from the first draw.
            let (nextState, drawKey) = MLXRandom.split(key: value)
            current = nextState
            return drawKey
        }
    }

    /// Fast-forward the key chain by `draws` splits without yielding keys,
    /// exactly as if ``next()`` had been called `draws` times. Because the
    /// chain is bit-identical to the single path's seeded `RandomState`
    /// (see ``next()``), the single→batch upgrade uses this to make a fresh
    /// holder CONTINUE the sequence at the draw the single iterator would
    /// have consumed next, instead of replaying draws 1..N. Unseeded holders
    /// stay keyless. No-op for `draws <= 0`.
    func advance(by draws: Int) {
        guard draws > 0 else { return }
        key.withLockUnchecked { current in
            guard var value = current else { return }
            for _ in 0 ..< draws {
                value = MLXRandom.split(key: value).0
            }
            // Materialize the advanced state so the skipped splits do not
            // accumulate as one deep lazy graph evaluated at the next draw.
            eval(value)
            current = value
        }
    }
}
