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

/// Build a `RowSampler` from OpenAI-style request parameters.
///
/// Behavior, matching the single-request `TopPSampler` in `Evaluate.swift`
/// (which mirrors `mlx_lm.sample_utils.make_sampler`):
///   - `temperature <= 0`: return `greedySampler` (no RNG, deterministic).
///   - `temperature > 0`: optionally mask the unscaled logprobs to top-P
///     then top-K, scale by `1/temperature`, and sample categorically. The
///     order (filter first, temperature at the draw) means one batched row
///     reproduces single-stream sampling for the same params/seed.
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
///   - seed: optional RNG seed; nil uses MLX's global PRNG.
public func makeRowSampler(
    temperature: Float = 0.0,
    topP: Float = 1.0,
    topK: Int = 0,
    seed: UInt64? = nil
) -> RowSampler {
    if temperature <= 0 {
        return greedySampler
    }

    let keyHolder = SamplerKeyHolder(seed: seed)
    let temp = temperature
    let p = topP
    let k = topK

    return { @Sendable logprobs in
        // Match the single-request `TopPSampler` exactly (Evaluate.swift):
        // filter on the *unscaled* logprobs in top-P -> top-K order, then
        // apply temperature at the categorical draw. The input is already
        // `logSoftmax(logits)` (see `DecodeBatch.step`), mirroring
        // `TopPSampler.sample`'s `logprobs = logSoftmax(logits)`. Applying
        // temperature before top-P would reshape the softmax mass and change
        // the nucleus boundary, so one batched row would diverge from
        // single-stream sampling for the same params/seed.
        var lp = logprobs
        if p > 0, p < 1 { lp = applyTopP(lp, p: p) }
        if k > 0 { lp = applyTopK(lp, k: k) }
        let key = keyHolder.next()
        return MLXRandom.categorical(lp * (1.0 / temp), axis: -1, key: key)
    }
}

// MARK: - Top-K

/// Mask all but the top-K logits along the last axis to `-inf`. If
/// `k >= vocab` the input is returned unchanged.
@usableFromInline
func applyTopK(_ logprobs: MLXArray, k: Int) -> MLXArray {
    let vocab = logprobs.shape.last ?? 0
    if k <= 0 || k >= vocab { return logprobs }

    let sortedIdx = argSort(-logprobs, axis: -1)
    let topIdx = sortedIdx[.ellipsis, 0 ..< k]
    let topVals = takeAlong(logprobs, topIdx, axis: -1)
    let threshold = topVals.min(axes: [-1], keepDims: true)
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
            let (subkey, nextKey) = MLXRandom.split(key: value)
            current = nextKey
            return subkey
        }
    }
}
