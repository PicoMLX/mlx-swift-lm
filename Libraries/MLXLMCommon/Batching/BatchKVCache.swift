// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// Lives here (the first batched-cache file in the PR chain) rather than in
// BatchRotatingKVCache.swift: BatchKVCache.finalize and the quantized cache
// also roll per-row, and the rotating cache stacks on top of this file.
/// Per-element roll along a specified axis.
///
/// Ported from Python mlx-lm's `dynamic_roll`. Each element along the batch
/// dimension is rolled by its own shift amount.
///
/// - Parameters:
///   - x: The input array.
///   - shifts: Per-batch shift amounts. Shape must broadcast with `x` along axes
///     other than `axis`.
///   - axis: The axis along which to roll.
/// - Returns: The rolled array.
internal func dynamicRoll(_ x: MLXArray, shifts: MLXArray, axis: Int) -> MLXArray {
    let n = x.dim(axis)

    // Build index shape for broadcasting.
    let ndim = x.ndim
    let positiveAxis = axis >= 0 ? axis : ndim + axis

    // arange indices along the roll axis
    let indices = MLXArray(Int32(0) ..< Int32(n))

    // Reshape indices so they broadcast: [1, ..., 1, n, 1, ..., 1]
    var idxShape = [Int](repeating: 1, count: ndim)
    idxShape[positiveAxis] = n
    let reshapedIndices = indices.reshaped(idxShape)

    // Reshape shifts to broadcast: add trailing dims after the axis
    // shifts shape: e.g. [B, 1] → needs to become [B, 1, 1, ..., 1]
    var shiftShape = [Int](repeating: 1, count: ndim)
    for d in 0 ..< shifts.ndim {
        if d < ndim {
            shiftShape[d] = shifts.dim(d)
        }
    }
    let reshapedShifts = shifts.reshaped(shiftShape)

    // Compute rolled indices: (indices - shifts) mod n
    // Use ((x % n) + n) % n to ensure non-negative result (Python-style modulo)
    // ((x % n) + n) % n keeps the result non-negative (Python-style modulo).
    // Using `%` avoids any overload ambiguity with Foundation/Darwin `remainder`.
    let nArr = MLXArray(Int32(n))
    let idx = ((reshapedIndices - reshapedShifts) % nArr + nArr) % nArr

    return takeAlong(x, idx.asType(.int32), axis: positiveAxis)
}

// MARK: - BatchKVCache

/// Batch-aware KV cache with left-padding strategy for continuous batching.
///
/// Ported from Python mlx-lm's `BatchKVCache`. The cache expects inputs to be
/// left-padded so that variable-length sequences align on the right.
///
/// For example, prompts `[1, 3, 5]`, `[7]`, and `[2, 6, 8, 9]` are padded:
/// ```
/// [0, 1, 3, 5]
/// [0, 0, 0, 7]
/// [2, 6, 8, 9]
/// ```
/// With `leftPadding = [1, 3, 0]`.
public class BatchKVCache: BaseKVCache, BatchPositionedKVCache, BatchedCache {

    /// Per-sequence left-padding amounts as an MLXArray of shape `[B]`.
    public internal(set) var leftPadding: MLXArray

    /// Per-sequence offset as an MLXArray of shape `[B]`.
    /// Starts negative (equal to `-leftPadding`) and advances with each update.
    public internal(set) var batchOffsets: MLXArray

    /// Internal buffer index tracking how far into the keys/values buffer we've written.
    internal var _idx: Int = 0

    /// Keys buffer: `[B, H, S_buf, D_k]`
    internal var keys: MLXArray?

    /// Values buffer: `[B, H, S_buf, D_v]`
    internal var values: MLXArray?

    /// Step size for buffer allocation (grow in chunks of this size).
    public var step: Int = 256

    /// The scalar offset (not meaningful for batch caches, returns `_idx`).
    public override var offset: Int {
        get { _idx }
        set { _idx = newValue }
    }

    /// Initialize a BatchKVCache with the given left-padding per sequence.
    ///
    /// - Parameter leftPadding: Array of integers specifying the left-padding for each sequence.
    public init(leftPadding: [Int]) {
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.batchOffsets = MLXArray(leftPadding.map { -Int32($0) })
        super.init()
    }

    /// Internal initializer for creating empty batch caches with pre-built MLXArrays.
    internal init(leftPaddingArray: MLXArray, batchOffsetsArray: MLXArray) {
        self.leftPadding = leftPaddingArray
        self.batchOffsets = batchOffsetsArray
        super.init()
    }

    // MARK: - KVCache Protocol

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    /// Update the cache with new keys and values.
    ///
    /// Keys/values have shape `[B, H, S, D]` where `S` is the number of new tokens.
    /// The cache buffer grows in steps of `step` size.
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let prev = _idx

        let reset: Bool
        if let currentKeys = self.keys, (prev + keys.dim(2)) <= currentKeys.dim(2) {
            reset = false
        } else {
            reset = true
        }

        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if prev % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<prev, 0...]
                    currentValues = currentValues[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        batchOffsets = batchOffsets + Int32(keys.dim(2))
        _idx += keys.dim(2)

        self.keys?[.ellipsis, prev ..< _idx, 0...] = keys
        self.values?[.ellipsis, prev ..< _idx, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<_idx, 0...]
        let returnedValues = self.values![.ellipsis, ..<_idx, 0...]

        return (returnedKeys, returnedValues)
    }

    // MARK: - State Serialization

    public override var state: [MLXArray] {
        get {
            // Always include batchOffsets and leftPadding, even when keys/values are nil
            // (e.g. fresh cache or cache emptied by filter(batchIndices: [])).
            guard let keys = self.keys, let values = self.values else {
                return [batchOffsets, leftPadding]
            }
            let k: MLXArray
            let v: MLXArray
            if _idx < keys.dim(2) {
                k = keys[.ellipsis, ..<_idx, 0...]
                v = values[.ellipsis, ..<_idx, 0...]
            } else {
                k = keys
                v = values
            }
            return [k, v, batchOffsets, leftPadding]
        }
        set {
            switch newValue.count {
            case 2:
                // Empty cache: only batchOffsets and leftPadding
                self.keys = nil
                self.values = nil
                self.batchOffsets = newValue[0]
                self.leftPadding = newValue[1]
                self._idx = 0
            case 4:
                // Populated cache: keys, values, batchOffsets, leftPadding
                self.keys = newValue[0]
                self.values = newValue[1]
                self.batchOffsets = newValue[2]
                self.leftPadding = newValue[3]
                self._idx = self.keys!.dim(2)
            default:
                fatalError(
                    "BatchKVCache state must have 2 arrays (batchOffsets, leftPadding) or 4 arrays (keys, values, batchOffsets, leftPadding)"
                )
            }
        }
    }

    public override var metaState: [String] {
        get { [String(_idx)] }
        set {
            guard newValue.count == 1 else {
                fatalError("BatchKVCache metaState must have exactly 1 value")
            }
            self._idx = Int(newValue[0]) ?? 0
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        // `_idx` is the padded maximum width across rows; `batchOffsets[i]` is the
        // live (post-padding) token count of row `i` (it starts at `-leftPadding`
        // and advances by the number of tokens written). Trimming the full `n` from
        // every row would drive a shorter row's offset negative, leaving its
        // `leftPadding` larger than the new `_idx`. A later `extract` then slices
        // `padding ..< _idx` with an invalid range and decode sees invalid RoPE
        // positions. Clamp the trim to the minimum live per-row length so no row can
        // underflow, and return the clamped amount.
        guard _idx > 0 else { return 0 }
        let minLiveRow = max(0, Int(batchOffsets.min().item(Int32.self)))
        let trimmed = min(n, _idx, minLiveRow)
        _idx -= trimmed
        batchOffsets = batchOffsets - Int32(trimmed)
        return trimmed
    }

    /// Rewind each row's tail by `perRow[i]` tokens, keeping every row's end
    /// aligned at the (reduced) shared write position.
    ///
    /// Same mechanics as ``finalize()`` in reverse: the shared `_idx` drops by
    /// the SMALLEST requested trim, and each row needing a larger trim is
    /// rolled right by the difference so its retained tail lands on the new
    /// common end column; the rolled-in garbage wraps into the row's
    /// left-padding region, which masks it out. Per-row trims are clamped to
    /// the row's live length; the clamped amounts are returned.
    public func trimBatched(perRow: [Int]) -> [Int] {
        precondition(
            perRow.count == batchSize,
            "trimBatched(perRow:) requires one entry per row")
        guard keys != nil, _idx > 0 else {
            return Array(repeating: 0, count: perRow.count)
        }
        let live = batchOffsets.asArray(Int32.self).map { max(0, Int($0)) }
        let trims = zip(perRow, live).map { min(max(0, $0), $1, _idx) }
        guard let minTrim = trims.min() else { return [] }
        if trims.allSatisfy({ $0 == 0 }) { return trims }

        // Rows trimming more than the minimum shift right to stay aligned.
        let shifts = trims.map { Int32($0 - minTrim) }
        if shifts.contains(where: { $0 > 0 }), let k = keys, let v = values {
            let shiftArr = MLXArray(shifts)[0..., .newAxis]
            // Roll only the populated `[0, _idx)` span, mirroring `finalize()`:
            // the buffer is allocated in `step`-sized blocks, so rolling the
            // full width would gather over the zero-filled spare tail (up to
            // ~stepx wasted work for small batches) and park the relocated
            // dead-draft garbage there instead of dropping it. The head slice
            // also shrinks the buffer to `_idx`; the next `update` re-grows it.
            let kHead = k[.ellipsis, ..<_idx, 0...]
            let vHead = v[.ellipsis, ..<_idx, 0...]
            keys = dynamicRoll(kHead, shifts: shiftArr, axis: 2)
            values = dynamicRoll(vHead, shifts: shiftArr, axis: 2)
            leftPadding = leftPadding + MLXArray(shifts)
        }
        _idx -= minTrim
        batchOffsets = batchOffsets - MLXArray(trims.map { Int32($0) })
        return trims
    }

    /// The batch size (number of sequences).
    public var batchSize: Int {
        leftPadding.dim(0)
    }

    /// Whether the cache is empty (no keys/values stored).
    public var isEmpty: Bool {
        keys == nil
    }

    // MARK: - BatchPositionedKVCache Conformance

    /// Per-sequence position offsets as an MLXArray of shape `[B]`.
    ///
    /// This is an alias for `batchOffsets`, providing the per-sequence position
    /// offsets needed for batch-aware RoPE application via `applyRotaryPosition()`.
    public var batchOffset: MLXArray {
        batchOffsets
    }

    /// Override `BaseKVCache`'s scalar `ropeOffset` so per-row offsets are used
    /// even when this cache flows through attention layers typed as `KVCache`.
    public override var ropeOffset: RoPEOffset { .batch(batchOffset + 0) }

    // MARK: - Batch Operations

    /// In-place filter to keep only the sequences at the given batch indices.
    ///
    /// After filtering, the minimum left-padding is subtracted from all sequences
    /// and the buffer is trimmed accordingly (shift left to reduce padding).
    ///
    /// - Parameter batchIndices: Array of batch indices to keep.
    public func filter(batchIndices: [Int]) {
        // Handle empty filter -> produce valid empty state
        guard !batchIndices.isEmpty else {
            keys = nil
            values = nil
            leftPadding = MLXArray([Int32]())
            batchOffsets = MLXArray([Int32]())
            _idx = 0
            // Clear pending right-padding so a later finalize() can't roll/subtract
            // with stale per-row metadata against the now-empty batch.
            _rightPadding = nil
            return
        }

        let indices = MLXArray(batchIndices.map { Int32($0) })

        // Filter along batch dimension (dim 0)
        keys = keys?[indices]
        values = values?[indices]
        batchOffsets = batchOffsets[indices]
        leftPadding = leftPadding[indices]
        // Transient right-padding metadata (set by prepareBatched, consumed by
        // finalize) is per-row too: filter it with the same indices so a cancel
        // between prepare and finalize doesn't leave finalize rolling/subtracting
        // against the pre-filter batch shape.
        _rightPadding = _rightPadding?[indices]

        // Shift left to reduce padding. Only meaningful once a KV buffer exists:
        // for a fresh/cancelled-before-prefill cache the tensor slices are no-ops,
        // but decrementing `_idx` would drive it negative and leave `batchOffsets`
        // inconsistent with the adjusted padding, corrupting the `prev` range used
        // by the next prefill. Skip the shift entirely while `keys == nil`.
        //
        // Also require `_idx >= minLeftPad`: during chunked prefill the KV buffer
        // can exist before the write pointer has advanced past the padding. Shifting
        // then would drive `_idx` negative and slice past the written region (keeping
        // uninitialized data), so skip until enough tokens have been written.
        let minLeftPad = leftPadding.min().item(Int32.self)
        if minLeftPad > 0, keys != nil, _idx >= Int(minLeftPad) {
            let padInt = Int(minLeftPad)
            keys = keys?[.ellipsis, padInt..., 0...]
            values = values?[.ellipsis, padInt..., 0...]
            _idx -= padInt
            leftPadding = leftPadding - minLeftPad
        }
    }

    /// In-place extend this cache with another BatchKVCache.
    ///
    /// The caches are right-justified: the shorter cache gets additional left-padding
    /// to align with the longer one along the sequence dimension.
    ///
    /// - Parameter other: The other BatchKVCache to merge into this one.
    ///
    /// Admission contract: the engine prefills every admitted sub-batch before
    /// extending the active batch, so a non-empty `other` always arrives with a
    /// populated KV buffer. This precondition makes that contract explicit and
    /// closes the "append empty admitted rows" hole: an `other` that carries row
    /// metadata (`batchSize > 0`) but no prefilled keys is rejected here instead
    /// of silently dropping those rows from the enlarged batch.
    public func extend(other: BatchKVCache) {
        precondition(
            other.keys != nil || other.batchSize == 0,
            "BatchKVCache.extend requires a non-empty `other` to be prefilled "
                + "(keys != nil) before extending; the engine prefills each "
                + "admitted sub-batch before calling extend."
        )
        // A pending `prepare(rightPadding:)` cycle must be finalized before the
        // row set changes: `_rightPadding` is shaped `[batchSize]`, so growing
        // the batch here would make a later `finalize()` roll with mismatched
        // shifts (a broadcast shape error -- or, for a batch-1 receiver, a
        // silent mis-roll of every appended row), and a prepared `other`'s
        // pending padding would be dropped with its right-pad columns left
        // permanently unmasked. Mirrors `filter`, which reconciles
        // `_rightPadding` on every row-set change. The engine never hits this
        // (`PrefillBatch.prompt` pairs prepare/finalize in one call); it guards
        // direct API use.
        precondition(
            _rightPadding == nil && other._rightPadding == nil,
            "BatchKVCache.extend requires both sides to have no pending "
                + "prepare(rightPadding:) cycle; call finalize() first."
        )
        guard let selfKeys = self.keys, let otherKeys = other.keys else {
            if self.keys == nil && other.keys == nil {
                // Both empty: concatenate row metadata so admitted rows survive
                // until their first prefill instead of being dropped.
                self.leftPadding = concatenated([self.leftPadding, other.leftPadding], axis: 0)
                self.batchOffsets = concatenated([self.batchOffsets, other.batchOffsets], axis: 0)
            } else if other.keys != nil {
                // Adoption replaces this cache's row set wholesale, so the
                // receiver must not be carrying admitted-but-unprefilled rows
                // (metadata without keys) that would silently vanish.
                precondition(
                    self.batchSize == 0,
                    "BatchKVCache.extend cannot adopt a prefilled `other` into "
                        + "a receiver holding un-prefilled rows; prefill the "
                        + "receiver's rows first."
                )
                // self empty, other populated: adopt other's state.
                self.keys = other.keys
                self.values = other.values
                self.batchOffsets = other.batchOffsets
                self.leftPadding = other.leftPadding
                self._idx = other._idx
            }
            return
        }

        let maxIdx = max(self._idx, other._idx)
        let maxSize = max(selfKeys.dim(2), otherKeys.dim(2))

        // Inner function to pad a cache's keys/values for right-justification.
        func pad(
            _ cache: BatchKVCache
        ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
            let left = maxIdx - cache._idx
            var right = maxSize - cache.keys!.dim(2) - left

            var k = cache.keys!
            var v = cache.values!

            if right < 0 {
                k = k[.ellipsis, ..<(k.dim(2) + right), 0...]
                v = v[.ellipsis, ..<(v.dim(2) + right), 0...]
                right = 0
            }

            if left != 0 || right != 0 {
                let padWidths: [IntOrPair] = [0, 0, .init((left, right)), 0]
                k = MLX.padded(k, widths: padWidths)
                v = MLX.padded(v, widths: padWidths)
            }

            let adjustedLeftPadding = cache.leftPadding + Int32(left)

            return (k, v, cache.batchOffsets, adjustedLeftPadding)
        }

        let (selfK, selfV, selfOff, selfLP) = pad(self)
        let (otherK, otherV, otherOff, otherLP) = pad(other)

        self.keys = concatenated([selfK, otherK], axis: 0)
        self.values = concatenated([selfV, otherV], axis: 0)
        self.batchOffsets = concatenated([selfOff, otherOff], axis: 0)
        self.leftPadding = concatenated([selfLP, otherLP], axis: 0)
        self._idx = maxIdx
    }

    /// Extract a single sequence from the batch as a `KVCacheSimple`.
    ///
    /// The returned cache has the left-padding stripped and contains only the
    /// valid (non-padded) key/value data.
    ///
    /// - Parameter idx: The batch index of the sequence to extract.
    /// - Returns: A `KVCacheSimple` with the extracted sequence data.
    public func extract(idx: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        // Clamp: a row still inside its left-padding prefix (metadata rows
        // preserved by `filter` before their first real token) has
        // padding > _idx, and `padding ..< _idx` would trap. Clamping yields
        // an empty single-row cache instead.
        let padding = min(Int(leftPadding[idx].item(Int32.self)), _idx)

        if let k = keys, let v = values {
            cache.keys = MLX.contiguous(k[idx ..< (idx + 1), 0..., padding ..< _idx, 0...])
            cache.values = MLX.contiguous(v[idx ..< (idx + 1), 0..., padding ..< _idx, 0...])
            cache.offset = cache.keys!.dim(2)
        }

        return cache
    }

    /// Create a BatchKVCache by merging multiple individual KVCache instances.
    ///
    /// Each cache is right-justified in the batch: shorter caches receive left-padding
    /// to match the longest sequence.
    ///
    /// - Parameter caches: An array of `KVCacheSimple` instances.
    /// - Returns: A new `BatchKVCache` containing all sequences.
    public class func merge(_ caches: [KVCache]) -> BatchKVCache {
        // The copy loop below reads data only from `KVCacheSimple` instances; a
        // non-simple cache would silently contribute an all-zero row that the
        // mask still exposes. Fail loudly instead. `ChunkedKVCache` subclasses
        // `KVCacheSimple` but keeps an absolute `offset` after front-trimming, so
        // it would mis-slice past its retained chunk here — reject it explicitly
        // (matching the factory's rejection of chunked topologies).
        precondition(
            caches.allSatisfy { $0 is KVCacheSimple && !($0 is ChunkedKVCache) },
            "BatchKVCache.merge requires non-chunked KVCacheSimple instances"
        )
        let lengths = caches.map { $0.offset }
        let maxLength = lengths.max() ?? 0
        let padding = lengths.map { maxLength - $0 }
        let B = caches.count

        // Find dimensions from first non-empty cache
        var H = 0
        var Dk = 0
        var Dv = 0
        var dt: DType = .float16
        var dtV: DType = .float16

        for c in caches {
            if let simple = c as? KVCacheSimple, let k = simple.keys {
                H = k.dim(1)
                Dk = k.dim(3)
                Dv = simple.values!.dim(3)
                dt = k.dtype
                // Track the value dtype separately -- allocating values with
                // the key dtype would silently cast mixed-dtype KV caches.
                dtV = simple.values!.dtype
                break
            }
        }

        guard H > 0 else {
            // All caches are empty
            return BatchKVCache(leftPadding: padding)
        }

        let keysArr = MLXArray.zeros([B, H, maxLength, Dk], dtype: dt)
        let valuesArr = MLXArray.zeros([B, H, maxLength, Dv], dtype: dtV)

        for (i, (p, c)) in zip(padding, caches).enumerated() {
            if let simple = c as? KVCacheSimple, let k = simple.keys, let v = simple.values {
                let seqLen = c.offset
                keysArr[i ..< (i + 1), 0..., p ..< (p + seqLen), 0...] =
                    k[.ellipsis, ..<seqLen, 0...]
                valuesArr[i ..< (i + 1), 0..., p ..< (p + seqLen), 0...] =
                    v[.ellipsis, ..<seqLen, 0...]
            }
        }

        let cache = BatchKVCache(leftPadding: padding)
        cache.keys = keysArr
        cache.values = valuesArr
        // After merge, offset should advance by maxLength for all sequences
        cache.batchOffsets = cache.batchOffsets + Int32(maxLength)
        cache._idx = maxLength

        return cache
    }

    /// Create a batch-1 BatchKVCache from a single KVCacheSimple.
    ///
    /// The resulting cache has `leftPadding = [0]` and identical data.
    ///
    /// - Parameter cache: A single `KVCacheSimple` to wrap.
    /// - Returns: A new `BatchKVCache` with batch size 1.
    public class func fromSingle(_ cache: KVCacheSimple) -> BatchKVCache {
        // ChunkedKVCache subclasses KVCacheSimple but keeps `offset` absolute
        // while maybeTrimFront() shrinks `keys` to the retained chunk;
        // adopting it would set `_idx` past the buffer and corrupt later
        // slices. Mirrors the identical guard in `merge`.
        precondition(
            !(cache is ChunkedKVCache),
            "BatchKVCache.fromSingle requires a non-chunked KVCacheSimple: a "
                + "front-trimmed ChunkedKVCache's offset exceeds its buffer."
        )
        let batchCache = BatchKVCache(leftPadding: [0])

        if let k = cache.keys, let v = cache.values {
            batchCache.keys = k
            batchCache.values = v
            batchCache._idx = cache.offset
            batchCache.batchOffsets = MLXArray([Int32(cache.offset)])
        }

        return batchCache
    }

    /// Convert a batch-1 BatchKVCache back to a KVCacheSimple.
    ///
    /// - Returns: A `KVCacheSimple` with the single sequence data.
    public func toSingle() -> KVCacheSimple {
        precondition(batchSize == 1, "toSingle() requires batch size of 1")
        return extract(idx: 0)
    }

    // MARK: - Mask Creation

    /// Create an attention mask for this batch cache.
    ///
    /// Unlike non-batch caches which return `.none` for `n=1`, batch caches
    /// MUST always produce a mask that excludes left-padded positions. This
    /// ensures that during single-token decode steps, padded positions are
    /// still correctly masked out.
    ///
    /// - Parameters:
    ///   - n: The sequence length for the new tokens
    ///   - windowSize: Optional sliding window size
    ///   - returnArray: Force return of array mask instead of symbolic
    /// - Returns: Attention mask mode for scaled dot product attention
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // Batch caches always need an explicit mask to handle left-padding,
        // even for n=1 decode steps.
        //
        // makeMask() runs before attentionWithCacheUpdate(), but that helper
        // appends the current step's keys/values before launching attention.
        // The attention kernel therefore sees the post-update cache width, so
        // the mask must span the existing cache plus the n incoming tokens.
        return .array(
            createCausalMask(
                n: n, offset: _idx, windowSize: windowSize, leftPadding: leftPadding
            )
        )
    }

    // MARK: - Prepare / Finalize (Cached-Prompt Prefill)

    /// Stored right-padding for the current prefill cycle.
    /// Set by `prepare(rightPadding:)` and consumed by `finalize()`.
    internal var _rightPadding: MLXArray?

    /// Prepare the cache for a cached-prompt batch prefill with right-padding.
    ///
    /// During mixed-depth cached-prompt prefill, suffix tokens are
    /// RIGHT-padded (shorter suffixes padded on the right to match the
    /// longest suffix). After prefill, the right-padding zeros sit at
    /// positions that `createCausalMask` does NOT mask out, corrupting
    /// attention. `finalize()` fixes this by rolling the right-padding
    /// zeros to the LEFT side of the buffer.
    ///
    /// Matches Python mlx-lm's `BatchKVCache.prepare()`.
    ///
    /// - Parameter rightPadding: Per-sequence right-padding amounts as
    ///   an MLXArray of shape `[B]`.
    public func prepare(rightPadding: MLXArray) {
        // Only store if there's any non-zero padding
        if rightPadding.max().item(Int32.self) > 0 {
            _rightPadding = rightPadding
        }
    }

    /// Finalize the cache after a cached-prompt batch prefill.
    ///
    /// If `prepare(rightPadding:)` was called, this method uses
    /// `dynamicRoll` to shift each sequence's KV data so that
    /// right-padding zeros move to the LEFT side of the buffer,
    /// then adjusts `leftPadding += rightPadding` and
    /// `batchOffsets -= rightPadding`.
    ///
    /// After finalize, all padding is contiguous on the left and
    /// the causal mask correctly excludes it.
    ///
    /// Matches Python mlx-lm's `BatchKVCache.finalize()`.
    public override func finalize() {
        guard let padding = _rightPadding else { return }

        if let k = keys, let v = values {
            // Roll only the populated `[0, _idx)` span. The buffer is allocated in
            // `step`-sized blocks, so `keys.dim(2)` is usually larger than `_idx`
            // with a zero-filled spare tail; rolling the whole buffer would wrap
            // that tail into the valid prefix and corrupt the next attention step.
            // Operating on the head slice also shrinks the buffer to `_idx`, which
            // the next `update` re-grows as needed.
            let kHead = k[.ellipsis, ..<_idx, 0...]
            let vHead = v[.ellipsis, ..<_idx, 0...]
            self.keys = dynamicRoll(kHead, shifts: padding[0..., .newAxis], axis: 2)
            self.values = dynamicRoll(vHead, shifts: padding[0..., .newAxis], axis: 2)
        }
        batchOffsets = batchOffsets - padding
        leftPadding = leftPadding + padding
        _rightPadding = nil
    }

    // MARK: - BatchedCache Conformance

    /// Bridges the `BatchedCache` protocol (MLXArray-indexed, used by the engine)
    /// onto this cache's concrete operations. `batchIndices` is small (`[B]`), so
    /// the host round-trip is negligible next to the per-token decode work.
    public func filterBatched(batchIndices: MLXArray) {
        filter(batchIndices: batchIndices.asArray(Int32.self).map(Int.init))
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? BatchKVCache else {
            preconditionFailure("BatchKVCache.extendBatched requires another BatchKVCache")
        }
        extend(other: other)
    }

    public func prepareBatched(leftPadding: [Int]?, lengths _: [Int]?, rightPadding: [Int]?) {
        if let leftPadding {
            precondition(
                keys == nil,
                "prepareBatched(leftPadding:) requires an empty BatchKVCache"
            )
            let additional = MLXArray(leftPadding.map { Int32($0) })
            self.leftPadding = self.leftPadding + additional
            self.batchOffsets = self.batchOffsets - additional
        }
        if let rightPadding, rightPadding.contains(where: { $0 > 0 }) {
            prepare(rightPadding: MLXArray(rightPadding.map { Int32($0) }))
        }
    }

    public func finalizeBatched() {
        finalize()
    }

    public func extractBatched(_ idx: Int) -> any KVCache {
        extract(idx: idx)
    }

    /// Full-attention caches have no chunk-local prefill metadata to advance.
    public func advanceBatched(_: Int) {}

    public override func copy() -> any KVCache {
        let c = BatchKVCache(
            leftPaddingArray: leftPadding[0...], batchOffsetsArray: batchOffsets[0...])
        c.keys = keys.map { $0[.ellipsis] }
        c.values = values.map { $0[.ellipsis] }
        c._idx = _idx
        c.step = step
        c._rightPadding = _rightPadding.map { $0[0...] }
        return c
    }

    public var debugDescription: String {
        "BatchKVCache batchSize: \(batchSize), _idx: \(_idx), keys: \(keys?.shape.description ?? "-"), values: \(values?.shape.description ?? "-")"
    }
}
