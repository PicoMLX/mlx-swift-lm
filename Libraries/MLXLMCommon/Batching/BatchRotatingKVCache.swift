// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - Dynamic Roll Helper

// MARK: - RotatingKVCache Internal Extension

extension RotatingKVCache {
    /// Returns temporally ordered keys/values suitable for merging into a batch cache.
    ///
    /// When the rotating cache has wrapped around (offset >= maxSize), the internal
    /// buffer may not be in temporal order. This method returns the state in correct
    /// temporal order, which is needed for `BatchRotatingKVCache.merge()`.
    ///
    /// The returned arrays have shape `[1, H, seqLen, D]` where `seqLen = min(offset, maxSize)`.
    internal var temporalState: [MLXArray] {
        // The `state` getter on RotatingKVCache already handles slicing:
        // - When offset < keys.dim(2): returns keys[..<offset] (temporal order, no wrap)
        // - When offset >= keys.dim(2): returns full buffer (may be rotated)
        //
        // For a rotated buffer, we need to reconstruct temporal order.
        // We read metaState to get the idx and reconstruct.
        let meta = self.metaState
        guard meta.count >= 5,
            let keep = Int(meta[0]),
            let ms = Int(meta[1]),
            let off = Int(meta[3]),
            let ix = Int(meta[4])
        else {
            return self.state
        }

        let rawState = self.state
        guard rawState.count == 2 else { return rawState }

        let k = rawState[0]
        let v = rawState[1]

        // No rotation needed if offset < maxSize (buffer hasn't wrapped)
        if off < ms {
            return [k, v]
        }

        // Buffer is full and may be rotated. Reconstruct temporal order.
        // The idx tells us where the next write would go, so data before idx
        // is newer and data from idx onwards is older.
        if ix == k.dim(2) {
            // No rotation happened or idx is at the end
            return [k, v]
        } else if ix < off {
            // Rotated: [keep tokens][newer tokens from idx..][older tokens keep..<idx]
            let orderedK = concatenated(
                [
                    k[.ellipsis, ..<keep, 0...],
                    k[.ellipsis, ix..., 0...],
                    k[.ellipsis, keep ..< ix, 0...],
                ], axis: 2)
            let orderedV = concatenated(
                [
                    v[.ellipsis, ..<keep, 0...],
                    v[.ellipsis, ix..., 0...],
                    v[.ellipsis, keep ..< ix, 0...],
                ], axis: 2)
            return [orderedK, orderedV]
        } else {
            return [k[.ellipsis, ..<ix, 0...], v[.ellipsis, ..<ix, 0...]]
        }
    }
}

// MARK: - BatchRotatingKVCache

/// Batch-aware rotating KV cache for models using sliding-window attention.
///
/// Ported from Python mlx-lm's `BatchRotatingKVCache`. Combines the left-padding
/// strategy of `BatchKVCache` with the sliding window rotation of `RotatingKVCache`.
///
/// For models with bounded context windows (e.g. Mistral, Gemma), this cache
/// manages multiple sequences simultaneously, each at potentially different
/// positions, with a fixed maximum cache size per sequence.
///
/// Like `BatchKVCache`, inputs are expected to be left-padded so that
/// variable-length sequences align on the right.
///
/// > Warning: `keep > 0` is only sound while every row has zero left padding.
/// > At the rotation wrap, padded rows roll their pads to the END of the buffer
/// > to protect the keep prefix, but the prefix-only `leftPadding` mask cannot
/// > express trailing garbage, so those slots would be attended until
/// > overwritten. `makeBatchedCacheFactories` therefore rejects keep-prefix
/// > topologies; direct construction with `keep > 0` is for equal-length
/// > batches only.
public class BatchRotatingKVCache: BaseKVCache, BatchPositionedKVCache, BatchedCache {

    /// Per-sequence left-padding amounts as an MLXArray of shape `[B]`.
    public internal(set) var leftPadding: MLXArray

    /// Per-sequence offset as an MLXArray of shape `[B]`.
    /// Starts negative (equal to `-leftPadding`) and advances with each update.
    public internal(set) var batchOffsets: MLXArray

    /// Internal buffer index tracking how far into the keys/values buffer we've written.
    internal var _idx: Int = 0

    /// Scalar offset tracking total tokens written (similar to RotatingKVCache._offset in Python).
    internal var _scalarOffset: Int = 0

    /// Whether the cache buffer has wrapped around (rotation has occurred).
    internal var rotated: Bool = false

    /// Keys buffer: `[B, H, S_buf, D_k]`
    internal var keys: MLXArray?

    /// Values buffer: `[B, H, S_buf, D_v]`
    internal var values: MLXArray?

    /// Maximum cache size (sliding window size).
    private var maxCacheSize: Int

    /// Number of tokens to always keep at the start of the cache during rotation.
    /// Mirrors `RotatingKVCache.keep`.
    public internal(set) var keep: Int = 0

    /// Stored lengths for right-padded inputs during cached-prompt prefill.
    /// Set by `prepare(rightPadding:lengths:)` and consumed by `finalize()`.
    internal var _lengths: MLXArray?

    /// Step size for buffer allocation.
    public var step: Int = 256

    /// The maximum size of this cache (sliding window size).
    public override var maxSize: Int? { maxCacheSize }

    /// The absolute number of tokens processed, exactly matching single-stream
    /// `RotatingKVCache.offset` (BaseKVCache's monotonic counter).
    ///
    /// `_idx` is the circular write pointer (reset to `keep` on wrap), so it is
    /// not a valid position to expose to generic callers: after the first
    /// rotation it jumps backward. And capping at `maxCacheSize` is equally
    /// wrong: models consume `cache.offset` as an absolute position — e.g.
    /// Gemma3n's `cachePosition` and Mistral3's llama-4 attention scaling — so
    /// a capped value would freeze those computations once the window wraps,
    /// diverging from the same model run single-stream. Mask math that needs
    /// the window-capped span derives it internally (`makeMask`); paths that
    /// derive masks from `cache.offset` route through `makeMask` via the
    /// `BatchPositionedKVCache` branch of `createAttentionMask(h:cache:)`.
    public override var offset: Int {
        get { _scalarOffset }
        set { _scalarOffset = newValue }
    }

    /// Initialize a BatchRotatingKVCache with a maximum size and left-padding per sequence.
    ///
    /// - Parameters:
    ///   - maxSize: The maximum cache size (sliding window size).
    ///   - leftPadding: Array of integers specifying the left-padding for each sequence.
    ///   - keep: Number of tokens to always keep at the start during rotation (default 0).
    public init(maxSize: Int, leftPadding: [Int], keep: Int = 0) {
        self.maxCacheSize = maxSize
        self.keep = keep
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.batchOffsets = MLXArray(leftPadding.map { -Int32($0) })
        super.init()
    }

    /// Internal initializer with pre-built MLXArrays.
    internal init(
        maxSize: Int, keep: Int = 0, leftPaddingArray: MLXArray, batchOffsetsArray: MLXArray
    ) {
        self.maxCacheSize = maxSize
        self.keep = keep
        self.leftPadding = leftPaddingArray
        self.batchOffsets = batchOffsetsArray
        super.init()
    }

    // MARK: - KVCache Protocol

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    // MARK: - Update

    /// Update the cache with new keys and values.
    ///
    /// Dispatches to the concat path for multi-token updates (prefill) or
    /// the in-place rotation path for single-token updates (decode).
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        // A prepared cached-prompt prefill (`_lengths != nil`) must always take the
        // concat/finalize path, even when it is one token wide — e.g. one uncached
        // token batched with a fully cached, width-1 right-padded row. `updateInPlace`
        // requires `_lengths == nil` and would otherwise crash that valid prefill.
        if keys.dim(2) == 1 && _lengths == nil {
            return updateInPlace(keys: keys, values: values)
        } else {
            return updateConcat(keys: keys, values: values)
        }
    }

    /// Multi-token concat path for prefill.
    ///
    /// Puts keys/values into temporal order, trims to maintain the sliding window,
    /// and concatenates new data.
    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            // Put keys/values in temporal order
            temporalOrder()

            // Slice off unused end
            if self.keys!.dim(2) > _idx {
                self.keys = self.keys![.ellipsis, ..<_idx, 0...]
                self.values = self.values![.ellipsis, ..<_idx, 0...]
            }

            // Roll right sequences that are padded to make sure that we don't
            // trim valid cache entries (cached-prompt prefill support)
            if let lengths = _lengths {
                let roll = MLX.maximum(MLXArray(Int32(0)), batchOffsets - lengths)
                self.keys = dynamicRoll(self.keys!, shifts: roll[0..., .newAxis], axis: 2)
                self.values = dynamicRoll(self.values!, shifts: roll[0..., .newAxis], axis: 2)
                leftPadding = leftPadding + roll
                batchOffsets = batchOffsets - roll
            }

            // The largest size is maxCacheSize + S - 1 to ensure
            // every token gets at least maxCacheSize context
            let trimSize = _idx - maxCacheSize + 1
            if trimSize > 0 {
                leftPadding = leftPadding - Int32(trimSize)
                self.keys = trim(trimSize: trimSize, self.keys!, append: keys)
                self.values = trim(trimSize: trimSize, self.values!, append: values)
            } else {
                self.keys = concatenated([self.keys!, keys], axis: 2)
                self.values = concatenated([self.values!, values], axis: 2)
            }
        }

        batchOffsets = batchOffsets + Int32(keys.dim(2))
        _scalarOffset += keys.dim(2)
        _idx = self.keys!.dim(2)

        return (self.keys!, self.values!)
    }

    /// Single-token in-place rotation path for decode.
    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        precondition(
            _lengths == nil,
            "finalize() should be called before decoding with BatchRotatingKVCache"
        )

        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let S = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = _scalarOffset

        // May not have hit the max size yet, so potentially keep growing
        if self.keys == nil
            || (prev >= self.keys!.dim(2) && self.keys!.dim(2) < maxCacheSize)
        {
            let newSize = min(step, maxCacheSize - prev)
            let kShape = [B, nKVHeads, newSize, kHeadDim]
            let vShape = [B, nKVHeads, newSize, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
            _idx = prev
        }

        // Trim if needed
        let trimSize = self.keys!.dim(2) - maxCacheSize
        if trimSize > 0 {
            self.keys = trim(trimSize: trimSize, self.keys!)
            self.values = trim(trimSize: trimSize, self.values!)
            _idx = maxCacheSize
            leftPadding = leftPadding - Int32(trimSize)
        }

        // Rotate — wrap to keep (not 0) so the first `keep` positions are never overwritten
        if _idx == maxCacheSize {
            // When keep > 0 and some sequences have left-padding, the keep zone
            // (positions 0..<keep) may contain padding zeros rather than the
            // sequence's actual keep-prefix tokens. Roll away the left-padding
            // so that each sequence's data starts at position 0, ensuring the
            // global keep zone correctly protects per-sequence keep prefixes.
            // On subsequent wraps leftPadding is already ≤ 0 so the roll is a no-op.
            if keep > 0 {
                let effectivePadding = MLX.maximum(MLXArray(Int32(0)), leftPadding)
                if effectivePadding.max().item(Int32.self) > 0 {
                    self.keys = dynamicRoll(
                        self.keys!, shifts: -effectivePadding[0..., .newAxis], axis: 2)
                    self.values = dynamicRoll(
                        self.values!, shifts: -effectivePadding[0..., .newAxis], axis: 2)
                    leftPadding = leftPadding - effectivePadding
                }
            }
            rotated = true
            _idx = keep
        }
        if rotated {
            leftPadding = leftPadding - Int32(S)
        }

        // Assign
        self.keys![.ellipsis, _idx ..< (_idx + S), 0...] = keys
        self.values![.ellipsis, _idx ..< (_idx + S), 0...] = values
        _scalarOffset += S
        batchOffsets = batchOffsets + Int32(S)
        _idx += S

        // If the buffer is not full, slice off the end
        if _scalarOffset < maxCacheSize {
            return (
                self.keys![.ellipsis, ..<_scalarOffset, 0...],
                self.values![.ellipsis, ..<_scalarOffset, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    // MARK: - Temporal Order

    /// Rearrange the cache into temporal order by unrolling rotation.
    ///
    /// When `keep > 0`, the first `keep` positions are fixed and the circular
    /// buffer operates on positions `keep..<maxCacheSize`. This mirrors the
    /// `RotatingKVCache.temporalOrder` logic.
    private func temporalOrder() {
        guard rotated else { return }
        guard let k = self.keys, let v = self.values else { return }

        let seqDim = k.dim(2)
        if _idx == seqDim {
            // idx at the end means data is already in temporal order
        } else if _idx < _scalarOffset && keep > 0 {
            // Rotated with keep prefix: [keep tokens][newer(keep..<_idx)][older(_idx..)]
            // Reorder to: [keep tokens][older(_idx..)][newer(keep..<_idx)]
            self.keys = concatenated(
                [
                    k[.ellipsis, ..<keep, 0...],
                    k[.ellipsis, _idx..., 0...],
                    k[.ellipsis, keep ..< _idx, 0...],
                ], axis: 2)
            self.values = concatenated(
                [
                    v[.ellipsis, ..<keep, 0...],
                    v[.ellipsis, _idx..., 0...],
                    v[.ellipsis, keep ..< _idx, 0...],
                ], axis: 2)
        } else if _idx < _scalarOffset {
            // Rotated without keep: simple roll
            self.keys = MLX.roll(k, shift: -_idx, axis: 2)
            self.values = MLX.roll(v, shift: -_idx, axis: 2)
        } else {
            // idx >= scalarOffset: slice off the end
            self.keys = k[.ellipsis, ..<_idx, 0...]
            self.values = v[.ellipsis, ..<_idx, 0...]
        }

        _idx = self.keys!.dim(2)
        rotated = false
    }

    // MARK: - Trim Helper

    /// Trim the oldest entries from a buffer (after keep tokens).
    ///
    /// Preserves the first `keep` positions and trims from the window portion,
    /// matching `RotatingKVCache.trim` semantics.
    private func trim(trimSize: Int, _ array: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var toCat: [MLXArray] = []
        if trimSize > 0 && keep > 0 {
            toCat = [
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, (trimSize + keep)..., 0...],
            ]
        } else if trimSize > 0 {
            toCat = [array[.ellipsis, trimSize..., 0...]]
        } else {
            toCat = [array]
        }
        if let append = append {
            toCat.append(append)
        }
        return concatenated(toCat, axis: 2)
    }

    // MARK: - State Serialization

    public override var state: [MLXArray] {
        get {
            // Mirror BatchKVCache: include the row metadata even when
            // keys/values are nil (fresh cache, or emptied by
            // filter(batchIndices: [])) so a get -> set round trip is
            // lossless instead of trapping in the setter.
            guard let keys = self.keys, let values = self.values else {
                return [batchOffsets, leftPadding]
            }
            let k: MLXArray
            let v: MLXArray
            if _scalarOffset < keys.dim(2) {
                k = keys[.ellipsis, ..<_scalarOffset, 0...]
                v = values[.ellipsis, ..<_scalarOffset, 0...]
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
            case 4:
                // Populated cache: keys, values, batchOffsets, leftPadding
                self.keys = newValue[0]
                self.values = newValue[1]
                self.batchOffsets = newValue[2]
                self.leftPadding = newValue[3]
            default:
                fatalError(
                    "BatchRotatingKVCache state must have 2 arrays (batchOffsets, "
                        + "leftPadding) or 4 arrays (keys, values, batchOffsets, leftPadding)"
                )
            }
        }
    }

    public override var metaState: [String] {
        get {
            [
                String(maxCacheSize), String(_scalarOffset), String(_idx),
                String(rotated), String(keep),
            ]
        }
        set {
            guard newValue.count == 5 else {
                fatalError("BatchRotatingKVCache metaState must have exactly 5 values")
            }
            self.maxCacheSize = Int(newValue[0]) ?? 0
            self._scalarOffset = Int(newValue[1]) ?? 0
            self._idx = Int(newValue[2]) ?? 0
            self.rotated = newValue[3] == "true"
            self.keep = Int(newValue[4]) ?? 0
        }
    }

    public override var isTrimmable: Bool {
        _scalarOffset < maxCacheSize
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        // `_scalarOffset` is the global padded progress, not the shortest row's
        // live length, and `batchOffsets[i]` is the live per-row offset (starts at
        // `-leftPadding`, advances per token written). Subtracting the full `n`
        // from every row drives a shorter row's `batchOffsets` negative and can
        // push `_idx` before its left padding, so subsequent extraction or decode
        // uses invalid slices/positions. Clamp the trim to the minimum live per-row
        // offset before applying it to all rows, and return the clamped amount.
        guard _scalarOffset > 0 else { return 0 }
        let minLiveRow = max(0, Int(batchOffsets.min().item(Int32.self)))
        let trimmed = min(n, _scalarOffset, minLiveRow)
        _scalarOffset -= trimmed
        _idx -= trimmed
        batchOffsets = batchOffsets - Int32(trimmed)
        return trimmed
    }

    // MARK: - Prepare / Finalize (Cached-Prompt Prefill)

    /// Prepare the cache for a cached-prompt batch prefill.
    ///
    /// During prefill with cached prompts of different lengths, some sequences
    /// may need right-padding to align. This method stores the state needed to
    /// roll back to left-padding on `finalize()`.
    ///
    /// Matches Python mlx-lm's `BatchRotatingKVCache.prepare()`.
    ///
    /// - Parameters:
    ///   - leftPadding: Optional additional left-padding to add (only valid on empty caches).
    ///   - lengths: Per-sequence token lengths (required when `rightPadding` is used).
    ///   - rightPadding: Per-sequence right-padding amounts. When provided,
    ///     stores `_lengths = lengths + offset` so that `finalize()` can roll
    ///     right-padded tokens back to left-padded order.
    public func prepare(
        leftPadding: [Int]? = nil, lengths: [Int]? = nil, rightPadding: [Int]? = nil
    ) {
        if let lp = leftPadding {
            precondition(
                keys == nil, "Left padding can only be added to an empty BatchRotatingKVCache")
            let lpArray = MLXArray(lp.map { Int32($0) })
            self.leftPadding = self.leftPadding + lpArray
            self.batchOffsets = self.batchOffsets - lpArray
        }

        if let rp = rightPadding, rp.contains(where: { $0 > 0 }), let lengths = lengths {
            self._lengths = MLXArray(lengths.map { Int32($0) }) + self.batchOffsets
        }
    }

    /// Finalize the cache after a cached-prompt batch prefill.
    ///
    /// If `prepare(rightPadding:lengths:)` was called, this method rolls
    /// right-padded key/value data back to left-padded order so that the
    /// cache is in the correct state for subsequent decode steps.
    ///
    /// Matches Python mlx-lm's `BatchRotatingKVCache.finalize()`.
    public override func finalize() {
        guard let lengths = _lengths else { return }
        let roll = MLX.maximum(MLXArray(Int32(0)), batchOffsets - lengths)
        if let k = keys, let v = values {
            self.keys = dynamicRoll(k, shifts: roll[0..., .newAxis], axis: 2)
            self.values = dynamicRoll(v, shifts: roll[0..., .newAxis], axis: 2)
        }
        self.leftPadding = self.leftPadding + roll
        self.batchOffsets = self.batchOffsets - roll
        self._lengths = nil
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
    public var batchOffset: MLXArray {
        batchOffsets
    }

    /// Override `BaseKVCache`'s scalar `ropeOffset` so per-row offsets are used
    /// even when this cache flows through attention layers typed as `KVCache`.
    public override var ropeOffset: RoPEOffset { .batch(batchOffset + 0) }

    // MARK: - BatchedCache Conformance

    /// Bridges the `BatchedCache` protocol onto this cache's concrete operations.
    /// `batchIndices` is small (`[B]`), so the host round-trip is negligible next
    /// to the per-token decode work.
    public func filterBatched(batchIndices: MLXArray) {
        filter(batchIndices: batchIndices.asArray(Int32.self).map(Int.init))
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? BatchRotatingKVCache else {
            preconditionFailure(
                "BatchRotatingKVCache.extendBatched requires another BatchRotatingKVCache")
        }
        extend(other: other)
    }

    public func prepareBatched(leftPadding: [Int]?, lengths: [Int]?, rightPadding: [Int]?) {
        prepare(leftPadding: leftPadding, lengths: lengths, rightPadding: rightPadding)
    }

    public func finalizeBatched() {
        finalize()
    }

    public func extractBatched(_ idx: Int) -> any KVCache {
        extract(idx: idx)
    }

    /// Sliding-window caches have no chunk-local prefill metadata to advance.
    public func advanceBatched(_: Int) {}

    public override func copy() -> any KVCache {
        let c = BatchRotatingKVCache(
            maxSize: maxCacheSize, keep: keep,
            leftPaddingArray: leftPadding[0...], batchOffsetsArray: batchOffsets[0...])
        c.keys = keys.map { $0[.ellipsis] }
        c.values = values.map { $0[.ellipsis] }
        c._idx = _idx
        c._scalarOffset = _scalarOffset
        c.rotated = rotated
        c.step = step
        c._lengths = _lengths.map { $0[0...] }
        return c
    }

    // MARK: - Batch Operations

    /// In-place filter to keep only the sequences at the given batch indices.
    ///
    /// - Parameter batchIndices: Array of batch indices to keep.
    public func filter(batchIndices: [Int]) {
        guard !batchIndices.isEmpty else {
            keys = nil
            values = nil
            leftPadding = MLXArray([Int32]())
            batchOffsets = MLXArray([Int32]())
            _idx = 0
            _scalarOffset = 0
            // Clear rotation/prefill state so a reused cache starts unrotated.
            rotated = false
            _lengths = nil
            return
        }

        let indices = MLXArray(batchIndices.map { Int32($0) })

        keys = keys?[indices]
        values = values?[indices]
        batchOffsets = batchOffsets[indices]
        leftPadding = leftPadding[indices]
        // Transient prepared-prefill lengths are per-row too: filter with the same
        // indices so a cancel between prepare and finalize doesn't roll surviving
        // rows with another row's padding (or hit a shape mismatch) in finalize.
        _lengths = _lengths?[indices]
    }

    /// In-place extend this cache with another BatchRotatingKVCache.
    ///
    /// If the rotation states differ, both caches are put into temporal order first.
    ///
    /// - Parameter other: The other BatchRotatingKVCache to merge into this one.
    ///
    /// Admission contract: the engine prefills every admitted sub-batch before
    /// extending the active batch, so a non-empty `other` always arrives with a
    /// populated KV buffer. This precondition makes that contract explicit and
    /// closes the "append empty admitted rows" hole: an `other` that carries row
    /// metadata (`batchSize > 0`) but no prefilled keys is rejected here instead
    /// of silently dropping those rows from the enlarged batch.
    public func extend(other: BatchRotatingKVCache) {
        precondition(
            other.keys != nil || other.batchSize == 0,
            "BatchRotatingKVCache.extend requires a non-empty `other` to be "
                + "prefilled (keys != nil) before extending; the engine prefills "
                + "each admitted sub-batch before calling extend."
        )
        // Rows from differently-configured windows cannot share a buffer:
        // rotation/trim math would silently corrupt whichever side's config
        // is ignored. The engine guarantees this (one factory per layer, and
        // `parametersAreBatchable` rejects per-request maxKVSize overrides),
        // so this only fires on direct misuse. Mirrors `merge`.
        precondition(
            other.maxCacheSize == maxCacheSize && other.keep == keep,
            "BatchRotatingKVCache.extend requires matching maxSize/keep "
                + "(self: \(maxCacheSize)/\(keep), other: "
                + "\(other.maxCacheSize)/\(other.keep))"
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
                    "BatchRotatingKVCache.extend cannot adopt a prefilled "
                        + "`other` into a receiver holding un-prefilled rows; "
                        + "prefill the receiver's rows first."
                )
                // self empty, other populated: adopt other's state.
                self.keys = other.keys
                self.values = other.values
                self.batchOffsets = other.batchOffsets
                self.leftPadding = other.leftPadding
                self._idx = other._idx
                self._scalarOffset = other._scalarOffset
                self.rotated = other.rotated
            }
            return
        }

        // If rotation states differ, put both in temporal order
        if self.rotated != other.rotated || self._idx != other._idx {
            self.temporalOrder()
            other.temporalOrder()
        }

        let maxIdx = max(self._idx, other._idx)
        let maxSize = max(selfKeys.dim(2), otherKeys.dim(2))

        func pad(_ cache: BatchRotatingKVCache) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
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
        self._scalarOffset = max(self._scalarOffset, other._scalarOffset)
    }

    /// Extract a single sequence from the batch as a `RotatingKVCache`.
    ///
    /// The returned cache has the left-padding stripped and contains only the
    /// valid (non-padded) key/value data. The `maxSize` is preserved.
    ///
    /// - Parameter idx: The batch index of the sequence to extract.
    /// - Returns: A `RotatingKVCache` with the extracted sequence data.
    public func extract(idx: Int) -> RotatingKVCache {
        let cache = RotatingKVCache(maxSize: maxCacheSize, keep: keep)
        let rawPadding = Int(leftPadding[idx].item(Int32.self))
        let seqOffset = Int(batchOffsets[idx].item(Int32.self))

        // After overflow (rotation), leftPadding can become negative because
        // updateInPlace decrements it each step. Clamp to non-negative for slicing:
        // the effective valid start is max(0, leftPadding).
        let padding = max(0, rawPadding)

        if let k = keys, let v = values {
            var extractedK = k[idx ..< (idx + 1)]
            var extractedV = v[idx ..< (idx + 1)]

            // If rotated, apply temporal ordering before extraction
            if rotated {
                if keep > 0 {
                    // With keep: keep prefix is fixed, only roll the window portion
                    let keepK = extractedK[.ellipsis, ..<keep, 0...]
                    let windowK = extractedK[.ellipsis, keep..., 0...]
                    let keepV = extractedV[.ellipsis, ..<keep, 0...]
                    let windowV = extractedV[.ellipsis, keep..., 0...]
                    extractedK = concatenated(
                        [keepK, MLX.roll(windowK, shift: -(self._idx - keep), axis: 2)], axis: 2)
                    extractedV = concatenated(
                        [keepV, MLX.roll(windowV, shift: -(self._idx - keep), axis: 2)], axis: 2)
                } else {
                    extractedK = MLX.roll(extractedK, shift: -_idx, axis: 2)
                    extractedV = MLX.roll(extractedV, shift: -_idx, axis: 2)
                }
                // Strip front padding only while the window is not yet full. Once
                // `seqOffset >= maxCacheSize` the row's window is logically full, so
                // keep the full `maxCacheSize` buffer: a shorter buffer with an
                // offset already at/over the window makes the extracted
                // RotatingKVCache's next `updateInPlace` compute a non-positive
                // growth (`maxSize - offset`) and write out of bounds. Residual left
                // padding is ~0 by the time a row wraps (updateInPlace decrements it
                // each step), so the kept front slots are the wrapped window, not pad.
                if seqOffset < maxCacheSize {
                    let seqEnd = maxCacheSize
                    extractedK = MLX.contiguous(extractedK[0..., 0..., padding ..< seqEnd, 0...])
                    extractedV = MLX.contiguous(extractedV[0..., 0..., padding ..< seqEnd, 0...])
                } else {
                    extractedK = MLX.contiguous(extractedK)
                    extractedV = MLX.contiguous(extractedV)
                }
            } else {
                extractedK = MLX.contiguous(extractedK[0..., 0..., padding ..< _idx, 0...])
                extractedV = MLX.contiguous(extractedV[0..., 0..., padding ..< _idx, 0...])
            }

            cache.state = [extractedK, extractedV]
            cache.offset = seqOffset
            // Restore `idx` (the RotatingKVCache ring write position) consistently
            // with the extracted state's layout. The two extraction branches above
            // produce different layouts:
            //
            //  • ROTATED: the slice keeps trailing padded slots beyond this row's
            //    `seqOffset` (a shorter/left-padded row in a globally-rotated
            //    batch), so `extractedK.dim(2)` would push the next write past
            //    those pad slots while `offset == seqOffset`, landing the next
            //    token outside the returned `..<(offset+1)` slice and losing the
            //    first post-extraction token. Clamp to the logical length
            //    (`min(seqOffset, maxCacheSize)`) and the physical tensor length so
            //    a full row restores `idx == maxCacheSize` (wraps to `keep`) while
            //    a shorter row writes at its true logical position.
            //
            //  • UNROTATED: the slice is exactly the live LINEAR temporal sequence
            //    (`padding ..< _idx`), with no trailing pad. For an oversized
            //    prompt (wider than the window, not yet trimmed) this length
            //    exceeds `maxCacheSize`. `RotatingKVCache.temporalOrder` only
            //    leaves the buffer untouched when `idx == state.dim(2)`; any
            //    smaller `idx` (e.g. a window-capped `maxCacheSize`) is treated as
            //    a wrapped ring and circularly reordered, corrupting the linear
            //    prompt on the next multi-token update. So serialize `idx` as the
            //    full physical length to mark it linear.
            let cacheIdx =
                rotated
                ? min(min(seqOffset, maxCacheSize), extractedK.dim(2))
                : extractedK.dim(2)
            cache.metaState = [
                String(keep), String(maxCacheSize), "256", String(seqOffset), String(cacheIdx),
            ]
        }

        return cache
    }

    /// Create a BatchRotatingKVCache by merging multiple individual RotatingKVCache instances.
    ///
    /// All caches must have the same `maxSize`. Shorter caches receive left-padding
    /// to match the longest sequence.
    ///
    /// - Parameter caches: An array of `RotatingKVCache` instances.
    /// - Returns: A new `BatchRotatingKVCache` containing all sequences.
    public class func merge(_ caches: [KVCache]) -> BatchRotatingKVCache {
        // Validate all caches have the same maxSize and keep
        var targetMaxSize: Int = 0
        var targetKeep: Int = -1
        for cache in caches {
            guard let rotCache = cache as? RotatingKVCache else {
                preconditionFailure(
                    "BatchRotatingKVCache.merge requires RotatingKVCache instances")
            }
            let ms = rotCache.maxSize ?? 0
            // RotatingKVCache.keep is private; read it via metaState[0] (= keep).
            let k = Int(rotCache.metaState.first ?? "0") ?? 0
            if targetMaxSize == 0 {
                targetMaxSize = ms
                targetKeep = k
            } else {
                precondition(
                    ms == targetMaxSize,
                    "BatchRotatingKVCache can only merge caches with the same maximum size"
                )
                precondition(
                    k == targetKeep,
                    "BatchRotatingKVCache can only merge caches with the same keep value"
                )
            }
        }

        // Over-window sources are not yet supported: `temporalState` is
        // oldest-first, so slicing its first `maxSize` entries below would keep
        // the OLDEST tokens and drop the newest — the opposite of sliding-window
        // semantics. Fail loudly instead of silently merging the wrong window.
        // (PR #8 review, Codex C4.)
        precondition(
            caches.allSatisfy { $0.offset <= targetMaxSize },
            "BatchRotatingKVCache.merge does not yet support caches whose offset "
                + "exceeds maxSize (wrapped/over-window sources)"
        )

        let lengths = caches.map { min($0.offset, targetMaxSize) }
        let maxLength = lengths.max() ?? 0
        let padding = lengths.map { maxLength - $0 }
        let offsets = caches.map { $0.offset }
        let B = caches.count

        // Find dimensions from first non-empty cache
        var H = 0
        var Dk = 0
        var Dv = 0
        var dt: DType = .float16

        for c in caches {
            if let rotCache = c as? RotatingKVCache {
                let temporalData = rotCache.temporalState
                if temporalData.count >= 2 {
                    let k = temporalData[0]
                    let v = temporalData[1]
                    H = k.dim(1)
                    Dk = k.dim(3)
                    Dv = v.dim(3)
                    dt = k.dtype
                    break
                }
            }
        }

        guard H > 0 else {
            return BatchRotatingKVCache(
                maxSize: targetMaxSize, leftPadding: padding, keep: max(targetKeep, 0))
        }

        let keysArr = MLXArray.zeros([B, H, maxLength, Dk], dtype: dt)
        let valuesArr = MLXArray.zeros([B, H, maxLength, Dv], dtype: dt)

        for (i, (p, c)) in zip(padding, caches).enumerated() {
            // Get temporally ordered keys/values from the RotatingKVCache.
            // Sources are guaranteed `offset <= maxSize` by the precondition
            // above, so `temporalState` is the full linear history and the
            // head slice below is exact.
            guard let rotCache = c as? RotatingKVCache else { continue }
            let temporalData = rotCache.temporalState
            if temporalData.count >= 2 {
                let k = temporalData[0]
                let v = temporalData[1]
                let seqLen = lengths[i]
                if seqLen > 0 {
                    keysArr[i ..< (i + 1), 0..., p ..< (p + seqLen), 0...] =
                        k[.ellipsis, ..<seqLen, 0...]
                    valuesArr[i ..< (i + 1), 0..., p ..< (p + seqLen), 0...] =
                        v[.ellipsis, ..<seqLen, 0...]
                }
            }
        }

        let cache = BatchRotatingKVCache(
            maxSize: targetMaxSize, leftPadding: padding, keep: max(targetKeep, 0))
        cache.keys = keysArr
        cache.values = valuesArr
        cache.batchOffsets = MLXArray(offsets.map { Int32($0) })
        cache._idx = maxLength
        cache._scalarOffset = maxLength

        return cache
    }

    /// Create a batch-1 BatchRotatingKVCache from a single RotatingKVCache.
    ///
    /// - Parameter cache: A single `RotatingKVCache` to wrap.
    /// - Returns: A new `BatchRotatingKVCache` with batch size 1.
    public class func fromSingle(_ cache: RotatingKVCache) -> BatchRotatingKVCache {
        let ms = cache.maxSize ?? 0
        // RotatingKVCache.keep is private; read it via metaState[0] (= keep).
        let k = Int(cache.metaState.first ?? "0") ?? 0
        let batchCache = BatchRotatingKVCache(maxSize: ms, leftPadding: [0], keep: k)

        let temporalData = cache.temporalState
        if temporalData.count >= 2 {
            let seqLen = min(cache.offset, ms)
            // `temporalState` is oldest-first and, for an over-window source
            // (offset > maxSize), can be longer than the window. Keep the newest
            // `seqLen` tokens (the live sliding window) rather than the oldest —
            // otherwise later state/extraction would expose stale prompt tokens.
            let total = temporalData[0].dim(2)
            if total > seqLen {
                batchCache.keys = temporalData[0][.ellipsis, (total - seqLen)..., 0...]
                batchCache.values = temporalData[1][.ellipsis, (total - seqLen)..., 0...]
            } else {
                batchCache.keys = temporalData[0]
                batchCache.values = temporalData[1]
            }
            batchCache._idx = seqLen
            // `_scalarOffset` holds the *absolute* offset (the `offset` getter caps
            // it at `maxCacheSize`); capping it here would desync from `batchOffsets`
            // after `trim()` when the source cache has wrapped (offset > maxSize).
            batchCache._scalarOffset = cache.offset
            batchCache.batchOffsets = MLXArray([Int32(cache.offset)])
        }

        return batchCache
    }

    /// Convert a batch-1 BatchRotatingKVCache back to a RotatingKVCache.
    ///
    /// - Returns: A `RotatingKVCache` with the single sequence data.
    public func toSingle() -> RotatingKVCache {
        precondition(batchSize == 1, "toSingle() requires batch size of 1")
        return extract(idx: 0)
    }

    // MARK: - Mask Creation

    /// Create an attention mask for this batch rotating cache.
    ///
    /// Accounts for both the sliding window size and left-padding. During
    /// rotation, the mask is rolled to match the rotated buffer layout.
    ///
    /// - Parameters:
    ///   - n: The sequence length for the new tokens
    ///   - windowSize: Optional sliding window size (defaults to maxSize)
    ///   - returnArray: Force return of array mask instead of symbolic
    /// - Returns: Attention mask mode for scaled dot product attention
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        var effectiveLeftPadding = self.leftPadding
        let effectiveWindowSize = windowSize ?? maxCacheSize
        let cappedOffset = min(maxCacheSize - 1, _scalarOffset)

        let rinds = MLXArray(Int32(0) ..< Int32(cappedOffset + n))
        var linds =
            cappedOffset != 0
            ? MLXArray(Int32(cappedOffset) ..< Int32(cappedOffset + n))
            : rinds
        linds = linds[0..., .newAxis]
        let rindsRow = rinds[.newAxis]

        // Causal mask: query can attend to keys at or before its position
        var mask = linds .>= rindsRow

        // Window mask: restrict attention to the window
        mask = mask & (linds .< rindsRow + Int32(effectiveWindowSize))

        // Adjust left_padding for the trim that `updateConcat` will apply. This
        // must mirror that trim exactly, but `makeMask` runs BEFORE the update so
        // the span we compute it from depends on the cache's rotation state. Two
        // distinct over-window cases must be handled separately:
        //
        //  1. RING-ROTATED (`rotated == true`): `_idx` is the circular write
        //     pointer (reset to `keep` on wrap), much smaller than the logical
        //     span. `updateConcat` calls `temporalOrder()` first, which restores
        //     `_idx` to the temporal length (== maxCacheSize once wrapped) before
        //     trimming. So mirror that with the LOGICAL window length
        //     `min(_scalarOffset, maxCacheSize)` (== maxCacheSize when wrapped).
        //     Capping here is correct: the ring already holds only one window.
        //
        //  2. UNROTATED OVERSIZED (`rotated == false`, `_idx >= maxCacheSize`): an
        //     initial prompt WIDER than the window that has not been trimmed yet.
        //     `temporalOrder()` is a no-op (it guards on `rotated`), so the trim
        //     uses the full PHYSICAL `_idx`, dropping `_idx - maxCacheSize + 1`
        //     old slots. The pad mask must be reduced by that full pending trim —
        //     NOT capped to the window — or a heavily left-padded short row (e.g.
        //     leftPadding 99 against a 16-col window) masks out every retained
        //     valid key. Use the physical `_idx` uncapped here.
        let spanForTrim = rotated ? min(_scalarOffset, maxCacheSize) : _idx
        // A prepared cache (`_lengths != nil`) routes even one-token updates
        // through `updateConcat` (see `update`), which trims one slot exactly
        // like a multi-token concat -- so the concat trim term applies to it
        // regardless of `n`.
        let trimSize = spanForTrim - maxCacheSize + (n > 1 || _lengths != nil ? 1 : 0)
        if trimSize > 0 {
            effectiveLeftPadding = effectiveLeftPadding - Int32(trimSize)
        }

        // Rotated single-token DECODE: the in-place update writes into the
        // ring slot, so the mask columns must be rolled to the ring layout.
        // A prepared (`_lengths != nil`) one-token update is NOT a decode --
        // it takes the concat path and returns temporally ordered K/V, so
        // rolling the mask here would misalign it against those keys.
        let isRotated = n == 1 && _lengths == nil && (rotated || _idx >= maxCacheSize)
        if isRotated {
            effectiveLeftPadding = effectiveLeftPadding - Int32(1)
        }

        // Apply left-padding mask
        let lp = effectiveLeftPadding[0..., .newAxis, .newAxis, .newAxis]
        mask = mask & (rindsRow .>= lp)

        // Roll mask for rotated buffer, accounting for keep prefix
        if isRotated {
            var currentIdx = _idx
            if currentIdx >= maxCacheSize {
                currentIdx = keep
            }
            if keep > 0 {
                // With keep: only roll the window portion (positions keep..<maxCacheSize),
                // leaving the first `keep` positions of the mask fixed.
                let keepMask = mask[.ellipsis, ..<keep]
                let windowMask = mask[.ellipsis, keep...]
                let rolledWindow = MLX.roll(
                    windowMask, shift: currentIdx - keep + 1, axis: -1)
                mask = concatenated([keepMask, rolledWindow], axis: -1)
            } else {
                mask = MLX.roll(mask, shift: currentIdx + 1, axis: -1)
            }
        }

        return .array(mask)
    }

    public var debugDescription: String {
        "BatchRotatingKVCache batchSize: \(batchSize), maxSize: \(maxCacheSize), keep: \(keep), _idx: \(_idx), _offset: \(_scalarOffset), rotated: \(rotated), keys: \(keys?.shape.description ?? "-")"
    }
}
