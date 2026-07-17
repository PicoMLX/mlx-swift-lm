// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - BatchQuantizedKVCache

/// Batch-aware quantized KV cache with left-padding strategy for continuous
/// batching.
///
/// Combines ``BatchKVCache``'s left-padding/batch bookkeeping with
/// ``QuantizedKVCache``'s quantized storage: keys/values are stored as
/// quantized `(weight, scales, biases)` triples and consumed through
/// ``quantizedScaledDotProductAttention`` via the
/// ``QuantizedKVCacheProtocol`` dispatch in `attentionWithCacheUpdate`.
///
/// Like ``BatchKVCache``, inputs are expected to be left-padded so that
/// variable-length sequences align on the right, with `leftPadding` recording
/// the per-sequence pad and `batchOffsets` the per-sequence RoPE position.
///
/// All row operations (filter / extend / extract / roll) act on the sequence
/// axis (`-2` of each triple component) or the batch axis (`0`), never the
/// packed quantization axis (`-1`), so they are exact on quantized storage.
public class BatchQuantizedKVCache: BaseKVCache, QuantizedKVCacheProtocol,
    BatchPositionedKVCache, BatchedCache
{

    /// Per-sequence left-padding amounts as an MLXArray of shape `[B]`.
    public internal(set) var leftPadding: MLXArray

    /// Per-sequence offset as an MLXArray of shape `[B]`.
    /// Starts negative (equal to `-leftPadding`) and advances with each update.
    public internal(set) var batchOffsets: MLXArray

    /// Internal buffer index tracking how far into the buffers we've written.
    internal var _idx: Int = 0

    /// Quantized keys: `(weight, scales, biases)`, each `[B, H, S_buf, *]`.
    internal var keys: (MLXArray, MLXArray, MLXArray?)?

    /// Quantized values: `(weight, scales, biases)`, each `[B, H, S_buf, *]`.
    internal var values: (MLXArray, MLXArray, MLXArray?)?

    /// Step size for buffer allocation (grow in chunks of this size).
    public var step: Int = 256

    public private(set) var groupSize: Int
    public private(set) var bits: Int
    public let mode: QuantizationMode

    /// The scalar offset (not meaningful for batch caches, returns `_idx`).
    public override var offset: Int {
        get { _idx }
        set { _idx = newValue }
    }

    /// Initialize a BatchQuantizedKVCache with the given left-padding per sequence.
    public init(
        leftPadding: [Int], groupSize: Int = 64, bits: Int = 8,
        mode: QuantizationMode = .affine
    ) {
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.batchOffsets = MLXArray(leftPadding.map { -Int32($0) })
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        super.init()
    }

    /// Internal initializer with pre-built MLXArrays.
    internal init(
        leftPaddingArray: MLXArray, batchOffsetsArray: MLXArray,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) {
        self.leftPadding = leftPaddingArray
        self.batchOffsets = batchOffsetsArray
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        super.init()
    }

    // MARK: - Triple helpers

    private static func mapTriple(
        _ transform: (MLXArray) -> MLXArray, _ tuple: (MLXArray, MLXArray, MLXArray?)
    ) -> (MLXArray, MLXArray, MLXArray?) {
        (transform(tuple.0), transform(tuple.1), tuple.2.map(transform))
    }

    private static func flatten(_ tuple: (MLXArray, MLXArray, MLXArray?)) -> [MLXArray] {
        [tuple.0, tuple.1, tuple.2].compactMap { $0 }
    }

    /// Quantize a zeros buffer of the given logical shape, yielding empty
    /// quantized storage with the right packed trailing dimensions.
    private func initQuant(dim: Int, shape: [Int], dtype: DType) -> (MLXArray, MLXArray, MLXArray?)
    {
        let temp = MLXArray.zeros(shape + [dim], dtype: dtype)
        let q = quantized(temp, groupSize: groupSize, bits: bits)
        return (q.wq, q.scales, q.biases)
    }

    /// Append `extra` zero-rows along the sequence axis of every component.
    private static func expandQuant(
        _ tuple: (MLXArray, MLXArray, MLXArray?), extra: Int
    ) -> (MLXArray, MLXArray, MLXArray?) {
        mapTriple(
            { array in
                var shape = array.shape
                shape[shape.count - 2] = extra
                let zeros = MLXArray.zeros(shape, dtype: array.dtype)
                return concatenated([array, zeros], axis: -2)
            }, tuple)
    }

    // MARK: - KVCache Protocol

    public override func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let keys { arrays.append(contentsOf: Self.flatten(keys)) }
        if let values { arrays.append(contentsOf: Self.flatten(values)) }
        return arrays
    }

    /// Required by ``KVCache`` but not usable on a quantized cache. Use
    /// `updateQuantized` (which `attentionWithCacheUpdate` dispatches to via
    /// ``QuantizedKVCacheProtocol``).
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("`update` was called on `BatchQuantizedKVCache`. Use `updateQuantized` instead.")
    }

    /// Update the cache with new (unquantized) keys/values and return the full
    /// quantized state, trimmed to the written length.
    ///
    /// Keys/values have shape `[B, H, S, D]`. The buffers grow in steps of
    /// `step` along the sequence axis, exactly like ``BatchKVCache``.
    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let B = keys.dim(0)
        let kvHeads = keys.dim(1)
        let S = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = _idx

        let effectiveGroupSize = resolvedKVQuantizationGroupSize(
            requested: groupSize, keyHeadDim: kHeadDim, valueHeadDim: vHeadDim)
        if let effectiveGroupSize, effectiveGroupSize != groupSize,
            self.keys == nil, self.values == nil, _idx == 0
        {
            self.groupSize = effectiveGroupSize
        }
        guard effectiveGroupSize != nil else {
            fatalError(
                "KV cache quantization requires head dimensions divisible by one of the supported group sizes (32, 64, 128). Requested group size: \(groupSize). Key head dim: \(kHeadDim). Value head dim: \(vHeadDim)."
            )
        }

        // Grow if needed.
        if self.keys == nil || (prev + S) > self.keys!.0.dim(-2) {
            let needed = prev + S
            let target = ((needed + step - 1) / step) * step
            if var currentKeys = self.keys, var currentValues = self.values {
                let width = currentKeys.0.dim(-2)
                if prev < width {
                    currentKeys = Self.mapTriple({ $0[.ellipsis, ..<prev, 0...] }, currentKeys)
                    currentValues = Self.mapTriple({ $0[.ellipsis, ..<prev, 0...] }, currentValues)
                }
                self.keys = Self.expandQuant(currentKeys, extra: target - prev)
                self.values = Self.expandQuant(currentValues, extra: target - prev)
            } else {
                let shape = [B, kvHeads, target]
                self.keys = initQuant(dim: kHeadDim, shape: shape, dtype: keys.dtype)
                self.values = initQuant(dim: vHeadDim, shape: shape, dtype: values.dtype)
            }
        }

        batchOffsets = batchOffsets + Int32(S)
        _idx += S

        let qKeys = quantized(keys, groupSize: groupSize, bits: bits)
        let qValues = quantized(values, groupSize: groupSize, bits: bits)

        guard let currentKeys = self.keys, let currentValues = self.values else {
            fatalError("BatchQuantizedKVCache not properly initialized")
        }

        currentKeys.0[.ellipsis, prev ..< _idx, 0...] = qKeys.wq
        currentKeys.1[.ellipsis, prev ..< _idx, 0...] = qKeys.scales
        if let biases = qKeys.biases {
            currentKeys.2?[.ellipsis, prev ..< _idx, 0...] = biases
        }
        currentValues.0[.ellipsis, prev ..< _idx, 0...] = qValues.wq
        currentValues.1[.ellipsis, prev ..< _idx, 0...] = qValues.scales
        if let biases = qValues.biases {
            currentValues.2?[.ellipsis, prev ..< _idx, 0...] = biases
        }

        let idx = _idx
        return (
            Self.mapTriple({ $0[.ellipsis, ..<idx, 0...] }, currentKeys),
            Self.mapTriple({ $0[.ellipsis, ..<idx, 0...] }, currentValues)
        )
    }

    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        guard let keys, let values else { return nil }
        let idx = _idx
        return (
            Self.mapTriple({ $0[.ellipsis, ..<idx, 0...] }, keys),
            Self.mapTriple({ $0[.ellipsis, ..<idx, 0...] }, values)
        )
    }

    // MARK: - State Serialization

    /// Empty: `[batchOffsets, leftPadding]` (2). Populated: key triple +
    /// value triple + `[batchOffsets, leftPadding]` (6 without biases, 8 with).
    public override var state: [MLXArray] {
        get {
            guard let keys, let values else {
                return [batchOffsets, leftPadding]
            }
            let idx = _idx
            let trim: (MLXArray) -> MLXArray =
                keys.0.dim(-2) > idx ? { $0[.ellipsis, ..<idx, 0...] } : { $0 }
            return Self.flatten(Self.mapTriple(trim, keys))
                + Self.flatten(Self.mapTriple(trim, values)) + [batchOffsets, leftPadding]
        }
        set {
            switch newValue.count {
            case 2:
                self.keys = nil
                self.values = nil
                self.batchOffsets = newValue[0]
                self.leftPadding = newValue[1]
                self._idx = 0
            case 6:
                self.keys = (newValue[0], newValue[1], nil)
                self.values = (newValue[2], newValue[3], nil)
                self.batchOffsets = newValue[4]
                self.leftPadding = newValue[5]
                self._idx = newValue[0].dim(-2)
            case 8:
                self.keys = (newValue[0], newValue[1], newValue[2])
                self.values = (newValue[3], newValue[4], newValue[5])
                self.batchOffsets = newValue[6]
                self.leftPadding = newValue[7]
                self._idx = newValue[0].dim(-2)
            default:
                fatalError(
                    "BatchQuantizedKVCache state must have 2 (empty), 6 (no biases) or 8 arrays")
            }
        }
    }

    public override var metaState: [String] {
        get { [String(_idx), String(step), String(groupSize), String(bits)] }
        set {
            guard newValue.count == 4 else {
                fatalError("BatchQuantizedKVCache metaState must have exactly 4 values")
            }
            self._idx = Int(newValue[0]) ?? 0
            self.step = Int(newValue[1]) ?? 256
            self.groupSize = Int(newValue[2]) ?? 64
            self.bits = Int(newValue[3]) ?? 8
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(_idx, n)
        _idx -= trimmed
        batchOffsets = batchOffsets - Int32(trimmed)
        return trimmed
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

    public var batchOffset: MLXArray {
        batchOffsets
    }

    /// Override `BaseKVCache`'s scalar `ropeOffset` so per-row offsets are used
    /// even when this cache flows through attention layers typed as `KVCache`.
    public override var ropeOffset: RoPEOffset { .batch(batchOffset + 0) }

    // MARK: - Mask Creation

    /// Batch caches always need an explicit mask so left-padded positions stay
    /// excluded, even for `n == 1` decode steps. Same contract as
    /// ``BatchKVCache/makeMask(n:windowSize:returnArray:)``; consumed by
    /// `quantizedScaledDotProductAttention`'s boolean-array mask path.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .array(
            createCausalMask(
                n: n, offset: _idx, windowSize: windowSize, leftPadding: leftPadding
            )
        )
    }

    // MARK: - Batch Operations

    /// In-place filter to keep only the sequences at the given batch indices.
    /// After filtering, the minimum left-padding is subtracted from all rows
    /// and the buffers are trimmed accordingly (shift left to reduce padding).
    public func filter(batchIndices: [Int]) {
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

        keys = keys.map { triple in Self.mapTriple({ $0[indices] }, triple) }
        values = values.map { triple in Self.mapTriple({ $0[indices] }, triple) }
        batchOffsets = batchOffsets[indices]
        leftPadding = leftPadding[indices]
        // Transient right-padding metadata (set by prepareBatched, consumed by
        // finalize) is per-row too: filter it with the same indices so a cancel
        // between prepare and finalize doesn't leave finalize rolling/subtracting
        // against the pre-filter batch shape. Same as `BatchKVCache.filter`.
        _rightPadding = _rightPadding?[indices]

        // Shift left to reduce padding. Skipped while `keys == nil` for the
        // same reason as `BatchKVCache.filter`: decrementing `_idx` on an
        // unpopulated cache would corrupt the next prefill's write range.
        // Also require `_idx >= minLeftPad`: during chunked prefill the KV
        // buffer can exist before the write pointer has advanced past the
        // padding, and shifting then would drive `_idx` negative and slice
        // past the written region.
        let minLeftPad = leftPadding.min().item(Int32.self)
        if minLeftPad > 0, keys != nil, _idx >= Int(minLeftPad) {
            let padInt = Int(minLeftPad)
            keys = keys.map { triple in
                Self.mapTriple({ $0[.ellipsis, padInt..., 0...] }, triple)
            }
            values = values.map { triple in
                Self.mapTriple({ $0[.ellipsis, padInt..., 0...] }, triple)
            }
            _idx -= padInt
            leftPadding = leftPadding - minLeftPad
        }
    }

    /// In-place extend this cache with another BatchQuantizedKVCache.
    /// The caches are right-justified: the shorter cache gets additional
    /// left-padding to align with the longer one along the sequence axis.
    ///
    /// Admission contract: same as ``BatchKVCache/extend(other:)`` — a
    /// non-empty `other` must arrive prefilled.
    public func extend(other: BatchQuantizedKVCache) {
        // `bits`/`mode` are fixed at construction and never resolved, so they
        // must always match. `groupSize`, however, is resolved lazily on the
        // first `updateQuantized`: an empty receiver still holds its unresolved
        // *requested* value, so a strict `groupSize` equality check would trap
        // when admitting the first prefilled sub-batch. Only enforce it once the
        // receiver is populated; an empty receiver adopts `other`'s resolved
        // `groupSize` in the adoption branch below.
        precondition(
            bits == other.bits && mode == other.mode,
            "BatchQuantizedKVCache.extend requires matching quantization parameters"
        )
        precondition(
            keys == nil || groupSize == other.groupSize,
            "BatchQuantizedKVCache.extend requires matching quantization parameters"
        )
        precondition(
            other.keys != nil || other.batchSize == 0,
            "BatchQuantizedKVCache.extend requires a non-empty `other` to be prefilled "
                + "(keys != nil) before extending"
        )
        guard let selfKeys = self.keys, let otherKeys = other.keys else {
            if self.keys == nil && other.keys == nil {
                self.leftPadding = concatenated([self.leftPadding, other.leftPadding], axis: 0)
                self.batchOffsets = concatenated([self.batchOffsets, other.batchOffsets], axis: 0)
            } else if other.keys != nil {
                self.keys = other.keys
                self.values = other.values
                self.batchOffsets = other.batchOffsets
                self.leftPadding = other.leftPadding
                self._idx = other._idx
                // Adopt `other`'s resolved `groupSize`: an empty receiver's was
                // still the unresolved requested value. (`bits`/`mode` are fixed
                // at construction and already validated to match above.)
                self.groupSize = other.groupSize
            }
            return
        }

        let maxIdx = max(self._idx, other._idx)
        let maxSize = max(selfKeys.0.dim(-2), otherKeys.0.dim(-2))

        func pad(
            _ cache: BatchQuantizedKVCache
        ) -> ((MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?), MLXArray, MLXArray)
        {
            let left = maxIdx - cache._idx
            var right = maxSize - cache.keys!.0.dim(-2) - left

            var k = cache.keys!
            var v = cache.values!

            if right < 0 {
                let cut = right
                k = Self.mapTriple({ $0[.ellipsis, ..<($0.dim(-2) + cut), 0...] }, k)
                v = Self.mapTriple({ $0[.ellipsis, ..<($0.dim(-2) + cut), 0...] }, v)
                right = 0
            }

            if left != 0 || right != 0 {
                // Pad along the sequence axis (-2). Component ranks match, so a
                // shared width spec is valid; padded zeros sit in masked
                // positions and are never attended.
                let padWidths: [IntOrPair] = [0, 0, .init((left, right)), 0]
                k = Self.mapTriple({ MLX.padded($0, widths: padWidths) }, k)
                v = Self.mapTriple({ MLX.padded($0, widths: padWidths) }, v)
            }

            let adjustedLeftPadding = cache.leftPadding + Int32(left)

            return (k, v, cache.batchOffsets, adjustedLeftPadding)
        }

        let (selfK, selfV, selfOff, selfLP) = pad(self)
        let (otherK, otherV, otherOff, otherLP) = pad(other)

        func concatTriple(
            _ a: (MLXArray, MLXArray, MLXArray?), _ b: (MLXArray, MLXArray, MLXArray?)
        ) -> (MLXArray, MLXArray, MLXArray?) {
            let biases: MLXArray?
            if let ab = a.2, let bb = b.2 {
                biases = concatenated([ab, bb], axis: 0)
            } else {
                biases = nil
            }
            return (
                concatenated([a.0, b.0], axis: 0),
                concatenated([a.1, b.1], axis: 0),
                biases
            )
        }

        self.keys = concatTriple(selfK, otherK)
        self.values = concatTriple(selfV, otherV)
        self.batchOffsets = concatenated([selfOff, otherOff], axis: 0)
        self.leftPadding = concatenated([selfLP, otherLP], axis: 0)
        self._idx = maxIdx
    }

    /// Extract a single sequence from the batch as a `QuantizedKVCache` with
    /// the left-padding stripped.
    public func extract(idx: Int) -> QuantizedKVCache {
        let cache = QuantizedKVCache(groupSize: groupSize, bits: bits, mode: mode)
        let padding = Int(leftPadding[idx].item(Int32.self))

        if let k = keys, let v = values {
            let slice: (MLXArray) -> MLXArray = {
                MLX.contiguous($0[idx ..< (idx + 1), 0..., padding ..< self._idx, 0...])
            }
            let extractedK = Self.mapTriple(slice, k)
            let extractedV = Self.mapTriple(slice, v)
            cache.state = Self.flatten(extractedK) + Self.flatten(extractedV)
            cache.metaState = [
                String(step), String(_idx - padding), String(groupSize), String(bits),
            ]
        }

        return cache
    }

    /// Create a batch-1 BatchQuantizedKVCache from a single QuantizedKVCache.
    /// The resulting cache has `leftPadding = [0]` and identical data.
    public class func fromSingle(_ cache: QuantizedKVCache) -> BatchQuantizedKVCache {
        let batchCache = BatchQuantizedKVCache(
            leftPadding: [0], groupSize: cache.groupSize, bits: cache.bits, mode: cache.mode)

        if let (k, v) = cache.getQuantizedState() {
            batchCache.keys = k
            batchCache.values = v
            batchCache._idx = cache.offset
            batchCache.batchOffsets = MLXArray([Int32(cache.offset)])
        }

        return batchCache
    }

    /// Convert a batch-1 BatchQuantizedKVCache back to a QuantizedKVCache.
    public func toSingle() -> QuantizedKVCache {
        precondition(batchSize == 1, "toSingle() requires batch size of 1")
        return extract(idx: 0)
    }

    // MARK: - Prepare / Finalize (Cached-Prompt Prefill)

    /// Stored right-padding for the current prefill cycle.
    internal var _rightPadding: MLXArray?

    /// Record right-padding before a ragged prefill so `finalize()` can roll
    /// it to the left side. Same contract as ``BatchKVCache/prepare(rightPadding:)``.
    public func prepare(rightPadding: MLXArray) {
        if rightPadding.max().item(Int32.self) > 0 {
            _rightPadding = rightPadding
        }
    }

    /// Roll right-padding zeros to the left side of the buffers and convert
    /// them to left-padding. The roll permutes whole positions along the
    /// sequence axis, so quantized rows stay bit-exact.
    public override func finalize() {
        guard let padding = _rightPadding else { return }

        if let k = keys, let v = values {
            let shifts = padding[0..., .newAxis]
            self.keys = Self.mapTriple({ dynamicRoll($0, shifts: shifts, axis: -2) }, k)
            self.values = Self.mapTriple({ dynamicRoll($0, shifts: shifts, axis: -2) }, v)
        }
        batchOffsets = batchOffsets - padding
        leftPadding = leftPadding + padding
        _rightPadding = nil
    }

    // MARK: - BatchedCache Conformance

    public func filterBatched(batchIndices: MLXArray) {
        filter(batchIndices: batchIndices.asArray(Int32.self).map(Int.init))
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? BatchQuantizedKVCache else {
            preconditionFailure(
                "BatchQuantizedKVCache.extendBatched requires another BatchQuantizedKVCache")
        }
        extend(other: other)
    }

    public func prepareBatched(leftPadding: [Int]?, lengths _: [Int]?, rightPadding: [Int]?) {
        if let leftPadding {
            precondition(
                keys == nil,
                "prepareBatched(leftPadding:) requires an empty BatchQuantizedKVCache"
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
        let c = BatchQuantizedKVCache(
            leftPaddingArray: leftPadding[0...], batchOffsetsArray: batchOffsets[0...],
            groupSize: groupSize, bits: bits, mode: mode)
        c.keys = keys.map { triple in Self.mapTriple({ $0[.ellipsis] }, triple) }
        c.values = values.map { triple in Self.mapTriple({ $0[.ellipsis] }, triple) }
        c._idx = _idx
        c.step = step
        c._rightPadding = _rightPadding.map { $0[0...] }
        return c
    }

    public var debugDescription: String {
        "BatchQuantizedKVCache batchSize: \(batchSize), _idx: \(_idx), groupSize: \(groupSize), bits: \(bits), keys: \(keys?.0.shape.description ?? "-")"
    }
}
