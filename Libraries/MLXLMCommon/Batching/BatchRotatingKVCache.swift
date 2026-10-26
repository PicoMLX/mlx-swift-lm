// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// A sliding-window KV cache for a batch of left-padded sequences.
///
/// Port of mlx-lm's `BatchRotatingKVCache`: the left padding of ``BatchKVCache`` with
/// the window of ``RotatingKVCache``. Every row keeps its last ``maxSize`` positions.
/// Unlike ``RotatingKVCache`` it has no pinned prefix (`keep`) and no rewind reserve.
public final class BatchRotatingKVCache: BaseKVCache, BatchPositionedKVCache, BatchedCache {
    /// Each row's left padding, shape `[B]`. It goes negative once the window wraps.
    public private(set) var leftPadding: MLXArray

    /// Each row's position, shape `[B]`. It starts at `-leftPadding`.
    public private(set) var batchOffset: MLXArray

    var keys: MLXArray?
    var values: MLXArray?

    private var window: Int
    /// The buffer position the next token is written to.
    private var idx = 0
    /// The tokens written to the buffer so far.
    private var written = 0
    /// Whether writes have wrapped around the buffer.
    private var rotated = false
    /// Each row's end during a ragged prefill, from `prepare` until `finalize`.
    private var lengths: MLXArray?

    public var step = 256

    /// Whether the window came from the model or from a requested capacity.
    package var capacityOrigin = RotatingKVCache.CapacityOrigin.modelNative

    public override var maxSize: Int? { window }

    /// The positions in use, at most the window. Reads like ``RotatingKVCache`` would
    /// for a mask built from the offset.
    public override var offset: Int {
        get { min(written, window) }
        set { written = newValue }
    }

    /// - Parameters:
    ///   - maxSize: the window
    ///   - leftPadding: the left padding of each row
    public init(maxSize: Int, leftPadding: [Int]) {
        self.window = maxSize
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.batchOffset = MLXArray(leftPadding.map { -Int32($0) })
        super.init()
    }

    /// The number of rows.
    public var batchSize: Int { leftPadding.dim(0) }

    /// Per-row positions, also when the cache is typed as `KVCache`.
    public override var ropeOffset: RoPEOffset { .batch(batchOffset + 0) }

    public override func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    /// Writes one decode token in place, and everything else (prefill, ragged steps)
    /// by concatenation.
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if keys.dim(2) == 1 && lengths == nil {
            return updateInPlace(keys: keys, values: values)
        }
        return updateConcat(keys: keys, values: values)
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            temporalOrder()
            if self.keys!.dim(2) > idx {
                self.keys = self.keys![.ellipsis, ..<idx, 0...]
                self.values = self.values![.ellipsis, ..<idx, 0...]
            }

            // Roll right-padded rows so the trim below keeps their tokens.
            if let lengths {
                let roll = MLX.maximum(MLXArray(Int32(0)), batchOffset - lengths)
                self.keys = dynamicRoll(self.keys!, shifts: roll[0..., .newAxis], axis: 2)
                self.values = dynamicRoll(self.values!, shifts: roll[0..., .newAxis], axis: 2)
                leftPadding = leftPadding + roll
                batchOffset = batchOffset - roll
            }

            // Keep `window - 1` old positions, so every new token sees a full window.
            let trimSize = idx - window + 1
            if trimSize > 0 {
                leftPadding = leftPadding - Int32(trimSize)
                self.keys = concatenated([self.keys![.ellipsis, trimSize..., 0...], keys], axis: 2)
                self.values = concatenated(
                    [self.values![.ellipsis, trimSize..., 0...], values], axis: 2)
            } else {
                self.keys = concatenated([self.keys!, keys], axis: 2)
                self.values = concatenated([self.values!, values], axis: 2)
            }
        }

        batchOffset = batchOffset + Int32(keys.dim(2))
        written += keys.dim(2)
        idx = self.keys!.dim(2)
        return (self.keys!, self.values!)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let count = keys.dim(2)
        let previous = written

        // Grow the buffer until it holds the window.
        if self.keys == nil || (previous >= self.keys!.dim(2) && self.keys!.dim(2) < window) {
            let size = min(step, window - previous)
            let newKeys = MLXArray.zeros(
                [keys.dim(0), keys.dim(1), size, keys.dim(3)], dtype: keys.dtype)
            let newValues = MLXArray.zeros(
                [values.dim(0), values.dim(1), size, values.dim(3)], dtype: values.dtype)
            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, newKeys], axis: 2)
                self.values = concatenated([currentValues, newValues], axis: 2)
            } else {
                self.keys = newKeys
                self.values = newValues
            }
            idx = previous
        }

        // A buffer longer than the window drops its oldest positions.
        let trimSize = self.keys!.dim(2) - window
        if trimSize > 0 {
            self.keys = self.keys![.ellipsis, trimSize..., 0...]
            self.values = self.values![.ellipsis, trimSize..., 0...]
            idx = window
            leftPadding = leftPadding - Int32(trimSize)
        }

        if idx == window {
            rotated = true
            idx = 0
        }
        if rotated {
            leftPadding = leftPadding - Int32(count)
        }

        self.keys![.ellipsis, idx ..< idx + count, 0...] = keys
        self.values![.ellipsis, idx ..< idx + count, 0...] = values
        written += count
        batchOffset = batchOffset + Int32(count)
        idx += count

        if written < window {
            return (
                self.keys![.ellipsis, ..<written, 0...], self.values![.ellipsis, ..<written, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    /// Puts a wrapped buffer back in temporal order.
    private func temporalOrder() {
        guard rotated, let keys, let values else { return }
        if idx < keys.dim(2) {
            if idx < written {
                self.keys = MLX.roll(keys, shift: -idx, axis: 2)
                self.values = MLX.roll(values, shift: -idx, axis: 2)
            } else {
                self.keys = keys[.ellipsis, ..<idx, 0...]
                self.values = values[.ellipsis, ..<idx, 0...]
            }
        }
        idx = self.keys!.dim(2)
        rotated = false
    }

    public override var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            if written < keys.dim(2) {
                return [
                    keys[.ellipsis, ..<written, 0...], values[.ellipsis, ..<written, 0...],
                    batchOffset, leftPadding,
                ]
            }
            return [keys, values, batchOffset, leftPadding]
        }
        set {
            guard newValue.count == 4 else {
                fatalError("BatchRotatingKVCache state must have 4 arrays, got \(newValue.count)")
            }
            keys = newValue[0]
            values = newValue[1]
            batchOffset = newValue[2]
            leftPadding = newValue[3]
        }
    }

    /// `[maxSize, written, idx, rotated]`, as in mlx-lm.
    public override var metaState: [String] {
        get { [String(window), String(written), String(idx), String(rotated)] }
        set {
            guard newValue.count == 4, let window = Int(newValue[0]),
                let written = Int(newValue[1]), let idx = Int(newValue[2]),
                let rotated = Bool(newValue[3])
            else {
                fatalError("Invalid BatchRotatingKVCache metaState \(newValue)")
            }
            self.window = window
            self.written = written
            self.idx = idx
            self.rotated = rotated
        }
    }

    public override var isTrimmable: Bool { isTrimmable(after: 0) }

    public override func isTrimmable(after positions: Int) -> Bool {
        written + positions < window
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        // No row can lose more tokens than the shortest row holds.
        let shortest = batchSize > 0 ? Int(batchOffset.min().item(Int32.self)) : 0
        let trimmed = min(written, max(0, shortest), n)
        written -= trimmed
        idx -= trimmed
        batchOffset = batchOffset - Int32(trimmed)
        return trimmed
    }

    /// Prepares a right-padded prefill of rows with the given lengths.
    public override func prepare(lengths: [Int]?) {
        guard let lengths, let longest = lengths.max(), lengths.contains(where: { $0 < longest })
        else { return }
        self.lengths = MLXArray(lengths.map { Int32($0) }) + batchOffset
    }

    public override func prepare(lengths: MLXArray?) {
        prepare(lengths: lengths?.asArray(Int.self))
    }

    /// Moves the right padding of a prefill to the left, where the mask excludes it.
    public override func finalize() {
        guard let lengths else { return }
        let roll = MLX.maximum(MLXArray(Int32(0)), batchOffset - lengths)
        if let keys, let values {
            self.keys = dynamicRoll(keys, shifts: roll[0..., .newAxis], axis: 2)
            self.values = dynamicRoll(values, shifts: roll[0..., .newAxis], axis: 2)
        }
        leftPadding = leftPadding + roll
        batchOffset = batchOffset - roll
        self.lengths = nil
    }

    public func filter(batchIndices: MLXArray) {
        let rows = batchIndices.asArray(Int32.self)
        guard !rows.isEmpty else {
            keys = nil
            values = nil
            lengths = nil
            idx = 0
            written = 0
            rotated = false
            leftPadding = MLXArray([Int32]())
            batchOffset = MLXArray([Int32]())
            return
        }
        let indices = MLXArray(rows)
        keys = keys?[indices]
        values = values?[indices]
        batchOffset = batchOffset[indices]
        leftPadding = leftPadding[indices]
        lengths = lengths?[indices]
    }

    /// Appends the rows of `other`, a `BatchRotatingKVCache` with prefilled rows and
    /// the same window, step and capacity origin.
    public func extend(other: any BatchedCache) {
        guard let other = other as? BatchRotatingKVCache else {
            preconditionFailure(
                "BatchRotatingKVCache can only be extended with a BatchRotatingKVCache")
        }
        precondition(
            other.keys != nil || other.batchSize == 0,
            "BatchRotatingKVCache.extend requires prefilled rows")
        precondition(
            other.window == window && other.step == step
                && other.capacityOrigin == capacityOrigin,
            "BatchRotatingKVCache.extend requires matching maxSize, step and capacityOrigin")
        guard keys != nil, other.keys != nil else {
            if keys == nil, other.keys == nil {
                leftPadding = concatenated([leftPadding, other.leftPadding])
                batchOffset = concatenated([batchOffset, other.batchOffset])
            } else if other.keys != nil {
                keys = other.keys
                values = other.values
                idx = other.idx
                written = other.written
                rotated = other.rotated
                leftPadding = other.leftPadding
                batchOffset = other.batchOffset
            }
            return
        }

        if rotated != other.rotated || idx != other.idx {
            temporalOrder()
            other.temporalOrder()
        }
        let end = max(idx, other.idx)
        let width = max(keys!.dim(2), other.keys!.dim(2))
        let own = rightJustify(keys!, values!, end: idx, to: end, width: width)
        let added = rightJustify(other.keys!, other.values!, end: other.idx, to: end, width: width)
        keys = concatenated([own.keys, added.keys])
        values = concatenated([own.values, added.values])
        leftPadding = concatenated([
            leftPadding + Int32(end - idx), other.leftPadding + Int32(end - other.idx),
        ])
        batchOffset = concatenated([batchOffset, other.batchOffset])
        idx = end
        written = max(written, other.written)
    }

    /// The row at `index` in temporal order without its left padding, as a
    /// ``RotatingKVCache`` with this cache's window, step and capacity origin.
    public func extract(_ index: Int) -> any KVCache {
        let cache = RotatingKVCache(maxSize: window, step: step)
        // Left padding goes negative once the window wraps.
        let padding = max(0, Int(leftPadding[index].item(Int32.self)))
        let rowOffset = max(0, Int(batchOffset[index].item(Int32.self)))
        var rowIdx = min(rowOffset, window)

        if let keys, let values {
            var rowKeys = keys[index ..< index + 1]
            var rowValues = values[index ..< index + 1]
            if rotated {
                rowKeys = MLX.roll(rowKeys, shift: -idx, axis: 2)
                rowValues = MLX.roll(rowValues, shift: -idx, axis: 2)
            }
            let end = rotated ? window : idx
            rowKeys = MLX.contiguous(rowKeys[.ellipsis, padding ..< end, 0...])
            rowValues = MLX.contiguous(rowValues[.ellipsis, padding ..< end, 0...])
            cache.state = [rowKeys, rowValues]
            // A concatenated buffer can hold more than a window, in temporal order.
            rowIdx = min(rowOffset, rowKeys.dim(2))
        }

        // RotatingKVCache's metadata: keep, maxSize, step, offset, idx, origin, wrapped.
        cache.metaState = [
            "0", String(window), String(step), String(rowOffset), String(rowIdx),
            capacityOrigin.rawValue, "false",
        ]
        return cache
    }

    /// A batch of one row holding the window of `cache`, or `nil` when `cache` has a
    /// pinned prefix or a rewind reserve, which a batch cannot hold.
    public static func fromSingle(_ cache: RotatingKVCache) -> BatchRotatingKVCache? {
        guard cache.preservedPrefixTokens == 0, cache.rewindCapacity == 0,
            let window = cache.maxSize
        else { return nil }
        let batch = BatchRotatingKVCache(maxSize: window, leftPadding: [0])
        batch.capacityOrigin = cache.capacityOrigin
        batch.step = cache.allocationStep
        if let (keys, values) = cache.logicalView(tail: window) {
            // New array objects, so writes to the batch leave `cache` unchanged.
            batch.keys = keys[.ellipsis]
            batch.values = values[.ellipsis]
            batch.idx = keys.dim(2)
            batch.written = keys.dim(2)
            batch.batchOffset = MLXArray([Int32(cache.offset)])
        }
        return batch
    }

    /// The mask for `n` new tokens: the window, the causal order and each row's left
    /// padding, laid out like the keys the matching update returns.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let windowSize = windowSize ?? window
        let offset = min(window - 1, written)
        let rinds = MLXArray(Int32(0) ..< Int32(offset + n))[.newAxis]
        let linds = MLXArray(Int32(offset) ..< Int32(offset + n))[0..., .newAxis]
        var mask = (linds .>= rinds) & (linds .< rinds + Int32(windowSize))

        // The matching update concatenates: it puts a wrapped buffer in order and then
        // trims it to `window - 1` positions. Predict that trim.
        let concatenates = n != 1 || lengths != nil
        var padding = leftPadding
        let trimSize = (concatenates && rotated ? window : idx) - window + (concatenates ? 1 : 0)
        if trimSize > 0 {
            padding = padding - Int32(trimSize)
        }

        // The matching update writes in place and wraps.
        let wraps = !concatenates && (rotated || idx >= window)
        if wraps {
            padding = padding - 1
        }
        mask = mask & (rinds .>= padding[0..., .newAxis, .newAxis, .newAxis])
        if wraps {
            let position = idx >= window ? 0 : idx
            mask = MLX.roll(mask, shift: position + 1, axis: -1)
        }
        return .array(mask)
    }

    public override func copy() -> any KVCache {
        let copy = BatchRotatingKVCache(maxSize: window, leftPadding: [])
        copy.leftPadding = leftPadding[0...]
        copy.batchOffset = batchOffset[0...]
        copy.keys = keys?[.ellipsis]
        copy.values = values?[.ellipsis]
        copy.idx = idx
        copy.written = written
        copy.rotated = rotated
        copy.lengths = lengths?[0...]
        copy.step = step
        copy.capacityOrigin = capacityOrigin
        return copy
    }
}
