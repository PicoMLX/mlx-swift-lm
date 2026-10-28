// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// A KV cache for a batch of left-padded sequences.
///
/// Port of mlx-lm's `BatchKVCache`. Rows are left-padded so that sequences of
/// different lengths end at the same position. Prompts `[1, 3, 5]`, `[7]` and
/// `[2, 6, 8, 9]` are stored as
/// ```
/// [0, 1, 3, 5]
/// [0, 0, 0, 7]
/// [2, 6, 8, 9]
/// ```
/// with ``leftPadding`` `[1, 3, 0]`. ``KVCache/offset`` is the width of the buffer in use.
public final class BatchKVCache: BaseKVCache, BatchPositionedKVCache, BatchedCache {
    /// Each row's left padding, shape `[B]`.
    public private(set) var leftPadding: MLXArray

    /// Each row's position, shape `[B]`. It starts at `-leftPadding`.
    public private(set) var batchOffset: MLXArray

    var keys: MLXArray?
    var values: MLXArray?

    /// The right padding of a ragged prefill, from `prepare` until `finalize`.
    private var rightPadding: MLXArray?

    /// Whether any row may have left padding. Kept on the CPU, so `makeMask` reads
    /// nothing back from the GPU.
    private var hasLeftPadding: Bool

    public var step = 256

    /// - Parameter leftPadding: the left padding of each row
    public init(leftPadding: [Int]) {
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.batchOffset = MLXArray(leftPadding.map { -Int32($0) })
        self.hasLeftPadding = leftPadding.contains { $0 != 0 }
        super.init()
    }

    /// The number of rows.
    public var batchSize: Int { leftPadding.dim(0) }

    /// Per-row positions, also when the cache is typed as `KVCache`.
    public override var ropeOffset: RoPEOffset { .batch(batchOffset + 0) }

    public override func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset
        if self.keys == nil || previous + keys.dim(2) > self.keys!.dim(2) {
            let length = (step + keys.dim(2) - 1) / step * step
            let newKeys = MLXArray.zeros(
                [keys.dim(0), keys.dim(1), length, keys.dim(3)], dtype: keys.dtype)
            let newValues = MLXArray.zeros(
                [values.dim(0), values.dim(1), length, values.dim(3)], dtype: values.dtype)
            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newKeys], axis: 2)
                self.values = concatenated([currentValues, newValues], axis: 2)
            } else {
                self.keys = newKeys
                self.values = newValues
            }
        }

        offset += keys.dim(2)
        batchOffset = batchOffset + Int32(keys.dim(2))
        self.keys![.ellipsis, previous ..< offset, 0...] = keys
        self.values![.ellipsis, previous ..< offset, 0...] = values
        return (self.keys![.ellipsis, ..<offset, 0...], self.values![.ellipsis, ..<offset, 0...])
    }

    public override var state: [MLXArray] {
        get {
            guard let keys, let values else { return [batchOffset, leftPadding] }
            return [
                keys[.ellipsis, ..<offset, 0...], values[.ellipsis, ..<offset, 0...],
                batchOffset, leftPadding,
            ]
        }
        set {
            switch newValue.count {
            case 2:
                keys = nil
                values = nil
                offset = 0
                batchOffset = newValue[0]
                leftPadding = newValue[1]
            case 4:
                keys = newValue[0]
                values = newValue[1]
                offset = newValue[0].dim(2)
                batchOffset = newValue[2]
                leftPadding = newValue[3]
            default:
                fatalError("BatchKVCache state must have 2 or 4 arrays, got \(newValue.count)")
            }
            hasLeftPadding = batchSize > 0 && leftPadding.max().item(Int32.self) > 0
        }
    }

    public override var isTrimmable: Bool { true }

    /// No row can lose more tokens than the shortest row holds.
    public override var maxTrimCount: Int {
        let shortest = batchSize > 0 ? Int(batchOffset.min().item(Int32.self)) : 0
        return min(offset, max(0, shortest))
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(maxTrimCount, n)
        offset -= trimmed
        batchOffset = batchOffset - Int32(trimmed)
        return trimmed
    }

    /// The symbolic mask while no row has left padding and the caller needs neither a
    /// window nor an array, which keeps the fused causal attention. Otherwise an explicit
    /// mask, even for one token, so the padding stays masked.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if !hasLeftPadding && windowSize == nil && !returnArray {
            return n == 1 ? .none : .causal
        }
        return .array(
            createCausalMask(
                n: n, offset: offset, windowSize: windowSize, leftPadding: leftPadding))
    }

    /// Prepares a right-padded prefill of rows with the given lengths.
    public override func prepare(lengths: [Int]?) {
        guard let lengths, let longest = lengths.max(), lengths.contains(where: { $0 < longest })
        else { return }
        rightPadding = MLXArray(lengths.map { Int32(longest - $0) })
    }

    public override func prepare(lengths: MLXArray?) {
        prepare(lengths: lengths?.asArray(Int.self))
    }

    /// Moves the right padding of a prefill to the left, where the mask excludes it.
    public override func finalize() {
        guard let rightPadding else { return }
        // A circular roll of the whole buffer lands each row's tokens at
        // `padding ..< offset`; what wraps around is masked or unused.
        if let keys, let values {
            self.keys = dynamicRoll(keys, shifts: rightPadding[0..., .newAxis], axis: 2)
            self.values = dynamicRoll(values, shifts: rightPadding[0..., .newAxis], axis: 2)
        }
        batchOffset = batchOffset - rightPadding
        leftPadding = leftPadding + rightPadding
        // `prepare` only keeps right padding when some row is shorter.
        hasLeftPadding = true
        self.rightPadding = nil
    }

    public func filter(batchIndices: MLXArray) {
        let rows = batchIndices.asArray(Int32.self)
        guard !rows.isEmpty else {
            keys = nil
            values = nil
            rightPadding = nil
            offset = 0
            leftPadding = MLXArray([Int32]())
            batchOffset = MLXArray([Int32]())
            hasLeftPadding = false
            return
        }
        let indices = MLXArray(rows)
        keys = keys?[indices]
        values = values?[indices]
        batchOffset = batchOffset[indices]
        leftPadding = leftPadding[indices]
        rightPadding = rightPadding?[indices]

        // Drop the left padding that every remaining row has, up to the positions stored
        // so far: in a chunked prefill the rest of it is still to come, and the rows keep
        // it, with their negative positions. Rows without keys keep all of theirs. Both
        // bounds come from one readback.
        let bounds = MLX.stacked([leftPadding.min(), leftPadding.max()]).asArray(Int32.self)
        let shared = keys != nil ? min(Int(bounds[0]), offset) : 0
        if shared > 0 {
            keys = keys?[.ellipsis, shared..., 0...]
            values = values?[.ellipsis, shared..., 0...]
            offset -= shared
            leftPadding = leftPadding - Int32(shared)
        }
        hasLeftPadding = Int(bounds[1]) > shared
    }

    /// Appends the rows of `other`. Rows without keys, on either side, join as rows of
    /// padding only, as in mlx-lm. Next to stored rows they start empty, at position 0.
    ///
    /// Rows with keys must have completed their prefill, including `finalize()` after a
    /// ragged one. Padding still to come would mask a row's next token from itself.
    public func extend(other: any BatchedCache) {
        guard let other = other as? BatchKVCache else {
            preconditionFailure("BatchKVCache can only be extended with a BatchKVCache")
        }
        guard other.batchSize > 0 else { return }
        guard batchSize > 0 else {
            // Take the rows as they are, without the copy a concatenation makes. New
            // array objects, so writes to either cache leave the other unchanged.
            keys = other.keys?[.ellipsis]
            values = other.values?[.ellipsis]
            offset = other.offset
            leftPadding = other.leftPadding
            batchOffset = other.batchOffset
            hasLeftPadding = other.hasLeftPadding
            return
        }
        guard let shapeKeys = keys ?? other.keys, let shapeValues = values ?? other.values
        else {
            // Neither cache holds keys yet.
            leftPadding = concatenated([leftPadding, other.leftPadding])
            batchOffset = concatenated([batchOffset, other.batchOffset])
            hasLeftPadding = hasLeftPadding || other.hasLeftPadding
            return
        }

        let end = max(offset, other.offset)
        let width = max(keys?.dim(2) ?? 0, other.keys?.dim(2) ?? 0)
        func justified(_ cache: BatchKVCache) -> (keys: MLXArray, values: MLXArray) {
            // A cache without keys holds rows of padding only.
            func empty(_ like: MLXArray) -> MLXArray {
                MLXArray.zeros([cache.batchSize, like.dim(1), 0, like.dim(3)], dtype: like.dtype)
            }
            return rightJustify(
                cache.keys ?? empty(shapeKeys), cache.values ?? empty(shapeValues),
                end: cache.offset, to: end, width: width)
        }
        func rows(_ cache: BatchKVCache) -> (leftPadding: MLXArray, batchOffset: MLXArray) {
            guard cache.keys != nil else {
                // Empty sequences whose first token comes next. Padding they kept for a
                // prefill cannot sit behind stored rows.
                let zeros = MLXArray.zeros([cache.batchSize], dtype: .int32)
                return (zeros + Int32(end), zeros)
            }
            return (cache.leftPadding + Int32(end - cache.offset), cache.batchOffset)
        }
        let own = justified(self)
        let added = justified(other)
        let ownRows = rows(self)
        let addedRows = rows(other)
        // Right-justifying the side that ends earlier pads its rows.
        hasLeftPadding = hasLeftPadding || other.hasLeftPadding || offset != other.offset
        keys = concatenated([own.keys, added.keys])
        values = concatenated([own.values, added.values])
        leftPadding = concatenated([ownRows.leftPadding, addedRows.leftPadding])
        batchOffset = concatenated([ownRows.batchOffset, addedRows.batchOffset])
        offset = end
    }

    /// The row at `index` without its left padding, as a ``KVCacheSimple``. A row that
    /// holds only padding so far, such as one in the middle of a chunked prefill, comes
    /// back empty.
    public func extract(_ index: Int) -> any KVCache {
        let cache = KVCacheSimple()
        let padding = Int(leftPadding[index].item(Int32.self))
        if let keys, let values, padding < offset {
            cache.keys = MLX.contiguous(keys[index ..< index + 1, 0..., padding ..< offset, 0...])
            cache.values = MLX.contiguous(
                values[index ..< index + 1, 0..., padding ..< offset, 0...])
            cache.offset = offset - padding
        }
        return cache
    }

    /// A batch of one row holding the sequence in `cache`.
    public static func fromSingle(_ cache: KVCacheSimple) -> BatchKVCache {
        // Exact type: a ChunkedKVCache can report an offset beyond the arrays it holds.
        precondition(
            type(of: cache) == KVCacheSimple.self,
            "BatchKVCache.fromSingle requires a KVCacheSimple")
        let batch = BatchKVCache(leftPadding: [0])
        if let keys = cache.keys, let values = cache.values {
            // New array objects, so writes to the batch leave `cache` unchanged.
            batch.keys = keys[.ellipsis]
            batch.values = values[.ellipsis]
            batch.offset = cache.offset
            batch.batchOffset = MLXArray([Int32(cache.offset)])
        }
        return batch
    }

    public override func copy() -> any KVCache {
        let copy = BatchKVCache(leftPadding: [])
        copy.leftPadding = leftPadding[0...]
        copy.batchOffset = batchOffset[0...]
        copy.keys = keys?[.ellipsis]
        copy.values = values?[.ellipsis]
        copy.rightPadding = rightPadding?[0...]
        copy.hasLeftPadding = hasLeftPadding
        copy.offset = offset
        copy.step = step
        return copy
    }
}

/// `keys` and `values`, whose rows end at position `end`, padded to end at `newEnd`
/// in a buffer `width` wide.
func rightJustify(
    _ keys: MLXArray, _ values: MLXArray, end: Int, to newEnd: Int, width: Int
) -> (keys: MLXArray, values: MLXArray) {
    var keys = keys
    var values = values
    let left = newEnd - end
    var right = width - keys.dim(2) - left
    if right < 0 {
        keys = keys[.ellipsis, ..<(keys.dim(2) + right), 0...]
        values = values[.ellipsis, ..<(values.dim(2) + right), 0...]
        right = 0
    }
    if left != 0 || right != 0 {
        let widths: [IntOrPair] = [0, 0, .init((left, right)), 0]
        keys = MLX.padded(keys, widths: widths)
        values = MLX.padded(values, widths: widths)
    }
    return (keys, values)
}

/// Rolls each row of `x` along `axis` by its own shift. Port of mlx-lm's `dynamic_roll`.
func dynamicRoll(_ x: MLXArray, shifts: MLXArray, axis: Int) -> MLXArray {
    let axis = axis >= 0 ? axis : x.ndim + axis
    let n = x.dim(axis)
    var positionShape = [Int](repeating: 1, count: x.ndim)
    positionShape[axis] = n
    var shiftShape = [Int](repeating: 1, count: x.ndim)
    for dimension in 0 ..< min(shifts.ndim, x.ndim) {
        shiftShape[dimension] = shifts.dim(dimension)
    }
    let positions = MLXArray(Int32(0) ..< Int32(n)).reshaped(positionShape)
    let size = MLXArray(Int32(n))
    // Adding `n` keeps the index non-negative for negative shifts.
    let indices = ((positions - shifts.reshaped(shiftShape)) % size + size) % size
    return takeAlong(x, indices.asType(.int32), axis: axis)
}
