// Copyright © 2026 Apple Inc.

import MLX

/// Left-padded rows of affine 4-bit or 8-bit KV state, kept packed while rows join and leave.
public final class BatchQuantizedKVCache: BaseKVCache, BatchPositionedKVCache,
    BatchableKVCache, QuantizedKVCacheProtocol
{
    private var storage: QuantizedKVCache
    private var positions: [Int32]
    private var positionsArray: MLXArray?
    private var leftPaddingValues: [Int32]
    private var rightPaddingValues: [Int32]?

    public private(set) var leftPadding: MLXArray

    public var batchSize: Int { positions.count }
    public var groupSize: Int { storage.groupSize }
    public var bits: Int { storage.bits }
    public var mode: QuantizationMode { .affine }

    public override var offset: Int {
        get { storage.offset }
        set { storage.offset = newValue }
    }

    public var batchOffset: MLXArray {
        if let positionsArray { return positionsArray }
        let array = MLXArray(positions)
        positionsArray = array
        return array
    }

    public override var ropeOffset: RoPEOffset {
        if let first = positions.first, positions.allSatisfy({ $0 == first }) {
            return .scalar(Int(first))
        }
        return .batch(batchOffset)
    }

    public init(leftPadding: [Int], groupSize: Int = 64, bits: Int = 8) {
        precondition(bits == 4 || bits == 8, "Batched affine KV caches support 4 or 8 bits")
        let padding = leftPadding.map { Int32($0) }
        storage = QuantizedKVCache(groupSize: groupSize, bits: bits)
        positions = padding.map { -$0 }
        leftPaddingValues = padding
        self.leftPadding = MLXArray(padding)
        super.init()
    }

    private func setPositions(_ values: [Int32]) {
        positions = values
        positionsArray = nil
    }

    private func setLeftPadding(_ values: [Int32]) {
        leftPaddingValues = values
        leftPadding = MLXArray(values)
    }

    private func replaceStorage(_ arrays: [MLXArray], offset: Int) {
        let replacement = QuantizedKVCache(groupSize: groupSize, bits: bits)
        if !arrays.isEmpty { replacement.state = arrays }
        replacement.offset = offset
        storage = replacement
    }

    public override func innerState() -> [MLXArray] { storage.innerState() }

    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        storage.getQuantizedState()
    }

    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        precondition(keys.dim(0) == batchSize, "One KV row is required per batch row")
        let result = storage.updateQuantized(keys: keys, values: values)
        let n = Int32(keys.dim(2))
        setPositions(positions.map { $0 + n })
        return result
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        preconditionFailure("Use updateQuantized with a BatchQuantizedKVCache")
    }

    /// Packed keys and values followed by per-row positions and left padding.
    public override var state: [MLXArray] {
        get { storage.state + [batchOffset, leftPadding] }
        set {
            precondition(
                newValue.count == 2 || newValue.count == 8,
                "BatchQuantizedKVCache state needs 2 or 8 arrays")
            let packed = Array(newValue.dropLast(2))
            replaceStorage(packed, offset: packed.first?.dim(2) ?? 0)
            setPositions(newValue[newValue.count - 2].asArray(Int32.self))
            setLeftPadding(newValue[newValue.count - 1].asArray(Int32.self))
            precondition(positions.count == leftPaddingValues.count, "Inconsistent row metadata")
            rightPaddingValues = nil
        }
    }

    public override var isTrimmable: Bool { true }

    public override func isTrimmable(after positions: Int) -> Bool {
        guard positions >= 0 else { return false }
        if positions == 0 { return true }
        // Padding writes and zero-row batches cannot rewind the whole proposed suffix.
        return !self.positions.isEmpty && self.positions.allSatisfy { $0 >= 0 }
            && storage.isTrimmable(after: positions)
    }

    public override var maxTrimCount: Int {
        min(offset, max(0, Int(positions.min() ?? 0)))
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(maxTrimCount, max(0, n))
        _ = storage.trim(trimmed)
        setPositions(positions.map { $0 - Int32(trimmed) })
        return trimmed
    }

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1, !leftPaddingValues.contains(where: { $0 != 0 }),
            windowSize.map({ $0 >= offset + n }) ?? true
        {
            return .none
        }
        return .array(
            createCausalMask(
                n: n, offset: offset, windowSize: windowSize, leftPadding: leftPadding))
    }

    public override func prepare(lengths: [Int]?) {
        guard let lengths else { return }
        precondition(lengths.count == batchSize, "One prefill length is required per row")
        guard let longest = lengths.max(), lengths.contains(where: { $0 < longest }) else {
            return
        }
        rightPaddingValues = lengths.map { Int32(longest - $0) }
    }

    public override func prepare(lengths: MLXArray?) {
        prepare(lengths: lengths?.asArray(Int.self))
    }

    public override func finalize() {
        guard let padding = rightPaddingValues else { return }
        let arrays = storage.innerState()
        if !arrays.isEmpty {
            let shifts = MLXArray(padding)[0..., .newAxis]
            replaceStorage(arrays.map { dynamicRoll($0, shifts: shifts, axis: 2) }, offset: offset)
        }
        setPositions(zip(positions, padding).map { $0 - $1 })
        setLeftPadding(zip(leftPaddingValues, padding).map { $0 + $1 })
        rightPaddingValues = nil
    }

    public func filter(batchIndices: MLXArray) {
        let rows = batchIndices.asArray(Int32.self).map { Int($0) }
        guard !rows.isEmpty else {
            replaceStorage([], offset: 0)
            setPositions([])
            setLeftPadding([])
            rightPaddingValues = nil
            return
        }
        let indices = MLXArray(rows.map { Int32($0) })
        var arrays = storage.innerState().map { $0[indices] }
        setPositions(rows.map { positions[$0] })
        var padding = rows.map { leftPaddingValues[$0] }
        rightPaddingValues = rightPaddingValues.map { values in rows.map { values[$0] } }
        let shared = min(padding.min() ?? 0, Int32(offset))
        var end = offset
        if shared > 0, !arrays.isEmpty {
            arrays = arrays.map { $0[.ellipsis, Int(shared)..., 0...] }
            end -= Int(shared)
            padding = padding.map { $0 - shared }
        }
        replaceStorage(arrays, offset: end)
        setLeftPadding(padding)
    }

    public func extend(other: any KVCache) {
        guard let other = other as? BatchQuantizedKVCache else {
            preconditionFailure("A BatchQuantizedKVCache needs another BatchQuantizedKVCache")
        }
        extend(other: other)
    }

    public func validateExtension(other: any KVCache) throws {
        guard let other = other as? BatchQuantizedKVCache,
            groupSize == other.groupSize && bits == other.bits && mode == other.mode
        else {
            throw KVCacheError(
                message: "Batched quantized caches must share type, group size and bits")
        }
        guard rightPaddingValues == nil && other.rightPaddingValues == nil else {
            throw KVCacheError(message: "Finalize a ragged prefill before extending its cache")
        }

        let own = innerState()
        let added = other.innerState()
        for (cache, arrays) in [(self, own), (other, added)] {
            guard cache.offset >= 0,
                cache.positions.count == cache.leftPaddingValues.count,
                zip(cache.positions, cache.leftPaddingValues).allSatisfy({ position, padding in
                    padding >= 0 && Int64(position) + Int64(padding) == Int64(cache.offset)
                }),
                arrays.isEmpty ? cache.offset == 0 : arrays.count == 6
            else {
                throw KVCacheError(message: "Invalid batched affine cache state")
            }
            guard
                arrays.allSatisfy({
                    $0.ndim == 4 && $0.dim(0) == cache.batchSize && $0.dim(2) >= cache.offset
                })
            else {
                throw KVCacheError(message: "Invalid batched affine tensor layout")
            }
            for start in stride(from: 0, to: arrays.count, by: 3) {
                let packed = arrays[start]
                let scales = arrays[start + 1]
                let biases = arrays[start + 2]
                guard cache.groupSize > 0, packed.dtype == .uint32,
                    [.float16, .bfloat16, .float32].contains(scales.dtype),
                    scales.dtype == biases.dtype && scales.shape == biases.shape,
                    packed.dim(1) == scales.dim(1) && packed.dim(2) == scales.dim(2),
                    packed.dim(3) * (32 / cache.bits) == scales.dim(3) * cache.groupSize
                else {
                    throw KVCacheError(message: "Invalid packed affine components")
                }
            }
        }
        guard
            own.isEmpty || added.isEmpty
                || zip(own, added).allSatisfy({ lhs, rhs in
                    lhs.dim(1) == rhs.dim(1) && lhs.dim(3) == rhs.dim(3) && lhs.dtype == rhs.dtype
                })
        else {
            throw KVCacheError(message: "Batched quantized caches must share one tensor layout")
        }
    }

    public func extend(other: BatchQuantizedKVCache) {
        precondition(
            groupSize == other.groupSize && bits == other.bits,
            "Batched quantized caches must share group size and bits")
        precondition(
            rightPaddingValues == nil && other.rightPaddingValues == nil,
            "Finalize a ragged prefill before extending its cache")
        guard other.batchSize > 0 else { return }
        guard batchSize > 0 else {
            replaceStorage(other.innerState().map { $0[.ellipsis] }, offset: other.offset)
            setPositions(other.positions)
            setLeftPadding(other.leftPaddingValues)
            return
        }

        let own = innerState()
        let added = other.innerState()
        guard !own.isEmpty || !added.isEmpty else {
            setPositions(positions + other.positions)
            setLeftPadding(leftPaddingValues + other.leftPaddingValues)
            return
        }
        let end = max(offset, other.offset)
        let width = max(own.first?.dim(2) ?? 0, added.first?.dim(2) ?? 0)
        let components = own.isEmpty ? added : own
        let merged = components.indices.map { index in
            let example = components[index]
            let lhs = own.isEmpty ? emptyRows(batchSize, like: example) : own[index]
            let rhs = added.isEmpty ? emptyRows(other.batchSize, like: example) : added[index]
            return concatenated([
                rightJustify(lhs, end: offset, to: end, width: width),
                rightJustify(rhs, end: other.offset, to: end, width: width),
            ])
        }
        let ownShift = Int32(end - offset)
        let addedShift = Int32(end - other.offset)
        replaceStorage(merged, offset: end)
        setPositions(positions + other.positions)
        setLeftPadding(
            leftPaddingValues.map { $0 + ownShift }
                + other.leftPaddingValues.map { $0 + addedShift })
    }

    public func extract(_ index: Int) -> any KVCache {
        precondition(positions.indices.contains(index), "Batch row index out of range")
        precondition(
            rightPaddingValues == nil,
            "BatchQuantizedKVCache.extract takes finalized rows; finalize a ragged prefill first")
        let row = QuantizedKVCache(groupSize: groupSize, bits: bits)
        let padding = Int(leftPaddingValues[index])
        guard padding < offset else { return row }
        let arrays = innerState()
        if !arrays.isEmpty {
            row.state = arrays.map {
                MLX.contiguous($0[index ..< index + 1, 0..., padding ..< offset, 0...])
            }
            row.offset = offset - padding
        }
        return row
    }

    /// The rows of `caches`, exactly `QuantizedKVCache` rows, as one batch. See
    /// ``QuantizedKVCache/merge(_:)``.
    public static func merge(_ caches: [any KVCache]) throws -> any BatchableKVCache {
        try QuantizedKVCache.merge(try batchRows(caches, of: QuantizedKVCache.self))
    }

    public override func copy() -> any KVCache {
        let copy = BatchQuantizedKVCache(leftPadding: [], groupSize: groupSize, bits: bits)
        copy.replaceStorage(innerState().map { $0[.ellipsis] }, offset: offset)
        copy.setPositions(positions)
        copy.setLeftPadding(leftPaddingValues)
        copy.rightPaddingValues = rightPaddingValues
        return copy
    }

    fileprivate convenience init(merging caches: [QuantizedKVCache]) {
        let longest = caches.map(\.offset).max() ?? 0
        self.init(
            leftPadding: caches.map { longest - $0.offset },
            groupSize: caches[0].groupSize, bits: caches[0].bits)
        guard let template = caches.first(where: { !$0.innerState().isEmpty }) else { return }
        let components = template.innerState()
        let merged = components.indices.map { index in
            concatenated(
                caches.map { cache in
                    let arrays = cache.innerState()
                    guard !arrays.isEmpty else {
                        return MLXArray.zeros(
                            [1, components[index].dim(1), longest, components[index].dim(3)],
                            dtype: components[index].dtype)
                    }
                    let array = arrays[index][.ellipsis, ..<cache.offset, 0...]
                    return rightJustify(array, end: cache.offset, to: longest, width: longest)
                })
        }
        replaceStorage(merged, offset: longest)
        setPositions(caches.map { Int32($0.offset) })
    }
}

extension QuantizedKVCache {
    /// Batches exactly these affine rows without dequantizing their packed components.
    public static func merge(_ caches: [QuantizedKVCache]) throws -> BatchQuantizedKVCache {
        guard let first = caches.first else { throw KVCacheError(message: "merge needs a cache") }
        guard caches.allSatisfy({ type(of: $0) == QuantizedKVCache.self }) else {
            throw KVCacheError(
                message: "QuantizedKVCache.merge takes exactly QuantizedKVCache rows")
        }
        guard first.mode == .affine, first.bits == 4 || first.bits == 8,
            caches.allSatisfy({
                $0.groupSize == first.groupSize && $0.bits == first.bits && $0.mode == first.mode
            })
        else {
            throw KVCacheError(
                message: "Batched KV rows require matching affine 4-bit or 8-bit storage")
        }
        let template = caches.first(where: { !$0.innerState().isEmpty })?.innerState()
        for cache in caches {
            let arrays = cache.innerState()
            guard cache.offset >= 0, arrays.isEmpty ? cache.offset == 0 : arrays.count == 6 else {
                throw KVCacheError(message: "Invalid affine cache state")
            }
            if let template, !arrays.isEmpty {
                guard
                    zip(arrays, template).allSatisfy({ array, example in
                        array.ndim == 4 && array.dim(0) == 1 && array.dim(2) >= cache.offset
                            && array.dim(1) == example.dim(1) && array.dim(3) == example.dim(3)
                            && array.dtype == example.dtype
                    })
                else {
                    throw KVCacheError(message: "Quantized KV rows must share one tensor layout")
                }
            }
        }
        return BatchQuantizedKVCache(merging: caches)
    }
}

private func emptyRows(_ count: Int, like array: MLXArray) -> MLXArray {
    MLXArray.zeros([count, array.dim(1), 0, array.dim(3)], dtype: array.dtype)
}

private func rightJustify(_ array: MLXArray, end: Int, to newEnd: Int, width: Int) -> MLXArray {
    let left = newEnd - end
    var result = array
    var right = width - array.dim(2) - left
    if right < 0 {
        result = result[.ellipsis, ..<(array.dim(2) + right), 0...]
        right = 0
    }
    if left != 0 || right != 0 {
        result = MLX.padded(result, widths: [0, 0, .init((left, right)), 0])
    }
    return result
}
