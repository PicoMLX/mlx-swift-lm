// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Batch-aware KV cache with left-padding support and efficient batch merging.
public class BatchKVCache: BaseKVCache {

    public var step = 256

    private var keys: MLXArray?
    private var values: MLXArray?
    public private(set) var offsets: MLXArray
    private var leftPaddingStorage: MLXArray
    private var currentLength: Int = 0
    private var pendingRightPadding: MLXArray?

    public var batchSize: Int { leftPaddingStorage.size }
    public var leftPadding: [Int] { leftPaddingStorage.asArray(Int.self) }

    public override var isTrimmable: Bool { true }

    public init(batchSize: Int, leftPadding: [Int]) {
        precondition(leftPadding.count == batchSize, "leftPadding must have exactly batchSize elements")
        self.leftPaddingStorage = MLXArray(leftPadding)
        self.offsets = MLXArray(leftPadding.map { -$0 })
        super.init()
    }

    public convenience init(leftPadding: [Int]) {
        self.init(batchSize: leftPadding.count, leftPadding: leftPadding)
    }

    public override func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = currentLength
        let tokenCount = keys.dim(2)

        ensureCapacity(for: keys, values: values, previousLength: previous)

        guard self.keys != nil, self.values != nil else {
            fatalError("BatchKVCache storage was not initialized before update")
        }

        currentLength += tokenCount
        offsets = offsets + tokenCount
        offset = currentLength

        self.keys?[.ellipsis, previous ..< currentLength, 0...] = keys
        self.values?[.ellipsis, previous ..< currentLength, 0...] = values

        return (
            self.keys![.ellipsis, ..<currentLength, 0...],
            self.values![.ellipsis, ..<currentLength, 0...]
        )
    }

    public override var state: [MLXArray] {
        get {
            guard var keys = self.keys, var values = self.values else { return [] }
            if currentLength < keys.dim(2) {
                keys = keys[.ellipsis, ..<currentLength, 0...]
                values = values[.ellipsis, ..<currentLength, 0...]
            }
            return [keys, values, offsets, leftPaddingStorage]
        }
        set {
            guard newValue.count >= 4 else {
                keys = nil
                values = nil
                offsets = MLXArray([])
                leftPaddingStorage = MLXArray([])
                currentLength = 0
                offset = 0
                return
            }

            keys = newValue[0]
            values = newValue[1]
            offsets = newValue[2]
            leftPaddingStorage = newValue[3]
            currentLength = keys?.dim(2) ?? 0
            offset = currentLength
        }
    }

    public override var metaState: [String] {
        get { [] }
        set {
            if !newValue.isEmpty {
                fatalError("BatchKVCache does not use metaState")
            }
        }
    }

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 && leftPadding.allSatisfy({ $0 == 0 }) && windowSize == nil {
            return .none
        }

        var mask = createCausalMask(n: n, offset: currentLength, windowSize: windowSize)
        mask = mask[.newAxis, .newAxis, 0..., 0...]

        if leftPaddingStorage.size > 0 && leftPadding.contains(where: { $0 > 0 }) {
            let total = currentLength + n
            let columnIndices = MLXArray(Int32(0) ..< Int32(total))[.newAxis, .newAxis, .newAxis, 0...]
            let leftMask = leftPaddingStorage[0..., .newAxis, .newAxis, .newAxis] .<= columnIndices
            mask = mask & leftMask
        }

        return .array(mask)
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(currentLength, n)
        if trimmed == 0 { return 0 }
        currentLength -= trimmed
        offsets = offsets - trimmed
        offset = currentLength
        return trimmed
    }

    public func prepare(rightPadding: [Int]?) {
        guard let rightPadding, rightPadding.contains(where: { $0 > 0 }) else {
            pendingRightPadding = nil
            return
        }
        pendingRightPadding = MLXArray(rightPadding)
    }

    public func finalizeBatchPrefill() {
        guard let pendingRightPadding,
            pendingRightPadding.size > 0,
            let keys,
            let values
        else {
            self.pendingRightPadding = nil
            return
        }

        self.keys = dynamicRoll(keys, shifts: pendingRightPadding, axis: 2)
        self.values = dynamicRoll(values, shifts: pendingRightPadding, axis: 2)
        offsets = offsets - pendingRightPadding
        leftPaddingStorage = leftPaddingStorage + pendingRightPadding
        self.pendingRightPadding = nil
    }

    public static func fromSingle(
        perSequenceCaches: [[KVCacheSimple]],
        leftPadding: [Int]? = nil
    ) -> [BatchKVCache] {
        let batchSize = perSequenceCaches.count
        guard batchSize > 0 else { return [] }

        let layerCount = perSequenceCaches[0].count
        let sequenceLengths = perSequenceCaches.map { $0.first?.offset ?? 0 }
        let maxOffset = sequenceLengths.max() ?? 0
        let padding = leftPadding ?? sequenceLengths.map { maxOffset - $0 }

        return (0 ..< layerCount).map { layerIndex in
            let layerCaches = perSequenceCaches.map { $0[layerIndex] }
            return makeBatchLayer(
                layerCaches: layerCaches,
                padding: padding,
                sequenceLengths: sequenceLengths,
                maxOffset: maxOffset
            )
        }
    }

    public func filter(batchIndices: MLXArray) {
        guard keys != nil, values != nil else { return }

        if batchIndices.size == 0 {
            keys = nil
            values = nil
            offsets = MLXArray([])
            leftPaddingStorage = MLXArray([])
            currentLength = 0
            offset = 0
            return
        }

        keys = keys?[batchIndices, .ellipsis]
        values = values?[batchIndices, .ellipsis]
        offsets = offsets[batchIndices]
        leftPaddingStorage = leftPaddingStorage[batchIndices]

        let minLeftPad = leftPaddingStorage.min().item(Int.self)
        if minLeftPad > 0 {
            let shift = min(minLeftPad, currentLength)
            if shift > 0 {
                keys = keys?[.ellipsis, shift..., 0...]
                values = values?[.ellipsis, shift..., 0...]
                currentLength -= shift
                offset = currentLength
            }
            leftPaddingStorage = leftPaddingStorage - MLXArray(minLeftPad)
        }
    }

    public func filter(keepIndices: [Int]) -> BatchKVCache {
        let filtered = BatchKVCache(leftPadding: leftPadding)
        filtered.state = state
        filtered.filter(batchIndices: MLXArray(keepIndices.map(Int32.init)))
        return filtered
    }

    public func extend(other: BatchKVCache) {
        guard let otherKeys = other.keys, let otherValues = other.values else { return }

        if keys == nil {
            keys = otherKeys
            values = otherValues
            offsets = other.offsets
            leftPaddingStorage = other.leftPaddingStorage
            currentLength = other.currentLength
            offset = currentLength
            return
        }

        guard let currentKeys = keys else {
            fatalError("BatchKVCache missing backing storage during extend")
        }

        let maxIndex = max(currentLength, other.currentLength)
        let maxCapacity = max(currentKeys.dim(2), otherKeys.dim(2))

        let paddedSelf = paddedForConcat(targetLength: maxIndex, targetCapacity: maxCapacity)
        let paddedOther = other.paddedForConcat(targetLength: maxIndex, targetCapacity: maxCapacity)

        keys = MLX.concatenated([paddedSelf.keys, paddedOther.keys], axis: 0)
        values = MLX.concatenated([paddedSelf.values, paddedOther.values], axis: 0)
        offsets = MLX.concatenated([paddedSelf.offsets, paddedOther.offsets], axis: 0)
        leftPaddingStorage = MLX.concatenated(
            [paddedSelf.leftPadding, paddedOther.leftPadding], axis: 0)
        currentLength = maxIndex
        offset = currentLength
    }

    public func extract(index: Int) -> KVCacheSimple {
        let simple = KVCacheSimple()
        guard let keys, let values else { return simple }

        let pad = leftPadding[index]
        guard currentLength > pad else { return simple }

        simple.state = [
            keys[index...(index), .ellipsis, pad..<currentLength, 0...],
            values[index...(index), .ellipsis, pad..<currentLength, 0...],
        ]
        return simple
    }

    private static func makeBatchLayer(
        layerCaches: [KVCacheSimple],
        padding: [Int],
        sequenceLengths: [Int],
        maxOffset: Int
    ) -> BatchKVCache {
        let batchCache = BatchKVCache(leftPadding: padding)
        batchCache.currentLength = maxOffset
        batchCache.offset = maxOffset
        batchCache.offsets = MLXArray(sequenceLengths)

        guard maxOffset > 0 else { return batchCache }

        var keysList: [MLXArray] = []
        var valuesList: [MLXArray] = []

        for (cache, pad) in zip(layerCaches, padding) {
            let state = cache.state
            guard state.count >= 2 else {
                keysList.append(MLXArray.zeros([1, 1, maxOffset, 1]))
                valuesList.append(MLXArray.zeros([1, 1, maxOffset, 1]))
                continue
            }

            var keys = state[0]
            var values = state[1]

            if pad > 0 {
                let keyPad = MLXArray.zeros([1, keys.dim(1), pad, keys.dim(3)], dtype: keys.dtype)
                let valuePad = MLXArray.zeros([1, values.dim(1), pad, values.dim(3)], dtype: values.dtype)
                keys = MLX.concatenated([keyPad, keys], axis: 2)
                values = MLX.concatenated([valuePad, values], axis: 2)
            }

            keysList.append(keys)
            valuesList.append(values)
        }

        let realIndex = keysList.firstIndex(where: { $0.dim(1) > 1 }) ?? 0
        if realIndex < keysList.count {
            let referenceKeys = keysList[realIndex]
            let referenceValues = valuesList[realIndex]
            for index in 0 ..< keysList.count where keysList[index].dim(1) == 1 && referenceKeys.dim(1) > 1 {
                keysList[index] = MLXArray.zeros(
                    [1, referenceKeys.dim(1), maxOffset, referenceKeys.dim(3)],
                    dtype: referenceKeys.dtype
                )
                valuesList[index] = MLXArray.zeros(
                    [1, referenceValues.dim(1), maxOffset, referenceValues.dim(3)],
                    dtype: referenceValues.dtype
                )
            }
        }

        batchCache.keys = MLX.concatenated(keysList, axis: 0)
        batchCache.values = MLX.concatenated(valuesList, axis: 0)
        return batchCache
    }

    private func ensureCapacity(for keys: MLXArray, values: MLXArray, previousLength: Int) {
        if self.keys == nil {
            let chunk = allocateChunk(for: keys, values: values)
            self.keys = chunk.keys
            self.values = chunk.values
            return
        }

        guard var currentKeys = self.keys, var currentValues = self.values else {
            fatalError("BatchKVCache missing backing storage")
        }

        let tokenCount = keys.dim(2)
        if previousLength + tokenCount <= currentKeys.dim(2) {
            return
        }

        if previousLength % step != 0 {
            currentKeys = currentKeys[.ellipsis, ..<previousLength, 0...]
            currentValues = currentValues[.ellipsis, ..<previousLength, 0...]
        }

        let chunk = allocateChunk(for: keys, values: values)
        self.keys = MLX.concatenated([currentKeys, chunk.keys], axis: 2)
        self.values = MLX.concatenated([currentValues, chunk.values], axis: 2)
    }

    private func allocateChunk(for keys: MLXArray, values: MLXArray) -> (keys: MLXArray, values: MLXArray) {
        let batch = keys.dim(0)
        let kvHeads = keys.dim(1)
        let tokenCount = keys.dim(2)
        let keyDim = keys.dim(3)
        let valueDim = values.dim(3)

        let steps = (step + tokenCount - 1) / step
        let capacity = steps * step

        return (
            MLXArray.zeros([batch, kvHeads, capacity, keyDim], dtype: keys.dtype),
            MLXArray.zeros([batch, kvHeads, capacity, valueDim], dtype: values.dtype)
        )
    }

    private func paddedForConcat(
        targetLength: Int,
        targetCapacity: Int
    ) -> (keys: MLXArray, values: MLXArray, offsets: MLXArray, leftPadding: MLXArray) {
        guard var keys = self.keys, var values = self.values else {
            fatalError("BatchKVCache has no backing tensors to pad")
        }

        let left = targetLength - currentLength
        var right = targetCapacity - keys.dim(2) - left

        if right < 0 {
            let end = keys.dim(2) + right
            keys = keys[.ellipsis, ..<end, 0...]
            values = values[.ellipsis, ..<end, 0...]
            right = 0
        }

        if left != 0 || right != 0 {
            let padWidths: [IntOrPair] = [
                IntOrPair(0),
                IntOrPair(0),
                IntOrPair((left, right)),
                IntOrPair(0),
            ]
            keys = MLX.padded(keys, widths: padWidths)
            values = MLX.padded(values, widths: padWidths)
        }

        let adjustedLeftPadding = leftPaddingStorage + MLXArray(left)
        return (keys, values, offsets, adjustedLeftPadding)
    }
}

/// Batch-aware rotating cache for sliding-window attention.
public class BatchRotatingKVCache: BaseKVCache {

    public var step = 256

    private let maxWindowSize: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    private var leftPadding: MLXArray
    public private(set) var offsets: MLXArray
    private var idx: Int = 0
    private var offsetCursor: Int = 0
    private var rotated = false
    private var pendingLengths: MLXArray?

    public override var maxSize: Int? { maxWindowSize }
    public override var isTrimmable: Bool { offsetCursor < maxWindowSize }

    public init(maxSize: Int, leftPadding: [Int]) {
        maxWindowSize = maxSize
        self.leftPadding = MLXArray(leftPadding)
        offsets = MLXArray(leftPadding.map { -$0 })
        super.init()
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if keys.dim(2) == 1 {
            return updateInPlace(keys: keys, values: values)
        } else {
            return updateConcat(keys: keys, values: values)
        }
    }

    public override var state: [MLXArray] {
        get {
            guard var keys = self.keys, var values = self.values else { return [] }
            if offsetCursor < keys.dim(2) {
                keys = keys[.ellipsis, ..<offsetCursor, 0...]
                values = values[.ellipsis, ..<offsetCursor, 0...]
            }
            return [keys, values, offsets, leftPadding]
        }
        set {
            guard newValue.count >= 4 else {
                keys = nil
                values = nil
                offsets = MLXArray([])
                leftPadding = MLXArray([])
                idx = 0
                offsetCursor = 0
                rotated = false
                offset = 0
                return
            }

            keys = newValue[0]
            values = newValue[1]
            offsets = newValue[2]
            leftPadding = newValue[3]
            offsetCursor = keys?.dim(2) ?? 0
            idx = offsetCursor
            offset = offsetCursor
            rotated = false
        }
    }

    public override var metaState: [String] {
        get {
            [
                String(maxWindowSize),
                String(offsetCursor),
                String(idx),
                rotated ? "1" : "0",
            ]
        }
        set {
            guard newValue.count == 4 else {
                fatalError("BatchRotatingKVCache metaState must have exactly 4 values")
            }
            guard let parsedMax = Int(newValue[0]) else {
                fatalError("Invalid maxSize in BatchRotatingKVCache metaState")
            }
            guard parsedMax == maxWindowSize else {
                fatalError("BatchRotatingKVCache maxSize mismatch during state load")
            }
            guard let parsedOffset = Int(newValue[1]), let parsedIndex = Int(newValue[2]) else {
                fatalError("Invalid offset/index in BatchRotatingKVCache metaState")
            }

            offsetCursor = parsedOffset
            idx = parsedIndex
            rotated = newValue[3] == "1"
            offset = offsetCursor
        }
    }

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 && leftPadding.asArray(Int.self).allSatisfy({ $0 == 0 }) && windowSize == nil {
            return .none
        }
        return makeMask(sequenceLength: n, returnArray: returnArray, windowSize: windowSize)
    }

    public func filter(batchIndices: MLXArray) {
        guard keys != nil, values != nil else { return }

        if batchIndices.size == 0 {
            keys = nil
            values = nil
            offsets = MLXArray([])
            leftPadding = MLXArray([])
            offsetCursor = 0
            idx = 0
            rotated = false
            offset = 0
            return
        }

        keys = keys?[batchIndices, .ellipsis]
        values = values?[batchIndices, .ellipsis]
        offsets = offsets[batchIndices]
        leftPadding = leftPadding[batchIndices]
    }

    public func extend(other: BatchRotatingKVCache) {
        guard let otherKeys = other.keys, let otherValues = other.values else { return }

        if keys == nil {
            keys = otherKeys
            values = otherValues
            offsets = other.offsets
            leftPadding = other.leftPadding
            offsetCursor = other.offsetCursor
            idx = other.idx
            rotated = other.rotated
            offset = offsetCursor
            return
        }

        ensureTemporalOrder()
        other.ensureTemporalOrder()

        guard let currentKeys = keys, values != nil else {
            fatalError("BatchRotatingKVCache missing backing storage during extend")
        }

        let maxIndex = max(idx, other.idx)
        let maxCapacity = max(currentKeys.dim(2), otherKeys.dim(2))

        func pad(_ cache: BatchRotatingKVCache) -> (
            keys: MLXArray, values: MLXArray, offsets: MLXArray, leftPadding: MLXArray
        ) {
            guard var keys = cache.keys, var values = cache.values else {
                fatalError("BatchRotatingKVCache has no backing tensors to pad")
            }

            let left = maxIndex - cache.idx
            var right = maxCapacity - keys.dim(2) - left
            if right < 0 {
                let end = keys.dim(2) + right
                keys = keys[.ellipsis, ..<end, 0...]
                values = values[.ellipsis, ..<end, 0...]
                right = 0
            }

            if left != 0 || right != 0 {
                let padWidths: [IntOrPair] = [
                    IntOrPair(0),
                    IntOrPair(0),
                    IntOrPair((left, right)),
                    IntOrPair(0),
                ]
                keys = MLX.padded(keys, widths: padWidths)
                values = MLX.padded(values, widths: padWidths)
            }

            return (keys, values, cache.offsets, cache.leftPadding + MLXArray(left))
        }

        let paddedSelf = pad(self)
        let paddedOther = pad(other)

        keys = MLX.concatenated([paddedSelf.keys, paddedOther.keys], axis: 0)
        values = MLX.concatenated([paddedSelf.values, paddedOther.values], axis: 0)
        offsets = MLX.concatenated([paddedSelf.offsets, paddedOther.offsets], axis: 0)
        leftPadding = MLX.concatenated([paddedSelf.leftPadding, paddedOther.leftPadding], axis: 0)
        idx = maxIndex
        offsetCursor = max(offsetCursor, other.offsetCursor)
        rotated = false
        offset = offsetCursor
    }

    public func prepare(lengths: [Int]?, rightPadding: [Int]?) {
        guard let rightPadding, rightPadding.contains(where: { $0 > 0 }) else {
            pendingLengths = nil
            return
        }
        let prefixLengths = lengths ?? Array(repeating: 0, count: rightPadding.count)
        pendingLengths = MLXArray(prefixLengths) + offsets
    }

    public func finalizeBatchPrefill() {
        guard let pendingLengths,
            let keys,
            let values
        else {
            self.pendingLengths = nil
            return
        }

        let roll = maximum(MLXArray(0), offsets - pendingLengths)
        self.keys = dynamicRoll(keys, shifts: roll, axis: 2)
        self.values = dynamicRoll(values, shifts: roll, axis: 2)
        leftPadding = leftPadding + roll
        offsets = offsets - roll
        self.pendingLengths = nil
    }

    public func extract(index: Int) -> RotatingKVCache {
        let cache = RotatingKVCache(maxSize: maxWindowSize, keep: 0)
        guard let keys, let values else { return cache }

        let padding = leftPadding[index].item(Int.self)
        let offsetValue = offsets[index].item(Int.self)

        var extractedKeys = keys[index ..< (index + 1), .ellipsis]
        var extractedValues = values[index ..< (index + 1), .ellipsis]
        var extractedIndex = idx

        if rotated {
            extractedKeys = MLX.roll(extractedKeys, shift: -idx, axis: 2)
            extractedValues = MLX.roll(extractedValues, shift: -idx, axis: 2)
            extractedIndex = maxWindowSize
        }

        cache.state = [
            extractedKeys[.ellipsis, padding ..< extractedIndex, 0...],
            extractedValues[.ellipsis, padding ..< extractedIndex, 0...],
        ]
        cache.metaState = [
            String(0),
            String(maxWindowSize),
            String(offsetValue),
            String(min(extractedIndex - padding, maxWindowSize)),
        ]
        return cache
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offsetCursor, n)
        if trimmed == 0 { return 0 }

        offsetCursor -= trimmed
        idx = max(idx - trimmed, 0)
        offsets = offsets - MLXArray(trimmed)
        offset = offsetCursor
        return trimmed
    }

    public func makeMask(
        sequenceLength: Int,
        returnArray: Bool = false,
        windowSize: Int? = nil
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let window = windowSize ?? maxWindowSize
        let offsetValue = min(maxWindowSize - 1, offsetCursor)
        let total = offsetValue + sequenceLength

        var rinds = MLXArray(Int32(0) ..< Int32(total))
        var linds: MLXArray
        if offsetValue > 0 {
            linds = MLXArray(Int32(offsetValue) ..< Int32(offsetValue + sequenceLength))
        } else {
            linds = rinds
        }

        linds = linds[0..., .newAxis]
        rinds = rinds[.newAxis]

        var mask = linds .>= rinds
        mask = mask & (linds .< (rinds + MLXArray(window)))

        var adjustedLeftPadding = leftPadding
        let trimSize = idx - maxWindowSize + (sequenceLength > 1 ? 1 : 0)
        if trimSize > 0 {
            adjustedLeftPadding = adjustedLeftPadding - MLXArray(trimSize)
        }

        let isRotated = sequenceLength == 1 && (rotated || idx >= maxWindowSize)
        if isRotated {
            adjustedLeftPadding = adjustedLeftPadding - MLXArray(1)
        }

        var mask4d = mask[.newAxis, .newAxis, 0..., 0...]
        let columnIndices = MLXArray(Int32(0) ..< Int32(total))[.newAxis, .newAxis, .newAxis, 0...]
        let leftMask = adjustedLeftPadding[0..., .newAxis, .newAxis, .newAxis] .<= columnIndices
        mask4d = mask4d & leftMask

        if isRotated {
            var shift = idx
            if shift >= maxWindowSize {
                shift = 0
            }
            mask4d = MLX.roll(mask4d, shift: shift + 1, axis: -1)
        }

        return .array(mask4d)
    }

    private func ensureTemporalOrder() {
        guard rotated, let keys, let values else { return }
        self.keys = MLX.roll(keys, shift: -idx, axis: 2)
        self.values = MLX.roll(values, shift: -idx, axis: 2)
        idx = self.keys?.dim(2) ?? 0
        rotated = false
    }

    private func trim(array: MLXArray, trimSize: Int, append: MLXArray? = nil) -> MLXArray {
        var result = array
        if trimSize > 0 {
            result = result[.ellipsis, trimSize..., 0...]
        }
        if let append {
            result = MLX.concatenated([result, append], axis: 2)
        }
        return result
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            ensureTemporalOrder()
            if let currentKeys = self.keys, currentKeys.dim(2) > idx {
                self.keys = currentKeys[.ellipsis, ..<idx, 0...]
            }
            if let currentValues = self.values, currentValues.dim(2) > idx {
                self.values = currentValues[.ellipsis, ..<idx, 0...]
            }

            if let pendingLengths,
                let currentKeys = self.keys,
                let currentValues = self.values
            {
                let roll = maximum(MLXArray(0), offsets - pendingLengths)
                self.keys = dynamicRoll(currentKeys, shifts: roll, axis: 2)
                self.values = dynamicRoll(currentValues, shifts: roll, axis: 2)
                leftPadding = leftPadding + roll
                offsets = offsets - roll
            }

            let trimSize = idx - maxWindowSize + 1
            if trimSize > 0 {
                leftPadding = leftPadding - MLXArray(trimSize)
                if let currentKeys = self.keys {
                    self.keys = trim(array: currentKeys, trimSize: trimSize, append: keys)
                }
                if let currentValues = self.values {
                    self.values = trim(array: currentValues, trimSize: trimSize, append: values)
                }
            } else {
                self.keys = trim(array: self.keys!, trimSize: 0, append: keys)
                self.values = trim(array: self.values!, trimSize: 0, append: values)
            }
        }

        offsetCursor += keys.dim(2)
        offsets = offsets + MLXArray(keys.dim(2))
        idx = self.keys?.dim(2) ?? 0
        offset = offsetCursor

        guard let currentKeys = self.keys, let currentValues = self.values else {
            fatalError("BatchRotatingKVCache failed to produce backing storage")
        }
        return (currentKeys, currentValues)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if pendingLengths != nil {
            fatalError("finalizeBatchPrefill() must be called before decode with BatchRotatingKVCache")
        }

        let tokenCount = keys.dim(2)
        let batch = keys.dim(0)
        let kvHeads = keys.dim(1)
        let keyDim = keys.dim(3)
        let valueDim = values.dim(3)
        let previous = offsetCursor

        if self.keys == nil
            || (previous >= (self.keys?.dim(2) ?? 0) && (self.keys?.dim(2) ?? 0) < maxWindowSize)
        {
            let newSize = min(step, maxWindowSize - previous)
            let newKeys = MLXArray.zeros([batch, kvHeads, newSize, keyDim], dtype: keys.dtype)
            let newValues = MLXArray.zeros([batch, kvHeads, newSize, valueDim], dtype: values.dtype)
            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = MLX.concatenated([currentKeys, newKeys], axis: 2)
                self.values = MLX.concatenated([currentValues, newValues], axis: 2)
            } else {
                self.keys = newKeys
                self.values = newValues
            }
        }

        guard var currentKeys = self.keys, var currentValues = self.values else {
            fatalError("BatchRotatingKVCache missing storage during in-place update")
        }

        let trimSize = currentKeys.dim(2) - maxWindowSize
        if trimSize > 0 {
            currentKeys = trim(array: currentKeys, trimSize: trimSize)
            currentValues = trim(array: currentValues, trimSize: trimSize)
            self.keys = currentKeys
            self.values = currentValues
            idx = maxWindowSize
            leftPadding = leftPadding - MLXArray(trimSize)
        }

        if idx == maxWindowSize {
            rotated = true
            idx = 0
        }
        if rotated {
            leftPadding = leftPadding - MLXArray(tokenCount)
        }

        self.keys?[.ellipsis, idx ..< (idx + tokenCount), 0...] = keys
        self.values?[.ellipsis, idx ..< (idx + tokenCount), 0...] = values

        offsetCursor += tokenCount
        offsets = offsets + MLXArray(tokenCount)
        idx += tokenCount
        offset = offsetCursor

        if offsetCursor < maxWindowSize {
            return (
                self.keys![.ellipsis, ..<offsetCursor, 0...],
                self.values![.ellipsis, ..<offsetCursor, 0...]
            )
        }

        return (self.keys!, self.values!)
    }
}

private func dynamicRoll(_ x: MLXArray, shifts: MLXArray, axis: Int) -> MLXArray {
    let normalizedAxis = axis >= 0 ? axis : x.ndim + axis
    let size = x.dim(normalizedAxis)

    var indexShape = Array(repeating: 1, count: x.ndim)
    indexShape[0] = x.dim(0)
    indexShape[normalizedAxis] = size

    let base = broadcast(
        MLXArray(Int32(0) ..< Int32(size)).reshaped(indexShape),
        to: x.shape
    )

    var shiftShape = Array(repeating: 1, count: x.ndim)
    shiftShape[0] = x.dim(0)
    let expandedShifts = broadcast(shifts.reshaped(shiftShape), to: x.shape)

    let indices = (base - expandedShifts.asType(.int32) + Int32(size)) % Int32(size)
    return take(x, indices, axis: normalizedAxis)
}
