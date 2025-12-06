// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Swift implementation of Python's `BatchKVCache`
/// (`mlx_lm/models/cache.py:662-781`). Handles left-padded batched KV storage,
/// dynamic capacity growth, filtering, and concatenation to mirror the Python
/// batching behaviour.
public class BatchKVCache: BaseKVCache {

    public var step = 256

    private var keys: MLXArray?
    private var values: MLXArray?
    /// Per-batch sequence offsets for RoPE positions
    public private(set) var offsets: MLXArray
    private var leftPadding: MLXArray
    private var currentLength: Int = 0

    public override var isTrimmable: Bool { true }

    public init(leftPadding: [Int]) {
        self.leftPadding = MLXArray(leftPadding)
        self.offsets = MLXArray(leftPadding.map { -$0 })
        super.init()
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = currentLength
        let tokenCount = keys.dim(2)

        ensureCapacity(for: keys, values: values, previousLength: previous)

        guard self.keys != nil, self.values != nil else {
            fatalError("BatchKVCache storage was not initialized before update")
        }

        currentLength += tokenCount

        // Update per-batch offsets
        offsets = offsets + tokenCount

        // For batch mode, set self.offset based on currentLength
        // Python tracks this as self._idx, avoiding GPU sync
        self.offset = currentLength

        self.keys?[.ellipsis, previous ..< currentLength, 0...] = keys
        self.values?[.ellipsis, previous ..< currentLength, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<currentLength, 0...]
        let returnedValues = self.values![.ellipsis, ..<currentLength, 0...]

        return (returnedKeys, returnedValues)
    }

    public override var state: [MLXArray] {
        get {
            guard var keys = self.keys, var values = self.values else { return [] }
            if currentLength < keys.dim(2) {
                keys = keys[.ellipsis, ..<currentLength, 0...]
                values = values[.ellipsis, ..<currentLength, 0...]
            }
            return [keys, values, offsets, leftPadding]
        }
        set {
            guard newValue.count >= 4 else {
                self.keys = nil
                self.values = nil
                self.offsets = MLXArray([])
                self.leftPadding = MLXArray([])
                currentLength = 0
                self.offset = 0
                return
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            self.offsets = newValue[2]
            self.leftPadding = newValue[3]
            currentLength = self.keys?.dim(2) ?? 0
            self.offset = currentLength
        }
    }

    public func filter(batchIndices: MLXArray) {
        guard self.keys != nil, self.values != nil else { return }

        if batchIndices.size == 0 {
            self.keys = nil
            self.values = nil
            offsets = MLXArray([])
            leftPadding = MLXArray([])
            currentLength = 0
            self.offset = 0
            return
        }

        self.keys = self.keys?[batchIndices, .ellipsis]
        self.values = self.values?[batchIndices, .ellipsis]
        offsets = offsets[batchIndices]
        leftPadding = leftPadding[batchIndices]

        let minLeftPad = leftPadding.min().item(Int.self)
        if minLeftPad > 0 {
            let shift = min(minLeftPad, currentLength)
            if shift > 0 {
                self.keys = self.keys?[.ellipsis, shift..., 0...]
                self.values = self.values?[.ellipsis, shift..., 0...]
                currentLength -= shift
                self.offset = currentLength
            }
            leftPadding = leftPadding - MLXArray(minLeftPad)
        }
    }

    public func extend(other: BatchKVCache) {
        guard let otherKeys = other.keys, let otherValues = other.values else { return }

        if keys == nil {
            self.keys = otherKeys
            self.values = otherValues
            self.offsets = other.offsets
            self.leftPadding = other.leftPadding
            self.currentLength = other.currentLength
            self.offset = currentLength
            return
        }

        guard let currentKeys = self.keys else {
            fatalError("BatchKVCache missing backing storage during extend")
        }

        let maxIndex = max(currentLength, other.currentLength)
        let maxSize = max(currentKeys.dim(2), otherKeys.dim(2))

        let selfPadded = paddedForConcat(targetLength: maxIndex, targetCapacity: maxSize)
        let otherPadded = other.paddedForConcat(targetLength: maxIndex, targetCapacity: maxSize)

        self.keys = MLX.concatenated([selfPadded.keys, otherPadded.keys], axis: 0)
        self.values = MLX.concatenated([selfPadded.values, otherPadded.values], axis: 0)
        self.offsets = MLX.concatenated([selfPadded.offsets, otherPadded.offsets], axis: 0)
        self.leftPadding = MLX.concatenated([selfPadded.leftPadding, otherPadded.leftPadding], axis: 0)
        self.currentLength = maxIndex
        self.offset = currentLength
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(currentLength, n)
        if trimmed == 0 { return 0 }
        currentLength -= trimmed
        self.offset = currentLength
        offsets = offsets - trimmed
        return trimmed
    }

    public func makeMask(
        sequenceLength: Int,
        returnArray: Bool = false,
        windowSize: Int? = nil
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        var mask = createCausalMask(
            n: sequenceLength,
            offset: currentLength,
            windowSize: windowSize
        )
        mask = mask[.newAxis, .newAxis, 0..., 0...]

        if leftPadding.size > 0 {
            let total = currentLength + sequenceLength
            let columnIndices = MLXArray(Int32(0) ..< Int32(total))[.newAxis, .newAxis, .newAxis, 0...]
            let leftMask = leftPadding[0..., .newAxis, .newAxis, .newAxis] .<= columnIndices
            mask = mask & leftMask
        }

        return .array(mask)
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

        let kShape = [batch, kvHeads, capacity, keyDim]
        let vShape = [batch, kvHeads, capacity, valueDim]

        return (
            MLXArray.zeros(kShape, dtype: keys.dtype),
            MLXArray.zeros(vShape, dtype: values.dtype)
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
            let padSpec: [IntOrPair] = [
                IntOrPair(0),
                IntOrPair(0),
                IntOrPair((left, right)),
                IntOrPair(0),
            ]
            keys = MLX.padded(keys, widths: padSpec)
            values = MLX.padded(values, widths: padSpec)
        }

        let adjustedLeftPadding = leftPadding + MLXArray(left)
        return (keys, values, offsets, adjustedLeftPadding)
    }
}

/// Batch-aware rotating cache mirroring Python's `BatchRotatingKVCache`
/// (`mlx_lm/models/cache.py:797-1007`). Supports sliding-window attention for
/// batched prompts with left padding.
public class BatchRotatingKVCache: BaseKVCache {

    public var step = 256

    private let maxWindowSize: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    private var leftPadding: MLXArray
    public private(set) var offsets: MLXArray
    private var idx: Int = 0
    private var offsetCursor: Int = 0
    private var rotated: Bool = false

    public override var isTrimmable: Bool { offsetCursor < maxWindowSize }

    public init(maxSize: Int, leftPadding: [Int]) {
        self.maxWindowSize = maxSize
        self.leftPadding = MLXArray(leftPadding)
        self.offsets = MLXArray(leftPadding.map { -$0 })
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
                self.keys = nil
                self.values = nil
                self.offsets = MLXArray([])
                self.leftPadding = MLXArray([])
                self.idx = 0
                self.offsetCursor = 0
                self.rotated = false
                self.offset = 0
                return
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            self.offsets = newValue[2]
            self.leftPadding = newValue[3]
            self.offsetCursor = self.keys?.dim(2) ?? 0
            self.idx = self.offsetCursor
            self.offset = offsetCursor
            self.rotated = false
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
            guard let parsedOffset = Int(newValue[1]),
                let parsedIdx = Int(newValue[2])
            else {
                fatalError("Invalid offset/index in BatchRotatingKVCache metaState")
            }
            self.offsetCursor = parsedOffset
            self.idx = parsedIdx
            self.rotated = newValue[3] == "1"
            self.offset = offsetCursor
        }
    }

    public override var maxSize: Int? { maxWindowSize }

    public func filter(batchIndices: MLXArray) {
        guard self.keys != nil, self.values != nil else { return }

        if batchIndices.size == 0 {
            self.keys = nil
            self.values = nil
            offsets = MLXArray([])
            leftPadding = MLXArray([])
            offsetCursor = 0
            idx = 0
            rotated = false
            self.offset = 0
            return
        }

        self.keys = self.keys?[batchIndices, .ellipsis]
        self.values = self.values?[batchIndices, .ellipsis]
        offsets = offsets[batchIndices]
        leftPadding = leftPadding[batchIndices]
    }

    public func extend(other: BatchRotatingKVCache) {
        guard let otherKeys = other.keys, let otherValues = other.values else { return }

        if keys == nil {
            self.keys = otherKeys
            self.values = otherValues
            self.offsets = other.offsets
            self.leftPadding = other.leftPadding
            self.offsetCursor = other.offsetCursor
            self.idx = other.idx
            self.rotated = other.rotated
            self.offset = offsetCursor
            return
        }

        ensureTemporalOrder()
        other.ensureTemporalOrder()

        guard let currentKeys = self.keys, let currentValues = self.values else {
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

            let adjustedLeft = cache.leftPadding + MLXArray(left)
            return (keys, values, cache.offsets, adjustedLeft)
        }

        let paddedSelf = pad(self)
        let paddedOther = pad(other)

        self.keys = MLX.concatenated([paddedSelf.keys, paddedOther.keys], axis: 0)
        self.values = MLX.concatenated([paddedSelf.values, paddedOther.values], axis: 0)
        self.offsets = MLX.concatenated([paddedSelf.offsets, paddedOther.offsets], axis: 0)
        self.leftPadding = MLX.concatenated([paddedSelf.leftPadding, paddedOther.leftPadding], axis: 0)

        self.idx = maxIndex
        self.offsetCursor = max(self.offsetCursor, other.offsetCursor)
        self.rotated = false
        self.offset = offsetCursor
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offsetCursor, n)
        if trimmed == 0 { return 0 }

        offsetCursor -= trimmed
        idx = max(idx - trimmed, 0)
        offsets = offsets - MLXArray(trimmed)
        self.offset = offsetCursor
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
        guard rotated, let keys = self.keys, let values = self.values else { return }
        self.keys = MLX.roll(keys, shift: -idx, axis: 2)
        self.values = MLX.roll(values, shift: -idx, axis: 2)
        self.idx = self.keys?.dim(2) ?? 0
        self.rotated = false
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
        self.offset = offsetCursor

        guard let currentKeys = self.keys, let currentValues = self.values else {
            fatalError("BatchRotatingKVCache failed to produce backing storage")
        }
        return (currentKeys, currentValues)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
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
            let kShape = [batch, kvHeads, newSize, keyDim]
            let vShape = [batch, kvHeads, newSize, valueDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)
            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = MLX.concatenated([currentKeys, newK], axis: 2)
                self.values = MLX.concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
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
        self.offset = offsetCursor

        if offsetCursor < maxWindowSize {
            let slices = offsetCursor
            return (
                self.keys![.ellipsis, ..<slices, 0...],
                self.values![.ellipsis, ..<slices, 0...]
            )
        }

        return (self.keys!, self.values!)
    }
}
