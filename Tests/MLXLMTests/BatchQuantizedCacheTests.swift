// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// MARK: - BatchQuantizedKVCache

@Suite(.serialized)
struct BatchQuantizedKVCacheTests {

    @Test("Lifecycle covers updateQuantized, filter, extend, and extract")
    func lifecycleRoundTrip() throws {
        let cache = BatchQuantizedKVCache(leftPadding: [0, 0], groupSize: 32, bits: 8)
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 32)
        let (qKeys, qValues) = cache.updateQuantized(keys: keys, values: values)

        #expect(qKeys.0.dim(0) == 2)
        #expect(qValues.0.dim(-2) == 3)
        #expect(cache.offset == 3)
        #expect(cache.batchOffsets.asArray(Int32.self) == [3, 3])

        cache.filter(batchIndices: [1])
        #expect(cache.batchSize == 1)

        // Row 1 carried the value 2 (keys) / 20 (values); the extracted single
        // cache must dequantize back to those constants.
        let extracted = cache.extract(idx: 0)
        #expect(extracted.offset == 3)
        let (xk, xv) = try #require(extracted.getQuantizedState())
        let dk = dequantized(
            xk.0, scales: xk.1, biases: xk.2, groupSize: cache.groupSize, bits: cache.bits,
            mode: cache.mode)
        let dv = dequantized(
            xv.0, scales: xv.1, biases: xv.2, groupSize: cache.groupSize, bits: cache.bits,
            mode: cache.mode)
        #expect(maxAbsDifference(dk, MLXArray.ones([1, 2, 3, 32]) * 2) < 1e-2)
        #expect(maxAbsDifference(dv, MLXArray.ones([1, 2, 3, 32]) * 20) < 1e-1)
    }

    @Test("Extend right-justifies the shorter cache with left padding")
    func extendRightJustifies() {
        let cache = BatchQuantizedKVCache(leftPadding: [0], groupSize: 32, bits: 8)
        let (k1, v1) = makeKV(batchSize: 1, heads: 2, seqLen: 4, headDim: 32, value: 1)
        _ = cache.updateQuantized(keys: k1, values: v1)

        let other = BatchQuantizedKVCache(leftPadding: [0], groupSize: 32, bits: 8)
        let (k2, v2) = makeKV(batchSize: 1, heads: 2, seqLen: 2, headDim: 32, value: 5)
        _ = other.updateQuantized(keys: k2, values: v2)

        cache.extend(other: other)

        #expect(cache.batchSize == 2)
        #expect(cache.offset == 4)
        #expect(cache.leftPadding.asArray(Int32.self) == [0, 2])
        // Logical positions are per-row and unaffected by re-justification.
        #expect(cache.batchOffsets.asArray(Int32.self) == [4, 2])
    }

    @Test("Filter shifts out shared left padding")
    func filterShiftsPadding() {
        let cache = BatchQuantizedKVCache(leftPadding: [0], groupSize: 32, bits: 8)
        let (k1, v1) = makeKV(batchSize: 1, heads: 2, seqLen: 4, headDim: 32, value: 1)
        _ = cache.updateQuantized(keys: k1, values: v1)

        let other = BatchQuantizedKVCache(leftPadding: [0], groupSize: 32, bits: 8)
        let (k2, v2) = makeKV(batchSize: 1, heads: 2, seqLen: 2, headDim: 32, value: 5)
        _ = other.updateQuantized(keys: k2, values: v2)
        cache.extend(other: other)

        // Keep only the padded row; its 2 columns of padding shift out.
        cache.filter(batchIndices: [1])
        #expect(cache.batchSize == 1)
        #expect(cache.leftPadding.asArray(Int32.self) == [0])
        #expect(cache.offset == 2)
        #expect(cache.batchOffsets.asArray(Int32.self) == [2])
    }

    @Test("makeMask honours left padding during decode")
    func makeMaskHonoursLeftPadding() throws {
        let cache = BatchQuantizedKVCache(leftPadding: [0, 2], groupSize: 32, bits: 8)
        let (keys, values) = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 32)
        _ = cache.updateQuantized(keys: keys, values: values)

        guard case .array(let mask) = cache.makeMask(n: 1, windowSize: nil, returnArray: true)
        else {
            Issue.record("expected an array mask")
            return
        }
        // Mask spans the existing cache plus the incoming token: width 4.
        #expect(mask.dim(-1) == 4)
        // Row 0 attends everything; row 1's first two positions are padding.
        #expect(mask[0, 0, 0, 0].item(Bool.self) == true)
        #expect(mask[1, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[1, 0, 0, 1].item(Bool.self) == false)
        #expect(mask[1, 0, 0, 2].item(Bool.self) == true)
    }

    @Test("fromSingle/toSingle preserve offset and data")
    func fromSingleToSingleRoundTrip() throws {
        let single = QuantizedKVCache(groupSize: 32, bits: 8)
        let (keys, values) = makeKV(batchSize: 1, heads: 2, seqLen: 5, headDim: 32, value: 3)
        _ = single.updateQuantized(keys: keys, values: values)

        let batched = BatchQuantizedKVCache.fromSingle(single)
        #expect(batched.batchSize == 1)
        #expect(batched.offset == 5)
        #expect(batched.batchOffsets.asArray(Int32.self) == [5])

        let restored = batched.toSingle()
        #expect(restored.offset == 5)
        let (rk, _) = try #require(restored.getQuantizedState())
        let dk = dequantized(
            rk.0, scales: rk.1, biases: rk.2, groupSize: 32, bits: 8, mode: .affine)
        #expect(maxAbsDifference(dk, MLXArray.ones([1, 2, 5, 32]) * 3) < 1e-2)
    }

    @Test("prepare/finalize roll right padding into left padding")
    func prepareFinalizeRollsPadding() {
        let cache = BatchQuantizedKVCache(leftPadding: [0, 0], groupSize: 32, bits: 8)
        cache.prepareBatched(leftPadding: nil, lengths: [4, 2], rightPadding: [0, 2])

        let (keys, values) = makeKV(batchSize: 2, heads: 2, seqLen: 4, headDim: 32)
        _ = cache.updateQuantized(keys: keys, values: values)
        cache.finalizeBatched()

        #expect(cache.leftPadding.asArray(Int32.self) == [0, 2])
        // Row 1 saw only 2 real tokens; its logical position rolls back to 2.
        #expect(cache.batchOffsets.asArray(Int32.self) == [4, 2])
    }

    @Test("trim rewinds the write position and per-row offsets")
    func trimRewinds() {
        let cache = BatchQuantizedKVCache(leftPadding: [0], groupSize: 32, bits: 8)
        let (keys, values) = makeKV(batchSize: 1, heads: 2, seqLen: 4, headDim: 32)
        _ = cache.updateQuantized(keys: keys, values: values)

        #expect(cache.isTrimmable)
        let trimmed = cache.trim(2)
        #expect(trimmed == 2)
        #expect(cache.offset == 2)
        #expect(cache.batchOffsets.asArray(Int32.self) == [2])
    }

    @Test("trim clamps to the minimum live per-row length")
    func trimClampsToMinimumLiveRow() {
        // Ragged batch: leftPadding [4, 0], then prefill 6 tokens. Row offsets
        // become [2, 6] (live lengths 2 and 6). Trimming by 5 must not drive the
        // shorter row's offset negative; the trim is clamped to the min live row
        // length (2) so no row underflows.
        let cache = BatchQuantizedKVCache(leftPadding: [4, 0], groupSize: 32, bits: 8)
        let (keys, values) = makeKV(batchSize: 2, heads: 2, seqLen: 6, headDim: 32)
        _ = cache.updateQuantized(keys: keys, values: values)
        #expect(cache.batchOffsets.asArray(Int32.self) == [2, 6])

        let trimmed = cache.trim(5)
        #expect(trimmed == 2)
        #expect(cache.offset == 4)
        // No row offset is driven negative.
        #expect(cache.batchOffsets.asArray(Int32.self) == [0, 4])

        // Extraction still round-trips after the clamped trim: the fully
        // rewound row extracts as an empty cache, the longer row keeps its
        // remaining live tokens.
        #expect(cache.extract(idx: 0).offset == 0)
        #expect(cache.extract(idx: 1).offset == 4)
    }

    @Test("Extract clamps a row still inside its left-padding prefix")
    func extractClampsRowInsidePaddingPrefix() throws {
        // During chunked prefill the first chunk can leave `_idx` short of a
        // ragged row's left padding; `filter`'s `_idx >= minLeftPad` guard then
        // legitimately preserves the row with leftPadding > _idx. Extracting
        // that row must yield an empty cache instead of trapping on the
        // invalid `padding ..< _idx` slice.
        let cache = BatchQuantizedKVCache(leftPadding: [0, 0], groupSize: 32, bits: 8)
        cache.prepareBatched(leftPadding: [4, 0], lengths: nil, rightPadding: nil)
        let (keys, values) = makeKV(batchSize: 2, heads: 2, seqLen: 2, headDim: 32)
        _ = cache.updateQuantized(keys: keys, values: values)

        // Keep only the padded row: minLeftPad (4) exceeds _idx (2), so the
        // padding shift is skipped and the row keeps leftPadding > _idx.
        cache.filter(batchIndices: [0])
        #expect(cache.leftPadding.asArray(Int32.self) == [4])
        #expect(cache.offset == 2)

        let extracted = cache.extract(idx: 0)
        #expect(extracted.offset == 0)
        let (xk, xv) = try #require(extracted.getQuantizedState())
        #expect(xk.0.dim(-2) == 0)
        #expect(xv.0.dim(-2) == 0)
    }
}

// MARK: - Quantized attention masking

@Suite(.serialized)
struct QuantizedAttentionMaskTests {

    @Test("Boolean mask suppresses masked positions")
    func booleanMaskSuppresses() {
        // All true scores are negative, so the pre-fix fill value (~+0.0)
        // would dominate the softmax and pull the output toward the masked
        // value row. With the fix the masked key contributes ~0 weight and the
        // output is the unmasked value row.
        let queries = MLXArray.ones([1, 2, 1, 32])
        let keys = concatenated(
            [MLXArray.ones([1, 2, 1, 32]) * -1, MLXArray.ones([1, 2, 1, 32]) * 3], axis: 2)
        let values = concatenated(
            [MLXArray.ones([1, 2, 1, 32]) * 5, MLXArray.ones([1, 2, 1, 32]) * 9], axis: 2)

        let qk = quantized(keys, groupSize: 32, bits: 8)
        let qv = quantized(values, groupSize: 32, bits: 8)

        // [B, 1, L, S] boolean mask: only key 0 is visible.
        let mask = MLXArray(converting: [1.0, 0.0]).reshaped([1, 1, 1, 2]) .> 0.5

        let output = quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: (qk.wq, qk.scales, qk.biases),
            quantizedValues: (qv.wq, qv.scales, qv.biases),
            scale: 1.0,
            mask: .array(mask),
            groupSize: 32,
            bits: 8
        )

        #expect(maxAbsDifference(output, MLXArray.ones([1, 2, 1, 32]) * 5) < 1e-1)
    }

    @Test("Batched 4-D mask broadcasts over grouped query heads")
    func batchedMaskBroadcastsOverGQA() {
        // nQHeads 4 over nKVHeads 2 → nRepeats 2, so scores are 5-D and a
        // [B, 1, L, S] mask needs the inserted group axis to broadcast.
        let queries = MLXArray.ones([2, 4, 1, 32])
        let keys = MLXArray.ones([2, 2, 3, 32])
        let values = MLXArray.ones([2, 2, 3, 32]) * 7

        let qk = quantized(keys, groupSize: 32, bits: 8)
        let qv = quantized(values, groupSize: 32, bits: 8)

        let leftPadding = MLXArray([Int32(0), Int32(2)])
        let mask = createCausalMask(n: 1, offset: 2, leftPadding: leftPadding)

        let output = quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: (qk.wq, qk.scales, qk.biases),
            quantizedValues: (qv.wq, qv.scales, qv.biases),
            scale: 1.0,
            mask: .array(mask),
            groupSize: 32,
            bits: 8
        )

        #expect(output.shape == [2, 4, 1, 32])
        // Every visible value row is the constant 7; both rows attend at least
        // their own position, so the output must be ~7 everywhere.
        #expect(maxAbsDifference(output, MLXArray.ones([2, 4, 1, 32]) * 7) < 1e-1)
    }
}

// MARK: - Helpers

private func makeKV(
    batchSize: Int,
    heads: Int,
    seqLen: Int,
    headDim: Int,
    value: Float = 1.0
) -> (MLXArray, MLXArray) {
    let keys = MLXArray.ones([batchSize, heads, seqLen, headDim]) * value
    let values = MLXArray.ones([batchSize, heads, seqLen, headDim]) * (value + 1)
    return (keys, values)
}

private func makeDistinctKV(
    batchSize: Int,
    heads: Int,
    seqLen: Int,
    headDim: Int
) -> (MLXArray, MLXArray) {
    var keysList = [MLXArray]()
    var valuesList = [MLXArray]()
    for index in 0 ..< batchSize {
        keysList.append(MLXArray.ones([1, heads, seqLen, headDim]) * Float(index + 1))
        valuesList.append(MLXArray.ones([1, heads, seqLen, headDim]) * Float((index + 1) * 10))
    }
    return (concatenated(keysList, axis: 0), concatenated(valuesList, axis: 0))
}

private func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
    abs(lhs - rhs).max().item(Float.self)
}
