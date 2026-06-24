// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// MARK: - Quantized attention masking

@Suite(.serialized)
struct QuantizedAttentionMaskTests {

    @Test("Boolean mask suppresses masked positions when true scores are negative")
    func booleanMaskSuppresses() {
        // All true scores are negative, so the pre-fix fill (~+0.0) would
        // dominate the softmax and pull the output toward the masked value row.
        // With the finite-minimum fill the masked key contributes ~0 weight, so
        // the output is the single unmasked value row.
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

    @Test("maskFill is finite and very negative for every float dtype")
    func maskFillIsFiniteAndNegative() {
        for dtype in [DType.float16, .bfloat16, .float32] {
            let value = MLXArray.maskFill(for: dtype).asType(.float32).item(Float.self)
            #expect(value.isFinite, "maskFill overflowed to -inf for \(dtype)")
            #expect(value < -1e4)
        }
    }

    @Test("All-masked row softmaxes to finite weights instead of NaN")
    func allMaskedRowStaysFinite() {
        // A fully-masked row fills every position with maskFill. A `-inf` fill
        // (what casting the Float32 max down to f16/bf16 produces) would softmax
        // to NaN; the finite minimum yields a uniform, finite distribution.
        for dtype in [DType.float16, .bfloat16] {
            let row = MLXArray.zeros([4], dtype: dtype) + MLXArray.maskFill(for: dtype)
            let sum = softmax(row, axis: -1).asType(.float32).sum().item(Float.self)
            #expect(sum.isFinite, "all-masked softmax produced NaN for \(dtype)")
            #expect(abs(sum - 1) < 1e-3)
        }
    }
}

// MARK: - Helpers

private func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
    abs(lhs - rhs).max().item(Float.self)
}
