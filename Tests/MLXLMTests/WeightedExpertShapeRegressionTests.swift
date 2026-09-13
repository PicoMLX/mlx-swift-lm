import MLX
import MLXLMCommon
import Testing

@Suite(.serialized)
struct WeightedExpertShapeRegressionTests {
    @Test("Expert reduction remains correct when the routed expert count changes")
    func changingExpertCount() {
        Device.withDefaultDevice(.gpu) {
            // Distinct ranks keep both first-call orders independent of other
            // model tests that may have warmed the process-wide compiled function.
            for (leadingRank, counts) in [(5, [1, 4, 2, 8, 1]), (6, [4, 1, 8, 2, 4])] {
                for dtype in [DType.float32, .float16, .bfloat16] {
                    for (index, k) in counts.enumerated() {
                        let rows = index.isMultiple(of: 2) ? 17 : 3
                        let prefix = Array(repeating: 1, count: leadingRank) + [rows]
                        let values =
                            (MLXArray(Array(1 ... (rows * k * 16))).asType(.float32) / 1024)
                            .asType(dtype).reshaped(prefix + [k, 16])
                        let weights = (MLXArray.ones(prefix + [k]) / Float(k)).asType(dtype)
                        let expected = (values * weights.expandedDimensions(axis: -1)).sum(axis: -2)
                        let actual = weightedExpertSum(values, weights)
                        eval(actual, expected)
                        #expect(actual.shape == prefix + [16])
                        #expect(actual.dtype == expected.dtype)
                        #expect(
                            actual.asType(.float32).asArray(Float.self)
                                == expected.asType(.float32).asArray(Float.self))
                    }
                }
            }
        }
    }
}
