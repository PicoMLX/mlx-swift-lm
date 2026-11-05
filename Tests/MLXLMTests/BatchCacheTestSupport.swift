// Copyright © 2026 Apple Inc.

import MLX

/// Keys and values with the same value at every position. Use this only for shape and
/// counter checks: it cannot show which positions an operation kept or their order.
func makeKV(
    batchSize: Int, heads: Int, seqLen: Int, headDim: Int, value: Float = 1
) -> (MLXArray, MLXArray) {
    let keys = MLXArray.ones([batchSize, heads, seqLen, headDim]) * value
    let values = MLXArray.ones([batchSize, heads, seqLen, headDim]) * (value + 1)
    return (keys, values)
}

/// Keys of `i + 1` and values of `(i + 1) * 10` in row `i`.
func makeDistinctKV(batchSize: Int, heads: Int, seqLen: Int, headDim: Int) -> (MLXArray, MLXArray) {
    let rows = MLXArray((0 ..< batchSize).map { Float($0 + 1) }).reshaped([batchSize, 1, 1, 1])
    let ones = MLXArray.ones([batchSize, heads, seqLen, headDim])
    return (ones * rows, ones * rows * 10)
}

/// Keys stamped with their position (plus `row * rowStride`), and values with the
/// key plus 100, so tests can check which positions survive and in what order.
func makePositionKV(
    positions: Range<Int>, heads: Int, headDim: Int, batchSize: Int = 1, rowStride: Float = 0
) -> (MLXArray, MLXArray) {
    var stamps = [Float]()
    for row in 0 ..< batchSize {
        stamps.append(contentsOf: positions.map { Float($0) + Float(row) * rowStride })
    }
    let stamp = MLXArray(stamps).reshaped([batchSize, 1, positions.count, 1])
    let ones = MLXArray.ones([batchSize, heads, positions.count, headDim])
    return (ones * stamp, ones * (stamp + 100))
}

func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
    abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
}

func batchRows(_ indices: Int...) -> MLXArray {
    MLXArray(indices.map { Int32($0) })
}
