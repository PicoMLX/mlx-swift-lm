// Copyright © 2026 Apple Inc.

import Foundation
import MLX

extension DType {
    /// The largest finite value representable in this float dtype
    /// (the analogue of Python's `mx.finfo(dtype).max`).
    ///
    /// Use it negated as the fill for masked attention scores, so masked
    /// positions collapse to ~0 after softmax. Prefer `MLXArray.maskFill(for:)`,
    /// which builds the scalar directly in the target dtype.
    ///
    /// It's per-dtype rather than `Float.greatestFiniteMagnitude` so the value
    /// stays finite in `.float16`/`.bfloat16` instead of overflowing to `inf`
    /// (a `-inf` fill would NaN an all-masked softmax row).
    public var greatestFiniteMagnitude: Float {
        switch self {
        case .float16: return Float(Float16.greatestFiniteMagnitude)  // 65504
        case .bfloat16: return Float(bitPattern: 0x7F7F_0000)  // ≈ 3.3895e38
        default: return .greatestFiniteMagnitude  // float32 and wider
        }
    }
}

extension MLXArray {
    /// A scalar masked-score fill — `-finfo(dtype).max` constructed directly in
    /// `dtype`, so masked positions vanish under softmax with nothing to cast.
    ///
    /// ```swift
    /// scores = MLX.where(causalMask, scores, MLXArray.maskFill(for: scores.dtype))
    /// ```
    public static func maskFill(for dtype: DType) -> MLXArray {
        MLXArray(-dtype.greatestFiniteMagnitude, dtype: dtype)
    }
}
