// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - applyRotaryPosition Helper

/// Apply rotary position embeddings, dispatching to the appropriate offset type
/// based on the cache.
///
/// - For `BatchPositionedKVCache`: uses `ArrayOffsetLayer` with per-sequence
///   `MLXArray` offsets for batched inference.
/// - For single caches (non-batch): uses `OffsetLayer` with scalar `Int` offset.
/// - For `nil` cache: uses `OffsetLayer` with offset `0`.
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - x: The input tensor to apply RoPE to.
///   - cache: The KV cache (determines offset type), or `nil` for offset 0.
/// - Returns: The input with rotary positional encoding applied.
@available(*, deprecated, message: "use applyRotaryPosition(_:to:offset:) instead")
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?)
    -> MLXArray
{
    applyRotaryPosition(rope, to: x, offset: cache?.ropeOffset)
}

/// Apply rotary position embeddings, using the cache offset when available.
///
/// This function enables models to use a single call site instead of
/// repeating conditional offset handling:
///
/// ```swift
/// let offset = cache?.ropeOffset
/// queries = applyRotaryPosition(rope, to: queries, offset: offset)
/// keys = applyRotaryPosition(rope, to: keys, offset: offset)
/// ```
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - x: The input tensor to apply RoPE to.
///   - offset: the offset into the rotary positional encoding.  0 if nil.
/// - Returns: The input with rotary positional encoding applied.
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, offset: RoPEOffset?)
    -> MLXArray
{
    switch offset {
    case nil:
        rope(x, offset: 0)
    case .scalar(let v):
        rope(x, offset: v)
    case .batch(let v):
        rope(x, offset: v)
    }
}
