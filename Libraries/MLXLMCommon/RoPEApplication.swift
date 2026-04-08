// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - BatchPositionedKVCache

/// Protocol for KV caches that expose per-sequence RoPE offsets.
///
/// This is a forward-compatible hook for batched caches. Current scalar-cache
/// code paths continue using `KVCache.offset`.
public protocol BatchPositionedKVCache: KVCache {
    /// Per-sequence RoPE offsets with shape `[B]`.
    var batchOffset: MLXArray { get }
}

// MARK: - applyRotaryPosition Helper

/// Apply rotary position embeddings, dispatching to the appropriate offset type
/// based on the cache.
///
/// - For `BatchPositionedKVCache`: uses `ArrayOffsetLayer` with per-sequence
///   `MLXArray` offsets for batched inference.
/// - For single caches (non-batch): uses `OffsetLayer` with scalar `Int` offset.
/// - For `nil` cache: uses `OffsetLayer` with offset `0`.
///
/// This function enables models to use a single call site instead of
/// repeating conditional offset handling:
/// ```swift
/// queries = applyRotaryPosition(rope, to: queries, cache: cache)
/// keys = applyRotaryPosition(rope, to: keys, cache: cache)
/// ```
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - x: The input tensor to apply RoPE to.
///   - cache: The KV cache (determines offset type), or `nil` for offset 0.
/// - Returns: The input with rotary positional encoding applied.
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?)
    -> MLXArray
{
    if let batchCache = cache as? BatchPositionedKVCache {
        return rope(x, offset: batchCache.batchOffset)
    } else {
        return rope(x, offset: cache?.offset ?? 0)
    }
    if let batchCache = cache as? BatchPositionedKVCache {
        return rope(x, offset: batchCache.batchOffset)
    } else {
        return rope(x, offset: cache?.offset ?? 0)
    }
}
