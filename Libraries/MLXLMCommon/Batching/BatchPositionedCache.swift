// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - BatchPositionedKVCache Protocol

/// Protocol for batch-aware KV caches that provide per-sequence positional offsets.
///
/// When applying rotary position embeddings (RoPE) in a batched context, each
/// sequence in the batch may be at a different position. This protocol provides
/// the per-sequence offsets as an `MLXArray` so that RoPE can be applied with
/// different offsets per batch element.
///
/// Conforming types expose `batchOffset: MLXArray` of shape `[B]` containing
/// the current position offset for each sequence in the batch.
public protocol BatchPositionedKVCache: KVCache {
    /// Per-sequence position offsets as an MLXArray of shape `[B]`.
    ///
    /// For a batch of sequences that have been prefilled to different lengths,
    /// this array contains the effective position index for each sequence,
    /// accounting for left-padding.
    var batchOffset: MLXArray { get }
}

// MARK: - isBatchCompatible

/// Check whether a list of per-layer caches is compatible with batch KV cache
/// merge/extend operations.
///
/// Returns `false` for:
/// - `CacheList` (composite caches used by hybrid models like Jamba)
/// - `MambaCache` (SSM state-space caches, not key-value based)
/// - `QuantizedKVCache` (stores quantized tuples incompatible with batch merge/extend)
///
/// Returns `true` for:
/// - `KVCacheSimple` (standard transformer KV cache)
/// - `RotatingKVCache` (sliding-window attention cache)
/// - Empty cache arrays
///
/// - Parameter caches: The per-layer cache array to check.
/// - Returns: `true` if all caches support batch operations, `false` otherwise.
public func isBatchCompatible(_ caches: [KVCache]) -> Bool {
    for cache in caches {
        if cache is CacheList || cache is MambaCache || cache is QuantizedKVCache {
            return false
        }
    }
    return true
}
