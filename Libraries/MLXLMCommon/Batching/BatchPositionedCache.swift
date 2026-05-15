// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

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
