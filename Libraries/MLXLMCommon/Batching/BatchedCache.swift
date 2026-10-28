// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// A KV cache that holds a batch of sequences as rows, with the row operations
/// continuous batching needs.
///
/// Ragged prompts are prefilled right-padded: call `prepare(lengths:)` with each row's
/// length before the prefill and ``KVCache/finalize()`` after it.
/// Conforming caches are not thread-safe.
public protocol BatchedCache: KVCache {
    /// Keep only the rows at `batchIndices`, in that order.
    func filter(batchIndices: MLXArray)

    /// Append the rows of `other`, which must be the same type of cache.
    func extend(other: any BatchedCache)

    /// The row at `index` as a single-sequence cache.
    func extract(_ index: Int) -> any KVCache
}

/// Makes the batched cache of one layer for `leftPadding.count` rows.
public typealias BatchedCacheFactory = (_ leftPadding: [Int]) -> any BatchedCache

/// A model cache that has no batched counterpart.
public enum BatchedCacheError: Error, CustomStringConvertible, Equatable {
    case unsupportedCacheTopology(layer: Int, path: String, cacheType: String, reason: String)

    public var description: String {
        switch self {
        case .unsupportedCacheTopology(let layer, let path, let cacheType, let reason):
            "Unsupported cache topology at layer \(layer), \(path): \(cacheType). \(reason)"
        }
    }
}

/// One ``BatchedCacheFactory`` per layer of `probe`, a model's single-sequence caches
/// such as those from `newCache(parameters:)`.
///
/// This checks only the cache types. A caller still has to check that the model
/// supports per-row positions.
///
/// - Throws: ``BatchedCacheError`` when a layer's cache has no batched counterpart.
public func makeBatchedCacheFactories(for probe: [any KVCache]) throws -> [BatchedCacheFactory] {
    try probe.enumerated().map { layer, cache in
        try makeBatchedCacheFactory(for: cache, layer: layer, path: "layer")
    }
}

private func makeBatchedCacheFactory(
    for cache: any KVCache, layer: Int, path: String
) throws -> BatchedCacheFactory {
    func unsupported(_ reason: String) -> BatchedCacheError {
        .unsupportedCacheTopology(
            layer: layer, path: path, cacheType: String(describing: type(of: cache)),
            reason: reason)
    }

    // Exact types: subclasses such as ChunkedKVCache behave differently.
    if type(of: cache) == KVCacheSimple.self {
        return { BatchKVCache(leftPadding: $0) }
    }
    throw unsupported("No batched cache implementation exists for this cache type.")
}
