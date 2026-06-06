// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// MARK: - BatchedCache

/// A ``KVCache`` that supports the continuous-batching primitives used by the
/// batched inference engine: in-place row filtering, concatenation of admitted
/// rows, ragged-prefill prepare/finalize, single-row extraction, and chunk-local
/// advancement.
///
/// Both full-attention caches (``BatchKVCache``, ``BatchRotatingKVCache``) and
/// SSM-style caches (``ArraysCache`` / ``MambaCache``) conform, so the engine can
/// treat a heterogeneous per-layer cache list uniformly.
///
/// Conformers are **not** thread-safe; the engine mutates them from a single
/// serial executor.
public protocol BatchedCache: KVCache {
    /// In-place keep only the rows at the given batch indices.
    func filterBatched(batchIndices: MLXArray)

    /// In-place append `other`'s rows. The runtime types must match.
    func extendBatched(_ other: any BatchedCache)

    /// Prepare cache metadata before a ragged prompt prefill.
    func prepareBatched(leftPadding: [Int]?, lengths: [Int]?, rightPadding: [Int]?)

    /// Finalize cache metadata after a ragged prompt prefill.
    func finalizeBatched()

    /// Extract one row as its corresponding single-request cache.
    func extractBatched(_ idx: Int) -> any KVCache

    /// Advance chunk-local metadata after a chunked prefill step.
    func advanceBatched(_ n: Int)
}

// MARK: - BatchedCacheList

/// Batched wrapper for composite per-layer caches. Some hybrid models keep
/// multiple cache objects per logical layer, so batching has to preserve that
/// nested topology instead of treating the composite as full attention.
public final class BatchedCacheList: CacheList, BatchedCache {

    private let batchedCaches: [any BatchedCache]

    internal init(caches: [any BatchedCache]) {
        self.batchedCaches = caches
        super.init(caches: caches.map { $0 as any KVCache })
    }

    public func filterBatched(batchIndices: MLXArray) {
        for cache in batchedCaches {
            cache.filterBatched(batchIndices: batchIndices)
        }
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? BatchedCacheList else {
            preconditionFailure("BatchedCacheList.extendBatched requires another BatchedCacheList")
        }
        precondition(
            batchedCaches.count == other.batchedCaches.count,
            "Cannot extend BatchedCacheList with different child count"
        )

        for (a, b) in zip(batchedCaches, other.batchedCaches) {
            a.extendBatched(b)
        }
    }

    public func prepareBatched(leftPadding: [Int]?, lengths: [Int]?, rightPadding: [Int]?) {
        for cache in batchedCaches {
            cache.prepareBatched(
                leftPadding: leftPadding,
                lengths: lengths,
                rightPadding: rightPadding
            )
        }
    }

    public func finalizeBatched() {
        for cache in batchedCaches {
            cache.finalizeBatched()
        }
    }

    public func extractBatched(_ idx: Int) -> any KVCache {
        CacheList(caches: batchedCaches.map { $0.extractBatched(idx) })
    }

    public func advanceBatched(_ n: Int) {
        for cache in batchedCaches {
            cache.advanceBatched(n)
        }
    }
}

// MARK: - SSM cache conformance

extension ArraysCache: BatchedCache {
    public func filterBatched(batchIndices: MLXArray) {
        filter(batchIndices: batchIndices)
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? ArraysCache else {
            preconditionFailure("ArraysCache.extendBatched requires another ArraysCache")
        }
        extend(other: other)
    }

    public func prepareBatched(leftPadding _: [Int]?, lengths: [Int]?, rightPadding _: [Int]?) {
        prepare(lengths: lengths)
    }

    public func finalizeBatched() {
        finalize()
    }

    public func extractBatched(_ idx: Int) -> any KVCache {
        extract(idx)
    }

    public func advanceBatched(_ n: Int) {
        advance(n)
    }
}

// MARK: - Factory

/// A closure that allocates one batched cache for `leftPadding.count` rows.
public typealias BatchedCacheFactory = (_ leftPadding: [Int]) -> any BatchedCache

/// Error thrown when a model's cache topology cannot be batched.
public enum BatchedCacheError: Error, CustomStringConvertible, Equatable {
    case unsupportedCacheTopology(layer: Int, path: String, cacheType: String, reason: String)

    public var description: String {
        switch self {
        case .unsupportedCacheTopology(let layer, let path, let cacheType, let reason):
            return "Unsupported cache topology at layer \(layer), \(path): "
                + "\(cacheType). \(reason)"
        }
    }
}

/// Build one ``BatchedCacheFactory`` per layer from a probe of single-stream
/// caches (typically `model.newCache(parameters:)`).
///
/// Validating the topology once at engine-construction time means per-admission
/// allocation is a cheap closure call that cannot fail.
///
/// - Throws: ``BatchedCacheError`` if any layer's cache type has no batched
///   implementation (e.g. quantized or chunked caches).
public func makeBatchedCacheFactories(for probe: [any KVCache]) throws -> [BatchedCacheFactory] {
    try probe.enumerated().map { layer, cache in
        try makeBatchedCacheFactory(for: cache, layer: layer, path: "layer")
    }
}

/// Convenience: build a fresh set of batched caches for `leftPadding.count` rows
/// directly from a probe of single-stream caches.
public func makeBatchedCache(
    for probe: [any KVCache], leftPadding: [Int]
) throws -> [any BatchedCache] {
    try makeBatchedCacheFactories(for: probe).map { $0(leftPadding) }
}

private func makeBatchedCacheFactory(
    for cache: any KVCache,
    layer: Int,
    path: String
) throws -> BatchedCacheFactory {
    let cacheType = String(describing: Swift.type(of: cache))

    func unsupported(_ reason: String) -> BatchedCacheError {
        .unsupportedCacheTopology(
            layer: layer,
            path: path,
            cacheType: cacheType,
            reason: reason
        )
    }

    if cache is QuantizedKVCache {
        throw unsupported("Quantized KV caches are not supported by continuous batching.")
    }

    if cache is ChunkedKVCache {
        throw unsupported("Chunked KV caches are not supported by continuous batching.")
    }

    if let cacheList = cache as? CacheList {
        let childFactories = try cacheList.children.enumerated().map { childIndex, child in
            try makeBatchedCacheFactory(
                for: child,
                layer: layer,
                path: "\(path).children[\(childIndex)]"
            )
        }
        return { leftPadding in
            BatchedCacheList(caches: childFactories.map { $0(leftPadding) })
        }
    }

    // Exact-type matches avoid misclassifying subclasses such as
    // MambaCache : ArraysCache and ChunkedKVCache : KVCacheSimple.
    if Swift.type(of: cache) == MambaCache.self {
        return { leftPadding in MambaCache(leftPadding: leftPadding) }
    }

    if Swift.type(of: cache) == ArraysCache.self, let arrays = cache as? ArraysCache {
        let slotCount = arrays.slotCount
        return { leftPadding in ArraysCache(size: slotCount, leftPadding: leftPadding) }
    }

    if let rotating = cache as? RotatingKVCache {
        guard let maxSize = rotating.maxSize else {
            throw unsupported("RotatingKVCache must have a non-nil maxSize.")
        }

        // `BatchRotatingKVCache` handles keep-prefix rotation, so (unlike a
        // keep-only sliding window) any `keep` value can be batched. metaState
        // layout is [keep, maxCacheSize, step, offset, idx].
        let keep = Int(rotating.metaState.first ?? "0") ?? 0

        return { leftPadding in
            BatchRotatingKVCache(maxSize: maxSize, leftPadding: leftPadding, keep: keep)
        }
    }

    if Swift.type(of: cache) == KVCacheSimple.self {
        return { leftPadding in BatchKVCache(leftPadding: leftPadding) }
    }

    throw unsupported("No batched cache implementation exists for this cache type.")
}
