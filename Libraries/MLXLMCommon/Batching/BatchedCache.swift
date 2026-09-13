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

/// Batched wrapper that preserves nested caches within a logical layer.
public final class BatchedCacheList: CacheList, BatchedCache {

    private var batchedCaches: [any BatchedCache] {
        guard let caches = children as? [any BatchedCache] else {
            preconditionFailure(
                "BatchedCacheList child rewrites must preserve BatchedCache conformance")
        }
        return caches
    }

    internal init(caches: [any BatchedCache]) {
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

    /// Delegate to the first attention child so masks respect per-row padding.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        for cache in batchedCaches where cache is BatchPositionedKVCache {
            return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
        }
        return super.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    /// Deep-copy the children while preserving their batched wrapper.
    public override func copy() -> any KVCache {
        BatchedCacheList(
            caches: batchedCaches.map { child in
                guard let copied = child.copy() as? any BatchedCache else {
                    preconditionFailure(
                        "BatchedCache.copy() must return a BatchedCache for "
                            + "BatchedCacheList snapshots"
                    )
                }
                return copied
            }
        )
    }
}

// MARK: - SSM cache conformance

extension ArraysCache {
    /// Extract one row, preserving ``MambaCache`` for downstream type checks.
    public func extract(_ idx: Int) -> ArraysCache {
        let extracted = self is MambaCache ? MambaCache() : ArraysCache(size: slotCount)
        for slot in 0 ..< slotCount {
            extracted[slot] = self[slot]?[idx ..< (idx + 1)]
        }
        extracted.offset = offset
        extracted.leftPadding = leftPadding?[idx ..< (idx + 1)]
        extracted.lengths = lengths?[idx ..< (idx + 1)]
        return extracted
    }
}

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

    public func advanceBatched(_: Int) {
        // The model advances recurrent caches during its forward pass.
        // Advancing again here would shift the next chunk's padding mask.
    }
}

// MARK: - Factory

/// A closure that allocates one batched cache for `leftPadding.count` rows.
public typealias BatchedCacheFactory = (_ leftPadding: [Int]) -> any BatchedCache

/// Normalize all-zero padding to nil so SSM prefill uses its lengths mask.
private func ssmLeftPadding(_ leftPadding: [Int]) -> [Int]? {
    // Preserve zero-row cardinality; nil defaults to one row.
    if leftPadding.isEmpty { return leftPadding }
    return leftPadding.allSatisfy { $0 == 0 } ? nil : leftPadding
}

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
/// This checks cache topology only; callers must separately validate the
/// model's batched position and recurrent-state semantics.
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
        return { leftPadding in MambaCache(leftPadding: ssmLeftPadding(leftPadding)) }
    }

    if Swift.type(of: cache) == ArraysCache.self, let arrays = cache as? ArraysCache {
        let slotCount = arrays.slotCount
        return { leftPadding in
            ArraysCache(size: slotCount, leftPadding: ssmLeftPadding(leftPadding))
        }
    }

    if let rotating = cache as? RotatingKVCache {
        guard let maxSize = rotating.maxSize else {
            throw unsupported("RotatingKVCache must have a non-nil maxSize.")
        }

        // RotatingKVCache.keep is private; metaState layout is
        // [keep, maxCacheSize, step, offset, idx].
        let keep = Int(rotating.metaState.first ?? "0") ?? 0

        // With keep > 0, rotation can move padding beyond the prefix mask.
        // Keep these topologies on the single-stream path.
        guard keep == 0 else {
            throw unsupported(
                "RotatingKVCache with keep > 0 is not supported by continuous batching."
            )
        }

        // Preserve capacity provenance for validation and runtime reporting.
        let capacityOrigin = rotating.capacityOrigin
        let step = Int(rotating.metaState[2]) ?? 256

        return { leftPadding in
            let cache = BatchRotatingKVCache(
                maxSize: maxSize, leftPadding: leftPadding, keep: keep)
            cache.capacityOrigin = capacityOrigin
            cache.step = step
            return cache
        }
    }

    if Swift.type(of: cache) == KVCacheSimple.self {
        return { leftPadding in BatchKVCache(leftPadding: leftPadding) }
    }

    throw unsupported("No batched cache implementation exists for this cache type.")
}
