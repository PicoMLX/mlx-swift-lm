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

    if let list = cache as? CacheList {
        let children = try list.children.enumerated().map { index, child in
            try makeBatchedCacheFactory(
                for: child, layer: layer, path: "\(path).children[\(index)]")
        }
        return { leftPadding in BatchedCacheList(caches: children.map { $0(leftPadding) }) }
    }

    // Exact types: subclasses such as ChunkedKVCache behave differently.
    if type(of: cache) == KVCacheSimple.self {
        return { BatchKVCache(leftPadding: $0) }
    }
    if type(of: cache) == RotatingKVCache.self, let rotating = cache as? RotatingKVCache {
        // A pinned prefix would pin padding, and a batch has no rewind reserve.
        guard rotating.preservedPrefixTokens == 0 else {
            throw unsupported("A RotatingKVCache with keep > 0 has no batched form.")
        }
        guard rotating.rewindCapacity == 0 else {
            throw unsupported("A RotatingKVCache with a rewind reserve has no batched form.")
        }
        let window = rotating.maxSize ?? 0
        let step = rotating.allocationStep
        let capacityOrigin = rotating.capacityOrigin
        return { leftPadding in
            let cache = BatchRotatingKVCache(maxSize: window, leftPadding: leftPadding)
            cache.step = step
            cache.capacityOrigin = capacityOrigin
            return cache
        }
    }
    if type(of: cache) == MambaCache.self {
        return { MambaCache(leftPadding: recurrentLeftPadding($0)) }
    }
    if type(of: cache) == ArraysCache.self, let arrays = cache as? ArraysCache {
        let slotCount = arrays.slotCount
        return { ArraysCache(size: slotCount, leftPadding: recurrentLeftPadding($0)) }
    }
    throw unsupported("No batched cache implementation exists for this cache type.")
}

/// `nil` for an all-zero left padding, so a recurrent cache masks with the lengths
/// `prepare(lengths:)` sets. An empty batch keeps its zero rows.
private func recurrentLeftPadding(_ leftPadding: [Int]) -> [Int]? {
    leftPadding.isEmpty || leftPadding.contains { $0 != 0 } ? leftPadding : nil
}

// MARK: - Composite and recurrent caches

/// A ``CacheList`` whose children are batched caches, for layers that hold several caches.
public final class BatchedCacheList: CacheList, BatchedCache {
    /// A batch cannot be restored from a prompt cache file.
    public override class var serializedClassName: String? { nil }

    init(caches: [any BatchedCache]) {
        super.init(caches: caches)
    }

    private var batchedChildren: [any BatchedCache] {
        children.map { child in
            guard let child = child as? any BatchedCache else {
                preconditionFailure("A BatchedCacheList child must stay a BatchedCache")
            }
            return child
        }
    }

    public func filter(batchIndices: MLXArray) {
        for child in batchedChildren {
            child.filter(batchIndices: batchIndices)
        }
    }

    public func extend(other: any BatchedCache) {
        guard let other = other as? BatchedCacheList, other.children.count == children.count
        else {
            preconditionFailure(
                "BatchedCacheList can only be extended with a BatchedCacheList of the same layout")
        }
        for (child, added) in zip(batchedChildren, other.batchedChildren) {
            child.extend(other: added)
        }
    }

    public func extract(_ index: Int) -> any KVCache {
        CacheList(caches: batchedChildren.map { $0.extract(index) })
    }

    /// The mask of the first child with per-row positions, so it excludes left padding.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if let child = batchedChildren.first(where: { $0 is any BatchPositionedKVCache }) {
            return child.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
        }
        return super.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public override func copy() -> any KVCache {
        BatchedCacheList(
            caches: batchedChildren.map { child in
                guard let copy = child.copy() as? any BatchedCache else {
                    preconditionFailure("A BatchedCache copy must stay a BatchedCache")
                }
                return copy
            })
    }
}

extension ArraysCache: BatchedCache {
    public func extend(other: any BatchedCache) {
        guard let other = other as? ArraysCache else {
            preconditionFailure("ArraysCache can only be extended with an ArraysCache")
        }
        extend(other: other)
    }

    /// The row at `index`, as a ``MambaCache`` when this is one.
    public func extract(_ index: Int) -> any KVCache {
        let row = self is MambaCache ? MambaCache() : ArraysCache(size: slotCount)
        for slot in 0 ..< slotCount {
            row[slot] = self[slot]?[index ..< index + 1]
        }
        row.offset = offset
        row.leftPadding = leftPadding?[index ..< index + 1]
        row.lengths = lengths?[index ..< index + 1]
        return row
    }
}
