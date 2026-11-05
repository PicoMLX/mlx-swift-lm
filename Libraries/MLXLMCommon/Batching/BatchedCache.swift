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

/// A cache that has no batched form.
public enum BatchedCacheError: Error, CustomStringConvertible, Equatable {
    case unsupportedCache(cacheType: String, reason: String)

    public var description: String {
        switch self {
        case .unsupportedCache(let cacheType, let reason):
            "No batched form for \(cacheType): \(reason)"
        }
    }
}

/// The rows of `caches`, single-sequence caches of one layer, as one batched cache.
///
/// Port of mlx-lm's `_merge_caches`: dispatches on the exact cache type, so a
/// `ChunkedKVCache` or a compressed cache is refused rather than treated as the
/// class it subclasses. Empty caches merge into an empty batch with one row each.
///
/// - Throws: ``BatchedCacheError`` when the type has no batched form, when the
///   rows do not share one type, or when a rotating cache keeps a prefix or a
///   rewind reserve.
public func mergeCaches(_ caches: [any KVCache]) throws -> any BatchedCache {
    guard let first = caches.first else {
        throw BatchedCacheError.unsupportedCache(cacheType: "none", reason: "merge needs a cache")
    }
    let cacheType = String(describing: type(of: first))
    func rows<T: KVCache>(as _: T.Type) throws -> [T] {
        try caches.map { cache in
            guard type(of: cache) == T.self, let row = cache as? T else {
                throw BatchedCacheError.unsupportedCache(
                    cacheType: cacheType, reason: "Rows must share one cache type.")
            }
            return row
        }
    }
    switch first {
    case is CacheList:
        return try CacheList.merge(try rows(as: CacheList.self))
    case is MambaCache:
        return ArraysCache.merge(try rows(as: MambaCache.self))
    case is ArraysCache:
        return ArraysCache.merge(try rows(as: ArraysCache.self))
    case is RotatingKVCache:
        return try RotatingKVCache.merge(try rows(as: RotatingKVCache.self))
    case is KVCacheSimple where type(of: first) == KVCacheSimple.self:
        return KVCacheSimple.merge(try rows(as: KVCacheSimple.self))
    default:
        throw BatchedCacheError.unsupportedCache(
            cacheType: cacheType,
            reason: "No batched cache implementation exists for this cache type.")
    }
}

/// Copy a prepared single-sequence cache into its continuous-batching representation.
///
/// Uses the same exact-type and topology checks as ``mergeCaches(_:)``. The source
/// remains independently usable; unsupported cache types return `nil`.
public func makeBatchedCache(from cache: any KVCache) -> (any BatchedCache)? {
    try? mergeCaches([cache])
}

/// Convert every layer of a prepared row, or reject its complete topology.
/// No source cache is mutated if a later layer cannot be represented.
public func makeBatchedCaches(from caches: [any KVCache]) -> [any BatchedCache]? {
    guard !caches.isEmpty else { return nil }
    return try? caches.map { try mergeCaches([$0]) }
}

// MARK: - Recurrent caches

extension ArraysCache: BatchedCache {
    /// The rows of `caches` as one cache. Recurrent state has no padding to
    /// align, so this is the first row extended by the rest; a batch of
    /// `MambaCache` stays a `MambaCache`. A row with no state yet counts as
    /// one row only once it holds arrays, so when any row is populated the
    /// empty ones get zero state of the same shape first. An all-empty merge
    /// stays empty and takes its row count from the first update.
    public static func merge(_ caches: [ArraysCache]) -> ArraysCache {
        precondition(!caches.isEmpty, "merge needs a cache")
        func isEmpty(_ cache: ArraysCache) -> Bool {
            (0 ..< cache.slotCount).allSatisfy { cache[$0] == nil }
        }
        let template = caches.first { !isEmpty($0) }
        let rows: [ArraysCache] = caches.map { cache in
            guard let row = cache.copy() as? ArraysCache else {
                preconditionFailure("An ArraysCache copy must stay an ArraysCache")
            }
            guard let template, isEmpty(row) else { return row }
            for slot in 0 ..< row.slotCount {
                if let example = template[slot] {
                    row[slot] = MLXArray.zeros(
                        [1] + Array(example.shape.dropFirst()), dtype: example.dtype)
                }
            }
            return row
        }
        let merged = rows[0]
        for row in rows.dropFirst() {
            merged.extend(other: row)
        }
        return merged
    }

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

// MARK: - Composite caches

extension CacheList: BatchedCache {
    /// The rows of `caches`, composite caches of one layout, as one composite
    /// whose children are batched. Each child position merges on its own type.
    public static func merge(_ caches: [CacheList]) throws -> CacheList {
        precondition(!caches.isEmpty, "merge needs a cache")
        let width = caches[0].children.count
        guard caches.allSatisfy({ $0.children.count == width }) else {
            throw BatchedCacheError.unsupportedCache(
                cacheType: "CacheList", reason: "Rows must share one layout.")
        }
        let merged = try (0 ..< width).map { index in
            try mergeCaches(caches.map { $0.children[index] }) as any KVCache
        }
        return CacheList(caches: merged)
    }

    private var batchedChildren: [any BatchedCache] {
        children.map { child in
            guard let child = child as? any BatchedCache else {
                preconditionFailure("A batched CacheList child must be a BatchedCache")
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
        guard let other = other as? CacheList, other.children.count == children.count
        else {
            preconditionFailure(
                "A CacheList can only be extended with a CacheList of the same layout")
        }
        for (child, added) in zip(batchedChildren, other.batchedChildren) {
            child.extend(other: added)
        }
    }

    public func extract(_ index: Int) -> any KVCache {
        CacheList(caches: batchedChildren.map { $0.extract(index) })
    }
}
