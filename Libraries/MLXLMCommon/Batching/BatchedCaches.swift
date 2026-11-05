// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The rows of `caches`, single-sequence caches of one layer, as one batched cache.
///
/// Port of mlx-lm's `_merge_caches` for one layer: dispatches on the exact cache type, so a
/// `ChunkedKVCache` or a compressed cache is refused rather than treated as the class it
/// subclasses. Empty caches merge into an empty batch with one row each.
///
/// - Throws: ``KVCacheError`` when the type has no batched form, when the rows do not share
///   one type, or when a rotating cache keeps a prefix or a rewind reserve.
func mergeCaches(_ caches: [any KVCache]) throws -> any KVCache {
    guard let first = caches.first else {
        throw KVCacheError(message: "merge needs a cache")
    }
    let cacheType = String(describing: type(of: first))
    func rows<T: KVCache>(as _: T.Type) throws -> [T] {
        try caches.map { cache in
            guard type(of: cache) == T.self, let row = cache as? T else {
                throw KVCacheError(message: "Rows of a batch must share one cache type")
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
        throw KVCacheError(message: "\(cacheType) has no batched form")
    }
}

// MARK: - Recurrent caches

extension ArraysCache {
    /// The rows of `caches` as one cache. Port of mlx-lm's `ArraysCache.merge`.
    ///
    /// Recurrent state has no padding to align, so this is the first row extended by the
    /// rest; a batch of `MambaCache` stays a `MambaCache`. When any row holds state, the
    /// empty rows get zero state of the same shape first, so every row counts. An all-empty
    /// merge stays empty and takes its row count from the first update.
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
        // Recurrent state keeps one position counter for the batch: the longest row's.
        merged.offset = caches.map(\.offset).max() ?? 0
        return merged
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

extension CacheList {
    /// The rows of `caches`, composite caches of one layout, as one composite whose children
    /// are batched. Each child position merges on its own type. Port of mlx-lm's
    /// `CacheList.merge`.
    ///
    /// - Throws: ``KVCacheError`` when the layouts differ or a child has no batched form.
    public static func merge(_ caches: [CacheList]) throws -> CacheList {
        precondition(!caches.isEmpty, "merge needs a cache")
        let width = caches[0].children.count
        guard caches.allSatisfy({ $0.children.count == width }) else {
            throw KVCacheError(message: "Rows of a CacheList batch must share one layout")
        }
        let merged = try (0 ..< width).map { index in
            try mergeCaches(caches.map { $0.children[index] })
        }
        return CacheList(caches: merged)
    }

    /// Keeps only the rows at `batchIndices`, in that order, in every child.
    public func filter(batchIndices: MLXArray) {
        for child in children {
            switch child {
            case let batch as BatchKVCache: batch.filter(batchIndices: batchIndices)
            case let batch as BatchRotatingKVCache: batch.filter(batchIndices: batchIndices)
            case let batch as ArraysCache: batch.filter(batchIndices: batchIndices)
            case let batch as CacheList: batch.filter(batchIndices: batchIndices)
            default: preconditionFailure("\(type(of: child)) is not a batched cache")
            }
        }
    }

    /// Appends the rows of `other`, a `CacheList` of the same layout, child by child.
    public func extend(other: CacheList) {
        precondition(
            other.children.count == children.count,
            "A CacheList can only be extended with a CacheList of the same layout")
        for (child, added) in zip(children, other.children) {
            switch (child, added) {
            case (let batch as BatchKVCache, let added as BatchKVCache):
                batch.extend(other: added)
            case (let batch as BatchRotatingKVCache, let added as BatchRotatingKVCache):
                batch.extend(other: added)
            case (let batch as ArraysCache, let added as ArraysCache):
                batch.extend(other: added)
            case (let batch as CacheList, let added as CacheList):
                batch.extend(other: added)
            default:
                preconditionFailure(
                    "\(type(of: child)) cannot be extended with \(type(of: added))")
            }
        }
    }

    /// The row at `index` as a composite of single-sequence caches.
    public func extract(_ index: Int) -> any KVCache {
        CacheList(
            caches: children.map { child -> any KVCache in
                switch child {
                case let batch as BatchKVCache: batch.extract(index)
                case let batch as BatchRotatingKVCache: batch.extract(index)
                case let batch as ArraysCache: batch.extract(index)
                case let batch as CacheList: batch.extract(index)
                default: preconditionFailure("\(type(of: child)) is not a batched cache")
                }
            })
    }
}
