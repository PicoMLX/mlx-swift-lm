// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The rows of `caches`, single-sequence caches of one layer, as one batched cache.
///
/// Port of mlx-lm's `_merge_caches` for one layer: the rows merge on their exact class, so a
/// `ChunkedKVCache` or a compressed cache is refused rather than treated as the class it
/// subclasses. Empty caches merge into an empty batch with one row each.
///
/// - Throws: ``KVCacheError`` when the type has no batched form, when the rows do not share
///   one type, or when a rotating cache keeps a prefix or a rewind reserve.
public func mergeCaches(_ caches: [any KVCache]) throws -> any KVCache {
    guard let first = caches.first else {
        throw KVCacheError(message: "merge needs a cache")
    }
    guard caches.allSatisfy({ type(of: $0) == type(of: first) }) else {
        throw KVCacheError(message: "Rows of a batch must share one cache type")
    }
    return try type(of: first).merge(caches)
}

/// `caches` as rows of exactly `T`, for a `merge` that takes only its own class.
func batchRows<T: KVCache>(_ caches: [any KVCache], of _: T.Type) throws -> [T] {
    try caches.map { cache in
        guard type(of: cache) == T.self, let row = cache as? T else {
            throw KVCacheError(message: "\(type(of: cache)) has no batched form")
        }
        return row
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
}
