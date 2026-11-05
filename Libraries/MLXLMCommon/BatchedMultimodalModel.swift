// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Media inputs whose prepared caches a model can safely adopt into ragged decoding.
public struct BatchedMediaCapabilities: OptionSet, Sendable {
    public let rawValue: Int

    public init(rawValue: Int) { self.rawValue = rawValue }

    public static let image = Self(rawValue: 1 << 0)
    public static let video = Self(rawValue: 1 << 1)
    public static let audio = Self(rawValue: 1 << 2)
}

/// A model validated for ragged decoding after independently preparing media requests.
///
/// Conforming models keep request-specific continuation data in their caches. Their
/// media `prepare` returns logits without a separate `LMOutput.State`, and the caches
/// support `mergeCaches(_:)`, row filtering, extraction and extension.
/// Media encoders may run separately before those caches join a running decode batch.
public protocol BatchedMultimodalModel: LanguageModel {
    var batchedMediaCapabilities: BatchedMediaCapabilities { get }

    /// Negative placeholders introduced by this model's processor, replaced before
    /// embedding lookup. Admission can validate the remaining token IDs normally.
    var batchTokenPlaceholderIDs: Set<Int> { get }
}

extension BatchedMultimodalModel {
    public var batchTokenPlaceholderIDs: Set<Int> { [] }
}

/// Whether a single media prefill can be adopted at `tokenCount` without losing state.
/// Attention leaves must consume the full prompt; recurrent and positional metadata
/// leaves must hold initialized single-row state rather than an attention offset.
public func areBatchedMediaCachesPrepared(_ caches: [any KVCache], tokenCount: Int) -> Bool {
    guard tokenCount > 0, !caches.isEmpty else { return false }
    var hasAttention = false
    func prepared(_ cache: any KVCache) -> Bool {
        if let list = cache as? CacheList {
            return !list.children.isEmpty && list.children.allSatisfy(prepared)
        }
        if let arrays = cache as? ArraysCache {
            return arrays.state.count == arrays.slotCount
                && arrays.state.allSatisfy { $0.ndim > 0 && $0.dim(0) == 1 }
        }
        guard type(of: cache) == KVCacheSimple.self || type(of: cache) == RotatingKVCache.self
        else {
            return false
        }
        hasAttention = true
        return cache.offset == tokenCount
    }
    return caches.allSatisfy(prepared) && hasAttention
}
