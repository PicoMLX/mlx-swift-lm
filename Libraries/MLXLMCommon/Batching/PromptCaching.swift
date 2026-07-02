// Copyright © 2024 Apple Inc.

import Foundation

// MARK: - Neutral result types

/// How a prompt-cache lookup matched the requested token sequence.
///
/// This is the protocol-neutral hit classification shared by all
/// ``PromptCaching`` conformers. ``LRUPromptCache`` exposes it under the
/// convenience name `LRUPromptCache.HitKind` for source compatibility.
public enum PromptCacheHitKind: String, Sendable, Equatable {
    /// No cached prefix matched the query.
    case none
    /// A cached entry covered exactly the requested tokens.
    case exact
    /// A cached entry covered a strict prefix of the requested tokens.
    case shorter
    /// A cached entry covered more tokens than requested and was trimmed.
    case longer
}

/// The result of a prompt-cache lookup.
///
/// This is the protocol-neutral result type returned by
/// ``PromptCaching/fetchNearestCacheResult(model:tokens:salt:)``. Keeping it
/// independent of any concrete cache prevents ``PromptCaching`` from coupling
/// to ``LRUPromptCache``; ``LRUPromptCache`` exposes it under the convenience
/// name `LRUPromptCache.FetchResult` for source compatibility.
///
/// Not `Sendable`: it carries non-`Sendable` `KVCache` values. The owning
/// ``PromptCaching`` conformer is `Sendable`; results are consumed locally by
/// the actor that performed the lookup (a freshly materialized deep copy).
public struct PromptCacheFetchResult {
    /// A deep copy of the matched cache, or `nil` if nothing matched.
    public let cache: [KVCache]?
    /// The tokens that still need processing after the cached prefix.
    public let remainder: [Int]
    /// How the lookup matched.
    public let hitKind: PromptCacheHitKind
    /// How many leading tokens the matched cache covers.
    public let matchedTokenCount: Int

    /// The number of prompt tokens whose KV state was reused from the cache.
    public var reusedTokenCount: Int { matchedTokenCount }

    public init(
        cache: [KVCache]?,
        remainder: [Int],
        hitKind: PromptCacheHitKind,
        matchedTokenCount: Int
    ) {
        self.cache = cache
        self.remainder = remainder
        self.hitKind = hitKind
        self.matchedTokenCount = matchedTokenCount
    }
}

// MARK: - PromptCaching

/// A cross-request KV-cache reuse store.
///
/// Conformers map token-sequence prefixes to KV caches so that a later request
/// sharing a prefix can skip re-prefilling those tokens. The protocol is kept
/// minimal and expressed in terms of the neutral ``PromptCacheFetchResult`` /
/// ``PromptCacheHitKind`` so that alternate implementations (e.g. a future
/// block/paged cache) can conform without inheriting ``LRUPromptCache``'s
/// concrete result types.
///
/// `Sendable` because a prompt cache is shared across the actors that drive
/// batched inference. Disk persistence is intentionally *not* part of this
/// protocol — see ``PersistablePromptCache``.
public protocol PromptCaching: Sendable {
    /// Fetch the nearest matching KV cache for the given token sequence.
    ///
    /// - Parameters:
    ///   - model: Model identifier used for isolation.
    ///   - tokens: The token sequence to look up.
    ///   - salt: Per-agent isolation salt (default `0`).
    /// - Returns: The match (a deep copy) plus the tokens still requiring work.
    func fetchNearestCacheResult(
        model: String,
        tokens: [Int],
        salt: UInt64
    ) -> PromptCacheFetchResult

    /// Insert a KV cache covering the given token sequence.
    ///
    /// - Parameters:
    ///   - model: Model identifier used for isolation.
    ///   - tokens: The token sequence the cache covers.
    ///   - promptCache: The KV cache layers to store.
    ///   - checkpoint: Whether this is a checkpoint entry (affects eviction).
    ///   - salt: Per-agent isolation salt (default `0`).
    func insertCache(
        model: String,
        tokens: [Int],
        promptCache: [KVCache],
        checkpoint: Bool,
        salt: UInt64
    )

    /// Evict entries until the cache is within the given limits.
    ///
    /// - Parameters:
    ///   - nSequences: Maximum number of entries to keep (`nil` = no limit).
    ///   - nBytes: Maximum total bytes to keep (`nil` = no limit).
    func trim(nSequences: Int?, nBytes: Int?)
}

// MARK: - Default arguments

/// Default-argument conveniences for `any PromptCaching` callers.
///
/// Protocol *requirements* cannot carry default parameter values, so a caller
/// holding the existential `any PromptCaching` would otherwise have to pass
/// `salt` / `checkpoint` explicitly. These overloads supply the documented
/// defaults (`salt: 0`, `checkpoint: false`) by forwarding to the requirements.
extension PromptCaching {
    /// Fetch the nearest matching KV cache, defaulting `salt` to `0`.
    public func fetchNearestCacheResult(
        model: String,
        tokens: [Int]
    ) -> PromptCacheFetchResult {
        fetchNearestCacheResult(model: model, tokens: tokens, salt: 0)
    }

    /// Insert a KV cache, defaulting `checkpoint` to `false` and `salt` to `0`.
    public func insertCache(
        model: String,
        tokens: [Int],
        promptCache: [KVCache],
        checkpoint: Bool = false
    ) {
        insertCache(
            model: model,
            tokens: tokens,
            promptCache: promptCache,
            checkpoint: checkpoint,
            salt: 0
        )
    }
}

// MARK: - PersistablePromptCache

/// A ``PromptCaching`` store that can be saved to and loaded from disk.
///
/// Disk persistence is a refining protocol so that in-memory-only caches are
/// not forced to implement it. Public surfaces typed as `any PromptCaching`
/// therefore do not expose persistence — callers that need `save`/`load` hold
/// a `PersistablePromptCache` (or the concrete cache) reference.
public protocol PersistablePromptCache: PromptCaching {
    /// Persist the cache's disk-persistable entries under `directory`.
    func save(to directory: URL, maxDiskBytes: Int?) async throws

    /// Load previously persisted entries from `directory`.
    func load(from directory: URL, allowedModels: Set<String>?) async throws
}
