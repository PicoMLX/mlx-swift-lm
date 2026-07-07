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

/// An opaque, immutable snapshot of per-layer KV state.
///
/// The storage is deliberately non-public so its representation can move from
/// today's deep-copied `[KVCache]` to a `Sendable` materialized payload
/// without a source break: conformers construct snapshots from live caches,
/// consumers materialize them back, and neither sees the representation.
///
/// > Note: **`MaterializedMLXArray` adoption plan.** mlx-swift is about to
/// > ship `MaterializedMLXArray` (a `Sendable` carrier for evaluated
/// > arrays). When it lands, this type's stored representation switches to
/// > materialized per-layer state and the type gains `Sendable` conformance
/// > — both changes are internal to this file, and none of the public
/// > surface (`init(caches:)`, `materialize()`, `PromptCacheFetchResult`)
/// > changes. Snapshots then cross actor boundaries directly, removing the
/// > deep-copy round trips at the scheduler/driver cache seams. `Sendable`
/// > conformance is intentionally deferred until then.
public struct PromptCacheSnapshot {
    /// Today: deep-copied, evaluated caches owned by whoever holds the
    /// snapshot. The future materialized representation replaces this
    /// stored property without changing the public surface.
    private let storage: [KVCache]

    /// Wrap already-detached (deep-copied, evaluated) per-layer caches.
    public init(caches: [KVCache]) {
        self.storage = caches
    }

    /// The live per-layer caches this snapshot carries.
    public func materialize() -> [KVCache] {
        storage
    }
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
    /// An opaque snapshot of the matched KV state, or `nil` if nothing
    /// matched. See ``PromptCacheSnapshot`` for the representation contract.
    public let snapshot: PromptCacheSnapshot?
    /// The live caches from ``snapshot`` (deep copies owned by the caller),
    /// or `nil` if nothing matched.
    public var cache: [KVCache]? { snapshot?.materialize() }
    /// The tokens that still need processing after the cached prefix.
    public let remainder: [Int]
    /// How the lookup matched.
    public let hitKind: PromptCacheHitKind
    /// How many leading tokens the matched cache covers.
    public let matchedTokenCount: Int

    /// The number of prompt tokens whose KV state was reused from the cache.
    public var reusedTokenCount: Int { matchedTokenCount }

    public init(
        snapshot: PromptCacheSnapshot?,
        remainder: [Int],
        hitKind: PromptCacheHitKind,
        matchedTokenCount: Int
    ) {
        self.snapshot = snapshot
        self.remainder = remainder
        self.hitKind = hitKind
        self.matchedTokenCount = matchedTokenCount
    }

    /// Convenience: wrap live caches in a snapshot. The caches must already
    /// be detached copies (every in-repo producer deep-copies at fetch).
    public init(
        cache: [KVCache]?,
        remainder: [Int],
        hitKind: PromptCacheHitKind,
        matchedTokenCount: Int
    ) {
        self.init(
            snapshot: cache.map(PromptCacheSnapshot.init(caches:)),
            remainder: remainder,
            hitKind: hitKind,
            matchedTokenCount: matchedTokenCount
        )
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
/// protocol — storage policy (where, when, and how to persist) belongs to the
/// embedding app, built on the library's KV-cache serialization primitives.
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
