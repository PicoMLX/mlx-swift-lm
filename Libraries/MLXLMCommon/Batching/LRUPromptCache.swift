// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import os

// MARK: - LRUPromptCache

/// Trie-based LRU cache storing KV caches keyed by token sequences.
///
/// Supports exact, shorter-prefix, and longer-prefix lookups. Fetch always
/// returns a deep copy (independent of the stored cache). Model isolation
/// (keyed by model identifier and a per-agent salt) ensures caches from
/// different models/agents don't cross-contaminate.
///
/// Thread safety is provided by an `OSAllocatedUnfairLock` wrapping the mutable
/// state, which makes the type checked-`Sendable` (no `@unchecked` on this
/// type). All mutating operations run inside `state.withLockUnchecked { ... }`
/// (`Unchecked` because the protected state holds non-`Sendable` `KVCache`
/// values; the lock supplies the synchronization).
///
/// Key operations:
/// - `insertCache(model:tokens:promptCache:)` — store a KV cache for a token sequence
/// - `fetchNearestCache(model:tokens:)` — find the best matching cached prefix
/// - `trimTo(nSequences:nBytes:)` — memory-aware eviction
public final class LRUPromptCache: PromptCaching {

    // MARK: - Types

    /// Convenience alias for the protocol-neutral ``PromptCacheHitKind``.
    ///
    /// Retained so existing call sites (and external consumers)
    /// can keep referring to `LRUPromptCache.HitKind`.
    public typealias HitKind = PromptCacheHitKind

    /// Convenience alias for the protocol-neutral ``PromptCacheFetchResult``.
    ///
    /// Retained for source compatibility; see ``HitKind``.
    public typealias FetchResult = PromptCacheFetchResult

    fileprivate struct RootKey: Hashable {
        let model: String
        let salt: UInt64
    }

    /// A single entry stored at a trie leaf.
    fileprivate final class CacheEntry {
        let promptCache: [KVCache]
        let nbytes: Int
        var lastUsed: TimeInterval

        init(
            promptCache: [KVCache],
            nbytes: Int,
            lastUsed: TimeInterval = Date().timeIntervalSince1970
        ) {
            self.promptCache = promptCache
            self.nbytes = nbytes
            self.lastUsed = lastUsed
        }
    }

    /// A node in the trie. Children are keyed by token ID.
    fileprivate final class TrieNode {
        var children: [Int32: TrieNode] = [:]
        var cache: CacheEntry?
    }

    /// LRU order tracking with support for checkpoint vs regular entries.
    fileprivate final class CacheOrder {
        /// Regular LRU entries (most-recently-used at the back).
        private var lru: [(key: RootKey, tokens: [Int])] = []
        /// Checkpoint LRU entries (most-recently-used at the back).
        private var lruCheckpoints: [(key: RootKey, tokens: [Int])] = []

        var count: Int { lru.count + lruCheckpoints.count }

        func push(key: RootKey, tokens: [Int], checkpoint: Bool = false) {
            if checkpoint {
                lruCheckpoints.append((key, tokens))
            } else {
                lru.append((key, tokens))
            }
        }

        /// Remove an entry, returning whether it was a checkpoint entry
        /// (`true`), a regular entry (`false`), or absent (`nil`). The return
        /// value lets `touch` re-insert the entry into its original eviction
        /// pool instead of silently demoting checkpoints into the regular LRU.
        @discardableResult
        func remove(key: RootKey, tokens: [Int]) -> Bool? {
            if let idx = lru.firstIndex(where: { $0.key == key && $0.tokens == tokens }) {
                lru.remove(at: idx)
                return false
            } else if let idx = lruCheckpoints.firstIndex(where: {
                $0.key == key && $0.tokens == tokens
            }) {
                lruCheckpoints.remove(at: idx)
                return true
            }
            return nil
        }

        /// Pop the least-recently-used entry. Pops from the longer list first
        /// (matching the Python behavior which pops from whichever deque is longer).
        func pop() -> (key: RootKey, tokens: [Int])? {
            if lru.count >= lruCheckpoints.count {
                return lru.isEmpty ? nil : lru.removeFirst()
            } else {
                return lruCheckpoints.isEmpty ? nil : lruCheckpoints.removeFirst()
            }
        }
    }

    /// Result of a trie search.
    fileprivate struct SearchResult {
        let key: RootKey
        /// Non-nil if an exact match was found.
        let exact: [Int]?
        /// Non-nil if a shorter prefix with a cached entry was found.
        let shorter: [Int]?
        /// Non-nil if a longer cached entry reachable from the query's path was found.
        let longer: [Int]?
        /// How many tokens of the query matched trie edges (may exceed cached depth).
        let commonPrefix: Int
    }

    /// All mutable state, guarded by the lock.
    ///
    /// The trie (`cache`), the LRU tracker (`lru`), and the running byte total
    /// (`nBytes`) are only ever touched inside `state.withLockUnchecked`. The
    /// mutating helpers below operate on `inout State` so there is a single
    /// synchronization domain and no `@unchecked Sendable` on this type.
    fileprivate struct State {
        var cache: [RootKey: TrieNode] = [:]
        let lru = CacheOrder()
        var nBytes: Int = 0
        /// Every `(model, salt)` namespace this instance has ever held a live
        /// entry for, accumulated on insert and never cleared by eviction/trim.
        /// Preserved as the cache's ownership identity for app-level
        /// persistence layers built on top of this store.
        var managedKeys: Set<RootKey> = []
    }

    // MARK: - Properties

    /// Maximum number of cached entries.
    public let maxSize: Int

    /// Maximum total bytes across all cached entries.
    public let maxBytes: Int

    /// Lock-protected mutable state.
    ///
    /// `OSAllocatedUnfairLock` is itself `Sendable`, so wrapping the mutable
    /// state in it makes ``LRUPromptCache`` a checked `Sendable` with no
    /// `@unchecked Sendable` conformance on this type (replacing an earlier
    /// `NSLock` + `@unchecked Sendable`). The `uncheckedState:` initializer and
    /// `withLockUnchecked` accessor are required because `State` transitively
    /// holds non-`Sendable` `KVCache` values; the lock supplies the actual
    /// synchronization, and every value handed out of the lock body is either a
    /// freshly materialized deep copy or a `Sendable` scalar.
    private let state = OSAllocatedUnfairLock(uncheckedState: State())

    // MARK: - Initializer

    /// Create a new LRUPromptCache.
    ///
    /// - Parameters:
    ///   - maxSize: Maximum number of cached entries (default: 10).
    ///   - maxBytes: Maximum total bytes across all entries (default: `Int.max`).
    public init(maxSize: Int = 10, maxBytes: Int = Int.max) {
        self.maxSize = maxSize
        self.maxBytes = maxBytes
    }

    // MARK: - Public API

    /// The number of cached entries.
    public var count: Int {
        state.withLockUnchecked { $0.lru.count }
    }

    /// The total byte size of all cached entries.
    public var nbytes: Int {
        state.withLockUnchecked { $0.nBytes }
    }

    /// Fetch the nearest matching KV cache for the given token sequence.
    ///
    /// Returns a deep copy of the matched cache (mutations don't affect the
    /// stored cache) and the remainder tokens that still need processing.
    ///
    /// Match priority:
    /// 1. **Exact match** — returns cache with empty remainder.
    /// 2. **Longer prefix** — if a cached entry covers more tokens than the query
    ///    and the cache is trimmable, returns a deep-copied and trimmed cache.
    /// 3. **Shorter prefix** — returns the deepest cached prefix with remainder tokens.
    ///
    /// - Parameters:
    ///   - model: Model identifier for isolation.
    ///   - tokens: The token sequence to look up.
    ///   - salt: Per-agent isolation salt.
    /// - Returns: A tuple of (cache, remainderTokens). Cache is nil if no match found;
    ///   remainder is the full token array if no match.
    public func fetchNearestCache(
        model: String,
        tokens: [Int],
        salt: UInt64 = 0
    ) -> ([KVCache]?, [Int]) {
        let result = fetchNearestCacheResult(model: model, tokens: tokens, salt: salt)
        return (result.cache, result.remainder)
    }

    public func fetchNearestCacheResult(
        model: String,
        tokens: [Int],
        salt: UInt64 = 0
    ) -> FetchResult {
        // An empty token list is always a miss. Without this guard, `search`
        // treats it as an exact match (`lastCacheIndex == tokens.count - 1`,
        // both `-1`) once any entry exists for the model/salt, and the exact
        // branch then force-unwraps the root node's `cache`, which is always nil
        // because empty inserts are rejected — crashing after the cache warms.
        guard !tokens.isEmpty else {
            return FetchResult(cache: nil, remainder: tokens, hitKind: .none, matchedTokenCount: 0)
        }
        return state.withLockUnchecked { state in
            state.fetchNearestCache(key: RootKey(model: model, salt: salt), tokens: tokens)
        }
    }

    /// Insert a KV cache for the given token sequence.
    ///
    /// If the cache is trimmable and a shorter prefix is encountered during insertion,
    /// it is removed (the new, longer cache supersedes it). After insertion, LRU and
    /// memory-based eviction is triggered if limits are exceeded.
    ///
    /// - Parameters:
    ///   - model: Model identifier for isolation.
    ///   - tokens: The token sequence this cache covers.
    ///   - promptCache: The KV cache layers to store.
    ///   - checkpoint: Whether this is a checkpoint entry (affects eviction priority).
    ///   - salt: Per-agent isolation salt.
    public func insertCache(
        model: String,
        tokens: [Int],
        promptCache: [KVCache],
        checkpoint: Bool = false,
        salt: UInt64 = 0
    ) {
        guard !tokens.isEmpty, Self.isCacheCompatible(promptCache) else { return }
        let snapshot = Self.materializedCopy(promptCache, tokenCount: tokens.count)
        // Require *every* layer to cover the full token key. A layer's `offset`
        // is its covered-token-count (`KVCacheSimple.state` sets
        // `offset = keys.dim(2)`; `RotatingKVCache` restores `offset` from
        // `metaState`, always copied by `materializedCopy`), mirroring how the
        // fetch path equates a stored entry's covered length with its token
        // count. Use the *minimum* covered length across layers so a mixed-depth
        // snapshot (one populated layer plus a short/empty layer) is rejected
        // rather than recorded for the full sequence — otherwise a later exact
        // fetch reports every token as reused while returning the shorter,
        // unusable layer. This also subsumes the empty-state case: covered
        // length 0 (an array of empty/unpopulated caches) is < non-empty
        // `tokens.count`, so it is skipped.
        let coveredLength = snapshot.map(\.offset).min() ?? 0
        guard !snapshot.isEmpty, coveredLength >= tokens.count else { return }

        state.withLockUnchecked { state in
            state.insertCache(
                key: RootKey(model: model, salt: salt),
                tokens: tokens,
                promptCache: snapshot,
                checkpoint: checkpoint,
                maxSize: maxSize,
                maxBytes: maxBytes
            )
        }
    }

    /// Evict entries until the cache is within the given limits.
    ///
    /// - Parameters:
    ///   - nSequences: Maximum number of entries to keep (nil = no limit).
    ///   - nBytes: Maximum total bytes to keep (nil = no limit).
    public func trimTo(nSequences: Int? = nil, nBytes: Int? = nil) {
        let seqLimit = nSequences.map { max(0, $0) } ?? Int.max
        let byteLimit = nBytes.map { max(0, $0) } ?? Int.max

        state.withLockUnchecked { state in
            state.trim(seqLimit: seqLimit, byteLimit: byteLimit)
        }
    }

    /// ``PromptCaching`` conformance: forwards to ``trimTo(nSequences:nBytes:)``.
    public func trim(nSequences: Int? = nil, nBytes: Int? = nil) {
        trimTo(nSequences: nSequences, nBytes: nBytes)
    }

    public static func isCacheCompatible(_ cache: [KVCache]) -> Bool {
        // Exact dynamic types on purpose: `ChunkedKVCache` subclasses
        // `KVCacheSimple` but keeps an absolute offset over a front-trimmed
        // buffer, which `materializedCopy`'s metaState write-back rejects
        // with a fatal error. Match `InferenceScheduler.migrateCaches`.
        !cache.isEmpty
            && cache.allSatisfy {
                Swift.type(of: $0) == KVCacheSimple.self
                    || Swift.type(of: $0) == RotatingKVCache.self
            }
    }

    /// Fold a stable fingerprint of a cache topology (per-layer concrete type
    /// and window size) into a prompt-cache salt.
    ///
    /// Entries are keyed by (model, salt): without this, the same model and
    /// salt used with different cache configurations — e.g. two
    /// `GenerateParameters.maxKVSize` values producing `RotatingKVCache`s with
    /// different windows — would collide, and a lookup could return KV state
    /// shaped for the wrong window. Callers mix the topology in before any
    /// fetch/insert (the scheduler does this once per request).
    public static func topologySalt(base: UInt64, caches: [KVCache]) -> UInt64 {
        // FNV-1a over per-layer "TypeName:maxSize;" descriptors.
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for cache in caches {
            let window = cache.maxSize.map { String($0) } ?? "-"
            for byte in "\(Swift.type(of: cache)):\(window);".utf8 {
                hash ^= UInt64(byte)
                hash = hash &* 0x100_0000_01b3
            }
        }
        return base ^ hash
    }

    static func canUsePromptCache(
        input: LMInput,
        parameters: GenerateParameters,
        model: any LanguageModel
    ) -> Bool {
        guard input.image == nil, input.video == nil, input.audio == nil else { return false }
        guard parameters.kvBits == nil, parameters.kvScheme == nil else { return false }
        guard model.defaultPromptCachePolicy == .exact else { return false }
        return isCacheCompatible(model.newCache(parameters: parameters))
    }


    // MARK: - Cache materialization (immutable / static)

    /// Deep-copy a KV cache by reading and writing its state.
    ///
    /// When `tokenCount` is supplied (the insert path), each layer is normalized
    /// down to that key length: `KVCacheSimple` is sliced by `stateSnapshot`, and
    /// a `RotatingKVCache` whose restored `offset` exceeds `tokenCount` is
    /// normalized by ``normalizeRotating`` so a stored entry's covered length can
    /// never overshoot its token key. When `tokenCount` is `nil` (the fetch
    /// path) the copy is faithful — offsets are preserved exactly.
    private static func materializedCopy(_ promptCache: [KVCache], tokenCount: Int?) -> [KVCache] {
        let copy = promptCache.map { original -> KVCache in
            var copy: KVCache
            if original is KVCacheSimple {
                copy = KVCacheSimple()
            } else if let rotating = original as? RotatingKVCache {
                copy = RotatingKVCache(maxSize: rotating.maxSize ?? 0)
            } else {
                // Fallback: KVCacheSimple for unknown types
                copy = KVCacheSimple()
            }
            let originalState = stateSnapshot(for: original, tokenCount: tokenCount)
            // Only restore state if the cache has data (non-empty state).
            // Empty state means keys/values are nil (e.g., mock model didn't
            // populate the cache), and setting empty state would crash.
            if !originalState.isEmpty {
                copy.state = originalState
            }
            copy.metaState = original.metaState
            if copy is RotatingKVCache, let tokenCount {
                copy = normalizeRotating(copy, tokenCount: tokenCount)
            }
            return copy
        }
        let arrays = copy.flatMap { $0.state }
        if !arrays.isEmpty {
            eval(arrays)
        }
        return copy
    }

    /// Normalize a freshly materialized `RotatingKVCache` copy down to
    /// `tokenCount` so its covered length (and therefore its rope offset) never
    /// exceeds the token key it will be stored under.
    ///
    /// `KVCacheSimple` is already truncated to the key by ``stateSnapshot`` (and
    /// its `state` setter resets `offset = keys.dim(2)`), but the rotating path
    /// restores `offset` from `metaState`, so an over-covered snapshot would
    /// otherwise be recorded as covering the full key while returning KV state +
    /// a rope offset that include extra tokens — letting an exact lookup skip
    /// evaluation from the wrong position.
    ///
    /// - If `offset <= tokenCount`, the copy already fits: return it unchanged.
    /// - If the cache is still linear (never rotated: `idx == offset`, so the
    ///   physical buffer holds tokens `0..<offset` in temporal order), `trim` the
    ///   excess. After trimming, `offset == idx == tokenCount` and the `state`
    ///   getter yields exactly the `..<tokenCount` temporal prefix.
    /// - If it has already wrapped (`idx != offset`), a position-sliced prefix is
    ///   not the temporal prefix, so a clean truncation to the key is impossible.
    ///   Return a fresh empty `RotatingKVCache` (offset 0) so the insert coverage
    ///   guard (`min` offset `>= tokens.count`) rejects the whole snapshot rather
    ///   than storing an unusable over-covered entry.
    private static func normalizeRotating(_ copy: KVCache, tokenCount: Int) -> KVCache {
        guard let rotating = copy as? RotatingKVCache, rotating.offset > tokenCount else {
            return copy
        }
        // metaState == [keep, maxCacheSize, step, offset, idx]; `idx == offset`
        // iff the cache has never rotated and the buffer is in temporal order.
        // Compare `idx` against the authoritative `offset` (not `meta[3]`) so a
        // missing/garbled value fails closed (treated as wrapped → rejected)
        // rather than matching `nil == nil`.
        let meta = rotating.metaState
        let isLinear = meta.count == 5 && Int(meta[4]) == rotating.offset
        if isLinear {
            rotating.trim(rotating.offset - tokenCount)
            // `trim` is logical (it only decrements offset/idx); the copied
            // backing buffers still hold the pre-trim length, while byte
            // accounting sums the `state` getter's `..<offset` slices. Re-set
            // the state from those slices (physically detached) so the entry
            // retains exactly what `maxBytes` accounting measures.
            let truncated = rotating.state.map {
                $0 + MLXArray.zeros($0.shape, dtype: $0.dtype)
            }
            let meta = rotating.metaState
            rotating.state = truncated
            rotating.metaState = meta
            return rotating
        }
        return RotatingKVCache(maxSize: rotating.maxSize ?? 0)
    }

    private static func stateSnapshot(for cache: KVCache, tokenCount: Int?) -> [MLXArray] {
        cache.state.map { array in
            var snapshot = array
            if cache is KVCacheSimple,
                let tokenCount,
                array.ndim >= 3,
                array.dim(2) > tokenCount
            {
                snapshot = array[.ellipsis, ..<tokenCount, 0...]
            }
            let copied = snapshot + MLXArray.zeros(snapshot.shape, dtype: snapshot.dtype)
            return copied
        }
    }

    private static func canSafelyTrim(_ cache: [KVCache], numTokens: Int) -> Bool {
        guard numTokens > 0 else { return true }
        let minCachedSeqLen =
            cache.map { layer -> Int in
                guard let first = layer.state.first, first.ndim >= 3 else { return 0 }
                return first.dim(2)
            }.min() ?? 0
        return numTokens < minCachedSeqLen
    }
}

// MARK: - State mutating operations

extension LRUPromptCache.State {

    /// Search the trie for the best match.
    private func search(key: LRUPromptCache.RootKey, tokens: [Int]) -> LRUPromptCache.SearchResult {
        guard let root = cache[key] else {
            return LRUPromptCache.SearchResult(
                key: key, exact: nil, shorter: nil, longer: nil, commonPrefix: 0)
        }

        var current = root
        var lastCacheIndex = -1
        var index = 0

        while index < tokens.count, let next = current.children[Int32(tokens[index])] {
            current = next
            if current.cache != nil {
                lastCacheIndex = index
            }
            index += 1
        }

        // Exact match: the deepest cached node is at the last token
        if lastCacheIndex == tokens.count - 1 {
            return LRUPromptCache.SearchResult(
                key: key, exact: tokens, shorter: nil, longer: nil, commonPrefix: 0)
        }

        // Shorter prefix
        var shorter: [Int]?
        if lastCacheIndex >= 0 {
            shorter = Array(tokens[...lastCacheIndex])
        }

        // Longer prefix: search for the shortest cached descendant from `current`
        var longer: [Int]?
        let commonPrefix = index
        if index > 0 {
            var best: [Int]?
            var stack: [(node: LRUPromptCache.TrieNode, extra: [Int])] = [(current, [])]
            while !stack.isEmpty {
                let (node, extra) = stack.removeLast()
                if node.cache != nil {
                    if best == nil || extra.count < best!.count {
                        best = extra
                    }
                } else {
                    for (tok, child) in node.children {
                        stack.append((child, extra + [Int(tok)]))
                    }
                }
            }
            if let best {
                longer = Array(tokens[..<index]) + best
            }
        }

        return LRUPromptCache.SearchResult(
            key: key, exact: nil, shorter: shorter, longer: longer,
            commonPrefix: commonPrefix)
    }

    /// Get the cache entry at the given path.
    private func get(key: LRUPromptCache.RootKey, tokens: [Int]) -> LRUPromptCache.CacheEntry {
        var current = cache[key]!
        for tok in tokens {
            current = current.children[Int32(tok)]!
        }
        return current.cache!
    }

    /// Delete a cache entry from the trie.
    mutating func delete(key: LRUPromptCache.RootKey, tokens: [Int]) {
        guard let root = cache[key] else { return }

        var path = [root]
        for tok in tokens {
            guard let next = path.last!.children[Int32(tok)] else { return }
            path.append(next)
        }

        guard let entry = path.last?.cache else { return }
        nBytes -= entry.nbytes
        path.last!.cache = nil

        // Clean up empty nodes from the bottom
        for i in stride(from: tokens.count - 1, through: 0, by: -1) {
            let child = path[i + 1]
            if child.children.isEmpty && child.cache == nil {
                path[i].children.removeValue(forKey: Int32(tokens[i]))
            } else {
                break
            }
        }
    }

    /// Refresh LRU recency for the given entry (move to most-recently-used).
    private func touch(key: LRUPromptCache.RootKey, tokens: [Int]) {
        let entry = get(key: key, tokens: tokens)
        entry.lastUsed = Date().timeIntervalSince1970
        // Preserve the entry's eviction class: re-push as a checkpoint if it was
        // one, so a recently-used checkpoint isn't demoted into the regular
        // eviction pool and evicted ahead of its checkpoint peers.
        let wasCheckpoint = lru.remove(key: key, tokens: tokens) ?? false
        lru.push(key: key, tokens: tokens, checkpoint: wasCheckpoint)
    }

    /// Internal fetch (must be called while holding the lock).
    func fetchNearestCache(key: LRUPromptCache.RootKey, tokens: [Int]) -> LRUPromptCache.FetchResult
    {
        let result = search(key: key, tokens: tokens)

        // Exact match
        if let exact = result.exact {
            let entry = get(key: result.key, tokens: exact)
            touch(key: result.key, tokens: exact)
            return LRUPromptCache.FetchResult(
                cache: LRUPromptCache.deepCopy(entry.promptCache),
                remainder: [],
                hitKind: .exact,
                matchedTokenCount: tokens.count
            )
        }

        let shortLength = result.shorter?.count ?? 0

        // Longer prefix: if the cached entry is longer than the query and trimmable
        if let longer = result.longer, result.commonPrefix > shortLength {
            let entry = get(key: result.key, tokens: longer)
            if canTrimPromptCache(entry.promptCache) {
                let prefix = min(tokens.count, result.commonPrefix)
                let numToTrim = longer.count - prefix
                if LRUPromptCache.canSafelyTrim(entry.promptCache, numTokens: numToTrim) {
                    let copy = LRUPromptCache.deepCopy(entry.promptCache)
                    trimPromptCache(copy, numTokens: numToTrim)
                    let remainder = prefix < tokens.count ? Array(tokens[prefix...]) : []
                    touch(key: result.key, tokens: longer)
                    return LRUPromptCache.FetchResult(
                        cache: copy,
                        remainder: remainder,
                        hitKind: .longer,
                        matchedTokenCount: prefix
                    )
                }
            }
        }

        // Shorter prefix
        if shortLength > 0 {
            let entry = get(key: result.key, tokens: result.shorter!)
            touch(key: result.key, tokens: result.shorter!)
            return LRUPromptCache.FetchResult(
                cache: LRUPromptCache.deepCopy(entry.promptCache),
                remainder: Array(tokens[shortLength...]),
                hitKind: .shorter,
                matchedTokenCount: shortLength
            )
        }

        // No match
        return LRUPromptCache.FetchResult(
            cache: nil, remainder: tokens, hitKind: .none, matchedTokenCount: 0)
    }

    /// Internal insert (must be called while holding the lock).
    mutating func insertCache(
        key: LRUPromptCache.RootKey,
        tokens: [Int],
        promptCache: [KVCache],
        checkpoint: Bool,
        maxSize: Int,
        maxBytes: Int
    ) {
        let isTrimmable = canTrimPromptCache(promptCache)

        // Record namespace ownership (survives later eviction/trim; used by
        // app-level persistence layers built on top of this store).
        managedKeys.insert(key)

        if cache[key] == nil {
            cache[key] = LRUPromptCache.TrieNode()
        }
        var current = cache[key]!

        for i in 0 ..< tokens.count {
            let tok = Int32(tokens[i])
            if current.children[tok] == nil {
                current.children[tok] = LRUPromptCache.TrieNode()
            }
            // If inserting a trimmable cache and we pass through an existing cached node,
            // remove it (the new longer cache supersedes the shorter one).
            if isTrimmable, current.cache != nil {
                nBytes -= current.cache!.nbytes
                current.cache = nil
                lru.remove(key: key, tokens: Array(tokens[..<i]))
            }
            current = current.children[tok]!
        }

        let cacheBytes = promptCache.reduce(0) { $0 + $1.state.reduce(0) { $0 + $1.nbytes } }
        if current.cache != nil {
            // Update existing entry: remove from LRU and reinsert
            nBytes -= current.cache!.nbytes
            lru.remove(key: key, tokens: tokens)
            current.cache = LRUPromptCache.CacheEntry(promptCache: promptCache, nbytes: cacheBytes)
            nBytes += cacheBytes
        } else {
            current.cache = LRUPromptCache.CacheEntry(promptCache: promptCache, nbytes: cacheBytes)
            nBytes += cacheBytes
        }

        lru.push(key: key, tokens: tokens, checkpoint: checkpoint)

        // Evict if over maxSize
        if lru.count > maxSize {
            if let evicted = lru.pop() {
                delete(key: evicted.key, tokens: evicted.tokens)
            }
        }

        // Evict if over maxBytes
        while nBytes > maxBytes {
            guard let evicted = lru.pop() else { break }
            delete(key: evicted.key, tokens: evicted.tokens)
        }
    }

    /// Evict entries until within both the sequence-count and byte limits
    /// (must be called while holding the lock).
    mutating func trim(seqLimit: Int, byteLimit: Int) {
        while lru.count > seqLimit {
            guard let evicted = lru.pop() else { break }
            delete(key: evicted.key, tokens: evicted.tokens)
        }
        while nBytes > byteLimit {
            guard let evicted = lru.pop() else { break }
            delete(key: evicted.key, tokens: evicted.tokens)
        }
    }
}

extension LRUPromptCache {
    /// Deep-copy a KV cache by reading and writing its state.
    fileprivate static func deepCopy(_ promptCache: [KVCache]) -> [KVCache] {
        materializedCopy(promptCache, tokenCount: nil)
    }
}
