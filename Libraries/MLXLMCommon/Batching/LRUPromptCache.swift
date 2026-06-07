// Copyright © 2024 Apple Inc.

import CryptoKit
import Dispatch
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
/// - `save(to:maxDiskBytes:)` / `load(from:allowedModels:)` — disk persistence
public final class LRUPromptCache: PersistablePromptCache {

    // MARK: - Types

    /// Convenience alias for the protocol-neutral ``PromptCacheHitKind``.
    ///
    /// Retained so existing call sites (and external consumers such as PicoCore)
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
    /// `@unchecked Sendable` conformance on this type (replacing batching3's
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
        !cache.isEmpty && cache.allSatisfy { $0 is KVCacheSimple || $0 is RotatingKVCache }
    }

    static func canUsePromptCache(
        input: LMInput,
        parameters: GenerateParameters,
        model: any LanguageModel
    ) -> Bool {
        guard input.image == nil, input.video == nil, input.audio == nil else { return false }
        guard parameters.kvBits == nil else { return false }
        guard model.defaultPromptCachePolicy == .exact else { return false }
        return isCacheCompatible(model.newCache(parameters: parameters))
    }

    public func save(to directory: URL, maxDiskBytes: Int? = nil) async throws {
        try await Self.runPersistenceTask {
            try self.saveSynchronously(to: directory, maxDiskBytes: maxDiskBytes)
        }
    }

    public func load(from directory: URL, allowedModels: Set<String>? = nil) async throws {
        try await Self.runPersistenceTask {
            try self.loadSynchronously(from: directory, allowedModels: allowedModels)
        }
    }

    private func saveSynchronously(to directory: URL, maxDiskBytes: Int? = nil) throws {
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )

        let snapshots = diskSnapshots()
        #if DEBUG
            let skipped = snapshots.filter { !Self.isDiskPersistable($0.promptCache) }.count
            if skipped > 0 {
                print("[PromptCache] skipping disk persistence for \(skipped) non-KVCacheSimple leaves")
            }
        #endif

        let liveStems = Set(
            snapshots.lazy
                .filter { Self.isDiskPersistable($0.promptCache) }
                .map(\.fileStem)
        )
        // The `(model, salt)` pairs this cache currently manages. Stale-sidecar
        // cleanup is scoped to these so a shared persistence directory is never
        // corrupted: saving a cache that only holds model A never deletes model
        // B's committed entries, and — because salt is part of the cache key —
        // saving one salt never deletes another salt's same-model entries.
        let liveKeys = Set(snapshots.map(\.key))

        for snapshot in snapshots where Self.isDiskPersistable(snapshot.promptCache) {
            let baseURL = directory.appendingPathComponent(snapshot.fileStem)
            let tensorURL = baseURL.appendingPathExtension("safetensors")
            let sidecarURL = baseURL.appendingPathExtension("json")
            let tempTensorURL = directory.appendingPathComponent(
                "\(snapshot.fileStem).\(UUID().uuidString).tmp.safetensors")
            let tempSidecarURL = directory.appendingPathComponent(
                "\(snapshot.fileStem).\(UUID().uuidString).tmp.json")
            defer {
                try? FileManager.default.removeItem(at: tempTensorURL)
                try? FileManager.default.removeItem(at: tempSidecarURL)
            }

            try savePromptCache(
                url: tempTensorURL,
                cache: snapshot.promptCache,
                metadata: [
                    "formatVersion": String(Self.diskFormatVersion),
                    "model": snapshot.key.model,
                    "salt": String(snapshot.key.salt),
                    "tokenHash": snapshot.tokenHash,
                ]
            )

            let sidecar = DiskSidecar(
                formatVersion: Self.diskFormatVersion,
                model: snapshot.key.model,
                salt: snapshot.key.salt,
                tokens: snapshot.tokens,
                tokenHash: snapshot.tokenHash,
                tokenCount: snapshot.tokens.count,
                byteCount: snapshot.nbytes,
                lastUsed: snapshot.lastUsed
            )
            let data = try JSONEncoder().encode(sidecar)
            try data.write(to: tempSidecarURL, options: .atomic)

            try? FileManager.default.removeItem(at: tensorURL)
            try FileManager.default.moveItem(at: tempTensorURL, to: tensorURL)

            // The sidecar is the discoverable commit record. Moving it last makes
            // interrupted saves load as cache misses instead of half-written entries.
            try? FileManager.default.removeItem(at: sidecarURL)
            try FileManager.default.moveItem(at: tempSidecarURL, to: sidecarURL)
        }

        try Self.removeStaleSidecars(
            directory: directory, liveStems: liveStems, liveKeys: liveKeys)

        if let maxDiskBytes {
            try Self.enforceDiskBudget(directory: directory, maxDiskBytes: maxDiskBytes)
        }
    }

    /// Delete this cache's own committed `entry_` sidecars (and their paired
    /// tensors) whose stem is no longer in the live snapshot set, so a later
    /// `load` doesn't resurrect prompts that were evicted or trimmed from
    /// memory since they were last written.
    ///
    /// Deletion is scoped two ways so a shared persistence directory is never
    /// corrupted: only `entry_`-prefixed final sidecars are considered (via
    /// ``isFinalSidecar``), and a stale sidecar is removed only when it decodes
    /// to valid metadata whose `(model, salt)` is one this cache manages
    /// (`liveKeys`). Salt is part of the cache key, so scoping on model alone
    /// would let one salt delete another salt's same-model entries. A sidecar
    /// that can't be decoded, or whose `(model, salt)` this cache doesn't own
    /// (for example another cache instance or model/salt scope sharing the
    /// directory, which `load(from:allowedModels:)` supports), is left
    /// untouched. When in doubt, do not delete.
    private static func removeStaleSidecars(
        directory: URL,
        liveStems: Set<String>,
        liveKeys: Set<RootKey>
    ) throws {
        let fileManager = FileManager.default
        let sidecars = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        .filter { isFinalSidecar($0) }

        let decoder = JSONDecoder()
        for sidecarURL in sidecars {
            let stem = sidecarURL.deletingPathExtension().lastPathComponent
            guard !liveStems.contains(stem) else { continue }
            // Only delete sidecars this cache owns: decode the metadata and
            // require its `(model, salt)` to be one of this cache's live keys.
            // Never delete an undecodable sidecar or one belonging to another
            // model or salt.
            guard let data = try? Data(contentsOf: sidecarURL),
                let sidecar = try? decoder.decode(DiskSidecar.self, from: data),
                liveKeys.contains(RootKey(model: sidecar.model, salt: sidecar.salt))
            else {
                continue
            }
            let tensorURL = sidecarURL.deletingPathExtension()
                .appendingPathExtension("safetensors")
            try? fileManager.removeItem(at: sidecarURL)
            try? fileManager.removeItem(at: tensorURL)
        }
    }

    private func loadSynchronously(from directory: URL, allowedModels: Set<String>? = nil) throws {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: directory.path) else { return }

        let sidecarURLs = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        )
        // Only committed final sidecars written by this cache. Excluding
        // `*.tmp.json` honors the "sidecar move is the commit record" ordering
        // in `saveSynchronously`: a crash after writing the temp pair but before
        // the final move must load as a miss, not a half-written entry. The
        // `entry_` prefix also keeps us from touching unrelated `.json` files in
        // a shared directory.
        .filter { Self.isFinalSidecar($0) }

        let decoder = JSONDecoder()
        var decoded = [(DiskSidecar, URL)]()
        for sidecarURL in sidecarURLs {
            guard let data = try? Data(contentsOf: sidecarURL),
                let sidecar = try? decoder.decode(DiskSidecar.self, from: data),
                sidecar.formatVersion == Self.diskFormatVersion,
                sidecar.tokenCount == sidecar.tokens.count,
                sidecar.tokenHash == Self.tokenHashHex(sidecar.tokens),
                allowedModels?.contains(sidecar.model) ?? true
            else {
                continue
            }
            decoded.append((sidecar, sidecarURL))
        }

        decoded.sort { $0.0.lastUsed < $1.0.lastUsed }

        for (sidecar, sidecarURL) in decoded {
            let tensorURL = sidecarURL.deletingPathExtension().appendingPathExtension("safetensors")
            guard fileManager.fileExists(atPath: tensorURL.path),
                let (promptCache, metadata) = try? loadPromptCache(url: tensorURL),
                metadata["formatVersion"] == String(Self.diskFormatVersion),
                metadata["model"] == sidecar.model,
                metadata["salt"] == String(sidecar.salt),
                metadata["tokenHash"] == sidecar.tokenHash,
                Self.isDiskPersistable(promptCache)
            else {
                continue
            }

            let entryBytes = Self.cacheByteCount(promptCache)
            guard entryBytes <= maxBytes else { continue }

            insertCache(
                model: sidecar.model,
                tokens: sidecar.tokens,
                promptCache: promptCache,
                salt: sidecar.salt
            )
        }
    }

    // MARK: - Disk persistence helpers

    private static let diskFormatVersion = 1
    private static let persistenceQueue = DispatchQueue(
        label: "org.ml-explore.mlx-swift-lm.lrucache.persistence",
        qos: .utility
    )

    private static func runPersistenceTask(
        _ work: @escaping @Sendable () throws -> Void
    ) async throws {
        try await withCheckedThrowingContinuation { continuation in
            persistenceQueue.async {
                do {
                    try work()
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private struct DiskSidecar: Codable {
        let formatVersion: Int
        let model: String
        let salt: UInt64
        let tokens: [Int]
        let tokenHash: String
        let tokenCount: Int
        let byteCount: Int
        let lastUsed: TimeInterval
    }

    private struct DiskSnapshot {
        let key: RootKey
        let tokens: [Int]
        let promptCache: [KVCache]
        let nbytes: Int
        let lastUsed: TimeInterval

        var tokenHash: String { LRUPromptCache.tokenHashHex(tokens) }

        var fileStem: String {
            "entry_\(LRUPromptCache.stringHashHex(key.model))_\(String(format: "%016llx", key.salt))_\(tokens.count)_\(tokenHash)"
        }
    }

    private func diskSnapshots() -> [DiskSnapshot] {
        state.withLockUnchecked { state in
            var snapshots = [DiskSnapshot]()
            var tokens = [Int]()
            for (key, root) in state.cache {
                tokens.removeAll(keepingCapacity: true)
                Self.collectDiskSnapshots(key: key, node: root, tokens: &tokens, into: &snapshots)
            }
            return snapshots
        }
    }

    private static func collectDiskSnapshots(
        key: RootKey,
        node: TrieNode,
        tokens: inout [Int],
        into snapshots: inout [DiskSnapshot]
    ) {
        if let entry = node.cache {
            // Deep-copy the cache *inside the lock* (this static helper is only
            // reached from `diskSnapshots()` while `state.withLockUnchecked` is
            // held). Handing out the trie's live, mutable `[KVCache]` would let
            // the off-thread `savePromptCache` serialization race against
            // concurrent mutations on another thread.
            snapshots.append(
                DiskSnapshot(
                    key: key,
                    tokens: tokens,
                    promptCache: LRUPromptCache.deepCopy(entry.promptCache),
                    nbytes: entry.nbytes,
                    lastUsed: entry.lastUsed
                )
            )
        }
        for (token, child) in node.children {
            tokens.append(Int(token))
            collectDiskSnapshots(
                key: key,
                node: child,
                tokens: &tokens,
                into: &snapshots
            )
            tokens.removeLast()
        }
    }

    private static func isDiskPersistable(_ cache: [KVCache]) -> Bool {
        !cache.isEmpty && cache.allSatisfy { $0 is KVCacheSimple }
    }

    private static func cacheByteCount(_ cache: [KVCache]) -> Int {
        cache.reduce(0) { total, layer in
            total + layer.state.reduce(0) { $0 + $1.nbytes }
        }
    }

    private static func tokenHashHex(_ tokens: [Int]) -> String {
        var data = Data()
        data.reserveCapacity(tokens.count * MemoryLayout<Int64>.size)
        for token in tokens {
            var value = Int64(token).littleEndian
            withUnsafeBytes(of: &value) { bytes in
                data.append(contentsOf: bytes)
            }
        }
        return sha256Hex(data)
    }

    private static func stringHashHex(_ string: String) -> String {
        sha256Hex(Data(string.utf8))
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func enforceDiskBudget(directory: URL, maxDiskBytes: Int) throws {
        let fileManager = FileManager.default
        let sidecars = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        )
        .filter { isFinalSidecar($0) }

        // Only enforce the budget against this cache's own entries: a committed
        // `entry_` sidecar that decodes to valid metadata. Undecodable or
        // foreign `.json` files in a shared directory are left untouched so the
        // budget never deletes data this cache doesn't own.
        var entries = sidecars.compactMap {
            sidecarURL -> (sidecar: URL, tensor: URL, lastUsed: TimeInterval, bytes: Int)? in
            guard let data = try? Data(contentsOf: sidecarURL),
                let sidecar = try? JSONDecoder().decode(DiskSidecar.self, from: data)
            else {
                return nil
            }
            let tensorURL = sidecarURL.deletingPathExtension().appendingPathExtension("safetensors")
            let bytes = fileSize(sidecarURL) + fileSize(tensorURL)
            return (sidecarURL, tensorURL, sidecar.lastUsed, bytes)
        }

        var totalBytes = entries.reduce(0) { $0 + $1.bytes }
        guard totalBytes > maxDiskBytes else { return }

        entries.sort { $0.lastUsed < $1.lastUsed }
        for entry in entries where totalBytes > maxDiskBytes {
            try? fileManager.removeItem(at: entry.sidecar)
            try? fileManager.removeItem(at: entry.tensor)
            totalBytes -= entry.bytes
        }
    }

    /// Whether `url` is a committed final sidecar written by this cache: an
    /// `entry_`-prefixed `.json` file that is not an in-flight `.tmp.json`
    /// temporary. Used by both `load` and the disk-budget cleanup to avoid
    /// touching interrupted saves or unrelated files in a shared directory.
    private static func isFinalSidecar(_ url: URL) -> Bool {
        let name = url.lastPathComponent
        return url.pathExtension == "json"
            && name.hasPrefix("entry_")
            && !name.hasSuffix(".tmp.json")
    }

    private static func fileSize(_ url: URL) -> Int {
        (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
    }

    // MARK: - Cache materialization (immutable / static)

    /// Deep-copy a KV cache by reading and writing its state.
    private static func materializedCopy(_ promptCache: [KVCache], tokenCount: Int?) -> [KVCache] {
        let copy = promptCache.map { original in
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
            return copy
        }
        let arrays = copy.flatMap { $0.state }
        if !arrays.isEmpty {
            eval(arrays)
        }
        return copy
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
        let minCachedSeqLen = cache.map { layer -> Int in
            guard let first = layer.state.first, first.ndim >= 3 else { return 0 }
            return first.dim(2)
        }.min() ?? 0
        return numTokens < minCachedSeqLen
    }
}

// MARK: - State mutating operations

extension LRUPromptCache.State {

    /// Search the trie for the best match.
    private func search(key: RootKey, tokens: [Int]) -> LRUPromptCache.SearchResult {
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
    private func get(key: RootKey, tokens: [Int]) -> LRUPromptCache.CacheEntry {
        var current = cache[key]!
        for tok in tokens {
            current = current.children[Int32(tok)]!
        }
        return current.cache!
    }

    /// Delete a cache entry from the trie.
    mutating func delete(key: RootKey, tokens: [Int]) {
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
    private func touch(key: RootKey, tokens: [Int]) {
        let entry = get(key: key, tokens: tokens)
        entry.lastUsed = Date().timeIntervalSince1970
        // Preserve the entry's eviction class: re-push as a checkpoint if it was
        // one, so a recently-used checkpoint isn't demoted into the regular
        // eviction pool and evicted ahead of its checkpoint peers.
        let wasCheckpoint = lru.remove(key: key, tokens: tokens) ?? false
        lru.push(key: key, tokens: tokens, checkpoint: wasCheckpoint)
    }

    /// Internal fetch (must be called while holding the lock).
    func fetchNearestCache(key: RootKey, tokens: [Int]) -> LRUPromptCache.FetchResult {
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
        key: RootKey,
        tokens: [Int],
        promptCache: [KVCache],
        checkpoint: Bool,
        maxSize: Int,
        maxBytes: Int
    ) {
        let isTrimmable = canTrimPromptCache(promptCache)

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
