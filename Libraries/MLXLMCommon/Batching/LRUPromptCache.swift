// Copyright © 2024 Apple Inc.

import Foundation
import Dispatch
import CryptoKit
import MLX

// MARK: - LRUPromptCache

/// Trie-based LRU cache storing KV caches keyed by token sequences.
///
/// Ported from Python mlx-lm's `LRUPromptCache`. Supports exact, shorter-prefix,
/// and longer-prefix lookups. Fetch always returns a deep copy (independent of
/// stored cache). Model isolation ensures caches from different models don't
/// cross-contaminate.
///
/// Thread safety is ensured via `NSLock`-based serialization.
///
/// Key operations:
/// - `insertCache(model:tokens:promptCache:)` — store a KV cache for a token sequence
/// - `fetchNearestCache(model:tokens:)` — find the best matching cached prefix
/// - `trimTo(nSequences:nBytes:)` — memory-aware eviction
public final class LRUPromptCache: @unchecked Sendable {

    // MARK: - Types

    public enum HitKind: String, Sendable, Equatable {
        case none
        case exact
        case shorter
        case longer
    }

    public struct FetchResult {
        public let cache: [KVCache]?
        public let remainder: [Int]
        public let hitKind: HitKind
        public let matchedTokenCount: Int

        public var reusedTokenCount: Int { matchedTokenCount }
    }

    private struct RootKey: Hashable {
        let model: String
        let salt: UInt64
    }

    /// A single entry stored at a trie leaf.
    private final class CacheEntry {
        let promptCache: [KVCache]
        let nbytes: Int
        var lastUsed: TimeInterval

        init(promptCache: [KVCache], nbytes: Int, lastUsed: TimeInterval = Date().timeIntervalSince1970) {
            self.promptCache = promptCache
            self.nbytes = nbytes
            self.lastUsed = lastUsed
        }
    }

    /// A node in the trie. Children are keyed by token ID.
    private final class TrieNode {
        var children: [Int32: TrieNode] = [:]
        var cache: CacheEntry?
    }

    /// LRU order tracking with support for checkpoint vs regular entries.
    private final class CacheOrder {
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

        func remove(key: RootKey, tokens: [Int]) {
            if let idx = lru.firstIndex(where: { $0.key == key && $0.tokens == tokens }) {
                lru.remove(at: idx)
            } else if let idx = lruCheckpoints.firstIndex(where: {
                $0.key == key && $0.tokens == tokens
            }) {
                lruCheckpoints.remove(at: idx)
            }
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
    private struct SearchResult {
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

    // MARK: - Properties

    /// Maximum number of cached entries.
    public let maxSize: Int

    /// Maximum total bytes across all cached entries.
    public let maxBytes: Int

    /// Root trie nodes keyed by model identifier.
    private var cache: [RootKey: TrieNode] = [:]

    /// LRU order tracker.
    private let lru = CacheOrder()

    /// Total byte size of all cached entries.
    private var _nBytes: Int = 0

    /// Lock for thread safety.
    private let lock = NSLock()

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
        lock.lock()
        defer { lock.unlock() }
        return lru.count
    }

    /// The total byte size of all cached entries.
    public var nbytes: Int {
        lock.lock()
        defer { lock.unlock() }
        return _nBytes
    }

    /// Fetch the nearest matching KV cache for the given token sequence.
    ///
    /// Returns a deep copy of the matched cache (mutations don't affect stored cache)
    /// and the remainder tokens that still need processing.
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
        lock.lock()
        defer { lock.unlock() }
        return _fetchNearestCache(key: RootKey(model: model, salt: salt), tokens: tokens)
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
    public func insertCache(
        model: String,
        tokens: [Int],
        promptCache: [KVCache],
        checkpoint: Bool = false,
        salt: UInt64 = 0
    ) {
        guard !tokens.isEmpty, Self.isCacheCompatible(promptCache) else { return }
        let snapshot = Self.materializedCopy(promptCache, tokenCount: tokens.count)
        guard !snapshot.isEmpty else { return }

        lock.lock()
        defer { lock.unlock() }
        _insertCache(
            key: RootKey(model: model, salt: salt),
            tokens: tokens,
            promptCache: snapshot,
            checkpoint: checkpoint
        )
    }

    /// Evict entries until the cache is within the given limits.
    ///
    /// - Parameters:
    ///   - nSequences: Maximum number of entries to keep (nil = no limit).
    ///   - nBytes: Maximum total bytes to keep (nil = no limit).
    public func trimTo(nSequences: Int? = nil, nBytes: Int? = nil) {
        lock.lock()
        defer { lock.unlock() }

        let seqLimit = nSequences.map { max(0, $0) } ?? Int.max
        let byteLimit = nBytes.map { max(0, $0) } ?? Int.max

        while lru.count > seqLimit {
            guard let evicted = lru.pop() else { break }
            _delete(key: evicted.key, tokens: evicted.tokens)
        }
        while _nBytes > byteLimit {
            guard let evicted = lru.pop() else { break }
            _delete(key: evicted.key, tokens: evicted.tokens)
        }
    }

    public static func isCacheCompatible(_ cache: [KVCache]) -> Bool {
        !cache.isEmpty && cache.allSatisfy { $0 is KVCacheSimple || $0 is RotatingKVCache }
    }

    static func canUsePromptCache(
        input: LMInput,
        parameters: GenerateParameters,
        model: any LanguageModel
    ) -> Bool {
        guard input.image == nil, input.video == nil else { return false }
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

        for snapshot in snapshots where Self.isDiskPersistable(snapshot.promptCache) {
            let baseURL = directory.appendingPathComponent(snapshot.fileStem)
            let tensorURL = baseURL.appendingPathExtension("safetensors")
            let sidecarURL = baseURL.appendingPathExtension("json")
            let tempTensorURL = directory.appendingPathComponent("\(snapshot.fileStem).\(UUID().uuidString).tmp.safetensors")
            let tempSidecarURL = directory.appendingPathComponent("\(snapshot.fileStem).\(UUID().uuidString).tmp.json")
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

        if let maxDiskBytes {
            try Self.enforceDiskBudget(directory: directory, maxDiskBytes: maxDiskBytes)
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
        .filter { $0.pathExtension == "json" }

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

    // MARK: - Private Implementation

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
        lock.lock()
        defer { lock.unlock() }

        var snapshots = [DiskSnapshot]()
        var tokens = [Int]()
        for (key, root) in cache {
            tokens.removeAll(keepingCapacity: true)
            collectDiskSnapshots(key: key, node: root, tokens: &tokens, into: &snapshots)
        }
        return snapshots
    }

    private func collectDiskSnapshots(
        key: RootKey,
        node: TrieNode,
        tokens: inout [Int],
        into snapshots: inout [DiskSnapshot]
    ) {
        if let entry = node.cache {
            snapshots.append(
                DiskSnapshot(
                    key: key,
                    tokens: tokens,
                    promptCache: entry.promptCache,
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
        .filter { $0.pathExtension == "json" }

        var entries = sidecars.map { sidecarURL -> (sidecar: URL, tensor: URL, lastUsed: TimeInterval, bytes: Int) in
            let tensorURL = sidecarURL.deletingPathExtension().appendingPathExtension("safetensors")
            let data = try? Data(contentsOf: sidecarURL)
            let sidecar = data.flatMap { try? JSONDecoder().decode(DiskSidecar.self, from: $0) }
            let bytes = fileSize(sidecarURL) + fileSize(tensorURL)
            return (sidecarURL, tensorURL, sidecar?.lastUsed ?? 0, bytes)
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

    private static func fileSize(_ url: URL) -> Int {
        (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
    }

    /// Search the trie for the best match.
    private func _search(key: RootKey, tokens: [Int]) -> SearchResult {
        guard let root = cache[key] else {
            return SearchResult(
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
            return SearchResult(
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
            var stack: [(node: TrieNode, extra: [Int])] = [(current, [])]
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

        return SearchResult(
            key: key, exact: nil, shorter: shorter, longer: longer,
            commonPrefix: commonPrefix)
    }

    /// Get the cache entry at the given path.
    private func _get(key: RootKey, tokens: [Int]) -> CacheEntry {
        var current = cache[key]!
        for tok in tokens {
            current = current.children[Int32(tok)]!
        }
        return current.cache!
    }

    /// Delete a cache entry from the trie.
    private func _delete(key: RootKey, tokens: [Int]) {
        guard let root = cache[key] else { return }

        var path = [root]
        for tok in tokens {
            guard let next = path.last!.children[Int32(tok)] else { return }
            path.append(next)
        }

        guard let entry = path.last?.cache else { return }
        _nBytes -= entry.nbytes
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

    /// Deep-copy a KV cache by reading and writing its state.
    private func _deepCopy(_ promptCache: [KVCache]) -> [KVCache] {
        Self.materializedCopy(promptCache, tokenCount: nil)
    }

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

    /// Refresh LRU recency for the given entry (move to most-recently-used).
    private func _touch(key: RootKey, tokens: [Int]) {
        let entry = _get(key: key, tokens: tokens)
        entry.lastUsed = Date().timeIntervalSince1970
        lru.remove(key: key, tokens: tokens)
        lru.push(key: key, tokens: tokens)
    }

    /// Internal fetch without locking.
    private func _fetchNearestCache(key: RootKey, tokens: [Int]) -> FetchResult {
        let result = _search(key: key, tokens: tokens)

        // Exact match
        if let exact = result.exact {
            let entry = _get(key: result.key, tokens: exact)
            _touch(key: result.key, tokens: exact)
            return FetchResult(
                cache: _deepCopy(entry.promptCache),
                remainder: [],
                hitKind: .exact,
                matchedTokenCount: tokens.count
            )
        }

        let shortLength = result.shorter?.count ?? 0

        // Longer prefix: if the cached entry is longer than the query and trimmable
        if let longer = result.longer, result.commonPrefix > shortLength {
            let entry = _get(key: result.key, tokens: longer)
            if canTrimPromptCache(entry.promptCache) {
                let prefix = min(tokens.count, result.commonPrefix)
                let numToTrim = longer.count - prefix
                if Self.canSafelyTrim(entry.promptCache, numTokens: numToTrim) {
                    let copy = _deepCopy(entry.promptCache)
                    trimPromptCache(copy, numTokens: numToTrim)
                    let remainder = prefix < tokens.count ? Array(tokens[prefix...]) : []
                    _touch(key: result.key, tokens: longer)
                    return FetchResult(
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
            let entry = _get(key: result.key, tokens: result.shorter!)
            _touch(key: result.key, tokens: result.shorter!)
            return FetchResult(
                cache: _deepCopy(entry.promptCache),
                remainder: Array(tokens[shortLength...]),
                hitKind: .shorter,
                matchedTokenCount: shortLength
            )
        }

        // No match
        return FetchResult(cache: nil, remainder: tokens, hitKind: .none, matchedTokenCount: 0)
    }

    private static func canSafelyTrim(_ cache: [KVCache], numTokens: Int) -> Bool {
        guard numTokens > 0 else { return true }
        let minCachedSeqLen = cache.map { layer -> Int in
            guard let first = layer.state.first, first.ndim >= 3 else { return 0 }
            return first.dim(2)
        }.min() ?? 0
        return numTokens < minCachedSeqLen
    }

    /// Internal insert without locking.
    private func _insertCache(
        key: RootKey, tokens: [Int], promptCache: [KVCache], checkpoint: Bool
    ) {
        let isTrimmable = canTrimPromptCache(promptCache)

        if cache[key] == nil {
            cache[key] = TrieNode()
        }
        var current = cache[key]!

        for i in 0 ..< tokens.count {
            let tok = Int32(tokens[i])
            if current.children[tok] == nil {
                current.children[tok] = TrieNode()
            }
            // If inserting a trimmable cache and we pass through an existing cached node,
            // remove it (the new longer cache supersedes the shorter one).
            if isTrimmable, current.cache != nil {
                _nBytes -= current.cache!.nbytes
                current.cache = nil
                lru.remove(key: key, tokens: Array(tokens[..<i]))
            }
            current = current.children[tok]!
        }

        let cacheBytes = promptCache.reduce(0) { $0 + $1.state.reduce(0) { $0 + $1.nbytes } }
        if current.cache != nil {
            // Update existing entry: remove from LRU and reinsert
            _nBytes -= current.cache!.nbytes
            lru.remove(key: key, tokens: tokens)
            current.cache = CacheEntry(promptCache: promptCache, nbytes: cacheBytes)
            _nBytes += cacheBytes
        } else {
            current.cache = CacheEntry(promptCache: promptCache, nbytes: cacheBytes)
            _nBytes += cacheBytes
        }

        lru.push(key: key, tokens: tokens, checkpoint: checkpoint)

        // Evict if over maxSize
        if lru.count > maxSize {
            if let evicted = lru.pop() {
                _delete(key: evicted.key, tokens: evicted.tokens)
            }
        }

        // Evict if over maxBytes
        while _nBytes > maxBytes {
            guard let evicted = lru.pop() else { break }
            _delete(key: evicted.key, tokens: evicted.tokens)
        }
    }
}
