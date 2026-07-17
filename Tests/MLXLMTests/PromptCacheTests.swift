// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

// MARK: - Helpers

private func makeSimpleCache(
    seqLen: Int,
    heads: Int = 2,
    headDim: Int = 4,
    value: Float = 1.0
) -> KVCacheSimple {
    let cache = KVCacheSimple()
    if seqLen > 0 {
        let keys = MLXArray.ones([1, heads, seqLen, headDim]) * value
        let values = MLXArray.ones([1, heads, seqLen, headDim]) * (value + 1)
        _ = cache.update(keys: keys, values: values)
    }
    return cache
}

private func makeSimplePromptCache(
    layers: Int = 2,
    seqLen: Int,
    value: Float = 1.0
) -> [KVCache] {
    (0 ..< layers).map { layer in
        makeSimpleCache(seqLen: seqLen, value: value + Float(layer))
    }
}

private func makeRotatingCache(
    seqLen: Int,
    maxSize: Int,
    heads: Int = 2,
    headDim: Int = 4,
    value: Float = 1.0
) -> RotatingKVCache {
    let cache = RotatingKVCache(maxSize: maxSize, keep: 4)
    if seqLen > 0 {
        let keys = MLXArray.ones([1, heads, seqLen, headDim]) * value
        let values = MLXArray.ones([1, heads, seqLen, headDim]) * (value + 1)
        _ = cache.update(keys: keys, values: values)
    }
    return cache
}

// MARK: - PromptCacheTests

@Suite("Prompt Cache", .serialized)
struct PromptCacheTests {

    @Test("Empty cache returns nil and the full remainder")
    func emptyCacheReturnsNil() {
        let cache = LRUPromptCache(maxSize: 10)
        let (result, remainder) = cache.fetchNearestCache(model: "model", tokens: [1, 2, 3])

        #expect(result == nil)
        #expect(remainder == [1, 2, 3])
    }

    @Test("Nearest-cache lookup prefers the longest prefix and trims longer entries")
    func nearestCacheLookupHandlesShorterAndLongerMatches() throws {
        let cache = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "model",
            tokens: [1, 2],
            promptCache: makeSimplePromptCache(seqLen: 2)
        )
        cache.insertCache(
            model: "model",
            tokens: [1, 2, 3, 4],
            promptCache: makeSimplePromptCache(seqLen: 4, value: 2)
        )

        let (shorterMatch, shorterRemainder) = cache.fetchNearestCache(
            model: "model",
            tokens: [1, 2, 3, 4, 5]
        )
        let shorterLayer = try #require(shorterMatch?.first)
        #expect(shorterRemainder == [5])
        #expect(shorterLayer.offset == 4)

        let (longerMatch, longerRemainder) = cache.fetchNearestCache(
            model: "model",
            tokens: [1, 2, 3]
        )
        let longerLayer = try #require(longerMatch?.first)
        #expect(longerRemainder.isEmpty)
        #expect(longerLayer.offset == 3)
    }

    @Test("Divergent queries return the shortest cached descendant of the shared prefix")
    func divergentQueryReturnsShortestCachedDescendant() throws {
        let cache = LRUPromptCache(maxSize: 10)
        let base = [1, 2, 3, 4, 5, 6, 7, 8]

        // A long stored chain (cheap 1-layer caches) and a much shorter entry
        // sharing the same 8-token prefix.
        let longTokens = base + Array(100 ..< 2100)
        cache.insertCache(
            model: "model",
            tokens: longTokens,
            promptCache: makeSimplePromptCache(layers: 1, seqLen: longTokens.count, value: 1)
        )
        let shortTokens = base + [9000, 9001]
        cache.insertCache(
            model: "model",
            tokens: shortTokens,
            promptCache: makeSimplePromptCache(layers: 1, seqLen: shortTokens.count, value: 5)
        )

        // The query shares the 8-token prefix and then diverges from both
        // stored continuations. The lookup must descend from the divergence
        // node and pick the SHORTEST cached descendant (the 10-token entry),
        // trimmed back to the shared prefix.
        let result = cache.fetchNearestCacheResult(model: "model", tokens: base + [7777])
        #expect(result.hitKind == .longer)
        #expect(result.matchedTokenCount == base.count)
        #expect(result.remainder == [7777])

        let layer = try #require(result.cache?.first)
        #expect(layer.offset == base.count)
        // The short entry was stored with keys == 5.0, the long chain with
        // keys == 1.0: seeing 5.0 proves the shallowest descendant won.
        let keys = try #require(layer.state.first)
        #expect(keys[0, 0, 0, 0].item(Float.self) == 5.0)
    }

    @Test("LRU eviction honors recency")
    func lruEvictionHonorsRecency() {
        let cache = LRUPromptCache(maxSize: 2)
        cache.insertCache(
            model: "model", tokens: [1], promptCache: makeSimplePromptCache(seqLen: 1))
        cache.insertCache(
            model: "model", tokens: [2], promptCache: makeSimplePromptCache(seqLen: 1))

        _ = cache.fetchNearestCache(model: "model", tokens: [1])
        cache.insertCache(
            model: "model", tokens: [3], promptCache: makeSimplePromptCache(seqLen: 1))

        #expect(cache.fetchNearestCache(model: "model", tokens: [1]).0 != nil)
        #expect(cache.fetchNearestCache(model: "model", tokens: [2]).0 == nil)
        #expect(cache.fetchNearestCache(model: "model", tokens: [3]).0 != nil)
    }

    @Test("Fetch returns deep copies and model names stay isolated")
    func fetchReturnsDeepCopiesAndHonorsModelIsolation() throws {
        let cache = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "model-a",
            tokens: [1, 2, 3],
            promptCache: makeSimplePromptCache(seqLen: 3)
        )

        let (copyA, _) = cache.fetchNearestCache(model: "model-a", tokens: [1, 2, 3])
        let (copyB, _) = cache.fetchNearestCache(model: "model-a", tokens: [1, 2, 3])
        let (otherModel, remainder) = cache.fetchNearestCache(model: "model-b", tokens: [1, 2, 3])

        let first = try #require(copyA?.first)
        let second = try #require(copyB?.first)
        first.trim(1)

        #expect(first.offset != second.offset)
        #expect(otherModel == nil)
        #expect(remainder == [1, 2, 3])
    }

    @Test("Fetches are isolated by salt")
    func fetchesAreIsolatedBySalt() throws {
        let cache = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "model",
            tokens: [1, 2, 3],
            promptCache: makeSimplePromptCache(seqLen: 3),
            salt: 1
        )

        let saltOne = cache.fetchNearestCacheResult(model: "model", tokens: [1, 2, 3], salt: 1)
        let saltTwo = cache.fetchNearestCacheResult(model: "model", tokens: [1, 2, 3], salt: 2)

        _ = try #require(saltOne.cache?.first)
        #expect(saltOne.hitKind == .exact)
        #expect(saltOne.remainder.isEmpty)
        #expect(saltTwo.cache == nil)
        #expect(saltTwo.hitKind == .none)
        #expect(saltTwo.remainder == [1, 2, 3])
    }

    @Test("Inserts under the same model but different salts do not collide")
    func insertsUnderDifferentSaltsDoNotCollide() throws {
        let cache = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "model",
            tokens: [1, 2, 3],
            promptCache: makeSimplePromptCache(seqLen: 3),
            salt: 7
        )
        cache.insertCache(
            model: "model",
            tokens: [1, 2, 3, 4, 5],
            promptCache: makeSimplePromptCache(seqLen: 5, value: 2),
            salt: 9
        )

        // Both salt namespaces retain their own entry.
        #expect(cache.count == 2)

        let sevenHit = cache.fetchNearestCacheResult(model: "model", tokens: [1, 2, 3], salt: 7)
        #expect(sevenHit.hitKind == .exact)
        #expect(sevenHit.remainder.isEmpty)

        let nineHit = cache.fetchNearestCacheResult(
            model: "model", tokens: [1, 2, 3, 4, 5], salt: 9)
        #expect(nineHit.hitKind == .exact)
        #expect(nineHit.remainder.isEmpty)

        // Salt 9's longer prompt is not visible to salt 7.
        let crossHit = cache.fetchNearestCacheResult(
            model: "model", tokens: [1, 2, 3, 4, 5], salt: 7)
        #expect(crossHit.hitKind == .shorter)
        #expect(crossHit.remainder == [4, 5])
    }

    @Test("Inserted snapshots are immune to later mutation of the source cache")
    func insertedSnapshotsAreDetachedFromSourceCache() throws {
        let cache = LRUPromptCache(maxSize: 10)
        let source = makeSimplePromptCache(seqLen: 4)

        cache.insertCache(model: "model", tokens: [1, 2, 3, 4], promptCache: source)
        source.first?.trim(3)

        let restored = cache.fetchNearestCacheResult(model: "model", tokens: [1, 2, 3, 4])
        let layer = try #require(restored.cache?.first)
        #expect(restored.hitKind == .exact)
        #expect(layer.offset == 4)
    }

    @Test("Longer-prefix rotating-cache trim preflight bails safely")
    func longerPrefixRotatingTrimPreflightBailsSafely() {
        let cache = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "model",
            tokens: [1, 2, 3, 4, 5, 6, 7, 8],
            promptCache: [makeRotatingCache(seqLen: 4, maxSize: 4)]
        )

        let result = cache.fetchNearestCacheResult(model: "model", tokens: [1])
        #expect(result.cache == nil)
        #expect(result.hitKind == .none)
        #expect(result.remainder == [1])
    }

    @Test("trimTo(nSequences:) evicts least-recently-used entries")
    func trimToSequencesEvictsLeastRecentlyUsed() {
        let cache = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "model", tokens: [1], promptCache: makeSimplePromptCache(seqLen: 1))
        cache.insertCache(
            model: "model", tokens: [2], promptCache: makeSimplePromptCache(seqLen: 1))
        cache.insertCache(
            model: "model", tokens: [3], promptCache: makeSimplePromptCache(seqLen: 1))
        #expect(cache.count == 3)

        // Touch [1] so it becomes most-recently-used.
        _ = cache.fetchNearestCache(model: "model", tokens: [1])
        cache.trimTo(nSequences: 1)

        #expect(cache.count == 1)
        #expect(cache.fetchNearestCache(model: "model", tokens: [1]).0 != nil)
        #expect(cache.fetchNearestCache(model: "model", tokens: [2]).0 == nil)
        #expect(cache.fetchNearestCache(model: "model", tokens: [3]).0 == nil)
    }

    @Test("trimTo(nBytes:) shrinks the cache to fit a byte budget")
    func trimToBytesShrinksCache() {
        let cache = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "model", tokens: [1], promptCache: makeSimplePromptCache(seqLen: 4))
        cache.insertCache(
            model: "model", tokens: [2], promptCache: makeSimplePromptCache(seqLen: 4))
        let perEntry = cache.nbytes / 2
        #expect(cache.count == 2)
        #expect(cache.nbytes > 0)

        // Keep only enough budget for a single entry.
        cache.trimTo(nBytes: perEntry)
        #expect(cache.count == 1)
        #expect(cache.nbytes <= perEntry)
    }

    @Test("isCacheCompatible accepts simple/rotating and rejects empty")
    func isCacheCompatibleClassifiesLeaves() {
        #expect(LRUPromptCache.isCacheCompatible([makeSimpleCache(seqLen: 1)]))
        #expect(LRUPromptCache.isCacheCompatible([makeRotatingCache(seqLen: 1, maxSize: 8)]))
        #expect(LRUPromptCache.isCacheCompatible([]) == false)
    }


    // MARK: - Protocol conformance

    @Test("LRUPromptCache satisfies the PromptCaching protocol surface")
    func conformsToPromptCaching() throws {
        let cache: any PromptCaching = LRUPromptCache(maxSize: 10)
        cache.insertCache(
            model: "model",
            tokens: [1, 2, 3],
            promptCache: makeSimplePromptCache(seqLen: 3),
            checkpoint: false,
            salt: 0
        )

        let result: PromptCacheFetchResult = cache.fetchNearestCacheResult(
            model: "model",
            tokens: [1, 2, 3],
            salt: 0
        )
        #expect(result.hitKind == .exact)
        #expect(result.reusedTokenCount == 3)
        _ = try #require(result.cache?.first)

        cache.trim(nSequences: 0, nBytes: nil)
        let afterTrim = cache.fetchNearestCacheResult(model: "model", tokens: [1, 2, 3], salt: 0)
        #expect(afterTrim.hitKind == .none)
    }

    @Test("topologySalt separates cache topologies and is stable")
    func topologySaltSeparatesTopologies() {
        // Same base salt, different topologies -> different keys, so entries
        // created under different maxKVSize windows can never collide.
        let base: UInt64 = 7
        let small = LRUPromptCache.topologySalt(
            base: base, caches: [RotatingKVCache(maxSize: 8)])
        let large = LRUPromptCache.topologySalt(
            base: base, caches: [RotatingKVCache(maxSize: 16)])
        let simple = LRUPromptCache.topologySalt(base: base, caches: [KVCacheSimple()])
        let smallAgain = LRUPromptCache.topologySalt(
            base: base, caches: [RotatingKVCache(maxSize: 8)])

        #expect(small == smallAgain)
        #expect(small != large)
        #expect(small != simple)
        #expect(large != simple)
        // The user-supplied base still participates in the key.
        #expect(
            small != LRUPromptCache.topologySalt(base: 8, caches: [RotatingKVCache(maxSize: 8)]))
        // Same window, different keep-prefix -> different keys (metaState[0]).
        #expect(
            small
                != LRUPromptCache.topologySalt(
                    base: base, caches: [RotatingKVCache(maxSize: 8, keep: 4)]))
    }
}
