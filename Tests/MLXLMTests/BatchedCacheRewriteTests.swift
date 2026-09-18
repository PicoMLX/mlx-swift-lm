// Copyright © 2026 Apple Inc.

import MLX
import Testing

@testable import MLXLMCommon

@Suite struct BatchedCacheRewriteTests {
    @Test(arguments: [false, true])
    func rewrittenChildrenDriveBatchOperations(useLeafRewrite: Bool) throws {
        let original = BatchKVCache(leftPadding: [0, 0])
        let initial = MLXArray.zeros([2, 1, 3, 1])
        _ = original.update(keys: initial, values: initial)
        let nested = BatchedCacheList(caches: [original])
        let root = BatchedCacheList(caches: [nested])
        let replacement = BatchKVCache(leftPadding: [0, 0])
        let keys = MLXArray([Float(1), 2, 3, 11, 12, 13], [2, 1, 3, 1])
        _ = replacement.update(keys: keys, values: keys + 100)
        if useLeafRewrite {
            root.rewriteLeaves(path: []) { _ in replacement }
        } else {
            root.mapChildren { _ in replacement }
        }

        let rewritten = try #require(nested[0] as? BatchKVCache)
        #expect(rewritten === replacement)
        root.filterBatched(batchIndices: MLXArray([Int32(1)]))
        #expect(replacement.batchSize == 1)
        #expect(original.batchSize == 2)
        let extracted = try #require(root.extractBatched(0) as? CacheList)
        let leaf = try #require(extracted[0] as? CacheList)[0]
        #expect(leaf.state[0].asArray(Float.self) == [11, 12, 13])
        #expect(leaf.state[1].asArray(Float.self) == [111, 112, 113])

        let copied = try #require(root.copy() as? BatchedCacheList)
        let copiedRow = try #require(copied.extractBatched(0) as? CacheList)
        let copiedLeaf = try #require(copiedRow[0] as? CacheList)[0]
        #expect(copiedLeaf.state[0].asArray(Float.self) == [11, 12, 13])
        #expect(copiedLeaf.state[1].asArray(Float.self) == [111, 112, 113])
        root.extendBatched(copied)
        #expect(replacement.batchSize == 2)
        #expect(replacement.state[0].asArray(Float.self) == [11, 12, 13, 11, 12, 13])
    }
}
