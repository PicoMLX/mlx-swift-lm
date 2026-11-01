// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import Testing

@Suite(.serialized)
struct CacheListOffsetTests {
    @Test("Copying and restoring a media cache preserves its admission position")
    func copyAndRestore() throws {
        try Device.withDefaultDevice(.cpu) {
            let attention = KVCacheSimple()
            let values = MLXArray.ones([1, 2, 7, 4])
            _ = attention.update(keys: values, values: values)
            let metadata = ArraysCache(size: 1)
            metadata[0] = MLXArray([Int32(3)]).reshaped(1, 1)
            let cache = CacheList(attention, metadata)
            #expect(cache.offset == 7)
            let copy = try #require(cache.copy() as? CacheList)
            #expect(copy.offset == 7)
            #expect(copy[0].offset == 7)
            let restored = try CacheList.restore(state: cache.state, metaState: cache.metaState)
            #expect(restored.offset == 7)
        }
    }

    @Test("Extracting a composite batch reports the selected row's cache position")
    func extractedRowPosition() throws {
        try Device.withDefaultDevice(.cpu) {
            let factories = try makeBatchedCacheFactories(for: [
                CacheList(KVCacheSimple(), ArraysCache(size: 1))
            ])
            let cache = try #require(factories[0]([0, 0]) as? BatchedCacheList)
            cache.prepare(lengths: [2, 5])
            _ = cache[0].update(
                keys: MLXArray.ones([2, 2, 5, 4]), values: MLXArray.ones([2, 2, 5, 4]))
            (cache[1] as? ArraysCache)?[0] = MLXArray([Int32(11), 22]).reshaped(2, 1)
            cache.finalize()
            let row = try #require(cache.extract(0) as? CacheList)
            #expect(row.offset == row[0].offset)
            #expect(row.offset == 2)
        }
    }
}
