// Copyright © 2026 Apple Inc.

import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct CacheListOffsetTests {
    @Test("Media admission distinguishes attention positions from recurrent and positional state")
    func mediaPrefillTopology() throws {
        try Device.withDefaultDevice(.cpu) {
            let attention = KVCacheSimple()
            let values = MLXArray.ones([1, 2, 5, 4])
            _ = attention.update(keys: values, values: values)
            let recurrent = MambaCache()
            recurrent[0] = MLXArray.zeros([1, 3, 8])
            recurrent[1] = MLXArray.zeros([1, 2, 4, 4])
            let metadata = ArraysCache(size: 1)
            metadata[0] = MLXArray([Int32(3)]).reshaped(1, 1)
            let caches: [any KVCache] = [recurrent, CacheList(attention, metadata)]
            #expect(areBatchedMediaCachesPrepared(caches, tokenCount: 5))
            #expect(!areBatchedMediaCachesPrepared(caches, tokenCount: 4))
            #expect(!areBatchedMediaCachesPrepared([recurrent], tokenCount: 5))
            metadata[0] = nil
            #expect(!areBatchedMediaCachesPrepared(caches, tokenCount: 5))
        }
    }

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
            let rows = (0 ..< 2).map { _ in CacheList(KVCacheSimple(), ArraysCache(size: 1)) }
            let cache = try #require(try mergeCaches(rows) as? CacheList)
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
    @Test("Prepared nested media caches adopt without sharing source state")
    func preparedNestedAdoption() throws {
        try Device.withDefaultDevice(.cpu) {
            func source(length: Int, mode: Int32) -> CacheList {
                let attention = KVCacheSimple()
                let positions = MLXArray(0 ..< length).asType(.float32).reshaped(1, 1, length, 1)
                _ = attention.update(keys: positions, values: positions + 100)
                let metadata = ArraysCache(size: 1)
                metadata[0] = MLXArray([mode], [1, 1])
                let recurrent = MambaCache()
                recurrent[0] = MLXArray([mode], [1, 1])
                recurrent[1] = MLXArray([mode + 1], [1, 1])
                return CacheList(CacheList(attention, metadata), recurrent)
            }
            let original = source(length: 5, mode: 9)
            let added = source(length: 2, mode: 17)
            let originalKeys = original[0].state[0] + 0
            let batch = try #require(makeBatchedCache(from: original) as? CacheList)
            let nested = try #require(batch[0] as? CacheList)
            #expect(nested[0] is BatchKVCache)
            #expect(batch[1] is MambaCache)
            #expect(batch.offset == 5)
            batch.extend(other: try #require(makeBatchedCache(from: added)))
            batch.filter(batchIndices: MLXArray([Int32(1), 0]))
            let row = try #require(batch.extract(0) as? CacheList)
            let rowOwner = try #require(row[0] as? CacheList)
            #expect(row.offset == 2)
            #expect((rowOwner[1] as? ArraysCache)?[0]?.asArray(Int32.self) == [17])
            #expect((row[1] as? MambaCache)?[0]?.asArray(Int32.self) == [17])
            (nested[1] as? ArraysCache)?[0] = MLXArray([Int32(88), 99], [2, 1])
            (batch[1] as? MambaCache)?[0] = MLXArray([Int32(66), 77], [2, 1])
            let key = MLXArray.full([2, 1, 1, 1], values: MLXArray(Float(200)))
            _ = nested[0].update(keys: key, values: key)
            let originalOwner = try #require(original[0] as? CacheList)
            #expect(originalOwner[0].offset == 5)
            #expect(abs(originalOwner[0].state[0] - originalKeys).max().item(Float.self) == 0)
            #expect((originalOwner[1] as? ArraysCache)?[0]?.asArray(Int32.self) == [9])
            #expect((original[1] as? MambaCache)?[0]?.asArray(Int32.self) == [9])
            #expect((added[1] as? MambaCache)?[0]?.asArray(Int32.self) == [17])
        }
    }

    @Test("Prepared cache adoption rejects unsupported subclasses atomically")
    func preparedAdoptionRejectsUnknownTopologies() {
        let source = KVCacheSimple()
        let values = MLXArray.ones([1, 1, 3, 1])
        _ = source.update(keys: values, values: values)
        #expect(makeBatchedCache(from: UnknownPreparedSimple()) == nil)
        #expect(makeBatchedCache(from: UnknownPreparedArrays(size: 1)) == nil)
        #expect(makeBatchedCache(from: UnknownPreparedComposite(source)) == nil)
        #expect(makeBatchedCache(from: CacheList(source, UnknownPreparedArrays(size: 1))) == nil)
        #expect(makeBatchedCache(from: RotatingKVCache(maxSize: 4, keep: 1)) == nil)
        #expect(makeBatchedCache(from: RotatingKVCache(maxSize: 4, rewindCapacity: 2)) == nil)
        #expect(makeBatchedCaches(from: [source, UnknownPreparedSimple()]) == nil)
        #expect(source.offset == 3)
        #expect(source.state[0].shape == [1, 1, 3, 1])
    }

    @Test("Prepared rotating caches retain their logical position during adoption")
    func preparedRotatingAdoption() throws {
        let source = RotatingKVCache(maxSize: 4)
        for position in 0 ..< 7 {
            let values = MLXArray.full([1, 1, 1, 1], values: MLXArray(Float(position)))
            _ = source.update(keys: values, values: values)
        }
        let batch = try #require(makeBatchedCache(from: source) as? BatchRotatingKVCache)
        let extracted = try #require(batch.extract(0) as? RotatingKVCache)
        #expect(extracted.offset == source.offset)
        #expect(extracted.offset == 7)
        #expect(source.offset == 7)
    }

}

private final class UnknownPreparedSimple: KVCacheSimple {}
private final class UnknownPreparedArrays: ArraysCache {}
private final class UnknownPreparedComposite: CacheList {}
