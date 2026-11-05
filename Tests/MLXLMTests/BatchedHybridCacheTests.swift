// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct BatchedHybridCacheTests {

    @Test("A recurrent cache prepares, finalizes and filters its rows")
    func recurrentLifecycle() throws {
        let mamba = MambaCache(leftPadding: [0, 0])
        mamba[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        mamba[1] = MLXArray(8 ..< 16).asType(.float32).reshaped([2, 4])

        mamba.prepare(lengths: [3, 5])
        #expect(mamba.lengths != nil)
        mamba.finalize()
        #expect(mamba.lengths == nil)

        mamba.filter(batchIndices: batchRows(0))
        #expect(mamba.batchSize == 1)
        #expect(
            maxAbsDifference(
                try #require(mamba[0]), MLXArray(0 ..< 4).asType(.float32).reshaped([1, 4])) == 0)
        #expect(
            maxAbsDifference(
                try #require(mamba[1]), MLXArray(8 ..< 12).asType(.float32).reshaped([1, 4])) == 0)
    }

    @Test("A merged empty recurrent batch masks by the prepared lengths")
    func mergedRecurrentMasksByLengths() throws {
        let cache = try #require(ArraysCache.merge([MambaCache(), MambaCache()]) as? MambaCache)
        #expect(
            cache.leftPadding?.asArray(Int32.self) == [0, 0], "every empty row has zero padding")
        #expect(cache.batchSize == 2)
        cache.prepare(lengths: [1, 2])
        #expect(cache.batchSize == 2)

        let mask = try #require(cache.makeMask(N: 2))
        #expect(mask.shape == [2, 2])
        #expect(mask.asArray(Bool.self) == [true, false, true, true])
    }

    @Test("merge keeps every empty recurrent row ahead of a populated one")
    func mergeKeepsEmptyRecurrentRows() throws {
        let populated = MambaCache()
        populated[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        populated[1] = MLXArray(8 ..< 16).asType(.float32).reshaped([2, 4])

        let merged = try #require(
            ArraysCache.merge([MambaCache(), MambaCache(), populated]) as? MambaCache)
        #expect(merged.batchSize == 4)
        let first = try #require(merged[0])
        #expect(first.shape == [4, 4])
        #expect(first[0 ..< 2].sum().item(Float.self) == 0, "empty rows hold zero state")
        #expect(maxAbsDifference(first[2 ..< 4], try #require(populated[0])) == 0)
        #expect(maxAbsDifference(try #require(merged[1])[2 ..< 4], try #require(populated[1])) == 0)

        // A hybrid list merged the same way keeps its children's rows aligned:
        // two empty sequences and one with recurrent state make three rows.
        let one = MambaCache()
        one[0] = MLXArray(20 ..< 24).asType(.float32).reshaped([1, 4])
        let list = try CacheList.merge([
            CacheList(KVCacheSimple(), MambaCache()), CacheList(KVCacheSimple(), MambaCache()),
            CacheList(KVCacheSimple(), one),
        ])
        #expect((list[0] as? BatchKVCache)?.batchSize == 3)
        #expect((list[1] as? MambaCache)?.batchSize == 3)
    }

    @Test("A recurrent batch with zero rows stays empty until extended")
    func recurrentZeroRows() throws {
        let empty = MambaCache(leftPadding: [])
        #expect(empty.batchSize == 0)
        #expect(empty.leftPaddingValues == [])
        #expect(empty.lengths == nil)

        let populated = MambaCache()
        populated[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        empty.extend(other: populated)
        #expect(empty.batchSize == 2)
        #expect(maxAbsDifference(try #require(empty[0]), try #require(populated[0])) == 0)
    }

    @Test("extract returns the requested row and keeps the MambaCache type")
    func extractKeepsMambaCache() throws {
        let mamba = MambaCache(leftPadding: [0, 0])
        mamba[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        mamba[1] = MLXArray(8 ..< 16).asType(.float32).reshaped([2, 4])

        for row in 0 ..< 2 {
            let extracted = try #require(mamba.extract(row) as? MambaCache)
            #expect(
                maxAbsDifference(
                    try #require(extracted[0]),
                    MLXArray((row * 4) ..< (row * 4 + 4)).asType(.float32).reshaped([1, 4]))
                    == 0)
            #expect(
                maxAbsDifference(
                    try #require(extracted[1]),
                    MLXArray((8 + row * 4) ..< (12 + row * 4)).asType(.float32).reshaped([1, 4]))
                    == 0)
        }
    }

    @Test("CacheList.merge batches each child position on its own type")
    func compositeMerge() throws {
        let list = try CacheList.merge([
            CacheList(KVCacheSimple(), MambaCache(), RotatingKVCache(maxSize: 16)),
            CacheList(KVCacheSimple(), MambaCache(), RotatingKVCache(maxSize: 16)),
        ])
        try #require(list.children.count == 3)
        #expect(list[0] is BatchKVCache)
        #expect(list[1] is MambaCache)
        #expect(list[2] is BatchRotatingKVCache)
        #expect((list[0] as? BatchKVCache)?.batchSize == 2)
        #expect((list[0] as? BatchKVCache)?.offset == 0)
        #expect((list[2] as? BatchRotatingKVCache)?.batchSize == 2)
        #expect((list[2] as? BatchRotatingKVCache)?.maxSize == 16)

        #expect(throws: KVCacheError.self) {
            _ = try CacheList.merge([CacheList(KVCacheSimple(), QuantizedKVCache())])
        }
        #expect(throws: KVCacheError.self) {
            _ = try CacheList.merge([
                CacheList(KVCacheSimple(), MambaCache()), CacheList(KVCacheSimple()),
            ])
        }
        for cache in [
            RotatingKVCache(maxSize: 16, keep: 4), RotatingKVCache(maxSize: 16, rewindCapacity: 4),
        ] {
            #expect(throws: KVCacheError.self) {
                _ = try CacheList.merge([CacheList(KVCacheSimple(), cache)])
            }
        }
    }

    @Test("mergeCaches refuses caches without a batched form and rows of mixed types")
    func mergeCachesRefusals() throws {
        #expect(throws: KVCacheError.self) {
            _ = try mergeCaches([QuantizedKVCache()])
        }
        #expect(throws: KVCacheError.self) {
            _ = try mergeCaches([ChunkedKVCache(chunkSize: 16)])
        }
        #expect(throws: KVCacheError.self) {
            _ = try mergeCaches([KVCacheSimple(), RotatingKVCache(maxSize: 4)])
        }
        #expect(throws: KVCacheError.self) {
            _ = try mergeCaches([])
        }
        #expect(try mergeCaches([KVCacheSimple(), KVCacheSimple()]) is BatchKVCache)
        #expect(try mergeCaches([MambaCache(), MambaCache()]) is MambaCache)
    }

    @Test("A merged CacheList filters, extends and extracts every child")
    func compositeLifecycle() throws {
        func composite() -> CacheList { CacheList(KVCacheSimple(), MambaCache()) }
        let list = try CacheList.merge([composite(), composite()])
        #expect(list.children.count == 2)
        let attention = try #require(list[0] as? BatchKVCache)
        #expect(attention.batchSize == 2)
        let recurrent = try #require(list[1] as? MambaCache)
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = attention.update(keys: keys, values: values)
        recurrent[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])

        list.filter(batchIndices: batchRows(1))
        #expect(attention.batchSize == 1)
        #expect(recurrent.batchSize == 1)

        let added = try CacheList.merge([composite()])
        #expect(added.children.count == 2)
        let addedKV = makePositionKV(positions: 0 ..< 3, heads: 2, headDim: 4)
        _ = try #require(added[0] as? BatchKVCache).update(keys: addedKV.0, values: addedKV.1)
        let addedRecurrent = try #require(added[1] as? MambaCache)
        addedRecurrent[0] = MLXArray(20 ..< 24).asType(.float32).reshaped([1, 4])
        list.extend(other: added)
        #expect(attention.batchSize == 2)
        #expect(recurrent.batchSize == 2)

        let first = try #require(list.extract(0) as? CacheList)
        let firstAttention = try #require(first[0] as? KVCacheSimple)
        let firstKeys = try #require(firstAttention.state.first)
        #expect(maxAbsDifference(firstKeys, MLXArray.ones(firstKeys.shape) * 2) == 0)
        let second = try #require(list.extract(1) as? CacheList)
        let secondAttention = try #require(second[0] as? KVCacheSimple)
        let secondKeys = try #require(secondAttention.state.first)
        #expect(maxAbsDifference(secondKeys, addedKV.0) == 0)
        let secondRecurrent = try #require(second[1] as? MambaCache)
        let secondState = try #require(secondRecurrent[0])
        #expect(
            maxAbsDifference(secondState, MLXArray(20 ..< 24).asType(.float32).reshaped([1, 4]))
                == 0)
    }

    @Test("An all-empty merge keeps its row count through a later merge and extension")
    func allEmptyMergeKeepsRows() throws {
        let empty = try #require(ArraysCache.merge([MambaCache(), MambaCache()]) as? MambaCache)
        let populated = MambaCache()
        populated[0] = MLXArray(0 ..< 4).asType(.float32).reshaped([1, 4])
        let merged = try #require(ArraysCache.merge([empty, populated]) as? MambaCache)
        #expect(merged.batchSize == 3)
        #expect(try #require(merged[0]).shape == [3, 4])

        // An all-empty composite batch takes prefilled rows with its children aligned.
        let list = try CacheList.merge([
            CacheList(KVCacheSimple(), MambaCache()), CacheList(KVCacheSimple(), MambaCache()),
        ])
        let added = try CacheList.merge([CacheList(KVCacheSimple(), MambaCache())])
        let kv = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
        _ = try #require(added[0] as? BatchKVCache).update(keys: kv.0, values: kv.1)
        try #require(added[1] as? MambaCache)[0] = MLXArray(0 ..< 4).asType(.float32).reshaped([
            1, 4,
        ])
        list.extend(other: added)
        #expect((list[0] as? BatchKVCache)?.batchSize == 3)
        #expect((list[1] as? MambaCache)?.batchSize == 3)
        #expect(try #require((list[1] as? MambaCache)?[0]).shape == [3, 4])
    }

    @Test("merge gives an empty input's rows zero state, however many it stands for")
    func mergeKeepsEmptyInputRows() throws {
        let populated = MambaCache()
        populated[0] = MLXArray(0 ..< 4).asType(.float32).reshaped([1, 4])
        let merged = try #require(
            ArraysCache.merge([MambaCache(leftPadding: [0, 0]), populated]) as? MambaCache)
        #expect(merged.batchSize == 3)
        #expect(try #require(merged[0]).shape == [3, 4])
        #expect(merged.leftPadding?.shape == [3])
        #expect(maxAbsDifference(try #require(merged[0])[2 ..< 3], try #require(populated[0])) == 0)

        let none = try #require(
            ArraysCache.merge([MambaCache(leftPadding: []), populated]) as? MambaCache)
        #expect(none.batchSize == 1)
        #expect(try #require(none[0]).shape == [1, 4])
    }

    @Test("Extending an empty composite batch with prefilled rows keeps its children aligned")
    func extendEmptyCompositeRows() throws {
        let list = try CacheList.merge([CacheList(KVCacheSimple(), MambaCache())])
        let added = try CacheList.merge([CacheList(KVCacheSimple(), MambaCache())])
        let kv = makePositionKV(positions: 0 ..< 3, heads: 2, headDim: 4)
        _ = try #require(added[0] as? BatchKVCache).update(keys: kv.0, values: kv.1)
        try #require(added[1] as? MambaCache)[0] = MLXArray(20 ..< 24).asType(.float32).reshaped([
            1, 4,
        ])
        list.extend(other: added)

        let attention = try #require(list[0] as? BatchKVCache)
        let recurrent = try #require(list[1] as? MambaCache)
        #expect(attention.batchSize == 2)
        #expect(recurrent.batchSize == 2)
        #expect(attention.leftPadding.asArray(Int32.self) == [3, 0])
        #expect(try #require(recurrent[0]).shape == [2, 4])
        #expect(try #require(recurrent[0])[0 ..< 1].sum().item(Float.self) == 0)
        list.filter(batchIndices: batchRows(1))
        #expect(attention.batchSize == 1)
        #expect(recurrent.batchSize == 1)
        #expect(
            maxAbsDifference(
                try #require(recurrent[0]), MLXArray(20 ..< 24).asType(.float32).reshaped([1, 4]))
                == 0)
    }

    @Test("A nested CacheList masks with its batched attention descendant")
    func nestedCompositeMask() throws {
        let longer = KVCacheSimple()
        let kv = makePositionKV(positions: 0 ..< 1, heads: 2, headDim: 4)
        _ = longer.update(keys: kv.0, values: kv.1)
        let list = try CacheList.merge([
            CacheList(MambaCache(), CacheList(KVCacheSimple())),
            CacheList(MambaCache(), CacheList(longer)),
        ])
        guard case .array(let mask) = list.makeMask(n: 1, windowSize: nil, returnArray: false)
        else {
            Issue.record("A nested batched CacheList must use its attention descendant's mask")
            return
        }
        #expect(mask.dim(0) == 2)
        #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[1, 0, 0, 0].item(Bool.self) == true)
    }

    @Test("A merged CacheList masks with its attention child's left padding")
    func compositeMask() throws {
        // Row 1 holds one token, row 0 none, so the merge pads row 0 by one.
        let longer = KVCacheSimple()
        let kv = makePositionKV(positions: 0 ..< 1, heads: 2, headDim: 4)
        _ = longer.update(keys: kv.0, values: kv.1)
        let list = try CacheList.merge([
            CacheList(MambaCache(), KVCacheSimple()), CacheList(MambaCache(), longer),
        ])
        #expect((list[1] as? BatchKVCache)?.leftPadding.asArray(Int32.self) == [1, 0])
        guard case .array(let mask) = list.makeMask(n: 1, windowSize: nil, returnArray: false)
        else {
            Issue.record("A batched CacheList must use its attention child's mask")
            return
        }
        #expect(mask.dim(0) == 2)
        #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[1, 0, 0, 0].item(Bool.self) == true)
    }
}
