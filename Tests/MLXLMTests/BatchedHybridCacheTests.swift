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

        let cache: any BatchedCache = mamba
        cache.prepare(lengths: [3, 5])
        #expect(mamba.lengths != nil)
        cache.finalize()
        #expect(mamba.lengths == nil)

        cache.filter(batchIndices: batchRows(0))
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
        #expect(cache.leftPadding == nil, "empty rows carry no padding")
        cache.prepare(lengths: [1, 2])
        #expect(cache.batchSize == 2)

        let mask = try #require(cache.makeMask(N: 2))
        #expect(mask.shape == [2, 2])
        #expect(mask.asArray(Bool.self) == [true, false, true, true])
    }

    @Test("A recurrent batch with zero rows stays empty until extended")
    func recurrentZeroRows() throws {
        let empty = MambaCache(leftPadding: [])
        #expect(empty.batchSize == 0)
        #expect(empty.leftPaddingValues == [])
        #expect(empty.lengths == nil)

        let populated = MambaCache()
        populated[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        empty.extend(other: populated as any BatchedCache)
        #expect(empty.batchSize == 2)
        #expect(maxAbsDifference(try #require(empty[0]), try #require(populated[0])) == 0)
    }

    @Test("CacheList.merge batches each child position on its own type")
    func compositeMerge() throws {
        let list = try CacheList.merge([
            CacheList(KVCacheSimple(), MambaCache()), CacheList(KVCacheSimple(), MambaCache()),
        ])
        try #require(list.children.count == 2)
        #expect(list[0] is BatchKVCache)
        #expect(list[1] is MambaCache)
        #expect((list[0] as? BatchKVCache)?.batchSize == 2)
        #expect((list[0] as? BatchKVCache)?.offset == 0)

        #expect(throws: BatchedCacheError.self) {
            _ = try CacheList.merge([CacheList(KVCacheSimple(), QuantizedKVCache())])
        }
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

    @Test("A merged CacheList cannot be saved, a multi-row recurrent cache still can")
    func promptCacheSaving() throws {
        let longer = KVCacheSimple()
        let kv = makePositionKV(positions: 0 ..< 1, heads: 2, headDim: 4)
        _ = longer.update(keys: kv.0, values: kv.1)
        let list = try CacheList.merge([
            CacheList(longer, MambaCache()), CacheList(KVCacheSimple(), MambaCache()),
        ])
        #expect((list[0] as? BatchKVCache)?.leftPadding.asArray(Int32.self) == [0, 1])
        let listURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("batched-list-\(UUID().uuidString).safetensors")
        #expect(throws: (any Error).self) {
            try savePromptCache(url: listURL, cache: [list])
        }
        #expect(!FileManager.default.fileExists(atPath: listURL.path))

        let mamba = MambaCache(leftPadding: [1, 0])
        mamba[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        mamba[1] = MLXArray(8 ..< 16).asType(.float32).reshaped([2, 4])
        mamba.prepare(lengths: [3, 5])
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mamba-cache-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [mamba])
        let restored = try #require(try loadPromptCache(url: url).0.first as? MambaCache)
        #expect(maxAbsDifference(try #require(restored[0]), try #require(mamba[0])) == 0)
        #expect(restored.leftPaddingValues == [1, 0])
        #expect(restored.lengthsValues == [3, 5])
    }

    @Test("A recurrent cache filtered to zero rows cannot be saved, also inside a list")
    func zeroRowsAreRefused() {
        let mamba = MambaCache(leftPadding: [0, 0])
        mamba[0] = MLXArray.ones([2, 4])
        mamba.filter(batchIndices: MLXArray([Int32]()))

        for cache in [mamba as any KVCache, CacheList(KVCacheSimple(), mamba)] {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("zero-rows-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            #expect(throws: (any Error).self) {
                try savePromptCache(url: url, cache: [cache])
            }
            #expect(!FileManager.default.fileExists(atPath: url.path))
        }
    }
}
