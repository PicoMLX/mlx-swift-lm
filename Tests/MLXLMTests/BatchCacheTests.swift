// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

// MARK: - BatchKVCache

@Suite(.serialized)
struct BatchKVCacheCoverageTests {

    @Test("Lifecycle covers update, filter, extend, and extract")
    func lifecycleRoundTrip() {
        // Zero-padding here so this test exercises lifecycle bookkeeping, not
        // left-padding normalization (padding behavior is covered elsewhere).
        let cache = BatchKVCache(leftPadding: [0, 0])
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)

        cache.filter(batchIndices: [0])
        #expect(cache.batchSize == 1)
        #expect(cache.leftPadding.shape == [1])
        #expect(cache.leftPadding[0].item(Int32.self) == 0)

        let extensionCache = BatchKVCache(leftPadding: [0])
        let extensionKV = makeKV(batchSize: 1, heads: 2, seqLen: 2, headDim: 4, value: 4)
        _ = extensionCache.update(keys: extensionKV.0, values: extensionKV.1)
        cache.extend(other: extensionCache)

        #expect(cache.batchSize == 2)
        #expect(cache.leftPadding[0].item(Int32.self) == 0)
        #expect(cache.leftPadding[1].item(Int32.self) == 1)
        #expect(cache.batchOffsets[0].item(Int32.self) == 3)
        #expect(cache.batchOffsets[1].item(Int32.self) == 2)

        let extractedFirst = cache.extract(idx: 0)
        let extractedSecond = cache.extract(idx: 1)
        #expect(extractedFirst.offset == 3)
        #expect(extractedSecond.offset == 2)
    }

    @Test("fromSingle/toSingle preserve cache data")
    func fromSingleRoundTripPreservesData() throws {
        let single = KVCacheSimple()
        let kv = makeKV(batchSize: 1, heads: 2, seqLen: 4, headDim: 4, value: 3)
        _ = single.update(keys: kv.0, values: kv.1)

        let batch = BatchKVCache.fromSingle(single)
        #expect(batch.batchSize == 1)

        let restored = batch.toSingle()
        let restoredKeys = try #require(restored.state.first)
        let originalKeys = try #require(single.state.first)
        #expect(restored.offset == single.offset)
        #expect(restoredKeys.shape == originalKeys.shape)
        #expect(maxAbsDifference(restoredKeys, originalKeys) == 0)
    }

    @Test("makeMask honours left padding during decode")
    func makeMaskUsesLeftPaddingDuringDecode() {
        let cache = BatchKVCache(leftPadding: [1, 3, 0])
        let mode = cache.makeMask(n: 2, windowSize: nil, returnArray: false)

        switch mode {
        case .array(let mask):
            #expect(mask.dim(0) == 3)
            #expect(mask.dim(2) == 2)
            // Row 0/1 left-padded slot 0 is masked out.
            #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
            #expect(mask[1, 0, 0, 0].item(Bool.self) == false)
        case .arrays, .causal, .none:
            Issue.record("BatchKVCache should produce an explicit array mask for this path")
        }
    }

    @Test("trim clamps to the minimum live per-row length")
    func trimClampsToMinimumLiveRow() {
        // Ragged batch: leftPadding [4, 0], then prefill 5 tokens. Row offsets
        // become [1, 5] (live lengths 1 and 5). Trimming by 2 must not drive the
        // shorter row's offset negative; the trim is clamped to the min live row
        // length (1) so no row underflows.
        let cache = BatchKVCache(leftPadding: [4, 0])
        let (keys, values) = makeKV(batchSize: 2, heads: 2, seqLen: 5, headDim: 4)
        _ = cache.update(keys: keys, values: values)
        #expect(cache.batchOffsets[0].item(Int32.self) == 1)
        #expect(cache.batchOffsets[1].item(Int32.self) == 5)

        let trimmed = cache.trim(2)
        #expect(trimmed == 1)
        #expect(cache.offset == 4)
        // No row offset is driven negative.
        #expect(cache.batchOffsets[0].item(Int32.self) == 0)
        #expect(cache.batchOffsets[1].item(Int32.self) == 4)
    }
}

// MARK: - BatchRotatingKVCache (keep > 0)

// MARK: - Factory + BatchedCache protocol surface

@Suite(.serialized)
struct BatchedCacheFactoryTests {

    @Test("Factory rejects SSM (Mamba/ArraysCache) caches")
    func factoryRejectsSSMCaches() {
        // SSM caches are only safe under continuous batching when the model
        // threads createSSMMask into its conv/SSM mixers; several in-repo Mamba
        // users do not (their SSM mask is hard-coded nil), so ragged batches would
        // corrupt recurrent state. The factory cannot see the model from the cache
        // probe, so it conservatively rejects all SSM topologies.
        #expect(throws: BatchedCacheError.self) {
            _ = try makeBatchedCacheFactories(for: [ArraysCache(size: 2)])
        }
        #expect(throws: BatchedCacheError.self) {
            _ = try makeBatchedCacheFactories(for: [MambaCache()])
        }
        // A composite layer cache containing an SSM child (e.g. BaichuanM1's
        // CacheList(MambaCache, KVCacheSimple)) is rejected too, since the child
        // factory throws.
        #expect(throws: BatchedCacheError.self) {
            _ = try makeBatchedCacheFactories(
                for: [CacheList(MambaCache(), KVCacheSimple())])
        }
    }

    @Test("Factory rejects chunked caches")
    func factoryRejectsUnsupportedTypes() {
        #expect(throws: BatchedCacheError.self) {
            _ = try makeBatchedCacheFactories(for: [ChunkedKVCache(chunkSize: 16)])
        }
    }

    @Test("BatchedCache protocol drives a full-attention cache")
    func protocolSurfaceFullAttention() {
        let cache: any BatchedCache = BatchKVCache(leftPadding: [0, 0])
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)

        cache.filterBatched(batchIndices: MLXArray([Int32(0), Int32(1)]))
        let extracted = cache.extractBatched(0)
        #expect(extracted is KVCacheSimple)
        // advanceBatched is a no-op for full attention; just confirm it is callable.
        cache.advanceBatched(1)
    }
}

// MARK: - SSM / composite batched lifecycle

@Suite(.serialized)
struct BatchedSSMCacheTests {

    @Test("ArraysCache conforms to the BatchedCache lifecycle")
    func arraysCacheBatchedLifecycle() {
        let mamba = MambaCache(leftPadding: [0, 0])
        mamba[0] = MLXArray.ones([2, 4])
        mamba[1] = MLXArray.ones([2, 4])

        let cache: any BatchedCache = mamba
        cache.prepareBatched(leftPadding: nil, lengths: [3, 5], rightPadding: nil)
        #expect(mamba.lengths != nil)

        cache.advanceBatched(1)
        cache.finalizeBatched()
        #expect(mamba.lengths == nil)

        cache.filterBatched(batchIndices: MLXArray([Int32(0)]))
        #expect(mamba.batchSize == 1)
    }

    @Test("BatchedCacheList preserves nested topology")
    func batchedCacheListNested() throws {
        let factories = try makeBatchedCacheFactories(
            for: [CacheList(KVCacheSimple(), KVCacheSimple())])
        let composite = factories[0]([0, 0])
        let list = try #require(composite as? BatchedCacheList)
        // Filtering routes through each child without flattening the topology.
        list.filterBatched(batchIndices: MLXArray([Int32(0)]))
        let extracted = list.extractBatched(0)
        #expect(extracted is CacheList)
    }

    @Test("BatchedCacheList delegates masking to its attention child")
    func batchedCacheListDelegatesMask() {
        // A composite layer cache (e.g. BaichuanM1's recurrent + attention pair)
        // must honour the attention child's per-row left padding when a model asks
        // it for an attention mask via createAttentionMask(h:cache: cache?.first).
        // Without the makeMask override the inherited BaseKVCache.makeMask would
        // return .none for n=1 / .causal otherwise, exposing left-padded KV slots.
        let attention = BatchKVCache(leftPadding: [1, 3])
        let (keys, values) = makeKV(batchSize: 2, heads: 2, seqLen: 4, headDim: 4)
        _ = attention.update(keys: keys, values: values)

        let list = BatchedCacheList(caches: [attention])
        // Offset/ropeOffset delegate to the attention child.
        #expect(list.offset == attention.offset)
        guard case .batch = list.ropeOffset else {
            Issue.record("BatchedCacheList.ropeOffset should delegate to the batched child")
            return
        }

        // Even for a single-token decode step the composite must emit an explicit
        // array mask that masks the left padding.
        let mode = list.makeMask(n: 1, windowSize: nil, returnArray: false)
        switch mode {
        case .array(let mask):
            #expect(mask.dim(0) == 2)
            // Row 1 has left padding 3 → its first padded slot is masked out.
            #expect(mask[1, 0, 0, 0].item(Bool.self) == false)
        case .arrays, .causal, .none:
            Issue.record("BatchedCacheList should delegate to the batched array mask")
        }
    }
}

// MARK: - Masking

@Suite(.serialized)
struct BatchMaskingTests {

    @Test("createCausalMask masks left padding per sequence")
    func causalMaskHonoursLeftPadding() {
        let leftPadding = MLXArray([Int32(1), Int32(2)])
        let mask = createCausalMask(n: 4, offset: 0, leftPadding: leftPadding)

        #expect(mask.dim(0) == 2)
        // Row 0: leftPadding 1 → position 0 masked, position 1 attendable on the diagonal.
        #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[0, 0, 1, 1].item(Bool.self) == true)
        // Row 1: leftPadding 2 → positions 0 and 1 masked, position 2 attendable.
        #expect(mask[1, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[1, 0, 0, 1].item(Bool.self) == false)
        #expect(mask[1, 0, 2, 2].item(Bool.self) == true)
    }
}

// MARK: - Helpers

private func makeKV(
    batchSize: Int,
    heads: Int,
    seqLen: Int,
    headDim: Int,
    value: Float = 1.0
) -> (MLXArray, MLXArray) {
    let keys = MLXArray.ones([batchSize, heads, seqLen, headDim]) * value
    let values = MLXArray.ones([batchSize, heads, seqLen, headDim]) * (value + 1)
    return (keys, values)
}

private func makeDistinctKV(
    batchSize: Int,
    heads: Int,
    seqLen: Int,
    headDim: Int
) -> (MLXArray, MLXArray) {
    var keysList = [MLXArray]()
    var valuesList = [MLXArray]()
    for index in 0 ..< batchSize {
        keysList.append(MLXArray.ones([1, heads, seqLen, headDim]) * Float(index + 1))
        valuesList.append(MLXArray.ones([1, heads, seqLen, headDim]) * Float((index + 1) * 10))
    }
    return (concatenated(keysList, axis: 0), concatenated(valuesList, axis: 0))
}

private func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
    abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
}
