// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite(
    "Batch KV Cache",
    .enabled(if: MLXMetalGuard.isAvailable, MLXMetalGuard.unavailableComment),
    .serialized
)
struct BatchKVCacheCoverageTests {

    @Test("Lifecycle covers update, filter, extend, and extract")
    func lifecycleRoundTrip() {
        // Use zero-padding here so this test exercises lifecycle bookkeeping,
        // not left-padding normalization. Padding behavior is covered elsewhere.
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
        let difference = maxAbsDifference(restoredKeys, originalKeys)
        #expect(difference == 0)
    }

    @Test("makeMask honours left padding during decode")
    func makeMaskUsesLeftPaddingDuringDecode() {
        let cache = BatchKVCache(leftPadding: [1, 3, 0])
        let mode = cache.makeMask(n: 2, windowSize: nil, returnArray: false)

        switch mode {
        case .array(let mask):
            #expect(mask.dim(0) == 3)
            #expect(mask.dim(2) == 2)
            #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
            #expect(mask[1, 0, 0, 0].item(Bool.self) == false)
        case .arrays, .causal, .none:
            Issue.record("BatchKVCache should produce an explicit array mask for this path")
        }
    }
}

@Suite(
    "Batch Rotating KV Cache",
    .enabled(if: MLXMetalGuard.isAvailable, MLXMetalGuard.unavailableComment),
    .serialized
)
struct BatchRotatingKVCacheCoverageTests {

    @Test("Lifecycle covers update, filter, extend, and extract")
    func lifecycleRoundTrip() {
        // Use zero-padding here so this test exercises lifecycle bookkeeping,
        // not left-padding normalization. Padding behavior is covered elsewhere.
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 0], keep: 2)
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)

        cache.filter(batchIndices: [0])
        #expect(cache.batchSize == 1)

        let extensionCache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0], keep: 2)
        let extensionKV = makeKV(batchSize: 1, heads: 2, seqLen: 2, headDim: 4, value: 5)
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
        #expect(extractedFirst.keep == 2)
        #expect(extractedSecond.keep == 2)
    }

    @Test("Overflow keeps the sliding window and preserve keep")
    func overflowPreservesKeepAndWindow() throws {
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [0], keep: 2)
        let first = makeKV(batchSize: 1, heads: 2, seqLen: 4, headDim: 4, value: 1)
        _ = cache.update(keys: first.0, values: first.1)

        let second = makeKV(batchSize: 1, heads: 2, seqLen: 1, headDim: 4, value: 9)
        _ = cache.update(keys: second.0, values: second.1)

        let extracted = cache.extract(idx: 0)
        let keys = try #require(extracted.state.first)
        #expect(extracted.keep == 2)
        #expect(extracted.maxSize == 4)
        #expect(keys.dim(2) <= 4)
    }

    @Test("prepare/finalize preserve extractable state")
    func prepareFinalizePreserveExtractableState() {
        let cache = BatchRotatingKVCache(maxSize: 32, leftPadding: [2, 0], keep: 4)
        cache.prepare(lengths: [3, 5], rightPadding: [2, 0])

        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 5, headDim: 4, value: 2)
        _ = cache.update(keys: kv.0, values: kv.1)
        #expect(cache._lengths != nil)

        cache.finalize()
        #expect(cache._lengths == nil)
        #expect(cache.keep == 4)
    }
}

@Suite(
    "Batch Masking And Position",
    .enabled(if: MLXMetalGuard.isAvailable, MLXMetalGuard.unavailableComment),
    .serialized
)
struct BatchMaskingAndPositionCoverageTests {

    @Test("createCausalMask masks left padding")
    func causalMaskHonoursLeftPadding() {
        let leftPadding = MLXArray([Int32(1), Int32(2)])
        let mask = createCausalMask(n: 4, offset: 0, leftPadding: leftPadding)

        #expect(mask.dim(0) == 2)
        #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[0, 0, 1, 1].item(Bool.self) == true)
        #expect(mask[1, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[1, 0, 0, 1].item(Bool.self) == false)
        #expect(mask[1, 0, 2, 2].item(Bool.self) == true)
    }

    @Test("batch compatibility rejects unsupported cache types")
    func batchCompatibilityMatrix() {
        #expect(isBatchCompatible([]))
        #expect(isBatchCompatible([KVCacheSimple(), RotatingKVCache(maxSize: 16)]))
        #expect(isBatchCompatible([QuantizedKVCache()]) == false)
        #expect(isBatchCompatible([MambaCache()]) == false)
        #expect(isBatchCompatible([CacheList(KVCacheSimple(), KVCacheSimple())]) == false)
    }

    @Test("applyRotaryPosition uses per-sequence offsets for batch caches")
    func rotaryPositionUsesPerSequenceOffsets() {
        let rope = RoPE(dimensions: 8)
        let input = MLXArray.ones([2, 4, 3, 8])

        let batchCache = BatchRotatingKVCache(maxSize: 16, leftPadding: [2, 0], keep: 4)
        let kv = makeKV(batchSize: 2, heads: 4, seqLen: 3, headDim: 8)
        _ = batchCache.update(keys: kv.0, values: kv.1)

        let result = applyRotaryPosition(rope, to: input, cache: batchCache)
        let expected = rope(input, offset: batchCache.batchOffset)

        #expect(result.shape == expected.shape)
        #expect(maxAbsDifference(result, expected) == 0)
    }
}

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
