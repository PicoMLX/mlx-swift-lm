// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

// MARK: - BatchKVCache Tests

@Suite("BatchKVCache")
struct BatchKVCacheTests {

    // MARK: Helpers

    private func makeSimpleCache(nHeads: Int = 8, headDim: Int = 64, seqLen: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        if seqLen > 0 {
            let k = MLXArray.ones([1, nHeads, seqLen, headDim])
            let v = MLXArray.ones([1, nHeads, seqLen, headDim]) * 2
            _ = cache.update(keys: k, values: v)
            eval(k, v)
        }
        return cache
    }

    // MARK: - Basic update

    @Test("BatchKVCache update grows cache correctly")
    func testBatchUpdate() {
        let batchSize = 2
        let nHeads = 4
        let headDim = 32
        let seqLen = 10

        let cache = BatchKVCache(batchSize: batchSize, leftPadding: [0, 0])

        let k = MLXArray.ones([batchSize, nHeads, seqLen, headDim])
        let v = MLXArray.ones([batchSize, nHeads, seqLen, headDim]) * 2
        let (retK, retV) = cache.update(keys: k, values: v)

        eval(retK, retV)
        #expect(cache.offset == seqLen)
        #expect(retK.shape == [batchSize, nHeads, seqLen, headDim])
        #expect(retV.shape == [batchSize, nHeads, seqLen, headDim])
    }

    @Test("BatchKVCache decode step appends single token")
    func testBatchDecodeAppend() {
        let batchSize = 2
        let nHeads = 4
        let headDim = 32

        let cache = BatchKVCache(batchSize: batchSize, leftPadding: [0, 0])

        // Simulate prefill (10 tokens)
        let prefillK = MLXArray.ones([batchSize, nHeads, 10, headDim])
        let prefillV = MLXArray.ones([batchSize, nHeads, 10, headDim]) * 2
        _ = cache.update(keys: prefillK, values: prefillV)
        eval(prefillK, prefillV)

        #expect(cache.offset == 10)

        // Simulate decode (1 token)
        let decodeK = MLXArray.ones([batchSize, nHeads, 1, headDim]) * 3
        let decodeV = MLXArray.ones([batchSize, nHeads, 1, headDim]) * 4
        let (retK, retV) = cache.update(keys: decodeK, values: decodeV)
        eval(retK, retV)

        #expect(cache.offset == 11)
        #expect(retK.shape == [batchSize, nHeads, 11, headDim])
    }

    // MARK: - fromSingle

    @Test("BatchKVCache.fromSingle creates correct batch from equal-length caches")
    func testFromSingleEqualLength() {
        let nHeads = 4
        let headDim = 32
        let seqLen = 20
        let nLayers = 3

        // Create 2 per-sequence sets of caches
        let seqACaches = (0 ..< nLayers).map { _ in makeSimpleCache(nHeads: nHeads, headDim: headDim, seqLen: seqLen) }
        let seqBCaches = (0 ..< nLayers).map { _ in makeSimpleCache(nHeads: nHeads, headDim: headDim, seqLen: seqLen) }

        let batchCaches = BatchKVCache.fromSingle(perSequenceCaches: [seqACaches, seqBCaches])

        #expect(batchCaches.count == nLayers)
        for layer in batchCaches {
            eval(layer.state)
            #expect(layer.batchSize == 2)
            #expect(layer.offset == seqLen)
            #expect(layer.leftPadding == [0, 0])
            // state[0] = keys: [B, nHeads, seqLen, headDim]
            #expect(layer.state.first?.shape == [2, nHeads, seqLen, headDim])
        }
    }

    @Test("BatchKVCache.fromSingle left-pads shorter sequence")
    func testFromSingleWithPadding() {
        let nHeads = 4
        let headDim = 32
        let nLayers = 2

        // Sequence A: 20 tokens, Sequence B: 12 tokens
        let seqACaches = (0 ..< nLayers).map { _ in makeSimpleCache(nHeads: nHeads, headDim: headDim, seqLen: 20) }
        let seqBCaches = (0 ..< nLayers).map { _ in makeSimpleCache(nHeads: nHeads, headDim: headDim, seqLen: 12) }

        let batchCaches = BatchKVCache.fromSingle(perSequenceCaches: [seqACaches, seqBCaches])

        for layer in batchCaches {
            eval(layer.state)
            #expect(layer.batchSize == 2)
            #expect(layer.offset == 20)            // Aligned to max (sequence A)
            #expect(layer.leftPadding == [0, 8])   // B needs 8 tokens of left padding
            // state[0] = keys: [B, nHeads, maxOffset, headDim]
            #expect(layer.state.first?.shape == [2, nHeads, 20, headDim])
        }
    }

    // MARK: - filter

    @Test("BatchKVCache.filter removes completed sequences")
    func testFilter() {
        let nHeads = 4
        let headDim = 32
        let nLayers = 2

        let seqACaches = (0 ..< nLayers).map { _ in makeSimpleCache(nHeads: nHeads, headDim: headDim, seqLen: 10) }
        let seqBCaches = (0 ..< nLayers).map { _ in makeSimpleCache(nHeads: nHeads, headDim: headDim, seqLen: 10) }
        let seqCCaches = (0 ..< nLayers).map { _ in makeSimpleCache(nHeads: nHeads, headDim: headDim, seqLen: 10) }

        let batchCaches = BatchKVCache.fromSingle(perSequenceCaches: [seqACaches, seqBCaches, seqCCaches])

        // Remove sequence B (index 1), keep A (0) and C (2)
        let filtered = batchCaches[0].filter(keepIndices: [0, 2])
        eval(filtered.state)

        #expect(filtered.batchSize == 2)
        #expect(filtered.offset == 10)
        #expect(filtered.state.first?.shape == [2, nHeads, 10, headDim])
    }

    // MARK: - extract

    @Test("BatchKVCache.extract returns single-sequence cache")
    func testExtract() {
        let nHeads = 4
        let headDim = 32

        let cache = BatchKVCache(batchSize: 2, leftPadding: [0, 5])
        let k = MLXArray.ones([2, nHeads, 20, headDim])
        let v = MLXArray.ones([2, nHeads, 20, headDim]) * 2
        _ = cache.update(keys: k, values: v)
        eval(k, v)

        // Extract sequence 1 (has 5 left-padding, so 15 real tokens)
        let extracted = cache.extract(index: 1)
        eval(extracted.state)

        #expect(extracted.offset == 15)
        if extracted.state.count >= 2 {
            #expect(extracted.state[0].shape == [1, nHeads, 15, headDim])
        }
    }

    // MARK: - makeMask

    @Test("BatchKVCache.makeMask returns .none for decode with no padding")
    func testMakeMaskDecodeNoPadding() {
        let cache = BatchKVCache(batchSize: 2, leftPadding: [0, 0])
        // Simulate some prefill
        let k = MLXArray.ones([2, 4, 10, 32])
        let v = MLXArray.ones([2, 4, 10, 32])
        _ = cache.update(keys: k, values: v)
        eval(k, v)

        let maskMode = cache.makeMask(n: 1, windowSize: nil, returnArray: false)
        switch maskMode {
        case .none:
            break  // Expected
        default:
            Issue.record("Expected .none mask for decode with no padding")
        }
    }

    @Test("BatchKVCache.makeMask returns array mask for decode with left-padding")
    func testMakeMaskDecodeWithPadding() {
        let cache = BatchKVCache(batchSize: 2, leftPadding: [0, 5])
        let k = MLXArray.ones([2, 4, 20, 32])
        let v = MLXArray.ones([2, 4, 20, 32])
        _ = cache.update(keys: k, values: v)
        eval(k, v)

        let maskMode = cache.makeMask(n: 1, windowSize: nil, returnArray: false)
        switch maskMode {
        case .array(let mask):
            eval(mask)
            // Mask should be [B, 1, n, total] = [2, 1, 1, 21]
            #expect(mask.shape == [2, 1, 1, 21])
        default:
            Issue.record("Expected .array mask for decode with left-padding")
        }
    }
}

// MARK: - DeviceEngine Tests

@Suite("DeviceEngine")
struct DeviceEngineTests {

    @Test("DeviceEngine.run executes synchronous work")
    func testRunSync() async throws {
        let result = try await DeviceEngine.shared.run {
            42
        }
        #expect(result == 42)
    }

    @Test("DeviceEngine.run propagates errors")
    func testRunThrows() async {
        do {
            try await DeviceEngine.shared.run {
                throw CancellationError()
            }
            Issue.record("Expected error to be thrown")
        } catch {
            // Expected
        }
    }
}

// MARK: - InferenceScheduler Config Tests

@Suite("InferenceScheduler.Config")
struct InferenceSchedulerConfigTests {

    @Test("Default config has expected values")
    func testDefaultConfig() {
        let config = InferenceScheduler.Config()
        #expect(config.completionBatchSize == 4)
        #expect(config.prefillBatchSize == 4)
        #expect(config.maxQueueSize == 32)
    }

    @Test("Custom config values are preserved")
    func testCustomConfig() {
        let config = InferenceScheduler.Config(
            completionBatchSize: 8,
            prefillBatchSize: 2,
            maxQueueSize: 16
        )
        #expect(config.completionBatchSize == 8)
        #expect(config.prefillBatchSize == 2)
        #expect(config.maxQueueSize == 16)
    }
}
