// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
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

    @Test("Filter during a ragged prefill keeps transient right-padding consistent")
    func filterDuringRaggedPrefillFiltersRightPadding() {
        let cache = BatchKVCache(leftPadding: [0, 0])
        cache.prepare(rightPadding: MLXArray([Int32(2), Int32(0)]))
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 5, headDim: 4, value: 2)
        _ = cache.update(keys: kv.0, values: kv.1)

        // Row 0 is cancelled mid-prefill; finalize must apply only the
        // surviving row's (zero) right padding, not the removed row's 2.
        cache.filter(batchIndices: [1])
        cache.finalize()

        let extracted = cache.extract(idx: 0)
        #expect(extracted.offset == 5)
        #expect(cache.leftPadding[0].item(Int32.self) == 0)
    }

    @Test("finalize preserves valid tokens when capacity exceeds the populated prefix")
    func finalizePreservesValidTokensWithSpareCapacity() throws {
        // The buffer allocates in 256-position steps, so after a 5-token
        // update the capacity far exceeds `_idx`. finalize's whole-buffer
        // circular roll must still land each row's valid data at
        // `[pad ..< _idx]` (see the invariant comment on `finalize()`).
        let cache = BatchKVCache(leftPadding: [0, 0])
        cache.prepare(rightPadding: MLXArray([Int32(2), Int32(0)]))

        // Row 0: 3 valid positions then 2 right-pad slots; row 1: 5 valid.
        var kVals = [Float](repeating: 0, count: 2 * 5)
        for row in 0 ..< 2 {
            let valid = row == 0 ? 3 : 5
            for pos in 0 ..< valid {
                kVals[row * 5 + pos] = Float(10 * (row + 1) + pos)
            }
        }
        let keys = MLXArray(kVals, [2, 1, 5, 1])
        _ = cache.update(keys: keys, values: keys * 2)

        cache.finalize()

        let row0 = cache.extract(idx: 0)
        #expect(row0.offset == 3)
        #expect(try #require(row0.keys).asArray(Float.self) == [10, 11, 12])

        let row1 = cache.extract(idx: 1)
        #expect(row1.offset == 5)
        #expect(try #require(row1.keys).asArray(Float.self) == [20, 21, 22, 23, 24])
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

    @Test("ropeOffset dispatches to .batch through the KVCache type")
    func ropeOffsetDispatchesThroughKVCache() {
        // Models read `cache?.ropeOffset` with `cache` typed as `KVCache?`, so the
        // batched override must win via witness-table dispatch (not the scalar
        // KVCache extension default).
        let full: any KVCache = BatchKVCache(leftPadding: [1, 0])
        guard case .batch = full.ropeOffset else {
            Issue.record("BatchKVCache.ropeOffset via KVCache should be .batch")
            return
        }
        let rotating: any KVCache = BatchRotatingKVCache(maxSize: 16, leftPadding: [1, 0], keep: 0)
        guard case .batch = rotating.ropeOffset else {
            Issue.record("BatchRotatingKVCache.ropeOffset via KVCache should be .batch")
            return
        }
    }
}

// MARK: - BatchRotatingKVCache (keep > 0)

@Suite(.serialized)
struct BatchRotatingKVCacheCoverageTests {

    @Test("Lifecycle covers update, filter, extend, and extract with keep > 0")
    func lifecycleRoundTrip() {
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

        let extractedFirst = cache.extract(idx: 0)
        let extractedSecond = cache.extract(idx: 1)
        #expect(extractedFirst.offset == 3)
        #expect(extractedSecond.offset == 2)
        // `keep` is exposed on the batched cache (RotatingKVCache.keep is private).
        #expect(cache.keep == 2)
    }

    @Test("Overflow keeps the sliding window and preserves keep")
    func overflowPreservesKeepAndWindow() throws {
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [0], keep: 2)
        let first = makeKV(batchSize: 1, heads: 2, seqLen: 4, headDim: 4, value: 1)
        _ = cache.update(keys: first.0, values: first.1)

        let second = makeKV(batchSize: 1, heads: 2, seqLen: 1, headDim: 4, value: 9)
        _ = cache.update(keys: second.0, values: second.1)

        let extracted = cache.extract(idx: 0)
        let keys = try #require(extracted.state.first)
        #expect(cache.keep == 2)
        #expect(extracted.maxSize == 4)
        #expect(keys.dim(2) <= 4)
    }

    @Test("isTrimmable(after:) predicts window overflow")
    func isTrimmableAfterPredictsOverflow() {
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0], keep: 0)
        let kv = makeKV(batchSize: 1, heads: 2, seqLen: 4, headDim: 4, value: 1)
        _ = cache.update(keys: kv.0, values: kv.1)

        // offset 4, window 8: trimmable now and through 3 more positions,
        // but not once the window would be full — matching RotatingKVCache.
        #expect(cache.isTrimmable)
        #expect(cache.isTrimmable(after: 3))
        #expect(cache.isTrimmable(after: 4) == false)
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

    @Test("fromSingle/toSingle keep the retained window, not the oldest one")
    func fromSingleRoundTripKeepsRetainedWindow() {
        // Per-position values, so returning the oldest window instead of the
        // retained one is visible rather than hidden behind uniform data.
        func positionalKV(seqLen: Int) -> (MLXArray, MLXArray) {
            let positions = MLXArray((0 ..< seqLen).map { Float($0 + 1) })
                .reshaped([1, 1, seqLen, 1])
            let keys = MLXArray.ones([1, 2, seqLen, 4]) * positions
            let values = MLXArray.ones([1, 2, seqLen, 4]) * (positions * Float(10))
            return (keys, values)
        }

        // Below the window, and exactly at it: the largest source `fromSingle`
        // accepts, and the one whose `_idx == maxCacheSize` drives the decode
        // path's front-trim arithmetic.
        for seqLen in [5, 8] {
            let single = RotatingKVCache(maxSize: 8, keep: 0)
            let kv = positionalKV(seqLen: seqLen)
            _ = single.update(keys: kv.0, values: kv.1)

            let batch = BatchRotatingKVCache.fromSingle(single)
            #expect(batch.batchSize == 1)

            let restored = batch.toSingle()
            #expect(restored.offset == single.offset)
            #expect(restored.state[0].dim(2) == single.state[0].dim(2))
            #expect(maxAbsDifference(restored.state[0], single.state[0]) == 0)
            #expect(maxAbsDifference(restored.state[1], single.state[1]) == 0)
        }
    }

    @Test("Requested-capacity provenance survives fromSingle and extract")
    func capacityOriginSurvivesBatching() {
        let single = RotatingKVCache(maxSize: 8, keep: 0)
        let kv = makeKV(batchSize: 1, heads: 2, seqLen: 4, headDim: 4, value: 1)
        _ = single.update(keys: kv.0, values: kv.1)

        // metaState's sixth field is the capacity origin. Mark this window as
        // coming from a requested capacity rather than the model architecture.
        var meta = single.metaState
        #expect(meta.count == 6)
        meta[5] = "requested"
        single.metaState = meta

        // A row restored through the legacy five-field format silently reverts
        // to `modelNative`, which exempts it from requested-capacity validation
        // and makes runtime status report the limit as model-defined.
        let restored = BatchRotatingKVCache.fromSingle(single).extract(idx: 0)
        #expect(restored.metaState.count == 6)
        #expect(restored.metaState[5] == "requested")

        // A model-native window keeps its own label.
        let native = RotatingKVCache(maxSize: 8, keep: 0)
        _ = native.update(keys: kv.0, values: kv.1)
        let nativeRestored = BatchRotatingKVCache.fromSingle(native).extract(idx: 0)
        #expect(nativeRestored.metaState[5] == "modelNative")
    }
}

// MARK: - Factory + BatchedCache protocol surface

@Suite(.serialized)
struct BatchedCacheFactoryTests {

    @Test("Factory routes supported cache types")
    func factoryRoutesSupportedTypes() throws {
        let simple = try makeBatchedCacheFactories(for: [KVCacheSimple()])
        #expect(simple[0]([0, 0]) is BatchKVCache)

        let rotating = try makeBatchedCacheFactories(for: [RotatingKVCache(maxSize: 16)])
        #expect(rotating[0]([0]) is BatchRotatingKVCache)

        let arrays = try makeBatchedCacheFactories(for: [ArraysCache(size: 2)])
        let arraysCache = arrays[0]([0])
        #expect(arraysCache is ArraysCache)
        #expect((arraysCache as? ArraysCache)?.slotCount == 2)

        let mamba = try makeBatchedCacheFactories(for: [MambaCache()])
        #expect(mamba[0]([0]) is MambaCache)

        let composite = try makeBatchedCacheFactories(
            for: [CacheList(KVCacheSimple(), RotatingKVCache(maxSize: 16))])
        #expect(composite[0]([0, 0]) is BatchedCacheList)
    }

    @Test("The factory carries requested-capacity provenance onto produced caches")
    func factoryPreservesCapacityOrigin() throws {
        // This is the path the engine builds caches through; `fromSingle` and
        // `merge` are conversion helpers. A cache produced here with the
        // default `modelNative` label would round-trip that wrong label
        // faithfully through `extract`, so covering only the helpers missed it.
        let probe = RotatingKVCache(maxSize: 16, keep: 0)
        var meta = probe.metaState
        meta[5] = "requested"
        probe.metaState = meta

        let factories = try makeBatchedCacheFactories(for: [probe])
        let produced = try #require(factories[0]([0]) as? BatchRotatingKVCache)
        // `extract` only writes metaState for a populated cache, so fill one
        // row before asking what provenance comes back.
        let kv = makeKV(batchSize: 1, heads: 2, seqLen: 3, headDim: 4, value: 1)
        _ = produced.update(keys: kv.0, values: kv.1)
        #expect(produced.extract(idx: 0).metaState[5] == "requested")

        // A model-native probe still yields model-native rows.
        let native = try makeBatchedCacheFactories(for: [RotatingKVCache(maxSize: 16, keep: 0)])
        let nativeProduced = try #require(native[0]([0]) as? BatchRotatingKVCache)
        _ = nativeProduced.update(keys: kv.0, values: kv.1)
        #expect(nativeProduced.extract(idx: 0).metaState[5] == "modelNative")
    }

    @Test("Rotating caches with keep > 0 are rejected (single-stream fallback)")
    func rotatingWithKeepIsRejected() {
        // keep-prefix rotation + per-row left padding cannot be represented by
        // the prefix-only padding mask after the wrap roll; the factory routes
        // these topologies back to single-stream until the mask model supports
        // trailing-garbage exclusion.
        #expect(throws: BatchedCacheError.self) {
            _ = try makeBatchedCacheFactories(for: [RotatingKVCache(maxSize: 16, keep: 4)])
        }
    }

    @Test("Factory rejects quantized and chunked caches")
    func factoryRejectsUnsupportedTypes() {
        #expect(throws: BatchedCacheError.self) {
            _ = try makeBatchedCacheFactories(for: [QuantizedKVCache()])
        }
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

    @Test("advanceBatched leaves recurrent padding metadata to the model")
    func arraysCacheAdvanceIsModelOwned() {
        let mamba = MambaCache(leftPadding: [0, 2])
        mamba[0] = MLXArray.ones([2, 4])
        let cache: any BatchedCache = mamba
        cache.prepareBatched(leftPadding: nil, lengths: [4, 2], rightPadding: nil)

        // What a mask-aware mixer does during the forward pass: Mamba2,
        // FalconH1, GraniteMoeHybrid, LFM2MoE, Qwen3.5 and Qwen3Next all call
        // `advance(chunk)` themselves.
        mamba.advance(2)
        let afterModel = mamba.currentLengths?.asArray(Int32.self)

        // The engine's post-chunk hook must not advance it a second time, or
        // `lengths` runs a full chunk ahead and the next chunk's SSM mask
        // suppresses valid tokens.
        cache.advanceBatched(2)
        #expect(mamba.currentLengths?.asArray(Int32.self) == afterModel)
    }

    @Test("BatchedCacheList preserves nested topology")
    func batchedCacheListNested() throws {
        let factories = try makeBatchedCacheFactories(
            for: [CacheList(KVCacheSimple(), RotatingKVCache(maxSize: 16))])
        let composite = factories[0]([0, 0])
        let list = try #require(composite as? BatchedCacheList)
        // Filtering routes through each child without flattening the topology.
        list.filterBatched(batchIndices: MLXArray([Int32(0)]))
        let extracted = list.extractBatched(0)
        #expect(extracted is CacheList)
    }

    @Test("BatchedCacheList delegates makeMask to its attention child")
    func batchedCacheListDelegatesMask() throws {
        // Hybrid models pass the layer's CacheList itself to
        // createAttentionMask; the composite must surface the batched
        // attention child's left-padding mask, not BaseKVCache's `.none`.
        let factories = try makeBatchedCacheFactories(
            for: [CacheList(KVCacheSimple(), RotatingKVCache(maxSize: 16))])
        let composite = factories[0]([1, 0])

        let mode = composite.makeMask(n: 1, windowSize: nil, returnArray: false)
        switch mode {
        case .array(let mask):
            #expect(mask.dim(0) == 2)
        case .arrays, .causal, .none:
            Issue.record("BatchedCacheList should delegate to the batched attention child")
        }
    }

    @Test("extract preserves the MambaCache subtype")
    func extractPreservesMambaSubtype() {
        let mamba = MambaCache(leftPadding: [0, 0])
        mamba[0] = MLXArray.ones([2, 4])
        mamba[1] = MLXArray.ones([2, 4])

        let extracted = mamba.extract(0)
        #expect(extracted is MambaCache)
        #expect(extracted.slotCount == 2)
    }
}

// MARK: - Model masking

@Suite(.serialized)
struct Gemma2BatchMaskTests {

    @Test("Fully masked left-padding rows stay finite in float16")
    func float16PaddingRowsStayFinite() throws {
        // Converting the old -1e9 mask fill to float16 overflows to -inf; a
        // query position inside a row's left padding is then fully masked and
        // its softmax produces NaN, contaminating later layers. The finite
        // dtype-aware fill keeps those rows finite.
        let json = """
            {
              "hidden_size": 16,
              "num_hidden_layers": 1,
              "intermediate_size": 32,
              "num_attention_heads": 2,
              "head_dim": 8,
              "rms_norm_eps": 1e-6,
              "vocab_size": 32,
              "num_key_value_heads": 1,
              "attn_logit_softcapping": 50.0,
              "final_logit_softcapping": 30.0,
              "query_pre_attn_scalar": 144.0
            }
            """
        let config = try JSONDecoder().decode(
            Gemma2Configuration.self, from: Data(json.utf8))
        let attention = Gemma2Attention(config)
        attention.update(parameters: attention.parameters().mapValues { $0.asType(.float16) })

        // Row 0 is left-padded by 2: its first two query positions can attend
        // only padded keys, i.e. a fully masked score row.
        let cache = BatchKVCache(leftPadding: [2, 0])
        let mask = cache.makeMask(n: 4, windowSize: nil, returnArray: false)
        let x = MLXArray.ones([2, 4, 16]).asType(.float16)

        let out = attention(x, mask: mask, cache: cache)
        eval(out)
        #expect(isNaN(out).any().item(Bool.self) == false)
    }
}

// MARK: - Serialization

@Suite(.serialized)
struct BatchCacheSerializationTests {

    @Test("savePromptCache fails closed for batched caches")
    func savePromptCacheRejectsBatchedCaches() {
        // Batched caches would otherwise serialize under the "KVCache" class
        // name and reload as a single-sequence cache with multi-row state.
        let cache = BatchKVCache(leftPadding: [0, 1])
        let (keys, values) = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("batch-cache-\(UUID().uuidString).safetensors")
        #expect(throws: (any Error).self) {
            try savePromptCache(url: url, cache: [cache])
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test("savePromptCache fails closed for multi-row array caches")
    func savePromptCacheRejectsMultiRowArraysCache() {
        // ArraysCache/MambaCache are batched by batch dimension, not by type:
        // a multi-row instance would serialize under its registered class name
        // and reload batch-sized recurrent state into a single-sequence
        // continuation. Single-row instances remain saveable.
        let batched = MambaCache(leftPadding: [0, 0])
        batched[0] = MLXArray.ones([2, 4])
        batched[1] = MLXArray.ones([2, 4])

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mamba-cache-\(UUID().uuidString).safetensors")
        #expect(throws: (any Error).self) {
            try savePromptCache(url: url, cache: [batched])
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))

        let single = MambaCache()
        single[0] = MLXArray.ones([1, 4])
        single[1] = MLXArray.ones([1, 4])
        let singleURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mamba-cache-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: singleURL) }
        #expect(throws: Never.self) {
            try savePromptCache(url: singleURL, cache: [single])
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

    @Test("Multi-token mask after rotation predicts the linearization trim")
    func multiTokenMaskAfterRotationPredictsTrim() {
        // maxSize 4, row 0 left-padded by 2. Prefilling 4 tokens fills the
        // ring exactly and one decode step wraps it (`rotated == true`,
        // `_idx` back at 0+1) while row 0 still has left padding 1. The next
        // multi-token update linearizes the ring first (temporalOrder resets
        // the temporal length to maxSize) and then trims one position, so
        // its mask must derive the trim from the linearized length: row 0's
        // first valid token ends up at key position 0 and must be attendable.
        // Computing the trim from the circular write pointer instead leaves
        // one stale pad in the row's effective left padding and masks it out.
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [2, 0])
        let (k4, v4) = makeKV(batchSize: 2, heads: 2, seqLen: 4, headDim: 4)
        _ = cache.update(keys: k4, values: v4)
        let (k1, v1) = makeKV(batchSize: 2, heads: 2, seqLen: 1, headDim: 4)
        _ = cache.update(keys: k1, values: v1)

        guard
            case .array(let mask) = cache.makeMask(n: 2, windowSize: nil, returnArray: true)
        else {
            Issue.record("expected an array mask from a batch rotating cache")
            return
        }
        // Key axis spans cappedOffset (3) + n (2) = 5 positions.
        #expect(mask.dim(-1) == 5)
        // Row 0, first query, key position 0: the row's first valid token
        // after the update's linearize-then-trim.
        #expect(mask[0, 0, 0, 0].item(Bool.self) == true)
        // The pre-existing single-token behavior is unchanged: row 1 (no
        // padding) attends its whole window.
        #expect(mask[1, 0, 0, 0].item(Bool.self) == true)
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
