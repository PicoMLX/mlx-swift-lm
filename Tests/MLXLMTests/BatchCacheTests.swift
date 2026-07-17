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

    @Test("trimBatched rewinds per-row tails and realigns rows")
    func trimBatchedRewindsPerRowTails() {
        // Two equal-length rows; rewind row 0 by 2 and row 1 by 0 (the
        // speculative-rollback shape: rejected draft counts differ per row).
        let cache = BatchKVCache(leftPadding: [0, 0])
        let (keys, values) = makeKV(batchSize: 2, heads: 2, seqLen: 5, headDim: 4)
        _ = cache.update(keys: keys, values: values)

        let trimmed = cache.trimBatched(perRow: [2, 0])
        #expect(trimmed == [2, 0])
        // Shared write position drops by the SMALLEST trim (0 here); row 0 is
        // rolled right by 2 and its padding grows so its retained tail stays
        // aligned at the common end.
        #expect(cache.batchOffsets.asArray(Int32.self) == [3, 5])
        #expect(cache.leftPadding.asArray(Int32.self) == [2, 0])

        // Clamping: requests beyond a row's live length trim only what exists.
        let overTrim = cache.trimBatched(perRow: [10, 1])
        #expect(overTrim == [3, 1])

        // The protocol default (non-rollback-capable caches) trims nothing.
        let mamba: any BatchedCache = MambaCache()
        #expect(mamba.trimBatched(perRow: [1]) == [0])
    }

    @Test("trim clamps to the minimum live per-row offset")
    func trimClampsToMinimumLiveRow() {
        // Ragged batch: leftPadding [4, 0], prefill 5 tokens → row offsets [1, 5].
        // Trimming by 2 must clamp to the min live row offset (1) so the shorter
        // row's batchOffsets is not driven negative and _idx stays >= its padding.
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [4, 0], keep: 0)
        let (keys, values) = makeKV(batchSize: 2, heads: 2, seqLen: 5, headDim: 4)
        _ = cache.update(keys: keys, values: values)
        #expect(cache.batchOffsets[0].item(Int32.self) == 1)
        #expect(cache.batchOffsets[1].item(Int32.self) == 5)

        let trimmed = cache.trim(2)
        #expect(trimmed == 1)
        #expect(cache.batchOffsets[0].item(Int32.self) == 0)
        #expect(cache.batchOffsets[1].item(Int32.self) == 4)
    }

    @Test("Rotated multi-token prefill mask uses the logical window span")
    func rotatedPrefillMaskUsesLogicalSpan() {
        // Drive the cache past its window so it wraps (rotated == true), which
        // resets the circular `_idx` to a small value (here `keep == 0`). A later
        // multi-token (prefill) mask must derive its trim from the LOGICAL window
        // length, not the ring index. With maxSize 4 and leftPadding [2]:
        //   prefill 4 → _idx 4, _scalarOffset 4
        //   decode 1 → wrap: rotated, _idx 1, _scalarOffset 5, leftPadding 1
        // makeMask(n: 2) then computes trimSize from the logical span (min(5,4)=4
        // → trimSize 1), reducing effectiveLeftPadding from 1 to 0 so the first
        // retained key column (col 0) stays attendable for the first query row.
        // The buggy ring-index path computed trimSize = _idx(1) - 4 + 1 = -2,
        // skipped the reduction, and masked column 0 out.
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [2], keep: 0)
        let prefill = makeKV(batchSize: 1, heads: 2, seqLen: 4, headDim: 4, value: 1)
        _ = cache.update(keys: prefill.0, values: prefill.1)
        let decode = makeKV(batchSize: 1, heads: 2, seqLen: 1, headDim: 4, value: 2)
        _ = cache.update(keys: decode.0, values: decode.1)

        let mode = cache.makeMask(n: 2, windowSize: nil, returnArray: false)
        switch mode {
        case .array(let mask):
            // First query row attends to retained key column 0 (true). Under the
            // ring-index bug this would have been masked out (false).
            #expect(mask[0, 0, 0, 0].item(Bool.self) == true)
        case .arrays, .causal, .none:
            Issue.record("Rotated multi-token prefill should produce an array mask")
        }
    }

    @Test("Unrotated oversized prefill mask subtracts the full pending trim")
    func unrotatedOversizedPrefillMaskSubtractsPendingTrim() {
        // An UNROTATED prompt WIDER than the window (not yet trimmed) leaves
        // `_idx` at the full physical width with `rotated == false`. The next
        // multi-token prefill's mask must reduce effectiveLeftPadding by the FULL
        // pending physical trim (`_idx - maxCacheSize + 1`), not a window-capped
        // amount, or a heavily left-padded short row masks out every retained key.
        //
        // maxSize 4, leftPadding [10, 0], prefill width 12 → _idx 12,
        // batchOffsets [2, 12], rotated false. makeMask(n: 2): trimSize =
        // 12 - 4 + 1 = 9, so row 0's effective padding is 10 - 9 = 1 (only key
        // column 0 masked). The buggy window-capped span gave trimSize 1 →
        // effective padding 9, masking the whole short row.
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [10, 0], keep: 0)
        let (keys, values) = makeKV(batchSize: 2, heads: 2, seqLen: 12, headDim: 4)
        _ = cache.update(keys: keys, values: values)
        #expect(cache.rotated == false)

        let mode = cache.makeMask(n: 2, windowSize: nil, returnArray: false)
        switch mode {
        case .array(let mask):
            // Padded row 0, first query: key column 0 stays masked (padding 1),
            // but column 1 is a retained valid key and must be attendable. Under
            // the window-capped bug the whole row was masked (column 1 false).
            #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
            #expect(mask[0, 0, 0, 1].item(Bool.self) == true)
        case .arrays, .causal, .none:
            Issue.record("Unrotated oversized prefill should produce an array mask")
        }
    }

    @Test("Oversized unrotated extract preserves linear state indices")
    func oversizedUnrotatedExtractPreservesLinearState() {
        // A single prompt wider than the window, never decoded (unrotated), keeps
        // the full linear temporal state on extract. The serialized RotatingKVCache
        // `idx` must equal the state width so `RotatingKVCache.temporalOrder` treats
        // it as linear (idx == dim) instead of circularly rolling it on the next
        // multi-token update, which would corrupt the prompt.
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [0], keep: 0)
        let (keys, values) = makeKV(batchSize: 1, heads: 2, seqLen: 6, headDim: 4)
        _ = cache.update(keys: keys, values: values)
        #expect(cache.rotated == false)

        let extracted = cache.extract(idx: 0)
        #expect(extracted.offset == 6)
        let stateWidth = extracted.state.first?.dim(2) ?? -1
        #expect(stateWidth == 6)
        // metaState layout is [keep, maxCacheSize, step, offset, idx]; idx must
        // equal the linear state width so the state is not treated as a ring.
        let idx = Int(extracted.metaState[4]) ?? -1
        #expect(idx == stateWidth)
    }

    @Test("offset reports the absolute processed count past the window")
    func offsetStaysAbsolutePastWindow() {
        // Single-stream RotatingKVCache.offset is BaseKVCache's monotonic
        // counter; models consume it as an absolute position (Gemma3n's
        // cachePosition, Mistral3's attention scaling). A window-capped value
        // would freeze those computations after the first wrap.
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [0], keep: 0)
        let prefill = makeKV(batchSize: 1, heads: 2, seqLen: 4, headDim: 4, value: 1)
        _ = cache.update(keys: prefill.0, values: prefill.1)
        #expect(cache.offset == 4)

        let decode = makeKV(batchSize: 1, heads: 2, seqLen: 1, headDim: 4, value: 2)
        _ = cache.update(keys: decode.0, values: decode.1)
        #expect(cache.offset == 5)
    }

    @Test("Empty-cache state round-trips through get/set")
    func emptyStateRoundTrips() {
        // A fresh (or filter-emptied) cache has no keys/values; its state is
        // the 2-element row-metadata form, which the setter must accept
        // instead of trapping -- mirroring BatchKVCache.
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [1, 2], keep: 0)
        let snapshot = cache.state
        #expect(snapshot.count == 2)

        let restored = BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 0], keep: 0)
        restored.state = snapshot
        #expect(restored.isEmpty)
        #expect(restored.leftPadding.asArray(Int32.self) == [1, 2])
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

// MARK: - Factory + BatchedCache protocol surface

@Suite(.serialized)
struct BatchedCacheFactoryTests {

    @Test("Factory routes supported cache types")
    func factoryRoutesSupportedTypes() throws {
        let simple = try makeBatchedCacheFactories(for: [KVCacheSimple()])
        #expect(simple[0]([0, 0]) is BatchKVCache)

        let rotating = try makeBatchedCacheFactories(for: [RotatingKVCache(maxSize: 16)])
        #expect(rotating[0]([0]) is BatchRotatingKVCache)

        let composite = try makeBatchedCacheFactories(
            for: [CacheList(KVCacheSimple(), RotatingKVCache(maxSize: 16))])
        #expect(composite[0]([0, 0]) is BatchedCacheList)
    }

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
            for: [CacheList(KVCacheSimple(), RotatingKVCache(maxSize: 16))])
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
