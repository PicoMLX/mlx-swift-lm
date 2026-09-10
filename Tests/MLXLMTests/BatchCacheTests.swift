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
    func lifecycleRoundTrip() throws {
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
        // Position-stamped: the appended row's assertion below compares against
        // this tensor, so a uniform fill would let an extend that reversed or
        // rotated the appended positions compare equal.
        let extensionKV = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
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

        // The appended row's actual tensors, not just bookkeeping: an extend
        // that fixed offsets and padding while duplicating or zero-filling
        // the appended state would pass every check above and silently attach
        // the wrong history to an admitted request.
        let appendedKeys = try #require(extractedSecond.state.first)
        #expect(maxAbsDifference(appendedKeys, extensionKV.0) == 0)
        let appendedValues = try #require(extractedSecond.state.last)
        #expect(maxAbsDifference(appendedValues, extensionKV.1) == 0)
    }

    @Test("Filter during a ragged prefill keeps transient right-padding consistent")
    func filterDuringRaggedPrefillFiltersRightPadding() throws {
        let cache = BatchKVCache(leftPadding: [0, 0])
        cache.prepare(rightPadding: MLXArray([Int32(2), Int32(0)]))
        // Row-distinct, so filtering the cancelled row's cache state (not just
        // its `_rightPadding` entry) is observable: the survivor must not
        // inherit row 0's attention history.
        let kv = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 5, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)

        // Row 0 is cancelled mid-prefill; finalize must apply only the
        // surviving row's (zero) right padding, not the removed row's 2.
        cache.filter(batchIndices: [1])
        cache.finalize()

        let extracted = cache.extract(idx: 0)
        #expect(extracted.offset == 5)
        #expect(cache.leftPadding[0].item(Int32.self) == 0)
        // makeDistinctKV keys row i with i+1 and values with (i+1)*10.
        let keys = try #require(extracted.state.first)
        #expect(maxAbsDifference(keys, MLXArray.ones(keys.shape) * 2) == 0)
        let values = try #require(extracted.state.last)
        #expect(maxAbsDifference(values, MLXArray.ones(values.shape) * 20) == 0)
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
        // Values too: a roll that lands the keys correctly while leaving the
        // value buffer padded or out of order feeds corrupted values to the
        // next attention step. The input sets values = keys * 2.
        #expect(try #require(row0.values).asArray(Float.self) == [20, 22, 24])

        let row1 = cache.extract(idx: 1)
        #expect(row1.offset == 5)
        #expect(try #require(row1.keys).asArray(Float.self) == [20, 21, 22, 23, 24])
        #expect(try #require(row1.values).asArray(Float.self) == [40, 42, 44, 46, 48])
    }

    @Test("fromSingle/toSingle preserve cache data")
    func fromSingleRoundTripPreservesData() throws {
        let single = KVCacheSimple()
        // Position-stamped: with a uniform fill, a round trip that reversed,
        // rolled, or duplicated a position across the sequence still satisfies
        // `maxAbsDifference == 0` -- in the one test whose whole claim is that
        // the data is preserved.
        let kv = makePositionKV(positions: 0 ..< 4, heads: 2, headDim: 4)
        _ = single.update(keys: kv.0, values: kv.1)

        let batch = BatchKVCache.fromSingle(single)
        #expect(batch.batchSize == 1)

        let restored = batch.toSingle()
        let restoredKeys = try #require(restored.state.first)
        let originalKeys = try #require(single.state.first)
        #expect(restored.offset == single.offset)
        #expect(restoredKeys.shape == originalKeys.shape)
        #expect(maxAbsDifference(restoredKeys, originalKeys) == 0)
        // Values too: a round trip that preserves keys but drops or swaps the
        // value tensor changes every subsequent attention output.
        let restoredValues = try #require(restored.state.last)
        let originalValues = try #require(single.state.last)
        #expect(maxAbsDifference(restoredValues, originalValues) == 0)
    }

    @Test("trim clamps to the shortest row of an unequal batch")
    func trimClampsToShortestRow() throws {
        // leftPadding [2, 0] + a 3-token update leaves logical row lengths
        // [1, 3] under a padded width of 3. A global trim of 2 must clamp to
        // the shorter row's single token, or that row's offset goes negative
        // and its padding exceeds the buffer, making a later extract slice an
        // invalid range.
        let cache = BatchKVCache(leftPadding: [2, 0])
        // Position-stamped rather than a constant fill: `trim` is pure index
        // arithmetic, so a variant that dropped the OLDEST tokens (advancing
        // the padding) instead of the newest (pulling the write head back)
        // lands on exactly the same offsets and satisfies every count-based
        // assertion here. Only the retained content separates the two.
        let kv = makePositionKV(
            positions: 0 ..< 3, heads: 2, headDim: 4, batchSize: 2, rowStride: 10)
        _ = cache.update(keys: kv.0, values: kv.1)

        let trimmed = cache.trim(2)
        #expect(trimmed == 1)
        #expect(cache.batchOffsets.asArray(Int32.self) == [0, 2])
        // The clamped state stays extractable for the short row.
        #expect(cache.extract(idx: 0).offset == 0)
        // The long row keeps its two OLDEST positions: trim drops from the
        // tail. Stamps of [11, 12] would mean the front was dropped instead,
        // and [0, 1] would mean row 0's data leaked into row 1.
        let row1 = cache.extract(idx: 1)
        #expect(row1.offset == 2)
        let row1Keys = try #require(row1.state.first)
        #expect(row1Keys[0, 0, 0..., 0].asArray(Float.self) == [10, 11])
        // Values carry the same stamps offset by 100: a trim that repositioned
        // only the key tensor would satisfy the keys assertion alone.
        let row1Values = try #require(row1.state.last)
        #expect(row1Values[0, 0, 0..., 0].asArray(Float.self) == [110, 111])

        let rotating = BatchRotatingKVCache(maxSize: 16, leftPadding: [2, 0], keep: 0)
        _ = rotating.update(keys: kv.0, values: kv.1)
        #expect(rotating.trim(2) == 1)
        #expect(rotating.batchOffsets.asArray(Int32.self) == [0, 2])
        // Same extractability check the non-rotating variant makes above: a
        // trim that moves the offsets but leaves stale per-row padding makes
        // the emptied shortest row slice an invalid range.
        #expect(rotating.extract(idx: 0).offset == 0)
        // And the same tail-drop check: the rotating trim decrements both the
        // window index and the scalar offset, so a front-dropping variant is
        // equally invisible to the offset assertions above.
        let rotatingRow1 = rotating.extract(idx: 1)
        #expect(rotatingRow1.offset == 2)
        let rotatingRow1Keys = try #require(rotatingRow1.state.first)
        #expect(rotatingRow1Keys[0, 0, 0..., 0].asArray(Float.self) == [10, 11])
        let rotatingRow1Values = try #require(rotatingRow1.state.last)
        #expect(rotatingRow1Values[0, 0, 0..., 0].asArray(Float.self) == [110, 111])
    }

    @Test("makeMask honours left padding during decode")
    func makeMaskUsesLeftPaddingDuringDecode() throws {
        let cache = BatchKVCache(leftPadding: [1, 3, 0])
        let mode = cache.makeMask(n: 2, windowSize: nil, returnArray: false)

        switch mode {
        case .array(let mask):
            // #require, not #expect: a one-row regression would otherwise
            // continue into the row-1 subscripts below and trap in MLX.
            try #require(mask.dim(0) == 3)
            try #require(mask.dim(2) == 2)
            // Row 0/1 left-padded slot 0 is masked out.
            #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
            #expect(mask[1, 0, 0, 0].item(Bool.self) == false)
            // Depth is per-row, not boolean: row 1 (padding 3) keeps slot 1
            // masked while row 0 (padding 1) can attend it — collapsing every
            // positive padding to one would pass the slot-0 checks alone.
            #expect(mask[0, 0, 1, 1].item(Bool.self) == true)
            #expect(mask[1, 0, 1, 1].item(Bool.self) == false)
        case .arrays, .causal, .none:
            Issue.record("BatchKVCache should produce an explicit array mask for this path")
        }
    }

    @Test("ropeOffset dispatches to .batch through the KVCache type")
    func ropeOffsetDispatchesThroughKVCache() {
        // Models read `cache?.ropeOffset` with `cache` typed as `KVCache?`, so the
        // batched override must win via witness-table dispatch (not the scalar
        // KVCache extension default).
        // Unequal, non-trivial offsets: layers position RoPE from the
        // associated values, so a witness returning `.batch` with zeroed,
        // scalarized or misordered offsets shifts token positions even though
        // the dispatch looks correct. leftPadding [1, 0] starts the rows at
        // [-1, 0]; a 3-token update lands them at [2, 3].
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4, value: 1)
        let fullCache = BatchKVCache(leftPadding: [1, 0])
        _ = fullCache.update(keys: kv.0, values: kv.1)
        let full: any KVCache = fullCache
        guard case .batch(let fullOffsets) = full.ropeOffset else {
            Issue.record("BatchKVCache.ropeOffset via KVCache should be .batch")
            return
        }
        #expect(fullOffsets.asArray(Int32.self) == [2, 3])

        let rotatingCache = BatchRotatingKVCache(maxSize: 16, leftPadding: [1, 0], keep: 0)
        _ = rotatingCache.update(keys: kv.0, values: kv.1)
        let rotating: any KVCache = rotatingCache
        guard case .batch(let rotatingOffsets) = rotating.ropeOffset else {
            Issue.record("BatchRotatingKVCache.ropeOffset via KVCache should be .batch")
            return
        }
        #expect(rotatingOffsets.asArray(Int32.self) == [2, 3])
    }
}

// MARK: - BatchRotatingKVCache (keep > 0)

@Suite(.serialized)
struct BatchRotatingKVCacheCoverageTests {

    @Test("Lifecycle covers update, filter, extend, and extract with keep > 0")
    func lifecycleRoundTrip() throws {
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 0], keep: 2)
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)

        cache.filter(batchIndices: [0])
        #expect(cache.batchSize == 1)

        let extensionCache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0], keep: 2)
        // Position-stamped for the same reason as the non-rotating lifecycle
        // test: the appended row is compared against this tensor by content.
        let extensionKV = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
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

        // The appended row's actual tensors: an extend that fixed the
        // bookkeeping while duplicating or zero-filling the appended state
        // would pass every check above and become another request's
        // attention history.
        let appendedKeys = try #require(extractedSecond.state.first)
        #expect(maxAbsDifference(appendedKeys, extensionKV.0) == 0)
        let appendedValues = try #require(extractedSecond.state.last)
        #expect(maxAbsDifference(appendedValues, extensionKV.1) == 0)
    }

    @Test("Overflow keeps the sliding window and preserves keep")
    func overflowPreservesKeepAndWindow() throws {
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [0], keep: 2)
        let first = makePositionKV(positions: 0 ..< 4, heads: 2, headDim: 4)
        _ = cache.update(keys: first.0, values: first.1)

        let second = makePositionKV(positions: 4 ..< 5, heads: 2, headDim: 4)
        _ = cache.update(keys: second.0, values: second.1)

        let extracted = cache.extract(idx: 0)
        let keys = try #require(extracted.state.first)
        #expect(cache.keep == 2)
        #expect(extracted.maxSize == 4)
        #expect(keys.dim(2) <= 4)
        // Each key is stamped with its absolute position, so the retained
        // window is checkable by content: the pinned keep prefix (0, 1)
        // plus the newest suffix (3, 4), with the oldest non-keep position
        // (2) evicted. Compared as a sorted set because the ring's internal
        // layout may rotate.
        let retained = keys[0, 0, 0..., 0].asArray(Float.self).sorted()
        #expect(retained == [0, 1, 3, 4])
        // Values ride the same eviction: a rotate that keeps the right keys
        // but leaves stale or misordered values feeds corrupted values to
        // attention (makePositionKV stamps them at +100).
        let values = try #require(extracted.state.last)
        // Pairing first, then the set: sorting each side independently would
        // accept a cache that retains the right values but permutes them
        // differently from the keys, pairing every key with a wrong value.
        #expect(maxAbsDifference(values, keys + 100) == 0)
        let retainedValues = values[0, 0, 0..., 0].asArray(Float.self).sorted()
        #expect(retainedValues == [100, 101, 103, 104])
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
    func prepareFinalizePreserveExtractableState() throws {
        // Zero construction-time left padding, matching how the engine builds
        // batched caches (`makeBatchedCache` always passes zeros and declares
        // ragged rows through `prepare(rightPadding:)`). Combining a
        // construction-time left padding with a prepare-time right padding
        // describes a row that is padded from both ends while `lengths` claims
        // the full span, and `finalize` then charges the row for both.
        let cache = BatchRotatingKVCache(maxSize: 32, leftPadding: [0, 0], keep: 4)
        cache.prepare(lengths: [3, 5], rightPadding: [2, 0])

        // Position-stamped so post-finalize extraction is checkable by
        // content, and row-strided so the rows are distinguishable: with
        // identical stamps a finalize that swapped or copied one row's state
        // into the other would still produce the expected sequences.
        let kv = makePositionKV(
            positions: 0 ..< 5, heads: 2, headDim: 4, batchSize: 2, rowStride: 10)
        _ = cache.update(keys: kv.0, values: kv.1)
        #expect(cache._lengths != nil)

        cache.finalize()
        #expect(cache._lengths == nil)
        #expect(cache.keep == 4)

        // Row 0 keeps its 3 real tokens (the trailing 2 of the update were
        // right padding), row 1 all 5 — each in order, with the row's own
        // offset, and free of padding.
        let row0 = cache.extract(idx: 0)
        #expect(row0.offset == 3)
        let row0Keys = try #require(row0.state.first)
        #expect(row0Keys[0, 0, 0..., 0].asArray(Float.self) == [0, 1, 2])
        // Values roll with the keys (makePositionKV stamps them at +100).
        let row0Values = try #require(row0.state.last)
        #expect(row0Values[0, 0, 0..., 0].asArray(Float.self) == [100, 101, 102])

        // Row 1 carries its own stamps (+10), so a row swap is visible.
        let row1 = cache.extract(idx: 1)
        #expect(row1.offset == 5)
        let row1Keys = try #require(row1.state.first)
        #expect(row1Keys[0, 0, 0..., 0].asArray(Float.self) == [10, 11, 12, 13, 14])
        let row1Values = try #require(row1.state.last)
        #expect(row1Values[0, 0, 0..., 0].asArray(Float.self) == [110, 111, 112, 113, 114])
    }

    @Test("One-token ragged prefill finalizes before decode")
    func oneTokenRaggedPrefillFinalizesBeforeDecode() throws {
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 0])
        // The longest prefix has one token; the empty row fills that slot with padding.
        cache.prepare(lengths: [1, 0], rightPadding: [0, 1])

        let prefillKeys = MLXArray([Float(10), 99]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: prefillKeys, values: prefillKeys + 100)
        #expect(cache._lengths != nil)

        cache.finalize()
        #expect(cache._lengths == nil)

        let firstPrefill = cache.extract(idx: 0)
        #expect(firstPrefill.offset == 1)
        #expect(try #require(firstPrefill.state.first).asArray(Float.self) == [10])
        #expect(try #require(firstPrefill.state.last).asArray(Float.self) == [110])

        let secondPrefill = cache.extract(idx: 1)
        #expect(secondPrefill.offset == 0)
        #expect(try #require(secondPrefill.state.first).size == 0)
        #expect(try #require(secondPrefill.state.last).size == 0)

        let decodeKeys = MLXArray([Float(20), 30]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: decodeKeys, values: decodeKeys + 100)

        let firstDecoded = cache.extract(idx: 0)
        #expect(firstDecoded.offset == 2)
        #expect(try #require(firstDecoded.state.first).asArray(Float.self) == [10, 20])
        #expect(try #require(firstDecoded.state.last).asArray(Float.self) == [110, 120])

        let secondDecoded = cache.extract(idx: 1)
        #expect(secondDecoded.offset == 1)
        #expect(try #require(secondDecoded.state.first).asArray(Float.self) == [30])
        #expect(try #require(secondDecoded.state.last).asArray(Float.self) == [130])
    }

    @Test("Ragged prefill accepts a trailing one-token chunk")
    func raggedPrefillAcceptsTrailingOneTokenChunk() throws {
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 0])
        cache.prepare(lengths: [2, 3], rightPadding: [1, 0])

        let first = makePositionKV(
            positions: 0 ..< 2, heads: 1, headDim: 1, batchSize: 2, rowStride: 10)
        _ = cache.update(keys: first.0, values: first.1)

        // Row 0 is already complete, so its final chunk entry is padding.
        let finalKeys = MLXArray([Float(99), 12]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: finalKeys, values: finalKeys + 100)
        #expect(cache._lengths != nil)
        cache.finalize()
        #expect(cache._lengths == nil)

        let firstRow = cache.extract(idx: 0)
        #expect(firstRow.offset == 2)
        #expect(try #require(firstRow.state.first).asArray(Float.self) == [0, 1])
        #expect(try #require(firstRow.state.last).asArray(Float.self) == [100, 101])

        let secondRow = cache.extract(idx: 1)
        #expect(secondRow.offset == 3)
        #expect(try #require(secondRow.state.first).asArray(Float.self) == [10, 11, 12])
        #expect(try #require(secondRow.state.last).asArray(Float.self) == [110, 111, 112])
    }

    @Test("fromSingle/toSingle keep the retained window, not the oldest one")
    func fromSingleRoundTripKeepsRetainedWindow() throws {
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
            // #require'd .first/.last, not [0]/[1]: a round trip returning
            // fewer than two state tensors should fail the test, not trap.
            let restoredKeys = try #require(restored.state.first)
            let restoredValues = try #require(restored.state.last)
            let singleKeys = try #require(single.state.first)
            let singleValues = try #require(single.state.last)
            #expect(restoredKeys.dim(2) == singleKeys.dim(2))
            #expect(maxAbsDifference(restoredKeys, singleKeys) == 0)
            #expect(maxAbsDifference(restoredValues, singleValues) == 0)
        }
    }

    @Test("Requested-capacity provenance survives fromSingle and extract")
    func capacityOriginSurvivesBatching() throws {
        let single = RotatingKVCache(maxSize: 8, keep: 0)
        let kv = makeKV(batchSize: 1, heads: 2, seqLen: 4, headDim: 4, value: 1)
        _ = single.update(keys: kv.0, values: kv.1)

        // metaState's sixth field is the capacity origin. Mark this window as
        // coming from a requested capacity rather than the model architecture.
        // #require rather than #expect: on a five-field regression the [5]
        // accesses below would trap and take the rest of the suite with them.
        var meta = single.metaState
        try #require(meta.count == 6)
        meta[5] = "requested"
        single.metaState = meta

        // A row restored through the legacy five-field format silently reverts
        // to `modelNative`, which exempts it from requested-capacity validation
        // and makes runtime status report the limit as model-defined.
        let restored = BatchRotatingKVCache.fromSingle(single).extract(idx: 0)
        try #require(restored.metaState.count == 6)
        #expect(restored.metaState[5] == "requested")

        // A model-native window keeps its own label.
        let native = RotatingKVCache(maxSize: 8, keep: 0)
        _ = native.update(keys: kv.0, values: kv.1)
        let nativeRestored = BatchRotatingKVCache.fromSingle(native).extract(idx: 0)
        let nativeMeta = nativeRestored.metaState
        try #require(nativeMeta.count == 6)
        #expect(nativeMeta[5] == "modelNative")
    }

    @Test("Extracting a row that never prefilled still restores its metadata")
    func extractRestoresMetadataForEmptyRow() throws {
        // A row can be extracted before its first update — early cancellation,
        // or an admitted row that never prefilled. It still has to carry the
        // window, keep prefix and capacity provenance: a default-labelled
        // `modelNative` row is exempted from requested-capacity validation and
        // reports its limit as model-defined.
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 2], keep: 4)
        cache.capacityOrigin = .requested
        #expect(cache.isEmpty)

        for row in 0 ..< 2 {
            let extracted = cache.extract(idx: row)
            let meta = extracted.metaState
            // #require: a shorter representation should fail the test, not
            // trap the suite on the subscripts below.
            try #require(meta.count == 6)
            #expect(meta[0] == "4")  // keep
            #expect(meta[1] == "16")  // maxSize
            #expect(meta[5] == "requested")
            // Nothing was written, so the row starts from zero rather than from
            // its negative pre-prefill batch offset.
            #expect(extracted.offset == 0)
        }
    }
}

// MARK: - Factory + BatchedCache protocol surface

@Suite(.serialized)
struct BatchedCacheFactoryTests {

    @Test("Factory routes supported cache types")
    func factoryRoutesSupportedTypes() throws {
        // #require on .first throughout: a factory array missing an entry
        // should fail the test, not trap the suite on a [0] subscript. The
        // required factory is bound to a local before it is called — invoking
        // the result of a #require expansion directly crashes the Swift 6.3
        // type checker (recordArgumentList assertion).
        let simple = try makeBatchedCacheFactories(for: [KVCacheSimple()])
        let simpleFactory = try #require(simple.first)
        #expect(simpleFactory([0, 0]) is BatchKVCache)

        let rotating = try makeBatchedCacheFactories(for: [RotatingKVCache(maxSize: 16)])
        let rotatingFactory = try #require(rotating.first)
        let producedRotating = try #require(rotatingFactory([0]) as? BatchRotatingKVCache)
        // The window must come from the probe, not a default: a hard-coded
        // maxSize would make the engine retain the wrong number of tokens.
        #expect(producedRotating.maxSize == 16)

        let arrays = try makeBatchedCacheFactories(for: [ArraysCache(size: 2)])
        let arraysFactory = try #require(arrays.first)
        let arraysCache = arraysFactory([0])
        #expect(arraysCache is ArraysCache)
        #expect((arraysCache as? ArraysCache)?.slotCount == 2)

        let mamba = try makeBatchedCacheFactories(for: [MambaCache()])
        let mambaFactory = try #require(mamba.first)
        #expect(mambaFactory([0]) is MambaCache)

        let composite = try makeBatchedCacheFactories(
            for: [CacheList(KVCacheSimple(), RotatingKVCache(maxSize: 16))])
        let compositeFactory = try #require(composite.first)
        #expect(compositeFactory([0, 0]) is BatchedCacheList)
    }

    @Test("The factory carries requested-capacity provenance onto produced caches")
    func factoryPreservesCapacityOrigin() throws {
        // This is the path the engine builds caches through; `fromSingle` and
        // `merge` are conversion helpers. A cache produced here with the
        // default `modelNative` label would round-trip that wrong label
        // faithfully through `extract`, so covering only the helpers missed it.
        let probe = RotatingKVCache(maxSize: 16, keep: 0)
        var meta = probe.metaState
        // #require before every [5] access: a five-field regression should
        // fail this test, not trap the suite.
        try #require(meta.count == 6)
        meta[5] = "requested"
        probe.metaState = meta

        let factories = try makeBatchedCacheFactories(for: [probe])
        let factory = try #require(factories.first)
        let produced = try #require(factory([0]) as? BatchRotatingKVCache)
        // `extract` only writes metaState for a populated cache, so fill one
        // row before asking what provenance comes back.
        let kv = makeKV(batchSize: 1, heads: 2, seqLen: 3, headDim: 4, value: 1)
        _ = produced.update(keys: kv.0, values: kv.1)
        let producedMeta = produced.extract(idx: 0).metaState
        try #require(producedMeta.count == 6)
        #expect(producedMeta[5] == "requested")

        // A model-native probe still yields model-native rows.
        let native = try makeBatchedCacheFactories(for: [RotatingKVCache(maxSize: 16, keep: 0)])
        let nativeFactory = try #require(native.first)
        let nativeProduced = try #require(nativeFactory([0]) as? BatchRotatingKVCache)
        _ = nativeProduced.update(keys: kv.0, values: kv.1)
        let nativeMeta = nativeProduced.extract(idx: 0).metaState
        try #require(nativeMeta.count == 6)
        #expect(nativeMeta[5] == "modelNative")
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
        // Recursive routing must reject it inside a composite too: a factory
        // validating only top-level types would batch a hybrid model's nested
        // keep-prefix cache.
        #expect(throws: BatchedCacheError.self) {
            _ = try makeBatchedCacheFactories(
                for: [CacheList(KVCacheSimple(), RotatingKVCache(maxSize: 16, keep: 4))])
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
    func protocolSurfaceFullAttention() throws {
        let cache: any BatchedCache = BatchKVCache(leftPadding: [0, 0])
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)

        // Shrink to row 1 only — an identity filter would let a missing or
        // no-op protocol witness pass. The surviving row must carry row 1's
        // distinct stamp (makeDistinctKV keys row i with value i+1).
        cache.filterBatched(batchIndices: MLXArray([Int32(1)]))
        let batch = try #require(cache as? BatchKVCache)
        #expect(batch.batchSize == 1)
        let extracted = cache.extractBatched(0)
        let simple = try #require(extracted as? KVCacheSimple)
        let extractedKeys = try #require(simple.state.first)
        #expect(maxAbsDifference(extractedKeys, MLXArray.ones(extractedKeys.shape) * 2) == 0)
        // advanceBatched is a no-op for full attention — and must stay one:
        // `PrefillBatch.prompt` calls it after every chunk, so a witness that
        // shifted offsets or state would move subsequent attention positions.
        let offsetsBefore = batch.batchOffsets.asArray(Int32.self)
        let paddingBefore = batch.leftPadding.asArray(Int32.self)
        cache.advanceBatched(1)
        #expect(batch.batchOffsets.asArray(Int32.self) == offsetsBefore)
        #expect(batch.leftPadding.asArray(Int32.self) == paddingBefore)
    }
}

// MARK: - SSM / composite batched lifecycle

@Suite(.serialized)
struct BatchedSSMCacheTests {

    @Test("ArraysCache conforms to the BatchedCache lifecycle")
    func arraysCacheBatchedLifecycle() throws {
        let mamba = MambaCache(leftPadding: [0, 0])
        // Distinguishable rows, so filtering is checkable by content below.
        mamba[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        mamba[1] = MLXArray(8 ..< 16).asType(.float32).reshaped([2, 4])

        let cache: any BatchedCache = mamba
        cache.prepareBatched(leftPadding: nil, lengths: [3, 5], rightPadding: nil)
        #expect(mamba.lengths != nil)

        cache.advanceBatched(1)
        cache.finalizeBatched()
        #expect(mamba.lengths == nil)

        cache.filterBatched(batchIndices: MLXArray([Int32(0)]))
        #expect(mamba.batchSize == 1)
        // The recurrent tensors themselves shrink to the selected row:
        // filtering only the metadata would hand a one-row input two-row
        // SSM state on the next model call.
        let slot0 = try #require(mamba[0])
        #expect(maxAbsDifference(slot0, MLXArray(0 ..< 4).asType(.float32).reshaped([1, 4])) == 0)
        let slot1 = try #require(mamba[1])
        #expect(maxAbsDifference(slot1, MLXArray(8 ..< 12).asType(.float32).reshaped([1, 4])) == 0)
    }

    @Test("advanceBatched leaves recurrent padding metadata to the model")
    func arraysCacheAdvanceIsModelOwned() throws {
        let mamba = MambaCache(leftPadding: [0, 2])
        mamba[0] = MLXArray.ones([2, 4])
        let cache: any BatchedCache = mamba
        cache.prepareBatched(leftPadding: nil, lengths: [4, 2], rightPadding: nil)

        // What a mask-aware mixer does during the forward pass: Mamba2,
        // FalconH1, GraniteMoeHybrid, LFM2MoE, Qwen3.5 and Qwen3Next all call
        // `advance(chunk)` themselves.
        mamba.advance(2)
        // #require and check the advanced value itself: a regression that
        // clears `currentLengths` would otherwise slip through the final
        // nil == nil comparison.
        let afterModel = try #require(mamba.currentLengths?.asArray(Int32.self))
        #expect(afterModel == [2, 0])

        // The engine's post-chunk hook must not advance it a second time, or
        // `lengths` runs a full chunk ahead and the next chunk's SSM mask
        // suppresses valid tokens.
        cache.advanceBatched(2)
        #expect(mamba.currentLengths?.asArray(Int32.self) == afterModel)
    }

    @Test("SSM factories drop an all-zero left padding so lengths bound the mask")
    func ssmFactoryZeroLeftPaddingKeepsLengthsBound() throws {
        // `ArraysCache.makeMask` treats a non-nil `leftPadding` as a
        // left-padded layout and makes it the only mask bound — even when
        // every entry is zero. The batched engine prefills right-padded and
        // always hands the factory an all-zero left padding, so the factory
        // must construct SSM caches with `nil`: otherwise `prepare(lengths:)`
        // stops excluding right padding and mask-aware mixers commit padding
        // tokens into their recurrent state.
        let factories = try makeBatchedCacheFactories(for: [MambaCache()])
        let factory = try #require(factories.first)
        let cache = try #require(factory([0, 0]) as? MambaCache)
        cache.prepare(lengths: [1, 2])

        let mask = try #require(cache.makeMask(N: 2))
        // Shape before contents: `asArray` flattens, so [4], [1, 4] or a
        // transposed layout with the same values would pass while broadcasting
        // along the wrong axis in a mask-aware mixer.
        #expect(mask.shape == [2, 2])
        // Row 0: length 1 -> position 0 only. Row 1: length 2 -> both.
        #expect(mask.asArray(Bool.self) == [true, false, true, true])
    }

    @Test("SSM factories preserve zero-row cardinality")
    func ssmFactoryPreservesZeroRowCardinality() throws {
        // An empty batch must stay empty: if the factory mapped `[]` to a nil
        // left padding, `ArraysCache.batchSize` would fall back to 1 and
        // `extend` would synthesize a phantom zero-filled row when allocating
        // the "missing" left-hand state.
        let factories = try makeBatchedCacheFactories(for: [MambaCache()])
        let factory = try #require(factories.first)
        let empty = try #require(factory([]) as? MambaCache)
        #expect(empty.batchSize == 0)

        let populated = MambaCache()
        populated[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        empty.extend(other: populated)
        // Content, not just the batch dimension: an extend that allocates
        // two correctly shaped rows but zero-fills the incoming recurrent
        // state would corrupt the admitted requests' next model call.
        let slot0 = try #require(empty[0])
        #expect(slot0.dim(0) == 2)
        #expect(maxAbsDifference(slot0, try #require(populated[0])) == 0)
        // Metadata has to grow with the tensor, or scheduling and later
        // filtering see a different batch size than the recurrent state.
        #expect(empty.batchSize == 2)
    }

    @Test("BatchedCacheList preserves nested topology")
    func batchedCacheListNested() throws {
        let factories = try makeBatchedCacheFactories(
            for: [CacheList(KVCacheSimple(), RotatingKVCache(maxSize: 16))])
        let factory = try #require(factories.first)
        let composite = factory([0, 0])
        let list = try #require(composite as? BatchedCacheList)
        // Give BOTH children two distinguishable rows so a filter that only
        // reaches one child is visible below (makeDistinctKV keys row i with
        // i+1); a composite with inconsistent child cardinalities selects
        // the wrong request or fails on later extraction.
        // Child count first: if the factory dropped the second nested cache —
        // the topology defect this test exists to catch — `list[1]` would trap
        // before Swift Testing could report it.
        try #require(list.children.count == 2)
        let attention = try #require(list[0] as? BatchKVCache)
        let rotating = try #require(list[1] as? BatchRotatingKVCache)
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = attention.update(keys: keys, values: values)
        _ = rotating.update(keys: keys, values: values)

        // Filtering routes through each child without flattening the
        // topology — and actually shrinks it: keep only row 1.
        list.filterBatched(batchIndices: MLXArray([Int32(1)]))
        #expect(attention.batchSize == 1)
        #expect(rotating.batchSize == 1)
        let extracted = list.extractBatched(0)
        let childList = try #require(extracted as? CacheList)
        try #require(childList.children.count == 2)
        let simple = try #require(childList[0] as? KVCacheSimple)
        let extractedKeys = try #require(simple.state.first)
        #expect(maxAbsDifference(extractedKeys, MLXArray.ones(extractedKeys.shape) * 2) == 0)
        let rotChild = try #require(childList[1] as? RotatingKVCache)
        let rotKeys = try #require(rotChild.state.first)
        #expect(maxAbsDifference(rotKeys, MLXArray.ones(rotKeys.shape) * 2) == 0)
        // Values too, from both children: a filter that selects row 1's keys
        // while retaining row 0's values would let the surviving hybrid
        // request attend another request's history (row 1 values are 20).
        let simpleValues = try #require(simple.state.last)
        #expect(maxAbsDifference(simpleValues, MLXArray.ones(simpleValues.shape) * 20) == 0)
        let rotValues = try #require(rotChild.state.last)
        #expect(maxAbsDifference(rotValues, MLXArray.ones(rotValues.shape) * 20) == 0)
    }

    @Test("BatchedCacheList extends every child")
    func batchedCacheListExtends() throws {
        // `DecodeBatch.extend` calls `extendBatched` on every layer cache when
        // a request is admitted mid-batch, composites included. A composite
        // that extended only one child, or appended inconsistent state, would
        // leave a dynamically admitted hybrid request running with mismatched
        // cache cardinalities — and nothing else in this suite covers it.
        let factories = try makeBatchedCacheFactories(
            for: [CacheList(KVCacheSimple(), RotatingKVCache(maxSize: 16))])
        let factory = try #require(factories.first)

        let base = try #require(factory([0, 0]) as? BatchedCacheList)
        try #require(base.children.count == 2)
        let baseAttention = try #require(base[0] as? BatchKVCache)
        let baseRotating = try #require(base[1] as? BatchRotatingKVCache)
        let baseKV = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = baseAttention.update(keys: baseKV.0, values: baseKV.1)
        _ = baseRotating.update(keys: baseKV.0, values: baseKV.1)

        let addition = try #require(factory([0]) as? BatchedCacheList)
        let additionAttention = try #require(addition[0] as? BatchKVCache)
        let additionRotating = try #require(addition[1] as? BatchRotatingKVCache)
        // Position-stamped: both children's appended state is compared against
        // this tensor, and a uniform fill cannot see a reordering inside it.
        let additionKV = makePositionKV(positions: 0 ..< 3, heads: 2, headDim: 4)
        _ = additionAttention.update(keys: additionKV.0, values: additionKV.1)
        _ = additionRotating.update(keys: additionKV.0, values: additionKV.1)

        base.extendBatched(addition)

        // Every child grew, and the appended row carries the incoming state.
        #expect(baseAttention.batchSize == 3)
        #expect(baseRotating.batchSize == 3)
        let appended = try #require(base.extractBatched(2) as? CacheList)
        try #require(appended.children.count == 2)
        let appendedSimple = try #require(appended[0] as? KVCacheSimple)
        let appendedKeys = try #require(appendedSimple.state.first)
        #expect(maxAbsDifference(appendedKeys, additionKV.0) == 0)
        let appendedRotating = try #require(appended[1] as? RotatingKVCache)
        let appendedRotatingKeys = try #require(appendedRotating.state.first)
        #expect(maxAbsDifference(appendedRotatingKeys, additionKV.0) == 0)
    }

    @Test("BatchedCacheList delegates makeMask to its attention child")
    func batchedCacheListDelegatesMask() throws {
        // Hybrid models pass the layer's CacheList itself to
        // createAttentionMask; the composite must surface the batched
        // attention child's left-padding mask, not BaseKVCache's `.none`.
        let factories = try makeBatchedCacheFactories(
            for: [CacheList(KVCacheSimple(), RotatingKVCache(maxSize: 16))])
        let factory = try #require(factories.first)
        let composite = factory([1, 0])

        let mode = composite.makeMask(n: 1, windowSize: nil, returnArray: false)
        switch mode {
        case .array(let mask):
            #expect(mask.dim(0) == 2)
            // Delegation semantics, not just shape: the child's [1, 0]
            // padding must survive — row 0's padded slot masked, row 1's
            // attendable. Any two-row array would pass the dim check alone.
            #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
            #expect(mask[1, 0, 0, 0].item(Bool.self) == true)
        case .arrays, .causal, .none:
            Issue.record("BatchedCacheList should delegate to the batched attention child")
        }
    }

    @Test("extract preserves the MambaCache subtype")
    func extractPreservesMambaSubtype() throws {
        let mamba = MambaCache(leftPadding: [0, 0])
        // Distinguishable rows: extraction that picks the wrong row or
        // zero-fills would corrupt subsequent SSM generation while passing a
        // subtype-and-slot-count check.
        mamba[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        mamba[1] = MLXArray(8 ..< 16).asType(.float32).reshaped([2, 4])

        let extracted = mamba.extract(0)
        let typed = try #require(extracted as? MambaCache)
        #expect(typed.slotCount == 2)
        let slot0 = try #require(typed[0])
        #expect(maxAbsDifference(slot0, MLXArray(0 ..< 4).asType(.float32).reshaped([1, 4])) == 0)
        let slot1 = try #require(typed[1])
        #expect(maxAbsDifference(slot1, MLXArray(8 ..< 12).asType(.float32).reshaped([1, 4])) == 0)

        // Row 1 as well: an extract that ignored its index argument and always
        // returned row 0 would pass every assertion above while handing the
        // engine another request's recurrent state.
        let extractedSecond = try #require(mamba.extract(1) as? MambaCache)
        let secondSlot0 = try #require(extractedSecond[0])
        #expect(
            maxAbsDifference(secondSlot0, MLXArray(4 ..< 8).asType(.float32).reshaped([1, 4])) == 0)
        let secondSlot1 = try #require(extractedSecond[1])
        #expect(
            maxAbsDifference(secondSlot1, MLXArray(12 ..< 16).asType(.float32).reshaped([1, 4]))
                == 0)
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
        // Finite, not merely non-NaN: an infinity that survives the masked
        // softmax without producing NaN would still corrupt downstream
        // computation.
        #expect(isNaN(out).any().item(Bool.self) == false)
        #expect(isInf(out).any().item(Bool.self) == false)
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

        // The rotating variant is just as unrestorable; a guard that only
        // recognized BatchKVCache would let a multi-row rotating snapshot
        // through to reload as an invalid single-sequence cache.
        let rotating = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 1], keep: 0)
        _ = rotating.update(keys: keys, values: values)
        let rotatingURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("batch-rotating-\(UUID().uuidString).safetensors")
        #expect(throws: (any Error).self) {
            try savePromptCache(url: rotatingURL, cache: [rotating])
        }
        #expect(!FileManager.default.fileExists(atPath: rotatingURL.path))
    }

    @Test("savePromptCache fails closed for zero-row array caches")
    func savePromptCacheRejectsZeroRowArraysCache() {
        // A populated cache filtered with an empty index set keeps tensors
        // with a batch dimension of 0; restoring those against a
        // single-request input fails later and less legibly than refusing
        // the save here.
        let mamba = MambaCache(leftPadding: [0, 0])
        mamba[0] = MLXArray.ones([2, 4])
        mamba.filter(batchIndices: MLXArray([Int32]()))

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("zero-row-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: (any Error).self) {
            try savePromptCache(url: url, cache: [mamba])
        }
        // Fail-closed means no artifact either: a save that writes the file
        // and then throws would leave callers an invalid cache on disk.
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test("savePromptCache keeps multi-row array-cache snapshots saveable")
    func savePromptCacheAllowsMultiRowArraysCache() throws {
        // The serialized format is batch-aware for ArraysCache/MambaCache —
        // state, lengths and left padding all round-trip (see the upstream
        // ArraysCache/MambaCache round-trip tests) — so multi-row instances
        // must stay saveable. Only zero-row instances and the Batch*
        // attention caches, which the restore registry cannot reconstruct,
        // are refused.
        let batched = MambaCache(leftPadding: [1, 0])
        batched[0] = MLXArray(0 ..< 8).asType(.float32).reshaped([2, 4])
        batched[1] = MLXArray(8 ..< 16).asType(.float32).reshaped([2, 4])
        // Unequal per-row lengths, so the metadata this test claims to cover
        // is actually exercised: a serializer that drops or reorders them
        // lets a resumed mask-aware mixer treat padding as live input.
        batched.prepare(lengths: [3, 5])

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mamba-cache-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [batched])

        let (loaded, _) = try loadPromptCache(url: url)
        // `loaded.first`, not `loaded[0]`: an empty restored array should
        // fail the #require, not trap before it can record the failure.
        let restored = try #require(loaded.first as? MambaCache)
        // Contents and batch metadata, not just shapes: state restored with
        // the right dimensions but wrong values would corrupt every
        // subsequent generation silently.
        let slot0 = try #require(restored[0])
        let slot1 = try #require(restored[1])
        #expect(maxAbsDifference(slot0, try #require(batched[0])) == 0)
        #expect(maxAbsDifference(slot1, try #require(batched[1])) == 0)
        #expect(restored.leftPaddingValues == [1, 0])
        #expect(restored.lengthsValues == [3, 5])
    }
}

// MARK: - Masking

@Suite(.serialized)
struct BatchMaskingTests {

    @Test("createCausalMask masks left padding per sequence")
    func causalMaskHonoursLeftPadding() throws {
        let leftPadding = MLXArray([Int32(1), Int32(2)])
        let mask = createCausalMask(n: 4, offset: 0, leftPadding: leftPadding)

        // #require every dimension the subscripts below rely on: a shorter
        // mask would otherwise trap in MLX after recording the failure.
        try #require(mask.dim(0) == 2)
        try #require(mask.dim(-1) >= 4)
        try #require(mask.dim(-2) >= 4)
        // Row 0: leftPadding 1 → position 0 masked, position 1 attendable on the diagonal.
        #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[0, 0, 1, 1].item(Bool.self) == true)
        // Row 1: leftPadding 2 → positions 0 and 1 masked, position 2 attendable.
        #expect(mask[1, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[1, 0, 0, 1].item(Bool.self) == false)
        #expect(mask[1, 0, 2, 2].item(Bool.self) == true)
        // The second padded position is what distinguishes padding 2 from
        // padding 1: its own diagonal must be masked, and a later query must
        // not attend the padded key either (causality alone would allow it).
        #expect(mask[1, 0, 1, 1].item(Bool.self) == false)
        #expect(mask[1, 0, 3, 1].item(Bool.self) == false)
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
        // Causality still holds within the update: key position 4 is the
        // second query's own slot, so the first query must not attend it. An
        // all-true array of the right width would pass every check above.
        #expect(mask[0, 0, 0, 4].item(Bool.self) == false)
    }
}

// MARK: - Helpers

/// Uniform KV: every position in every row holds the same value.
///
/// Use this only when the assertion is about shape, counters, or metadata. If
/// the assertion is about *which* elements survive or *in what order* -- trims,
/// round trips, window retention, appended rows -- reach for `makePositionKV`
/// instead: a uniform fill makes an operation and its inverse produce identical
/// tensors, so the test passes either way.
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

/// KV whose every key/value is stamped with its absolute position, so
/// window-retention tests can assert which positions survived by content.
/// `rowStride` offsets each batch row's stamps so multi-row tests can also
/// tell the rows apart — without it every row carries identical values and a
/// cache that swaps or copies rows passes unnoticed.
private func makePositionKV(
    positions: Range<Int>, heads: Int, headDim: Int, batchSize: Int = 1,
    rowStride: Float = 0
) -> (MLXArray, MLXArray) {
    var stampValues = [Float]()
    for row in 0 ..< batchSize {
        stampValues.append(contentsOf: positions.map { Float($0) + Float(row) * rowStride })
    }
    let stamps = MLXArray(stampValues).reshaped([batchSize, 1, positions.count, 1])
    let ones = MLXArray.ones([batchSize, heads, positions.count, headDim])
    return (ones * stamps, ones * (stamps + 100))
}
