// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct BatchKVCacheTests {

    @Test("Update, filter, extend and extract keep each row's tokens")
    func lifecycle() throws {
        let cache = BatchKVCache(leftPadding: [0, 0])
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)

        cache.filter(batchIndices: batchRows(0))
        #expect(cache.batchSize == 1)
        #expect(cache.leftPadding.asArray(Int32.self) == [0])

        let added = BatchKVCache(leftPadding: [0])
        let addedKV = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
        _ = added.update(keys: addedKV.0, values: addedKV.1)
        cache.extend(other: added)

        #expect(cache.batchSize == 2)
        #expect(cache.leftPadding.asArray(Int32.self) == [0, 1])
        #expect(cache.batchOffset.asArray(Int32.self) == [3, 2])
        #expect(cache.extract(0).offset == 3)

        let second = cache.extract(1)
        #expect(second.offset == 2)
        #expect(maxAbsDifference(try #require(second.state.first), addedKV.0) == 0)
        #expect(maxAbsDifference(try #require(second.state.last), addedKV.1) == 0)
    }

    @Test("Filtering during a ragged prefill keeps the survivors' right padding")
    func filterDuringRaggedPrefill() throws {
        let cache = BatchKVCache(leftPadding: [0, 0])
        cache.prepare(lengths: [3, 5])
        let kv = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 5, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)

        cache.filter(batchIndices: batchRows(1))
        cache.finalize()

        let row = cache.extract(0)
        #expect(row.offset == 5)
        #expect(cache.leftPadding.asArray(Int32.self) == [0])
        let keys = try #require(row.state.first)
        #expect(maxAbsDifference(keys, MLXArray.ones(keys.shape) * 2) == 0)
        let values = try #require(row.state.last)
        #expect(maxAbsDifference(values, MLXArray.ones(values.shape) * 20) == 0)
    }

    @Test("finalize keeps every row's tokens when the buffer has spare capacity")
    func finalizeWithSpareCapacity() throws {
        let cache = BatchKVCache(leftPadding: [0, 0])
        cache.prepare(lengths: [3, 5])
        // Row 0 holds 3 tokens and 2 right-padding slots, row 1 holds 5 tokens.
        var stamps = [Float](repeating: 0, count: 10)
        for row in 0 ..< 2 {
            for position in 0 ..< (row == 0 ? 3 : 5) {
                stamps[row * 5 + position] = Float(10 * (row + 1) + position)
            }
        }
        let keys = MLXArray(stamps, [2, 1, 5, 1])
        _ = cache.update(keys: keys, values: keys * 2)

        cache.finalize()

        let first = try #require(cache.extract(0) as? KVCacheSimple)
        #expect(first.offset == 3)
        #expect(try #require(first.keys).asArray(Float.self) == [10, 11, 12])
        #expect(try #require(first.values).asArray(Float.self) == [20, 22, 24])
        let second = try #require(cache.extract(1) as? KVCacheSimple)
        #expect(second.offset == 5)
        #expect(try #require(second.keys).asArray(Float.self) == [20, 21, 22, 23, 24])
        #expect(try #require(second.values).asArray(Float.self) == [40, 42, 44, 46, 48])
    }

    @Test("fromSingle and extract round-trip a sequence")
    func fromSingleRoundTrip() throws {
        let single = KVCacheSimple()
        let kv = makePositionKV(positions: 0 ..< 4, heads: 2, headDim: 4)
        _ = single.update(keys: kv.0, values: kv.1)

        let batch = BatchKVCache.fromSingle(single)
        #expect(batch.batchSize == 1)

        let restored = batch.extract(0)
        #expect(restored.offset == single.offset)
        #expect(
            maxAbsDifference(try #require(restored.state.first), try #require(single.state.first))
                == 0)
        #expect(
            maxAbsDifference(try #require(restored.state.last), try #require(single.state.last))
                == 0)

        // Writing to the batch leaves the source unchanged.
        let next = makePositionKV(positions: 4 ..< 5, heads: 2, headDim: 4)
        _ = batch.update(keys: next.0, values: next.1)
        #expect(single.offset == 4)
        #expect(maxAbsDifference(try #require(single.state.first), kv.0) == 0)
    }

    @Test("trim drops the newest tokens, no more than the shortest row holds")
    func trimClampsToShortestRow() throws {
        let cache = BatchKVCache(leftPadding: [2, 0])
        let kv = makePositionKV(
            positions: 0 ..< 3, heads: 2, headDim: 4, batchSize: 2, rowStride: 10)
        _ = cache.update(keys: kv.0, values: kv.1)

        #expect(cache.trim(2) == 1)
        #expect(cache.batchOffset.asArray(Int32.self) == [0, 2])
        #expect(cache.extract(0).offset == 0)
        let second = cache.extract(1)
        #expect(second.offset == 2)
        #expect(try #require(second.state.first)[0, 0, 0..., 0].asArray(Float.self) == [10, 11])
        #expect(try #require(second.state.last)[0, 0, 0..., 0].asArray(Float.self) == [110, 111])
    }

    @Test("maxTrimCount reports only what trim can remove")
    func maxTrimCountMatchesTrim() {
        let cache = BatchKVCache(leftPadding: [2, 0])
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)

        #expect(cache.maxTrimCount == 1)
        #expect(!canTrimPromptCache([cache], numTokens: 2))
        #expect(canTrimPromptCache([cache], numTokens: 1))
        #expect(cache.trim(1) == 1)
    }

    @Test("An empty cache extended from another does not share its arrays")
    func extendEmptyCacheCopies() throws {
        let source = BatchKVCache(leftPadding: [0])
        let prompt = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
        _ = source.update(keys: prompt.0, values: prompt.1)
        let cache = BatchKVCache(leftPadding: [])
        cache.extend(other: source)

        // Both writes land inside the buffer the source allocated.
        let next = makePositionKV(positions: 50 ..< 51, heads: 2, headDim: 4)
        _ = cache.update(keys: next.0, values: next.1)
        let sourceNext = makePositionKV(positions: 70 ..< 71, heads: 2, headDim: 4)
        _ = source.update(keys: sourceNext.0, values: sourceNext.1)

        let keys = try #require(cache.extract(0).state.first)
        #expect(keys[0, 0, 0..., 0].asArray(Float.self) == [0, 1, 50])
        let sourceKeys = try #require(source.extract(0).state.first)
        #expect(sourceKeys[0, 0, 0..., 0].asArray(Float.self) == [0, 1, 70])
    }

    @Test("Rows without keys join as padding only, in either direction")
    func extendKeepsRowsWithoutKeys() throws {
        let kv = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)

        // Rows that were never prefilled receive a prefilled one.
        let receiver = BatchKVCache(leftPadding: [0, 0])
        let prefilled = BatchKVCache(leftPadding: [0])
        _ = prefilled.update(keys: kv.0, values: kv.1)
        receiver.extend(other: prefilled)
        #expect(receiver.batchSize == 3)
        #expect(receiver.leftPadding.asArray(Int32.self) == [2, 2, 0])
        #expect(receiver.batchOffset.asArray(Int32.self) == [0, 0, 2])
        #expect(receiver.extract(0).offset == 0)
        let joined = try #require(receiver.extract(2).state.first)
        #expect(joined[0, 0, 0..., 0].asArray(Float.self) == [0, 1])

        // A prefilled batch receives a row that was never prefilled.
        let batch = BatchKVCache(leftPadding: [0])
        _ = batch.update(keys: kv.0, values: kv.1)
        batch.extend(other: BatchKVCache(leftPadding: [0]))
        #expect(batch.batchSize == 2)
        #expect(batch.leftPadding.asArray(Int32.self) == [0, 2])
        #expect(batch.batchOffset.asArray(Int32.self) == [2, 0])
        let kept = try #require(batch.extract(0).state.first)
        #expect(kept[0, 0, 0..., 0].asArray(Float.self) == [0, 1])
        #expect(maskKind(batch, n: 1) == "array")
    }

    @Test("Extending with an empty cache changes nothing")
    func extendWithEmptyCache() throws {
        let cache = BatchKVCache(leftPadding: [0])
        let kv = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)
        cache.extend(other: BatchKVCache(leftPadding: []))
        #expect(cache.batchSize == 1)
        #expect(cache.offset == 2)
        #expect(maskKind(cache, n: 1) == "none")
        let keys = try #require(cache.extract(0).state.first)
        #expect(keys[0, 0, 0..., 0].asArray(Float.self) == [0, 1])

        let unprefilled = BatchKVCache(leftPadding: [1, 0])
        unprefilled.extend(other: BatchKVCache(leftPadding: []))
        #expect(unprefilled.leftPadding.asArray(Int32.self) == [1, 0])
    }

    @Test("A padded row without keys starts empty when it joins stored rows")
    func paddedRowWithoutKeysStartsEmpty() throws {
        let kv = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)

        // The row without keys receives a prefilled one.
        let receiver = BatchKVCache(leftPadding: [1])
        let prefilled = BatchKVCache(leftPadding: [0])
        _ = prefilled.update(keys: kv.0, values: kv.1)
        receiver.extend(other: prefilled)
        #expect(receiver.leftPadding.asArray(Int32.self) == [2, 0])
        #expect(receiver.batchOffset.asArray(Int32.self) == [0, 2])
        #expect(receiver.extract(0).offset == 0)

        // A prefilled batch receives the row without keys.
        let batch = BatchKVCache(leftPadding: [0])
        _ = batch.update(keys: kv.0, values: kv.1)
        batch.extend(other: BatchKVCache(leftPadding: [1]))
        #expect(batch.leftPadding.asArray(Int32.self) == [0, 2])
        #expect(batch.batchOffset.asArray(Int32.self) == [2, 0])

        // Filtering down to that row leaves an empty sequence that takes new tokens.
        batch.filter(batchIndices: batchRows(1))
        #expect(batch.offset == 0)
        #expect(batch.leftPadding.asArray(Int32.self) == [0])
        let next = makePositionKV(positions: 7 ..< 8, heads: 2, headDim: 4)
        _ = batch.update(keys: next.0, values: next.1)
        let keys = try #require(batch.extract(0).state.first)
        #expect(keys[0, 0, 0..., 0].asArray(Float.self) == [7])

        // Without stored rows, the padding still waits for the prefill.
        let pending = BatchKVCache(leftPadding: [1])
        pending.extend(other: BatchKVCache(leftPadding: [2]))
        #expect(pending.leftPadding.asArray(Int32.self) == [1, 2])
        #expect(pending.batchOffset.asArray(Int32.self) == [-1, -2])
    }

    @Test("The first token of a joined row without keys attends like a single sequence")
    func joinedRowFirstTokenMatchesSingleSequence() {
        let batch = BatchKVCache(leftPadding: [0])
        let history = makePositionKV(positions: 0 ..< 3, heads: 2, headDim: 4)
        _ = batch.update(keys: history.0, values: history.1)
        batch.extend(other: BatchKVCache(leftPadding: [1]))

        // One decode step as a model runs it: mask, cache update, attention.
        let step = makePositionKV(
            positions: 3 ..< 4, heads: 2, headDim: 4, batchSize: 2, rowStride: 10)
        let queries = MLXArray.ones([2, 2, 1, 4])
        let mask = batch.makeMask(n: 1, windowSize: nil, returnArray: false)
        let (keys, values) = batch.update(keys: step.0, values: step.1)
        let batched = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 0.5, mask: mask)

        let single = KVCacheSimple()
        let singleMask = single.makeMask(n: 1, windowSize: nil, returnArray: false)
        let (rowKeys, rowValues) = single.update(keys: step.0[1 ..< 2], values: step.1[1 ..< 2])
        let expected = MLXFast.scaledDotProductAttention(
            queries: queries[1 ..< 2], keys: rowKeys, values: rowValues, scale: 0.5,
            mask: singleMask)
        #expect(maxAbsDifference(batched[1 ..< 2], expected) < 1e-5)
    }

    @Test("filter clears the padding flag only when it removes the padding")
    func filterRecomputesPaddingFlag() {
        // Stored rows with two positions of shared padding: filter removes them.
        let stored = BatchKVCache(leftPadding: [2, 2, 0])
        let kv = makeKV(batchSize: 3, heads: 2, seqLen: 3, headDim: 4)
        _ = stored.update(keys: kv.0, values: kv.1)
        stored.filter(batchIndices: batchRows(0, 1))
        #expect(stored.leftPadding.asArray(Int32.self) == [0, 0])
        #expect(maskKind(stored, n: 1) == "none")

        // Rows without keys keep the padding that waits for their prefill.
        let pending = BatchKVCache(leftPadding: [2, 2, 0])
        pending.filter(batchIndices: batchRows(0, 1))
        #expect(pending.leftPadding.asArray(Int32.self) == [2, 2])
        #expect(maskKind(pending, n: 1) == "array")
    }

    @Test("A request admitted to an empty batch keeps the symbolic mask")
    func firstAdmissionKeepsSymbolicMask() {
        let request = BatchKVCache(leftPadding: [0])
        let kv = makeKV(batchSize: 1, heads: 2, seqLen: 3, headDim: 4)
        _ = request.update(keys: kv.0, values: kv.1)
        let batch = BatchKVCache(leftPadding: [])
        batch.extend(other: request)
        #expect(maskKind(batch, n: 1) == "none")
        #expect(maskKind(batch, n: 2) == "causal")
    }

    @Test("Without padding, a window or a requested array still gets the explicit mask")
    func explicitMaskForWindowOrRequestedArray() throws {
        let cache = BatchKVCache(leftPadding: [0])
        let kv = makeKV(batchSize: 1, heads: 2, seqLen: 8, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)
        #expect(maskKind(cache, n: 1, returnArray: true) == "array")

        guard case .array(let mask) = cache.makeMask(n: 2, windowSize: 4, returnArray: false)
        else {
            Issue.record("A window needs an explicit mask")
            return
        }
        // The token at position 8 sees positions 5 through 8 only.
        #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[0, 0, 0, 4].item(Bool.self) == false)
        #expect(mask[0, 0, 0, 5].item(Bool.self) == true)
    }

    @Test("The mask is symbolic only while no row has left padding")
    func symbolicMaskWithoutPadding() throws {
        let cache = BatchKVCache(leftPadding: [0, 0])
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)
        #expect(maskKind(cache, n: 1) == "none")
        #expect(maskKind(cache, n: 2) == "causal")

        // A shorter row joins with one position of left padding.
        let added = BatchKVCache(leftPadding: [0])
        let addedKV = makeKV(batchSize: 1, heads: 2, seqLen: 2, headDim: 4)
        _ = added.update(keys: addedKV.0, values: addedKV.1)
        cache.extend(other: added)
        #expect(maskKind(cache, n: 1) == "array")
        #expect(maskKind(try #require(cache.copy() as? BatchKVCache), n: 1) == "array")

        // Without the padded row, the mask is symbolic again.
        cache.filter(batchIndices: batchRows(0, 1))
        #expect(maskKind(cache, n: 1) == "none")

        // A ragged prefill leaves the shorter row padded, also after a state restore.
        let ragged = BatchKVCache(leftPadding: [0, 0])
        ragged.prepare(lengths: [2, 3])
        _ = ragged.update(keys: kv.0, values: kv.1)
        ragged.finalize()
        #expect(maskKind(ragged, n: 1) == "array")
        let restored = BatchKVCache(leftPadding: [])
        restored.state = ragged.state
        #expect(maskKind(restored, n: 1) == "array")
    }

    @Test("The mask excludes each row's left padding, also for one token")
    func maskExcludesLeftPadding() throws {
        let cache = BatchKVCache(leftPadding: [1, 3, 0])
        for n in [1, 2] {
            guard case .array(let mask) = cache.makeMask(n: n, windowSize: nil, returnArray: false)
            else {
                Issue.record("BatchKVCache must return an array mask")
                return
            }
            try #require(mask.dim(0) == 3)
            #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
            #expect(mask[1, 0, 0, 0].item(Bool.self) == false)
            #expect(mask[2, 0, 0, 0].item(Bool.self) == true)
        }
        guard case .array(let mask) = cache.makeMask(n: 2, windowSize: nil, returnArray: false)
        else { return }
        // Row 0 (padding 1) attends position 1; row 1 (padding 3) does not.
        #expect(mask[0, 0, 1, 1].item(Bool.self) == true)
        #expect(mask[1, 0, 1, 1].item(Bool.self) == false)
    }

    @Test("ropeOffset is per row through the KVCache type")
    func ropeOffsetThroughKVCache() {
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        let batch = BatchKVCache(leftPadding: [1, 0])
        _ = batch.update(keys: kv.0, values: kv.1)
        let cache: any KVCache = batch
        guard case .batch(let offsets) = cache.ropeOffset else {
            Issue.record("BatchKVCache.ropeOffset must be .batch")
            return
        }
        #expect(offsets.asArray(Int32.self) == [2, 3])
    }

    @Test("createCausalMask masks each row's left padding")
    func causalMaskWithLeftPadding() throws {
        let mask = createCausalMask(n: 4, offset: 0, leftPadding: MLXArray([Int32(1), 2]))
        try #require(mask.shape == [2, 1, 4, 4])
        #expect(mask[0, 0, 0, 0].item(Bool.self) == false)
        #expect(mask[0, 0, 1, 1].item(Bool.self) == true)
        #expect(mask[1, 0, 1, 1].item(Bool.self) == false)
        #expect(mask[1, 0, 3, 1].item(Bool.self) == false)
        #expect(mask[1, 0, 2, 2].item(Bool.self) == true)
    }

    @Test("The factory batches KVCacheSimple and refuses caches without a batched form")
    func factory() throws {
        let factories = try makeBatchedCacheFactories(for: [KVCacheSimple()])
        let factory = try #require(factories.first)
        #expect((factory([0, 2]) as? BatchKVCache)?.leftPadding.asArray(Int32.self) == [0, 2])

        #expect(throws: BatchedCacheError.self) {
            _ = try makeBatchedCacheFactories(for: [QuantizedKVCache()])
        }
        #expect(throws: BatchedCacheError.self) {
            _ = try makeBatchedCacheFactories(for: [ChunkedKVCache(chunkSize: 16)])
        }
    }

    @Test("The BatchedCache protocol drives a BatchKVCache")
    func protocolSurface() throws {
        let cache: any BatchedCache = BatchKVCache(leftPadding: [0, 0])
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)

        cache.filter(batchIndices: batchRows(1))
        #expect((cache as? BatchKVCache)?.batchSize == 1)
        let row = try #require(cache.extract(0) as? KVCacheSimple)
        let rowKeys = try #require(row.state.first)
        #expect(maxAbsDifference(rowKeys, MLXArray.ones(rowKeys.shape) * 2) == 0)
    }

    @Test("A batched cache cannot be saved as a prompt cache")
    func savePromptCacheRefuses() {
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
}

/// The kind of mask `cache` returns for `n` new tokens.
private func maskKind(_ cache: BatchKVCache, n: Int, returnArray: Bool = false) -> String {
    switch cache.makeMask(n: n, windowSize: nil, returnArray: returnArray) {
    case .none: "none"
    case .causal: "causal"
    case .array: "array"
    case .arrays: "arrays"
    }
}
