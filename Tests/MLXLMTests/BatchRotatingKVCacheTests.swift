// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct BatchRotatingKVCacheTests {

    @Test("Update, filter, extend and extract keep each row's tokens")
    func lifecycle() throws {
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 0])
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)

        cache.filter(batchIndices: batchRows(0))
        #expect(cache.batchSize == 1)

        let added = BatchRotatingKVCache(maxSize: 16, leftPadding: [0])
        let addedKV = makePositionKV(positions: 0 ..< 2, heads: 2, headDim: 4)
        _ = added.update(keys: addedKV.0, values: addedKV.1)
        cache.extend(other: added)

        #expect(cache.batchSize == 2)
        #expect(cache.leftPadding.asArray(Int32.self) == [0, 1])
        #expect(cache.extract(0).offset == 3)
        let second = cache.extract(1)
        #expect(second.offset == 2)
        #expect(maxAbsDifference(try #require(second.state.first), addedKV.0) == 0)
        #expect(maxAbsDifference(try #require(second.state.last), addedKV.1) == 0)
    }

    @Test("isTrimmable(after:) predicts when the window fills")
    func trimmability() {
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0])
        let kv = makeKV(batchSize: 1, heads: 2, seqLen: 4, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)
        #expect(cache.isTrimmable)
        #expect(cache.isTrimmable(after: 3))
        #expect(!cache.isTrimmable(after: 4))
    }

    @Test("trim drops the newest tokens, no more than the shortest row holds")
    func trimClampsToShortestRow() throws {
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [2, 0])
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
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [2, 0])
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)

        #expect(cache.maxTrimCount == 1)
        #expect(!canTrimPromptCache([cache], numTokens: 2))
        #expect(canTrimPromptCache([cache], numTokens: 1))
        #expect(cache.trim(1) == 1)
    }

    @Test("An empty cache extended from another does not share its arrays")
    func extendEmptyCacheCopies() throws {
        // One token at a time, so the source's buffer has room past its tokens.
        let source = BatchRotatingKVCache(maxSize: 8, leftPadding: [0])
        append(0 ..< 1, to: source)
        append(1 ..< 2, to: source)
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [])
        cache.extend(other: source)

        // Both writes land inside the buffer the source allocated.
        append(50 ..< 51, to: cache)
        append(70 ..< 71, to: source)

        #expect(try view(row(cache)).0.asArray(Float.self) == [0, 1, 50])
        #expect(try view(row(source)).0.asArray(Float.self) == [0, 1, 70])
    }

    @Test("ropeOffset is per row through the KVCache type")
    func ropeOffsetThroughKVCache() {
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        let batch = BatchRotatingKVCache(maxSize: 16, leftPadding: [1, 0])
        _ = batch.update(keys: kv.0, values: kv.1)
        let cache: any KVCache = batch
        guard case .batch(let offsets) = cache.ropeOffset else {
            Issue.record("BatchRotatingKVCache.ropeOffset must be .batch")
            return
        }
        #expect(offsets.asArray(Int32.self) == [2, 3])
    }

    @Test("A ragged prefill keeps each row's tokens after finalize")
    func raggedPrefill() throws {
        let cache = BatchRotatingKVCache(maxSize: 32, leftPadding: [0, 0])
        cache.prepare(lengths: [3, 5])
        let kv = makePositionKV(
            positions: 0 ..< 5, heads: 2, headDim: 4, batchSize: 2, rowStride: 10)
        _ = cache.update(keys: kv.0, values: kv.1)
        cache.finalize()

        let first = cache.extract(0)
        #expect(first.offset == 3)
        #expect(try #require(first.state.first)[0, 0, 0..., 0].asArray(Float.self) == [0, 1, 2])
        #expect(
            try #require(first.state.last)[0, 0, 0..., 0].asArray(Float.self) == [100, 101, 102])
        let second = cache.extract(1)
        #expect(second.offset == 5)
        #expect(
            try #require(second.state.first)[0, 0, 0..., 0].asArray(Float.self)
                == [10, 11, 12, 13, 14])
    }

    @Test("A one-token ragged prefill finalizes before decode")
    func oneTokenRaggedPrefill() throws {
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 0])
        cache.prepare(lengths: [1, 0])
        let prefill = MLXArray([Float(10), 99]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: prefill, values: prefill + 100)
        cache.finalize()

        #expect(cache.extract(0).offset == 1)
        #expect(try #require(cache.extract(0).state.first).asArray(Float.self) == [10])
        #expect(cache.extract(1).offset == 0)
        #expect(try #require(cache.extract(1).state.first).size == 0)

        let decode = MLXArray([Float(20), 30]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: decode, values: decode + 100)
        #expect(try #require(cache.extract(0).state.first).asArray(Float.self) == [10, 20])
        #expect(try #require(cache.extract(0).state.last).asArray(Float.self) == [110, 120])
        #expect(cache.extract(1).offset == 1)
        #expect(try #require(cache.extract(1).state.first).asArray(Float.self) == [30])
    }

    @Test("A ragged prefill accepts a trailing one-token chunk")
    func trailingOneTokenChunk() throws {
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 0])
        cache.prepare(lengths: [2, 3])
        let first = makePositionKV(
            positions: 0 ..< 2, heads: 1, headDim: 1, batchSize: 2, rowStride: 10)
        _ = cache.update(keys: first.0, values: first.1)
        // Row 0 is already complete, so its final chunk entry is padding.
        let last = MLXArray([Float(99), 12]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: last, values: last + 100)
        cache.finalize()

        #expect(cache.extract(0).offset == 2)
        #expect(try #require(cache.extract(0).state.first).asArray(Float.self) == [0, 1])
        #expect(cache.extract(1).offset == 3)
        #expect(try #require(cache.extract(1).state.first).asArray(Float.self) == [10, 11, 12])

        let decode = MLXArray([Float(2), 13]).reshaped([2, 1, 1, 1])
        _ = cache.update(keys: decode, values: decode + 100)
        #expect(try #require(cache.extract(0).state.first).asArray(Float.self) == [0, 1, 2])
        #expect(
            try #require(cache.extract(1).state.first).asArray(Float.self) == [10, 11, 12, 13])
    }

    @Test("fromSingle keeps the retained window, below and at capacity")
    func fromSingleKeepsWindow() throws {
        for length in [5, 8] {
            let single = RotatingKVCache(maxSize: 8)
            append(0 ..< length, to: single)
            let batch = try #require(BatchRotatingKVCache.fromSingle(single))
            #expect(batch.batchSize == 1)
            try expectHistory(row(batch), single)
        }
    }

    @Test("fromSingle refuses a pinned prefix or a rewind reserve")
    func fromSingleRefusesUnbatchableCaches() {
        #expect(BatchRotatingKVCache.fromSingle(RotatingKVCache(maxSize: 8, keep: 2)) == nil)
        #expect(
            BatchRotatingKVCache.fromSingle(RotatingKVCache(maxSize: 8, rewindCapacity: 4))
                == nil)
    }

    @Test("A trimmed wrapped source keeps its rows through batching and decode")
    func trimmedWrappedSource() throws {
        let single = RotatingKVCache(maxSize: 8)
        for position in 0 ..< 17 { append(position ..< position + 1, to: single) }
        #expect(single.trim(2) == 2)
        #expect(single.offset == 15)

        let batch = try #require(BatchRotatingKVCache.fromSingle(single))
        try expectHistory(row(batch), single)
        for position in 15 ..< 20 {
            append(position ..< position + 1, to: single)
            append(position ..< position + 1, to: batch)
            #expect(batch.batchOffset.asArray(Int.self) == [position + 1])
            try expectHistory(row(batch), single)
        }
    }

    @Test("Wrapped and over-long histories survive conversion and further writes")
    func conversion() throws {
        try Device.withDefaultDevice(.cpu) {
            for overlong in [false, true] {
                let source = RotatingKVCache(maxSize: 8, step: 3)
                if overlong {
                    append(0 ..< 11, to: source)
                } else {
                    append(0 ..< 8, to: source)
                    for position in 8 ..< 11 { append(position ..< position + 1, to: source) }
                }
                let original = source.state.map { $0.asArray(Float.self) }
                let reference = try #require(source.copy() as? RotatingKVCache)
                let batch = try #require(BatchRotatingKVCache.fromSingle(source))
                #expect(batch.batchOffset.asArray(Int.self) == [11])
                #expect(batch.step == 3)
                #expect(
                    try view(row(batch)).0.asArray(Float.self) == Array(3 ... 10).map(Float.init))

                var position = 11
                for count in [1, 2, 1, 1, 1, 1, 1, 1, 1, 1] {
                    append(position ..< position + count, to: batch)
                    append(position ..< position + count, to: reference)
                    position += count
                    let extracted = try row(batch)
                    try expectHistory(extracted, reference)
                    #expect(extracted.allocationStep == 3)

                    // A multi-token write continues from the restored write position.
                    let continued = try #require(reference.copy() as? RotatingKVCache)
                    append(position ..< position + 2, to: extracted)
                    append(position ..< position + 2, to: continued)
                    try expectHistory(extracted, continued)
                }
                #expect(source.state.map { $0.asArray(Float.self) } == original)
            }
        }
    }

    @Test("Converting a full window leaves the source independently usable")
    func sourceIsolation() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = RotatingKVCache(maxSize: 8)
            append(0 ..< 8, to: source)
            let original = source.state.map { $0.asArray(Float.self) }
            let metadata = source.metaState
            let batch = try #require(BatchRotatingKVCache.fromSingle(source))
            append(8 ..< 9, to: batch)
            #expect(source.metaState == metadata)
            #expect(source.state.map { $0.asArray(Float.self) } == original)
            append(20 ..< 21, to: source)
            #expect(try view(source).0.asArray(Float.self) == [1, 2, 3, 4, 5, 6, 7, 20])
            #expect(try view(row(batch)).0.asArray(Float.self) == [1, 2, 3, 4, 5, 6, 7, 8])
        }
    }

    @Test("Extraction after a concatenating write continues with another one")
    func extractedConcatContinuation() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = RotatingKVCache(maxSize: 4)
            append(0 ..< 4, to: source)
            let batch = try #require(BatchRotatingKVCache.fromSingle(source))
            append(4 ..< 6, to: batch)
            append(4 ..< 6, to: source)
            let extracted = try row(batch)
            try expectHistory(extracted, source)
            append(6 ..< 8, to: extracted)
            append(6 ..< 8, to: source)
            try expectHistory(extracted, source)
        }
    }

    @Test("Converted rows extend with shorter histories, reorder and continue")
    func raggedExtension() throws {
        try Device.withDefaultDevice(.cpu) {
            let sources = [
                RotatingKVCache(maxSize: 8), RotatingKVCache(maxSize: 8),
                RotatingKVCache(maxSize: 8),
            ]
            append(0 ..< 8, to: sources[0])
            for position in 8 ..< 11 { append(position ..< position + 1, to: sources[0]) }
            append(100 ..< 113, to: sources[1])
            append(200 ..< 203, to: sources[2])
            let batch = try #require(BatchRotatingKVCache.fromSingle(sources[0]))
            batch.extend(other: try #require(BatchRotatingKVCache.fromSingle(sources[1])))
            batch.extend(other: try #require(BatchRotatingKVCache.fromSingle(sources[2])))
            for index in sources.indices { try expectHistory(row(batch, index), sources[index]) }

            batch.filter(batchIndices: batchRows(2, 0))
            let survivors = [sources[2], sources[0]]
            for step in 0 ..< 10 {
                let next = MLXArray([Float(203 + step), Float(11 + step)]).reshaped([2, 1, 1, 1])
                _ = batch.update(keys: next, values: next + 100)
                append(203 + step ..< 204 + step, to: survivors[0])
                append(11 + step ..< 12 + step, to: survivors[1])
                for index in survivors.indices {
                    try expectHistory(row(batch, index), survivors[index])
                }
            }
        }
    }

    @Test("One-token prepared masks match the returned keys at capacity", arguments: [false, true])
    func preparedMask(wrapped: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [2, 0])
            let initial = MLXArray([Float(90), 91, 0, 1, 10, 11, 12, 13]).reshaped([2, 1, 4, 1])
            _ = cache.update(keys: initial, values: initial + 100)
            if wrapped {
                let token = MLXArray([Float(2), 14]).reshaped([2, 1, 1, 1])
                _ = cache.update(keys: token, values: token + 100)
            }
            cache.prepare(lengths: [1, 0])
            guard case .array(let mask) = cache.makeMask(n: 1, windowSize: 3, returnArray: true)
            else {
                Issue.record("Expected an array mask")
                return
            }
            let next = MLXArray([Float(wrapped ? 3 : 2), 99]).reshaped([2, 1, 1, 1])
            let (keys, values) = cache.update(keys: next, values: next + 100)
            let latest = Float(wrapped ? 3 : 2)
            let expectedMask = keys[0, 0, 0..., 0].asArray(Float.self).map {
                $0 >= latest - 2 && $0 <= latest
            }
            #expect(mask[0, 0, 0].asArray(Bool.self) == expectedMask)
            let scores = MLX.where(mask, MLXArray(Float(0)), MLXArray(-Float.infinity))
            let attended = matmul(softmax(scores, axis: -1), values)[0].item(Float.self)
            #expect(abs(attended - (latest - 1 + 100)) < 0.0001)

            cache.finalize()
            #expect(cache.batchOffset.asArray(Int.self) == [wrapped ? 4 : 3, wrapped ? 5 : 4])
            let decode = MLXArray([latest + 1, Float(wrapped ? 15 : 14)]).reshaped([2, 1, 1, 1])
            _ = cache.update(keys: decode, values: decode + 100)
            let first = try view(row(cache, 0)).0.asArray(Float.self)
            #expect(first == (wrapped ? [1, 2, 3, 4] : [0, 1, 2, 3]))
            let second = try view(row(cache, 1)).0.asArray(Float.self)
            #expect(second == (wrapped ? [12, 13, 14, 15] : [11, 12, 13, 14]))
        }
    }

    @Test("A multi-token mask after a wrap predicts the trim of the matching update")
    func multiTokenMaskAfterWrap() {
        let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [2, 0])
        let (k4, v4) = makeKV(batchSize: 2, heads: 2, seqLen: 4, headDim: 4)
        _ = cache.update(keys: k4, values: v4)
        let (k1, v1) = makeKV(batchSize: 2, heads: 2, seqLen: 1, headDim: 4)
        _ = cache.update(keys: k1, values: v1)

        guard case .array(let mask) = cache.makeMask(n: 2, windowSize: nil, returnArray: true)
        else {
            Issue.record("Expected an array mask")
            return
        }
        #expect(mask.dim(-1) == 5)
        #expect(mask[0, 0, 0, 0].item(Bool.self) == true)
        #expect(mask[1, 0, 0, 0].item(Bool.self) == true)
        #expect(mask[0, 0, 0, 4].item(Bool.self) == false)
    }

    @Test("Extraction keeps the window, step and capacity origin, also for an empty row")
    func extractionMetadata() throws {
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 2])
        cache.capacityOrigin = .requested
        cache.step = 3
        for index in 0 ..< 2 {
            let extracted = try row(cache, index)
            #expect(extracted.maxSize == 16)
            #expect(extracted.allocationStep == 3)
            #expect(extracted.capacityOrigin == .requested)
            #expect(extracted.offset == 0)
        }
        let kv = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: kv.0, values: kv.1)
        #expect(try row(cache, 0).capacityOrigin == .requested)
    }

    @Test("The factory carries the window, step and capacity origin")
    func factory() throws {
        let probe = RotatingKVCache(maxSize: 16, step: 3)
        probe.capacityOrigin = .requested
        let factory = try #require(try makeBatchedCacheFactories(for: [probe]).first)
        let cache = try #require(factory([0]) as? BatchRotatingKVCache)
        #expect(cache.maxSize == 16)
        #expect(cache.step == 3)
        #expect(cache.capacityOrigin == .requested)

        let list = try #require(
            try makeBatchedCacheFactories(
                for: [CacheList(KVCacheSimple(), RotatingKVCache(maxSize: 16))]
            ).first)
        let composite = try #require(list([0, 0]) as? BatchedCacheList)
        #expect(composite[1] is BatchRotatingKVCache)
    }

    @Test("The factory refuses a pinned prefix or a rewind reserve, also when nested")
    func factoryRefusesUnbatchableCaches() {
        for cache in [
            RotatingKVCache(maxSize: 16, keep: 4), RotatingKVCache(maxSize: 16, rewindCapacity: 4),
        ] {
            #expect(throws: BatchedCacheError.self) {
                _ = try makeBatchedCacheFactories(for: [cache])
            }
            #expect(throws: BatchedCacheError.self) {
                _ = try makeBatchedCacheFactories(for: [CacheList(KVCacheSimple(), cache)])
            }
        }
    }

    @Test("Extending rejects a different allocation step")
    func mismatchedSteps() async throws {
        let result = try await #require(
            processExitsWith: .failure, observing: [\.standardErrorContent]
        ) {
            Device.withDefaultDevice(.cpu) {
                let receiver = BatchRotatingKVCache(maxSize: 8, leftPadding: [0])
                let other = BatchRotatingKVCache(maxSize: 8, leftPadding: [0])
                other.step = 3
                let kv = MLXArray.ones([1, 1, 1, 1])
                _ = receiver.update(keys: kv, values: kv)
                _ = other.update(keys: kv, values: kv)
                receiver.extend(other: other)
            }
        }
        #expect(
            String(decoding: result.standardErrorContent, as: UTF8.self).contains(
                "requires matching maxSize, step and capacityOrigin"))
    }

    @Test("A batched rotating cache cannot be saved as a prompt cache")
    func savePromptCacheRefuses() {
        let cache = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 1])
        let (keys, values) = makeKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = cache.update(keys: keys, values: values)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("batch-rotating-\(UUID().uuidString).safetensors")
        #expect(throws: (any Error).self) {
            try savePromptCache(url: url, cache: [cache])
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }
}

private func append(_ positions: Range<Int>, to cache: any KVCache) {
    let keys = MLXArray(positions.map(Float.init)).reshaped([1, 1, positions.count, 1])
    _ = cache.update(keys: keys, values: keys + 100)
}

private func row(_ cache: BatchRotatingKVCache, _ index: Int = 0) throws -> RotatingKVCache {
    try #require(cache.extract(index) as? RotatingKVCache)
}

private func view(_ cache: RotatingKVCache) throws -> (MLXArray, MLXArray) {
    let window = try #require(cache.maxSize)
    return try #require(cache.logicalView(tail: window))
}

private func expectHistory(_ actual: RotatingKVCache, _ expected: RotatingKVCache) throws {
    #expect(actual.offset == expected.offset)
    #expect(actual.maxSize == expected.maxSize)
    let a = try view(actual)
    let b = try view(expected)
    #expect(a.0.asArray(Float.self) == b.0.asArray(Float.self))
    #expect(a.1.asArray(Float.self) == b.1.asArray(Float.self))
}
