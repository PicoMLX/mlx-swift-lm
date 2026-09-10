import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct WrappedRotatingCacheTests {
    @Test("Converting a chronological full window leaves the source independently usable")
    func sourceIsolation() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = RotatingKVCache(maxSize: 8)
            append(0 ..< 8, to: source)
            let original = source.state.map { $0.asArray(Float.self) }
            let metadata = source.metaState
            let batch = BatchRotatingKVCache.fromSingle(source)
            append(8 ..< 9, to: batch)
            #expect(source.metaState == metadata)
            #expect(source.state.map { $0.asArray(Float.self) } == original)
            append(20 ..< 21, to: source)
            let sourceKeys = try view(source).0
            let batchKeys = try view(batch.toSingle()).0
            #expect(sourceKeys.asArray(Float.self) == [1, 2, 3, 4, 5, 6, 7, 20])
            #expect(batchKeys.asArray(Float.self) == [1, 2, 3, 4, 5, 6, 7, 8])
        }
    }

    @Test(
        "Wrapped adoption and cached ragged prefill preserve real-model logits",
        arguments: [false, true])
    func modelParity(overlong: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let config = Gemma3TextConfiguration(
                modelType: "gemma3_text", hiddenSize: 16, hiddenLayers: 4, intermediateSize: 32,
                attentionHeads: 2, headDim: 8, rmsNormEps: 1e-5, vocabularySize: 16, kvHeads: 1,
                ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000, ropeTraditional: false,
                queryPreAttnScalar: 8, slidingWindow: 8, slidingWindowPattern: 2,
                maxPositionEmbeddings: 128)
            let model = withRandomState(MLXRandom.RandomState(seed: 503)) {
                let model = Gemma3TextModel(config)
                eval(model)
                return model
            }
            let singles = try [model.newCache(parameters: nil), model.newCache(parameters: nil)]
            if overlong {
                eval(model(MLXArray(Array(1 ... 11)).reshaped([1, 11]), cache: singles[0]))
            } else {
                eval(model(MLXArray(Array(1 ... 8)).reshaped([1, 8]), cache: singles[0]))
                for token in 9 ... 11 {
                    eval(model(MLXArray([token]).reshaped([1, 1]), cache: singles[0]))
                }
            }
            eval(model(MLXArray([1, 2, 3]).reshaped([1, 3]), cache: singles[1]))
            func convert(_ cache: any KVCache) throws -> any BatchedCache {
                if let rotating = cache as? RotatingKVCache {
                    return BatchRotatingKVCache.fromSingle(rotating)
                }
                return BatchKVCache.fromSingle(try #require(cache as? KVCacheSimple))
            }
            let batched = try singles[0].map(convert)
            for layer in batched.indices {
                batched[layer].extendBatched(try convert(singles[1][layer]))
                batched[layer].prepareBatched(
                    leftPadding: nil, lengths: [2, 1], rightPadding: [0, 1])
            }
            let first = model(MLXArray([7, 8]).reshaped([2, 1]), cache: batched)
            for (row, token) in [7, 8].enumerated() {
                let expected = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
                expectLogits(first[row ..< row + 1], expected)
            }
            let trailing = model(MLXArray([9, 0]).reshaped([2, 1]), cache: batched)
            let expectedTrailing = model(MLXArray([9]).reshaped([1, 1]), cache: singles[0])
            expectLogits(trailing[0 ..< 1], expectedTrailing)
            batched.forEach { $0.finalizeBatched() }
            try expectModelHistories(batched, singles)

            for step in 0 ..< 6 {
                let token = 4 + step
                let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: batched)
                for row in 0 ..< 2 {
                    let expected = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
                    expectLogits(output[row ..< row + 1], expected)
                }
                try expectModelHistories(batched, singles)
            }
            batched.forEach { $0.filterBatched(batchIndices: MLXArray([Int32(0)])) }
            let continuation = MLXArray([4, 5]).reshaped([1, 2])
            expectLogits(
                model(continuation, cache: batched), model(continuation, cache: singles[0]))
            try expectModelHistories(batched, [singles[0]])
            let extracted = batched.map { $0.extractBatched(0) }
            expectLogits(
                model(continuation, cache: extracted), model(continuation, cache: singles[0]))
        }
    }

    @Test(
        "Wrapped and over-long native histories survive conversion and continued writes",
        arguments: [0, 2], [false, true])
    func conversion(keep: Int, overlong: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let source = RotatingKVCache(maxSize: 8, keep: keep, step: 3)
            var metadata = source.metaState
            metadata[5] = "requested"
            source.metaState = metadata
            if overlong {
                append(0 ..< 11, to: source)
            } else {
                append(0 ..< 8, to: source)
                for position in 8 ..< 11 { append(position ..< position + 1, to: source) }
            }
            let original = source.state.map { $0.asArray(Float.self) }
            let originalMetadata = source.metaState
            let reference = try #require(source.copy() as? RotatingKVCache)
            let batch = BatchRotatingKVCache.fromSingle(source)
            #expect(batch.batchOffsets.asArray(Int.self) == [11])
            #expect(batch.step == 3)
            let expected: [Float] =
                keep == 0 ? Array(3 ... 10).map(Float.init) : [0, 1, 5, 6, 7, 8, 9, 10]
            #expect(try view(batch.toSingle()).0.asArray(Float.self) == expected)
            try expectHistory(batch.toSingle(), reference)

            var position = 11
            for count in [1, 2, 1, 1, 1, 1, 1, 1, 1, 1] {
                append(position ..< position + count, to: batch)
                append(position ..< position + count, to: reference)
                position += count
                let extracted = batch.toSingle()
                try expectHistory(extracted, reference)
                #expect(extracted.metaState[2] == "3")
                #expect(extracted.metaState[5] == "requested")
                #expect(batch.batchOffsets.asArray(Int.self) == [position])

                // Multi-token continuation exposes a restored native write pointer
                // that incorrectly treats chronological over-long storage as a ring.
                let continued = try #require(reference.copy() as? RotatingKVCache)
                append(position ..< position + 2, to: extracted)
                append(position ..< position + 2, to: continued)
                try expectHistory(extracted, continued)
            }
            #expect(source.metaState == originalMetadata)
            #expect(source.state.map { $0.asArray(Float.self) } == original)
        }
    }

    @Test(
        "Conversion and extraction preserve allocation step even before first write",
        arguments: [false, true])
    func allocationStep(populated: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let source = RotatingKVCache(maxSize: 8, step: 3)
            if populated { append(0 ..< 3, to: source) }
            let batch = BatchRotatingKVCache.fromSingle(source)
            #expect(batch.step == 3)
            let extracted = batch.toSingle()
            #expect(extracted.metaState[2] == "3")
            #expect(extracted.offset == source.offset)
        }
    }

    @Test("Extraction after concat can continue with another multi-token write")
    func extractedConcatContinuation() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = RotatingKVCache(maxSize: 4)
            append(0 ..< 4, to: source)
            let batch = BatchRotatingKVCache.fromSingle(source)
            append(4 ..< 6, to: batch)
            append(4 ..< 6, to: source)
            let extracted = batch.toSingle()
            try expectHistory(extracted, source)
            append(6 ..< 8, to: extracted)
            append(6 ..< 8, to: source)
            try expectHistory(extracted, source)
        }
    }

    @Test(
        "One-token prepared masks follow the returned chronological K/V at capacity",
        arguments: [false, true])
    func preparedMask(wrapped: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = BatchRotatingKVCache(maxSize: 4, leftPadding: [2, 0])
            let initial = MLXArray([Float(90), 91, 0, 1, 10, 11, 12, 13]).reshaped([2, 1, 4, 1])
            _ = cache.update(keys: initial, values: initial + 100)
            if wrapped {
                let token = MLXArray([Float(2), 14]).reshaped([2, 1, 1, 1])
                _ = cache.update(keys: token, values: token + 100)
            }
            cache.prepare(lengths: [1, 0], rightPadding: [0, 1])
            guard case .array(let mask) = cache.makeMask(n: 1, windowSize: 3, returnArray: true)
            else {
                Issue.record("Expected an array mask")
                return
            }
            let next = MLXArray([Float(wrapped ? 3 : 2), 99]).reshaped([2, 1, 1, 1])
            let (keys, values) = cache.update(keys: next, values: next + 100)
            let row = keys[0, 0, 0..., 0].asArray(Float.self)
            let latest = Float(wrapped ? 3 : 2)
            let expectedMask = row.map { $0 >= latest - 2 && $0 <= latest }
            #expect(mask[0, 0, 0].asArray(Bool.self) == expectedMask)
            let scores = MLX.where(mask, MLXArray(Float(0)), MLXArray(-Float.infinity))
            let actual = matmul(softmax(scores, axis: -1), values)[0].item(Float.self)
            let expected = latest - 1 + 100
            #expect(abs(actual - expected) < 0.0001)
            cache.finalize()
            #expect(cache.batchOffsets.asArray(Int.self) == [wrapped ? 4 : 3, wrapped ? 5 : 4])
            let decode = MLXArray([latest + 1, Float(wrapped ? 15 : 14)]).reshaped([2, 1, 1, 1])
            _ = cache.update(keys: decode, values: decode + 100)
            let first = try view(cache.extract(idx: 0))
            #expect(first.0.asArray(Float.self) == (wrapped ? [1, 2, 3, 4] : [0, 1, 2, 3]))
            let second = try view(cache.extract(idx: 1))
            #expect(
                second.0.asArray(Float.self) == (wrapped ? [12, 13, 14, 15] : [11, 12, 13, 14]))
        }
    }

    @Test("Converted wrapped rows extend with shorter histories, reorder, and continue")
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
            let batch = BatchRotatingKVCache.fromSingle(sources[0])
            batch.extend(other: BatchRotatingKVCache.fromSingle(sources[1]))
            batch.extend(other: BatchRotatingKVCache.fromSingle(sources[2]))
            for row in sources.indices { try expectHistory(batch.extract(idx: row), sources[row]) }
            batch.filter(batchIndices: [2, 0])
            let survivors = [sources[2], sources[0]]
            for step in 0 ..< 10 {
                let next = MLXArray([Float(203 + step), Float(11 + step)]).reshaped([2, 1, 1, 1])
                _ = batch.update(keys: next, values: next + 100)
                append(203 + step ..< 204 + step, to: survivors[0])
                append(11 + step ..< 12 + step, to: survivors[1])
                for row in survivors.indices {
                    try expectHistory(batch.extract(idx: row), survivors[row])
                }
            }
        }
    }
}

private func append(_ positions: Range<Int>, to cache: any KVCache) {
    let keys = MLXArray(positions.map(Float.init)).reshaped([1, 1, positions.count, 1])
    _ = cache.update(keys: keys, values: keys + 100)
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

private func expectLogits(_ actual: MLXArray, _ expected: MLXArray) {
    #expect(abs(actual - expected).max().item(Float.self) < 0.0001)
    #expect(
        actual.argMax(axis: -1).asArray(Int.self) == expected.argMax(axis: -1).asArray(Int.self))
}

private func expectModelHistories(_ batched: [any BatchedCache], _ singles: [[any KVCache]]) throws
{
    func state(_ cache: any KVCache) throws -> [MLXArray] {
        guard let rotating = cache as? RotatingKVCache else { return cache.state }
        let (keys, values) = try view(rotating)
        return [keys, values]
    }
    for (row, caches) in singles.enumerated() {
        for layer in batched.indices {
            let extracted = batched[layer].extractBatched(row)
            #expect(extracted.offset == caches[layer].offset)
            let actual = try state(extracted)
            let expected = try state(caches[layer])
            try #require(actual.count == expected.count)
            for (a, b) in zip(actual, expected) {
                try #require(a.shape == b.shape)
                #expect(abs(a - b).max().item(Float.self) < 0.0001)
            }
        }
    }
}
