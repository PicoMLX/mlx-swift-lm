// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

/// Batched caches must give every row the logits it gets on its own.
@Suite(.serialized)
struct BatchedDecodeParityTests {

    @Test("A ragged batch of Llama prompts decodes like separate sequences")
    func llama() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = withRandomState(MLXRandom.RandomState(seed: 7)) {
                let model = LlamaModel(
                    LlamaConfiguration(
                        hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64,
                        attentionHeads: 4, rmsNormEps: 1e-5, vocabularySize: 64, kvHeads: 2))
                eval(model)
                return model
            }
            try expectBatchedRowsMatchSingles(model)
        }
    }

    @Test("A ragged batch of sliding-window Gemma 3 prompts decodes past its window")
    func gemma3() throws {
        try Device.withDefaultDevice(.cpu) {
            try expectBatchedRowsMatchSingles(tinyGemma3())
        }
    }

    @Test(
        "Adopted wrapped rows and a cached ragged prefill decode like separate sequences",
        arguments: [false, true])
    func gemma3AdoptedRows(overlong: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyGemma3()
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

            func batched(_ cache: any KVCache) throws -> any BatchedCache {
                if let rotating = cache as? RotatingKVCache {
                    return try #require(BatchRotatingKVCache.fromSingle(rotating))
                }
                return BatchKVCache.fromSingle(try #require(cache as? KVCacheSimple))
            }
            let batch = try singles[0].map(batched)
            for layer in batch.indices {
                batch[layer].extend(other: try batched(singles[1][layer]))
                // Row 0 continues with two tokens, row 1 with one.
                batch[layer].prepare(lengths: [2, 1])
            }
            let caches = batch.map { $0 as any KVCache }

            let first = model(MLXArray([7, 8]).reshaped([2, 1]), cache: caches)
            for (row, token) in [7, 8].enumerated() {
                let expected = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
                #expect(maxAbsDifference(first[row ..< row + 1], expected) < 1e-4)
            }
            let trailing = model(MLXArray([9, 0]).reshaped([2, 1]), cache: caches)
            let expectedTrailing = model(MLXArray([9]).reshaped([1, 1]), cache: singles[0])
            #expect(maxAbsDifference(trailing[0 ..< 1], expectedTrailing) < 1e-4)
            batch.forEach { $0.finalize() }

            for step in 0 ..< 6 {
                let token = 4 + step
                let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: caches)
                for row in 0 ..< 2 {
                    let expected = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
                    #expect(
                        maxAbsDifference(output[row ..< row + 1], expected) < 1e-4,
                        "step \(step), row \(row)")
                }
            }

            batch.forEach { $0.filter(batchIndices: batchRows(0)) }
            let continuation = MLXArray([4, 5]).reshaped([1, 2])
            #expect(
                maxAbsDifference(
                    model(continuation, cache: caches), model(continuation, cache: singles[0]))
                    < 1e-4)
            let extracted = batch.map { $0.extract(0) }
            #expect(
                maxAbsDifference(
                    model(continuation, cache: extracted), model(continuation, cache: singles[0]))
                    < 1e-4)
        }
    }
}

private func tinyGemma3() -> Gemma3TextModel {
    let configuration = Gemma3TextConfiguration(
        modelType: "gemma3_text", hiddenSize: 16, hiddenLayers: 4, intermediateSize: 32,
        attentionHeads: 2, headDim: 8, rmsNormEps: 1e-5, vocabularySize: 16, kvHeads: 1,
        ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000, ropeTraditional: false,
        queryPreAttnScalar: 8, slidingWindow: 8, slidingWindowPattern: 2,
        maxPositionEmbeddings: 128)
    return withRandomState(MLXRandom.RandomState(seed: 503)) {
        let model = Gemma3TextModel(configuration)
        eval(model)
        return model
    }
}

/// Runs two prompts of different lengths as one right-padded batch, decodes four tokens
/// and drops a row, comparing every row's logits with the same prompt run alone.
private func expectBatchedRowsMatchSingles(_ model: any LanguageModel) throws {
    let prompts: [[Int32]] = [[1, 2, 3], [4, 5, 6, 7, 8]]
    let longest = prompts.map(\.count).max()!

    var singles: [[any KVCache]] = []
    var expected: [MLXArray] = []
    for prompt in prompts {
        let cache = try model.newCache(parameters: nil)
        let logits = model(MLXArray(prompt).reshaped([1, prompt.count]), cache: cache)
        singles.append(cache)
        expected.append(logits[0..., (prompt.count - 1) ..< prompt.count, 0...])
    }

    let factories = try makeBatchedCacheFactories(for: model.newCache(parameters: nil))
    let batched = factories.map { $0([0, 0]) }
    let caches = batched.map { $0 as any KVCache }
    batched.forEach { $0.prepare(lengths: prompts.map(\.count)) }
    let padded = prompts.flatMap { $0 + Array(repeating: 0, count: longest - $0.count) }
    let logits = model(MLXArray(padded).reshaped([prompts.count, longest]), cache: caches)
    batched.forEach { $0.finalize() }
    for (row, prompt) in prompts.enumerated() {
        let actual = logits[row ..< row + 1, (prompt.count - 1) ..< prompt.count, 0...]
        #expect(maxAbsDifference(actual, expected[row]) < 1e-4, "prefill, row \(row)")
    }

    for step in 0 ..< 4 {
        let token = Int32(10 + step)
        let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: caches)
        for row in prompts.indices {
            let single = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
            #expect(
                maxAbsDifference(output[row ..< row + 1], single) < 1e-4,
                "step \(step), row \(row)")
        }
    }

    batched.forEach { $0.filter(batchIndices: batchRows(1)) }
    let output = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: caches)
    let single = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: singles[1])
    #expect(maxAbsDifference(output, single) < 1e-4, "after filtering")
}
