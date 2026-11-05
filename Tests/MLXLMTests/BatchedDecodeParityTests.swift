// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

/// Batched caches must give every row the logits and the keys and values it gets on its own.
@Suite(.serialized)
struct BatchedDecodeParityTests {

    @Test("A ragged batch of Llama prompts decodes like separate sequences")
    func llama() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            try expectBatchedRowsMatchSingles(model)
            try expectMergedRowsMatchSingles(model)
        }
    }

    @Test("A ragged batch of Gemma 2 prompts decodes like separate sequences")
    func gemma2() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyGemma2()
            try expectBatchedRowsMatchSingles(model)
            try expectMergedRowsMatchSingles(model)
        }
    }
}

func tinyLlama() -> LlamaModel {
    withRandomState(MLXRandom.RandomState(seed: 7)) {
        let model = LlamaModel(
            LlamaConfiguration(
                hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64,
                attentionHeads: 4, rmsNormEps: 1e-5, vocabularySize: 64, kvHeads: 2))
        eval(model)
        return model
    }
}

/// The caches of `model`, as the single-sequence caches a batch is built from.
private func singleCaches(_ model: any LanguageModel) throws -> [KVCacheSimple] {
    try model.newCache(parameters: nil).map { cache in
        try #require(cache as? KVCacheSimple, "the parity models use plain caches")
    }
}

/// Merges fresh caches, prefills two prompts of different lengths as one right-padded batch,
/// decodes four tokens and drops a row, comparing every row's logits and extracted keys and
/// values with the same prompt run alone.
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

    // One fresh single cache per row, merged per layer into the empty batch.
    let fresh = try prompts.map { _ in try singleCaches(model) }
    let batched = fresh[0].indices.map { layer in KVCacheSimple.merge(fresh.map { $0[layer] }) }
    let caches = batched.map { $0 as any KVCache }
    batched.forEach { $0.prepare(lengths: prompts.map(\.count)) }
    let padded = prompts.flatMap { $0 + Array(repeating: 0, count: longest - $0.count) }
    let logits = model(MLXArray(padded).reshaped([prompts.count, longest]), cache: caches)
    batched.forEach { $0.finalize() }
    for (row, prompt) in prompts.enumerated() {
        let actual = logits[row ..< row + 1, (prompt.count - 1) ..< prompt.count, 0...]
        #expect(maxAbsDifference(actual, expected[row]) < 1e-4, "prefill, row \(row)")
    }
    try expectExtractedRowsMatch(batched, singles)

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
    try expectExtractedRowsMatch(batched, singles)

    batched.forEach { $0.filter(batchIndices: batchRows(1)) }
    let output = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: caches)
    let single = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: singles[1])
    #expect(maxAbsDifference(output, single) < 1e-4, "after filtering")
    try expectExtractedRowsMatch(batched, [singles[1]])
}

/// Prefills two prompts alone, merges their caches and decodes three tokens, comparing
/// every row with the same sequence continued alone.
private func expectMergedRowsMatchSingles(_ model: any LanguageModel) throws {
    let prompts: [[Int32]] = [[1, 2, 3, 4, 5, 6], [7, 8]]
    var singles: [[any KVCache]] = []
    var rows: [[KVCacheSimple]] = []
    for prompt in prompts {
        let single = try model.newCache(parameters: nil)
        eval(model(MLXArray(prompt).reshaped([1, prompt.count]), cache: single))
        singles.append(single)
        let row = try singleCaches(model)
        eval(model(MLXArray(prompt).reshaped([1, prompt.count]), cache: row))
        rows.append(row)
    }

    let batched = rows[0].indices.map { layer in KVCacheSimple.merge(rows.map { $0[layer] }) }
    #expect(batched[0].batchOffset.asArray(Int.self) == prompts.map(\.count))
    #expect(batched[0].leftPadding.asArray(Int.self) == [0, 4])
    let caches = batched.map { $0 as any KVCache }
    try expectExtractedRowsMatch(batched, singles)

    for step in 0 ..< 3 {
        let token = Int32(10 + step)
        let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: caches)
        for row in prompts.indices {
            let single = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
            #expect(
                maxAbsDifference(output[row ..< row + 1], single) < 1e-4,
                "step \(step), row \(row)")
        }
    }
    try expectExtractedRowsMatch(batched, singles)
}

/// Every row extracted from `batch` holds the offset, keys and values of its single cache.
private func expectExtractedRowsMatch(_ batch: [BatchKVCache], _ singles: [[any KVCache]])
    throws
{
    for (row, caches) in singles.enumerated() {
        try #require(caches.count == batch.count)
        for layer in batch.indices {
            let actual = batch[layer].extract(row)
            #expect(actual.offset == caches[layer].offset, "row \(row), layer \(layer)")
            let expected = caches[layer].state
            try #require(actual.state.count == expected.count)
            for (actualArray, expectedArray) in zip(actual.state, expected) {
                try #require(actualArray.shape == expectedArray.shape)
                #expect(
                    maxAbsDifference(actualArray, expectedArray) < 1e-4,
                    "row \(row), layer \(layer)")
            }
        }
    }
}
