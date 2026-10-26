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
    let output = model(MLXArray([Int32(20)]).reshaped([1, 1]), cache: caches)
    let single = model(MLXArray([Int32(20)]).reshaped([1, 1]), cache: singles[1])
    #expect(maxAbsDifference(output, single) < 1e-4, "after filtering")
}
