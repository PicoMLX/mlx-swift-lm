// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Recurrent and composite batched caches must give every row the logits it gets on its own.
///
/// The rows are prefilled one at a time and then merged: a right-padded prefill of these
/// models is not exact yet, because their state-space scans do not honour the prepared
/// lengths. The scans run as Metal kernels, so the tests stay on the default device, where a
/// one-row and a two-row forward differ by the matmul precision (see ``MatmulPrecision``).
@Suite(.serialized)
struct BatchedRecurrentParityTests {

    @Test("Merged Mamba 2 prompts decode like separate sequences")
    func mamba2() throws {
        let model = try tinyMamba2()
        try expectMergedRowsMatchSingles(model, layers: { $0 is MambaCache })
    }

    @Test("Merged Falcon H1 prompts decode like separate sequences")
    func falconH1() throws {
        let model = try tinyFalconH1()
        try expectMergedRowsMatchSingles(model, layers: { $0 is CacheList })
    }
}

private func tinyMamba2() throws -> Mamba2Model {
    let configuration = try JSONDecoder().decode(
        Mamba2Configuration.self,
        from: Data(
            """
            {
              "model_type": "mamba2", "num_heads": 4, "head_dim": 4, "vocab_size": 32,
              "hidden_size": 16, "state_size": 8, "num_hidden_layers": 2,
              "layer_norm_epsilon": 1e-5, "conv_kernel": 4, "n_groups": 1, "use_bias": false,
              "use_conv_bias": true, "tie_word_embeddings": false, "time_step_limit": [0.0, 100.0]
            }
            """.utf8))
    return withRandomState(MLXRandom.RandomState(seed: 601)) {
        let model = Mamba2Model(configuration)
        eval(model)
        return model
    }
}

private func tinyFalconH1() throws -> FalconH1Model {
    let configuration = try JSONDecoder().decode(
        FalconH1Configuration.self,
        from: Data(
            """
            {
              "model_type": "falcon_h1", "hidden_size": 32, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
              "vocab_size": 100, "mamba_d_ssm": 16, "mamba_d_state": 8, "mamba_d_head": 8,
              "mamba_n_heads": 2, "mamba_n_groups": 1, "mamba_d_conv": 4, "mamba_chunk_size": 64,
              "attention_in_multiplier": 2.0, "key_multiplier": 0.5
            }
            """.utf8))
    return withRandomState(MLXRandom.RandomState(seed: 602)) {
        let model = FalconH1Model(configuration)
        eval(model)
        return model
    }
}

/// Prefills two prompts of different lengths alone, merges the prefilled caches, decodes four
/// tokens and drops a row, comparing every row's logits with the prompt continued alone and
/// every extracted row's state with its single cache.
private func expectMergedRowsMatchSingles(
    _ model: any LanguageModel, layers: (any KVCache) -> Bool
) throws {
    let prompts: [[Int32]] = [[1, 2, 3], [4, 5, 6, 7, 8]]

    var singles: [[any KVCache]] = []
    var rows: [[any KVCache]] = []
    for prompt in prompts {
        let single = try model.newCache(parameters: nil)
        eval(model(MLXArray(prompt).reshaped([1, prompt.count]), cache: single))
        singles.append(single)
        let row = try model.newCache(parameters: nil)
        eval(model(MLXArray(prompt).reshaped([1, prompt.count]), cache: row))
        rows.append(row)
    }
    let hasExpectedLayers = singles[0].contains(where: layers)
    #expect(hasExpectedLayers)

    let batched = try rows[0].indices.map { layer in
        try mergeCaches(rows.map { $0[layer] })
    }
    for cache in batched {
        if let attention = cache as? BatchKVCache {
            #expect(attention.batchOffset.asArray(Int.self) == prompts.map(\.count))
        }
        if let list = cache as? CacheList, let attention = list[1] as? BatchKVCache {
            #expect(attention.batchOffset.asArray(Int.self) == prompts.map(\.count))
        }
    }
    try expectExtractedRowsMatch(batched, singles)

    for step in 0 ..< 4 {
        let token = Int32(10 + step)
        let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: batched)
        for row in prompts.indices {
            let single = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
            #expect(
                maxAbsDifference(output[row ..< row + 1], single) < MatmulPrecision.splitTolerance,
                "step \(step), row \(row)")
        }
    }
    try expectExtractedRowsContinueLikeSingles(model, batched, singles)

    batched.forEach { filterRows($0, keeping: batchRows(1)) }
    let output = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: batched)
    let single = model(MLXArray([Int32(14)]).reshaped([1, 1]), cache: singles[1])
    #expect(maxAbsDifference(output, single) < MatmulPrecision.splitTolerance, "after filtering")
    try expectExtractedRowsContinueLikeSingles(model, batched, [singles[1]])
}

/// Every row extracted from `batch` decodes the next token like its single cache. The
/// recurrent state arrays themselves are not compared after a decode: a one-row and a two-row
/// scan leave them in different numerical form even for copies of one row.
private func expectExtractedRowsContinueLikeSingles(
    _ model: any LanguageModel, _ batch: [any KVCache], _ singles: [[any KVCache]]
) throws {
    for (row, caches) in singles.enumerated() {
        let extracted = try batch.map { try extractRow($0, row) }
        for (layer, cache) in extracted.enumerated() where !(cache is ArraysCache) {
            #expect(cache.offset == caches[layer].offset, "row \(row), layer \(layer)")
        }
        let token = MLXArray([Int32(20 + row)]).reshaped([1, 1])
        let actual = model(token, cache: extracted)
        let expected = model(token, cache: caches.map { $0.copy() })
        #expect(
            maxAbsDifference(actual, expected) < MatmulPrecision.splitTolerance,
            "extracted row \(row)")
    }
}

private func filterRows(_ cache: any KVCache, keeping indices: MLXArray) {
    switch cache {
    case let batch as CacheList: batch.filter(batchIndices: indices)
    case let batch as ArraysCache: batch.filter(batchIndices: indices)
    case let batch as BatchKVCache: batch.filter(batchIndices: indices)
    default: Issue.record("not a batched cache: \(type(of: cache))")
    }
}

private func extractRow(_ cache: any KVCache, _ index: Int) throws -> any KVCache {
    switch cache {
    case let batch as CacheList: return batch.extract(index)
    case let batch as ArraysCache: return batch.extract(index)
    case let batch as BatchKVCache: return batch.extract(index)
    default: throw KVCacheError(message: "not a batched cache: \(type(of: cache))")
    }
}

/// Every row extracted from `batch` holds the state of its single cache, and the offset of
/// an attention cache. Recurrent state keeps one offset for the whole batch.
private func expectExtractedRowsMatch(_ batch: [any KVCache], _ singles: [[any KVCache]])
    throws
{
    for (row, caches) in singles.enumerated() {
        try #require(caches.count == batch.count)
        for layer in batch.indices {
            let actual = try extractRow(batch[layer], row)
            if !(actual is ArraysCache) {
                #expect(actual.offset == caches[layer].offset, "row \(row), layer \(layer)")
            }
            let a = actual.state
            let b = caches[layer].state
            try #require(a.count == b.count, "row \(row), layer \(layer)")
            for (actualArray, expectedArray) in zip(a, b) {
                try #require(actualArray.shape == expectedArray.shape, "row \(row), layer \(layer)")
                // Recurrent state grows with the sequence, so the bound is relative.
                let mismatch = MatmulPrecision.mismatch(
                    actualArray, expectedArray, rtol: 1e-2, atol: 1e-4, reducedAtol: 2e-2)
                #expect(mismatch == nil, "row \(row), layer \(layer): \(mismatch ?? "")")
            }
        }
    }
}
