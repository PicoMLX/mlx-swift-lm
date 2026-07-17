// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Pins the M-13 fix: per-row llama-4 attention scaling on the batched path
/// (Libraries/MLXLLM/Models/Mistral3Text.swift).
///
/// Mistral3's llama-4 scaling multiplies queries by
/// `1 + beta * log(1 + floor(position / original_max_position_embeddings))`.
/// Before the fix, a ragged batch computed that scale from the scalar
/// `cache[0].offset` — the LONGEST row's absolute position — so a short row
/// decoding at position 4 was scaled as if it sat at position 24 once any
/// batch-mate passed `original_max_position_embeddings`. The fix reads each
/// row's own position from the batched cache (`BatchPositionedKVCache
/// .batchOffset`).
///
/// Method (mirroring the regression suite's merge pattern): prefill each row
/// single-stream, merge the per-layer caches into hand-built batch caches
/// (`BatchKVCache` / `BatchRotatingKVCache`, so BOTH cache types appear), then
/// greedy-decode two steps batched and compare each row against the same model
/// run single-stream. With `original_max_position_embeddings = 8`, the 24-token
/// row decodes at positions ≥ 24 (scale > 1) while the 4-token row must keep
/// scale exactly 1 — under the scalar-offset bug the short row diverges.
///
/// No Metal-availability guard, matching BatchModelRegressionTests: the real
/// gate is the macOS/Metal CI, which runs these end to end.
@Suite("Mistral3 batch parity", .serialized)
struct Mistral3BatchParityTests {

    // 24 vs 4 tokens: the long row's positions pass
    // original_max_position_embeddings (8) during prefill AND decode, so its
    // llama-4 scale exceeds 1 while the short row's must stay exactly 1.
    private let longPrompt: [Int32] = [
        3, 9, 27, 17, 40, 5, 61, 33, 12, 48, 22, 7,
        51, 30, 2, 44, 19, 36, 58, 10, 25, 41, 6, 14,
    ]
    private let shortPrompt: [Int32] = [7, 11, 13, 17]

    /// Shared fixture config. `rope_parameters` uses Mistral3Text's exact
    /// decoding keys (`llama_4_scaling_beta`, `original_max_position_embeddings`,
    /// flat `[String: StringOrNumber]` dict), and `layer_types` mixes a
    /// full-attention layer (→ `KVCacheSimple`/`BatchKVCache`) with a
    /// sliding-attention layer (→ `RotatingKVCache`/`BatchRotatingKVCache`).
    /// `sliding_window` (32) exceeds prompt+decode length so the rotating
    /// caches never wrap (BatchRotatingKVCache.merge rejects wrapped sources).
    private static let configurationJSON = """
        {
          "model_type": "ministral3",
          "hidden_size": 16,
          "num_hidden_layers": 2,
          "intermediate_size": 32,
          "num_attention_heads": 4,
          "head_dim": 4,
          "rms_norm_eps": 0.000001,
          "vocab_size": 64,
          "num_key_value_heads": 2,
          "max_position_embeddings": 128,
          "rope_theta": 10000.0,
          "tie_word_embeddings": false,
          "layer_types": ["full_attention", "sliding_attention"],
          "sliding_window": 32,
          "rope_parameters": {
            "rope_theta": 10000.0,
            "llama_4_scaling_beta": 0.75,
            "original_max_position_embeddings": 8
          }
        }
        """

    private func makeModel(seed: UInt64) throws -> Mistral3TextModel {
        let configuration: Mistral3TextConfiguration = try decodeConfiguration(
            Self.configurationJSON)

        let model = withRandomState(MLXRandom.RandomState(seed: seed)) {
            Mistral3TextModel(configuration)
        }
        // Run in float16 (real-world deployment dtype). On hardware with a
        // NAX, float32 batched matmuls silently run in tf32 while single-
        // stream runs full float32, which skews batch-vs-single comparisons;
        // float16 uses the same kernels on both paths (see
        // SpeculativeDecodingTests for the same move).
        model.apply {
            if $0.dtype == .float32 {
                $0.asType(.float16)
            } else {
                $0
            }
        }
        eval(model)
        return model
    }

    // MARK: - Config guard

    /// Guards the parity tests against becoming vacuous: if the rope keys ever
    /// drifted, `ropeParameters` would silently decode without llama-4 scaling
    /// and the batched scale path under test would never engage.
    @Test("Fixture config parses llama-4 scaling and mixed layer types")
    func fixtureParsesLlama4Scaling() throws {
        let configuration: Mistral3TextConfiguration = try decodeConfiguration(
            Self.configurationJSON)

        let rope = try #require(configuration.ropeParameters)
        #expect(rope["llama_4_scaling_beta"]?.asFloat() == 0.75)
        #expect(rope["original_max_position_embeddings"]?.asInt() == 8)
        #expect(configuration.layerTypes == ["full_attention", "sliding_attention"])
        #expect(configuration.slidingWindow == 32)

        // Both cache topologies must actually appear, and the first layer's
        // cache must be the one the model's `cache?.first as?
        // BatchPositionedKVCache` check sees after migration.
        let model = try makeModel(seed: 299)
        let caches = model.newCache(parameters: nil)
        #expect(caches.count == 2)
        #expect(caches[0] is KVCacheSimple)
        #expect(caches[1] is RotatingKVCache)
    }

    // MARK: - Ragged 2-row parity

    @Test("Ragged 2-row batched decode matches single-stream rows")
    func raggedBatchedDecodeMatchesSingle() throws {
        let model = try makeModel(seed: 300)
        let prompts = [longPrompt, shortPrompt]

        // Single-stream prefill per row: the per-row baseline AND the source
        // the batch caches are merged from.
        var singleCaches: [[KVCache]] = []
        var currentTokens: [Int32] = []
        for prompt in prompts {
            let (logits, cache) = prefillSingle(model: model, prompt: prompt)
            let last = logits[0..., (prompt.count - 1)..., 0...].asType(.float32)
            #expect(
                abs(last).max().item(Float.self).isFinite,
                "Single prefill produced non-finite logits for a \(prompt.count)-token prompt"
            )
            singleCaches.append(cache)
            currentTokens.append(argmaxToken(last))
        }

        let batchCaches = mergeCaches(rows: singleCaches)

        // Greedy decode two steps. The chain is driven by the single-stream
        // argmax so both paths always see identical inputs; each step then
        // compares the batched row's logits (and, margin permitting, argmax)
        // against the single-stream run.
        for step in 0 ..< 2 {
            let batchedInput = MLXArray(currentTokens, [prompts.count, 1])
            let batchedLogits = model.callAsFunction(batchedInput, cache: batchCaches)
            materialize(arrays: [batchedLogits], cache: batchCaches)

            var nextTokens: [Int32] = []
            for row in prompts.indices {
                let input = MLXArray([currentTokens[row]])[.newAxis, .ellipsis]
                let singleLogits = model.callAsFunction(input, cache: singleCaches[row])
                materialize(arrays: [singleLogits], cache: singleCaches[row])

                let batchRow = batchedLogits[row ..< (row + 1), 0..., 0...].asType(.float32)
                let single = singleLogits.asType(.float32)

                #expect(batchRow.shape == single.shape)
                // Under the scalar-offset bug the short row's queries are
                // scaled by 1 + 0.75 * log(4) ≈ 2.04 instead of 1.0 at this
                // step, which moves its logits far beyond this tolerance.
                #expect(
                    maxAbsDifference(batchRow, single) <= 0.02,
                    "Decode step \(step) diverged for row \(row) (prompt length \(prompts[row].count))"
                )

                // Argmax parity, gated on a clear top-2 margin so a numeric
                // near-tie cannot flip the winner and flake the test.
                let singleTop = argmaxToken(single)
                if topTwoGap(single) > 0.05 {
                    #expect(
                        argmaxToken(batchRow) == singleTop,
                        "Decode step \(step) picked a different token for row \(row)"
                    )
                }
                nextTokens.append(singleTop)
            }
            currentTokens = nextTokens
        }
    }

    // MARK: - Batch-of-1 parity

    @Test("Batch-of-1 decode matches single-stream")
    func batchOfOneMatchesSingle() throws {
        let model = try makeModel(seed: 301)

        // Use the long prompt so decode positions (≥ 24) are past
        // original_max_position_embeddings and the scaling path is live.
        let (logits, singleCache) = prefillSingle(model: model, prompt: longPrompt)
        let batchCaches = mergeCaches(rows: [singleCache])

        var currentToken = argmaxToken(
            logits[0..., (longPrompt.count - 1)..., 0...].asType(.float32))

        for step in 0 ..< 2 {
            let batchedInput = MLXArray([currentToken], [1, 1])
            let batchedLogits = model.callAsFunction(batchedInput, cache: batchCaches)
            materialize(arrays: [batchedLogits], cache: batchCaches)

            let input = MLXArray([currentToken])[.newAxis, .ellipsis]
            let singleLogits = model.callAsFunction(input, cache: singleCache)
            materialize(arrays: [singleLogits], cache: singleCache)

            let batchRow = batchedLogits.asType(.float32)
            let single = singleLogits.asType(.float32)

            #expect(batchRow.shape == single.shape)
            #expect(
                maxAbsDifference(batchRow, single) <= 0.02,
                "Batch-of-1 decode step \(step) diverged from single-stream"
            )

            let singleTop = argmaxToken(single)
            if topTwoGap(single) > 0.05 {
                #expect(argmaxToken(batchRow) == singleTop)
            }
            currentToken = singleTop
        }
    }

    // MARK: - Helpers

    private func decodeConfiguration<T: Decodable>(_ json: String) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(json.utf8))
    }

    private func prefillSingle(
        model: Mistral3TextModel, prompt: [Int32]
    ) -> (logits: MLXArray, cache: [KVCache]) {
        let cache = model.newCache(parameters: nil)
        let input = MLXArray(prompt)[.newAxis, .ellipsis]
        let logits = model.callAsFunction(input, cache: cache)
        materialize(arrays: [logits], cache: cache)
        return (logits, cache)
    }

    /// Merge per-row single caches into one batched cache per layer,
    /// right-justified with per-row left padding — `BatchKVCache.merge` for
    /// full-attention layers and `BatchRotatingKVCache.merge` for
    /// sliding-attention layers. After the merge, each row's `batchOffset`
    /// carries ITS absolute position (24 vs 4), which is exactly what the
    /// per-row llama-4 scale must consume.
    private func mergeCaches(rows: [[KVCache]]) -> [KVCache] {
        guard let layerCount = rows.first?.count else { return [] }
        return (0 ..< layerCount).map { layer in
            let layerCaches = rows.map { $0[layer] }
            if layerCaches.allSatisfy({ $0 is RotatingKVCache }) {
                return BatchRotatingKVCache.merge(layerCaches)
            } else {
                return BatchKVCache.merge(layerCaches)
            }
        }
    }

    private func argmaxToken(_ logits: MLXArray) -> Int32 {
        Int32(logits.argMax(axis: -1).item(Int.self))
    }

    /// Margin between the best and second-best logit of a `[1, 1, V]` row.
    private func topTwoGap(_ logits: MLXArray) -> Float {
        let flat = logits.reshaped([-1]).asType(.float32)
        let top = flat.max()
        let second = MLX.where(flat .>= top, MLXArray(Float(-1e9)), flat).max()
        return (top - second).item(Float.self)
    }

    private func materialize(arrays: [MLXArray], cache: [KVCache]) {
        if !arrays.isEmpty {
            eval(arrays)
        }
        let cacheState = cache.flatMap { $0.state }
        if !cacheState.isEmpty {
            eval(cacheState)
        }
    }

    private func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs - rhs).max().item(Float.self)
    }
}
