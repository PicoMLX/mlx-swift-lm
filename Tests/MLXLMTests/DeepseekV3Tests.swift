import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class DeepseekV3Tests: XCTestCase {

    // Tiny MoE config: 4 routed experts in 2 groups, top-1 group, top-2 experts.
    private func makeConfig(numHiddenLayers: Int = 2) throws -> DeepseekV3Configuration {
        let json = """
            {
                "model_type": "deepseek_v3",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": \(numHiddenLayers),
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "n_group": 2,
                "topk_group": 1,
                "norm_topk_prob": true,
                "routed_scaling_factor": 1.0,
                "first_k_dense_replace": 1,
                "moe_layer_freq": 1,
                "q_lora_rank": 4,
                "kv_lora_rank": 4,
                "qk_rope_head_dim": 2,
                "qk_nope_head_dim": 2,
                "v_head_dim": 4,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "max_position_embeddings": 128,
                "attention_bias": false
            }
            """
        return try JSONDecoder().decode(DeepseekV3Configuration.self, from: Data(json.utf8))
    }

    /// Attention must update the KV cache exactly once per forward pass.
    ///
    /// The attention path builds `keys`/`values` and hands them to
    /// `attentionWithCacheUpdate`, which performs the cache update itself. An
    /// additional explicit `cache.update(...)` beforehand appended the returned
    /// history a second time, so a prompt of L tokens left the cache at 2L and
    /// every subsequent token attended over duplicated keys.
    func testForwardPassUpdatesCacheExactlyOnce() throws {
        let model = DeepseekV3Model(try makeConfig())
        let cache = try model.newCache(parameters: nil)
        let tokenCount = 5
        let inputs = MLXArray(Array(Int32(1) ... Int32(tokenCount))).reshaped(1, tokenCount)

        let logits = model(inputs, cache: cache)
        eval(logits)

        for (i, layerCache) in cache.enumerated() {
            XCTAssertEqual(
                layerCache.offset, tokenCount,
                "layer \(i) cache advanced by \(layerCache.offset) for \(tokenCount) tokens")
        }
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = DeepseekV3Model(try makeConfig())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testSanitizeDropsPredictionLayerAfterDeepseekV3Stack() throws {
        let model = DeepseekV3Model(try makeConfig(numHiddenLayers: 61))
        let sanitized = model.sanitize(weights: [
            "model.layers.60.self_attn.q_proj.weight": MLXArray.zeros([1]),
            "model.layers.61.self_attn.q_proj.weight": MLXArray.zeros([1]),
        ])

        XCTAssertNotNil(sanitized["model.layers.60.self_attn.q_proj.weight"])
        XCTAssertNil(sanitized["model.layers.61.self_attn.q_proj.weight"])
    }

    func testSanitizeUsesConfiguredLayerCount() throws {
        let model = DeepseekV3Model(try makeConfig(numHiddenLayers: 78))
        let sanitized = model.sanitize(weights: [
            "model.layers.61.self_attn.q_proj.weight": MLXArray.zeros([1]),
            "model.layers.77.self_attn.q_proj.weight": MLXArray.zeros([1]),
            "model.layers.78.self_attn.q_proj.weight": MLXArray.zeros([1]),
        ])

        XCTAssertNotNil(sanitized["model.layers.61.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["model.layers.77.self_attn.q_proj.weight"])
        XCTAssertNil(sanitized["model.layers.78.self_attn.q_proj.weight"])
    }

    /// The official checkpoints store E4M3 FP8 weights with 128 x 128 block scales.
    func testSanitizeDecodesFP8WeightsBeforeScaling() throws {
        let model = DeepseekV3Model(try makeConfig())
        // 130 rows span two row blocks, scaled by 2 and by 4.
        let values = MLXArray((0 ..< 130).flatMap { _ in [Float(1.5), -0.25] }).reshaped([130, 2])
        let key = "model.layers.0.self_attn.q_a_proj.weight"
        let sanitized = model.sanitize(weights: [
            key: toFP8(values),
            key + "_scale_inv": MLXArray([Float(2), 4]).reshaped([2, 1]),
        ])

        let decoded = try XCTUnwrap(sanitized[key])
        let expected = (0 ..< 130).flatMap { row -> [Float] in row < 128 ? [3, -0.5] : [6, -1] }
        XCTAssertEqual(decoded.asType(.float32).asArray(Float.self), expected)
        XCTAssertEqual(decoded.dtype, .bfloat16)
        XCTAssertNil(sanitized[key + "_scale_inv"])
    }

    /// Experts are stacked from their decoded weights, and their per-expert keys removed.
    func testSanitizeStacksDecodedFP8Experts() throws {
        let model = DeepseekV3Model(try makeConfig())
        let values = MLXArray([Float(1.5), -0.25, 0.5, 2]).reshaped([2, 2])
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 4 {
            let key = "model.layers.1.mlp.experts.\(expert).gate_proj.weight"
            weights[key] = toFP8(values)
            weights[key + "_scale_inv"] = MLXArray([Float(expert + 1)]).reshaped([1, 1])
        }
        let sanitized = model.sanitize(weights: weights)

        let stacked = try XCTUnwrap(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"])
        let expected = (1 ... 4).flatMap { scale in [1.5, -0.25, 0.5, 2].map { $0 * Float(scale) } }
        XCTAssertEqual(stacked.asType(.float32).asArray(Float.self), expected)
        XCTAssertEqual(stacked.dtype, .bfloat16)
        XCTAssertEqual(sanitized.keys.filter { $0.contains(".experts.") }, [])
    }
}
