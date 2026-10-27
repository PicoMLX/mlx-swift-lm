// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class MiMoV2FlashTests: XCTestCase {

    /// The official checkpoints store E4M3 FP8 weights with 128 x 128 block scales.
    func testSanitizeDecodesFP8WeightsBeforeScaling() throws {
        let json = """
            {
                "model_type": "mimo_v2_flash", "num_experts_per_tok": 2,
                "hybrid_layer_pattern": [1, 0], "moe_layer_freq": [0, 0],
                "add_swa_attention_sink_bias": true, "add_full_attention_sink_bias": true,
                "sliding_window_size": 8, "vocab_size": 32, "hidden_size": 8,
                "intermediate_size": 16, "moe_intermediate_size": 8, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "topk_method": "greedy", "scoring_func": "softmax", "norm_topk_prob": true,
                "n_group": 1, "topk_group": 1, "max_position_embeddings": 128,
                "layernorm_epsilon": 1e-6, "rope_theta": 10000.0, "swa_rope_theta": 10000.0,
                "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1,
                "head_dim": 4, "v_head_dim": 4, "swa_head_dim": 4, "swa_v_head_dim": 4,
                "partial_rotary_factor": 1.0
            }
            """
        let model = MiMoV2FlashModel(
            try JSONDecoder().decode(MiMoV2FlashConfiguration.self, from: Data(json.utf8)))

        // 130 rows span two row blocks, scaled by 2 and by 4.
        let values = MLXArray((0 ..< 130).flatMap { _ in [Float(1.5), -0.25] }).reshaped([130, 2])
        let key = "model.layers.0.self_attn.q_proj.weight"
        let sanitized = model.sanitize(weights: [
            key: toFP8(values),
            key + "_scale_inv": MLXArray([Float(2), 4]).reshaped([2, 1]),
        ])

        let decoded = try XCTUnwrap(sanitized[key])
        let expected = (0 ..< 130).flatMap { row -> [Float] in row < 128 ? [3, -0.5] : [6, -1] }
        XCTAssertEqual(decoded.asType(.float32).asArray(Float.self), expected)
        XCTAssertEqual(decoded.dtype, .bfloat16)
    }
}
