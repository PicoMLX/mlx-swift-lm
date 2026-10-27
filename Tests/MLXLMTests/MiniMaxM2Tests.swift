// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class MiniMaxM2Tests: XCTestCase {

    private let configuration = Data(
        """
        {
            "model_type": "minimax_m2",
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 1024,
            "num_experts_per_tok": 2,
            "num_local_experts": 4,
            "shared_intermediate_size": 0,
            "num_hidden_layers": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000,
            "rotary_dim": 8,
            "vocab_size": 64,
            "tie_word_embeddings": false,
            "scoring_func": "sigmoid",
            "use_qk_norm": true
        }
        """.utf8)

    /// MiniMax M2 checkpoints name their model type `minimax_m2`.
    func testMiniMaxM2ModelTypeCreatesTheMiniMaxModel() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configuration, modelType: "minimax_m2")
        XCTAssertTrue(model is MiniMaxModel)
    }

    /// The official M2 checkpoints store E4M3 FP8 weights with 128 x 128 block scales.
    func testSanitizeDecodesFP8WeightsBeforeScaling() async throws {
        let created = try await LLMTypeRegistry.shared.createModel(
            configuration: configuration, modelType: "minimax_m2")
        let model = try XCTUnwrap(created as? MiniMaxModel)

        // 130 rows span two row blocks, scaled by 2 and by 4.
        let values = MLXArray((0 ..< 130).flatMap { _ in [Float(1.5), -0.25] }).reshaped([130, 2])
        let key = "model.layers.0.self_attn.q_proj.weight"
        let weights = model.sanitize(weights: [
            key: toFP8(values),
            key + "_scale_inv": MLXArray([Float(2), 4]).reshaped([2, 1]),
        ])

        let decoded = try XCTUnwrap(weights[key])
        let expected = (0 ..< 130).flatMap { row -> [Float] in row < 128 ? [3, -0.5] : [6, -1] }
        XCTAssertEqual(decoded.asType(.float32).asArray(Float.self), expected)
        XCTAssertEqual(decoded.dtype, .bfloat16)
    }
}
