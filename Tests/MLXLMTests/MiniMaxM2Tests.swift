// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class MiniMaxM2Tests: XCTestCase {

    /// MiniMax M2 checkpoints name their model type `minimax_m2`.
    func testMiniMaxM2ModelTypeCreatesTheMiniMaxModel() async throws {
        let json = Data(
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
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: json, modelType: "minimax_m2")
        XCTAssertTrue(model is MiniMaxModel)
    }
}
