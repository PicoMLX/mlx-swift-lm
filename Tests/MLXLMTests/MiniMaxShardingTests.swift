// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Tensor-parallel sharding of MiniMax models, whose query and key norms span all heads.
final class MiniMaxShardingTests: XCTestCase {

    private func makeModel(quantized: Bool) throws -> MiniMaxModel {
        try makeShardingModel(
            """
            {
                "model_type": "minimax_m2", "hidden_size": 64, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                "max_position_embeddings": 1024, "num_experts_per_tok": 2,
                "num_local_experts": 4, "shared_intermediate_size": 0, "num_hidden_layers": 2,
                "rms_norm_eps": 1e-6, "rope_theta": 10000, "rotary_dim": 8, "vocab_size": 64,
                "tie_word_embeddings": false, "scoring_func": "sigmoid", "use_qk_norm": true
            }
            """, configuration: MiniMaxConfiguration.self, quantized: quantized,
            MiniMaxModel.init)
    }

    /// The sharded query and key norms compute in float32 rather than through the fused
    /// kernel, which the shared tolerance allows for.
    func testShardingKeepsTheModel() throws {
        try assertShardingKeepsTheModel(makeModel)
    }

    func testShardingReplacesTheProjectionsAndNorms() throws {
        let model = try makeModel(quantized: true)
        try model.shard(group: MLXDistributed.initialize())

        XCTAssertEqual(model.model.layers.count, 2)
        for layer in model.model.layers {
            let attention = layer.selfAttn
            XCTAssertTrue(attention.wq is QuantizedAllToShardedLinear)
            XCTAssertTrue(attention.wk is QuantizedAllToShardedLinear)
            XCTAssertTrue(attention.wv is QuantizedAllToShardedLinear)
            XCTAssertTrue(attention.wo is QuantizedShardedToAllLinear)
            XCTAssertTrue(attention.qNorm is MiniMaxShardedRMSNorm)
            XCTAssertTrue(attention.kNorm is MiniMaxShardedRMSNorm)
            XCTAssertNotNil(layer.blockSparseMoe.shardingGroup)
        }
    }

    /// A sharded projection is rebuilt from its base weights, so an unfused adapter would
    /// be dropped without a trace. The model must be refused whole.
    func testModelWithUnfusedAdapterIsRejectedAndLeftWhole() throws {
        let model = try makeModel(quantized: false)
        let rank = 4
        let adapter = LoRAContainer(
            configuration: .init(
                numLayers: 2,
                loraParameters: .init(rank: rank, scale: 1.0, keys: ["self_attn.q_proj"])),
            parameters: ModuleParameters.unflattened([
                "model.layers.0.self_attn.q_proj.lora_a": MLXRandom.normal([64, rank]),
                "model.layers.0.self_attn.q_proj.lora_b": MLXRandom.normal([rank, 64]),
                "model.layers.1.self_attn.q_proj.lora_a": MLXRandom.normal([64, rank]),
                "model.layers.1.self_attn.q_proj.lora_b": MLXRandom.normal([rank, 64]),
            ]))
        try adapter.load(into: model)
        let adapted = try shardingLogits(model)

        XCTAssertThrowsError(try model.shard(group: MLXDistributed.initialize())) { error in
            XCTAssertEqual(error as? TensorParallelError, .unfusedAdapters)
        }
        assertClose(try shardingLogits(model), adapted, tolerance: 0, "after the refused call")
    }
}
