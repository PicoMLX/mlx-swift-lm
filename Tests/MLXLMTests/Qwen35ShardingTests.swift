// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Tensor-parallel sharding of the Qwen 3.5 text model: gated delta, full attention,
/// dense MLP and mixture-of-experts layers.
final class Qwen35ShardingTests: XCTestCase {

    private func makeModel(quantized: Bool, experts: Int) throws -> Qwen35TextModel {
        try makeShardingModel(
            """
            {
                "model_type": "qwen3_5_text", "hidden_size": 64, "num_hidden_layers": 4,
                "intermediate_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 32, "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4, "vocab_size": 64, "full_attention_interval": 2,
                "num_experts": \(experts), "num_experts_per_tok": \(min(experts, 2)),
                "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64
            }
            """, configuration: Qwen35TextConfiguration.self, quantized: quantized,
            Qwen35TextModel.init)
    }

    func testShardingKeepsADenseModel() throws {
        try assertShardingKeepsTheModel { try makeModel(quantized: $0, experts: 0) }
    }

    func testShardingKeepsAMixtureOfExpertsModel() throws {
        try assertShardingKeepsTheModel { try makeModel(quantized: $0, experts: 4) }
    }

    func testShardingReplacesTheProjectionsWithShardedLayers() throws {
        let model = try makeModel(quantized: true, experts: 4)
        try model.shard(group: MLXDistributed.initialize())

        for layer in model.model.layers {
            if let linear = layer.linearAttn {
                XCTAssertTrue(linear.inProjQKV is QuantizedAllToShardedLinear)
                XCTAssertTrue(linear.inProjZ is QuantizedAllToShardedLinear)
                XCTAssertTrue(linear.outProj is QuantizedShardedToAllLinear)
            }
            if let attention = layer.selfAttn {
                XCTAssertTrue(attention.qProj is QuantizedAllToShardedLinear)
                XCTAssertTrue(attention.oProj is QuantizedShardedToAllLinear)
            }
            let moe = try XCTUnwrap(layer.mlp as? Qwen35SparseMoeBlock)
            XCTAssertNotNil(moe.shardingGroup)
        }
    }

    /// A sharded projection is rebuilt from its base weights, so an unfused adapter would
    /// be dropped without a trace. The model must be refused whole.
    func testModelWithUnfusedAdapterIsRejectedAndLeftWhole() throws {
        let model = try makeModel(quantized: false, experts: 0)
        let rank = 4
        let adapter = LoRAContainer(
            configuration: .init(
                numLayers: 4,
                loraParameters: .init(rank: rank, scale: 1.0, keys: ["self_attn.q_proj"])),
            parameters: ModuleParameters.unflattened([
                "model.layers.1.self_attn.q_proj.lora_a": MLXRandom.normal([64, rank]),
                "model.layers.1.self_attn.q_proj.lora_b": MLXRandom.normal([rank, 128]),
                "model.layers.3.self_attn.q_proj.lora_a": MLXRandom.normal([64, rank]),
                "model.layers.3.self_attn.q_proj.lora_b": MLXRandom.normal([rank, 128]),
            ]))
        try adapter.load(into: model)
        let adapted = try shardingLogits(model)

        XCTAssertThrowsError(try model.shard(group: MLXDistributed.initialize())) { error in
            XCTAssertEqual(error as? TensorParallelError, .unfusedAdapters)
        }
        XCTAssertFalse(model.model.isSharded)
        assertClose(try shardingLogits(model), adapted, tolerance: 0, "after the refused call")
    }
}
