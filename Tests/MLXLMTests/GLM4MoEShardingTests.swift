// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Tensor-parallel sharding of GLM-4 mixture-of-experts models.
final class GLM4MoEShardingTests: XCTestCase {

    /// One dense layer, then two expert layers with a shared expert.
    private func makeModel(quantized: Bool) throws -> GLM4MoEModel {
        MLXRandom.seed(11)
        let model = GLM4MoEModel(
            try JSONDecoder().decode(
                GLM4MoEConfiguration.self,
                from: Data(
                    """
                    {
                        "model_type": "glm4_moe", "vocab_size": 64, "hidden_size": 64,
                        "intermediate_size": 128, "max_position_embeddings": 1024,
                        "moe_intermediate_size": 64, "norm_topk_prob": true,
                        "num_attention_heads": 4, "n_group": 1, "head_dim": 16, "topk_group": 1,
                        "n_shared_experts": 1, "n_routed_experts": 4,
                        "routed_scaling_factor": 1.0, "num_experts_per_tok": 2,
                        "first_k_dense_replace": 1, "num_hidden_layers": 3,
                        "num_key_value_heads": 2, "rms_norm_eps": 1e-5, "rope_theta": 10000,
                        "use_qk_norm": true, "tie_word_embeddings": false,
                        "attention_bias": true, "partial_rotary_factor": 0.5
                    }
                    """.utf8)))
        // The router starts at zero, which would send every token to the same experts.
        let routers = model.parameters().flattened()
            .filter { $0.0.hasSuffix("mlp.gate.weight") }
            .map { ($0.0, MLXRandom.normal($0.1.shape)) }
        try model.update(parameters: ModuleParameters.unflattened(routers), verify: [])
        if quantized {
            quantize(model: model, groupSize: 64, bits: 4)
        }
        eval(model)
        return model
    }

    func testShardingKeepsTheModel() throws {
        try assertShardingKeepsTheModel(makeModel)
    }

    func testShardingReplacesTheProjectionsInDenseAndExpertLayers() throws {
        let model = try makeModel(quantized: true)
        try model.shard(group: MLXDistributed.initialize())

        XCTAssertEqual(model.model.layers.count, 3)
        for layer in model.model.layers {
            XCTAssertTrue(layer.attention.wq is QuantizedAllToShardedLinear)
            XCTAssertTrue(layer.attention.wo is QuantizedShardedToAllLinear)
        }
        let dense = try XCTUnwrap(model.model.layers[0].mlp as? GLM4MoEMLP)
        XCTAssertTrue(dense.gateProj is QuantizedAllToShardedLinear)
        XCTAssertTrue(dense.downProj is QuantizedShardedToAllLinear)
        for layer in model.model.layers.dropFirst() {
            let moe = try XCTUnwrap(layer.mlp as? GLM4MoE)
            XCTAssertNotNil(moe.shardingGroup)
        }
    }

    /// A sharded projection is rebuilt from its base weights, so an unfused adapter would
    /// be dropped without a trace. The model must be refused whole.
    func testModelWithUnfusedAdapterIsRejectedAndLeftWhole() throws {
        let model = try makeModel(quantized: false)
        let rank = 4
        let adapter = LoRAContainer(
            configuration: .init(
                numLayers: 3,
                loraParameters: .init(rank: rank, scale: 1.0, keys: ["self_attn.q_proj"])),
            parameters: ModuleParameters.unflattened(
                (0 ..< 3).flatMap { layer in
                    [
                        (
                            "model.layers.\(layer).self_attn.q_proj.lora_a",
                            MLXRandom.normal([64, rank])
                        ),
                        (
                            "model.layers.\(layer).self_attn.q_proj.lora_b",
                            MLXRandom.normal([rank, 64])
                        ),
                    ]
                }))
        try adapter.load(into: model)
        let adapted = try shardingLogits(model)

        XCTAssertThrowsError(try model.shard(group: MLXDistributed.initialize())) { error in
            XCTAssertEqual(error as? TensorParallelError, .unfusedAdapters)
        }
        assertClose(try shardingLogits(model), adapted, tolerance: 0, "after the refused call")
    }
}
