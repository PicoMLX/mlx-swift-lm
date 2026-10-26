// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Tensor-parallel sharding of the Qwen3-Next model.
final class Qwen3NextShardingTests: XCTestCase {

    /// Gated delta and attention layers alternate; the last layer has a dense MLP and the
    /// others have experts with a shared expert.
    private let configurationJSON = """
        {
            "model_type": "qwen3_next", "hidden_size": 64, "num_hidden_layers": 4,
            "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "linear_num_value_heads": 4, "linear_num_key_heads": 2,
            "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
            "num_experts": 4, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
            "shared_expert_intermediate_size": 64, "mlp_only_layers": [3],
            "moe_intermediate_size": 64, "rms_norm_eps": 1e-6, "vocab_size": 64,
            "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
            "max_position_embeddings": 128, "norm_topk_prob": true,
            "tie_word_embeddings": false, "attention_bias": false, "full_attention_interval": 2
        }
        """

    private func makeModel(quantized: Bool) throws -> Qwen3NextModel {
        try makeShardingModel(
            configurationJSON, configuration: Qwen3NextConfiguration.self, quantized: quantized,
            Qwen3NextModel.init)
    }

    func testShardingKeepsTheModel() throws {
        try assertShardingKeepsTheModel(makeModel)
    }

    /// A whole float16 model decodes through the compiled segments, which were built for
    /// whole layers; a sharded one must not.
    func testShardedModelSkipsTheCompiledDecodeSegments() throws {
        func halfModel() throws -> Qwen3NextModel {
            let model = try makeModel(quantized: false)
            let half = model.parameters().flattened().map { ($0.0, $0.1.asType(.float16)) }
            try model.update(parameters: ModuleParameters.unflattened(half), verify: [])
            return model
        }
        let whole = try halfModel()
        let expected = try shardingLogits(whole).map { $0.asType(.float32) }
        XCTAssertGreaterThan(whole.model.compiledDecodeSegmentCount, 0)

        let sharded = try halfModel()
        try sharded.shard(group: MLXDistributed.initialize())
        let actual = try shardingLogits(sharded).map { $0.asType(.float32) }

        XCTAssertEqual(sharded.model.compiledDecodeSegmentCount, 0)
        assertClose(actual, expected, tolerance: 1e-2, "float16")
    }

    func testShardingReplacesTheProjectionsWithShardedLayers() throws {
        let model = try makeModel(quantized: true)
        try model.shard(group: MLXDistributed.initialize())

        let layers = model.model.children().flattened()
            .compactMap { $0.1 as? Qwen3NextDecoderLayer }
        XCTAssertEqual(layers.count, 4)
        for layer in layers {
            if let attention = layer.selfAttn {
                XCTAssertTrue(attention.qProj is QuantizedAllToShardedLinear)
                XCTAssertTrue(attention.oProj is QuantizedShardedToAllLinear)
            }
            if let linear = layer.linearAttn {
                XCTAssertTrue(linear.inProjQKVZ is QuantizedAllToShardedLinear)
                XCTAssertTrue(linear.inProjBA is QuantizedAllToShardedLinear)
                XCTAssertTrue(linear.outProj is QuantizedShardedToAllLinear)
            }
            if let mlp = layer.mlp as? Qwen3NextMLP {
                XCTAssertTrue(mlp.gateProj is QuantizedAllToShardedLinear)
                XCTAssertTrue(mlp.downProj is QuantizedShardedToAllLinear)
            } else {
                let moe = try XCTUnwrap(layer.mlp as? Qwen3NextSparseMoeBlock)
                XCTAssertNotNil(moe.shardingGroup)
            }
        }
    }

    /// A sharded projection is rebuilt from its base weights, so an unfused adapter would
    /// be dropped without a trace. The model must be refused whole.
    func testModelWithUnfusedAdapterIsRejectedAndLeftWhole() throws {
        let model = try makeModel(quantized: false)
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
