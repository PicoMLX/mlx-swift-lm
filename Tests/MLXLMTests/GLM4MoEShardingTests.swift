// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Tensor-parallel sharding of the GLM-4 MoE model.
///
/// One test process can only form a group of one, where nothing is split. These
/// tests cover what that reaches: the sharded code path and its bookkeeping.
/// Splitting weights across processes needs a multi-process run.
final class GLM4MoEShardingTests: XCTestCase {

    /// One dense layer, then two expert layers with a shared expert.
    private let configurationJSON = """
        {
            "model_type": "glm4_moe",
            "vocab_size": 64,
            "hidden_size": 64,
            "intermediate_size": 128,
            "max_position_embeddings": 1024,
            "moe_intermediate_size": 64,
            "norm_topk_prob": true,
            "num_attention_heads": 4,
            "n_group": 1,
            "head_dim": 16,
            "topk_group": 1,
            "n_shared_experts": 1,
            "n_routed_experts": 4,
            "routed_scaling_factor": 1.0,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 1,
            "num_hidden_layers": 3,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000,
            "use_qk_norm": true,
            "tie_word_embeddings": false,
            "attention_bias": true,
            "partial_rotary_factor": 0.5
        }
        """

    private func makeModel(quantized: Bool) throws -> GLM4MoEModel {
        MLXRandom.seed(11)
        let model = GLM4MoEModel(
            try JSONDecoder().decode(
                GLM4MoEConfiguration.self, from: Data(configurationJSON.utf8)))
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

    /// Last-position logits after a prompt and after each of three decode steps.
    private func logits(_ model: GLM4MoEModel) throws -> [MLXArray] {
        let cache = try model.newCache(parameters: nil)
        let steps: [[Int32]] = [[3, 17, 42, 8, 29, 51], [7], [23], [60]]
        return steps.map { tokens in
            let out = model(MLXArray(tokens).reshaped(1, -1), cache: cache)[0, -1]
            eval(out)
            return out
        }
    }

    private func assertClose(
        _ actual: [MLXArray], _ expected: [MLXArray], tolerance: Float, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (step, (a, e)) in zip(actual, expected).enumerated() {
            XCTAssertLessThanOrEqual(
                abs(a - e).max().item(Float.self), tolerance,
                "\(label), step \(step)", file: file, line: line)
        }
    }

    func testGroupOfOneLeavesTheModelWhole() throws {
        let model = try makeModel(quantized: false)
        let whole = try logits(model)

        try model.shard(group: MLXDistributed.initialize())

        XCTAssertFalse(model.model.isSharded)
        assertClose(try logits(model), whole, tolerance: 0, "unsharded")
    }

    /// With a group of one every shard is the whole weight, so the sharded layers
    /// must give the same logits.
    func testShardedPathReproducesTheWholeModel() throws {
        for quantized in [false, true] {
            let model = try makeModel(quantized: quantized)
            let whole = try logits(model)

            let group = try MLXDistributed.initialize()
            XCTAssertEqual(group.size, 1)
            try model.model.shard(group: group)

            XCTAssertTrue(model.model.isSharded)
            assertClose(
                try logits(model), whole, tolerance: 1e-4, quantized ? "4-bit" : "float")
        }
    }

    func testShardingReplacesTheProjectionsInDenseAndExpertLayers() throws {
        let model = try makeModel(quantized: true)
        try model.model.shard(group: MLXDistributed.initialize())

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

    /// Sharding again would split the shards a second time.
    func testRepeatedShardingIsRejected() throws {
        let model = try makeModel(quantized: false)
        let group = try MLXDistributed.initialize()
        try model.model.shard(group: group)
        let sharded = try logits(model)

        XCTAssertThrowsError(try model.model.shard(group: group)) { error in
            XCTAssertEqual(error as? GLM4MoEShardingError, .alreadySharded)
        }
        XCTAssertTrue(model.model.isSharded)
        assertClose(try logits(model), sharded, tolerance: 0, "after the refused call")
    }

    /// A sharded projection is rebuilt from its base weights, so an unfused
    /// adapter would be dropped without a trace. The model must be refused whole.
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
        let adapted = try logits(model)

        XCTAssertThrowsError(try model.model.shard(group: MLXDistributed.initialize())) { error in
            XCTAssertEqual(error as? GLM4MoEShardingError, .unfusedAdapters)
        }
        XCTAssertFalse(model.model.isSharded)
        assertClose(try logits(model), adapted, tolerance: 0, "after the refused call")
    }
}
