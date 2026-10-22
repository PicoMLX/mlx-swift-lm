// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Tensor-parallel sharding of the Qwen 3.5 text model.
///
/// One test process can only form a group of one, where nothing is split. These
/// tests cover what that reaches: the sharded code path and its bookkeeping.
/// Splitting weights across processes needs a multi-process run.
final class Qwen35ShardingTests: XCTestCase {

    private func configuration(experts: Int = 0) throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 64,
                "full_attention_interval": 2,
                "num_experts": \(experts),
                "num_experts_per_tok": \(min(experts, 2)),
                "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64
            }
            """
        return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    private func makeModel(quantized: Bool, experts: Int = 0) throws -> Qwen35TextModel {
        MLXRandom.seed(11)
        let model = Qwen35TextModel(try configuration(experts: experts))
        if quantized {
            quantize(model: model, groupSize: 64, bits: 4)
        }
        eval(model)
        return model
    }

    /// Last-position logits after a prompt and after each of three decode steps.
    private func logits(_ model: Qwen35TextModel) throws -> [MLXArray] {
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
    /// must give the same logits. The whole model decodes through the compiled
    /// path and the sharded one does not, hence the tolerance this suite's other
    /// cross-path checks use.
    func testShardedPathReproducesTheWholeModel() throws {
        for quantized in [false, true] {
            let model = try makeModel(quantized: quantized)
            let whole = try logits(model)

            let group = try MLXDistributed.initialize()
            XCTAssertEqual(group.size, 1)
            try model.model.shard(group: group)

            XCTAssertTrue(model.model.isSharded)
            assertClose(
                try logits(model), whole, tolerance: 1e-3, quantized ? "4-bit" : "float")
        }
    }

    func testShardingReplacesTheProjectionsWithSharedLayers() throws {
        let model = try makeModel(quantized: true)
        try model.model.shard(group: MLXDistributed.initialize())

        let layers = model.model.children().flattened().compactMap { $0.1 as? Qwen35DecoderLayer }
        XCTAssertEqual(layers.count, 4)
        for layer in layers {
            let mlp = try XCTUnwrap(layer.mlp as? Qwen3NextMLP)
            XCTAssertTrue(mlp.gateProj is QuantizedAllToShardedLinear)
            XCTAssertTrue(mlp.downProj is QuantizedShardedToAllLinear)
            if let attention = layer.selfAttn {
                XCTAssertTrue(attention.qProj is QuantizedAllToShardedLinear)
                XCTAssertTrue(attention.oProj is QuantizedShardedToAllLinear)
            }
            if let linear = layer.linearAttn {
                XCTAssertTrue(linear.inProjQKV is QuantizedAllToShardedLinear)
                XCTAssertTrue(linear.outProj is QuantizedShardedToAllLinear)
                XCTAssertFalse(linear.hasFusedInputProjection)
            }
        }
    }

    /// Sharding again would split the shards a second time.
    func testRepeatedShardingIsRejected() throws {
        let model = try makeModel(quantized: false)
        let group = try MLXDistributed.initialize()
        try model.model.shard(group: group)
        let sharded = try logits(model)

        XCTAssertThrowsError(try model.model.shard(group: group)) { error in
            XCTAssertEqual(error as? Qwen35ShardingError, .alreadySharded)
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
                numLayers: 4,
                loraParameters: .init(rank: rank, scale: 1.0, keys: ["self_attn.q_proj"])),
            parameters: ModuleParameters.unflattened([
                "model.layers.1.self_attn.q_proj.lora_a": MLXRandom.normal([64, rank]),
                "model.layers.1.self_attn.q_proj.lora_b": MLXRandom.normal([rank, 128]),
                "model.layers.3.self_attn.q_proj.lora_a": MLXRandom.normal([64, rank]),
                "model.layers.3.self_attn.q_proj.lora_b": MLXRandom.normal([rank, 128]),
            ]))
        try adapter.load(into: model)
        let adapted = try logits(model)

        XCTAssertThrowsError(try model.model.shard(group: MLXDistributed.initialize())) { error in
            XCTAssertEqual(error as? Qwen35ShardingError, .unfusedAdapters)
        }
        XCTAssertFalse(model.model.isSharded)
        assertClose(try logits(model), adapted, tolerance: 0, "after the refused call")
    }

    /// The expert blocks shard too: in a group of one every expert keeps its
    /// whole width, so the sharded blocks must give the whole model's logits.
    func testShardedMixtureOfExpertsReproducesTheWholeModel() throws {
        for quantized in [false, true] {
            let label = quantized ? "4-bit MoE" : "float MoE"
            let model = try makeModel(quantized: quantized, experts: 4)
            let whole = try logits(model)

            let group = try MLXDistributed.initialize()
            try model.model.shard(group: group)

            XCTAssertTrue(model.model.isSharded, label)
            assertClose(try logits(model), whole, tolerance: 1e-3, label)
            // A second pass would split every expert again.
            XCTAssertThrowsError(try model.model.shard(group: group), label) { error in
                XCTAssertEqual(error as? Qwen35ShardingError, .alreadySharded, label)
            }
        }
    }
}
