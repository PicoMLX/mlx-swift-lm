// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Tensor-parallel sharding of the MiniMax model.
///
/// One test process can only form a group of one, where nothing is split. These
/// tests cover what that reaches: the sharded code path and its bookkeeping.
/// Splitting weights across processes needs a multi-process run.
final class MiniMaxShardingTests: XCTestCase {

    private let configurationJSON = """
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
        """

    private func makeModel(quantized: Bool) throws -> MiniMaxModel {
        MLXRandom.seed(11)
        let model = MiniMaxModel(
            try JSONDecoder().decode(
                MiniMaxConfiguration.self, from: Data(configurationJSON.utf8)))
        if quantized {
            quantize(model: model, groupSize: 64, bits: 4)
        }
        eval(model)
        return model
    }

    /// Last-position logits after a prompt and after each of three decode steps.
    private func logits(_ model: MiniMaxModel) throws -> [MLXArray] {
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

    /// MiniMax M2 checkpoints name their model type `minimax_m2`.
    func testMiniMaxM2ModelTypeCreatesTheMiniMaxModel() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(configurationJSON.utf8), modelType: "minimax_m2")
        XCTAssertTrue(model is MiniMaxModel)
        XCTAssertTrue(model is any TensorParallelModel)
    }

    func testGroupOfOneLeavesTheModelWhole() throws {
        let model = try makeModel(quantized: false)
        let whole = try logits(model)

        try model.shard(group: MLXDistributed.initialize())

        XCTAssertFalse(model.model.isSharded)
        assertClose(try logits(model), whole, tolerance: 0, "unsharded")
    }

    /// With a group of one every shard is the whole weight, so the sharded layers
    /// must give the same logits. The sharded query and key norms compute in
    /// float32 rather than through the fused kernel, hence a small tolerance.
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

    func testShardingReplacesTheProjectionsAndNorms() throws {
        let model = try makeModel(quantized: true)
        try model.model.shard(group: MLXDistributed.initialize())

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

    /// Sharding again would split the shards a second time.
    func testRepeatedShardingIsRejected() throws {
        let model = try makeModel(quantized: false)
        let group = try MLXDistributed.initialize()
        try model.model.shard(group: group)
        let sharded = try logits(model)

        XCTAssertThrowsError(try model.model.shard(group: group)) { error in
            XCTAssertEqual(error as? MiniMaxShardingError, .alreadySharded)
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
                numLayers: 2,
                loraParameters: .init(rank: rank, scale: 1.0, keys: ["self_attn.q_proj"])),
            parameters: ModuleParameters.unflattened([
                "model.layers.0.self_attn.q_proj.lora_a": MLXRandom.normal([64, rank]),
                "model.layers.0.self_attn.q_proj.lora_b": MLXRandom.normal([rank, 64]),
                "model.layers.1.self_attn.q_proj.lora_a": MLXRandom.normal([64, rank]),
                "model.layers.1.self_attn.q_proj.lora_b": MLXRandom.normal([rank, 64]),
            ]))
        try adapter.load(into: model)
        let adapted = try logits(model)

        XCTAssertThrowsError(try model.model.shard(group: MLXDistributed.initialize())) { error in
            XCTAssertEqual(error as? MiniMaxShardingError, .unfusedAdapters)
        }
        XCTAssertFalse(model.model.isSharded)
        assertClose(try logits(model), adapted, tolerance: 0, "after the refused call")
    }
}
