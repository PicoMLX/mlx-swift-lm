// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Tensor-parallel sharding of the Qwen3-Next model.
///
/// One test process can only form a group of one, where nothing is split. These
/// tests cover what that reaches: the sharded code path and its bookkeeping.
/// Splitting weights across processes needs a multi-process run.
final class Qwen3NextShardingTests: XCTestCase {

    /// Gated delta and attention layers alternate; the last layer has a dense MLP
    /// and the others have experts with a shared expert.
    private let configurationJSON = """
        {
            "model_type": "qwen3_next",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 1,
            "shared_expert_intermediate_size": 64,
            "mlp_only_layers": [3],
            "moe_intermediate_size": 64,
            "rms_norm_eps": 1e-6,
            "vocab_size": 64,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "max_position_embeddings": 128,
            "norm_topk_prob": true,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "full_attention_interval": 2
        }
        """

    private enum Weights: String, CaseIterable {
        case float, float16, quantized
    }

    private func makeModel(_ weights: Weights) throws -> Qwen3NextModel {
        MLXRandom.seed(11)
        let model = Qwen3NextModel(
            try JSONDecoder().decode(
                Qwen3NextConfiguration.self, from: Data(configurationJSON.utf8)))
        switch weights {
        case .float:
            break
        case .float16:
            // float16 decode runs the compiled segments and the fused convolution.
            let half = model.parameters().flattened().map { ($0.0, $0.1.asType(.float16)) }
            try model.update(parameters: ModuleParameters.unflattened(half), verify: [])
        case .quantized:
            quantize(model: model, groupSize: 64, bits: 4)
        }
        eval(model)
        return model
    }

    /// Last-position logits after a prompt and after each of three decode steps.
    private func logits(_ model: Qwen3NextModel) throws -> [MLXArray] {
        let cache = try model.newCache(parameters: nil)
        let steps: [[Int32]] = [[3, 17, 42, 8, 29, 51], [7], [23], [60]]
        return steps.map { tokens in
            let out = model(MLXArray(tokens).reshaped(1, -1), cache: cache)[0, -1]
                .asType(.float32)
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
        let model = try makeModel(.float)
        let whole = try logits(model)

        try model.shard(group: MLXDistributed.initialize())

        XCTAssertFalse(model.model.isSharded)
        assertClose(try logits(model), whole, tolerance: 0, "unsharded")
    }

    /// With a group of one every shard is the whole weight, so the sharded layers
    /// must give the same logits. A whole float16 model decodes through the
    /// compiled segments and the sharded one does not, hence its looser bound.
    func testShardedPathReproducesTheWholeModel() throws {
        for weights in Weights.allCases {
            let model = try makeModel(weights)
            let whole = try logits(model)

            let group = try MLXDistributed.initialize()
            XCTAssertEqual(group.size, 1)
            try model.model.shard(group: group)

            XCTAssertTrue(model.model.isSharded)
            assertClose(
                try logits(model), whole, tolerance: weights == .float16 ? 1e-2 : 1e-4,
                weights.rawValue)
        }
    }

    /// A sharded model skips the compiled decode segments, which were built for
    /// whole layers.
    func testShardedModelSkipsTheCompiledDecodeSegments() throws {
        let model = try makeModel(.float16)
        try model.model.shard(group: MLXDistributed.initialize())

        _ = try logits(model)

        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 0)
    }

    func testShardingReplacesTheProjectionsWithSharedLayers() throws {
        let model = try makeModel(.quantized)
        try model.model.shard(group: MLXDistributed.initialize())

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

    /// Sharding again would split the shards a second time.
    func testRepeatedShardingIsRejected() throws {
        let model = try makeModel(.float)
        let group = try MLXDistributed.initialize()
        try model.model.shard(group: group)
        let sharded = try logits(model)

        XCTAssertThrowsError(try model.model.shard(group: group)) { error in
            XCTAssertEqual(error as? Qwen3NextShardingError, .alreadySharded)
        }
        XCTAssertTrue(model.model.isSharded)
        assertClose(try logits(model), sharded, tolerance: 0, "after the refused call")
    }

    /// A sharded projection is rebuilt from its base weights, so an unfused
    /// adapter would be dropped without a trace. The model must be refused whole.
    func testModelWithUnfusedAdapterIsRejectedAndLeftWhole() throws {
        let model = try makeModel(.float)
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
            XCTAssertEqual(error as? Qwen3NextShardingError, .unfusedAdapters)
        }
        XCTAssertFalse(model.model.isSharded)
        assertClose(try logits(model), adapted, tolerance: 0, "after the refused call")
    }
}
