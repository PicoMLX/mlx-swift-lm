// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Tensor-parallel sharding of the Llama model, which also backs `mistral`.
///
/// One test process can only form a group of one, where every shard is the whole weight:
/// these cover the sharded code path and its checks.
final class LlamaShardingTests: XCTestCase {

    /// Biases on every projection, so the sharded layers must split or keep each.
    private func makeModel(quantized: Bool) throws -> LlamaModel {
        MLXRandom.seed(11)
        let model = LlamaModel(
            try JSONDecoder().decode(
                LlamaConfiguration.self,
                from: Data(
                    """
                    {
                        "model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
                        "intermediate_size": 128, "num_attention_heads": 4,
                        "num_key_value_heads": 2, "rms_norm_eps": 1e-5, "vocab_size": 64,
                        "attention_bias": true, "mlp_bias": true
                    }
                    """.utf8)))
        if quantized {
            quantize(model: model, groupSize: 64, bits: 4)
        }
        eval(model)
        return model
    }

    /// Last-position logits after a prompt and after each of three decode steps.
    private func logits(_ model: LlamaModel) throws -> [MLXArray] {
        let cache = try model.newCache(parameters: nil)
        return [[3, 17, 42, 8, 29, 51], [7], [23], [60]].map { (tokens: [Int32]) in
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
                abs(a - e).max().item(Float.self), tolerance, "\(label), step \(step)",
                file: file, line: line)
        }
    }

    func testShardedPathReproducesTheWholeModel() throws {
        for quantized in [false, true] {
            let model = try makeModel(quantized: quantized)
            let whole = try logits(model)

            try model.shard(group: MLXDistributed.initialize())

            assertClose(
                try logits(model), whole, tolerance: 1e-4, quantized ? "4-bit" : "float")
        }
    }

    func testShardingReplacesTheProjectionsWithShardedLayers() throws {
        let model = try makeModel(quantized: true)
        try model.shard(group: MLXDistributed.initialize())

        XCTAssertEqual(model.model.layers.count, 2)
        for layer in model.model.layers {
            XCTAssertTrue(layer.attention.wq is QuantizedAllToShardedLinear)
            XCTAssertTrue(layer.attention.wk is QuantizedAllToShardedLinear)
            XCTAssertTrue(layer.attention.wv is QuantizedAllToShardedLinear)
            XCTAssertTrue(layer.attention.wo is QuantizedShardedToAllLinear)
            XCTAssertTrue(layer.mlp.gate is QuantizedAllToShardedLinear)
            XCTAssertTrue(layer.mlp.up is QuantizedAllToShardedLinear)
            XCTAssertTrue(layer.mlp.down is QuantizedShardedToAllLinear)
        }
    }

    /// Sharding again would split the shards a second time.
    func testRepeatedShardingIsRejected() throws {
        let model = try makeModel(quantized: false)
        let group = try MLXDistributed.initialize()
        try model.shard(group: group)
        let sharded = try logits(model)

        XCTAssertThrowsError(try model.shard(group: group)) { error in
            XCTAssertEqual(error as? TensorParallelError, .alreadySharded)
        }
        assertClose(try logits(model), sharded, tolerance: 0, "after the refused call")
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
        let adapted = try logits(model)

        XCTAssertThrowsError(try model.shard(group: MLXDistributed.initialize())) { error in
            XCTAssertEqual(error as? TensorParallelError, .unfusedAdapters)
        }
        assertClose(try logits(model), adapted, tolerance: 0, "after the refused call")
    }
}
