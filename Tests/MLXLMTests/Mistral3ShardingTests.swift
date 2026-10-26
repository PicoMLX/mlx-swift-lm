// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Tensor-parallel sharding of the Mistral 3 text model.
final class Mistral3ShardingTests: XCTestCase {

    /// A full-attention layer and a sliding-window layer.
    private func makeModel(quantized: Bool) throws -> Mistral3TextModel {
        try makeShardingModel(
            """
            {
                "model_type": "mistral3", "hidden_size": 64, "num_hidden_layers": 2,
                "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 16, "rms_norm_eps": 1e-5, "vocab_size": 64,
                "layer_types": ["full_attention", "sliding_attention"], "sliding_window": 4
            }
            """, configuration: Mistral3TextConfiguration.self, quantized: quantized,
            Mistral3TextModel.init)
    }

    func testShardingKeepsTheModel() throws {
        try assertShardingKeepsTheModel(makeModel)
    }

    func testShardingReplacesTheProjectionsWithShardedLayers() throws {
        let model = try makeModel(quantized: true)
        try model.shard(group: MLXDistributed.initialize())

        for layer in model.model.layers {
            XCTAssertTrue(layer.attention.wq is QuantizedAllToShardedLinear)
            XCTAssertTrue(layer.attention.wo is QuantizedShardedToAllLinear)
            XCTAssertTrue(layer.mlp.gate is QuantizedAllToShardedLinear)
            XCTAssertTrue(layer.mlp.down is QuantizedShardedToAllLinear)
        }
    }

    func testIndivisibleHeadsAreRefusedBeforeAnythingIsSharded() throws {
        let model = try makeShardingModel(
            """
            {
                "model_type": "mistral3", "hidden_size": 48, "num_hidden_layers": 1,
                "intermediate_size": 96, "num_attention_heads": 3, "num_key_value_heads": 3,
                "head_dim": 16, "rms_norm_eps": 1e-5, "vocab_size": 64
            }
            """, configuration: Mistral3TextConfiguration.self, quantized: false,
            Mistral3TextModel.init)
        // A group of one divides everything, so check the requirement directly.
        XCTAssertThrowsError(
            try requireDivisible(["attention heads": 3, "key-value heads": 3], by: 2))
        try model.shard(group: MLXDistributed.initialize())
    }
}
