// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Records what `loadShardedWeights` asks of a model.
private final class RecordingModel: Module, BaseLanguageModel, TensorParallelModel {
    struct Refused: Error {}

    @ModuleInfo(key: "layer") var layer: Linear
    var refuses = false
    private(set) var group: MLXDistributed.Group?
    /// Derived from the weight as it was when `shard` ran, without reading it. It holds the
    /// checkpoint's values only if the weights were set before the model was sharded.
    private(set) var weightWhenSharded: MLXArray?

    override init() {
        _layer.wrappedValue = Linear(2, 2, bias: false)
    }

    func shard(group: MLXDistributed.Group?) throws {
        if refuses { throw Refused() }
        self.group = group
        weightWhenSharded = layer.weight + 0
    }
}

/// `loadShardedWeights`: the weights are set on the model unread, the model is sharded, and
/// only then are they read.
///
/// One test process can only form a group of one, where sharding keeps every weight. These
/// cover the order of the steps and that the result is the model the eager loader gives.
/// Keeping half the weights across two processes needs a multi-process run.
final class ShardedLoadTests: XCTestCase {

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "ShardedLoadTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func qwenConfiguration() throws -> Qwen35TextConfiguration {
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
                "num_experts": 0,
                "num_experts_per_tok": 0
            }
            """
        return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    /// A Qwen 3.5 checkpoint with random weights, stored the way a converted one is.
    private func writeQwenCheckpoint(quantized: Bool) throws -> URL {
        MLXRandom.seed(21)
        let model = Qwen35TextModel(try qwenConfiguration())
        if quantized {
            quantize(model: model, groupSize: 64, bits: 4)
        }
        eval(model)

        let directory = try temporaryDirectory()
        try save(
            arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
            url: directory.appending(component: "model.safetensors"))
        return directory
    }

    private func lastLogits(_ model: Qwen35TextModel) throws -> MLXArray {
        let cache = try model.newCache(parameters: nil)
        let tokens = MLXArray([Int32(3), 17, 42, 8, 29, 51]).reshaped(1, -1)
        let logits = model(tokens, cache: cache)[0, -1]
        eval(logits)
        return logits
    }

    func testShardFirstLoadGivesTheModelTheEagerLoadGives() throws {
        for quantized in [false, true] {
            let directory = try writeQwenCheckpoint(quantized: quantized)
            defer { try? FileManager.default.removeItem(at: directory) }
            let quantization =
                quantized ? BaseConfiguration.Quantization(groupSize: 64, bits: 4) : nil

            let eager = Qwen35TextModel(try qwenConfiguration())
            try loadWeights(modelDirectory: directory, model: eager, quantization: quantization)

            let sharded = Qwen35TextModel(try qwenConfiguration())
            try loadShardedWeights(
                modelDirectory: directory, model: sharded, group: MLXDistributed.initialize(),
                quantization: quantization)

            let expected = Dictionary(uniqueKeysWithValues: eager.parameters().flattened())
            let actual = Dictionary(uniqueKeysWithValues: sharded.parameters().flattened())
            XCTAssertEqual(Set(actual.keys), Set(expected.keys))
            for (key, value) in expected {
                let loaded = try XCTUnwrap(actual[key], key)
                XCTAssertTrue(arrayEqual(loaded, value).item(Bool.self), "\(key) differs")
            }
            XCTAssertTrue(
                arrayEqual(try lastLogits(sharded), try lastLogits(eager)).item(Bool.self),
                "\(quantized ? "4-bit" : "float") logits differ")
        }
    }

    func testModelIsShardedAfterItsWeightsAreSetAndWithTheGivenGroup() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let weight = MLXArray([Float(1), 2, 3, 4]).reshaped(2, 2)
        try save(
            arrays: ["layer.weight": weight],
            url: directory.appending(component: "model.safetensors"))

        let model = RecordingModel()
        let group = try MLXDistributed.initialize()
        try loadShardedWeights(modelDirectory: directory, model: model, group: group)

        XCTAssertTrue(model.group === group)
        XCTAssertEqual(model.weightWhenSharded?.asArray(Float.self), [1, 2, 3, 4])
        XCTAssertEqual(model.layer.weight.asArray(Float.self), [1, 2, 3, 4])
    }

    func testModelThatRefusesToShardFailsTheLoad() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(
            arrays: ["layer.weight": MLXArray.ones([2, 2])],
            url: directory.appending(component: "model.safetensors"))

        let model = RecordingModel()
        model.refuses = true

        XCTAssertThrowsError(
            try loadShardedWeights(
                modelDirectory: directory, model: model, group: MLXDistributed.initialize())
        ) { error in
            XCTAssertTrue(error is RecordingModel.Refused)
        }
    }

    /// The unread weights must merge like the eager loader's: a later file wins a duplicate
    /// name, and the metadata is the first file's.
    func testOpenedFilesMergeLikeTheEagerLoader() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let urls = ["model-00001.safetensors", "model-00002.safetensors"].map {
            directory.appending(component: $0)
        }
        try save(
            arrays: ["a": MLXArray([Float(1)]), "shared": MLXArray([Float(1)])],
            metadata: ["format": "first"], url: urls[0])
        try save(
            arrays: ["b": MLXArray([Float(2)]), "shared": MLXArray([Float(2)])],
            metadata: ["format": "second"], url: urls[1])

        let opened = try openWeightArrays(urls: urls)
        let loaded = try loadWeightArrays(urls: urls)

        XCTAssertEqual(Set(opened.weights.keys), Set(loaded.weights.keys))
        XCTAssertEqual(opened.weights["shared"]?.asArray(Float.self), [2])
        XCTAssertEqual(opened.metadata, loaded.metadata)
        XCTAssertEqual(opened.metadata["format"], "first")
    }
}
