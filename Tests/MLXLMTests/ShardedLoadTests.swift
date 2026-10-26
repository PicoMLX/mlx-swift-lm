// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXScriptedLM
import MLXVLM
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Records what the shard-first loader asks of a model.
private class RecordingModel: Module, LanguageModel, KVCacheDimensionProvider, TensorParallelModel {
    struct Refused: Error {}

    let kvHeads: [Int] = []
    @ModuleInfo(key: "layer") var layer: Linear
    var refuses = false
    private(set) var group: MLXDistributed.Group?
    /// Derived from the weight as it was when `shard` ran, without reading it. It holds the
    /// checkpoint's values only if the weights were set before the model was sharded.
    private(set) var weightWhenSharded: MLXArray?

    override init() {
        _layer.wrappedValue = Linear(2, 2, bias: false)
    }

    func shard(group: MLXDistributed.Group) throws {
        if refuses { throw Refused() }
        self.group = group
        weightWhenSharded = layer.weight + 0
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        inputs
    }
}

/// A model without tensor parallelism.
private final class WholeOnlyModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads: [Int] = []
    @ModuleInfo(key: "layer") var layer: Linear

    override init() {
        _layer.wrappedValue = Linear(2, 2, bias: false)
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        inputs
    }
}

private struct TestTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        PseudoWordTokenizer()
    }
}

/// The shard-first loader sets the weights unread, shards the model and only then reads
/// them, and a configuration naming a group routes the LLM factory to it.
///
/// One test process can only form a group of one, where every shard is the whole
/// weight. Splitting weights across processes needs a multi-process run.
final class ShardedLoadTests: XCTestCase {

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "ShardedLoadTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private let llamaConfiguration = LlamaConfiguration(
        hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128, attentionHeads: 4,
        rmsNormEps: 1e-5, vocabularySize: 64, kvHeads: 2)

    /// A Llama checkpoint with random weights, stored the way a converted one is.
    private func writeLlamaCheckpoint(quantized: Bool) throws -> URL {
        MLXRandom.seed(21)
        let model = LlamaModel(llamaConfiguration)
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

    private func lastLogits(_ model: LlamaModel) throws -> MLXArray {
        let logits = model(
            MLXArray([Int32(3), 17, 42, 8, 29, 51]).reshaped(1, -1),
            cache: try model.newCache(parameters: nil))[0, -1]
        eval(logits)
        return logits
    }

    func testShardFirstLoadGivesTheModelTheEagerLoadGives() async throws {
        for quantized in [false, true] {
            let directory = try writeLlamaCheckpoint(quantized: quantized)
            defer { try? FileManager.default.removeItem(at: directory) }
            let quantization =
                quantized ? BaseConfiguration.Quantization(groupSize: 64, bits: 4) : nil

            let eager = LlamaModel(llamaConfiguration)
            try await loadWeights(
                modelDirectory: directory, model: eager, quantization: quantization)

            let sharded = LlamaModel(llamaConfiguration)
            try await loadShardedWeights(
                modelDirectory: directory, model: sharded, group: MLXDistributed.initialize(),
                quantization: quantization)

            let expected = Dictionary(uniqueKeysWithValues: eager.parameters().flattened())
            let actual = Dictionary(uniqueKeysWithValues: sharded.parameters().flattened())
            XCTAssertEqual(Set(actual.keys), Set(expected.keys))
            for (key, value) in expected {
                let loaded = try XCTUnwrap(actual[key], key)
                XCTAssertTrue(arrayEqual(loaded, value).item(Bool.self), "\(key) differs")
            }
            XCTAssertLessThanOrEqual(
                abs(try lastLogits(sharded) - lastLogits(eager)).max().item(Float.self), 1e-4,
                "\(quantized ? "4-bit" : "float") logits differ")
        }
    }

    func testModelIsShardedAfterItsWeightsAreSetAndWithTheGivenGroup() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(
            arrays: ["layer.weight": MLXArray([Float(1), 2, 3, 4]).reshaped(2, 2)],
            url: directory.appending(component: "model.safetensors"))

        let model = RecordingModel()
        let group = try MLXDistributed.initialize()
        try await loadShardedWeights(modelDirectory: directory, model: model, group: group)

        XCTAssertTrue(model.group === group)
        XCTAssertEqual(model.weightWhenSharded?.asArray(Float.self), [1, 2, 3, 4])
        XCTAssertEqual(model.layer.weight.asArray(Float.self), [1, 2, 3, 4])
    }

    func testModelThatRefusesToShardFailsTheLoad() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(
            arrays: ["layer.weight": MLXArray.ones([2, 2])],
            url: directory.appending(component: "model.safetensors"))

        let model = RecordingModel()
        model.refuses = true
        do {
            try await loadShardedWeights(
                modelDirectory: directory, model: model, group: MLXDistributed.initialize())
            XCTFail("Expected the load to fail")
        } catch {
            XCTAssertTrue(error is RecordingModel.Refused)
        }
    }

    // MARK: - Factories

    private func checkpoint(modelType: String) throws -> URL {
        let directory = try temporaryDirectory()
        try Data(#"{"model_type":"\#(modelType)"}"#.utf8)
            .write(to: directory.appending(component: "config.json"))
        try save(
            arrays: ["layer.weight": MLXArray([Float(1), 2, 3, 4]).reshaped(2, 2)],
            url: directory.appending(component: "model.safetensors"))
        return directory
    }

    private func factory() async -> LLMModelFactory {
        let registry = ModelTypeRegistry<LanguageModel>()
        await registry.registerModelType("shardable_test") { _ in RecordingModel() }
        await registry.registerModelType("whole_only_test") { _ in WholeOnlyModel() }
        return LLMModelFactory(typeRegistry: registry, modelRegistry: AbstractModelRegistry())
    }

    func testConfigurationWithAGroupLoadsShardFirst() async throws {
        let directory = try checkpoint(modelType: "shardable_test")
        defer { try? FileManager.default.removeItem(at: directory) }
        let group = try MLXDistributed.initialize()
        var configuration = ResolvedModelConfiguration(directory: directory)
        configuration.tensorParallelGroup = group

        let context = try await factory()._load(
            configuration: configuration, tokenizerLoader: TestTokenizerLoader())

        let model = try XCTUnwrap(context.model as? RecordingModel)
        XCTAssertTrue(model.group === group)
        XCTAssertEqual(model.layer.weight.asArray(Float.self), [1, 2, 3, 4])
        XCTAssertTrue(context.configuration.tensorParallelGroup === group)
    }

    func testConfigurationWithoutAGroupLoadsTheWholeModel() async throws {
        let directory = try checkpoint(modelType: "shardable_test")
        defer { try? FileManager.default.removeItem(at: directory) }

        let context = try await factory()._load(
            configuration: ResolvedModelConfiguration(directory: directory),
            tokenizerLoader: TestTokenizerLoader())

        let model = try XCTUnwrap(context.model as? RecordingModel)
        XCTAssertNil(model.group)
        XCTAssertEqual(model.layer.weight.asArray(Float.self), [1, 2, 3, 4])
        XCTAssertNil(context.configuration.tensorParallelGroup)
    }

    func testModelThatCannotBeShardedFailsTheLoad() async throws {
        let directory = try checkpoint(modelType: "whole_only_test")
        defer { try? FileManager.default.removeItem(at: directory) }
        var configuration = ResolvedModelConfiguration(directory: directory)
        configuration.tensorParallelGroup = try MLXDistributed.initialize()

        do {
            _ = try await factory()._load(
                configuration: configuration, tokenizerLoader: TestTokenizerLoader())
            XCTFail("Expected the load to be refused")
        } catch ModelFactoryError.invalidConfiguration(let message) {
            XCTAssertTrue(message.contains("whole_only_test"), message)
        }
    }

    func testVisionFactoryRefusesAGroup() async throws {
        let directory = try checkpoint(modelType: "shardable_test")
        defer { try? FileManager.default.removeItem(at: directory) }
        var configuration = ResolvedModelConfiguration(directory: directory)
        configuration.tensorParallelGroup = try MLXDistributed.initialize()

        do {
            _ = try await VLMModelFactory.shared._load(
                configuration: configuration, tokenizerLoader: TestTokenizerLoader())
            XCTFail("Expected the load to be refused")
        } catch ModelFactoryError.invalidConfiguration {
        }
    }

    func testResolvingKeepsTheGroup() throws {
        let directory = URL(filePath: "/tmp/model")
        let group = try MLXDistributed.initialize()
        var configuration = ModelConfiguration(directory: directory)
        XCTAssertNil(
            configuration.resolved(modelDirectory: directory, tokenizerDirectory: directory)
                .tensorParallelGroup)

        configuration.tensorParallelGroup = group

        XCTAssertTrue(
            configuration.resolved(modelDirectory: directory, tokenizerDirectory: directory)
                .tensorParallelGroup === group)
        XCTAssertNotEqual(configuration, ModelConfiguration(directory: directory))
    }
}
