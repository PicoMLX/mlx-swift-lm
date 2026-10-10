// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXVLM
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

private class WholeOnlyModel: Module, LanguageModel, KVCacheDimensionProvider {
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

/// Records the group it was asked to shard across.
private final class ShardableModel: WholeOnlyModel, TensorParallelModel {
    private(set) var group: MLXDistributed.Group?

    func shard(group: MLXDistributed.Group?) throws {
        self.group = group
    }
}

private struct TestTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        TestTokenizer()
    }
}

/// A configuration that names a distributed group makes the LLM factory load shard-first,
/// and makes the factories that cannot shard refuse.
///
/// One test process can only form a group of one, so these cover which loader the factory
/// picks. `ShardedLoadTests` covers what that loader does.
final class FactoryShardedLoadTests: XCTestCase {

    private func checkpoint(modelType: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "FactoryShardedLoadTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(#"{"model_type":"\#(modelType)"}"#.utf8)
            .write(to: directory.appending(component: "config.json"))
        try save(
            arrays: ["layer.weight": MLXArray([Float(1), 2, 3, 4]).reshaped(2, 2)],
            url: directory.appending(component: "model.safetensors"))
        return directory
    }

    private func factory() async -> LLMModelFactory {
        let registry = ModelTypeRegistry<LanguageModel>()
        await registry.registerModelType("shardable_test") { _ in ShardableModel() }
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

        let model = try XCTUnwrap(context.model as? ShardableModel)
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

        let model = try XCTUnwrap(context.model as? ShardableModel)
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
