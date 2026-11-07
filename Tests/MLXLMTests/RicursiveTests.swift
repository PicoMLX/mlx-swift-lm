// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

// Module tests use XCTest because this package's Swift Testing host cannot
// currently initialize MLX modules. Each test owns its model and random state.
final class RicursiveTests: XCTestCase {

    private let tokens: [Int32] = [1, 3, 5, 2, 9, 4, 8, 7, 6, 12, 10, 11, 14]

    private func resource(_ name: String, extension suffix: String) throws -> URL {
        try XCTUnwrap(Bundle.module.url(forResource: "ricursive-\(name)", withExtension: suffix))
    }

    private func configurationData(_ overrides: [String: Any] = [:]) throws -> Data {
        let data = try Data(contentsOf: resource("config", extension: "json"))
        var values = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        values.merge(overrides) { _, value in value }
        return try JSONSerialization.data(withJSONObject: values)
    }

    private func configuration(_ overrides: [String: Any] = [:]) throws -> RicursiveConfiguration {
        try JSONDecoder().decode(RicursiveConfiguration.self, from: configurationData(overrides))
    }

    private func emptyModel(_ config: RicursiveConfiguration) -> RicursiveModel {
        withRandomState(MLXRandom.RandomState(seed: 17)) {
            RicursiveModel(config)
        }
    }

    private func model(
        _ overrides: [String: Any] = [:], dtype: DType = .float32
    ) throws -> RicursiveModel {
        let model = emptyModel(try configuration(overrides))
        let parameterKeys = Set(model.parameters().flattened().map(\.0))
        let weights = try MLX.loadArrays(url: resource("weights", extension: "safetensors"))
            .filter { parameterKeys.contains($0.key) }
            .mapValues { $0.asType(dtype) }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        return model
    }

    private func reference(_ name: String = "reference") throws -> MLXArray {
        let arrays = try MLX.loadArrays(url: resource("logits", extension: "safetensors"))
        return try XCTUnwrap(arrays[name])
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 2e-5,
        context: String = "", file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, context, file: file, line: line)
        let difference = abs(actual.asType(.float32) - expected.asType(.float32))
        XCTAssertLessThanOrEqual(
            difference.max().item(Float.self), tolerance, context, file: file, line: line)
    }

    func testFloat32MatchesOfficialPyTorchForward() throws {
        try Device.withDefaultDevice(.cpu) {
            let variants: [(String, [String: Any])] = [
                ("reference", [:]),
                ("ungated", ["use_gated_attention": false]),
                ("without_value_residual", ["use_valres": false]),
                ("unclipped", ["ffn_clamp_threshold": NSNull()]),
                ("tied_embeddings", ["tie_word_embeddings": true]),
            ]
            for (name, overrides) in variants {
                let output = try model(overrides)(MLXArray(tokens)[.newAxis], cache: nil)
                assertClose(output, try reference(name), context: name)
            }
        }
    }

    func testBFloat16MatchesOfficialPyTorchForward() throws {
        for device in [Device.cpu, Device.gpu] {
            try Device.withDefaultDevice(device) {
                let output = try model(dtype: .bfloat16)(MLXArray(tokens)[.newAxis], cache: nil)
                let expected = try reference("bfloat16")
                let difference = abs(output.asType(.float32) - expected)
                // PyTorch's own CPU default/math SDPA kernels differ by 0.043 maximum
                // and 0.009 mean for this BF16 fixture; provenance records that comparison.
                assertClose(output, expected, tolerance: 0.05, context: "\(device)")
                XCTAssertLessThanOrEqual(difference.mean().item(Float.self), 0.01, "\(device)")
            }
        }
    }

    func testCachedDecodeAndChunkedPrefillMatchOfficialFullForward() throws {
        try Device.withDefaultDevice(.cpu) {
            try assertCachedMatchesReference(tolerance: 2e-5)
        }
    }

    func testMetalForwardAndCachedChunksMatchOfficialReference() throws {
        try Device.withDefaultDevice(.gpu) {
            // MLX enables reduced-precision TF32 matmuls on supported GPUs by default.
            let output = try model()(MLXArray(tokens)[.newAxis], cache: nil)
            assertClose(output, try reference(), tolerance: 0.01)
            try assertCachedMatchesReference(tolerance: 0.01)
        }
    }

    private func assertCachedMatchesReference(tolerance: Float) throws {
        let model = try model()
        let expected = try reference()
        let chunkings = [
            Array(repeating: 1, count: tokens.count),
            [3, 4, 6],
            [6, 1, 6],
            [4, 4, 4, 1],
            [13],
        ]
        for lengths in chunkings {
            let cache = try model.newCache(parameters: nil)
            var offset = 0
            var outputs = [MLXArray]()
            for length in lengths {
                let input = MLXArray(Array(tokens[offset ..< offset + length]))[.newAxis]
                let output = model(input, cache: cache)
                eval(output)
                outputs.append(output)
                offset += length
            }
            assertClose(
                concatenated(outputs, axis: 1), expected,
                tolerance: tolerance, context: "chunks \(lengths)")
        }
    }

    func testLocalCachesStayBoundedAndGlobalCacheRetainsHistory() throws {
        let model = try model()
        let caches = try model.newCache(parameters: nil)
        XCTAssertEqual(caches.count, 5)
        for token in tokens {
            eval(model(MLXArray([token])[.newAxis], cache: caches))
            for cache in caches.prefix(4) {
                XCTAssertLessThanOrEqual(try XCTUnwrap(cache.state.first).dim(2), 4)
            }
        }
        XCTAssertTrue(caches.prefix(4).allSatisfy { $0 is RotatingKVCache })
        XCTAssertTrue(caches.last is KVCacheSimple)
        XCTAssertEqual(try XCTUnwrap(caches.last?.state.first).dim(2), tokens.count)
        XCTAssertTrue(caches.allSatisfy { $0.offset == tokens.count })
    }

    func testLegacyCapacityCapsLocalAndGlobalCaches() throws {
        let model = try model()
        let caches = try model.newCache(parameters: GenerateParameters(maxKVSize: 2))
        XCTAssertEqual(caches.map(\.maxSize), Array(repeating: 2, count: 5))
        for cache in caches.prefix(4) {
            XCTAssertEqual(try XCTUnwrap(cache as? RotatingKVCache).keepCount, 0)
        }
        XCTAssertEqual(try XCTUnwrap(caches.last as? RotatingKVCache).keepCount, 1)
    }

    func testTypedCapacityPreservesNativeLocalWindow() throws {
        let capacity = try KVCacheConfiguration.Capacity(
            maxTokens: 2, preservedPrefixTokens: 1)
        let parameters = GenerateParameters(kvCache: KVCacheConfiguration(capacity: capacity))
        let caches = try model().newCache(parameters: parameters)
        XCTAssertEqual(caches.map(\.maxSize), [4, 4, 4, 4, 2])
        for cache in caches.prefix(4) {
            let local = try XCTUnwrap(cache as? RotatingKVCache)
            XCTAssertEqual(local.keepCount, 0)
            XCTAssertEqual(local.capacityOrigin, .modelNative)
        }
        let global = try XCTUnwrap(caches.last as? RotatingKVCache)
        XCTAssertEqual(global.keepCount, 1)
        XCTAssertEqual(global.capacityOrigin, .requested)
    }

    func testConfigurationPreservesExplicitAttentionWidthAndDefaultPattern() throws {
        let config = try configuration(["layer_types": NSNull()])
        XCTAssertEqual(config.attentionHeads * config.headDim, config.hiddenSize * 2)
        XCTAssertEqual(
            config.layerTypes,
            Array(repeating: "sliding_attention", count: 4)
                + ["full_attention"])
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.embedScale, 2.5)
        XCTAssertEqual(config.ffnClampThreshold, 0.6)
    }

    func testRegistryAndChatConventions() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configurationData(), modelType: "ricursive")
        XCTAssertTrue(model is RicursiveModel)
        XCTAssertEqual(model.toolCallFormat, .json)
        let reasoning = try XCTUnwrap(model.reasoningConfig)
        XCTAssertEqual(reasoning.startDelimiter, "<think>")
        XCTAssertEqual(reasoning.endDelimiter, "</think>")
        XCTAssertEqual(reasoning.promptStrategy, .none)
        XCTAssertFalse(reasoning.isSpecialToken)
        XCTAssertEqual(reasoning.startPolicy, .responseStart)
        XCTAssertEqual(reasoning.whitespacePolicy, .singleNewline)
        XCTAssertNil(reasoning.budgetTransition)
    }

    func testSanitizePreservesResidualScalarAndQuantizationMetadata() throws {
        let scalar = MLXArray([Float(0.75)])
        let weights = [
            "model.layers.0.self_attn.value_res_lambda": scalar,
            "model.layers.0.self_attn.g_proj.scales": scalar,
            "model.layers.0.self_attn.rotary_emb.inv_freq": scalar,
            "lm_head.weight": scalar,
        ]
        let untied = try model().sanitize(weights: weights)
        XCTAssertEqual(
            Set(untied.keys),
            Set(weights.keys).subtracting([
                "model.layers.0.self_attn.rotary_emb.inv_freq"
            ]))
        let tied = try model(["tie_word_embeddings": true]).sanitize(weights: weights)
        XCTAssertNil(tied["lm_head.weight"])
        XCTAssertNotNil(tied["model.layers.0.self_attn.value_res_lambda"])
        XCTAssertNotNil(tied["model.layers.0.self_attn.g_proj.scales"])
    }

    func testAffineAndMixedPrecisionCheckpointsLoadThroughCommonLoader() throws {
        let config = try configuration([
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 64,
            "layer_types": ["sliding_attention", "full_attention"],
        ])
        for mixed in [false, true] {
            let source = emptyModel(config)
            quantize(model: source) { path, _ in
                (groupSize: 64, bits: mixed && path.hasSuffix("g_proj") ? 8 : 4, mode: .affine)
            }
            let arrays = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
            let directory = FileManager.default.temporaryDirectory
                .appending(component: "ricursive-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            try MLX.save(arrays: arrays, url: directory.appending(component: "model.safetensors"))

            var quantization: [String: Any] = ["group_size": 64, "bits": 4]
            if mixed {
                for index in 0 ..< config.hiddenLayers {
                    quantization["model.layers.\(index).self_attn.g_proj"] = [
                        "group_size": 64, "bits": 8,
                    ]
                }
            }
            let metadata = try JSONDecoder().decode(
                BaseConfiguration.self,
                from: JSONSerialization.data(withJSONObject: [
                    "model_type": "ricursive", "quantization": quantization,
                ]))
            let loaded = emptyModel(config)
            try loadWeights(
                modelDirectory: directory, model: loaded,
                perLayerQuantization: metadata.perLayerQuantization)
            let input = MLXArray(tokens)[.newAxis]
            assertClose(loaded(input, cache: nil), source(input, cache: nil), tolerance: 1e-6)
            let modules = Dictionary(uniqueKeysWithValues: loaded.leafModules().flattened())
            let gate = try XCTUnwrap(
                modules["model.layers.0.self_attn.g_proj"] as? QuantizedLinear)
            XCTAssertEqual(gate.bits, mixed ? 8 : 4)
        }
    }
}
