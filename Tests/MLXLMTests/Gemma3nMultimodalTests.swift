// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

@Suite(.serialized)
struct Gemma3nMultimodalTests {
    @Test func referenceAudioAndAttentionMatchPython() throws {
        try Device.withDefaultDevice(.gpu) {
            let fixture = try arrays()
            var config = Gemma3nAudioConfiguration()
            config.inputFeatSize = 8
            config.hiddenSize = 8
            config.confNumAttentionHeads = 2
            config.confNumHiddenLayers = 2
            config.confAttentionChunkSize = 4
            config.confAttentionContextLeft = 3
            config.confAttentionContextRight = 1
            config.confReductionFactor = 2
            config.sscpConvChannelSize = [2, 2]
            let model = Gemma3nAudioModel(config)
            try load(model, prefix: "audio", fixture: fixture)
            var stage = model.subsampling(try value("audio.input", fixture))
            try expectClose(stage, value("audio.subsampled", fixture))
            let stageMask = try value("audio.mask", fixture).take(
                MLXArray(0 ..< stage.dim(1)) * 4, axis: 1)
            for (i, block) in model.conformer.enumerated() {
                stage = block(stage, paddingMask: stageMask)
                try expectClose(stage, value("audio.block\(i)", fixture))
            }
            let (output, mask) = model(
                try value("audio.input", fixture),
                paddingMask: try value("audio.mask", fixture))
            try expectClose(output, value("audio.output0", fixture))
            #expect(arrayEqual(mask, try value("audio.output1", fixture)).item(Bool.self))

            let attention = Gemma3nAudioAttention(config)
            try load(attention, prefix: "attention", fixture: fixture)
            try expectClose(
                attention(
                    value("attention.input", fixture),
                    paddingMask: value("attention.mask", fixture)),
                value("attention.output", fixture))
        }
    }

    @Test func referenceMobileNetBlocksMatchPython() throws {
        try Device.withDefaultDevice(.cpu) {
            let fixture = try arrays()
            let residual = Gemma3nInvertedResidual(
                8, 16, startKernel: 3,
                middleKernel: 5, stride: 2, expansion: 2)
            try load(residual, prefix: "inverted", fixture: fixture)
            try expectClose(
                residual(value("inverted.input", fixture)), value("inverted.output", fixture))
            let attention = Gemma3nMobileAttention(8, heads: 2, headDim: 4, stride: 2)
            try load(attention, prefix: "vision_attention", fixture: fixture)
            try expectClose(
                attention(value("vision_attention.input", fixture)),
                value("vision_attention.output", fixture))
            let fusion = Gemma3nMultiscaleFusion(
                inputChannels: 16, outputChannels: 16, resolution: 2)
            try load(fusion, prefix: "fusion", fixture: fixture)
            try expectClose(
                fusion([value("fusion.first", fixture), value("fusion.second", fixture)]),
                value("fusion.output", fixture))
        }
    }

    @Test func referenceHardAndSoftEmbeddingsMatchPython() throws {
        let fixture = try arrays()
        let embedder = Gemma3nMultimodalEmbedder(
            dimensions: 8, textDimensions: 16,
            vocabularySize: 8, offset: 32, eps: 1e-6)
        try load(embedder, prefix: "embedder", fixture: fixture)
        try expectClose(
            embedder.hard(value("embedder.ids", fixture)), value("embedder.output", fixture))
        try expectClose(
            embedder.soft(value("embedder.soft", fixture)), value("embedder.soft_output", fixture))
    }

    @Test func referenceAudioFeaturesMatchTransformers() throws {
        try Device.withDefaultDevice(.cpu) {
            let fixture = try arrays()
            var config = Gemma3nFeatureExtractorConfiguration()
            config.featureSize = 8
            let features = try Gemma3nAudioFeatureExtractor(config: config).features(
                samples: value("features.input", fixture))
            try expectClose(features, value("features.output", fixture), tolerance: 3e-4)
        }
    }

    @Test func imageProcessorHonorsPublishedPixelSettings() throws {
        let fixture = try arrays()
        var config = Gemma3nProcessorConfiguration()
        config.imageSize = 8
        config.doNormalize = false
        let source = try value("image.input", fixture)
        let expected = try value("image.output", fixture)
        try expectClose(gemma3nImagePixels(source, config: config), expected, tolerance: 1.01 / 255)
        config.doNormalize = true
        try expectClose(
            gemma3nImagePixels(source, config: config), expected * 2 - 1, tolerance: 2.01 / 255)
    }

    @Test func featureScatterPreservesRowsAndRejectsMismatches() throws {
        let tokens = MLXArray([1, 7, 7, 2, 7, 3]).reshaped(2, 3)
        let embeddings = MLXArray(0 ..< 24).asType(.float32).reshaped(2, 3, 4)
        let features = (MLXArray(0 ..< 12).asType(.float32) + 100).reshaped(3, 4)
        let output = try gemma3nMergeFeatures(
            embeddings, tokens: tokens, tokenId: 7,
            features: features, kind: "image")
        #expect(arrayEqual(output[0, 0], embeddings[0, 0]).item(Bool.self))
        #expect(arrayEqual(output[0, 1], features[0]).item(Bool.self))
        #expect(arrayEqual(output[0, 2], features[1]).item(Bool.self))
        #expect(arrayEqual(output[1, 1], features[2]).item(Bool.self))
        #expect(
            throws: Gemma3nError.placeholderMismatch(kind: "image", placeholders: 3, features: 2)
        ) {
            try gemma3nMergeFeatures(
                embeddings, tokens: tokens, tokenId: 7,
                features: features[..<2], kind: "image")
        }
    }

    @Test(arguments: [false, true])
    func raggedRowsMatchSoloAcrossWindowAndReordering(sparse: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyGemma3n(sparse: sparse)
            let prompts: [[Int32]] = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10]]
            let singles = try prompts.map { _ in try model.newCache(parameters: nil) }
            var expected: [MLXArray] = []
            for (row, prompt) in prompts.enumerated() {
                expected.append(model(MLXArray(prompt).reshaped(1, -1), cache: singles[row])[0, -1])
            }
            let factories = try makeBatchedCacheFactories(for: model.newCache(parameters: nil))
            let batched = factories.map { $0([0, 0]) }
            #expect(batched.count == 2)
            batched.forEach { $0.prepare(lengths: [3, 7]) }
            let padded = prompts.flatMap { $0 + Array(repeating: 0, count: 7 - $0.count) }
            let caches = batched.map { $0 as any KVCache }
            let output = model(MLXArray(padded).reshaped(2, 7), cache: caches)
            batched.forEach { $0.finalize() }
            for row in 0 ..< 2 {
                try expectClose(output[row, prompts[row].count - 1], expected[row])
            }
            batched.forEach { $0.filter(batchIndices: MLXArray([1, 0])) }
            for step in 0 ..< 7 {
                let tokens = [Int32(20 + step), Int32(30 + step)]
                let output = model(MLXArray(tokens).reshaped(2, 1), cache: caches)
                for (row, original) in [1, 0].enumerated() {
                    let solo = model(
                        MLXArray([tokens[row]]).reshaped(1, 1), cache: singles[original])
                    try expectClose(output[row ..< row + 1], solo)
                }
            }
            let extracted = batched.map { $0.extract(0) }
            let next = MLXArray([Int32(42)]).reshaped(1, 1)
            try expectClose(model(next, cache: extracted), model(next, cache: singles[1]))
        }
    }

    @Test func preparedMediaRowsCanJoinTextRows() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyGemma3n(sparse: true)
            let language = model.languageModel
            let mediaTokens = MLXArray([1, 50, 50, 2]).reshaped(1, 4)
            let textTokens = MLXArray([3, 4, 5]).reshaped(1, 3)
            let mediaCache = try model.newCache(parameters: nil)
            let textCache = try model.newCache(parameters: nil)
            let features = MLXRandom.normal([2, 32])
            let embedded = try gemma3nMergeFeatures(
                language.embeddings(mediaTokens), tokens: mediaTokens,
                tokenId: 50, features: features, kind: "image")
            _ = language(
                inputsEmbeds: embedded, cache: mediaCache.map { Optional($0) },
                perLayerInputs: language.getPerLayerInputs(mediaTokens))
            _ = model(textTokens, cache: textCache)
            eval(mediaCache, textCache)
            let references = [mediaCache.map { $0.copy() }, textCache.map { $0.copy() }]
            func adopted(_ cache: KVCache) throws -> any BatchedCache {
                if let rotating = cache as? RotatingKVCache {
                    return try #require(BatchRotatingKVCache.fromSingle(rotating))
                }
                return BatchKVCache.fromSingle(try #require(cache as? KVCacheSimple))
            }
            let batch = try mediaCache.map(adopted)
            for layer in batch.indices { batch[layer].extend(other: try adopted(textCache[layer])) }
            for step in 0 ..< 6 {
                let token = Int32(11 + step)
                let output = model(
                    MLXArray([token, token]).reshaped(2, 1), cache: batch.map { $0 as KVCache })
                for row in 0 ..< 2 {
                    let solo = model(MLXArray([token]).reshaped(1, 1), cache: references[row])
                    try expectClose(output[row ..< row + 1], solo)
                }
            }
        }
    }

    @Test(arguments: [false, true], [false, true])
    func singletonShardingPreservesDenseAndSparseModels(sparse: Bool, quantized: Bool) throws {
        let model = try tinyGemma3n(sparse: sparse)
        if quantized {
            quantize(
                model: model, groupSize: 32, bits: 4,
                filter: { _, module in
                    if let linear = module as? Linear { return linear.weight.dim(-1) % 32 == 0 }
                    if let embedding = module as? Embedding {
                        return embedding.weight.dim(-1) % 32 == 0
                    }
                    return false
                })
        }
        let tokens = MLXArray([1, 2, 3, 4, 5]).reshaped(1, 5)
        let expected = model(tokens, cache: nil)
        eval(expected)
        try model.shard(group: MLXDistributed.initialize())
        try expectClose(model(tokens, cache: nil), expected)
        #expect(throws: TensorParallelError.alreadySharded) {
            try model.shard(group: MLXDistributed.initialize())
        }
    }

    @Test func processorIsRegistered() async {
        #expect(await VLMTypeRegistry.shared.contains("gemma3n"))
    }

    @Test("Actual media preparation preserves ragged cache positions", arguments: [false, true])
    func raggedMediaPreparation(chunked: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            var text = try #require(
                JSONSerialization.jsonObject(
                    with: JSONEncoder().encode(tinyGemma3n(sparse: true).config)) as? [String: Any])
            text["vocab_size_per_layer_input"] = 32
            let data = try JSONSerialization.data(withJSONObject: [
                "model_type": "gemma3n", "text_config": text,
                "vision_config": [
                    "hidden_size": 2048, "image_size": 32,
                    "vocab_size": 8, "vocab_offset": 32,
                ],
                "audio_config": [
                    "input_feat_size": 8, "hidden_size": 8,
                    "conf_num_attention_heads": 2, "conf_num_hidden_layers": 1,
                    "conf_attention_chunk_size": 4, "conf_attention_context_left": 3,
                    "conf_attention_context_right": 1, "conf_reduction_factor": 2,
                    "sscp_conv_channel_size": [2, 2], "vocab_size": 16, "vocab_offset": 40,
                ],
                "image_token_id": 34, "audio_token_id": 42, "audio_soft_tokens_per_image": 3,
            ])
            let configuration = try JSONDecoder().decode(Gemma3nConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            let model = Gemma3n(configuration)
            let mediaIDs: [Int32] = [1, 42, 42, 42, 2]
            let textIDs: [Int32] = [3, 4]
            let audio = LMInput.ProcessedAudio(
                features: sin(MLXArray(0 ..< 152).asType(.float32) * 0.09).reshaped(1, 19, 8),
                mask: MLXArray.ones([1, 19], dtype: .bool))
            let singles = try [model.newCache(parameters: nil), model.newCache(parameters: nil)]
            let prefill = PrefillParameters(stepSize: 2, chunking: chunked ? .balanced : .unchunked)
            let media = LMInput(
                text: .init(tokens: MLXArray(mediaIDs).reshaped(1, 5)), audio: audio)
            guard
                case .logits(let expected) = try model.prepare(
                    media, cache: singles[0], state: nil, prefill: prefill)
            else {
                Issue.record("Media preparation must return logits")
                return
            }
            eval(model(MLXArray(textIDs).reshaped(1, 2), cache: singles[1]))
            let factories = try makeBatchedCacheFactories(for: model.newCache(parameters: nil))
            let batch = factories.map { $0([0, 0]) }
            let caches = batch.map { $0 as any KVCache }
            let input = LMInput(
                text: .init(
                    tokens: MLXArray(mediaIDs + textIDs + [0, 0, 0]).reshaped(2, 5),
                    mask: MLXArray([1, 1, 1, 1, 1, 1, 1, 0, 0, 0]).reshaped(2, 5)), audio: audio)
            guard
                case .logits(let actual) = try model.prepare(
                    input, cache: caches, state: nil, prefill: prefill)
            else {
                Issue.record("Batched media preparation must return logits")
                return
            }
            try expectClose(actual.logits[0, -1], expected.logits[0, -1])
            for layer in batch {
                let positioned = try #require(layer as? any BatchPositionedKVCache)
                #expect(positioned.batchOffset.asArray(Int.self) == [5, 2])
            }
            for step in 0 ..< 3 {
                let tokens: [Int32] = [Int32(9 + step), Int32(19 + step)]
                let output = model(MLXArray(tokens).reshaped(2, 1), cache: caches)
                for row in 0 ..< 2 {
                    try expectClose(
                        output[row ..< row + 1],
                        model(MLXArray([tokens[row]]).reshaped(1, 1), cache: singles[row]))
                }
            }
        }
    }

    @Test func processorRetainsSplitImageAndAudioSidecars() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(
            """
            {"processor_class":"Gemma3nProcessor","image_seq_length":256,"audio_seq_length":188}
            """.utf8
        ).write(to: directory.appendingPathComponent("processor_config.json"))
        try Data(
            """
            {"feature_size":8,"sampling_rate":8000,"max_frequency":3800,"preemphasis":0.5}
            """.utf8
        ).write(to: directory.appendingPathComponent("audio_preprocessor_config.json"))
        let merged = try Gemma3nProcessorConfiguration.mergedData(
            directory: directory,
            modelConfiguration: Data(
                "{\"audio_config\":{\"input_feat_size\":8},\"image_token_id\":42}".utf8),
            selected: Data(
                "{\"size\":{\"height\":768,\"width\":768},\"image_mean\":[0.1,0.2,0.3],\"do_normalize\":false}"
                    .utf8))
        let configuration = try JSONDecoder().decode(
            Gemma3nProcessorConfiguration.self, from: merged)
        #expect(configuration.imageSize == 768)
        #expect(configuration.imageMean == [0.1, 0.2, 0.3])
        #expect(!configuration.doNormalize)
        #expect(configuration.imageTokenId == 42)
        #expect(configuration.featureExtractor.featureSize == 8)
        #expect(configuration.featureExtractor.samplingRate == 8000)
        #expect(configuration.featureExtractor.preemphasis == 0.5)
    }

    private func arrays() throws -> [String: MLXArray] {
        let url = try #require(
            Bundle.module.url(forResource: "gemma3n-reference", withExtension: "safetensors"))
        return try loadArrays(url: url)
    }

    private func value(_ name: String, _ fixture: [String: MLXArray]) throws -> MLXArray {
        try #require(fixture[name], "Missing fixture \(name)")
    }

    private func load(_ model: Module, prefix: String, fixture: [String: MLXArray]) throws {
        let marker = prefix + ".weights."
        let weights = fixture.filter { $0.key.hasPrefix(marker) }.map {
            (String($0.key.dropFirst(marker.count)), $0.value)
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
    }

    private func expectClose(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 1e-4)
        throws
    {
        #expect(actual.shape == expected.shape)
        guard actual.shape == expected.shape else { return }
        eval(actual, expected)
        #expect(
            abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
                < tolerance)
    }
}

func tinyGemma3n(sparse: Bool = false) throws -> Gemma3nTextModel {
    var values = try #require(
        JSONSerialization.jsonObject(with: JSONEncoder().encode(Gemma3nTextConfiguration()))
            as? [String: Any])
    values.merge([
        "hidden_size": 32, "num_hidden_layers": 4, "intermediate_size": [128, 128, 128, 128],
        "num_attention_heads": 4, "head_dim": 16, "vocab_size": 64,
        "num_key_value_heads": 2, "num_kv_shared_layers": 2, "vocab_size_per_layer_input": 48,
        "hidden_size_per_layer_input": 8, "altup_num_inputs": 2, "laurel_rank": 8,
        "sliding_window": 4,
        "layer_types": [
            "sliding_attention", "full_attention", "sliding_attention", "full_attention",
        ],
        "activation_sparsity_pattern": sparse ? [0.5, 0.5, 0.0, 0.0] : [0.0, 0.0, 0.0, 0.0],
    ]) { _, new in new }
    let configuration = try JSONDecoder().decode(
        Gemma3nTextConfiguration.self,
        from: JSONSerialization.data(withJSONObject: values))
    return withRandomState(MLXRandom.RandomState(seed: 719)) {
        let model = Gemma3nTextModel(config: configuration)
        eval(model)
        return model
    }
}
