// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

@Suite(.serialized)
struct Qwen3OmniTests {
    @Test func publishedVisionDefaultsAndRopeParameterAliasDecode() throws {
        let url = try #require(
            Bundle.module.url(forResource: "qwen3-omni-reference", withExtension: "json"))
        var root = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        var thinker = try #require(root["thinker_config"] as? [String: Any])
        var vision = try #require(thinker["vision_config"] as? [String: Any])
        vision.removeValue(forKey: "num_position_embeddings")
        var text = try #require(thinker["text_config"] as? [String: Any])
        text.removeValue(forKey: "rope_scaling")
        text.removeValue(forKey: "rope_theta")
        text["rope_parameters"] = [
            "rope_type": "default", "rope_theta": 4321.0, "mrope_section": [3, 3, 2],
        ]
        thinker["vision_config"] = vision
        thinker["text_config"] = text
        root["thinker_config"] = thinker
        let configuration = try JSONDecoder().decode(
            Qwen3OmniConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
        #expect(configuration.thinker.vision.numPositionEmbeddings == 2304)
        #expect(configuration.thinker.text.ropeTheta == 4321)
        #expect(configuration.thinker.text.ropeScaling?.mropeSection == [3, 3, 2])
    }

    @Test func processorIncludesAudioAndRetainsSidecarsAndModelMetadata() throws {
        let generated = Qwen3OmniMessageGenerator().generate(
            message: .user("Describe", audios: [.array(MLXArray.zeros([1000]))]))
        let parts = try #require(generated["content"] as? [[String: String]])
        #expect(parts.filter { $0["type"] == "audio" }.count == 1)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{\"feature_size\":8,\"sampling_rate\":8000,\"n_fft\":200,\"hop_length\":80}".utf8)
            .write(to: directory.appendingPathComponent("audio_preprocessor_config.json"))
        let data = try Qwen3OmniProcessorConfiguration.mergedData(
            directory: directory,
            modelConfiguration: Data(
                contentsOf: #require(
                    Bundle.module.url(forResource: "qwen3-omni-reference", withExtension: "json"))),
            selected: Data("{\"image_mean\":[0.1,0.2,0.3]}".utf8))
        let c = try JSONDecoder().decode(Qwen3OmniProcessorConfiguration.self, from: data)
        #expect(c.melBins == 8)
        #expect(c.sampleRate == 8000)
        #expect(c.hop == 80)
        #expect(c.vision.patchSize == 2)
        #expect(c.positions.imageToken == 60)
        #expect(c.positions.audioEndToken == 59)
    }

    @Test func rawFusedAndPerExpertCheckpointsMapToSharedCore() throws {
        let model = Qwen3Omni(try configuration())
        let gate = MLXArray(0 ..< 4 * 128 * 64).asType(.float32).reshaped(4, 128, 64)
        let down = MLXArray.zeros([4, 64, 64])
        let mapped = model.sanitize(weights: [
            "thinker.model.layers.0.mlp.experts.gate_up_proj": gate,
            "thinker.model.layers.0.mlp.experts.down_proj": down,
            "talker.dummy": MLXArray(1), "code2wav.dummy": MLXArray(1),
        ])
        #expect(mapped["talker.dummy"] == nil)
        try close(
            #require(mapped["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]),
            gate[0..., ..<64])
        try close(
            #require(mapped["language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"]),
            gate[0..., 64...])
        let perExpert = Dictionary(
            uniqueKeysWithValues: (0 ..< 4).map { index in
                ("thinker.model.layers.0.mlp.experts.\(index).gate_proj.weight", gate[index, ..<64])
            })
        let stacked = model.sanitize(weights: perExpert)
        try close(
            #require(stacked["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]),
            gate[0..., ..<64])
        let normalized = try model.prepareCheckpoint(
            .init(
                weights: [:],
                perLayerQuantization: .init(
                    quantization: .init(groupSize: 32, bits: 4),
                    perLayerQuantization: [
                        "thinker.model.layers.0.mlp.gate": .quantize(.init(groupSize: 32, bits: 8)),
                        "thinker.model.layers.0.mlp.experts.gate_up_proj": .quantize(
                            .init(groupSize: 32, bits: 4)),
                        "talker.model": .skip,
                    ])))
        #expect(
            normalized.perLayerQuantization?.quantization(
                layer: "language_model.model.layers.0.mlp.gate")?.bits == 8)
        #expect(normalized.perLayerQuantization?.perLayerQuantization["talker.model"] == nil)
        #expect(
            normalized.perLayerQuantization?.perLayerQuantization[
                "language_model.model.layers.0.mlp.switch_mlp.up_proj"] != nil)
    }

    @Test func audioVisionAndDeepstackMatchPinnedPython() throws {
        try Device.withDefaultDevice(.cpu) {
            let f = try fixture()
            let config = try configuration()
            let audio = Qwen3OmniAudio.Model(config.thinker.audio)
            try load(audio, prefix: "audio", fixture: f)
            try close(
                audio(value("audio.input", f), lengths: [101, 137]), value("audio.output", f),
                tolerance: 4e-5)
            let vision = Qwen3VLVision.VisionModel(config.thinker.vision)
            try load(vision, prefix: "vision", fixture: f)
            let (features, residuals) = vision(
                try value("vision.input", f), gridTHW: [THW(1, 4, 4), THW(2, 4, 4)])
            try close(features, value("vision.output", f))
            for index in residuals.indices {
                try close(residuals[index], value("vision.deepstack\(index)", f))
            }
        }
    }

    @Test func interleavedMRopeAndBatchedDeepstackMatchPinnedPython() throws {
        try Device.withDefaultDevice(.cpu) {
            let f = try fixture()
            let language = Qwen3OmniLanguage(try configuration().thinker.text)
            try load(language, prefix: "language", fixture: f)
            let residual = try value("language.deepstack", f)
            let hidden = language.model(
                try value("language.tokens", f), cache: nil, inputEmbeddings: nil,
                mask: nil, positionIds: try value("language.positions", f),
                visualMask: MLXArray.ones([2, 5], dtype: .bool),
                deepstackEmbeds: (0 ..< 2).map { residual[0..., 0..., $0, 0...].reshaped(-1, 64) })
            try close(language.logits(hidden), value("language.output", f))
        }
    }

    @Test func whisperFeaturesMatchTransformers() throws {
        try Device.withDefaultDevice(.cpu) {
            let f = try fixture()
            let c = try JSONDecoder().decode(
                Qwen3OmniProcessorConfiguration.self, from: Data("{\"feature_size\":8}".utf8))
            try close(
                Qwen3OmniWhisperFeatures(config: c)(value("whisper.samples", f)),
                value("whisper.output", f), tolerance: 3e-5)
        }
    }

    @Test func fullMultimodalChunkingMatchesPythonAndStateIsCacheOwned() throws {
        try Device.withDefaultDevice(.cpu) {
            let f = try fixture()
            let model = try loadedModel(f)
            let input = try mediaInput(f)
            let reference = try value("multimodal.output", f)
            for step in [1, 7, 512] {
                let cache = try model.newCache(parameters: nil)
                guard
                    case .logits(let output) = try model.prepare(
                        input, cache: cache, state: nil, prefill: .init(stepSize: step))
                else {
                    Issue.record("Expected media prefill logits")
                    return
                }
                try close(output.logits[0, -1], reference[0, -1], tolerance: 2e-4)
                #expect(output.state == nil)
                #expect(cache.allSatisfy { $0.offset == input.text.tokens.dim(1) })
                let copied = cache.map { $0.copy() }
                let token = LMInput.Text(tokens: MLXArray([Int32(19)]).reshaped(1, 1))
                try close(
                    model(token, cache: cache, state: nil).logits,
                    model(token, cache: copied, state: nil).logits)
            }
        }
    }

    @Test func timeCoordinatesAndSynchronizedInterleavingAreExact() throws {
        let c = try configuration().thinker
        let f = try fixture()
        let actual = try Qwen3OmniPositions.make(
            tokens: value("multimodal.tokens", f), mask: nil,
            images: [THW(1, 4, 4)], videos: [THW(2, 4, 4)], secondsPerVideoGrid: [1],
            audioLengths: [101], config: .init(c))
        try close(actual.positions, value("multimodal.positions", f))
        // Two video tokens at t=0/1.5s interleaved with three audio positions.
        let tokens = MLXArray([56, 58, 61, 62, 62, 62, 61, 59, 57]).reshaped(1, -1)
        let positions = try Qwen3OmniPositions.make(
            tokens: tokens, mask: nil,
            images: [], videos: [THW(2, 2, 2)], secondsPerVideoGrid: [1.5],
            audioLengths: [17], config: .init(c))
        #expect(positions.positions[0, 0, 6].item(Float.self) == 21.5)
        #expect(positions.positions[0, 0, 7].item(Float.self) == 22.5)
        #expect(throws: (any Error).self) {
            _ = try Qwen3OmniPositions.make(
                tokens: tokens, mask: nil,
                images: [], videos: [THW(2, 2, 2)], secondsPerVideoGrid: [0],
                audioLengths: [17], config: .init(c))
        }
    }

    @Test func raggedPrefillDecodeFilterExtractAndMediaAdoptionMatchSolo() throws {
        try Device.withDefaultDevice(.cpu) {
            let f = try fixture()
            let model = try loadedModel(f)
            let prompts: [[Int32]] = [[3, 4, 5], [6, 7, 8, 9, 10, 11]]
            let singles = try prompts.map { _ in try model.newCache(parameters: nil) }
            var expected: [MLXArray] = []
            for (row, prompt) in prompts.enumerated() {
                expected.append(
                    model(
                        LMInput.Text(tokens: MLXArray(prompt).reshaped(1, -1)), cache: singles[row],
                        state: nil
                    ).logits[0, -1])
            }
            let factories = try makeBatchedCacheFactories(for: model.newCache(parameters: nil))
            let batched = factories.map { $0([0, 0]) }
            let caches = batched.map { $0 as KVCache }
            for cache in batched { cache.prepare(lengths: [3, 6]) }
            let padded = prompts.flatMap { $0 + Array(repeating: 0, count: 6 - $0.count) }
            let output = model(
                LMInput.Text(tokens: MLXArray(padded).reshaped(2, 6)), cache: caches, state: nil
            ).logits
            for cache in batched { cache.finalize() }
            for row in 0 ..< 2 { try close(output[row, prompts[row].count - 1], expected[row]) }
            for cache in batched { cache.filter(batchIndices: MLXArray([1, 0])) }
            for step in 0 ..< 5 {
                let ids = [Int32(20 + step), Int32(30 + step)]
                let output = model(
                    LMInput.Text(tokens: MLXArray(ids).reshaped(2, 1)), cache: caches, state: nil
                ).logits
                for (row, original) in [1, 0].enumerated() {
                    let solo = model(
                        LMInput.Text(tokens: MLXArray([ids[row]]).reshaped(1, 1)),
                        cache: singles[original], state: nil
                    ).logits
                    try close(output[row ..< row + 1], solo)
                }
            }
            let extracted = batched.map { $0.extract(0) }
            let next = LMInput.Text(tokens: MLXArray([Int32(42)]).reshaped(1, 1))
            try close(
                model(next, cache: extracted, state: nil).logits,
                model(next, cache: singles[1], state: nil).logits)

            let mediaCache = try model.newCache(parameters: nil)
            _ = try model.prepare(
                mediaInput(f), cache: mediaCache, state: nil, prefill: .init(stepSize: 7))
            let references = [mediaCache.map { $0.copy() }, singles[0].map { $0.copy() }]
            func adopt(_ cache: KVCache) throws -> any BatchedCache {
                if let list = cache as? CacheList {
                    return BatchedCacheList(caches: [
                        try adopt(list[0]), try #require(list[1].copy() as? ArraysCache),
                    ])
                }
                return BatchKVCache.fromSingle(try #require(cache as? KVCacheSimple))
            }
            let mixed = try mediaCache.map(adopt)
            for index in mixed.indices { mixed[index].extend(other: try adopt(singles[0][index])) }
            for step in 0 ..< 4 {
                let token = Int32(45 + step)
                let output = model(
                    LMInput.Text(tokens: MLXArray([token, token]).reshaped(2, 1)),
                    cache: mixed.map { $0 as KVCache }, state: nil
                ).logits
                for row in 0 ..< 2 {
                    try close(
                        output[row ..< row + 1],
                        model(
                            LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1)),
                            cache: references[row], state: nil
                        ).logits)
                }
            }
        }
    }

    @Test(arguments: [false, true]) func singletonShardingAndCheckpointMappings(quantized: Bool)
        throws
    {
        try Device.withDefaultDevice(.cpu) {
            let model = try loadedModel(fixture())
            if quantized { quantize(model: model.language, groupSize: 32, bits: 4) }
            let tokens = LMInput.Text(tokens: MLXArray([1, 2, 3, 4]).reshaped(1, 4))
            let expected = model(tokens, cache: nil, state: nil).logits
            eval(expected)
            try model.shard(group: MLXDistributed.initialize())
            try close(model(tokens, cache: nil, state: nil).logits, expected)
            #expect(throws: TensorParallelError.alreadySharded) {
                try model.shard(group: MLXDistributed.initialize())
            }
        }
    }

    private func fixture() throws -> [String: MLXArray] {
        try loadArrays(
            url: #require(
                Bundle.module.url(forResource: "qwen3-omni-reference", withExtension: "safetensors")
            ))
    }
    private func configuration() throws -> Qwen3OmniConfiguration {
        try JSONDecoder().decode(
            Qwen3OmniConfiguration.self,
            from: Data(
                contentsOf:
                    #require(
                        Bundle.module.url(
                            forResource: "qwen3-omni-reference", withExtension: "json"))))
    }
    private func loadedModel(_ fixture: [String: MLXArray]) throws -> Qwen3Omni {
        let model = Qwen3Omni(try configuration())
        try load(model.audio, prefix: "audio", fixture: fixture)
        try load(model.vision, prefix: "vision", fixture: fixture)
        try load(model.language, prefix: "language", fixture: fixture)
        return model
    }
    private func mediaInput(_ f: [String: MLXArray]) throws -> LMInput {
        LMInput(
            text: .init(tokens: try value("multimodal.tokens", f)),
            image: .init(pixels: try value("vision.input", f)[..<16], frames: [THW(1, 4, 4)]),
            video: .init(pixels: try value("vision.input", f)[16...], frames: [THW(2, 4, 4)]),
            audio: .init(
                features: try value("audio.input", f)[0..., ..<101].expandedDimensions(axis: 0)))
    }
    private func value(_ name: String, _ f: [String: MLXArray]) throws -> MLXArray {
        try #require(f[name])
    }
    private func load(_ model: Module, prefix: String, fixture: [String: MLXArray]) throws {
        let marker = prefix + ".weights."
        let weights = fixture.filter { $0.key.hasPrefix(marker) }.map {
            (String($0.key.dropFirst(marker.count)), $0.value)
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
    }
    private func close(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 1e-4) throws {
        #expect(actual.shape == expected.shape)
        guard actual.shape == expected.shape else { return }
        #expect(
            abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
                < tolerance)
    }
}
