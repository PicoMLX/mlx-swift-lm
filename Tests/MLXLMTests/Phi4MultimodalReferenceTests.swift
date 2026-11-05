// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

@Suite(.serialized)
struct Phi4MultimodalReferenceTests {
    private struct PixelFixture: Decodable {
        struct Case: Decodable {
            let name: String
            let height: Int
            let width: Int
            let targetHeight: Int
            let targetWidth: Int
            let pixels: [UInt8]
            let resized: [UInt8]
        }
        let cases: [Case]
    }

    private func pixelCase(_ name: String) throws -> PixelFixture.Case {
        let url = try #require(
            Bundle.module.url(forResource: "phi4mm-pixels", withExtension: "json"))
        let fixture = try JSONDecoder().decode(PixelFixture.self, from: Data(contentsOf: url))
        return try #require(fixture.cases.first { $0.name == name })
    }

    @Test(
        "RGB pixels match Pillow byte for byte after both resize passes",
        arguments: ["mild_downscale", "upscale", "asymmetric", "singleton", "identity"])
    func pillowPixels(name: String) throws {
        try Device.withDefaultDevice(.cpu) {
            let fixture = try pixelCase(name)
            let pixels =
                MLXArray(fixture.pixels).asType(.float32)
                .reshaped(1, fixture.height, fixture.width, 3).transposed(0, 3, 1, 2) / 255
            let resized = Phi4MultimodalPixelResampling.resize(
                pixels, height: fixture.targetHeight, width: fixture.targetWidth)
            let actual = floor(resized * 255 + 0.5).asType(.uint8).transposed(0, 2, 3, 1)
                .flattened()
            #expect(
                arrayEqual(actual, MLXArray(fixture.resized)).item(Bool.self), "\(fixture.name)")
        }
    }

    @Test("The image processor uses the RGB filter for a high-frequency 1.2x downscale")
    func pillowProcessorPixels() throws {
        try Device.withDefaultDevice(.cpu) {
            let fixture = try pixelCase("mild_downscale")
            var configuration = Phi4MultimodalProcessorConfiguration()
            configuration.patchSize = 1
            configuration.minimumPatches = fixture.targetHeight * fixture.targetWidth
            configuration.maximumPatches = configuration.minimumPatches
            configuration.imageMean = [0, 0, 0]
            configuration.imageStd = [1, 1, 1]
            let processor = Phi4MultimodalProcessor(configuration, tokenizer: PixelTestTokenizer())
            let image = UserInput.Image.array(
                MLXArray(fixture.pixels).reshaped(fixture.height, fixture.width, 3))
            let (patches, grid) = try processor.imagePatches(image)
            #expect(grid.h == fixture.targetHeight && grid.w == fixture.targetWidth)
            let actual = floor(patches * 255 + 0.5).asType(.uint8).flattened()
            #expect(arrayEqual(actual, MLXArray(fixture.resized)).item(Bool.self))
        }
    }

    @Test(
        "Phi-4 multimodal components and modality adapters match the pinned Python implementation")
    func pythonReference() throws {
        try Device.withDefaultDevice(.cpu) {
            let weightsURL = try #require(
                Bundle.module.url(forResource: "phi4mm-reference", withExtension: "safetensors"))
            let configurationURL = try #require(
                Bundle.module.url(forResource: "phi4mm-reference", withExtension: "json"))
            let config = try JSONDecoder().decode(
                Phi4MultimodalConfiguration.self, from: Data(contentsOf: configurationURL))
            let fixture = try loadArrays(url: weightsURL)
            let model = Phi4Multimodal(config)
            let weights = Dictionary(
                uniqueKeysWithValues: fixture.compactMap { key, value in
                    key.hasPrefix("weights.") ? (String(key.dropFirst(8)), value) : nil
                })
            let checkpoint = try model.prepareCheckpoint(
                .init(weights: weights, metadata: ["format": "mlx"]))
            try model.update(
                parameters: ModuleParameters.unflattened(checkpoint.weights), verify: [.all])
            let tokens = try #require(fixture["input.tokens"])
            for (mode, flags) in [
                ("base", [Float(0), 0]), ("vision", [Float(1), 0]),
                ("speech", [Float(0), 1]), ("both", [Float(1), 1]),
            ] {
                let cache = try model.newCache(parameters: nil)
                try model.setModes(MLXArray(flags, [1, 2]), cache: cache)
                let actual = model(tokens, cache: cache)
                #expect(
                    maxAbsDifference(actual, try #require(fixture["expected.\(mode).prefill"]))
                        < 2e-4, "\(mode) prefill")
                let next = model(MLXArray([Int32(4)]).reshaped(1, 1), cache: cache)
                #expect(
                    maxAbsDifference(next, try #require(fixture["expected.\(mode).decode"])) < 2e-4,
                    "\(mode) decode")
            }
            let probeInput = try #require(fixture["probe.projection_input"])
            let probeAttention = model.language.model.layers[0].attention
            let probeModes = MLXArray([Float(1), 0]).reshaped(1, 2)
            let projection = concatenated(
                [
                    probeAttention.query(probeInput, modes: probeModes),
                    probeAttention.key(probeInput, modes: probeModes),
                    probeAttention.value(probeInput, modes: probeModes),
                ], axis: -1)
            #expect(
                maxAbsDifference(projection, try #require(fixture["probe.projection_merged"]))
                    < 2e-4)
            #expect(
                maxAbsDifference(projection, try #require(fixture["probe.projection_unfused"]))
                    < 2e-4)
            let gpuExpected = try #require(fixture["probe.projection_unfused_gpu"])
            Device.withDefaultDevice(.gpu) {
                let gpuProjection = concatenated(
                    [
                        probeAttention.query(probeInput, modes: probeModes),
                        probeAttention.key(probeInput, modes: probeModes),
                        probeAttention.value(probeInput, modes: probeModes),
                    ], axis: -1)
                #expect(maxAbsDifference(gpuProjection, gpuExpected) < 2e-4)
            }
            let patches = try #require(fixture["input.patches"])
            let vision = model.vision(patches, height: 2, width: 2)[0]
            #expect(maxAbsDifference(vision, try #require(fixture["expected.vision"])) < 2e-4)
            #expect(
                maxAbsDifference(
                    model.projector(vision), try #require(fixture["expected.projector"])) < 2e-4)
            let rectangular = model.vision(patches[0..., ..<2, 0...], height: 1, width: 2)[0]
            #expect(
                maxAbsDifference(rectangular, try #require(fixture["expected.vision_rectangular"]))
                    < 2e-4)
            let sound = model.audio(try #require(fixture["input.audio"]))
            #expect(maxAbsDifference(sound, try #require(fixture["expected.audio"])) < 2e-4)
            #expect(
                maxAbsDifference(
                    model.audioProjection.speech(sound),
                    try #require(fixture["expected.audio_projector"])) < 2e-4)
            let mel = try Device.withDefaultDevice(.cpu) {
                try Phi4MultimodalAudioFeatures.extract(#require(fixture["input.samples"]))
            }
            let expectedMel = try #require(fixture["expected.log_mel"])
            #expect(maxAbsDifference(mel, expectedMel) < 2e-4)
            let samples = try #require(fixture["input.samples"])
            try Device.withDefaultDevice(.gpu) {
                let gpuMel = try Phi4MultimodalAudioFeatures.extract(samples)
                // Different FFT/GEMM backends: keep the strict CPU/NumPy assertion
                // above and independently bound Metal's relative feature error.
                let scale = max(abs(expectedMel).max().item(Float.self), 1)
                #expect(maxAbsDifference(gpuMel, expectedMel) / scale < 1e-4)
            }
        }
    }
}

private struct PixelTestTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
}
