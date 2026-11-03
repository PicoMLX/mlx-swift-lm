// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

@Suite(.serialized)
struct Phi4MultimodalReferenceTests {
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
