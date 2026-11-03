// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

@Suite(.serialized)
struct MiniCPMReferenceTests {
    @Test(
        "All three MiniCPM paths match pinned Python vision, language, canvas and audio fixtures",
        arguments: ["minicpmo", "minicpmv4_6", "minicpmv4_7"])
    func python(_ variant: String) throws {
        // Fixtures use Metal. Matching that backend avoids CPU/GPU LayerNorm rounding drift.
        try Device.withDefaultDevice(.gpu) {
            let url = try #require(
                Bundle.module.url(forResource: variant + "-reference", withExtension: "safetensors")
            )
            let configURL = try #require(
                Bundle.module.url(forResource: variant + "-reference", withExtension: "json"))
            let config = try JSONDecoder().decode(
                MiniCPMConfiguration.self, from: Data(contentsOf: configURL))
            let fixture = try loadArrays(url: url)
            func value(_ key: String) throws -> MLXArray { try #require(fixture[key]) }
            let model = MiniCPM(config)
            let weights = Dictionary(
                uniqueKeysWithValues: fixture.compactMap { key, value in
                    key.hasPrefix("weights.") ? (String(key.dropFirst(8)), value) : nil
                })
            let checkpoint = try model.prepareCheckpoint(
                .init(weights: weights, metadata: ["format": "mlx"]))
            try model.update(
                parameters: ModuleParameters.unflattened(checkpoint.weights), verify: [.all])
            let tokens = try value("input.tokens")
            let cache = try model.newCache(parameters: nil)
            #expect(
                maxAbsDifference(model(tokens, cache: cache), try value("expected.text_prefill"))
                    < 4e-4)
            #expect(
                maxAbsDifference(
                    model(MLXArray([Int32(4)]).reshaped(1, 1), cache: cache),
                    try value("expected.text_decode")) < 4e-4)
            let patches = try value("input.patches")
            if variant == "minicpmo" {
                let vision = model.vision(patches, height: 4, width: 4).0
                #expect(maxAbsDifference(vision, try value("expected.vision")) < 4e-4)
            }
            let projected = try #require(
                model.visualFeatures(patches[.newAxis, 0..., 0...], grids: [THW(1, 4, 4)]).first)
            let visualDifference = maxAbsDifference(
                projected, try value("expected.visual_projector"))

            #expect(visualDifference < 4e-4)
            let nativeTokens = try value("input.media_tokens").asArray(Int.self)
            let internalTokens = nativeTokens.map { $0 == 60 ? MiniCPM.imageToken : $0 }
            let input = LMInput(
                text: .init(tokens: MLXArray(internalTokens).reshaped(1, internalTokens.count)),
                image: .init(pixels: patches[.newAxis, 0..., 0...], frames: [THW(1, 4, 4)]))
            #expect(
                maxAbsDifference(
                    try model.inputEmbeddings(input), try value("expected.media_embeddings")) < 4e-4
            )
            let mediaCache = try model.newCache(parameters: nil)
            #expect(
                maxAbsDifference(
                    try miniCPMPrepared(model, input, cache: mediaCache),
                    try value("expected.media_prefill")) < 5e-4)
            #expect(
                maxAbsDifference(
                    model(MLXArray([Int32(4)]).reshaped(1, 1), cache: mediaCache),
                    try value("expected.media_decode")) < 5e-4)
            if !config.isOmni {
                let projected4 = try #require(
                    model.visualFeatures(
                        patches[.newAxis, 0..., 0...], grids: [THW(1, 4, 4)],
                        downsampleModes: ["4x"]
                    ).first)
                #expect(
                    maxAbsDifference(projected4, try value("expected.visual_projector_4x")) < 4e-4)
                let native4 = try value("input.media_tokens_4x").asArray(Int.self)
                let ids4 = native4.map { $0 == 60 ? MiniCPM.imageToken : $0 }
                let input4 = LMInput(
                    text: .init(tokens: MLXArray(ids4).reshaped(1, ids4.count)),
                    image: .init(pixels: patches[.newAxis, 0..., 0...], frames: [THW(1, 4, 4)]))
                #expect(
                    maxAbsDifference(
                        try model.inputEmbeddings(input4), try value("expected.media_embeddings_4x")
                    ) < 4e-4)
                let cache4 = try model.newCache(parameters: nil)
                #expect(
                    maxAbsDifference(
                        try miniCPMPrepared(model, input4, cache: cache4),
                        try value("expected.media_prefill_4x")) < 5e-4)
                #expect(
                    maxAbsDifference(
                        model(MLXArray([Int32(4)]).reshaped(1, 1), cache: cache4),
                        try value("expected.media_decode_4x")) < 5e-4)
            }
            if let audio = model.audio, let projection = model.audioProjection {
                let encoded = audio(try value("input.audio"), length: 12)
                let audioDifference = maxAbsDifference(encoded, try value("expected.audio"))

                #expect(audioDifference < 4e-4)
                #expect(
                    maxAbsDifference(
                        projection(encoded, length: 12, step: 2),
                        try value("expected.audio_projector")) < 4e-4)
                let mel = try MiniCPMAudioFeatures.extract(value("input.samples"), melBins: 8)
                let melDifference = maxAbsDifference(mel, try value("expected.log_mel"))
                #expect(melDifference < 4e-4)
                #expect(
                    maxAbsDifference(
                        MiniCPMImageProcessing.resize(
                            try value("input.resize"), height: 8, width: 12),
                        try value("expected.resize")) <= 1)
            }
            if let native = fixture["input.canvas_tokens"] {
                let ids = native.asArray(Int.self).map {
                    $0 == 60 ? MiniCPM.imageToken : $0 == 61 ? MiniCPM.videoToken : $0
                }
                let (positions, deltas) = try MiniCPMCanvas.positions(
                    tokens: ids, lengths: [ids.count], width: ids.count,
                    images: Array(repeating: THW(1, 4, 4), count: 5), videos: [THW(1, 4, 4)],
                    factor: 4,
                    markers: .init(
                        imageStart: 50, imageEnd: 51, sliceStart: 52, sliceEnd: 53, newline: 54))
                #expect(maxAbsDifference(positions, try value("expected.canvas_layout")) == 0)
                #expect(maxAbsDifference(deltas, try value("expected.canvas_deltas")) == 0)
            }
        }
    }
}
