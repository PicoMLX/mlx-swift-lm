// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

package func tinyPhi4Multimodal(adapters: Bool = false) throws -> Phi4Multimodal {
    var c = Phi4MultimodalConfiguration()
    c.vocabularySize = 64
    c.hiddenSize = 32
    c.layers = 2
    c.intermediateSize = 64
    c.heads = 4
    c.kvHeads = 2
    c.partialRotaryFactor = 0.5
    c.maxPositionEmbeddings = 128
    c.originalMaxPositionEmbeddings = 128
    c.audioToken = 60
    c.minPatches = 1
    c.maxPatches = 16
    c.vision.hiddenSize = 16
    c.vision.intermediateSize = 32
    c.vision.heads = 2
    c.vision.layers = 2
    c.vision.patchSize = 2
    c.vision.imageSize = 4
    var audio = Phi4MultimodalAudioConfiguration()
    audio.dimension = 8
    audio.heads = 2
    audio.layers = 1
    audio.intermediateSize = 16
    audio.inputSize = 8
    audio.reduction = 2
    audio.convChannels = 4
    audio.pointwiseChannels = 8
    audio.depthwiseChannels = 8
    audio.relativeDistance = 3
    c.audioProcessor = .init(config: audio)
    if adapters {
        var adapter = Phi4MultimodalAdapterConfiguration()
        adapter.rank = 2
        adapter.alpha = 2
        c.visionAdapter = adapter
        c.speechAdapter = adapter
    }
    try c.validateModelConfiguration()
    return Phi4Multimodal(c)
}

@Suite(.serialized)
struct Phi4MultimodalTests {
    @Test("A ragged mixed-modality Phi-4 batch preserves each row's adapter")
    func mixedModeBatch() throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(817)
            let model = try tinyPhi4Multimodal(adapters: true)
            let prompts: [[Int32]] = [[1, 2, 3], [4, 5, 6, 7, 8], [9, 10]]
            let modes = MLXArray([Float(0), 0, 1, 0, 0, 1], [3, 2])
            let singles = try prompts.indices.map { row -> [any KVCache] in
                let cache = try model.newCache(parameters: nil)
                try model.setModes(modes[row ..< row + 1], cache: cache)
                return cache
            }
            let expected = prompts.enumerated().map { row, prompt in
                model(MLXArray(prompt).reshaped(1, prompt.count), cache: singles[row])[0, -1]
            }
            let emptyRows = try prompts.map { _ in try model.newCache(parameters: nil) }
            let batched = try emptyRows[0].indices.map { layer in
                try mergeCaches(emptyRows.map { $0[layer] })
            }
            let caches = batched.map { $0 as any KVCache }
            try model.setModes(modes, cache: caches)
            batched.forEach { $0.prepare(lengths: prompts.map(\.count)) }
            let padded = prompts.flatMap { $0 + Array(repeating: 0, count: 5 - $0.count) }
            let actual = model(MLXArray(padded).reshaped(3, 5), cache: caches)
            batched.forEach { $0.finalize() }
            for row in prompts.indices {
                #expect(maxAbsDifference(actual[row, prompts[row].count - 1], expected[row]) < 1e-4)
            }
            for token: Int32 in [11, 12, 13] {
                let output = model(MLXArray([token, token, token]).reshaped(3, 1), cache: caches)
                for row in prompts.indices {
                    let expected = model(MLXArray([token]).reshaped(1, 1), cache: singles[row])
                    #expect(maxAbsDifference(output[row ..< row + 1], expected) < 1e-4)
                }
            }
            batched.forEach { $0.filter(batchIndices: MLXArray([Int32(2), 0])) }
            let filtered = model(MLXArray([Int32(14), 14]).reshaped(2, 1), cache: caches)
            for (row, original) in [2, 0].enumerated() {
                let expected = model(MLXArray([Int32(14)]).reshaped(1, 1), cache: singles[original])
                #expect(maxAbsDifference(filtered[row ..< row + 1], expected) < 1e-4)
            }
            let extracted = batched.map { $0.extract(0) }
            let token = MLXArray([Int32(15)]).reshaped(1, 1)
            #expect(
                maxAbsDifference(model(token, cache: extracted), model(token, cache: singles[2]))
                    < 1e-4)
            let restored = try model.newCache(parameters: nil).map { try mergeCaches([$0]) }
            let restoredKV = restored.map { $0 as any KVCache }
            try model.setModes(MLXArray([Float(1), 1], [1, 2]), cache: restoredKV)
            eval(model(MLXArray([Int32(2), 3]).reshaped(1, 2), cache: restoredKV))
            let clone = restored.map { $0.copy() }
            for i in batched.indices { batched[i].extend(other: restored[i]) }
            let output = model(MLXArray([Int32(16), 16, 16]).reshaped(3, 1), cache: caches)
            #expect(
                maxAbsDifference(
                    output[2 ..< 3], model(MLXArray([Int32(16)]).reshaped(1, 1), cache: clone))
                    < 1e-4)
        }
    }

    @Test("Image and audio features replace exactly their reserved token positions")
    func mediaPreparation() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyPhi4Multimodal(adapters: true)
            let cache = try model.newCache(parameters: nil)
            let image = MLXArray.zeros([1, 4, 12])
            let sound = MLXArray.zeros([1, 6, 8])
            let input = LMInput(
                text: .init(
                    tokens: MLXArray([Int32(1), -200, -200, -200, -200, 60, 60, 60, 2]).reshaped(
                        1, 9)),
                image: .init(pixels: image, frames: [THW(1, 2, 2)]), audio: .init(features: sound))
            let embeddings = try model.inputEmbeddings(input, cache: cache)
            #expect(embeddings.shape == [1, 9, 32])
            #expect(phi4Modes(cache[0], batch: 1).asArray(Float.self) == [1, 1])
            let imageFeatures = model.projector(model.vision(image, height: 2, width: 2))
            #expect(maxAbsDifference(embeddings[0..., 1 ..< 5, 0...], imageFeatures) == 0)
            let audioFeatures = model.audioProjection.speech(model.audio(sound))
            #expect(maxAbsDifference(embeddings[0..., 5 ..< 8, 0...], audioFeatures) == 0)
            let mismatch = LMInput(
                text: .init(tokens: MLXArray([Int32(-200)]).reshaped(1, 1)), image: input.image)
            #expect(throws: (any Error).self) { try model.inputEmbeddings(mismatch, cache: cache) }
        }
    }

    @Test("Fused projection rows and quantization auxiliaries split by semantic Q/K/V sections")
    func fusedCheckpoint() throws {
        let model = try tinyPhi4Multimodal()
        let prefix = "model.layers.0.self_attn.qkv_proj"
        let marker = MLXArray((0 ..< 64).map(Float.init))[0..., .newAxis]
        let source = [
            prefix + ".weight": broadcast(marker, to: [64, 4]), prefix + ".scales": marker,
            prefix + ".biases": marker,
        ]
        let weights = try model.sanitize(weights: source)
        for (projection, rows) in [
            ("q_proj", 0 ..< 32), ("k_proj", 32 ..< 48), ("v_proj", 48 ..< 64),
        ] {
            for suffix in ["weight", "scales", "biases"] {
                let key = "language_model.model.layers.0.self_attn.\(projection).base.\(suffix)"
                let tensor = try #require(weights[key])
                #expect(tensor.dim(0) == rows.count)
                #expect(tensor[0].flattened()[0].item(Float.self) == Float(rows.lowerBound))
            }
        }
    }

    @Test("Quantized checkpoints can replace linears throughout both media towers")
    func quantizedMediaTowers() throws {
        var configuration = try tinyPhi4Multimodal().config
        configuration.vision.hiddenSize = 32
        configuration.vision.intermediateSize = 64
        var audio = configuration.audio
        audio.dimension = 32
        audio.intermediateSize = 64
        audio.pointwiseChannels = 32
        audio.depthwiseChannels = 32
        audio.convChannels = 8
        configuration.audioProcessor = .init(config: audio)
        let model = Phi4Multimodal(configuration)
        quantize(
            model: model, groupSize: 32, bits: 4,
            filter: { _, module in
                guard let linear = module as? Linear else { return false }
                return linear.weight.dim(-1) % 32 == 0
            })
        #expect(model.vision.model.encoder.layers[0].mlp.fc1 is QuantizedLinear)
        #expect(model.audio.embed.out is QuantizedLinear)
        #expect(model.audio.encoders[0].feedIn.first.linear is QuantizedLinear)
        let cache = try model.newCache(parameters: nil)
        let input = LMInput(
            text: .init(
                tokens: MLXArray([Int32(1), -200, -200, -200, -200, 60, 60, 60, 2]).reshaped(1, 9)),
            image: .init(pixels: MLXArray.zeros([1, 4, 12]), frames: [THW(1, 2, 2)]),
            audio: .init(features: MLXArray.zeros([1, 6, 8])))
        let embedding = try model.inputEmbeddings(input, cache: cache)
        #expect(embedding.shape == [1, 9, 32])
        #expect(all(isFinite(embedding)).item(Bool.self))
    }

    @Test("Both vision and speech inputs require their declared placeholders")
    func missingMediaFailsClosed() throws {
        let model = try tinyPhi4Multimodal()
        let cache = try model.newCache(parameters: nil)
        #expect(throws: (any Error).self) {
            try model.inputEmbeddings(
                LMInput(tokens: MLXArray([Int32(-200)]).reshaped(1, 1)), cache: cache)
        }
    }

    @Test("Tensor parallelism preserves float and 4-bit decoder outputs and rejects reshards")
    func sharding() throws {
        try Device.withDefaultDevice(.cpu) {
            for quantized in [false, true] {
                let model = try tinyPhi4Multimodal(adapters: true)
                if quantized {
                    quantize(
                        model: model, groupSize: 32, bits: 4,
                        filter: { path, _ in path.hasSuffix(".base") })
                }
                let cache = try model.newCache(parameters: nil)
                try model.setModes(MLXArray([Float(1), 1], [1, 2]), cache: cache)
                let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
                let whole = model(tokens, cache: cache)
                eval(whole)
                try model.shard(group: MLXDistributed.initialize())
                let shardedCache = try model.newCache(parameters: nil)
                try model.setModes(MLXArray([Float(1), 1], [1, 2]), cache: shardedCache)
                #expect(maxAbsDifference(model(tokens, cache: shardedCache), whole) < 1e-4)
                #expect(throws: TensorParallelError.alreadySharded) {
                    try model.shard(group: MLXDistributed.initialize())
                }
            }
        }
    }
}
