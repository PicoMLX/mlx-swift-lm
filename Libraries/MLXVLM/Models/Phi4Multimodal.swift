// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm phi4mm at 4693cd0 (MIT).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Phi-4 Multimodal text, image and audio understanding.
/// Media encoders are replicated; tensor parallelism splits the language decoder.
public final class Phi4Multimodal: Module, VLMModel, KVCacheDimensionProvider, TensorParallelModel {
    public let config: Phi4MultimodalConfiguration
    @ModuleInfo(key: "language_model") package var language: Phi4MultimodalLanguage
    @ModuleInfo(key: "vision_tower") package var vision: Phi4MultimodalVision
    @ModuleInfo(key: "mm_projector") package var projector: Phi4MultimodalProjector
    @ModuleInfo(key: "audio_encoder") package var audio: Phi4MultimodalAudio
    @ModuleInfo(key: "audio_projection") package var audioProjection: Phi4MultimodalAudioProjection
    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] {
        language.model.layers.map { $0.attention.key.base.weight.dim(0) / config.headDimension }
    }
    public var headDim: IntOrPair { .init(config.headDimension) }
    public var loraLayers: [Module] { language.model.layers }

    public init(_ config: Phi4MultimodalConfiguration) {
        self.config = config
        _language.wrappedValue = .init(config)
        _vision.wrappedValue = .init(config.vision)
        _projector.wrappedValue = .init(input: config.vision.hiddenSize, output: config.hiddenSize)
        _audio.wrappedValue = .init(config.audio)
        _audioProjection.wrappedValue = .init(
            input: config.audio.dimension, output: config.hiddenSize)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try (0 ..< config.layers).map { _ in
            let modes = ArraysCache(size: 1)
            modes[0] = MLXArray([Float(1), 0], [1, 2])
            return CacheList(try makeAttentionKVCache(parameters: parameters), modes)
        }
    }

    /// Adapter routing lives in an ordinary composite cache and follows its row transforms.
    package func setModes(_ modes: MLXArray, cache: [any KVCache]) throws {
        guard modes.ndim == 2, modes.dim(1) == 2 else {
            throw VLMError.processing("Phi-4 modes must have shape [batch, 2]")
        }
        for layer in cache {
            guard let list = layer as? CacheList, let state = list[1] as? ArraysCache else {
                throw VLMError.processing("Phi-4 requires its composite modality cache")
            }
            state[0] = modes
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        language(language.model.embedding(inputs), cache: cache)
    }

    package func inputEmbeddings(_ input: LMInput, cache: [any KVCache]) throws -> MLXArray {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens[.newAxis, 0...] }
        guard tokens.ndim == 2 else { throw VLMError.processing("Phi-4 expects a token matrix") }
        let b = tokens.dim(0)
        let n = tokens.dim(1)
        let ids = tokens.asArray(Int.self)
        var features: [Int: MLXArray] = [:]
        if let image = input.image {
            guard image.pixels.ndim == 3, let shapes = image.frames,
                shapes.count == image.pixels.dim(0)
            else {
                throw VLMError.processing("Phi-4 image patches require one spatial grid per image")
            }
            var imageFeatures: [MLXArray] = []
            for (i, shape) in shapes.enumerated() {
                let count = shape.h * shape.w
                guard shape.t == 1, count > 0, count <= image.pixels.dim(1) else {
                    throw VLMError.processing("Invalid Phi-4 image patch grid")
                }
                imageFeatures.append(
                    projector(
                        vision(
                            image.pixels[i ..< i + 1, ..<count, 0...], height: shape.h,
                            width: shape.w))[0])
            }
            if !imageFeatures.isEmpty { features[config.imageToken] = concatenated(imageFeatures) }
        }
        if let sound = input.audio {
            guard sound.features.ndim == 3, sound.features.dim(2) == config.audio.inputSize else {
                throw VLMError.processing(
                    "Phi-4 audio must contain log-mel features [audio, time, mel]")
            }
            var audioFeatures: [MLXArray] = []
            for i in 0 ..< sound.features.dim(0) {
                let length =
                    sound.mask.map { $0[i].asType(.int32).sum().item(Int.self) }
                    ?? sound.features.dim(1)
                guard length > 0, length <= sound.features.dim(1) else {
                    throw VLMError.processing("Invalid Phi-4 audio feature length")
                }
                audioFeatures.append(
                    audioProjection.speech(audio(sound.features[i ..< i + 1, ..<length, 0...]))[0])
            }
            if !audioFeatures.isEmpty { features[config.audioToken] = concatenated(audioFeatures) }
        }
        var safeIDs = ids
        var rowModes = phi4Modes(cache.first, batch: b).asArray(Float.self)
        for row in 0 ..< b {
            let rowIDs = ids[(row * n) ..< ((row + 1) * n)]
            if features.keys.contains(where: { rowIDs.contains($0) }) {
                rowModes[row * 2] = 0
                rowModes[row * 2 + 1] = 0
            }
        }
        for (token, encoded) in features {
            let indices = ids.indices.filter { ids[$0] == token }
            guard indices.count == encoded.dim(0) else {
                throw VLMError.processing(
                    "Phi-4 media token count does not match its encoder features")
            }
            for index in indices {
                safeIDs[index] = 0
                rowModes[(index / n) * 2 + (token == config.imageToken ? 0 : 1)] = 1
            }
        }
        guard safeIDs.allSatisfy({ $0 >= 0 && $0 < config.vocabularySize }) else {
            throw VLMError.processing(
                "Phi-4 received an unmatched media placeholder or invalid token")
        }
        var embedding = language.model.embedding(MLXArray(safeIDs).reshaped(b, n))
        for (token, encoded) in features {
            let indices = ids.indices.filter { ids[$0] == token }
            let flat = embedding.reshaped(b * n, config.hiddenSize)
            flat[MLXArray(indices)] = encoded.asType(flat.dtype)
            embedding = flat.reshaped(b, n, config.hiddenSize)
        }
        if !features.isEmpty {
            // A continuation without media retains the cache's existing adapter mode.
            try setModes(MLXArray(rowModes, [b, 2]), cache: cache)
        }
        return embedding
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.video == nil else {
            throw VLMError.processing("Phi-4 Multimodal accepts images and audio, not video")
        }
        let embeddings = try inputEmbeddings(input, cache: cache)
        if let lengths = input.text.sequenceLengths {
            for layer in cache { layer.prepare(lengths: lengths) }
        }
        let count = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: count) { range in
            _ = language(embeddings[0..., range, 0...], cache: cache)
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        let logits = language(embeddings[0..., processed..., 0...], cache: cache)
        for layer in cache { layer.finalize() }
        prefill.progress?(count, count)
        return .logits(LMOutput(logits: logits))
    }

    public func shard(group: MLXDistributed.Group) throws {
        try validateShardable(self)
        try requireDivisible(
            [
                "attention heads": config.heads, "key-value heads": config.kvHeads,
                "MLP width": config.intermediateSize,
            ], by: group.size)
        for layer in language.model.layers {
            for projection in [
                layer.attention.query, layer.attention.key, layer.attention.value,
                layer.mlp.gate, layer.mlp.up,
            ] { try projection.shard(column: true, group: group) }
            try layer.attention.output.shard(column: false, group: group)
            try layer.mlp.down.shard(column: false, group: group)
        }
    }
}
