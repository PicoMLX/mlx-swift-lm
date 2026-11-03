// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm gemma3n at 4693cd0 (MIT).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public enum Gemma3nError: Error, Equatable, LocalizedError {
    case invalidConfiguration(String)
    case invalidInput(String)
    case placeholderMismatch(kind: String, placeholders: Int, features: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message): "Invalid Gemma3n configuration: \(message)"
        case .invalidInput(let message): "Invalid Gemma3n input: \(message)"
        case .placeholderMismatch(let kind, let placeholders, let features):
            "Gemma3n \(kind) has \(placeholders) prompt placeholders and \(features) feature tokens."
        }
    }
}

extension Gemma3nConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        try textConfig.validateModelConfiguration()
        guard modelType == "gemma3n", visionConfig.hiddenSize == 2048,
            visionConfig.vocabSize > 0, audioConfig.vocabSize > 0,
            visionSoftTokensPerImage == 256, audioSoftTokensPerImage > 0,
            audioConfig.hiddenSize > 0, audioConfig.hiddenSize % 2 == 0,
            audioConfig.confNumAttentionHeads > 0,
            audioConfig.hiddenSize % audioConfig.confNumAttentionHeads == 0,
            audioConfig.confAttentionChunkSize > 0,
            audioConfig.confAttentionContextLeft >= 0,
            audioConfig.confAttentionContextRight >= 0,
            audioConfig.confReductionFactor > 0,
            audioConfig.confConvKernelSize > 0,
            audioConfig.sscpConvChannelSize.count == 2,
            audioConfig.sscpConvKernelSize.count == 2,
            audioConfig.sscpConvStrideSize.count == 2,
            audioConfig.sscpConvChannelSize.allSatisfy({ $0 > 0 }),
            audioConfig.sscpConvKernelSize.allSatisfy({ $0.count == 2 && $0.allSatisfy { $0 > 0 } }
            ),
            audioConfig.sscpConvStrideSize.allSatisfy({ $0.count == 2 && $0.allSatisfy { $0 > 0 } })
        else {
            throw Gemma3nError.invalidConfiguration(
                "Unsupported tower dimensions or sampling parameters.")
        }
    }
}

final class Gemma3nMultimodalEmbedder: Module {
    @ModuleInfo var embedding: Embedding
    @ModuleInfo(key: "hard_embedding_norm") var hardNorm: Gemma3nModalityNorm
    @ModuleInfo(key: "soft_embedding_norm") var softNorm: Gemma3nModalityNorm
    @ModuleInfo(key: "embedding_projection") var projection: Linear
    @ModuleInfo(key: "embedding_post_projection_norm") var postNorm: Gemma3nModalityNorm
    let offset: Int
    let vocabularySize: Int

    init(dimensions: Int, textDimensions: Int, vocabularySize: Int, offset: Int, eps: Float) {
        self.offset = offset
        self.vocabularySize = vocabularySize
        _embedding.wrappedValue = Embedding(embeddingCount: vocabularySize, dimensions: dimensions)
        _hardNorm.wrappedValue = Gemma3nModalityNorm(dimensions, eps: eps)
        _softNorm.wrappedValue = Gemma3nModalityNorm(dimensions, eps: eps)
        _projection.wrappedValue = Linear(dimensions, textDimensions, bias: false)
        _postNorm.wrappedValue = Gemma3nModalityNorm(textDimensions, eps: eps, scale: false)
    }

    func hard(_ tokenIds: MLXArray) -> MLXArray {
        postNorm(projection(hardNorm(embedding(tokenIds - offset))))
    }

    func soft(_ features: MLXArray) -> MLXArray { postNorm(projection(softNorm(features))) }
}

func gemma3nMergeFeatures(
    _ embeddings: MLXArray, tokens: MLXArray, tokenId: Int,
    features: MLXArray, kind: String
) throws -> MLXArray {
    let positions = tokens.flattened().asArray(Int.self).enumerated().compactMap {
        $0.element == tokenId ? Int32($0.offset) : nil
    }
    let dimensions = embeddings.dim(-1)
    guard features.ndim >= 2, features.dim(-1) == dimensions,
        features.size / dimensions == positions.count
    else {
        throw Gemma3nError.placeholderMismatch(
            kind: kind, placeholders: positions.count,
            features: features.size / max(dimensions, 1))
    }
    guard !positions.isEmpty else { return embeddings }
    let flattened = embeddings.reshaped(-1, dimensions)
    flattened[MLXArray(positions)] = features.asType(embeddings.dtype).reshaped(-1, dimensions)
    return flattened.reshaped(embeddings.shape)
}

/// Gemma3n E2B/E4B with the MobileNetV5 and Conformer input towers.
public final class Gemma3n: Module, VLMModel, TensorParallelModel, KVCacheDimensionProvider {
    public let config: Gemma3nConfiguration
    public var vocabularySize: Int { config.textConfig.vocabSize }
    public var kvHeads: [Int] { languageModel.cacheHeadCounts }
    public var headDim: IntOrPair { .init(config.textConfig.headDim) }
    @ModuleInfo(key: "language_model") var languageModel: Gemma3nLanguageModel
    @ModuleInfo(key: "vision_tower") var visionTower: Gemma3nVisionModel
    @ModuleInfo(key: "audio_tower") var audioTower: Gemma3nAudioModel
    @ModuleInfo(key: "embed_vision") var embedVision: Gemma3nMultimodalEmbedder
    @ModuleInfo(key: "embed_audio") var embedAudio: Gemma3nMultimodalEmbedder

    public init(_ config: Gemma3nConfiguration) {
        self.config = config
        _languageModel.wrappedValue = Gemma3nLanguageModel(config.textConfig)
        _visionTower.wrappedValue = Gemma3nVisionModel(config.visionConfig)
        _audioTower.wrappedValue = Gemma3nAudioModel(config.audioConfig)
        _embedVision.wrappedValue = Gemma3nMultimodalEmbedder(
            dimensions: config.visionConfig.hiddenSize,
            textDimensions: config.textConfig.hiddenSize,
            vocabularySize: config.visionConfig.vocabSize, offset: config.visionConfig.vocabOffset,
            eps: config.visionConfig.rmsNormEps)
        _embedAudio.wrappedValue = Gemma3nMultimodalEmbedder(
            dimensions: config.audioConfig.hiddenSize, textDimensions: config.textConfig.hiddenSize,
            vocabularySize: config.audioConfig.vocabSize, offset: config.audioConfig.vocabOffset,
            eps: config.audioConfig.rmsNormEps)
    }

    public var loraLayers: [Module] { languageModel.loraLayers }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    public func shard(group: MLXDistributed.Group) throws {
        try validateShardable(self)
        try languageModel.shard(group: group)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs: inputs, cache: cache?.map { Optional($0) })
    }

    func inputEmbeddings(_ input: LMInput) throws -> (MLXArray, MLXArray) {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        guard tokens.ndim == 2, tokens.dim(1) > 0,
            all(logicalAnd(tokens .>= 0, tokens .< vocabularySize)).item(Bool.self)
        else {
            throw Gemma3nError.invalidInput(
                "Token IDs must be a nonempty rank-two vocabulary tensor.")
        }
        guard input.video == nil else {
            throw Gemma3nError.invalidInput("Use individual images for sampled video frames.")
        }
        var embeddings = languageModel.embeddings(tokens)
        let perLayerInputs = languageModel.getPerLayerInputs(tokens)
        guard input.image != nil || input.audio != nil else { return (embeddings, perLayerInputs) }

        let visionMask = logicalAnd(
            tokens .>= embedVision.offset,
            tokens .< embedAudio.offset)
        let visionIds = MLX.where(
            visionMask, tokens,
            MLXArray(embedVision.offset + embedVision.vocabularySize - 1))
        embeddings = MLX.where(
            visionMask.expandedDimensions(axis: -1), embedVision.hard(visionIds), embeddings)
        let audioMask = tokens .>= embedAudio.offset
        let audioIds = MLX.where(
            audioMask, tokens,
            MLXArray(embedAudio.offset + embedAudio.vocabularySize - 1))
        embeddings = MLX.where(
            audioMask.expandedDimensions(axis: -1), embedAudio.hard(audioIds), embeddings)

        if let image = input.image {
            guard image.pixels.ndim == 4, image.pixels.dim(1) == 3 else {
                throw Gemma3nError.invalidInput("Image pixels must be [images, 3, height, width].")
            }
            let output = visionTower(image.pixels)
            let features = embedVision.soft(
                output.reshaped(output.dim(0), -1, config.visionConfig.hiddenSize)
                    * sqrt(Float(config.visionConfig.hiddenSize)))
            embeddings = try gemma3nMergeFeatures(
                embeddings, tokens: tokens,
                tokenId: config.imageTokenId, features: features, kind: "image")
        }
        if let audio = input.audio {
            guard audio.features.ndim == 3,
                audio.features.dim(2) == config.audioConfig.inputFeatSize,
                audio.features.dim(1) > 0
            else {
                throw Gemma3nError.invalidInput("Audio features must be [clips, frames, mel bins].")
            }
            let valid =
                audio.mask ?? MLXArray.ones(Array(audio.features.shape.prefix(2)), dtype: .bool)
            guard valid.shape == Array(audio.features.shape.prefix(2)) else {
                throw Gemma3nError.invalidInput("Audio feature masks must match clips and frames.")
            }
            let (output, padding) = audioTower(
                audio.features, paddingMask: logicalNot(valid.asType(.bool)))
            let projected = embedAudio.soft(output)
            let paddingEmbedding = embedAudio.hard(
                MLXArray([embedAudio.offset + embedAudio.vocabularySize - 1]))
            var features = MLX.where(
                padding.expandedDimensions(axis: -1), paddingEmbedding, projected)
            guard features.dim(1) <= config.audioSoftTokensPerImage else {
                throw Gemma3nError.invalidInput("Audio exceeds the configured soft-token budget.")
            }
            let extra = config.audioSoftTokensPerImage - features.dim(1)
            if extra > 0 {
                features = concatenated(
                    [
                        features,
                        broadcast(
                            paddingEmbedding,
                            to: [features.dim(0), extra, features.dim(2)]),
                    ], axis: 1)
            }
            embeddings = try gemma3nMergeFeatures(
                embeddings, tokens: tokens,
                tokenId: config.audioTokenId, features: features, kind: "audio")
        }
        return (embeddings, perLayerInputs)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.image != nil || input.audio != nil || input.video != nil else {
            return .tokens(input.text)
        }
        let (embeddings, perLayer) = try inputEmbeddings(input)
        return try withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            let caches = cache.map { Optional($0) }
            let total = embeddings.dim(1)
            let processed = try prefill.forEachChunk(total: total) { range in
                _ = languageModel(
                    inputsEmbeds: embeddings[0..., range, 0...], cache: caches,
                    perLayerInputs: perLayer[0..., range, 0..., 0...])
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }
            let logits = languageModel(
                inputsEmbeds: embeddings[0..., processed..., 0...], cache: caches,
                perLayerInputs: perLayer[0..., processed..., 0..., 0...])
            prefill.progress?(total, total)
            return .logits(.init(logits: logits))
        }
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        let firstShared = languageModel.owningLayerCount
        let totalLayers = config.textConfig.numHiddenLayers
        var result = try checkpoint.mapNames { name in
            var key = name.hasPrefix("model.") ? String(name.dropFirst(6)) : name
            key = key.replacingOccurrences(of: "language_model.model.", with: "language_model.")
            if key.contains("rotary_emb.inv_freq") || key.split(separator: ".").contains("lm_head")
            {
                return nil
            }
            for layer in firstShared ..< totalLayers {
                for projection in ["k_proj", "v_proj", "k_norm", "v_norm"] {
                    let prefix = "language_model.layers.\(layer).self_attn.\(projection)"
                    if key == prefix || key.hasPrefix(prefix + ".") { return nil }
                }
            }
            return key
        }
        let expected = Dictionary(
            uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) })
        func rawConvolutionLayout(prefix: String, dimensions: Int) -> Bool {
            result.weights.contains { name, value in
                guard name.hasPrefix(prefix), name.contains("conv"), name.hasSuffix("weight"),
                    value.ndim == dimensions, let shape = expected[name], value.shape != shape
                else { return false }
                return dimensions == 4
                    ? value.transposed(0, 2, 3, 1).shape == shape
                    : value.transposed(0, 2, 1).shape == shape
            }
        }
        let rawVision = rawConvolutionLayout(prefix: "vision_tower.", dimensions: 4)
        let rawAudio2d = rawConvolutionLayout(prefix: "audio_tower.", dimensions: 4)
        let rawAudio1d = rawConvolutionLayout(prefix: "audio_tower.", dimensions: 3)
        for (name, value) in result.weights {
            if value.ndim == 4,
                (rawVision && name.hasPrefix("vision_tower."))
                    || (rawAudio2d && name.hasPrefix("audio_tower."))
            {
                result.weights[name] = value.transposed(0, 2, 3, 1)
                continue
            }
            if value.ndim == 3, rawAudio1d, name.hasPrefix("audio_tower."), name.contains("conv") {
                result.weights[name] = value.transposed(0, 2, 1)
                continue
            }
            guard let shape = expected[name], value.shape != shape else { continue }
            if value.ndim == 4, value.transposed(0, 2, 3, 1).shape == shape {
                result.weights[name] = value.transposed(0, 2, 3, 1)
            } else if value.ndim == 3, value.transposed(0, 2, 1).shape == shape {
                result.weights[name] = value.transposed(0, 2, 1)
            }
        }
        return result
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }
}
