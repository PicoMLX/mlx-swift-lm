// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm phi4mm at 4693cd0 (MIT).

import Foundation
import MLXLMCommon

public struct Phi4MultimodalVisionConfiguration: Codable, Sendable {
    public var hiddenSize: Int = 1152
    public var intermediateSize: Int = 4304
    public var heads: Int = 16
    public var layers: Int = 27
    public var patchSize: Int = 14
    public var imageSize: Int = 448
    public var channels: Int = 3
    public var eps: Float = 1e-6

    public init() {}

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case heads = "num_attention_heads"
        case layers = "num_hidden_layers"
        case patchSize = "patch_size"
        case imageSize = "image_size"
        case channels = "num_channels"
        case eps = "layer_norm_eps"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
        heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? 16
        layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? 27
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 448
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
        eps = try c.decodeIfPresent(Float.self, forKey: .eps) ?? 1e-6
    }
}

public struct Phi4MultimodalAudioConfiguration: Codable, Sendable {
    public var dimension: Int = 1024
    public var heads: Int = 16
    public var layers: Int = 24
    public var intermediateSize: Int = 1536
    public var inputSize: Int = 80
    public var reduction: Int = 8
    public var kernelSize: Int = 3
    public var activation: String = "swish"
    public var convActivation: String = "swish"
    public var gluActivation: String = "swish"
    public var biasInGLU: Bool = true
    public var pointwiseChannels: Int = 1024
    public var depthwiseChannels: Int = 1024
    public var depthwiseMultiplier: Int = 1
    public var causal: Bool = true
    public var convChannels: Int = 1024
    public var relativeDistance: Int = 500

    public init() {}

    enum CodingKeys: String, CodingKey {
        case dimension = "attention_dim"
        case heads = "attention_heads"
        case layers = "num_blocks"
        case intermediateSize = "linear_units"
        case inputSize = "input_size"
        case reduction = "time_reduction"
        case kernelSize = "kernel_size"
        case activation = "activation"
        case convActivation = "conv_activation"
        case gluActivation = "conv_glu_type"
        case biasInGLU = "bias_in_glu"
        case pointwiseChannels = "ext_pw_out_channel"
        case depthwiseChannels = "depthwise_seperable_out_channel"
        case depthwiseMultiplier = "depthwise_multiplier"
        case causal = "causal"
        case convChannels = "conv_channels"
        case relativeDistance = "t5_bias_max_distance"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dimension = try c.decodeIfPresent(Int.self, forKey: .dimension) ?? 1024
        heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? 16
        layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? 24
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 1536
        inputSize = try c.decodeIfPresent(Int.self, forKey: .inputSize) ?? 80
        reduction = try c.decodeIfPresent(Int.self, forKey: .reduction) ?? 8
        kernelSize = try c.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 3
        activation = try c.decodeIfPresent(String.self, forKey: .activation) ?? "swish"
        convActivation = try c.decodeIfPresent(String.self, forKey: .convActivation) ?? "swish"
        gluActivation = try c.decodeIfPresent(String.self, forKey: .gluActivation) ?? "swish"
        biasInGLU = try c.decodeIfPresent(Bool.self, forKey: .biasInGLU) ?? true
        pointwiseChannels = try c.decodeIfPresent(Int.self, forKey: .pointwiseChannels) ?? 1024
        depthwiseChannels = try c.decodeIfPresent(Int.self, forKey: .depthwiseChannels) ?? 1024
        depthwiseMultiplier = try c.decodeIfPresent(Int.self, forKey: .depthwiseMultiplier) ?? 1
        causal = try c.decodeIfPresent(Bool.self, forKey: .causal) ?? true
        convChannels = try c.decodeIfPresent(Int.self, forKey: .convChannels) ?? 1024
        relativeDistance = try c.decodeIfPresent(Int.self, forKey: .relativeDistance) ?? 500
    }
}

public struct Phi4MultimodalAdapterConfiguration: Codable, Sendable {
    public var rank: Int = 0
    public var alpha: Float = 1
    public var targets: [String] = []

    public init() {}

    enum CodingKeys: String, CodingKey {
        case rank = "r"
        case alpha = "lora_alpha"
        case targets = "target_modules"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        rank = try c.decodeIfPresent(Int.self, forKey: .rank) ?? 0
        alpha = try c.decodeIfPresent(Float.self, forKey: .alpha) ?? 1
        targets = try c.decodeIfPresent([String].self, forKey: .targets) ?? []
    }
}

public struct Phi4MultimodalConfiguration: Codable, Sendable {
    public var modelType: String = "phi4mm"
    public var vocabularySize: Int = 200064
    public var hiddenSize: Int = 3072
    public var layers: Int = 32
    public var intermediateSize: Int = 8192
    public var heads: Int = 24
    public var kvHeads: Int = 8
    public var eps: Float = 1e-5
    public var ropeTheta: Float = 10000
    public var ropeTraditional: Bool = false
    public var partialRotaryFactor: Float = 0.75
    public var maxPositionEmbeddings: Int = 131072
    public var originalMaxPositionEmbeddings: Int = 4096
    public var tieWordEmbeddings: Bool = true
    public var imageToken: Int = -200
    public var audioToken: Int = 200011
    public var vision: Phi4MultimodalVisionConfiguration = .init()
    public var audioProcessor: AudioProcessor? = nil
    public var visionAdapter: Phi4MultimodalAdapterConfiguration? = nil
    public var speechAdapter: Phi4MultimodalAdapterConfiguration? = nil
    public var ropeScaling: RopeScaling? = nil
    public var maxPatches: Int = 3600
    public var minPatches: Int = 256

    public struct AudioProcessor: Codable, Sendable {
        public var config: Phi4MultimodalAudioConfiguration
    }
    public struct RopeScaling: Codable, Sendable {
        public var type: String?
        public var shortFactor: [Float]?
        public var longFactor: [Float]?
        public var factor: Float?
        enum CodingKeys: String, CodingKey {
            case type, factor
            case shortFactor = "short_factor"
            case longFactor = "long_factor"
        }
    }
    public var audio: Phi4MultimodalAudioConfiguration { audioProcessor?.config ?? .init() }
    public var headDimension: Int { hiddenSize / heads }

    public init() {}

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case layers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case heads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case eps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case imageToken = "image_token_index"
        case audioToken = "audio_token_index"
        case vision = "vision_config"
        case audioProcessor = "audio_processor"
        case visionAdapter = "vision_lora"
        case speechAdapter = "speech_lora"
        case ropeScaling = "rope_scaling"
        case maxPatches = "max_num_patches"
        case minPatches = "min_num_patches"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "phi4mm"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 200064
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3072
        layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? 32
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8192
        heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? 24
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        eps = try c.decodeIfPresent(Float.self, forKey: .eps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        partialRotaryFactor =
            try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.75
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        originalMaxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .originalMaxPositionEmbeddings) ?? 4096
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        imageToken = try c.decodeIfPresent(Int.self, forKey: .imageToken) ?? -200
        audioToken = try c.decodeIfPresent(Int.self, forKey: .audioToken) ?? 200011
        vision =
            try c.decodeIfPresent(Phi4MultimodalVisionConfiguration.self, forKey: .vision)
            ?? .init()
        audioProcessor = try c.decodeIfPresent(AudioProcessor.self, forKey: .audioProcessor) ?? nil
        visionAdapter =
            try c.decodeIfPresent(Phi4MultimodalAdapterConfiguration.self, forKey: .visionAdapter)
            ?? nil
        speechAdapter =
            try c.decodeIfPresent(Phi4MultimodalAdapterConfiguration.self, forKey: .speechAdapter)
            ?? nil
        ropeScaling = try c.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling) ?? nil
        maxPatches = try c.decodeIfPresent(Int.self, forKey: .maxPatches) ?? 3600
        minPatches = try c.decodeIfPresent(Int.self, forKey: .minPatches) ?? 256
    }
}

extension Phi4MultimodalConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard heads > 0, kvHeads > 0, hiddenSize > 0, hiddenSize % heads == 0,
            heads % kvHeads == 0, layers > 0, intermediateSize > 0, vocabularySize > 0,
            partialRotaryFactor > 0, partialRotaryFactor <= 1,
            Int(Float(headDimension) * partialRotaryFactor) % 2 == 0,
            vision.hiddenSize > 0, vision.heads > 0, vision.hiddenSize % vision.heads == 0,
            vision.patchSize > 0, vision.layers >= 2, minPatches > 0, maxPatches >= minPatches,
            audio.dimension > 0, audio.heads > 0, audio.dimension % audio.heads == 0,
            audio.reduction >= 2, audio.reduction.nonzeroBitCount == 1,
            audio.pointwiseChannels == audio.dimension, audio.depthwiseChannels == audio.dimension,
            audio.inputSize > 0, audio.relativeDistance > 0
        else { throw ModelFactoryError.invalidConfiguration("Invalid Phi-4 multimodal dimensions") }
    }
}
