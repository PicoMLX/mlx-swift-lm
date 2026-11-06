// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm gemma3n at 4693cd0 (MIT).

import Foundation
import MLXLLM
import MLXLMCommon

public struct Gemma3nVisionConfiguration: Codable, Sendable {
    public var modelType = "gemma3n_vision"
    public var hiddenSize = 2048
    public var imageSize = 224
    public var rmsNormEps: Float = 1e-6
    public var vocabSize = 128
    public var vocabOffset = 262_144

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case imageSize = "image_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabOffset = "vocab_offset"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? imageSize
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? rmsNormEps
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? vocabSize
        vocabOffset = try c.decodeIfPresent(Int.self, forKey: .vocabOffset) ?? vocabOffset
    }

    package init() {}
}

public struct Gemma3nAudioConfiguration: Codable, Sendable {
    public var inputFeatSize = 128
    public var hiddenSize = 1536
    public var confAttentionChunkSize = 12
    public var confAttentionContextLeft = 13
    public var confAttentionContextRight = 0
    public var confAttentionInvalidLogitsValue: Float = -1e9
    public var confAttentionLogitCap: Float = 50
    public var confNumAttentionHeads = 8
    public var confNumHiddenLayers = 12
    public var confConvKernelSize = 5
    public var confReductionFactor = 4
    public var confResidualWeight: Float = 0.5
    public var sscpConvChannelSize = [128, 32]
    public var sscpConvKernelSize = [[3, 3], [3, 3]]
    public var sscpConvStrideSize = [[2, 2], [2, 2]]
    public var sscpConvEps: Float = 1e-3
    public var rmsNormEps: Float = 1e-6
    public var gradientClipping: Float = 1e10
    public var vocabSize = 128
    public var vocabOffset = 262_272

    enum CodingKeys: String, CodingKey {
        case inputFeatSize = "input_feat_size"
        case hiddenSize = "hidden_size"
        case confAttentionChunkSize = "conf_attention_chunk_size"
        case confAttentionContextLeft = "conf_attention_context_left"
        case confAttentionContextRight = "conf_attention_context_right"
        case confAttentionInvalidLogitsValue = "conf_attention_invalid_logits_value"
        case confAttentionLogitCap = "conf_attention_logit_cap"
        case confNumAttentionHeads = "conf_num_attention_heads"
        case confNumHiddenLayers = "conf_num_hidden_layers"
        case confConvKernelSize = "conf_conv_kernel_size"
        case confReductionFactor = "conf_reduction_factor"
        case confResidualWeight = "conf_residual_weight"
        case sscpConvChannelSize = "sscp_conv_channel_size"
        case sscpConvKernelSize = "sscp_conv_kernel_size"
        case sscpConvStrideSize = "sscp_conv_stride_size"
        case sscpConvEps = "sscp_conv_eps"
        case rmsNormEps = "rms_norm_eps"
        case gradientClipping = "gradient_clipping"
        case vocabSize = "vocab_size"
        case vocabOffset = "vocab_offset"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        inputFeatSize = try c.decodeIfPresent(Int.self, forKey: .inputFeatSize) ?? inputFeatSize
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        confAttentionChunkSize =
            try c.decodeIfPresent(Int.self, forKey: .confAttentionChunkSize)
            ?? confAttentionChunkSize
        confAttentionContextLeft =
            try c.decodeIfPresent(Int.self, forKey: .confAttentionContextLeft)
            ?? confAttentionContextLeft
        confAttentionContextRight =
            try c.decodeIfPresent(Int.self, forKey: .confAttentionContextRight)
            ?? confAttentionContextRight
        confAttentionInvalidLogitsValue =
            try c.decodeIfPresent(Float.self, forKey: .confAttentionInvalidLogitsValue)
            ?? confAttentionInvalidLogitsValue
        confAttentionLogitCap =
            try c.decodeIfPresent(Float.self, forKey: .confAttentionLogitCap)
            ?? confAttentionLogitCap
        confNumAttentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .confNumAttentionHeads) ?? confNumAttentionHeads
        confNumHiddenLayers =
            try c.decodeIfPresent(Int.self, forKey: .confNumHiddenLayers) ?? confNumHiddenLayers
        confConvKernelSize =
            try c.decodeIfPresent(Int.self, forKey: .confConvKernelSize) ?? confConvKernelSize
        confReductionFactor =
            try c.decodeIfPresent(Int.self, forKey: .confReductionFactor) ?? confReductionFactor
        confResidualWeight =
            try c.decodeIfPresent(Float.self, forKey: .confResidualWeight) ?? confResidualWeight
        sscpConvChannelSize =
            try c.decodeIfPresent([Int].self, forKey: .sscpConvChannelSize) ?? sscpConvChannelSize
        sscpConvKernelSize =
            try c.decodeIfPresent([[Int]].self, forKey: .sscpConvKernelSize) ?? sscpConvKernelSize
        sscpConvStrideSize =
            try c.decodeIfPresent([[Int]].self, forKey: .sscpConvStrideSize) ?? sscpConvStrideSize
        sscpConvEps = try c.decodeIfPresent(Float.self, forKey: .sscpConvEps) ?? sscpConvEps
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? rmsNormEps
        gradientClipping =
            try c.decodeIfPresent(Float.self, forKey: .gradientClipping) ?? gradientClipping
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? vocabSize
        vocabOffset = try c.decodeIfPresent(Int.self, forKey: .vocabOffset) ?? vocabOffset
    }

    package init() {}
}

public struct Gemma3nConfiguration: Codable {
    public let textConfig: Gemma3nTextConfiguration
    public let visionConfig: Gemma3nVisionConfiguration
    public let audioConfig: Gemma3nAudioConfiguration
    public var modelType = "gemma3n"
    public var imageTokenId = 262_145
    public var audioTokenId = 262_273
    public var visionSoftTokensPerImage = 256
    public var audioSoftTokensPerImage = 188

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case audioConfig = "audio_config"
        case modelType = "model_type"
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case audioSoftTokensPerImage = "audio_soft_tokens_per_image"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try c.decode(Gemma3nTextConfiguration.self, forKey: .textConfig)
        visionConfig = try c.decode(Gemma3nVisionConfiguration.self, forKey: .visionConfig)
        audioConfig = try c.decode(Gemma3nAudioConfiguration.self, forKey: .audioConfig)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? imageTokenId
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? audioTokenId
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: .visionSoftTokensPerImage)
            ?? visionSoftTokensPerImage
        audioSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: .audioSoftTokensPerImage)
            ?? audioSoftTokensPerImage
    }
}
