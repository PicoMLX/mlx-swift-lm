// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm MiniCPM models at 4693cd0 (MIT).
import Foundation
import MLXLLM
import MLXLMCommon

public struct MiniCPMVisionConfiguration: Codable, Sendable {
    public var hiddenSize: Int = 1152
    public var intermediateSize: Int = 4304
    public var heads: Int = 16
    public var layers: Int = 27
    public var patchSize: Int = 14
    public var imageSize: Int = 448
    public var channels: Int = 3
    public var hiddenAct: String = "gelu_pytorch_tanh"
    public var eps: Float = 1e-6
    public var window: [Int] = [2, 2]
    public init() {}
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case heads = "num_attention_heads"
        case layers = "num_hidden_layers"
        case patchSize = "patch_size"
        case imageSize = "image_size"
        case channels = "num_channels"
        case hiddenAct = "hidden_act"
        case eps = "layer_norm_eps"
        case window = "window_kernel_size"
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
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu_pytorch_tanh"
        eps = try c.decodeIfPresent(Float.self, forKey: .eps) ?? 1e-6
        window = try c.decodeIfPresent([Int].self, forKey: .window) ?? [2, 2]
    }
}

public struct MiniCPMAudioConfiguration: Codable, Sendable {
    public var dimension: Int = 1024
    public var heads: Int = 16
    public var layers: Int = 24
    public var intermediateSize: Int = 4096
    public var melBins: Int = 80
    public var positions: Int = 1500
    public var eps: Float = 1e-5
    public init() {}
    enum CodingKeys: String, CodingKey {
        case dimension = "d_model"
        case heads = "encoder_attention_heads"
        case layers = "encoder_layers"
        case intermediateSize = "encoder_ffn_dim"
        case melBins = "num_mel_bins"
        case positions = "max_source_positions"
        case eps = "layer_norm_eps"
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dimension = try c.decodeIfPresent(Int.self, forKey: .dimension) ?? 1024
        heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? 16
        layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? 24
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
        melBins = try c.decodeIfPresent(Int.self, forKey: .melBins) ?? 80
        positions = try c.decodeIfPresent(Int.self, forKey: .positions) ?? 1500
        eps = try c.decodeIfPresent(Float.self, forKey: .eps) ?? 1e-5
    }
}

public struct MiniCPMTextGeometry: Codable, Sendable {
    public var hiddenSize: Int = 4096
    public var layers: Int = 32
    public var vocabularySize: Int = 151936
    public var heads: Int = 32
    public var kvHeads: Int = 8
    public var headDimension: Int = 128
    public var tied: Bool = false
    public init() {}
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case layers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case heads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimension = "head_dim"
        case tied = "tie_word_embeddings"
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? 32
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151936
        heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        headDimension = try c.decodeIfPresent(Int.self, forKey: .headDimension) ?? 128
        tied = try c.decodeIfPresent(Bool.self, forKey: .tied) ?? false
    }
}

public struct MiniCPMConfiguration: Codable, Sendable {
    public let modelType: String
    public let text: Qwen3Configuration?
    public let hybrid: Qwen35TextConfiguration?
    public let geometry: MiniCPMTextGeometry
    public let vision: MiniCPMVisionConfiguration
    public let audio: MiniCPMAudioConfiguration?
    public let queryCount: Int
    public let audioPoolStep: Int
    public let insertLayer: Int
    public let downsampleMode: String
    public let mergeKernel: [Int]
    public let mergeTimes: Int
    public let sliceScale: Int
    public let maxSlices: Int
    public let sliceMode: Bool
    public let imageTokenID: Int?
    public let videoTokenID: Int?
    public let imageStartID: Int?
    public let imageEndID: Int?
    public let sliceStartID: Int?
    public let sliceEndID: Int?
    public let newlineID: Int?
    public var isOmni: Bool { modelType == "minicpmo" }
    public var isCanvas: Bool { modelType == "minicpmv4_7" }
    public var mergeFactor: Int { downsampleMode == "4x" ? 2 : 4 }
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
        case vision = "vision_config"
        case audio = "audio_config"
        case initAudio = "init_audio"
        case queryCount = "query_num"
        case audioPoolStep = "audio_pool_step"
        case insertLayer = "insert_layer_id"
        case downsampleMode = "downsample_mode"
        case mergeKernel = "merge_kernel_size"
        case mergeTimes = "merger_times"
        case slice = "slice_config"
        case sliceMode = "slice_mode"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case imageStartID = "image_start_id"
        case imageEndID = "image_end_id"
        case sliceStartID = "slice_start_id"
        case sliceEndID = "slice_end_id"
        case newlineID = "newline_id"
    }
    struct Slice: Codable {
        var scaleResolution: Int?
        var maxSliceNums: Int?
        enum CodingKeys: String, CodingKey {
            case scaleResolution = "scale_resolution"
            case maxSliceNums = "max_slice_nums"
        }
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        let textDecoder = c.contains(.text) ? try c.superDecoder(forKey: .text) : decoder
        geometry = try MiniCPMTextGeometry(from: textDecoder)
        if modelType == "minicpmo" {
            text = try Qwen3Configuration(from: textDecoder)
            hybrid = nil
        } else {
            text = nil
            hybrid = try Qwen35TextConfiguration(from: textDecoder)
        }
        vision = try c.decodeIfPresent(MiniCPMVisionConfiguration.self, forKey: .vision) ?? .init()
        audio =
            (try c.decodeIfPresent(Bool.self, forKey: .initAudio) ?? true)
            ? try c.decodeIfPresent(MiniCPMAudioConfiguration.self, forKey: .audio) : nil
        queryCount = try c.decodeIfPresent(Int.self, forKey: .queryCount) ?? 64
        audioPoolStep = try c.decodeIfPresent(Int.self, forKey: .audioPoolStep) ?? 5
        insertLayer = try c.decodeIfPresent(Int.self, forKey: .insertLayer) ?? 6
        downsampleMode = try c.decodeIfPresent(String.self, forKey: .downsampleMode) ?? "16x"
        mergeKernel = try c.decodeIfPresent([Int].self, forKey: .mergeKernel) ?? [2, 2]
        mergeTimes = try c.decodeIfPresent(Int.self, forKey: .mergeTimes) ?? 1
        let slice = try c.decodeIfPresent(Slice.self, forKey: .slice)
        sliceScale = slice?.scaleResolution ?? 448
        maxSlices = slice?.maxSliceNums ?? 9
        sliceMode = try c.decodeIfPresent(Bool.self, forKey: .sliceMode) ?? true
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID)
        videoTokenID = try c.decodeIfPresent(Int.self, forKey: .videoTokenID)
        imageStartID = try c.decodeIfPresent(Int.self, forKey: .imageStartID)
        imageEndID = try c.decodeIfPresent(Int.self, forKey: .imageEndID)
        sliceStartID = try c.decodeIfPresent(Int.self, forKey: .sliceStartID)
        sliceEndID = try c.decodeIfPresent(Int.self, forKey: .sliceEndID)
        newlineID = try c.decodeIfPresent(Int.self, forKey: .newlineID)
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        if let text { try c.encode(text, forKey: .text) }
        if let hybrid { try c.encode(hybrid, forKey: .text) }
        try c.encode(vision, forKey: .vision)
        try c.encodeIfPresent(audio, forKey: .audio)
        try c.encode(audio != nil, forKey: .initAudio)
        try c.encode(queryCount, forKey: .queryCount)
        try c.encode(audioPoolStep, forKey: .audioPoolStep)
        try c.encode(insertLayer, forKey: .insertLayer)
        try c.encode(downsampleMode, forKey: .downsampleMode)
        try c.encode(mergeKernel, forKey: .mergeKernel)
        try c.encode(mergeTimes, forKey: .mergeTimes)
        try c.encode(Slice(scaleResolution: sliceScale, maxSliceNums: maxSlices), forKey: .slice)
        try c.encode(sliceMode, forKey: .sliceMode)
        try c.encodeIfPresent(imageTokenID, forKey: .imageTokenID)
        try c.encodeIfPresent(videoTokenID, forKey: .videoTokenID)
        try c.encodeIfPresent(imageStartID, forKey: .imageStartID)
        try c.encodeIfPresent(imageEndID, forKey: .imageEndID)
        try c.encodeIfPresent(sliceStartID, forKey: .sliceStartID)
        try c.encodeIfPresent(sliceEndID, forKey: .sliceEndID)
        try c.encodeIfPresent(newlineID, forKey: .newlineID)
    }
}

extension MiniCPMConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard ["minicpmo", "minicpmv4_6", "minicpmv4_7"].contains(modelType),
            geometry.hiddenSize > 0, geometry.heads > 0, geometry.kvHeads > 0,
            geometry.heads % geometry.kvHeads == 0, geometry.layers > 0,
            geometry.vocabularySize > 0,
            vision.patchSize > 0, vision.imageSize % vision.patchSize == 0,
            vision.heads > 0, vision.hiddenSize % vision.heads == 0, vision.layers > 0,
            vision.window.count == 2, vision.window.allSatisfy({ $0 > 0 }),
            mergeKernel.count == 2, mergeKernel.allSatisfy({ $0 > 0 }), mergeTimes > 0,
            audioPoolStep > 0, queryCount > 0, sliceScale > 0, maxSlices > 0,
            ["4x", "16x"].contains(downsampleMode)
        else { throw ModelFactoryError.invalidConfiguration("Invalid MiniCPM model geometry") }
        if let text { try text.validateModelConfiguration() }
        if let hybrid {
            guard hybrid.linearNumKeyHeads > 0, hybrid.linearNumValueHeads > 0,
                hybrid.linearNumValueHeads % hybrid.linearNumKeyHeads == 0,
                hybrid.linearKeyHeadDim > 0, hybrid.linearValueHeadDim > 0,
                hybrid.linearConvKernelDim > 1,
                hybrid.numExperts >= 0, hybrid.numExpertsPerTok >= 0,
                hybrid.numExperts == 0
                    || (hybrid.numExpertsPerTok > 0 && hybrid.numExpertsPerTok <= hybrid.numExperts
                        && hybrid.moeIntermediateSize > 0)
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "Invalid MiniCPM hybrid decoder geometry")
            }
        }
    }
}
