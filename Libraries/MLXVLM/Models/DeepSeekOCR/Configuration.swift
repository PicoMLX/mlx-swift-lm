// Port of Blaizzy/mlx-vlm's deepseekocr and deepseekocr_2 (MIT).

import Foundation
import MLXLMCommon

struct OCRKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init(_ string: String) { stringValue = string }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}

extension KeyedDecodingContainer where Key == OCRKey {
    func value<T: Decodable>(_ key: String, _ fallback: T) throws -> T {
        try decodeIfPresent(T.self, forKey: OCRKey(key)) ?? fallback
    }
}

/// The shared language decoder and the two DeepSeek OCR vision configurations.
public struct DeepSeekOCRConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public struct Text: Decodable, Sendable {
        public var hiddenSize = 1280
        public var layers = 30
        public var intermediateSize = 6848
        public var heads = 32
        public var kvHeads = 32
        public var vocabularySize = 102400
        public var moeIntermediateSize = 896
        public var experts: Int? = 64
        public var sharedExperts: Int? = 2
        public var expertsPerToken = 6
        public var firstDenseLayers = 0
        public var moeFrequency = 1
        public var rmsNormEps: Float = 1e-6
        public var ropeTheta: Float = 10000
        public var ropeTraditional = false
        public var ropeScaling: [String: StringOrNumber]?
        public var maxPositionEmbeddings = 2048
        public var attentionBias = false
        public var routingScale: Float = 1
        public var scoringFunction = "softmax"
        public var topKMethod = "greedy"
        public var groups = 1
        public var topKGroups = 1
        public var qLoraRank: Int? = 1536
        public var kvLoraRank = 512
        public var qkRopeHeadDim = 0
        public var qkNopeHeadDim = 0
        public var vHeadDim = 128

        public init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: OCRKey.self)
            hiddenSize = try c.value("hidden_size", hiddenSize)
            layers = try c.value("num_hidden_layers", layers)
            intermediateSize = try c.value("intermediate_size", intermediateSize)
            heads = try c.value("num_attention_heads", heads)
            kvHeads = try c.value("num_key_value_heads", heads)
            vocabularySize = try c.value("vocab_size", vocabularySize)
            moeIntermediateSize = try c.value("moe_intermediate_size", moeIntermediateSize)
            if c.contains(OCRKey("n_routed_experts")) {
                experts = try c.decodeIfPresent(Int.self, forKey: OCRKey("n_routed_experts"))
            }
            if c.contains(OCRKey("n_shared_experts")) {
                sharedExperts = try c.decodeIfPresent(Int.self, forKey: OCRKey("n_shared_experts"))
            }
            expertsPerToken = try c.value("num_experts_per_tok", expertsPerToken)
            firstDenseLayers = try c.value("first_k_dense_replace", firstDenseLayers)
            moeFrequency = try c.value("moe_layer_freq", moeFrequency)
            rmsNormEps = try c.value("rms_norm_eps", rmsNormEps)
            ropeTheta = try c.value("rope_theta", ropeTheta)
            ropeTraditional = try c.value("rope_traditional", ropeTraditional)
            ropeScaling = try c.decodeIfPresent(
                [String: StringOrNumber].self, forKey: OCRKey("rope_scaling"))
            maxPositionEmbeddings = try c.value("max_position_embeddings", maxPositionEmbeddings)
            attentionBias = try c.value("attention_bias", attentionBias)
            routingScale = try c.value("routed_scaling_factor", routingScale)
            scoringFunction = try c.value("scoring_func", scoringFunction)
            topKMethod = try c.value("topk_method", topKMethod)
            groups = try c.value("n_group", groups)
            topKGroups = try c.value("topk_group", groups)
            if c.contains(OCRKey("q_lora_rank")) {
                qLoraRank = try c.decodeIfPresent(Int.self, forKey: OCRKey("q_lora_rank"))
            }
            kvLoraRank = try c.value("kv_lora_rank", kvLoraRank)
            qkRopeHeadDim = try c.value("qk_rope_head_dim", qkRopeHeadDim)
            qkNopeHeadDim = try c.value("qk_nope_head_dim", qkNopeHeadDim)
            vHeadDim = try c.value("v_head_dim", vHeadDim)
        }

        var usesMLA: Bool { qkNopeHeadDim > 0 }
        var headDim: Int { hiddenSize / heads }
    }

    public struct Vision: Decodable, Sendable {
        public var hiddenSize = 1024
        public var layers = 24
        public var intermediateSize = 4096
        public var heads = 16
        public var patchSize = 14
        public var layerNormEps: Float = 1e-6
        public var encoderDim = 896
        public var encoderLayers = 24
        public var encoderHeads = 14
        public var encoderKVHeads = 2
        public var encoderIntermediateSize = 4864
        public var encoderRopeTheta: Float = 1_000_000
        public var queryGlobalCount = 256
        public var queryLocalCount = 144

        public init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: OCRKey.self)
            hiddenSize = try c.value("hidden_size", hiddenSize)
            layers = try c.value("layers", layers)
            intermediateSize = try c.value("intermediate_size", intermediateSize)
            heads = try c.value("num_attention_heads", heads)
            patchSize = try c.value("patch_size", patchSize)
            layerNormEps = try c.value("layer_norm_eps", layerNormEps)
            let params = try c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: OCRKey("params"))
            let widths = try c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: OCRKey("width"))
            let qwen = params?["qwen2"] ?? widths?["qwen2-0-5b"] ?? [:]
            encoderDim = qwen["dim"]?.asInt() ?? encoderDim
            encoderLayers = qwen["layers"]?.asInt() ?? encoderLayers
            encoderHeads = qwen["heads"]?.asInt() ?? encoderHeads
            encoderKVHeads = qwen["kv_heads"]?.asInt() ?? encoderKVHeads
            encoderIntermediateSize = qwen["intermediate_size"]?.asInt() ?? encoderIntermediateSize
            encoderRopeTheta = qwen["rope_theta"]?.asFloat() ?? encoderRopeTheta
            queryGlobalCount = try c.value("query_global_count", queryGlobalCount)
            queryLocalCount = try c.value("query_local_count", queryLocalCount)
        }
    }

    public struct Projector: Decodable, Sendable {
        public var type = "linear"
        public var inputDim = 2048
        public var embedDim = 1280
        public var depth = 2
        public var mlpRatio = 1
        public var downsampleRatio = 2
        public init() {}
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: OCRKey.self)
            type = try c.value("projector_type", type)
            inputDim = try c.value("input_dim", inputDim)
            embedDim = try c.value("n_embed", embedDim)
            depth = try c.value("depth", depth)
            mlpRatio = try c.value("mlp_ratio", mlpRatio)
            downsampleRatio = try c.value("downsample_ratio", downsampleRatio)
        }
    }

    public struct SAM: Decodable, Sendable {
        public var imageSize = 1024
        public var patchSize = 16
        public var width = 768
        public var layers = 12
        public var heads = 12
        public var windowSize = 14
        public var globalAttentionLayers = [2, 5, 8, 11]
        public var neckChannels = 256
        public var downsampleChannels = 512
        public init() {}
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: OCRKey.self)
            imageSize = try c.value("image_size", imageSize)
            patchSize = try c.value("patch_size", patchSize)
            width = try c.value("width", width)
            layers = try c.value("layers", layers)
            heads = try c.value("heads", heads)
            windowSize = try c.value("window_size", windowSize)
            globalAttentionLayers = try c.value("global_attn_indexes", globalAttentionLayers)
            neckChannels = try c.value("neck_channels", neckChannels)
            downsampleChannels = try c.value("downsample_channels", downsampleChannels)
        }
    }

    public var modelType = "deepseekocr"
    public var text = Text()
    public var vision = Vision()
    public var projector = Projector()
    public var sam = SAM()
    public var imageTokenId = 128815
    public var globalViewPosition = "head"
    public var isOCR2: Bool { modelType == "deepseekocr_2" }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: OCRKey.self)
        modelType = try c.value("model_type", modelType)
        text =
            try c.decodeIfPresent(Text.self, forKey: OCRKey("text_config"))
            ?? c.decodeIfPresent(Text.self, forKey: OCRKey("language_config")) ?? Text()
        vision = try c.value("vision_config", vision)
        projector = try c.value("projector_config", projector)
        sam = try c.value("sam_config", sam)
        imageTokenId = try c.value("image_token_index", imageTokenId)
        globalViewPosition = try c.value("global_view_pos", globalViewPosition)
        if try c.value("tile_tag", "2D") != "2D" {
            throw DecodingError.dataCorruptedError(
                forKey: OCRKey("tile_tag"), in: c,
                debugDescription: "DeepSeek OCR requires 2D tiles.")
        }
    }

    public func validateModelConfiguration() throws {
        guard ["deepseekocr", "deepseekocr_2"].contains(modelType), text.hiddenSize > 0,
            text.layers > 0, text.heads > 0, text.kvHeads > 0,
            text.hiddenSize.isMultiple(of: text.heads), text.heads.isMultiple(of: text.kvHeads),
            text.moeFrequency > 0, imageTokenId >= 0, text.vocabularySize > imageTokenId,
            sam.patchSize > 0, sam.width > 0, sam.heads > 0,
            sam.width.isMultiple(of: sam.heads), sam.layers >= 0, sam.windowSize > 0,
            projector.embedDim == text.hiddenSize, projector.inputDim > 0,
            ["linear", "downsample_mlp_gelu"].contains(projector.type), projector.depth >= 2,
            projector.downsampleRatio > 0, vision.hiddenSize > 0, vision.heads > 0,
            vision.hiddenSize.isMultiple(of: vision.heads), vision.layers >= 0,
            vision.encoderDim > 0, vision.encoderHeads > 0, vision.encoderKVHeads > 0,
            vision.encoderDim.isMultiple(of: vision.encoderHeads),
            vision.encoderHeads.isMultiple(of: vision.encoderKVHeads), vision.encoderLayers >= 0,
            ["softmax", "sigmoid"].contains(text.scoringFunction),
            ["greedy", "noaux_tc"].contains(text.topKMethod)
        else { throw VLMError.processing("Invalid DeepSeek OCR model configuration.") }
        if let experts = text.experts {
            guard experts > 0, text.expertsPerToken > 0, text.expertsPerToken <= experts,
                text.groups > 0, experts.isMultiple(of: text.groups),
                text.topKGroups > 0, text.topKGroups <= text.groups
            else { throw VLMError.processing("Invalid DeepSeek OCR expert routing.") }
        }
    }
}
