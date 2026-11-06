// Copyright © 2026 Apple Inc.
// Adapted from mlx-vlm (MIT), pinned source attribution in ThirdPartyNotices/Qwen3Omni.

import Foundation

public struct Qwen3OmniConfiguration: Decodable, Sendable {
    public struct AudioConfiguration: Decodable, Sendable {
        public let dimensions: Int
        public let layers: Int
        public let heads: Int
        public let intermediateSize: Int
        public let melBins: Int
        public let outputDimensions: Int
        public let downsampleDimensions: Int
        public let window: Int
        public let inferenceWindow: Int
        public let maxPositions: Int
        public let convolutionChunkSize: Int
        public let activation: String

        enum CodingKeys: String, CodingKey {
            case dimensions = "d_model"
            case layers = "encoder_layers"
            case heads = "encoder_attention_heads"
            case intermediateSize = "encoder_ffn_dim"
            case melBins = "num_mel_bins"
            case outputDimensions = "output_dim"
            case downsampleDimensions = "downsample_hidden_size"
            case window = "n_window"
            case inferenceWindow = "n_window_infer"
            case maxPositions = "max_source_positions"
            case convolutionChunkSize = "conv_chunksize"
            case activation = "activation_function"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            dimensions = try c.decodeIfPresent(Int.self, forKey: .dimensions) ?? 1280
            layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? 32
            heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? 20
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5120
            melBins = try c.decodeIfPresent(Int.self, forKey: .melBins) ?? 128
            outputDimensions = try c.decodeIfPresent(Int.self, forKey: .outputDimensions) ?? 2048
            downsampleDimensions =
                try c.decodeIfPresent(Int.self, forKey: .downsampleDimensions) ?? 480
            window = try c.decodeIfPresent(Int.self, forKey: .window) ?? 50
            inferenceWindow = try c.decodeIfPresent(Int.self, forKey: .inferenceWindow) ?? 800
            maxPositions = try c.decodeIfPresent(Int.self, forKey: .maxPositions) ?? 1500
            activation = try c.decodeIfPresent(String.self, forKey: .activation) ?? "gelu"
            convolutionChunkSize =
                try c.decodeIfPresent(Int.self, forKey: .convolutionChunkSize) ?? 500
        }
    }

    public struct ThinkerConfiguration: Decodable, Sendable {
        public let text: Qwen3VLMoEConfiguration.TextConfiguration
        public let vision: Qwen3VLConfiguration.VisionConfiguration
        public let audio: AudioConfiguration
        public let imageToken: Int
        public let videoToken: Int
        public let audioToken: Int
        public let visionStartToken: Int
        public let visionEndToken: Int
        public let audioStartToken: Int
        public let audioEndToken: Int
        public let positionsPerSecond: Double

        enum CodingKeys: String, CodingKey {
            case text = "text_config"
            case vision = "vision_config"
            case audio = "audio_config"
            case imageToken = "image_token_id"
            case videoToken = "video_token_id"
            case audioToken = "audio_token_id"
            case visionStartToken = "vision_start_token_id"
            case visionEndToken = "vision_end_token_id"
            case audioStartToken = "audio_start_token_id"
            case audioEndToken = "audio_end_token_id"
            case positionsPerSecond = "position_id_per_seconds"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            text = try c.decode(Qwen3VLMoEConfiguration.TextConfiguration.self, forKey: .text)
            vision = try c.decode(Qwen3VLConfiguration.VisionConfiguration.self, forKey: .vision)
            audio = try c.decode(AudioConfiguration.self, forKey: .audio)
            imageToken = try c.decodeIfPresent(Int.self, forKey: .imageToken) ?? 151_655
            videoToken = try c.decodeIfPresent(Int.self, forKey: .videoToken) ?? 151_656
            audioToken = try c.decodeIfPresent(Int.self, forKey: .audioToken) ?? 151_675
            visionStartToken = try c.decodeIfPresent(Int.self, forKey: .visionStartToken) ?? 151_652
            visionEndToken = try c.decodeIfPresent(Int.self, forKey: .visionEndToken) ?? 151_653
            audioStartToken = try c.decodeIfPresent(Int.self, forKey: .audioStartToken) ?? 151_669
            audioEndToken = try c.decodeIfPresent(Int.self, forKey: .audioEndToken) ?? 151_670
            positionsPerSecond =
                try c.decodeIfPresent(Double.self, forKey: .positionsPerSecond) ?? 13
            struct RopePolicy: Decodable {
                let type: String?, ropeType: String?
                enum CodingKeys: String, CodingKey {
                    case type
                    case ropeType = "rope_type"
                }
            }
            struct TextPolicy: Decodable {
                let useSlidingWindow: Bool?
                let ropeScaling: RopePolicy?
                let ropeParameters: RopePolicy?
                enum CodingKeys: String, CodingKey {
                    case useSlidingWindow = "use_sliding_window"
                    case ropeScaling = "rope_scaling"
                    case ropeParameters = "rope_parameters"
                }
            }
            let policy = try c.decode(TextPolicy.self, forKey: .text)
            let ropeType =
                policy.ropeParameters?.ropeType ?? policy.ropeParameters?.type
                ?? policy.ropeScaling?.ropeType ?? policy.ropeScaling?.type ?? "default"
            guard text.hiddenSize > 0, text.numHiddenLayers > 0, text.vocabSize > 0,
                text.numAttentionHeads > 0, text.numKeyValueHeads > 0,
                text.numAttentionHeads % text.numKeyValueHeads == 0, text.headDim > 0,
                text.headDim % 2 == 0, text.decoderSparseStep > 0,
                text.numExperts > 0, text.numExpertsPerTok > 0,
                text.numExpertsPerTok <= text.numExperts, text.moeIntermediateSize > 0,
                !text.attentionBias, text.hiddenAct == "silu",
                text.ropeScaling?.type == nil || text.ropeScaling?.type == "default",
                ropeType == "default", policy.useSlidingWindow != true,
                audio.activation == "gelu",
                vision.outHiddenSize == text.hiddenSize, audio.outputDimensions == text.hiddenSize,
                vision.depth > 0, vision.numHeads > 0, vision.hiddenSize > 0,
                vision.hiddenSize % vision.numHeads == 0,
                (vision.hiddenSize / vision.numHeads) % 4 == 0,
                vision.patchSize > 0, vision.spatialMergeSize > 0, vision.temporalPatchSize > 0,
                vision.intermediateSize > 0, vision.numPositionEmbeddings > 0,
                vision.deepstackVisualIndexes.allSatisfy({ $0 >= 0 && $0 < vision.depth }),
                audio.dimensions > 2, audio.dimensions % 2 == 0,
                audio.layers > 0, audio.maxPositions > 0,
                audio.heads > 0, audio.dimensions % audio.heads == 0,
                audio.window > 0, audio.inferenceWindow >= audio.window * 2,
                audio.inferenceWindow % (audio.window * 2) == 0,
                audio.melBins > 0, audio.downsampleDimensions > 0, audio.convolutionChunkSize > 0,
                [
                    imageToken, videoToken, audioToken, visionStartToken, visionEndToken,
                    audioStartToken, audioEndToken,
                ].allSatisfy({ $0 >= 0 && $0 < text.vocabSize }),
                positionsPerSecond.isFinite, positionsPerSecond > 0
            else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: decoder.codingPath,
                        debugDescription:
                            "Invalid or unsupported Qwen3 Omni Thinker dimensions/attention configuration"
                    ))
            }
        }
    }

    public let thinker: ThinkerConfiguration
    enum CodingKeys: String, CodingKey { case thinker = "thinker_config" }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if c.contains(.thinker) {
            thinker = try c.decode(ThinkerConfiguration.self, forKey: .thinker)
        } else {
            thinker = try ThinkerConfiguration(from: decoder)
        }
    }
}
