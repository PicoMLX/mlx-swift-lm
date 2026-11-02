// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm phi4mm at 4693cd0 (MIT).

import Foundation
import MLX
import MLXLMCommon

extension Phi4Multimodal {
    private func checkpointName(_ source: String) -> String? {
        if source.contains("position_ids") || source.contains("glb_GN") || source.contains("sub_GN")
            || source.contains("img_processor.head.")
        {
            return nil
        }
        var name = source
        let namespaces = [
            ("model.embed_tokens_extend.audio_embed.encoder.", "audio_encoder."),
            ("model.embed_tokens_extend.audio_embed.audio_projection.", "audio_projection."),
            ("model.embed_tokens_extend.image_embed.img_processor.", "vision_tower.vision_tower."),
            ("model.embed_tokens_extend.image_embed.img_projection.", "mm_projector."),
        ]
        for (prefix, replacement) in namespaces where name.hasPrefix(prefix) {
            name = replacement + name.dropFirst(prefix.count)
            break
        }
        if name.hasPrefix("model.") || name.hasPrefix("lm_head.") {
            name = "language_model." + name
        }
        if config.tieWordEmbeddings && name.hasPrefix("language_model.lm_head.") { return nil }
        for (old, new) in [
            ("mm_projector.0.", "mm_projector.proj_0."),
            ("mm_projector.2.", "mm_projector.proj_2."),
            (".lora_A.vision", ".vision_a"), (".lora_B.vision", ".vision_b"),
            (".lora_A.speech", ".speech_a"), (".lora_B.speech", ".speech_b"),
            (".base_layer", ".base"), (".net.0.linear.", ".net_0.linear."),
            (".net.2.", ".net_2."), ("audio_projection.speech.0", "audio_projection.speech.proj_0"),
            ("audio_projection.speech.2", "audio_projection.speech.proj_2"),
            ("audio_projection.vision.0", "audio_projection.vision.proj_0"),
            ("audio_projection.vision.2", "audio_projection.vision.proj_2"),
        ] { name = name.replacingOccurrences(of: old, with: new) }
        if name.hasPrefix("audio_encoder."), let range = name.range(of: "embed.conv.") {
            let suffix = String(name[range.upperBound...]).split(separator: ".", maxSplits: 1)
            guard suffix.count == 2, let index = Int(suffix[0]) else { return name }
            let layers = [
                0: "conv_0", 2: "dw_pw_layers.0.dw", 3: "dw_pw_layers.0.pw",
                5: "dw_pw_layers.1.dw", 6: "dw_pw_layers.1.pw",
            ]
            guard let mapped = layers[index] else { return nil }
            name = String(name[..<range.lowerBound]) + "embed." + mapped + "." + suffix[1]
        }
        return name
    }

    /// Splitting the fused output rows preserves Q/K/V ownership on every rank.
    private func projectionDestinations(_ name: String) -> [(name: String, rows: Range<Int>?)] {
        guard name.hasPrefix("language_model.model.layers.") else { return [(name, nil)] }
        for (fused, parts) in [
            (
                "qkv_proj",
                [
                    ("q_proj", config.heads * config.headDimension),
                    ("k_proj", config.kvHeads * config.headDimension),
                    ("v_proj", config.kvHeads * config.headDimension),
                ]
            ),
            (
                "gate_up_proj",
                [("gate_proj", config.intermediateSize), ("up_proj", config.intermediateSize)]
            ),
        ] {
            if let range = name.range(of: "." + fused) {
                let prefix = String(name[..<range.lowerBound]) + "."
                var suffix = String(name[range.upperBound...])
                if suffix.isEmpty {
                    suffix = ".base"
                } else if !suffix.hasPrefix(".base") && !suffix.hasPrefix(".vision_")
                    && !suffix.hasPrefix(".speech_")
                {
                    suffix = ".base" + suffix
                }
                let replicatedA = suffix.hasPrefix(".vision_a") || suffix.hasPrefix(".speech_a")
                var start = 0
                return parts.map { part, count in
                    defer { start += count }
                    return (prefix + part + suffix, replicatedA ? nil : start ..< start + count)
                }
            }
        }
        for projection in [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ] {
            if let range = name.range(of: "." + projection) {
                let prefix = String(name[..<range.upperBound])
                let suffix = String(name[range.upperBound...])
                if suffix.isEmpty { return [(prefix + ".base", nil)] }
                if !suffix.hasPrefix(".base") && !suffix.hasPrefix(".vision_")
                    && !suffix.hasPrefix(".speech_")
                {
                    return [(prefix + ".base" + suffix, nil)]
                }
            }
        }
        return [(name, nil)]
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        let mlxFormat = checkpoint.metadata["format"] == "mlx"
        var normalized: [String: MLXArray] = [:]
        for source in checkpoint.weights.keys.sorted() {
            guard let name = checkpointName(source), var tensor = checkpoint.weights[source] else {
                continue
            }
            if name.contains("patch_embedding.weight"), tensor.ndim == 4 {
                tensor = tensor.transposed(0, 2, 3, 1).reshaped(tensor.dim(0), -1)
            } else if name.hasPrefix("audio_encoder.") {
                if (name.contains("glu.b1") || name.contains("glu.b2")) && tensor.ndim == 3 {
                    tensor = tensor.reshaped(-1)
                } else if !mlxFormat && name.hasSuffix("weight") {
                    if tensor.ndim == 4 { tensor = tensor.transposed(0, 2, 3, 1) }
                    if tensor.ndim == 3 {
                        if name.contains("glu.ext_pw_conv_1d")
                            || name.contains("conv.ext_pw_conv_1d")
                        {
                            tensor = tensor[0..., 0..., 0]
                        } else {
                            tensor = tensor.transposed(0, 2, 1)
                        }
                    }
                }
            }
            for (destination, rows) in projectionDestinations(name) {
                guard normalized[destination] == nil else {
                    throw ModelFactoryError.invalidConfiguration(
                        "Duplicate Phi-4 checkpoint tensor: \(destination)")
                }
                normalized[destination] = rows.map { tensor[$0] } ?? tensor
            }
        }
        for (i, layer) in language.model.layers.enumerated() {
            let prefix = "language_model.model.layers.\(i)."
            for (name, projection) in [
                ("self_attn.q_proj", layer.attention.query),
                ("self_attn.k_proj", layer.attention.key),
                ("self_attn.v_proj", layer.attention.value),
                ("self_attn.o_proj", layer.attention.output),
                ("mlp.gate_proj", layer.mlp.gate), ("mlp.up_proj", layer.mlp.up),
                ("mlp.down_proj", layer.mlp.down),
            ] { projection.removeAbsentAdapters(prefix: prefix + name, weights: normalized) }
        }
        var result = checkpoint
        result.weights = normalized
        if let precision = checkpoint.perLayerQuantization {
            var options: [String: BaseConfiguration.QuantizationOption] = [:]
            for (source, option) in precision.perLayerQuantization {
                guard let name = checkpointName(source) else { continue }
                for destination in projectionDestinations(name) {
                    guard options[destination.name] == nil else {
                        throw ModelFactoryError.invalidConfiguration(
                            "Duplicate Phi-4 quantization setting: \(destination.name)")
                    }
                    options[destination.name] = option
                }
            }
            result.perLayerQuantization = .init(
                quantization: precision.quantization, perLayerQuantization: options)
        }
        return result
    }
}
