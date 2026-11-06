// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm MiniCPM checkpoint mapping at 4693cd0 (MIT).
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

extension MiniCPM {
    private func checkpointName(_ source: String) -> String? {
        var name = source
        if name.hasPrefix("model.") { name = String(name.dropFirst(6)) }
        if name.hasPrefix("tts.") || name.hasPrefix("audio_avg_pooler.")
            || name.contains("position_ids") || name.contains("rotary_emb.inv_freq")
        {
            return nil
        }
        for (old, new) in [
            ("llm.", "language_model."), ("vpm.", "vision_tower."), ("visual.", "vision_tower."),
            ("apm.", "audio_tower."),
            ("vision_tower.vit_merger.", "vit_merger."),
        ] where name.hasPrefix(old) {
            name = new + name.dropFirst(old.count)
            break
        }
        if name.hasPrefix("lm_head.") { name = "language_model." + name }
        if name.hasPrefix("language_model.") && !name.hasPrefix("language_model.model.")
            && !name.hasPrefix("language_model.lm_head.")
        {
            name = "language_model.model." + name.dropFirst("language_model.".count)
        }
        return name
    }
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }
    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var normalized = try checkpoint.mapNames(checkpointName)
        var fusedWeight: MLXArray?
        var fusedBias: MLXArray?
        for name in normalized.weights.keys.sorted() {
            guard var tensor = normalized.weights[name] else { continue }
            if name == "resampler.attn.in_proj_weight" {
                fusedWeight = tensor
                normalized.weights.removeValue(forKey: name)
                continue
            }
            if name == "resampler.attn.in_proj_bias" {
                fusedBias = tensor
                normalized.weights.removeValue(forKey: name)
                continue
            }
            if name.hasSuffix("embeddings.patch_embedding.weight"), tensor.ndim == 4,
                tensor.dim(1) == config.vision.channels
            {
                tensor = tensor.transposed(0, 2, 3, 1)
            }
            if name == "audio_tower.conv1.weight" || name == "audio_tower.conv2.weight",
                tensor.ndim == 3, tensor.dim(1) != 3
            {
                tensor = tensor.transposed(0, 2, 1)
            }
            normalized.weights[name] = tensor
        }
        for (part, index) in [("q_proj", 0), ("k_proj", 1), ("v_proj", 2)] {
            for (tensor, suffix) in [(fusedWeight, "weight"), (fusedBias, "bias")] {
                if let tensor {
                    guard tensor.dim(0) == config.geometry.hiddenSize * 3 else {
                        throw ModelFactoryError.invalidConfiguration(
                            "Invalid fused MiniCPM resampler attention")
                    }
                    let name = "resampler.attn.\(part).\(suffix)"
                    guard normalized.weights[name] == nil else {
                        throw ModelFactoryError.invalidConfiguration(
                            "Duplicate MiniCPM resampler projection")
                    }
                    normalized.weights[name] =
                        tensor[
                            (index * config.geometry.hiddenSize)
                                ..< ((index + 1) * config.geometry.hiddenSize)]
                }
            }
        }
        let prefix = "language_model."
        var text = try normalized.mapNames {
            $0.hasPrefix(prefix) ? String($0.dropFirst(prefix.count)) : nil
        }
        if let hybrid = config.hybrid, hybrid.numExperts > 0 {
            text = try prepareExperts(text, count: hybrid.numExperts)
        }
        let prepared: ModelCheckpoint
        if let q = language as? Qwen3Model {
            prepared = try q.prepareCheckpoint(text)
        } else {
            prepared = try (language as! Qwen35TextModel).prepareCheckpoint(text)
        }
        let namespaced = try prepared.mapNames { prefix + $0 }
        var result = try normalized.mapNames { $0.hasPrefix(prefix) ? nil : $0 }
        result.weights.merge(namespaced.weights) { _, _ in
            preconditionFailure("MiniCPM checkpoint namespaces overlap")
        }
        if let precision = normalized.perLayerQuantization {
            var options = result.perLayerQuantization?.perLayerQuantization ?? [:]
            options.merge(namespaced.perLayerQuantization?.perLayerQuantization ?? [:]) { a, _ in a
            }
            result.perLayerQuantization = .init(
                quantization: precision.quantization, perLayerQuantization: options)
        }
        return result
    }
    private func prepareExperts(_ checkpoint: ModelCheckpoint, count: Int) throws -> ModelCheckpoint
    {
        var result = checkpoint
        func assign(_ tensor: MLXArray, to name: String) throws {
            guard result.weights[name] == nil else {
                throw ModelFactoryError.invalidConfiguration(
                    "Duplicate MiniCPM expert tensor: \(name)")
            }
            result.weights[name] = tensor
        }
        for layer in 0 ..< config.geometry.layers {
            let prefix = "model.layers.\(layer).mlp"
            let fused = "\(prefix).experts.gate_up_proj"
            let gate = "\(prefix).switch_mlp.gate_proj"
            let up = "\(prefix).switch_mlp.up_proj"
            for suffix in ["", ".weight", ".scales", ".biases", "_scales", "_biases"] {
                guard let tensor = result.weights.removeValue(forKey: fused + suffix) else {
                    continue
                }
                guard tensor.ndim == 3, tensor.dim(1) % 2 == 0 else {
                    throw ModelFactoryError.invalidConfiguration(
                        "Invalid MiniCPM fused expert rows")
                }
                let half = tensor.dim(1) / 2
                let parameter =
                    suffix.isEmpty ? ".weight" : suffix.replacingOccurrences(of: "_", with: ".")
                try assign(tensor[0..., ..<half, 0...], to: gate + parameter)
                try assign(tensor[0..., half..., 0...], to: up + parameter)
            }
            if let option = result.perLayerQuantization?.perLayerQuantization.removeValue(
                forKey: fused)
            {
                result.perLayerQuantization?.perLayerQuantization[gate] = option
                result.perLayerQuantization?.perLayerQuantization[up] = option
            }
            let source = "\(prefix).experts.down_proj"
            let destination = "\(prefix).switch_mlp.down_proj"
            for suffix in ["", ".weight", ".scales", ".biases", "_scales", "_biases"] {
                if let tensor = result.weights.removeValue(forKey: source + suffix) {
                    try assign(
                        tensor,
                        to: destination
                            + (suffix.isEmpty
                                ? ".weight" : suffix.replacingOccurrences(of: "_", with: ".")))
                }
            }
            if let option = result.perLayerQuantization?.perLayerQuantization.removeValue(
                forKey: source)
            {
                result.perLayerQuantization?.perLayerQuantization[destination] = option
            }
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let modules = (0 ..< count).map { "\(prefix).experts.\($0).\(projection)" }
                let output = "\(prefix).switch_mlp.\(projection)"
                guard modules.contains(where: { result.weights[$0 + ".weight"] != nil }) else {
                    continue
                }
                for parameter in [".weight", ".scales", ".biases"] {
                    guard modules.contains(where: { result.weights[$0 + parameter] != nil }) else {
                        continue
                    }
                    let tensors = try modules.map { module -> MLXArray in
                        guard let tensor = result.weights.removeValue(forKey: module + parameter)
                        else {
                            throw ModelFactoryError.invalidConfiguration(
                                "Incomplete MiniCPM expert checkpoint")
                        }
                        return tensor
                    }
                    guard tensors.allSatisfy({ $0.shape == tensors[0].shape }) else {
                        throw ModelFactoryError.invalidConfiguration(
                            "MiniCPM experts have inconsistent dimensions")
                    }
                    try assign(stacked(tensors), to: output + parameter)
                }
                if let precision = result.perLayerQuantization {
                    let options = modules.map {
                        precision.perLayerQuantization[$0] ?? precision.quantization.map {
                            .quantize($0)
                        } ?? .skip
                    }
                    guard
                        options.dropFirst().allSatisfy({ candidate in
                            switch (candidate, options[0]) {
                            case (.skip, .skip): return true
                            case (.quantize(let a), .quantize(let b)): return a == b
                            default: return false
                            }
                        })
                    else {
                        throw ModelFactoryError.invalidConfiguration(
                            "MiniCPM experts have inconsistent precision")
                    }
                    for module in modules {
                        result.perLayerQuantization?.perLayerQuantization.removeValue(
                            forKey: module)
                    }
                    result.perLayerQuantization?.perLayerQuantization[output] = options[0]
                }
            }
        }
        return result
    }

}
