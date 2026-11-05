// Copyright © 2026 Apple Inc.
// Thinker input-to-text port. See ThirdPartyNotices/Qwen3Omni for pinned upstream sources.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Qwen3OmniLanguage: Module {
    @ModuleInfo var model: Qwen3VLMoELanguage.Model
    @ModuleInfo(key: "lm_head") var head: Linear?
    init(_ config: Qwen3VLMoEConfiguration.TextConfiguration) {
        _model.wrappedValue = Qwen3VLMoELanguage.Model(config)
        _head.wrappedValue =
            config.tieWordEmbeddings
            ? nil : Linear(config.hiddenSize, config.vocabSize, bias: false)
    }
    func logits(_ hidden: MLXArray) -> MLXArray {
        if let head { return head(hidden) }
        return model.embedTokens.asLinear(hidden)
    }
}

/// Qwen3 Omni MoE Thinker: images, timestamped video and audio input, text output.
/// Language positions and modality continuation deltas belong to each request's cache.
public final class Qwen3Omni: Module, VLMModel, KVCacheDimensionProvider, TensorParallelModel,
    BatchedMultimodalModel
{
    public let config: Qwen3OmniConfiguration
    @ModuleInfo(key: "language_model") var language: Qwen3OmniLanguage
    @ModuleInfo(key: "vision_tower") var vision: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "audio_tower") var audio: Qwen3OmniAudio.Model
    public var excludedCheckpointNamespaces: [String] { ["talker", "code2wav"] }
    public var batchedMediaCapabilities: BatchedMediaCapabilities { [.image, .video, .audio] }
    public var loraLayers: [Module] { language.model.layers }
    public var kvHeads: [Int] { language.model.layers.map { $0.attention.kvHeads } }
    private var sharded = false

    public init(_ config: Qwen3OmniConfiguration) {
        self.config = config
        _language.wrappedValue = Qwen3OmniLanguage(config.thinker.text)
        _vision.wrappedValue = Qwen3VLVision.VisionModel(config.thinker.vision)
        _audio.wrappedValue = Qwen3OmniAudio.Model(config.thinker.audio)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        var caches = try kvHeads.map { _ in try makeAttentionKVCache(parameters: parameters) }
        caches[0] = CacheList(caches[0], ArraysCache(size: 1))
        return caches
    }

    private func splitCache(_ cache: [KVCache]?) -> (
        layers: [KVCache]?, metadata: ArraysCache?, list: CacheList?
    ) {
        guard var layers = cache, !layers.isEmpty else { return (nil, nil, nil) }
        guard let list = layers[0] as? CacheList, let metadata = list[1] as? ArraysCache else {
            preconditionFailure(
                "Qwen3 Omni requires caches from newCache, including request-local MRoPE metadata")
        }
        layers[0] = list[0]
        return (layers, metadata, list)
    }

    private func offsets(_ cache: KVCache?, batch: Int) -> MLXArray {
        if let cache = cache as? any BatchPositionedKVCache {
            return cache.batchOffset.asType(.float32).reshaped(batch, 1)
        }
        return (MLXArray.zeros([batch, 1]) + Float(cache?.offset ?? 0))
    }

    private func forward(
        _ tokens: MLXArray, cache: [KVCache]?, embeddings: MLXArray? = nil,
        positions: MLXArray? = nil, visualMask: MLXArray? = nil, deepstack: [MLXArray]? = nil
    ) -> LMOutput {
        let parts = splitCache(cache)
        let batch = tokens.dim(0)
        let length = tokens.dim(1)
        let delta = parts.metadata?[0] ?? MLXArray.zeros([batch, 1])
        let positions =
            positions
            ?? broadcast(
                (offsets(parts.layers?.first, batch: batch)
                    + delta + MLXArray(0 ..< length).asType(.float32)[.newAxis])[.newAxis],
                to: [3, batch, length])
        if parts.metadata?[0] == nil { parts.metadata?[0] = delta }
        let hidden = language.model(
            tokens, cache: parts.layers, inputEmbeddings: embeddings,
            mask: nil, positionIds: positions, visualMask: visualMask, deepstackEmbeds: deepstack)
        if let offset = parts.layers?.first?.offset {
            parts.metadata?.offset = offset
            parts.list?.offset = offset
        }
        return LMOutput(logits: language.logits(hidden))
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        forward(input.tokens, cache: cache)
    }

    /// Strict, batch-major scatter also handles arbitrary interleaving of image/video/audio tokens.
    static func scatter(_ embeddings: MLXArray, tokens: MLXArray, token: Int, features: MLXArray)
        throws -> MLXArray
    {
        let indices = tokens.asArray(Int.self).enumerated().compactMap {
            $0.element == token ? UInt32($0.offset) : nil
        }
        guard features.ndim == 2, features.dim(0) == indices.count,
            features.dim(1) == embeddings.dim(-1)
        else {
            throw VLMError.processing(
                "Qwen3 Omni feature/token mismatch for token \(token): \(indices.count) tokens, \(features.shape) features"
            )
        }
        let flattened = embeddings.reshaped(-1, embeddings.dim(-1))
        if !indices.isEmpty { flattened[MLXArray(indices)] = features.asType(embeddings.dtype) }
        return flattened.reshaped(embeddings.shape)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.image != nil || input.video != nil || input.audio != nil else {
            return .tokens(input.text)
        }
        let tokens = input.text.tokens
        guard tokens.ndim == 2, tokens.dim(1) > 0 else {
            throw VLMError.processing("Omni requires nonempty batched tokens")
        }
        var embeddings = language.model.embedTokens(tokens)
        let ids = tokens.asArray(Int.self)
        var deepstack = Array(
            repeating: MLXArray.zeros(embeddings.shape, dtype: embeddings.dtype),
            count: config.thinker.vision.deepstackVisualIndexes.count)
        for (media, token) in [
            (input.image.map { ($0.pixels, $0.frames) }, config.thinker.imageToken),
            (input.video.map { ($0.pixels, $0.frames) }, config.thinker.videoToken),
        ] {
            if let media {
                guard let grids = media.1, !grids.isEmpty else {
                    throw VLMError.processing("Omni vision input requires grids")
                }
                let (features, residuals) = vision(
                    media.0.asType(vision.patchEmbed.proj.weight.dtype), gridTHW: grids)
                embeddings = try Self.scatter(
                    embeddings, tokens: tokens, token: token, features: features)
                guard residuals.count == deepstack.count else {
                    throw VLMError.processing("Omni deepstack layer count mismatch")
                }
                for index in deepstack.indices {
                    deepstack[index] = try Self.scatter(
                        deepstack[index], tokens: tokens, token: token, features: residuals[index])
                }
            } else if ids.contains(token) {
                throw VLMError.processing("Omni visual placeholders have no corresponding media")
            }
        }
        var audioLengths: [Int] = []
        if let prepared = input.audio {
            let features = prepared.features
            guard features.ndim == 3, features.dim(1) == config.thinker.audio.melBins else {
                throw VLMError.processing(
                    "Omni audio input must contain [items, mel, frames] features")
            }
            audioLengths =
                prepared.mask?.asType(.int32).sum(axis: -1).asArray(Int.self)
                ?? Array(repeating: features.dim(2), count: features.dim(0))
            guard audioLengths.count == features.dim(0),
                audioLengths.allSatisfy({ $0 > 0 && $0 <= features.dim(2) })
            else {
                throw VLMError.processing("Omni audio mask must describe valid mel frames per item")
            }
            let packed = concatenated(
                audioLengths.enumerated().map { features[$0.offset, 0..., 0 ..< $0.element] },
                axis: 1)
            embeddings = try Self.scatter(
                embeddings, tokens: tokens, token: config.thinker.audioToken,
                features: audio(packed, lengths: audioLengths))
        } else if ids.contains(config.thinker.audioToken) {
            throw VLMError.processing("Omni audio placeholders have no corresponding audio")
        }

        // Processor-provided positions retain actual video timestamps instead of assuming a frame rate.
        let localPositions: MLXArray
        if let provided = input.video?.positionIds ?? input.image?.positionIds {
            localPositions = provided
        } else {
            localPositions = try Qwen3OmniPositions.make(
                tokens: tokens, mask: input.text.mask,
                images: input.image?.frames ?? [], videos: input.video?.frames ?? [],
                secondsPerVideoGrid: Array(repeating: 1, count: input.video?.frames?.count ?? 0),
                audioLengths: audioLengths, config: .init(config.thinker)
            ).positions
        }
        guard localPositions.shape == [3, tokens.dim(0), tokens.dim(1)] else {
            throw VLMError.processing("Omni media positions must cover all prompt tokens")
        }
        let parts = splitCache(cache)
        let anchor =
            offsets(parts.layers?.first, batch: tokens.dim(0))
            + (parts.metadata?[0] ?? MLXArray.zeros([tokens.dim(0), 1]))
        let positions = localPositions + anchor[.newAxis]
        let validLengths =
            input.text.mask?.sum(axis: -1, keepDims: true).asType(.float32)
            ?? (MLXArray.zeros([tokens.dim(0), 1]) + Float(tokens.dim(1)))
        let localEnd = localPositions.max(axis: 0).max(axis: -1, keepDims: true) + 1
        parts.metadata?[0] =
            anchor + localEnd - offsets(parts.layers?.first, batch: tokens.dim(0)) - validLengths
        var output: LMOutput?
        func consume(_ range: Range<Int>) {
            let chunkResiduals = deepstack.map { $0[0..., range, 0...] }
            // Reuse the shared language core with every chunk position marked: residuals are already full-token tensors.
            let visualMask = MLXArray.ones([tokens.dim(0), range.count], dtype: .bool)
            output = forward(
                tokens[0..., range], cache: cache, embeddings: embeddings[0..., range, 0...],
                positions: positions[0..., 0..., range], visualMask: visualMask,
                deepstack: chunkResiduals.map { $0.reshaped(-1, $0.dim(-1)) })
            if let output { eval([output.logits] + cache.flatMap { $0.innerState() }) }
        }
        let processed = try prefill.forEachChunk(total: tokens.dim(1), reserving: 1) { consume($0) }
        consume(processed ..< tokens.dim(1))
        prefill.progress?(tokens.dim(1), tokens.dim(1))
        return .logits(output!)
    }

    public func shard(group: MLXDistributed.Group) throws {
        guard !sharded else { throw TensorParallelError.alreadySharded }
        try validateShardable(self)
        let c = config.thinker.text
        let size = try group.size
        try requireDivisible(
            [
                "attention heads": c.numAttentionHeads, "KV heads": c.numKeyValueHeads,
                "dense MLP width": c.intermediateSize, "expert MLP width": c.moeIntermediateSize,
            ], by: size)
        for layer in language.model.layers {
            try layer.attention.shard(group: group)
            if let mlp = layer.mlp as? Qwen3VLMoELanguage.MLP { try mlp.shard(group: group) }
            if let moe = layer.mlp as? Qwen3VLMoELanguage.SparseMoeBlock {
                try moe.shard(group: group)
            }
        }
        sharded = true
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        // Quantization overrides use the same paths as weights, including the Thinker prefix.
        var result = try checkpoint.mapNames(Self.checkpointName)
        result.weights = sanitize(weights: result.weights)
        if let quantization = result.perLayerQuantization {
            var settings: [String: BaseConfiguration.QuantizationOption] = [:]
            func same(
                _ lhs: BaseConfiguration.QuantizationOption,
                _ rhs: BaseConfiguration.QuantizationOption
            ) -> Bool {
                switch (lhs, rhs) {
                case (.skip, .skip): return true
                case (.quantize(let lhs), .quantize(let rhs)): return lhs == rhs
                default: return false
                }
            }
            for (name, option) in quantization.perLayerQuantization {
                var destinations = [name]
                if let range = name.range(of: ".experts.") {
                    let prefix = String(name[..<range.lowerBound]) + ".switch_mlp."
                    let suffix = name[range.upperBound...]
                    if suffix == "gate_up_proj" {
                        destinations = [prefix + "gate_proj", prefix + "up_proj"]
                    } else if suffix == "down_proj" {
                        destinations = [prefix + "down_proj"]
                    } else {
                        let pieces = suffix.split(separator: ".", maxSplits: 1)
                        if pieces.count == 2, Int(pieces[0]) != nil {
                            destinations = [prefix + pieces[1]]
                        }
                    }
                }
                for destination in destinations {
                    if let existing = settings[destination], !same(existing, option) {
                        throw VLMError.processing(
                            "Omni experts stacked into \(destination) must use identical quantization settings"
                        )
                    }
                    settings[destination] = option
                }
            }
            result.perLayerQuantization = .init(
                quantization: quantization.quantization, perLayerQuantization: settings)
        }
        return result
    }

    private static func checkpointName(_ raw: String) -> String? {
        if raw == "talker" || raw.hasPrefix("talker.") || raw == "code2wav"
            || raw.hasPrefix("code2wav.")
        {
            return nil
        }
        var key = raw.hasPrefix("thinker.") ? String(raw.dropFirst(8)) : raw
        if key.hasPrefix("model.") { key = "language_model." + key }
        if key.hasPrefix("lm_head.") { key = "language_model." + key }
        key = key.replacingOccurrences(of: "visual.", with: "vision_tower.")
        key = key.replacingOccurrences(of: ".merger_list.", with: ".deepstack_merger_list.")
        if key.contains("merger") {
            key = key.replacingOccurrences(of: ".ln_q", with: ".norm")
                .replacingOccurrences(of: ".mlp.0", with: ".linear_fc1")
                .replacingOccurrences(of: ".mlp.2", with: ".linear_fc2")
        }
        return key
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        var experts: [String: [Int: MLXArray]] = [:]
        for (raw, value) in weights {
            guard let key = Self.checkpointName(raw) else { continue }
            if config.thinker.text.tieWordEmbeddings && key.hasPrefix("language_model.lm_head.") {
                continue
            }
            if let range = key.range(of: ".experts.gate_up_proj"), value.ndim == 3 {
                let prefix = String(key[..<range.lowerBound]) + ".switch_mlp."
                let suffix =
                    key[range.upperBound...].isEmpty ? ".weight" : String(key[range.upperBound...])
                let width = config.thinker.text.moeIntermediateSize
                if value.dim(1) == width * 2 {
                    result[prefix + "gate_proj" + suffix] = value[0..., ..<width, 0...]
                    result[prefix + "up_proj" + suffix] = value[0..., width..., 0...]
                } else {
                    precondition(value.dim(2) == width * 2, "Invalid fused Omni expert dimensions")
                    result[prefix + "gate_proj" + suffix] = value[0..., 0..., ..<width].transposed(
                        0, 2, 1)
                    result[prefix + "up_proj" + suffix] = value[0..., 0..., width...].transposed(
                        0, 2, 1)
                }
                continue
            }
            if let range = key.range(of: ".experts.down_proj"), value.ndim == 3 {
                let suffix =
                    key[range.upperBound...].isEmpty ? ".weight" : String(key[range.upperBound...])
                result[String(key[..<range.lowerBound]) + ".switch_mlp.down_proj" + suffix] = value
                continue
            }
            if let range = key.range(of: ".experts.") {
                let suffix = key[range.upperBound...].split(separator: ".", maxSplits: 1)
                if suffix.count == 2, let expert = Int(suffix[0]) {
                    let destination = String(key[..<range.lowerBound]) + ".switch_mlp." + suffix[1]
                    experts[destination, default: [:]][expert] = value
                    continue
                }
            }
            if key.hasPrefix("audio_tower.conv2d"), key.hasSuffix(".weight"), value.ndim == 4 {
                let expected =
                    key.contains("conv2d1") ? audio.conv1.weight.shape : audio.conv2.weight.shape
                result[key] = value.shape == expected ? value : value.transposed(0, 2, 3, 1)
            } else {
                result[key] = value
            }
        }
        for (key, values) in experts {
            precondition(
                values.count == config.thinker.text.numExperts,
                "Incomplete Omni expert checkpoint for \(key)")
            result[key] = stacked((0 ..< values.count).map { values[$0]! })
        }
        return vision.sanitize(weights: result)
    }
}
