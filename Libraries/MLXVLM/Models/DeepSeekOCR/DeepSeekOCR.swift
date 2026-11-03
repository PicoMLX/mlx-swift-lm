// Port of Blaizzy/mlx-vlm's deepseekocr and deepseekocr_2 (MIT).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// DeepSeek OCR and OCR2. Vision is replicated; the decoder supports tensor parallelism.
public final class DeepSeekOCR: Module, VLMModel, KVCacheDimensionProvider, TensorParallelModel,
    BatchedMultimodalModel
{
    public var batchedMediaCapabilities: BatchedMediaCapabilities { [.image] }
    public let config: DeepSeekOCRConfiguration
    @ModuleInfo(key: "language_model") var language: OCRLanguageModel
    @ModuleInfo(key: "sam_model") var sam: OCRSAMEncoder
    @ModuleInfo(key: "vision_model") var vision: Module
    let projector: OCRProjector
    @ParameterInfo(key: "view_separator") var viewSeparator: MLXArray
    @ParameterInfo(key: "image_newline") var newline: MLXArray?

    public var kvHeads: [Int] { language.model.layers.map { $0.attention.kvHeads } }
    public var loraLayers: [Module] { language.model.layers }

    public init(_ config: DeepSeekOCRConfiguration) {
        self.config = config
        _language.wrappedValue = OCRLanguageModel(config.text)
        _sam.wrappedValue = OCRSAMEncoder(
            config.sam,
            outputChannels: config.isOCR2 ? config.vision.encoderDim : config.vision.hiddenSize)
        _vision.wrappedValue =
            config.isOCR2 ? OCRVision2(config.vision) : OCRCLIPVision(config.vision)
        projector = OCRProjector(config.projector)
        _viewSeparator.wrappedValue = zeros([config.text.hiddenSize])
        if !config.isOCR2 { _newline.wrappedValue = zeros([config.text.hiddenSize]) }
    }

    func imageFeatures(_ pixels: MLXArray) -> MLXArray {
        let encoded = sam(pixels.transposed(0, 2, 3, 1))
        if let vision = vision as? OCRVision2 { return projector(vision(encoded)) }
        let clip = (vision as! OCRCLIPVision)(encoded)[0..., 1..., 0...]
        return projector(
            concatenated([clip, encoded.reshaped(encoded.dim(0), -1, encoded.dim(-1))], axis: -1))
    }

    private func withNewlines(_ features: MLXArray, rows: Int, columns: Int) -> MLXArray {
        let side = Int(sqrt(Double(features.dim(1))))
        let hidden = features.dim(-1)
        let spatial = features.reshaped(rows, columns, side, side, hidden)
            .transposed(0, 2, 1, 3, 4).reshaped(rows * side, columns * side, hidden)
        return concatenated([spatial, broadcast(newline!, to: [rows * side, 1, hidden])], axis: 1)
            .reshaped(-1, hidden)
    }

    /// Each frame records its text row (`t`) and local crop grid (`h`, `w`).
    func inputEmbeddings(_ input: LMInput) throws -> MLXArray {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        let embeddings = language.model.embedTokens(tokens)
        guard let image = input.image else { return embeddings }
        guard image.pixels.ndim == 4, image.pixels.dim(1) == 3,
            image.pixels.dim(2) > 0, image.pixels.dim(2) == image.pixels.dim(3)
        else { throw VLMError.processing("DeepSeek OCR expects square NCHW RGB images.") }
        let count = image.pixels.dim(0)
        let frames = image.frames ?? (0 ..< count).map { THW(tokens.dim(0) == 1 ? 0 : $0, 0, 0) }
        guard frames.count == count else {
            throw VLMError.processing("DeepSeek OCR image layout count does not match pixels.")
        }
        for frame in frames {
            guard frame.t >= 0, frame.t < tokens.dim(0), frame.h >= 0, frame.w >= 0,
                frame.h <= 64, frame.w <= 64, frame.h * frame.w <= 64,
                (frame.h == 0) == (frame.w == 0)
            else { throw VLMError.processing("Invalid DeepSeek OCR crop grid or text row.") }
        }
        let expectedCrops = frames.reduce(0) { $0 + $1.h * $1.w }
        if let crops = image.crops {
            guard crops.ndim == 4, crops.dim(1) == 3, crops.dim(2) > 0,
                crops.dim(2) == crops.dim(3), crops.dim(0) == expectedCrops
            else {
                throw VLMError.processing(
                    "DeepSeek OCR crop tensors do not match the declared layout.")
            }
        } else if expectedCrops != 0 {
            throw VLMError.processing("DeepSeek OCR local crops are missing.")
        }
        let global = imageFeatures(image.pixels)
        let local: MLXArray? = expectedCrops == 0 ? nil : image.crops.map(imageFeatures)
        var rowFeatures = [[MLXArray]](repeating: [], count: tokens.dim(0))
        var localOffset = 0
        for (index, frame) in frames.enumerated() {
            let cropCount = frame.h * frame.w
            var parts = [MLXArray]()
            if cropCount > 0 {
                guard let local, localOffset + cropCount <= local.dim(0) else {
                    throw VLMError.processing("DeepSeek OCR local crops are missing.")
                }
                let crops = local[localOffset ..< localOffset + cropCount, 0..., 0...]
                parts.append(
                    config.isOCR2
                        ? crops.reshaped(-1, config.text.hiddenSize)
                        : withNewlines(crops, rows: frame.h, columns: frame.w))
                localOffset += cropCount
            }
            let whole = global[index ..< index + 1, 0..., 0...]
            parts.append(config.isOCR2 ? whole[0] : withNewlines(whole, rows: 1, columns: 1))
            parts.append(viewSeparator.expandedDimensions(axis: 0))
            rowFeatures[frame.t].append(concatenated(parts, axis: 0))
        }
        guard localOffset == (local?.dim(0) ?? 0) else {
            throw VLMError.processing("DeepSeek OCR crop tensors exceed the declared layout.")
        }
        for row in 0 ..< tokens.dim(0) {
            let indices = tokens[row].asArray(Int.self).enumerated().compactMap {
                $0.element == config.imageTokenId ? $0.offset : nil
            }
            let features = rowFeatures[row]
            let featureCount = features.reduce(0) { $0 + $1.dim(0) }
            guard indices.count == featureCount else {
                throw VLMError.processing(
                    "DeepSeek OCR image features (\(featureCount)) do not match placeholder positions (\(indices.count))."
                )
            }
            if !indices.isEmpty {
                embeddings[row, MLXArray(indices)] = concatenated(features, axis: 0).asType(
                    embeddings.dtype)
            }
        }
        return embeddings
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.audio == nil, input.video == nil else {
            throw VLMError.processing("DeepSeek OCR accepts text and images.")
        }
        let embeddings = try inputEmbeddings(input)
        return try withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            let total = embeddings.dim(1)
            let processed = try prefill.forEachChunk(total: total) { range in
                _ = language(embeddings[0..., range, 0...], cache: cache)
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }
            let logits = language(embeddings[0..., processed..., 0...], cache: cache)
            prefill.progress?(total, total)
            return .logits(LMOutput(logits: logits))
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        language(language.model.embedTokens(inputs), cache: cache)
    }

    public func shard(group: MLXDistributed.Group) throws {
        try validateShardable(self)
        let text = config.text
        try requireDivisible(
            [
                "attention heads": text.heads,
                "key-value heads": text.usesMLA ? text.heads : text.kvHeads,
                "dense intermediate width": text.intermediateSize,
                "expert intermediate width": text.moeIntermediateSize,
            ], by: group.size)
        for layer in language.model.layers { try layer.shard(group: group) }
    }

    func mappedCheckpointName(_ original: String) -> String? {
        if original.contains("position_ids") { return nil }
        var name = original
        for (source, destination) in [
            ("model.qwen2_model.model.model.layers", "vision_model.qwen2_encoder.layers"),
            ("model.qwen2_model.model.model.norm", "vision_model.qwen2_encoder.norm"),
            ("model.qwen2_model.query_1024.weight", "vision_model.qwen2_encoder.query_1024"),
            ("model.qwen2_model.query_768.weight", "vision_model.qwen2_encoder.query_768"),
            ("model.qwen2_model.query_1024", "vision_model.qwen2_encoder.query_1024"),
            ("model.qwen2_model.query_768", "vision_model.qwen2_encoder.query_768"),
            ("model.vision_model", "vision_model"), ("model.sam_model", "sam_model"),
            ("model.projector", "projector"), ("model.view_seperator", "view_separator"),
            ("model.image_newline", "image_newline"),
        ] where name.hasPrefix(source) { name = destination + name.dropFirst(source.count) }
        if name.hasPrefix("model.layers.") || name.hasPrefix("model.embed_tokens.")
            || name.hasPrefix("model.norm.")
        {
            name = "language_model." + name
        }
        if name.hasPrefix("lm_head.") { name = "language_model." + name }
        if config.projector.type == "linear",
            name == "projector.layers" || name.hasPrefix("projector.layers.")
        {
            let suffix = String(name.dropFirst("projector.layers".count))
            if !suffix.hasPrefix(".0") { name = "projector.layers.0" + suffix }
        }
        return name
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = try checkpoint.mapNames(mappedCheckpointName)
        checkpoint.weights = try sanitizeMapped(checkpoint.weights)
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try sanitizeMapped(ModelCheckpoint(weights: weights).mapNames(mappedCheckpointName).weights)
    }

    private func sanitizeMapped(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var result = weights
        if let experts = config.text.experts {
            for layer in 0 ..< config.text.layers {
                let prefix = "language_model.model.layers.\(layer).mlp"
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    for suffix in ["weight", "scales", "biases"] {
                        let keys = (0 ..< experts).map {
                            "\(prefix).experts.\($0).\(projection).\(suffix)"
                        }
                        if keys.contains(where: { result[$0] != nil }) {
                            guard keys.allSatisfy({ result[$0] != nil }) else {
                                throw VLMError.processing(
                                    "Incomplete DeepSeek OCR expert checkpoint.")
                            }
                            result["\(prefix).switch_mlp.\(projection).\(suffix)"] = stacked(
                                keys.map { result.removeValue(forKey: $0)! })
                        }
                    }
                }
            }
        }
        let expected = Dictionary(uniqueKeysWithValues: parameters().flattened())
        for (key, value) in result where value.ndim == 4 {
            if let shape = expected[key]?.shape, value.shape != shape,
                value.transposed(0, 2, 3, 1).shape == shape
            {
                result[key] = value.transposed(0, 2, 3, 1)
            }
        }
        return result
    }
}
