// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm MiniCPM models at 4693cd0 (MIT).
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// MiniCPM-o 4.5 and MiniCPM-V 4.6/4.7 media understanding, with text output.
/// Visual/audio encoders are replicated; the Qwen language decoder is tensor-parallel.
public final class MiniCPM: Module, VLMModel, KVCacheDimensionProvider, TensorParallelModel,
    BatchedMultimodalModel
{
    package static let imageToken = -200, videoToken = -201, audioToken = -202
    public let config: MiniCPMConfiguration
    private var shardLocalKVHeads: Int
    @ModuleInfo(key: "language_model") var language: Module
    @ModuleInfo(key: "vision_tower") package var vision: MiniCPMVision
    package var resampler: MiniCPMResampler?
    @ModuleInfo(key: "vit_merger") package var vitMerger: MiniCPMViTMerger?
    package var merger: MiniCPMMerger?
    @ModuleInfo(key: "audio_tower") package var audio: MiniCPMAudio?
    @ModuleInfo(key: "audio_projection_layer") package var audioProjection: MiniCPMAudioProjection?
    public var batchedMediaCapabilities: BatchedMediaCapabilities {
        config.isOmni && config.audio != nil ? [.image, .video, .audio] : [.image, .video]
    }
    public var batchTokenPlaceholderIDs: Set<Int> {
        config.isOmni && config.audio != nil
            ? [Self.imageToken, Self.videoToken, Self.audioToken]
            : [Self.imageToken, Self.videoToken]
    }
    private var metadataLayer: Int {
        config.hybrid?.resolvedLayerTypes.firstIndex(of: "full_attention") ?? 0
    }
    public var vocabularySize: Int { config.geometry.vocabularySize }
    public var kvHeads: [Int] {
        Array(repeating: shardLocalKVHeads, count: config.geometry.layers)
    }
    public var headDim: IntOrPair { .init(config.geometry.headDimension) }
    public var loraLayers: [Module] {
        if let q = language as? Qwen3Model { return q.loraLayers }
        return (language as! Qwen35TextModel).loraLayers
    }
    public var excludedCheckpointNamespaces: [String] { ["tts", "audio_avg_pooler"] }
    public init(_ config: MiniCPMConfiguration) {
        self.config = config
        self.shardLocalKVHeads = config.geometry.kvHeads
        _language.wrappedValue =
            config.text.map { Qwen3Model($0) } ?? Qwen35TextModel(config.hybrid!)
        _vision.wrappedValue = MiniCPMVision(config.vision)
        if config.isOmni {
            resampler = MiniCPMResampler(config)
            if let a = config.audio {
                _audio.wrappedValue = MiniCPMAudio(a)
                _audioProjection.wrappedValue = MiniCPMAudioProjection(
                    input: a.intermediateSize / 4, output: config.geometry.hiddenSize)
            }
        } else {
            _vitMerger.wrappedValue = MiniCPMViTMerger(config.vision, tanhGELU: config.isCanvas)
            merger = MiniCPMMerger(config)
        }
    }
    public func prepare() throws { if let q = language as? Qwen35TextModel { try q.prepare() } }
    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        var cache: [any KVCache]
        if let q = language as? Qwen3Model {
            cache = try q.newCache(parameters: parameters)
        } else {
            cache = try (language as! Qwen35TextModel).newCache(parameters: parameters)
        }
        let positions = ArraysCache(size: 1)
        positions[0] = MLXArray.zeros([1, 1], dtype: .int32)
        cache[metadataLayer] = CacheList(cache[metadataLayer], positions)
        return cache
    }
    package func decoderCaches(_ cache: [any KVCache]?) -> [any KVCache]? {
        cache?.enumerated().map { index, c in
            index == metadataLayer ? (c as? CacheList)?[0] ?? c : c
        }
    }
    private func nextPositions(_ cache: [any KVCache]?, batch: Int) -> MLXArray {
        guard let c = cache?[metadataLayer] as? CacheList, let state = c[1] as? ArraysCache,
            let values = state[0]
        else {
            return MLXArray.zeros([batch, 1], dtype: .int32)
        }
        return broadcast(values, to: [batch, 1])
    }
    private func setNextPositions(_ values: MLXArray, cache: [any KVCache]?) {
        if let c = cache?[metadataLayer] as? CacheList, let state = c[1] as? ArraysCache {
            state[0] = values
        }
    }
    private func forward(_ embeddings: MLXArray, cache: [any KVCache]?, positions: MLXArray?)
        -> MLXArray
    {
        let core = decoderCaches(cache)
        let result: MLXArray
        if let q = language as? Qwen3Model {
            result = q.embeddedLogits(embeddings, cache: core)
        } else {
            result = (language as! Qwen35TextModel).embeddedLogits(
                embeddings, cache: core, positionIds: positions)
        }
        // CacheList owns row metadata but its inherited offset is not forwarded automatically.
        if let c = cache?[metadataLayer] as? CacheList {
            c.offset = core?.map { $0.offset }.max() ?? 0
            (c[1] as? ArraysCache)?.advance(embeddings.dim(1))
        }
        return result
    }
    private func tokenEmbeddings(_ tokens: MLXArray) -> MLXArray {
        if let q = language as? Qwen3Model { return q.tokenEmbeddings(tokens) }
        return (language as! Qwen35TextModel).tokenEmbeddings(tokens)
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let b = inputs.ndim == 1 ? 1 : inputs.dim(0)
        let n = inputs.dim(-1)
        let start = nextPositions(cache, batch: b)
        let positions = broadcast(
            (start + MLXArray(0 ..< n)[.newAxis, 0...])[.newAxis, 0..., 0...], to: [3, b, n])
        let pending = (cache?[metadataLayer] as? CacheList).flatMap { $0[1] as? ArraysCache }?
            .currentLengths
        let consumed =
            pending.map { clip($0, min: 0, max: n).reshaped(b, 1) }
            ?? MLXArray.full([b, 1], values: MLXArray(Int32(n)), dtype: .int32)
        let output = forward(
            tokenEmbeddings(inputs), cache: cache, positions: config.isOmni ? nil : positions)
        setNextPositions(start + consumed, cache: cache)
        return output
    }
    private func visualLayout(_ input: LMInput) throws -> (
        images: [String], videos: [String], factors: [Int]
    ) {
        let tokens = input.text.tokens
        let batch = tokens.ndim == 1 ? 1 : tokens.dim(0)
        let width = tokens.dim(-1)
        if config.isOmni { return ([], [], Array(repeating: 1, count: batch)) }
        let ids = tokens.asArray(Int.self)
        let images = input.image?.frames ?? []
        let videos = input.video?.frames ?? []
        var imageModes: [String] = []
        var videoModes: [String] = []
        var factors = Array(repeating: config.mergeFactor, count: batch)
        for row in 0 ..< batch {
            var cursor = row * width
            var seenFactor: Int?
            while cursor < (row + 1) * width {
                let token = ids[cursor]
                guard token == Self.imageToken || token == Self.videoToken else {
                    cursor += 1
                    continue
                }
                let start = cursor
                while cursor < (row + 1) * width && ids[cursor] == token { cursor += 1 }
                let grids = token == Self.imageToken ? images : videos
                let index = token == Self.imageToken ? imageModes.count : videoModes.count
                guard index < grids.count else {
                    throw VLMError.processing("MiniCPM crop spans have no matching grid")
                }
                let grid = grids[index]
                let count = cursor - start
                let factor: Int
                if grid.h * grid.w == count * 4 {
                    factor = 2
                } else if grid.h * grid.w == count * 16 {
                    factor = 4
                } else {
                    throw VLMError.processing(
                        "MiniCPM crop bounds do not match 4x or 16x compression")
                }
                if let seenFactor, seenFactor != factor {
                    throw VLMError.processing(
                        "MiniCPM requires one visual compression mode within each request row")
                }
                seenFactor = factor
                factors[row] = factor
                if token == Self.imageToken {
                    imageModes.append(factor == 2 ? "4x" : "16x")
                } else {
                    videoModes.append(factor == 2 ? "4x" : "16x")
                }
            }
        }
        guard imageModes.count == images.count, videoModes.count == videos.count else {
            throw VLMError.processing("MiniCPM received unused crop grids")
        }
        return (imageModes, videoModes, factors)
    }
    package func visualFeatures(_ pixels: MLXArray, grids: [THW], downsampleModes: [String]? = nil)
        throws -> [MLXArray]
    {
        guard downsampleModes == nil || downsampleModes!.count == grids.count,
            pixels.ndim == 3, pixels.dim(0) == grids.count,
            pixels.dim(2) == config.vision.patchSize * config.vision.patchSize
                * config.vision.channels
        else {
            throw VLMError.processing(
                "MiniCPM pixels must be [crop,patches,patch²×channels] with one grid per crop")
        }
        return try grids.enumerated().map { index, grid in
            let mode = downsampleModes?[index] ?? config.downsampleMode
            let factor = config.isOmni ? 1 : mode == "4x" ? 2 : 4
            guard grid.t == 1, grid.h > 0, grid.w > 0, grid.h * grid.w <= pixels.dim(1),
                grid.h % factor == 0, grid.w % factor == 0
            else {
                throw VLMError.processing(
                    "MiniCPM crop grid is invalid for its spatial compression")
            }
            let (hidden, h, w) = vision(
                pixels[index, ..<(grid.h * grid.w), 0...], height: grid.h, width: grid.w,
                merger: config.isOmni || mode == "4x" ? nil : vitMerger,
                insertLayer: config.insertLayer)
            if let resampler { return resampler(hidden[.newAxis, 0..., 0...], height: h, width: w) }
            return merger!(hidden, height: h, width: w)
        }
    }
    package func inputEmbeddings(_ input: LMInput) throws -> MLXArray {
        let tokens =
            input.text.tokens.ndim == 1 ? input.text.tokens[.newAxis, 0...] : input.text.tokens
        guard tokens.ndim == 2 else { throw VLMError.processing("MiniCPM expects a token matrix") }
        let b = tokens.dim(0)
        let n = tokens.dim(1)
        let ids = tokens.asArray(Int.self)
        let layout = try visualLayout(input)
        var features: [Int: MLXArray] = [:]
        if let image = input.image {
            guard let grids = image.frames else {
                throw VLMError.processing("MiniCPM images require crop grids")
            }
            let arrays = try visualFeatures(
                image.pixels, grids: grids, downsampleModes: config.isOmni ? nil : layout.images)
            if !arrays.isEmpty { features[Self.imageToken] = concatenated(arrays) }
        }
        if let video = input.video {
            guard let grids = video.frames else {
                throw VLMError.processing("MiniCPM video requires frame crop grids")
            }
            let arrays = try visualFeatures(
                video.pixels, grids: grids, downsampleModes: config.isOmni ? nil : layout.videos)
            if !arrays.isEmpty { features[Self.videoToken] = concatenated(arrays) }
        }
        if let sound = input.audio {
            guard let audio, let audioProjection, let a = config.audio else {
                throw VLMError.processing("This MiniCPM checkpoint has no audio encoder")
            }
            guard sound.features.ndim == 3, sound.features.dim(2) == a.melBins,
                (sound.features.dim(1) + 1) / 2 <= a.positions
            else { throw VLMError.processing("Invalid MiniCPM-o log-mel features") }
            var arrays: [MLXArray] = []
            for index in 0 ..< sound.features.dim(0) {
                let length =
                    sound.mask.map { $0[index].asType(.int32).sum().item(Int.self) }
                    ?? sound.features.dim(1)
                guard length > 0, length <= sound.features.dim(1),
                    sound.features.dim(1) >= config.audioPoolStep * 2
                else {
                    throw VLMError.processing("MiniCPM-o audio is too short for its pooling stride")
                }
                arrays.append(
                    audioProjection(
                        audio(sound.features[index ..< (index + 1), 0..., 0...], length: length),
                        length: length, step: config.audioPoolStep))
            }
            if !arrays.isEmpty { features[Self.audioToken] = concatenated(arrays) }
        }
        var safe = ids
        for (token, encoded) in features {
            let indices = ids.indices.filter { ids[$0] == token }
            guard indices.count == encoded.dim(0) else {
                throw VLMError.processing("MiniCPM placeholder count does not match encoded media")
            }
            for index in indices { safe[index] = 0 }
        }
        guard safe.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw VLMError.processing("MiniCPM received an unmatched media placeholder")
        }
        let embedding = tokenEmbeddings(MLXArray(safe, [b, n])).reshaped(
            b * n, config.geometry.hiddenSize)
        for (token, encoded) in features {
            embedding[MLXArray(ids.indices.filter { ids[$0] == token })] = encoded.asType(
                embedding.dtype)
        }
        return embedding.reshaped(b, n, config.geometry.hiddenSize)
    }
    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let embeddings = try inputEmbeddings(input)
        let b = embeddings.dim(0)
        let n = embeddings.dim(1)
        let lengths = input.text.sequenceLengths ?? Array(repeating: n, count: b)
        guard lengths.count == b, lengths.allSatisfy({ $0 > 0 && $0 <= n }) else {
            throw VLMError.processing("Invalid MiniCPM prompt lengths")
        }
        let start = nextPositions(cache, batch: b)
        var positions = broadcast(MLXArray(0 ..< n)[.newAxis, .newAxis, 0...], to: [3, b, n])
        if config.isCanvas {
            if let supplied = input.image?.positionIds ?? input.video?.positionIds {
                guard supplied.shape == [3, b, n] else {
                    throw VLMError.processing(
                        "MiniCPM canvas positions must have shape [3,batch,sequence]")
                }
                positions = supplied
            } else if input.image != nil || input.video != nil {
                guard let isID = config.imageStartID, let ieID = config.imageEndID,
                    let ssID = config.sliceStartID,
                    let seID = config.sliceEndID, let nlID = config.newlineID
                else {
                    throw VLMError.processing(
                        "MiniCPM canvas needs processor positions or all marker IDs")
                }
                positions = try MiniCPMCanvas.positions(
                    tokens: input.text.tokens.asArray(Int.self), lengths: lengths, width: n,
                    images: input.image?.frames ?? [], videos: input.video?.frames ?? [],
                    factor: config.mergeFactor, rowFactors: try visualLayout(input).factors,
                    markers: .init(
                        imageStart: isID, imageEnd: ieID, sliceStart: ssID, sliceEnd: seID,
                        newline: nlID)
                ).0
            }
        }
        positions = positions + start[.newAxis, 0..., 0...]
        let valid = MLXArray(0 ..< n)[.newAxis, 0...] .< MLXArray(lengths)[0..., .newAxis]
        let next =
            which(valid[.newAxis, 0..., 0...], positions, MLXArray(Int32(-1))).max(axis: 0).max(
                axis: 1, keepDims: true) + 1
        for layer in cache { layer.prepare(lengths: lengths) }
        let processed = try prefill.forEachChunk(total: n) { range in
            _ = forward(
                embeddings[0..., range, 0...], cache: cache,
                positions: config.isOmni ? nil : positions[0..., 0..., range])
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        let logits = forward(
            embeddings[0..., processed..., 0...], cache: cache,
            positions: config.isOmni ? nil : positions[0..., 0..., processed...])
        setNextPositions(next, cache: cache)
        for layer in cache { layer.finalize() }
        prefill.progress?(n, n)
        return .logits(.init(logits: logits))
    }
    public func shard(group: MLXDistributed.Group) throws {
        if let q = language as? Qwen3Model {
            try q.shard(group: group)
        } else {
            try (language as! Qwen35TextModel).shard(group: group)
        }
        shardLocalKVHeads /= try group.size
    }
}
