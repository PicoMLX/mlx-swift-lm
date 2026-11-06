// Copyright © 2026 Apple Inc.
// Whisper feature extraction and Omni token interleaving follow Transformers (Apache-2.0).

@preconcurrency import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon

public struct Qwen3OmniProcessorConfiguration: Decodable, Sendable {
    let vision: Qwen3VLProcessorConfiguration
    let melBins: Int
    let sampleRate: Int
    let fftSize: Int
    let hop: Int
    let positions: Qwen3OmniPositions.Parameters
    let maximumVideoFrames: Int

    public init(from decoder: Decoder) throws {
        // Checkpoints use both flat and nested image/audio processor metadata.
        struct Values: Decodable {
            let imageMean: [Double]?, imageStd: [Double]?, minPixels: Int?, maxPixels: Int?
            let patchSize: Int?, mergeSize: Int?, temporalPatchSize: Int?
            let featureSize: Int?, samplingRate: Int?, nFft: Int?, hopLength: Int?
            let positionIdPerSeconds: Double?, nWindow: Int?
            let imageTokenId: Int?, videoTokenId: Int?, audioTokenId: Int?
            let visionStartTokenId: Int?, visionEndTokenId: Int?
            let audioStartTokenId: Int?, audioEndTokenId: Int?
            let maxFrames: Int?
            enum CodingKeys: String, CodingKey {
                case imageMean = "image_mean"
                case imageStd = "image_std"
                case minPixels = "min_pixels"
                case maxPixels = "max_pixels"
                case patchSize = "patch_size"
                case mergeSize = "merge_size"
                case temporalPatchSize = "temporal_patch_size"
                case featureSize = "feature_size"
                case samplingRate = "sampling_rate"
                case nFft = "n_fft"
                case hopLength = "hop_length"
                case positionIdPerSeconds = "position_id_per_seconds"
                case nWindow = "n_window"
                case imageTokenId = "image_token_id"
                case videoTokenId = "video_token_id"
                case audioTokenId = "audio_token_id"
                case visionStartTokenId = "vision_start_token_id"
                case visionEndTokenId = "vision_end_token_id"
                case audioStartTokenId = "audio_start_token_id"
                case audioEndTokenId = "audio_end_token_id"
                case maxFrames = "max_frames"
            }
        }
        struct Nested: Decodable {
            let imageProcessor: Values?, featureExtractor: Values?, videoProcessor: Values?
            enum CodingKeys: String, CodingKey {
                case imageProcessor = "image_processor"
                case featureExtractor = "feature_extractor"
                case videoProcessor = "video_processor"
            }
        }
        let flat = try Values(from: decoder)
        let nested = try Nested(from: decoder)
        let image = nested.imageProcessor ?? flat
        let audio = nested.featureExtractor ?? flat
        maximumVideoFrames = nested.videoProcessor?.maxFrames ?? 768
        let merge = image.mergeSize ?? 2
        let object: [String: Any] = [
            "image_mean": image.imageMean ?? [0.5, 0.5, 0.5],
            "image_std": image.imageStd ?? [0.5, 0.5, 0.5],
            "min_pixels": image.minPixels ?? 4 * 32 * 32,
            "max_pixels": image.maxPixels ?? 1280 * 32 * 32,
            "patch_size": image.patchSize ?? 16, "merge_size": merge,
            "temporal_patch_size": image.temporalPatchSize ?? 2,
            "image_processor_type": "Qwen2VLImageProcessorFast",
        ]
        vision = try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self,
            from: JSONSerialization.data(withJSONObject: object))
        melBins = audio.featureSize ?? 128
        sampleRate = audio.samplingRate ?? 16_000
        fftSize = audio.nFft ?? 400
        hop = audio.hopLength ?? 160
        var p = Qwen3OmniPositions.Parameters()
        p.positionsPerSecond = flat.positionIdPerSeconds ?? 13
        p.merge = merge
        p.audioWindow = flat.nWindow ?? 50
        p.imageToken = flat.imageTokenId ?? p.imageToken
        p.videoToken = flat.videoTokenId ?? p.videoToken
        p.audioToken = flat.audioTokenId ?? p.audioToken
        p.visionStartToken = flat.visionStartTokenId ?? p.visionStartToken
        p.visionEndToken = flat.visionEndTokenId ?? p.visionEndToken
        p.audioStartToken = flat.audioStartTokenId ?? p.audioStartToken
        p.audioEndToken = flat.audioEndTokenId ?? p.audioEndToken
        positions = p
        guard melBins > 0, sampleRate > 0, fftSize > 0, fftSize % 2 == 0,
            hop > 0, merge > 0, maximumVideoFrames > 0, vision.patchSize > 0,
            vision.temporalPatchSize > 0,
            vision.imageMean.count == 3, vision.imageStd.count == 3,
            vision.imageStd.allSatisfy({ $0 > 0 }), positions.audioWindow > 0,
            positions.positionsPerSecond > 0
        else {
            throw VLMError.processing("Invalid Qwen3 Omni processor dimensions")
        }
    }
}

extension Qwen3OmniProcessorConfiguration {
    /// Resolve split HF sidecars and model-owned modality IDs before decoding.
    static func mergedData(directory: URL, modelConfiguration: Data, selected: Data) throws -> Data
    {
        func object(_ data: Data) throws -> [String: Any] {
            guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw VLMError.processing("Omni processor metadata must be a JSON object")
            }
            return value
        }
        var result: [String: Any] = [:]
        for filename in ["processor_config.json", "preprocessor_config.json"] {
            let url = directory.appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: url.path) {
                result.merge(try object(Data(contentsOf: url))) { _, newest in newest }
            }
        }
        result.merge(try object(selected)) { _, newest in newest }
        for (filename, destination) in [
            ("audio_preprocessor_config.json", "feature_extractor"),
            ("feature_extractor_config.json", "feature_extractor"),
            ("video_preprocessor_config.json", "video_processor"),
        ] {
            let url = directory.appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: url.path) {
                var nested = result[destination] as? [String: Any] ?? [:]
                nested.merge(try object(Data(contentsOf: url))) { _, newest in newest }
                result[destination] = nested
            }
        }
        let root = try object(modelConfiguration)
        let thinker = root["thinker_config"] as? [String: Any] ?? root
        for name in [
            "image_token_id", "video_token_id", "audio_token_id", "vision_start_token_id",
            "vision_end_token_id", "audio_start_token_id", "audio_end_token_id",
            "position_id_per_seconds",
        ] {
            if let value = thinker[name] { result[name] = value }
        }
        if let vision = thinker["vision_config"] as? [String: Any] {
            var image = result["image_processor"] as? [String: Any] ?? result
            for (source, destination) in [
                ("patch_size", "patch_size"),
                ("spatial_merge_size", "merge_size"),
                ("temporal_patch_size", "temporal_patch_size"),
            ] {
                if let value = vision[source] { image[destination] = value }
            }
            result["image_processor"] = image
        }
        if let audio = thinker["audio_config"] as? [String: Any] {
            var features = result["feature_extractor"] as? [String: Any] ?? result
            if let bins = audio["num_mel_bins"] { features["feature_size"] = bins }
            result["feature_extractor"] = features
            if let window = audio["n_window"] { result["n_window"] = window }
        }
        return try JSONSerialization.data(withJSONObject: result)
    }
}

struct Qwen3OmniWhisperFeatures {
    let config: Qwen3OmniProcessorConfiguration
    func callAsFunction(_ samples: MLXArray, paddedSampleCount: Int? = nil) throws -> MLXArray {
        guard samples.ndim == 1, samples.size > 0 else {
            throw VLMError.processing("Omni audio must be nonempty mono samples")
        }
        let half = config.fftSize / 2
        let count =
            (max(samples.size, paddedSampleCount ?? samples.size) + config.hop - 1) / config.hop
        let paddedCount = count * config.hop
        let wave = concatenated([
            samples.asType(.float32), MLXArray.zeros([paddedCount - samples.size]),
        ])
        // torch.stft(center=True, pad_mode="reflect"), dropping its final analysis frame.
        let period = max(1, 2 * (paddedCount - 1))
        let reflected = (-half ..< paddedCount + half).map { index -> Int32 in
            let folded = ((index % period) + period) % period
            return Int32(folded < paddedCount ? folded : period - folded)
        }
        let centered = wave[MLXArray(reflected)]
        let frames = asStrided(centered, [count, config.fftSize], strides: [config.hop, 1])
        let window = MLXArray(
            (0 ..< config.fftSize).map {
                Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(config.fftSize)))
            })
        let spectrum = abs(MLXFFT.rfft(frames * window, axis: -1))
        let mel = matmul(spectrum * spectrum, Self.filters(config))
        var result = log10(maximum(mel, MLXArray(Float(1e-10))))
        result = maximum(result, result.max() - 8)
        return ((result + 4) / 4).transposed()
    }

    static func filters(_ config: Qwen3OmniProcessorConfiguration) -> MLXArray {
        func mel(_ hz: Double) -> Double {
            hz < 1000 ? hz * 3 / 200 : 15 + log(hz / 1000) * 27 / log(6.4)
        }
        func hz(_ m: Double) -> Double {
            m < 15 ? m * 200 / 3 : 1000 * exp((m - 15) * log(6.4) / 27)
        }
        let maximum = mel(Double(config.sampleRate) / 2)
        let edges = (0 ... config.melBins + 1).map {
            hz(Double($0) * maximum / Double(config.melBins + 1))
        }
        var weights: [Float] = []
        for bin in 0 ... config.fftSize / 2 {
            let f = Double(bin * config.sampleRate) / Double(config.fftSize)
            for channel in 0 ..< config.melBins {
                let triangle = max(
                    0,
                    min(
                        (f - edges[channel]) / (edges[channel + 1] - edges[channel]),
                        (edges[channel + 2] - f) / (edges[channel + 2] - edges[channel + 1])))
                weights.append(Float(triangle * 2 / (edges[channel + 2] - edges[channel])))
            }
        }
        return MLXArray(weights, [config.fftSize / 2 + 1, config.melBins])
    }
}

public struct Qwen3OmniMessageGenerator: MessageGenerator {
    public init() {}
    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        var result = Qwen3VLMessageGenerator().generate(message: message)
        var parts = contentParts(for: message, layout: .imagesThenVideosThenText)
        parts.insert(
            contentsOf: message.audios.map { _ in ["type": "audio"] }, at: max(0, parts.count - 1))
        result["content"] = parts
        return result
    }
}

public struct Qwen3OmniProcessor: UserInputProcessor {
    let config: Qwen3OmniProcessorConfiguration
    let tokenizer: any Tokenizer
    public init(_ config: Qwen3OmniProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// Keep presentation-time gaps when extracting a video's audio track, so
    /// nonzero track starts and edits remain aligned with decoded video timestamps.
    private func synchronizedAudio(_ asset: AVAsset) async throws -> MLXArray {
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            throw VLMError.processing("Synchronized Omni video has no audio track")
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: Double(config.sampleRate),
                AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true, AVLinearPCMIsNonInterleaved: false,
            ])
        reader.add(output)
        reader.startReading()
        var samples: [Float] = []
        while let buffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let count = CMBlockBufferGetDataLength(block) / MemoryLayout<Float>.size
            var chunk = [Float](repeating: 0, count: count)
            CMBlockBufferCopyDataBytes(
                block, atOffset: 0, dataLength: count * MemoryLayout<Float>.size,
                destination: &chunk)
            let time = CMSampleBufferGetPresentationTimeStamp(buffer).seconds
            guard time.isFinite else {
                throw VLMError.processing("Omni video audio has an invalid presentation timestamp")
            }
            let start = Int((time * Double(config.sampleRate)).rounded())
            if start > samples.count {
                samples += Array(repeating: 0, count: start - samples.count)
            }
            let skip = max(0, samples.count - start)
            if skip < count { samples.append(contentsOf: chunk[skip...]) }
        }
        guard reader.status == .completed, !samples.isEmpty else {
            throw reader.error
                ?? VLMError.processing("Unable to decode synchronized Omni video audio")
        }
        return MLXArray(samples)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let input = input.removingSpecialTokenLabels(using: tokenizer)
        let messages = Qwen3OmniMessageGenerator().generate(from: input)
        let raw = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)
        let imageProcessor = Qwen3VLProcessor(config.vision, tokenizer: tokenizer)
        let images = try input.images.map {
            try imageProcessor.preprocess(images: [$0.asCIImage()], processing: input.processing)
        }
        let useAudioInVideo = input.additionalContext?["use_audio_in_video"] as? Bool ?? false
        var videos: [(MLXArray, THW)] = []
        var times: [[Double]] = []
        var videoSamples: [MLXArray] = []
        var audioProcessing = input.processing.audio
        audioProcessing.sampleRate = Double(config.sampleRate)
        audioProcessing.channels = 1
        let extractor = Qwen3OmniWhisperFeatures(config: config)
        for video in input.videos {
            var frames: [CIImage] = []
            let sequence = try await MediaProcessing.asProcessedSequence(
                video,
                processing: input.processing.video, targetFPS: { _ in 2 },
                maxFrames: config.maximumVideoFrames,
                frameProcessing: { frame in
                    frames.append(try frame.image.asCIImage())
                    return frame
                })
            guard !frames.isEmpty, frames.count == sequence.timestamps.count else {
                throw VLMError.processing("Omni video yielded no timestamped frames")
            }
            let result = try imageProcessor.preprocess(images: frames, processing: input.processing)
            videos.append(result)
            let initial = useAudioInVideo ? 0 : sequence.timestamps[0].seconds
            times.append(
                (0 ..< result.1.t).map { index in
                    sequence.timestamps[
                        min(index * config.vision.temporalPatchSize, sequence.timestamps.count - 1)
                    ].seconds - initial
                })
            if useAudioInVideo {
                let asset: AVAsset
                switch video.source {
                case .url(let value): asset = AVURLAsset(url: value)
                case .avAsset(let value): asset = value
                case .frames:
                    throw VLMError.processing(
                        "Synchronized in-memory video requires precomputed interleaved LMInput audio and timestamp positions"
                    )
                }
                videoSamples.append(try await synchronizedAudio(asset))
            }
        }
        var standaloneSamples: [MLXArray] = []
        for audio in input.audios {
            standaloneSamples.append(try await audio.asMLXArray(processing: audioProcessing))
        }
        let maximumSamples = (videoSamples + standaloneSamples).map { $0.size }.max() ?? 0
        func features(_ samples: MLXArray) throws -> MLXArray {
            let validFrames = (samples.size + config.hop - 1) / config.hop
            return try extractor(samples, paddedSampleCount: maximumSamples)[0..., ..<validFrames]
        }
        let videoAudio = try videoSamples.map(features)
        let standalone = try standaloneSamples.map(features)
        let p = config.positions
        var expanded: [Int] = []
        var audioFeatures: [MLXArray] = []
        var imageIndex = 0
        var videoIndex = 0
        var audioIndex = 0
        var cursor = 0
        while cursor < raw.count {
            let token = raw[cursor]
            if token == p.imageToken {
                guard imageIndex < images.count else {
                    throw VLMError.processing(
                        "Omni prompt has an image placeholder without an image")
                }
                expanded += Array(
                    repeating: token, count: images[imageIndex].1.product / (p.merge * p.merge))
                imageIndex += 1
            } else if token == p.audioToken {
                guard audioIndex < standalone.count else {
                    throw VLMError.processing("Omni prompt has an audio placeholder without audio")
                }
                let feature = standalone[audioIndex]
                audioIndex += 1
                audioFeatures.append(feature)
                expanded += Array(
                    repeating: token,
                    count: Qwen3OmniAudio.outputLength(feature.dim(1), window: p.audioWindow))
            } else if token == p.videoToken {
                guard videoIndex < videos.count else {
                    throw VLMError.processing("Omni prompt has a video placeholder without video")
                }
                let grid = videos[videoIndex].1
                let visualCount = grid.product / (p.merge * p.merge)
                if useAudioInVideo {
                    guard expanded.last == p.visionStartToken, cursor + 1 < raw.count,
                        raw[cursor + 1] == p.visionEndToken
                    else {
                        throw VLMError.processing(
                            "Omni synchronized video must have vision boundary tokens")
                    }
                    expanded.append(p.audioStartToken)
                    let feature = videoAudio[videoIndex]
                    audioFeatures.append(feature)
                    let audioCount = Qwen3OmniAudio.outputLength(
                        feature.dim(1), window: p.audioWindow)
                    let spatial = (grid.h / p.merge) * (grid.w / p.merge)
                    var v = 0
                    var a = 0
                    while v < visualCount || a < audioCount {
                        if v < visualCount
                            && (a == audioCount
                                || times[videoIndex][v / spatial] * p.positionsPerSecond
                                    <= Double(a))
                        {
                            expanded.append(p.videoToken)
                            v += 1
                        } else {
                            expanded.append(p.audioToken)
                            a += 1
                        }
                    }
                    expanded.append(p.audioEndToken)
                } else {
                    expanded += Array(repeating: token, count: visualCount)
                }
                videoIndex += 1
            } else {
                expanded.append(token)
            }
            cursor += 1
        }
        guard imageIndex == images.count, videoIndex == videos.count, audioIndex == standalone.count
        else {
            throw VLMError.processing(
                "Omni chat template omitted one or more supplied media inputs")
        }
        let tokens = MLXArray(expanded).expandedDimensions(axis: 0)
        let mask = MLXArray.ones(tokens.shape, dtype: .int32)
        let positions = try Qwen3OmniPositions.make(
            tokens: tokens, mask: mask,
            images: images.map { $0.1 }, videos: videos.map { $0.1 },
            secondsPerVideoGrid: Array(repeating: 1, count: videos.count),
            audioLengths: audioFeatures.map { $0.dim(1) }, config: p, videoTimes: times
        ).positions
        let preparedAudio: LMInput.ProcessedAudio?
        if let maximum = audioFeatures.map({ $0.dim(1) }).max() {
            let features = MLXArray.zeros([audioFeatures.count, config.melBins, maximum])
            let audioMask = MLXArray.zeros([audioFeatures.count, maximum], dtype: .int32)
            for (index, feature) in audioFeatures.enumerated() {
                features[index, 0..., 0 ..< feature.dim(1)] = feature
                audioMask[index, 0 ..< feature.dim(1)] = MLXArray(1)
            }
            preparedAudio = .init(features: features, mask: audioMask)
        } else {
            preparedAudio = nil
        }
        return LMInput(
            text: .init(tokens: tokens, mask: mask),
            image: images.isEmpty
                ? nil
                : .init(
                    pixels: concatenated(images.map { $0.0 }), positionIds: positions,
                    frames: images.map { $0.1 }),
            video: videos.isEmpty
                ? nil
                : .init(
                    pixels: concatenated(videos.map { $0.0 }), positionIds: positions,
                    frames: videos.map { $0.1 }),
            audio: preparedAudio)
    }
}
