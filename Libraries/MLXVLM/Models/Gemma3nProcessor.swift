// Copyright © 2026 Apple Inc.
// Copyright 2025 Google LLC.
// Gemma3n feature extraction follows Google's Transformers implementation (Apache-2.0).
// See ThirdPartyNotices/Gemma3n for the license and adaptation notice.

import CoreImage
import Foundation
import MLX
import MLXLMCommon

public struct Gemma3nFeatureExtractorConfiguration: Codable, Sendable {
    public var featureSize = 128
    public var samplingRate = 16_000
    public var frameLengthMs: Float = 32
    public var hopLengthMs: Float = 10
    public var minFrequency: Float = 125
    public var maxFrequency: Float = 7600
    public var preemphasis: Float = 0.97
    public var preemphasisHtkFlavor = true
    public var fftOverdrive = true
    public var dither: Float = 0
    public var inputScaleFactor: Float = 1
    public var melFloor: Float = 1e-5
    public var perBinMean: [Float]?
    public var perBinStddev: [Float]?

    enum CodingKeys: String, CodingKey {
        case featureSize = "feature_size"
        case samplingRate = "sampling_rate"
        case frameLengthMs = "frame_length_ms"
        case hopLengthMs = "hop_length_ms"
        case minFrequency = "min_frequency"
        case maxFrequency = "max_frequency"
        case preemphasis
        case preemphasisHtkFlavor = "preemphasis_htk_flavor"
        case fftOverdrive = "fft_overdrive"
        case dither
        case inputScaleFactor = "input_scale_factor"
        case melFloor = "mel_floor"
        case perBinMean = "per_bin_mean"
        case perBinStddev = "per_bin_stddev"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        featureSize = try c.decodeIfPresent(Int.self, forKey: .featureSize) ?? featureSize
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? samplingRate
        frameLengthMs = try c.decodeIfPresent(Float.self, forKey: .frameLengthMs) ?? frameLengthMs
        hopLengthMs = try c.decodeIfPresent(Float.self, forKey: .hopLengthMs) ?? hopLengthMs
        minFrequency = try c.decodeIfPresent(Float.self, forKey: .minFrequency) ?? minFrequency
        maxFrequency = try c.decodeIfPresent(Float.self, forKey: .maxFrequency) ?? maxFrequency
        preemphasis = try c.decodeIfPresent(Float.self, forKey: .preemphasis) ?? preemphasis
        preemphasisHtkFlavor =
            try c.decodeIfPresent(Bool.self, forKey: .preemphasisHtkFlavor) ?? preemphasisHtkFlavor
        fftOverdrive = try c.decodeIfPresent(Bool.self, forKey: .fftOverdrive) ?? fftOverdrive
        dither = try c.decodeIfPresent(Float.self, forKey: .dither) ?? dither
        inputScaleFactor =
            try c.decodeIfPresent(Float.self, forKey: .inputScaleFactor) ?? inputScaleFactor
        melFloor = try c.decodeIfPresent(Float.self, forKey: .melFloor) ?? melFloor
        perBinMean = try c.decodeIfPresent([Float].self, forKey: .perBinMean)
        perBinStddev = try c.decodeIfPresent([Float].self, forKey: .perBinStddev)
        guard featureSize > 0, samplingRate > 0, frameLengthMs.isFinite, frameLengthMs > 0,
            hopLengthMs.isFinite, hopLengthMs > 0,
            minFrequency.isFinite, maxFrequency.isFinite, minFrequency >= 0,
            maxFrequency > minFrequency, maxFrequency <= Float(samplingRate) / 2,
            preemphasis.isFinite, dither.isFinite, dither >= 0,
            inputScaleFactor.isFinite, melFloor.isFinite, melFloor > 0,
            perBinMean.map({ $0.count == featureSize && $0.allSatisfy(\.isFinite) }) ?? true,
            perBinStddev.map({ $0.count == featureSize && $0.allSatisfy { $0.isFinite && $0 > 0 } })
                ?? true
        else { throw Gemma3nError.invalidConfiguration("Invalid audio feature extractor.") }
    }

    package init() {}
}

struct Gemma3nAudioFeatureExtractor: Sendable {
    let config: Gemma3nFeatureExtractorConfiguration

    func features(samples: MLXArray) throws -> MLXArray {
        guard samples.ndim == 1 else { throw Gemma3nError.invalidInput("Audio must be mono.") }
        let frame = Int((Float(config.samplingRate) * config.frameLengthMs / 1000).rounded())
        let hop = Int((Float(config.samplingRate) * config.hopLengthMs / 1000).rounded())
        guard frame > 0, hop > 0, samples.dim(0) > frame else {
            throw Gemma3nError.invalidInput("Audio must contain at least one analysis frame.")
        }
        var fftLength = 1
        while fftLength < frame { fftLength *= 2 }
        if config.fftOverdrive { fftLength *= 2 }
        var wave = samples.asType(.float32) * config.inputScaleFactor
        if config.dither > 0 { wave = wave + MLXRandom.normal(wave.shape) * config.dither }
        let frames = asStrided(
            wave, [(wave.dim(0) - frame - 1) / hop + 1, frame + 1], strides: [hop, 1])
        var analysis: MLXArray
        if config.preemphasis > 0 {
            if config.preemphasisHtkFlavor {
                analysis = concatenated(
                    [
                        frames[0..., ..<1] * (1 - config.preemphasis),
                        frames[0..., 1 ..< frame] - frames[0..., ..<(frame - 1)]
                            * config.preemphasis,
                    ], axis: -1)
            } else {
                analysis = frames[0..., 1...] - frames[0..., ..<frame] * config.preemphasis
            }
        } else {
            analysis = frames[0..., ..<frame]
        }
        let window = (0 ..< frame).map {
            Float(0.5 * (1 - cos(2 * Double.pi * Double($0) / Double(frame))))
        }
        let magnitude = abs(MLXFFT.rfft(analysis * MLXArray(window), n: fftLength, axis: -1))
        let filters = Self.melFilters(config, fftLength: fftLength)
        var result = log(
            maximum(
                matmul(
                    magnitude,
                    MLXArray(
                        filters,
                        [fftLength / 2 + 1, config.featureSize])), MLXArray(config.melFloor)))
        if let average = config.perBinMean { result = result - MLXArray(average) }
        if let deviation = config.perBinStddev { result = result / MLXArray(deviation) }
        return result
    }

    static func melFilters(_ config: Gemma3nFeatureExtractorConfiguration, fftLength: Int)
        -> [Float]
    {
        func mel(_ hertz: Double) -> Double { 2595 * log10(1 + hertz / 700) }
        let low = mel(Double(config.minFrequency))
        let high = mel(Double(config.maxFrequency))
        let edges = (0 ... config.featureSize + 1).map {
            700
                * (pow(
                    10, (low + Double($0) * (high - low) / Double(config.featureSize + 1)) / 2595)
                    - 1)
        }
        var result: [Float] = []
        for bin in 0 ... fftLength / 2 {
            let frequency = Double(bin) * Double(config.samplingRate) / Double(fftLength)
            for channel in 0 ..< config.featureSize {
                let rising = (frequency - edges[channel]) / (edges[channel + 1] - edges[channel])
                let falling =
                    (edges[channel + 2] - frequency) / (edges[channel + 2] - edges[channel + 1])
                result.append(Float(max(0, min(rising, falling))))
            }
        }
        return result
    }
}

public struct Gemma3nProcessorConfiguration: Codable, Sendable {
    public var imageSeqLength = 256
    public var audioSeqLength = 188
    public var imageSize = 768
    public var imageMean: [CGFloat] = [0.5, 0.5, 0.5]
    public var imageStd: [CGFloat] = [0.5, 0.5, 0.5]
    public var doNormalize = true
    public var doRescale = true
    public var rescaleFactor: Float = 1 / 255
    public var resample = 2
    public var imageTokenId = 262_145
    public var audioTokenId = 262_273
    public var featureExtractor = Gemma3nFeatureExtractorConfiguration()

    enum CodingKeys: String, CodingKey {
        case imageSeqLength = "image_seq_length"
        case audioSeqLength = "audio_seq_length"
        case imageSize = "image_size"
        case size
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case featureExtractor = "feature_extractor"
        case imageProcessor = "image_processor"
        case doNormalize = "do_normalize"
        case doRescale = "do_rescale"
        case rescaleFactor = "rescale_factor"
        case resample
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let image =
            try c.contains(.imageProcessor)
            ? c.nestedContainer(keyedBy: CodingKeys.self, forKey: .imageProcessor) : c
        imageSeqLength = try c.decodeIfPresent(Int.self, forKey: .imageSeqLength) ?? imageSeqLength
        audioSeqLength = try c.decodeIfPresent(Int.self, forKey: .audioSeqLength) ?? audioSeqLength
        if let size = try? image.decode([String: Int].self, forKey: .size) {
            imageSize = size["height"] ?? size["shortest_edge"] ?? imageSize
        } else {
            imageSize = try image.decodeIfPresent(Int.self, forKey: .imageSize) ?? imageSize
        }
        imageMean = try image.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? imageMean
        imageStd = try image.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? imageStd
        doNormalize = try image.decodeIfPresent(Bool.self, forKey: .doNormalize) ?? doNormalize
        doRescale = try image.decodeIfPresent(Bool.self, forKey: .doRescale) ?? doRescale
        rescaleFactor =
            try image.decodeIfPresent(Float.self, forKey: .rescaleFactor) ?? rescaleFactor
        resample = try image.decodeIfPresent(Int.self, forKey: .resample) ?? resample
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? imageTokenId
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? audioTokenId
        featureExtractor =
            try c.decodeIfPresent(
                Gemma3nFeatureExtractorConfiguration.self, forKey: .featureExtractor)
            ?? featureExtractor
        guard imageSeqLength == 256, audioSeqLength > 0, imageSize > 0,
            imageMean.count == 3, imageMean.allSatisfy(\.isFinite),
            imageStd.count == 3, imageStd.allSatisfy({ $0.isFinite && $0 > 0 }),
            rescaleFactor.isFinite, rescaleFactor > 0, [2, 3].contains(resample)
        else {
            throw Gemma3nError.invalidConfiguration("Invalid image processor or soft-token budget.")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(imageSeqLength, forKey: .imageSeqLength)
        try c.encode(audioSeqLength, forKey: .audioSeqLength)
        try c.encode(imageSize, forKey: .imageSize)
        try c.encode(imageMean, forKey: .imageMean)
        try c.encode(imageStd, forKey: .imageStd)
        try c.encode(doNormalize, forKey: .doNormalize)
        try c.encode(doRescale, forKey: .doRescale)
        try c.encode(rescaleFactor, forKey: .rescaleFactor)
        try c.encode(resample, forKey: .resample)
        try c.encode(imageTokenId, forKey: .imageTokenId)
        try c.encode(audioTokenId, forKey: .audioTokenId)
        try c.encode(featureExtractor, forKey: .featureExtractor)
    }

    package init() {}

    /// Some exports split image, feature-extractor and sequence settings into
    /// separate sidecars. Preserve all three when the factory selects one file.
    static func mergedData(directory: URL, modelConfiguration: Data, selected: Data) throws -> Data
    {
        func object(_ data: Data) throws -> [String: Any] {
            guard let result = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw Gemma3nError.invalidConfiguration("Processor metadata must be a JSON object.")
            }
            return result
        }
        func sidecar(_ filename: String) throws -> [String: Any]? {
            let url = directory.appendingPathComponent(filename)
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            do { return try object(Data(contentsOf: url)) } catch {
                throw ProcessorConfigError(filename: filename, underlying: error)
            }
        }
        let model = try object(modelConfiguration)
        var combined = try sidecar("processor_config.json") ?? [:]
        let selected = try object(selected)
        combined.merge(selected) { _, value in value }
        if combined["image_processor"] == nil {
            combined["image_processor"] = selected
        }
        if let audio = try sidecar("audio_preprocessor_config.json")
            ?? sidecar("feature_extractor_config.json")
        {
            combined["feature_extractor"] = audio
        }
        var extractor = selected
        extractor.merge(combined["feature_extractor"] as? [String: Any] ?? [:]) { _, value in value
        }
        if extractor["feature_size"] == nil,
            let audio = model["audio_config"] as? [String: Any],
            let size = audio["input_feat_size"]
        {
            extractor["feature_size"] = size
        }
        combined["feature_extractor"] = extractor
        for (modelKey, processorKey) in [
            "image_token_id": "image_token_id", "audio_token_id": "audio_token_id",
            "vision_soft_tokens_per_image": "image_seq_length",
            "audio_soft_tokens_per_image": "audio_seq_length",
        ] {
            if let value = model[modelKey] { combined[processorKey] = value }
        }
        return try JSONSerialization.data(withJSONObject: combined)
    }
}

public struct Gemma3nMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        var dictionary = Qwen2VLMessageGenerator().generate(message: message)
        var parts = contentParts(for: message, layout: .imagesThenText)
        let text = parts.popLast()
        parts.append(contentsOf: message.audios.map { _ in ["type": "audio"] })
        if let text { parts.append(text) }
        dictionary["content"] = parts
        return dictionary
    }
}

/// Pillow's separable, antialiased triangle filter for the published bilinear image processor.
func gemma3nImagePixels(_ image: MLXArray, config: Gemma3nProcessorConfiguration) -> MLXArray {
    var pixels = image
    if config.resample == 2, image.dim(1) != config.imageSize || image.dim(2) != config.imageSize {
        func weights(_ input: Int, _ output: Int) -> MLXArray {
            let scale = Float(input) / Float(output)
            let support = max(scale, 1)
            var values = [Float](repeating: 0, count: input * output)
            for destination in 0 ..< output {
                let center = (Float(destination) + 0.5) * scale
                let start = max(0, Int(ceil(center - support - 0.5)))
                let end = min(input, Int(ceil(center + support - 0.5)))
                var total: Float = 0
                for source in start ..< end {
                    let value = max(0, 1 - abs((Float(source) + 0.5 - center) / support))
                    values[destination * input + source] = value
                    total += value
                }
                for source in start ..< end { values[destination * input + source] /= total }
            }
            return MLXArray(values, [output, input])
        }
        let height = config.imageSize
        let width = config.imageSize
        let channels = image.dim(3)
        let horizontal = matmul(
            weights(image.dim(2), width),
            image[0].transposed(1, 0, 2).reshaped(image.dim(2), -1)
        )
        .reshaped(width, image.dim(1), channels).transposed(1, 0, 2)
        let roundedHorizontal = MLX.round(horizontal * Float(255)) / Float(255)
        pixels = matmul(weights(image.dim(1), height), roundedHorizontal.reshaped(image.dim(1), -1))
            .reshaped(1, height, width, channels)
        pixels = MLX.round(pixels * Float(255)) / Float(255)
    }
    pixels = pixels * (config.doRescale ? 255 * config.rescaleFactor : 255)
    if config.doNormalize {
        pixels =
            (pixels - MLXArray(config.imageMean.map(Float.init)))
            / MLXArray(config.imageStd.map(Float.init))
    }
    return pixels
}

public struct Gemma3nProcessor: UserInputProcessor {
    private let config: Gemma3nProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma3nProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else {
            throw Gemma3nError.invalidInput("Video inputs require sampled images.")
        }
        let messages = Gemma3nMessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))
        let rawTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools, additionalContext: input.additionalContext)
        let imageToken = tokenizer.singleTokenID(for: "<image_soft_token>") ?? config.imageTokenId
        let audioToken = tokenizer.singleTokenID(for: "<audio_soft_token>") ?? config.audioTokenId
        func block(_ kind: String, token: Int, count: Int) throws -> [Int] {
            guard let begin = tokenizer.singleTokenID(for: "<start_of_\(kind)>"),
                let end = tokenizer.singleTokenID(for: "<end_of_\(kind)>")
            else {
                throw Gemma3nError.invalidInput("Tokenizer is missing \(kind) boundary tokens.")
            }
            let whitespace = tokenizer.encode(text: "\n\n", addSpecialTokens: false)
            return whitespace + [begin] + Array(repeating: token, count: count) + [end] + whitespace
        }
        var tokens: [Int] = []
        var images = 0
        var audios = 0
        for token in rawTokens {
            if token == imageToken, !input.images.isEmpty {
                images += 1
                tokens += try block("image", token: imageToken, count: config.imageSeqLength)
            } else if token == audioToken, !input.audios.isEmpty {
                audios += 1
                tokens += try block("audio", token: audioToken, count: config.audioSeqLength)
            } else {
                tokens.append(token)
            }
        }
        guard images == input.images.count else {
            throw Gemma3nError.placeholderMismatch(
                kind: "image", placeholders: images, features: input.images.count)
        }
        guard audios == input.audios.count else {
            throw Gemma3nError.placeholderMismatch(
                kind: "audio", placeholders: audios, features: input.audios.count)
        }
        var pixels: MLXArray?
        if !input.images.isEmpty {
            let size = CGSize(width: config.imageSize, height: config.imageSize)
            pixels = try concatenated(
                input.images.map { source in
                    let image = MediaProcessing.inSRGBToneCurveSpace(try source.asCIImage())
                    let original =
                        config.resample == 2
                        ? image : MediaProcessing.resampleBicubic(image, to: size)
                    return gemma3nImagePixels(
                        MediaProcessing.asMLXArray(original).transposed(0, 2, 3, 1), config: config)
                }
            ).transposed(0, 3, 1, 2)
        }
        var audio: LMInput.ProcessedAudio?
        if !input.audios.isEmpty {
            var waveforms: [MLXArray] = []
            for source in input.audios {
                var processing = UserInput.AudioProcessing()
                processing.sampleRate = Double(config.featureExtractor.samplingRate)
                let waveform = try await source.asMLXArray(processing: processing)
                guard waveform.ndim == 1, waveform.dim(0) > 0 else {
                    throw Gemma3nError.invalidInput("Audio must contain mono samples.")
                }
                waveforms.append(waveform[..<min(waveform.dim(0), 480_000)])
            }
            let longest = waveforms.map { $0.dim(0) }.max() ?? 0
            let paddedLength = ((longest + 127) / 128) * 128
            let extractor = Gemma3nAudioFeatureExtractor(config: config.featureExtractor)
            let features = try waveforms.map { waveform in
                try extractor.features(
                    samples: padded(
                        waveform,
                        widths: [.init((0, paddedLength - waveform.dim(0)))]))
            }
            let count = features[0].dim(0)
            let hop = Int(
                (Float(config.featureExtractor.samplingRate) * config.featureExtractor.hopLengthMs
                    / 1000).rounded())
            let masks = waveforms.map { MLXArray(0 ..< count) * hop .< $0.dim(0) }
            audio = .init(features: stacked(features), mask: stacked(masks))
        }
        let text = LMInput.Text(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
        let result = LMInput(text: text, image: pixels.map { .init(pixels: $0) }, audio: audio)
        try MLX.checkedEval(
            [text.tokens] + (pixels.map { [$0] } ?? [])
                + (audio.map { [$0.features, $0.mask!] } ?? []))
        return result
    }
}
