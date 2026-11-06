// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm MiniCPM processors at 4693cd0 (MIT).
import CoreImage
import Foundation
import MLX
import MLXLMCommon

public struct MiniCPMProcessorConfiguration: Codable, Sendable {
    public var modelType = "minicpmv4_6", patchSize = 14, scale = 448, maxSlices = 9,
        queryCount = 64
    public var sliceMode = true, useImageID = true, downsampleMode = "16x", audioPoolStep = 5,
        melBins = 80
    public var maxFrames = 128, stackFrames = 1
    public var mean: [Float] = [0.5, 0.5, 0.5], std: [Float] = [0.5, 0.5, 0.5]
    public init() {}
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case patchSize = "patch_size"
        case scale = "scale_resolution"
        case maxSlices = "max_slice_nums"
        case queryCount = "image_feature_size"
        case queryNumber = "query_num"
        case sliceMode = "slice_mode"
        case useImageID = "use_image_id"
        case downsampleMode = "downsample_mode"
        case audioPoolStep = "audio_pool_step"
        case melBins = "feature_size"
        case maxFrames = "max_num_frames"
        case stackFrames = "stack_frames"
        case mean = "image_mean"
        case std = "image_std"
        case normMean = "norm_mean"
        case normStd = "norm_std"
        case image = "image_processor"
        case video = "video_processor"
        case slice = "slice_config"
        case vision = "vision_config"
        case audio = "audio_config"
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let nested = try c.decodeIfPresent(Self.self, forKey: .image) { self = nested }
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? patchSize
        scale = try c.decodeIfPresent(Int.self, forKey: .scale) ?? scale
        maxSlices = try c.decodeIfPresent(Int.self, forKey: .maxSlices) ?? maxSlices
        queryCount =
            try c.decodeIfPresent(Int.self, forKey: .queryCount) ?? c.decodeIfPresent(
                Int.self, forKey: .queryNumber) ?? queryCount
        sliceMode = try c.decodeIfPresent(Bool.self, forKey: .sliceMode) ?? sliceMode
        useImageID = try c.decodeIfPresent(Bool.self, forKey: .useImageID) ?? useImageID
        downsampleMode =
            try c.decodeIfPresent(String.self, forKey: .downsampleMode) ?? downsampleMode
        audioPoolStep = try c.decodeIfPresent(Int.self, forKey: .audioPoolStep) ?? audioPoolStep
        melBins = try c.decodeIfPresent(Int.self, forKey: .melBins) ?? melBins
        maxFrames = try c.decodeIfPresent(Int.self, forKey: .maxFrames) ?? maxFrames
        stackFrames = try c.decodeIfPresent(Int.self, forKey: .stackFrames) ?? stackFrames
        mean =
            try c.decodeIfPresent([Float].self, forKey: .mean) ?? c.decodeIfPresent(
                [Float].self, forKey: .normMean) ?? mean
        std =
            try c.decodeIfPresent([Float].self, forKey: .std) ?? c.decodeIfPresent(
                [Float].self, forKey: .normStd) ?? std
        if let slice = try c.decodeIfPresent(MiniCPMConfiguration.Slice.self, forKey: .slice) {
            scale = slice.scaleResolution ?? scale
            maxSlices = slice.maxSliceNums ?? maxSlices
        }
        if let vision = try c.decodeIfPresent(MiniCPMVisionConfiguration.self, forKey: .vision) {
            patchSize = vision.patchSize
        }
        if let audio = try c.decodeIfPresent(MiniCPMAudioConfiguration.self, forKey: .audio) {
            melBins = audio.melBins
        }
        if let video = try c.decodeIfPresent(Self.self, forKey: .video) {
            maxFrames = video.maxFrames
            stackFrames = video.stackFrames
        }
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(patchSize, forKey: .patchSize)
        try c.encode(scale, forKey: .scale)
        try c.encode(maxSlices, forKey: .maxSlices)
        try c.encode(queryCount, forKey: .queryCount)
        try c.encode(sliceMode, forKey: .sliceMode)
        try c.encode(useImageID, forKey: .useImageID)
        try c.encode(downsampleMode, forKey: .downsampleMode)
        try c.encode(audioPoolStep, forKey: .audioPoolStep)
        try c.encode(melBins, forKey: .melBins)
        try c.encode(maxFrames, forKey: .maxFrames)
        try c.encode(stackFrames, forKey: .stackFrames)
        try c.encode(mean, forKey: .mean)
        try c.encode(std, forKey: .std)
    }
}
public struct MiniCPMMessageGenerator: MessageGenerator {
    public init() {}
    public func generate(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .text(let text):
            generate(messages: [
                .user(text, images: input.images, videos: input.videos, audios: input.audios)
            ])
        case .chat(let messages): generate(messages: messages)
        case .messages(let messages): messages
        }
    }
    public func generate(messages: [Chat.Message]) -> [Message] {
        messages.map { message in
            func missing(_ marker: String, _ attached: Int) -> String {
                let existing = message.content.components(separatedBy: marker).count - 1
                return String(repeating: marker, count: max(0, attached - existing))
            }
            var prefix = missing("<image>", message.images.count)
            prefix += missing("<video>", message.videos.count)
            prefix += missing("<audio>", message.audios.count)
            var result: Message = [
                "role": message.role.rawValue, "content": prefix + message.content,
            ]
            addReasoning(to: &result, for: message)
            addToolMetadata(to: &result, for: message)
            return result
        }
    }
}

package enum MiniCPMImageProcessing {
    static func divide(_ length: Double, _ unit: Int) -> Int {
        max(Int((length / Double(unit)).rounded(.toNearestOrEven)) * unit, unit)
    }
    static func bestSize(width: Int, height: Int, scale: Int, patch: Int, upscale: Bool, omni: Bool)
        -> (Int, Int)
    {
        var w = width
        var h = height
        if w * h > scale * scale || upscale {
            let ratio = Double(w) / Double(h)
            h = Int(Double(scale) / sqrt(max(ratio, 1e-6)))
            w = Int(Double(h) * ratio)
        }
        let unit = patch * (omni ? 1 : 4)
        return (divide(Double(w), unit), divide(Double(h), unit))
    }
    static func sliceGrid(width: Int, height: Int, scale: Int, maxSlices: Int) -> (Int, Int)? {
        let multiple = min(Int(ceil(Double(width * height) / Double(scale * scale))), maxSlices)
        guard multiple > 1 else { return nil }
        var best = (1, 1)
        var error = Double.infinity
        for count in (multiple - 1) ... (multiple + 1) where count > 1 && count <= maxSlices {
            for columns in 1 ... count where count % columns == 0 {
                let rows = count / columns
                let delta = abs(
                    log(Double(width) / Double(height)) - log(Double(columns) / Double(rows)))
                if delta < error {
                    error = delta
                    best = (columns, rows)
                }
            }
        }
        return best
    }
    /// Pillow's antialiased cubic kernel (a = -0.5), with half-pixel centers.
    static func weights(input: Int, output: Int) -> MLXArray {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        func cubic(_ value: Double) -> Double {
            let x = abs(value)
            if x < 1 { return (1.5 * x - 2.5) * x * x + 1 }
            if x < 2 { return ((-0.5 * x + 2.5) * x - 4) * x + 2 }
            return 0
        }
        var values = [Float](repeating: 0, count: input * output)
        for i in 0 ..< output {
            let center = (Double(i) + 0.5) * scale
            let first = max(Int(center - support + 0.5), 0)
            let last = min(Int(center + support + 0.5), input)
            let taps = (first ..< last).map { cubic((Double($0) + 0.5 - center) / filterScale) }
            let total = taps.reduce(0, +)
            for (j, tap) in taps.enumerated() {
                values[i * input + first + j] = Float(
                    (tap / total * 4_194_304).rounded() / 4_194_304)
            }
        }
        return MLXArray(values, [output, input])
    }
    /// Input and output are uint8-valued Float32, HWC. Each separable pass rounds like Pillow RGB.
    static func resize(_ pixels: MLXArray, height: Int, width: Int) -> MLXArray {
        if pixels.dim(0) == height && pixels.dim(1) == width { return pixels }
        let x = pixels.transposed(2, 0, 1)
        let horizontal = clip(
            floor(matmul(x, weights(input: pixels.dim(1), output: width).transposed()) + 0.5),
            min: 0, max: 255)
        return clip(
            floor(matmul(weights(input: pixels.dim(0), output: height), horizontal) + 0.5), min: 0,
            max: 255
        ).transposed(1, 2, 0)
    }
    struct Crops { var pixels: [MLXArray], grids: [THW], layout: (Int, Int)? }
    static func crops(_ pixels: MLXArray, config: MiniCPMProcessorConfiguration, omni: Bool)
        -> Crops
    {
        let height = pixels.dim(0)
        let width = pixels.dim(1)
        let p = config.patchSize
        let grid =
            omni || !config.sliceMode
            ? nil
            : sliceGrid(
                width: width, height: height, scale: config.scale, maxSlices: config.maxSlices)
        let (w, h) = bestSize(
            width: width, height: height, scale: config.scale, patch: p, upscale: grid == nil,
            omni: omni)
        var crops = [resize(pixels, height: h, width: w)]
        if let (columns, rows) = grid {
            let refinedW = divide(Double(width), columns)
            let refinedH = divide(Double(height), rows)
            let (cellW, cellH) = bestSize(
                width: refinedW / columns, height: refinedH / rows, scale: config.scale, patch: p,
                upscale: true, omni: false)
            let detailed = resize(pixels, height: cellH * rows, width: cellW * columns)
            for row in 0 ..< rows {
                for column in 0 ..< columns {
                    crops.append(
                        detailed[
                            (row * cellH) ..< ((row + 1) * cellH),
                            (column * cellW) ..< ((column + 1) * cellW), 0...])
                }
            }
        }
        let grids = crops.map { THW(1, $0.dim(0) / p, $0.dim(1) / p) }
        let patches = crops.map { image in
            let x = (image / 255 - MLXArray(config.mean)) / MLXArray(config.std)
            return x.reshaped(x.dim(0) / p, p, x.dim(1) / p, p, 3).transposed(0, 2, 1, 3, 4)
                .reshaped(-1, p * p * 3)
        }
        return .init(pixels: patches, grids: grids, layout: grid)
    }
}

public struct MiniCPMProcessor: UserInputProcessor {
    private var config: MiniCPMProcessorConfiguration
    private let tokenizer: any Tokenizer
    public init(_ config: MiniCPMProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }
    package init(
        _ config: MiniCPMProcessorConfiguration, tokenizer: any Tokenizer, modelType: String
    ) {
        self.config = config
        self.config.modelType = modelType
        self.tokenizer = tokenizer
    }
    private var omni: Bool { config.modelType == "minicpmo" }
    private func marker(_ text: String) throws -> Int {
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        guard ids.count == 1 else {
            throw VLMError.processing("MiniCPM tokenizer is missing required special token \(text)")
        }
        return ids[0]
    }
    private func pixels(_ image: UserInput.Image) throws -> MLXArray {
        let ci = try image.asCIImage().settingProperties([
            CIImageOption.applyOrientationProperty: true
        ])
        guard !ci.extent.isEmpty, !ci.extent.isInfinite else {
            throw VLMError.processing("Invalid MiniCPM image extent")
        }
        return floor(
            clip(
                MediaProcessing.asMLXArray(MediaProcessing.inSRGBToneCurveSpace(ci))[0].transposed(
                    1, 2, 0) * 255, min: 0, max: 255))
    }
    private func placeholder(
        _ crops: MiniCPMImageProcessing.Crops, index: Int, video: Bool, includeID: Bool = true
    ) throws -> [Int] {
        let sentinel = video ? MiniCPM.videoToken : MiniCPM.imageToken
        var ids: [Int] = []
        if !omni && config.useImageID && includeID {
            ids += tokenizer.encode(text: "<image_id>\(index)</image_id>", addSpecialTokens: false)
        }
        let factor = config.downsampleMode == "4x" ? 2 : 4
        for (crop, grid) in crops.grids.enumerated() {
            if crop > 1, let (columns, _) = crops.layout, (crop - 1) % columns == 0 {
                ids += tokenizer.encode(text: "\n", addSpecialTokens: false)
            }
            ids.append(try marker(crop == 0 ? "<image>" : "<slice>"))
            ids += Array(
                repeating: sentinel,
                count: omni ? config.queryCount : grid.h * grid.w / (factor * factor))
            ids.append(try marker(crop == 0 ? "</image>" : "</slice>"))
        }
        return ids
    }
    public func prepare(input: UserInput) async throws -> LMInput {
        guard config.patchSize > 0, config.scale > 0, config.maxSlices > 0, config.queryCount > 0,
            config.audioPoolStep > 0, config.maxFrames > 0, config.stackFrames > 0,
            config.mean.count == 3, config.std.count == 3, config.std.allSatisfy({ $0 > 0 }),
            ["4x", "16x"].contains(config.downsampleMode)
        else { throw VLMError.processing("Invalid MiniCPM processor geometry") }
        guard omni || input.audios.isEmpty else {
            throw VLMError.processing("MiniCPM-V checkpoints do not contain an audio encoder")
        }
        let messages = MiniCPMMessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))
        let template = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        if input.images.isEmpty && input.videos.isEmpty && input.audios.isEmpty {
            return LMInput(tokens: MLXArray(template)[.newAxis, 0...])
        }
        var imagePatches: [MLXArray] = []
        var imageGrids: [THW] = []
        var imageBlocks: [[Int]] = []
        for (index, image) in input.images.enumerated() {
            let crops = MiniCPMImageProcessing.crops(try pixels(image), config: config, omni: omni)
            imagePatches += crops.pixels
            imageGrids += crops.grids
            imageBlocks.append(try placeholder(crops, index: index, video: false))
        }
        var videoPatches: [MLXArray] = []
        var videoGrids: [THW] = []
        var videoBlocks: [[Int]] = []
        for (index, video) in input.videos.enumerated() {
            let decoded = try await MediaProcessing.asProcessedSequence(
                video, processing: input.processing.video,
                targetFPS: { _ in 1 }, maxFrames: config.maxFrames)
            guard !decoded.frames.isEmpty else {
                throw VLMError.processing("MiniCPM could not decode any video frames")
            }
            var frames = decoded.frames.map {
                floor(clip($0[0].transposed(1, 2, 0) * 255, min: 0, max: 255))
            }
            if config.stackFrames > 1 {
                var units: [MLXArray] = []
                for start in stride(from: 0, to: frames.count, by: config.stackFrames) {
                    let chunk = Array(
                        frames[start ..< min(start + config.stackFrames, frames.count)])
                    let width = chunk[0].dim(1)
                    let height = chunk[0].dim(0)
                    let count = chunk.count
                    let rows =
                        count == 4
                        ? 2
                        : count == 3 && width < height ? 3 : count == 2 && width < height ? 2 : 1
                    let columns = (count + rows - 1) / rows
                    let canvas = MLXArray.zeros([
                        rows * height + (rows - 1) * 6, columns * width + (columns - 1) * 6, 3,
                    ])
                    for (i, frame) in chunk.enumerated() {
                        canvas[
                            (i / columns * (height + 6)) ..< (i / columns * (height + 6) + height),
                            (i % columns * (width + 6)) ..< (i % columns * (width + 6) + width),
                            0...] = MiniCPMImageProcessing.resize(
                                frame, height: height, width: width)
                    }
                    units.append(canvas)
                }
                frames = units
            }
            var block: [Int] = []
            if !omni && config.modelType != "minicpmv4_7" && config.useImageID {
                block += tokenizer.encode(
                    text: "<image_id>\(index)</image_id>", addSpecialTokens: false)
            }
            for frame in frames {
                let crops = MiniCPMImageProcessing.crops(frame, config: config, omni: omni)
                videoPatches += crops.pixels
                videoGrids += crops.grids
                block += try placeholder(crops, index: index, video: true, includeID: false)
            }
            videoBlocks.append(block)
        }
        var waves: [MLXArray] = []
        var waveGroups: [Range<Int>] = []
        var waveSampleCounts: [Int] = []
        for audio in input.audios {
            var processing = input.processing.audio
            processing.sampleRate = 16000
            processing.channels = 1
            let wave = try await audio.asMLXArray(processing: processing)
            let mono = wave.squeezed()
            guard mono.ndim == 1, mono.size > 200 else {
                throw VLMError.processing(
                    "MiniCPM-o audio requires at least 13 ms of mono 16 kHz samples")
            }
            let first = waves.count
            for start in stride(from: 0, to: mono.size, by: 480000) {
                let segment = mono[start ..< min(start + 480000, mono.size)]
                waveSampleCounts.append(segment.size)
                waves.append(
                    segment.size > 200
                        ? segment : concatenated([segment, MLXArray.zeros([201 - segment.size])]))
            }
            waveGroups.append(first ..< waves.count)
        }
        let longest = max(
            ((waves.map { $0.size }.max() ?? 0) + 159) / 160 * 160, config.audioPoolStep * 320)
        var audioFeatures: [MLXArray] = []
        var audioLengths: [Int] = []
        var audioBlocks: [[Int]] = []
        for (index, wave) in waves.enumerated() {
            let features = try MiniCPMAudioFeatures.extract(
                wave, paddedSamples: longest, melBins: config.melBins)
            let frames = (waveSampleCounts[index] + 159) / 160
            let count = max(1, ((frames + 1) / 2) / config.audioPoolStep)
            audioFeatures.append(features)
            audioLengths.append(frames)
            audioBlocks.append(
                [try marker("<|audio_start|>")] + Array(repeating: MiniCPM.audioToken, count: count)
                    + [try marker("<|audio_end|>")])
        }
        audioBlocks = waveGroups.map { range in range.flatMap { audioBlocks[$0] } }
        var decoded = tokenizer.decode(tokenIds: template, skipSpecialTokens: false)
        if config.modelType == "minicpmv4_6" {
            decoded = decoded.replacingOccurrences(of: "<think>\n\n</think>\n\n", with: "")
        }
        let expression = try NSRegularExpression(
            pattern: #"<image>\./</image>|<image>|<video>|<audio>\./</audio>|<audio>"#)
        let matches = expression.matches(
            in: decoded, range: NSRange(decoded.startIndex..., in: decoded))
        var ids: [Int] = []
        var previous = decoded.startIndex
        var imageIndex = 0
        var videoIndex = 0
        var audioIndex = 0
        for match in matches {
            guard let range = Range(match.range, in: decoded) else { continue }
            ids += tokenizer.encode(
                text: String(decoded[previous ..< range.lowerBound]), addSpecialTokens: false)
            let value = String(decoded[range])
            if value.hasPrefix("<image>") {
                guard imageIndex < imageBlocks.count else {
                    throw VLMError.processing("MiniCPM image placeholder count mismatch")
                }
                ids += imageBlocks[imageIndex]
                imageIndex += 1
            } else if value == "<video>" {
                guard videoIndex < videoBlocks.count else {
                    throw VLMError.processing("MiniCPM video placeholder count mismatch")
                }
                ids += videoBlocks[videoIndex]
                videoIndex += 1
            } else {
                guard audioIndex < audioBlocks.count else {
                    throw VLMError.processing("MiniCPM audio placeholder count mismatch")
                }
                ids += audioBlocks[audioIndex]
                audioIndex += 1
            }
            previous = range.upperBound
        }
        ids += tokenizer.encode(text: String(decoded[previous...]), addSpecialTokens: false)
        if omni {
            let listen = tokenizer.encode(text: "<|listen|>", addSpecialTokens: false)
            if listen.count == 1 { ids.removeAll { $0 == listen[0] } }
        }
        guard imageIndex == imageBlocks.count, videoIndex == videoBlocks.count,
            audioIndex == audioBlocks.count
        else {
            throw VLMError.processing("MiniCPM attached media is missing its prompt placeholders")
        }
        func padded(_ arrays: [MLXArray]) -> MLXArray? {
            guard let longest = arrays.map({ $0.dim(0) }).max() else { return nil }
            return stacked(
                arrays.map {
                    $0.dim(0) == longest
                        ? $0
                        : concatenated([
                            $0, MLXArray.zeros([longest - $0.dim(0), $0.dim(1)], dtype: $0.dtype),
                        ])
                })
        }
        var positions: MLXArray?
        if config.modelType == "minicpmv4_7" {
            positions = try MiniCPMCanvas.positions(
                tokens: ids, lengths: [ids.count], width: ids.count, images: imageGrids,
                videos: videoGrids,
                factor: config.downsampleMode == "4x" ? 2 : 4,
                markers: .init(
                    imageStart: try marker("<image>"), imageEnd: try marker("</image>"),
                    sliceStart: try marker("<slice>"),
                    sliceEnd: try marker("</slice>"), newline: try marker("\n"))
            ).0
        }
        let image = padded(imagePatches).map {
            LMInput.ProcessedImage(pixels: $0, positionIds: positions, frames: imageGrids)
        }
        let video = padded(videoPatches).map {
            LMInput.ProcessedVideo(pixels: $0, positionIds: positions, frames: videoGrids)
        }
        let audio =
            audioFeatures.isEmpty
            ? nil
            : LMInput.ProcessedAudio(
                features: stacked(audioFeatures),
                mask: MLXArray(0 ..< audioFeatures[0].dim(0))[.newAxis, 0...]
                    .< MLXArray(audioLengths)[0..., .newAxis])
        return LMInput(
            text: .init(tokens: MLXArray(ids)[.newAxis, 0...]), image: image, video: video,
            audio: audio)
    }
}
public struct MiniCPMProcessorLoadingResolver: VLMProcessorLoadingResolver {
    public init() {}
    private func type(_ model: String) -> String? {
        switch model {
        case "minicpmo": "MiniCPMOProcessor"
        case "minicpmv4_6": "MiniCPMVProcessor"
        case "minicpmv4_7": "MiniCPMV4_7Processor"
        default: nil
        }
    }
    public func processorType(
        for context: VLMProcessorLoadingContext, declaredProcessorType: String?
    ) throws -> String? { type(context.modelType) }
    public func fallbackProcessorConfiguration(for context: VLMProcessorLoadingContext) throws
        -> VLMProcessorConfiguration?
    {
        type(context.modelType).map { .init(data: context.configurationData, processorType: $0) }
    }
}
