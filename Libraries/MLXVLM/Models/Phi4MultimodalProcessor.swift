// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm phi4mm at 4693cd0 (MIT).

import CoreImage
import Foundation
import MLX
import MLXLMCommon

public struct Phi4MultimodalProcessorConfiguration: Codable, Sendable {
    public var patchSize = 14
    public var minimumPatches = 256
    public var maximumPatches = 3600
    public var imageMean: [Float] = [0.5, 0.5, 0.5]
    public var imageStd: [Float] = [0.5, 0.5, 0.5]
    public var audioReduction = 8
    public var imageToken = -200
    public var audioToken = 200011
    public init() {}
    enum CodingKeys: String, CodingKey {
        case patchSize = "patch_size"
        case minimumPatches = "min_num_patches"
        case maximumPatches = "max_num_patches"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case vision = "vision_config"
        case audio = "audio_processor"
        case audioReduction = "audio_compression_rate"
        case imageToken = "image_token_index"
        case audioToken = "audio_token_index"
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let vision = try c.decodeIfPresent(Phi4MultimodalVisionConfiguration.self, forKey: .vision)
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? vision?.patchSize ?? 14
        minimumPatches = try c.decodeIfPresent(Int.self, forKey: .minimumPatches) ?? 256
        maximumPatches = try c.decodeIfPresent(Int.self, forKey: .maximumPatches) ?? 3600
        imageMean = try c.decodeIfPresent([Float].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([Float].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        audioReduction =
            try c.decodeIfPresent(Int.self, forKey: .audioReduction) ?? c.decodeIfPresent(
                Phi4MultimodalConfiguration.AudioProcessor.self, forKey: .audio)?.config.reduction
            ?? 8
        imageToken = try c.decodeIfPresent(Int.self, forKey: .imageToken) ?? -200
        audioToken = try c.decodeIfPresent(Int.self, forKey: .audioToken) ?? 200011
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(patchSize, forKey: .patchSize)
        try c.encode(minimumPatches, forKey: .minimumPatches)
        try c.encode(maximumPatches, forKey: .maximumPatches)
        try c.encode(imageMean, forKey: .imageMean)
        try c.encode(imageStd, forKey: .imageStd)
        try c.encode(audioReduction, forKey: .audioReduction)
        try c.encode(imageToken, forKey: .imageToken)
        try c.encode(audioToken, forKey: .audioToken)
    }
}

public struct Phi4MultimodalMessageGenerator: MessageGenerator {
    public init() {}
    public func generate(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .text(let text):
            generate(messages: [.user(text, images: input.images, audios: input.audios)])
        case .chat(let messages): generate(messages: messages)
        case .messages(let messages): messages
        }
    }
    public func generate(messages: [Chat.Message]) -> [Message] {
        var image = 0
        var audio = 0
        return messages.map { message in
            var content = ""
            for _ in message.images {
                image += 1
                content += "<|image_\(image)|>\n"
            }
            for _ in message.audios {
                audio += 1
                content += "<|audio_\(audio)|>\n"
            }
            var result: Message = [
                "role": message.role.rawValue, "content": content + message.content,
            ]
            addToolMetadata(to: &result, for: message)
            return result
        }
    }
}

package enum Phi4MultimodalAudioFeatures {
    static func melFilters() -> MLXArray {
        let bins = 257
        let filters = 80
        let high = 1127 * log(1 + 7690.0 / 700)
        let step = high / Double(filters + 1)
        var matrix = [Float](repeating: 0, count: bins * filters)
        for m in 0 ..< filters {
            let left = Double(m) * step
            let center = Double(m + 1) * step
            let right = Double(m + 2) * step
            for bin in 1 ..< Int(7690.0 * 512 / 16000 + 0.5) {
                let mel = 1127 * log(1 + Double(bin) * 16000 / (512 * 700))
                if mel > left && mel < right {
                    matrix[bin * filters + m] = Float(1 - abs(center - mel) / step)
                }
            }
        }
        return MLXArray(matrix, [bins, filters])
    }
    static func extract(_ samples: MLXArray) throws -> MLXArray {
        let wave = samples.squeezed()
        guard wave.ndim == 1, wave.size >= 400 else {
            throw VLMError.processing("Phi-4 audio needs at least 25 ms of mono 16 kHz samples")
        }
        let count = (wave.size - 400) / 160 + 1
        let starts = MLXArray(0 ..< count)[0..., .newAxis] * 160
        let offsets = MLXArray(0 ..< 400)[.newAxis, 0...]
        let frames = wave[starts + offsets]
        let previous = concatenated([frames[0..., 0 ..< 1], frames[0..., ..<399]], axis: 1)
        let window = MLXArray(
            (0 ..< 400).map { Float(0.54 - 0.46 * cos(2 * Double.pi * Double($0) / 399)) })
        let emphasized = (frames - 0.97 * previous) * 32768 * window
        let magnitude = abs(MLXFFT.rfft(emphasized, n: 512))
        return log(maximum(matmul(magnitude * magnitude, melFilters()), MLXArray(1)))
    }
}

/// Pillow RGB bilinear uses antialiased triangle coefficients and rounds each pass to bytes.
/// See ThirdPartyNotices/Phi4Multimodal-Pillow-LICENSE.txt for Pillow's MIT-CMU notice.
package enum Phi4MultimodalPixelResampling {
    private static let precision = 22

    private static func coefficients(input: Int, output: Int) -> (MLXArray, MLXArray) {
        let scale = Double(input) / Double(output)
        let support = max(scale, 1)
        var rows: [(indices: [Int32], weights: [Int32])] = []
        for destination in 0 ..< output {
            let center = (Double(destination) + 0.5) * scale
            let first = max(0, Int(center - support + 0.5))
            let end = min(input, Int(center + support + 0.5))
            let weights = (first ..< end).map { source in
                max(0, 1 - abs((Double(source) + 0.5 - center) / support))
            }
            let total = weights.reduce(0, +)
            rows.append(
                (
                    (first ..< end).map(Int32.init),
                    weights.map { Int32(floor($0 / total * Double(1 << precision) + 0.5)) }
                ))
        }
        let taps = rows.map { $0.indices.count }.max()!
        let indices = rows.flatMap {
            $0.indices + Array(repeating: 0, count: taps - $0.indices.count)
        }
        let weights = rows.flatMap {
            $0.weights + Array(repeating: 0, count: taps - $0.weights.count)
        }
        return (MLXArray(indices, [output, taps]), MLXArray(weights, [output, taps]))
    }

    private static func pass(_ pixels: MLXArray, axis: Int, count: Int) -> MLXArray {
        if pixels.dim(axis) == count { return pixels }
        let (indices, weights) = coefficients(input: pixels.dim(axis), output: count)
        var parts: [MLXArray] = []
        // Bound gather intermediates for large source photos.
        for start in stride(from: 0, to: count, by: 64) {
            let end = min(start + 64, count)
            var shape = Array(repeating: 1, count: pixels.ndim + 1)
            shape[axis] = end - start
            shape[axis + 1] = weights.dim(1)
            let gathered = pixels.take(indices[start ..< end], axis: axis)
            let sum = (gathered * weights[start ..< end].reshaped(shape)).sum(axis: axis + 1)
            let rounded = clip((sum + (1 << (precision - 1))) >> precision, min: 0, max: 255)
            eval(rounded)
            parts.append(rounded)
        }
        return concatenated(parts, axis: axis)
    }

    static func resize(_ pixels: MLXArray, height: Int, width: Int) -> MLXArray {
        let bytes = floor(clip(pixels * 255, min: 0, max: 255) + 0.5).asType(.int32)
        // Pillow 13 changes pass order when the height reduction dominates.
        let verticalReduction = pixels.dim(2) - height
        if verticalReduction > 0 && verticalReduction > 2 * (pixels.dim(3) - width) {
            let vertical = pass(bytes, axis: 2, count: height)
            return pass(vertical, axis: 3, count: width).asType(.float32) / 255
        }
        let horizontal = pass(bytes, axis: 3, count: width)
        return pass(horizontal, axis: 2, count: height).asType(.float32) / 255
    }
}

public struct Phi4MultimodalProcessor: UserInputProcessor {
    private let config: Phi4MultimodalProcessorConfiguration
    private let tokenizer: any Tokenizer
    public init(_ config: Phi4MultimodalProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    package func imagePatches(_ image: UserInput.Image) throws -> (MLXArray, THW) {
        guard config.patchSize > 0, config.minimumPatches > 0,
            config.maximumPatches >= config.minimumPatches, config.imageMean.count == 3,
            config.imageStd.count == 3, config.imageStd.allSatisfy({ $0 > 0 })
        else {
            throw VLMError.processing("Invalid Phi-4 image processor dimensions")
        }
        let oriented = try image.asCIImage().settingProperties([
            CIImageOption.applyOrientationProperty: true
        ])
        guard !oriented.extent.isEmpty, !oriented.extent.isInfinite else {
            throw VLMError.processing("Invalid Phi-4 image extent")
        }
        let image = MediaProcessing.inSRGBToneCurveSpace(oriented)
        let height = Int(image.extent.height)
        let width = Int(image.extent.width)
        let patch = config.patchSize
        let raw = max((height / patch) * (width / patch), 1)
        let budget = min(max(raw, config.minimumPatches), config.maximumPatches)
        let ratio = Double(width) / Double(height)
        var rows = Int(sqrt(Double(budget) / ratio))
        var columns = Int(Double(rows) * ratio)
        while rows * columns > budget {
            if rows > columns { rows -= 1 } else { columns -= 1 }
        }
        rows = max(rows, 1)
        columns = max(columns, 1)
        guard rows * columns <= config.maximumPatches else {
            throw VLMError.processing("Phi-4 image aspect ratio exceeds its patch budget")
        }
        let rawPixels = MediaProcessing.asMLXArray(image)
        var pixels = Phi4MultimodalPixelResampling.resize(
            rawPixels, height: rows * patch, width: columns * patch)
        pixels = pixels[0].transposed(1, 2, 0)
        pixels = (pixels - MLXArray(config.imageMean)) / MLXArray(config.imageStd)
        let patches = pixels.reshaped(rows, patch, columns, patch, 3).transposed(0, 2, 1, 3, 4)
            .reshaped(rows * columns, patch * patch * 3)
        return (patches, THW(1, rows, columns))
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard config.audioReduction > 0 else {
            throw VLMError.processing("Invalid Phi-4 audio reduction")
        }
        guard input.videos.isEmpty else {
            throw VLMError.processing("Phi-4 Multimodal does not accept video")
        }
        let messages = Phi4MultimodalMessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))
        let template = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        if input.images.isEmpty && input.audios.isEmpty {
            return LMInput(tokens: MLXArray(template)[.newAxis, 0...])
        }
        var imageArrays: [MLXArray] = []
        var grids: [THW] = []
        for image in input.images {
            let (patches, grid) = try imagePatches(image)
            imageArrays.append(patches)
            grids.append(grid)
        }
        var audioArrays: [MLXArray] = []
        for sound in input.audios {
            var processing = input.processing.audio
            processing.sampleRate = 16000
            processing.channels = 1
            let wave = try await sound.asMLXArray(processing: processing)
            audioArrays.append(try Phi4MultimodalAudioFeatures.extract(wave))
        }
        let decoded = tokenizer.decode(tokenIds: template, skipSpecialTokens: false)
            .replacingOccurrences(
                of: #"<\|image_\d+\|>"#, with: "<image>", options: .regularExpression
            )
            .replacingOccurrences(
                of: #"<\|audio_\d+\|>"#, with: "<|endoftext11|>", options: .regularExpression)
        let expression = try NSRegularExpression(pattern: #"<image>|<\|endoftext11\|>"#)
        let matches = expression.matches(
            in: decoded, range: NSRange(decoded.startIndex..., in: decoded))
        var result: [Int] = []
        var previous = decoded.startIndex
        var imageIndex = 0
        var audioIndex = 0
        for match in matches {
            guard let range = Range(match.range, in: decoded) else { continue }
            result += tokenizer.encode(
                text: String(decoded[previous ..< range.lowerBound]), addSpecialTokens: false)
            if decoded[range] == "<image>" {
                guard imageIndex < grids.count else {
                    throw VLMError.processing("Phi-4 image placeholder count mismatch")
                }
                result += Array(
                    repeating: config.imageToken, count: grids[imageIndex].h * grids[imageIndex].w)
                imageIndex += 1
            } else {
                guard audioIndex < audioArrays.count else {
                    throw VLMError.processing("Phi-4 audio placeholder count mismatch")
                }
                result += Array(
                    repeating: config.audioToken,
                    count: (audioArrays[audioIndex].dim(0) + config.audioReduction - 1)
                        / config.audioReduction)
                audioIndex += 1
            }
            previous = range.upperBound
        }
        result += tokenizer.encode(text: String(decoded[previous...]), addSpecialTokens: false)
        guard imageIndex == grids.count, audioIndex == audioArrays.count else {
            throw VLMError.processing("Phi-4 attached media is missing its prompt placeholders")
        }
        func paddedStack(_ arrays: [MLXArray]) -> MLXArray? {
            guard let longest = arrays.map({ $0.dim(0) }).max() else { return nil }
            return stacked(
                arrays.map { array in
                    if array.dim(0) == longest { return array }
                    return concatenated([
                        array,
                        MLXArray.zeros([longest - array.dim(0), array.dim(1)], dtype: array.dtype),
                    ])
                })
        }
        let image = paddedStack(imageArrays).map {
            LMInput.ProcessedImage(pixels: $0, frames: grids)
        }
        let audio = paddedStack(audioArrays).map { features in
            let lengths = MLXArray(audioArrays.map { $0.dim(0) })
            let mask = MLXArray(0 ..< features.dim(1))[.newAxis, 0...] .< lengths[0..., .newAxis]
            return LMInput.ProcessedAudio(features: features, mask: mask)
        }
        let text = LMInput.Text(tokens: MLXArray(result)[.newAxis, 0...])
        // Finish media preprocessing on the executor that created its streams
        // before passing the prepared tensors to the generation executor.
        try MLX.checkedEval(
            [text.tokens] + (image.map { [$0.pixels] } ?? [])
                + (audio.map { [$0.features] + ($0.mask.map { [$0] } ?? []) } ?? []))
        return LMInput(text: text, image: image, audio: audio)
    }
}

public struct Phi4MultimodalProcessorLoadingResolver: VLMProcessorLoadingResolver {
    public init() {}
    public func processorType(
        for context: VLMProcessorLoadingContext, declaredProcessorType: String?
    ) throws -> String? {
        context.modelType == "phi4mm" ? "Phi4MultimodalProcessor" : nil
    }
    public func fallbackProcessorConfiguration(for context: VLMProcessorLoadingContext) throws
        -> VLMProcessorConfiguration?
    {
        guard context.modelType == "phi4mm" else { return nil }
        return .init(data: context.configurationData, processorType: "Phi4MultimodalProcessor")
    }
}
