// Port of Blaizzy/mlx-vlm's DeepSeek OCR image preprocessing (MIT).

import CoreImage
import Foundation
import MLX
import MLXLMCommon

public struct DeepSeekOCRProcessorConfiguration: Decodable, Sendable {
    public var globalSize = 1024
    public var cropSize = 640
    public var minimumCrops = 2
    public var maximumCrops = 9
    public var cropping = true
    public var imageTokenId = 128815
    public var imageToken = "<image>"
    public var mean: [Float] = [0.5, 0.5, 0.5]
    public var std: [Float] = [0.5, 0.5, 0.5]
    public var normalize = true
    public var isOCR2 = false

    public init() {}
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: OCRKey.self)
        globalSize = try c.value("base_size", globalSize)
        cropSize = try c.value("image_size", cropSize)
        minimumCrops = try c.value("min_patches", minimumCrops)
        maximumCrops = try c.value("max_patches", maximumCrops)
        cropping = try c.value("cropping", cropping)
        imageTokenId = try c.value("image_token_index", imageTokenId)
        imageToken = try c.value("image_token", imageToken)
        mean = try c.value("image_mean", mean)
        std = try c.value("image_std", std)
        normalize = try c.value("normalize", normalize)
        isOCR2 = try c.value("model_type", "deepseekocr") == "deepseekocr_2"
        if isOCR2 {
            cropSize = try c.value("image_size", 768)
            minimumCrops = try c.value("min_patches", 1)
            maximumCrops = try c.value("max_patches", 6)
        }
        guard globalSize > 0, globalSize <= 4096, cropSize > 0, cropSize <= 4096,
            minimumCrops >= 0, maximumCrops > 0, maximumCrops <= 64,
            minimumCrops <= maximumCrops, mean.count == 3, std.count == 3,
            std.allSatisfy({ $0.isFinite && $0 > 0 }), mean.allSatisfy(\.isFinite)
        else { throw VLMError.processing("Invalid DeepSeek OCR image processor configuration.") }
    }
}

public struct DeepSeekOCRProcessor: UserInputProcessor {
    let config: DeepSeekOCRProcessorConfiguration
    let tokenizer: any Tokenizer
    public init(_ config: DeepSeekOCRProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    static func grid(width: Int, height: Int, cropSize: Int, minimum: Int, maximum: Int) -> (
        rows: Int, columns: Int
    ) {
        let ratio = Double(width) / Double(height)
        let candidates = (1 ... maximum).flatMap { columns in
            (1 ... maximum).compactMap { rows -> (Int, Int)? in
                let count = rows * columns
                return count >= max(1, minimum) && count <= maximum ? (rows, columns) : nil
            }
        }.sorted { a, b in
            let ac = a.0 * a.1
            let bc = b.0 * b.1
            return ac == bc ? (a.1 == b.1 ? a.0 < b.0 : a.1 < b.1) : ac < bc
        }
        var best = (rows: 1, columns: 1)
        var difference = Double.infinity
        for (rows, columns) in candidates {
            let d = abs(ratio - Double(columns) / Double(rows))
            if d < difference
                || (d == difference
                    && Double(width * height) > 0.5 * Double(cropSize * cropSize * rows * columns))
            {
                best = (rows, columns)
                difference = d
            }
        }
        return best
    }

    static func globalPixels(_ pixels: MLXArray, size: Int, mean: [Float]) -> MLXArray {
        let height = pixels.dim(2)
        let width = pixels.dim(3)
        let scale = min(Double(size) / Double(height), Double(size) / Double(width))
        let h = min(size, max(1, Int((Double(height) * scale).rounded(.toNearestOrEven))))
        let w = min(size, max(1, Int((Double(width) * scale).rounded(.toNearestOrEven))))
        let resized = resizedPixels(pixels, height: h, width: w)
        let fill = MLXArray(mean.map { floor($0 * 255) / 255 }).reshaped(1, 3, 1, 1)
        let top = (size - h) / 2
        let left = (size - w) / 2
        return padded(
            resized - fill,
            widths: [0, 0, .init((top, size - h - top)), .init((left, size - w - left))]) + fill
    }

    /// Pillow's RGB bicubic path rounds each separable pass to eight-bit pixels.
    /// Vision positional embeddings use floating-point interpolation separately.
    static func resizedPixels(_ pixels: MLXArray, height: Int, width: Int) -> MLXArray {
        let rows = EmbeddingGemma2ImageProcessor.bicubicWeights(
            input: pixels.dim(2), output: height)
        let columns = EmbeddingGemma2ImageProcessor.bicubicWeights(
            input: pixels.dim(3), output: width)
        let rgb = floor(clip(pixels * 255, min: 0, max: 255) + 0.5)
        let horizontal = floor(clip(matmul(rgb, columns.T), min: 0, max: 255) + 0.5)
        return floor(clip(matmul(rows, horizontal), min: 0, max: 255) + 0.5) / 255
    }

    func preparedImage(_ image: UserInput.Image) throws -> (
        global: MLXArray, crops: [MLXArray], frame: THW, tokens: Int
    ) {
        let ciImage = try EmbeddingGemma2ImageProcessor.prepared(image.asCIImage())
        let pixels = MediaProcessing.asMLXArray(ciImage)
        let width = pixels.dim(3)
        let height = pixels.dim(2)
        let shouldCrop =
            config.cropping && config.minimumCrops > 0
            && (config.isOCR2 || width > config.cropSize || height > config.cropSize)
        let grid =
            shouldCrop
            ? Self.grid(
                width: width, height: height, cropSize: config.cropSize,
                minimum: config.minimumCrops, maximum: config.maximumCrops) : (rows: 0, columns: 0)
        var crops = [MLXArray]()
        if grid.rows * grid.columns > 0 {
            let resized = Self.resizedPixels(
                pixels, height: grid.rows * config.cropSize, width: grid.columns * config.cropSize)
            for row in 0 ..< grid.rows {
                for column in 0 ..< grid.columns {
                    crops.append(
                        resized[
                            0..., 0..., row * config.cropSize ..< (row + 1) * config.cropSize,
                            column * config.cropSize ..< (column + 1) * config.cropSize])
                }
            }
        }
        var global = Self.globalPixels(pixels, size: config.globalSize, mean: config.mean)
        if config.normalize {
            let mean = MLXArray(config.mean).reshaped(1, 3, 1, 1)
            let std = MLXArray(config.std).reshaped(1, 3, 1, 1)
            global = (global - mean) / std
            crops = crops.map { ($0 - mean) / std }
        }
        let globalSide = (config.globalSize / 16 + 3) / 4
        let cropSide = (config.cropSize / 16 + 3) / 4
        let count: Int
        if config.isOCR2 {
            count = globalSide * globalSide + crops.count * cropSide * cropSide + 1
        } else {
            count =
                globalSide * (globalSide + 1) + 1
                + (crops.isEmpty ? 0 : grid.rows * cropSide * (grid.columns * cropSide + 1))
        }
        return (
            global.asType(.bfloat16), crops.map { $0.asType(.bfloat16) },
            THW(0, grid.rows, grid.columns), count
        )
    }

    private func content(_ text: String, images: Int) -> String {
        if text.contains(config.imageToken) { return text }
        return String(repeating: config.imageToken + "\n", count: images) + text
    }

    private func promptTokens(_ input: UserInput) throws -> [Int] {
        switch input.prompt {
        case .text(let text):
            return [0]
                + tokenizer.encode(
                    text: content(text, images: input.images.count), addSpecialTokens: false)
        case .chat(let messages):
            let rendered: [Message] = messages.map {
                ["role": $0.role.rawValue, "content": content($0.content, images: $0.images.count)]
            }
            return try template(rendered, input: input)
        case .messages(let messages):
            return try template(messages, input: input)
        }
    }

    private func template(_ messages: [Message], input: UserInput) throws -> [Int] {
        do {
            var tokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext)
            if tokens.first != 0 { tokens.insert(0, at: 0) }
            return tokens
        } catch TokenizerError.missingChatTemplate {
            // The upstream OCR processor's default template joins message content with spaces.
            let text = messages.compactMap { $0["content"] as? String }.joined(separator: " ") + " "
            return [0] + tokenizer.encode(text: text, addSpecialTokens: false)
        }
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.audios.isEmpty, input.videos.isEmpty else {
            throw VLMError.processing("DeepSeek OCR accepts text and images.")
        }
        let tokens = try promptTokens(input)
        let imageId = tokenizer.convertTokenToId(config.imageToken) ?? config.imageTokenId
        guard imageId != tokenizer.unknownTokenId else {
            throw VLMError.processing("DeepSeek OCR tokenizer has no image token.")
        }
        let imageCount = tokens.filter { $0 == imageId }.count
        guard imageCount == input.images.count else {
            throw VLMError.processing(
                "DeepSeek OCR placeholder count (\(imageCount)) does not match images (\(input.images.count))."
            )
        }
        if input.images.isEmpty { return LMInput(tokens: MLXArray(tokens).reshaped(1, -1)) }
        let images = try input.images.map(preparedImage)
        var expanded = [Int]()
        var index = 0
        for token in tokens {
            if token == imageId {
                expanded += Array(repeating: config.imageTokenId, count: images[index].tokens)
                index += 1
            } else {
                expanded.append(token)
            }
        }
        let allCrops = images.flatMap(\.crops)
        return LMInput(
            text: .init(tokens: MLXArray(expanded).reshaped(1, -1)),
            image: .init(
                pixels: concatenated(images.map(\.global), axis: 0), frames: images.map(\.frame),
                crops: allCrops.isEmpty ? nil : concatenated(allCrops, axis: 0)))
    }
}

public struct DeepSeekOCR2Processor: UserInputProcessor {
    let processor: DeepSeekOCRProcessor
    public init(_ config: DeepSeekOCRProcessorConfiguration, tokenizer: any Tokenizer) {
        var config = config
        if !config.isOCR2 {
            config.isOCR2 = true
            config.cropSize = 768
            config.minimumCrops = 1
            config.maximumCrops = 6
        }
        processor = DeepSeekOCRProcessor(config, tokenizer: tokenizer)
    }
    public func prepare(input: UserInput) async throws -> LMInput {
        try await processor.prepare(input: input)
    }
}

struct DeepSeekOCRProcessorLoadingResolver: VLMProcessorLoadingResolver {
    func processorType(for context: VLMProcessorLoadingContext, declaredProcessorType: String?)
        -> String?
    {
        switch context.modelType {
        case "deepseekocr": "DeepSeekOCRProcessor"
        case "deepseekocr_2": "DeepSeekOCR2Processor"
        default: nil
        }
    }
    func fallbackProcessorConfiguration(for context: VLMProcessorLoadingContext) throws
        -> VLMProcessorConfiguration?
    {
        guard let type = processorType(for: context, declaredProcessorType: nil) else { return nil }
        let model = try JSONDecoder.json5().decode(
            DeepSeekOCRConfiguration.self, from: context.configurationData)
        let data = try JSONSerialization.data(withJSONObject: [
            "model_type": context.modelType, "image_token_index": model.imageTokenId,
        ])
        return .init(data: data, processorType: type)
    }
}
