import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

@Suite(.serialized)
struct DeepSeekOCRTests {
    static func tinyConfiguration(ocr2: Bool = false) -> DeepSeekOCRConfiguration {
        var config = DeepSeekOCRConfiguration()
        config.modelType = ocr2 ? "deepseekocr_2" : "deepseekocr"
        config.imageTokenId = 3
        config.text.hiddenSize = 64
        config.text.vocabularySize = 64
        config.text.layers = 2
        config.text.heads = 4
        config.text.kvHeads = 2
        config.text.intermediateSize = 128
        config.text.moeIntermediateSize = 64
        config.text.experts = 4
        config.text.sharedExperts = 1
        config.text.expertsPerToken = 2
        config.sam.imageSize = 16
        config.sam.patchSize = 2
        config.sam.width = 8
        config.sam.layers = 1
        config.sam.heads = 2
        config.sam.windowSize = 4
        config.sam.globalAttentionLayers = []
        config.sam.neckChannels = 8
        config.sam.downsampleChannels = 8
        config.vision.hiddenSize = 8
        config.vision.layers = 1
        config.vision.heads = 2
        config.vision.intermediateSize = 16
        config.vision.encoderDim = 8
        config.vision.encoderLayers = 1
        config.vision.encoderHeads = 2
        config.vision.encoderKVHeads = 1
        config.vision.encoderIntermediateSize = 16
        config.vision.queryGlobalCount = 4
        config.vision.queryLocalCount = 1
        config.projector.inputDim = ocr2 ? 8 : 16
        config.projector.embedDim = 64
        return config
    }

    static func tiny(ocr2: Bool = false, quantized: Bool = false) -> DeepSeekOCR {
        withRandomState(MLXRandom.RandomState(seed: 81)) {
            let model = DeepSeekOCR(tinyConfiguration(ocr2: ocr2))
            if quantized { quantize(model: model.language, groupSize: 32, bits: 4) }
            eval(model)
            return model
        }
    }

    static func imageInput(ocr2: Bool = false, row: Int = 0) -> LMInput {
        let count = ocr2 ? 5 : 7
        return LMInput(
            text: .init(
                tokens: MLXArray([1] + Array(repeating: 3, count: count) + [2, 4]).reshaped(1, -1)),
            image: .init(
                pixels: MLXArray(0 ..< 768).asType(.float32).reshaped(1, 3, 16, 16) / 768,
                frames: [THW(row, 0, 0)]))
    }

    @Test("Both OCR architectures are registered")
    func registry() async throws {
        for type in ["deepseekocr", "deepseekocr_2"] {
            #expect(await VLMTypeRegistry.shared.contains(type))
        }
    }

    @Test("Upstream Python weights, image embeddings and logits match", arguments: [false, true])
    func pythonParity(ocr2: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let url = try #require(
                Bundle.module.url(
                    forResource: "deepseek-ocr-reference", withExtension: "safetensors"))
            let (arrays, metadata) = try loadArraysAndMetadata(url: url)
            let prefix = ocr2 ? "ocr2" : "ocr1"
            let json = try #require(metadata["\(prefix).configuration"])
            let config = try JSONDecoder().decode(
                DeepSeekOCRConfiguration.self, from: Data(json.utf8))
            try config.validateModelConfiguration()
            let model = DeepSeekOCR(config)
            let weightPrefix = "\(prefix).weights."
            let weights = Dictionary(
                uniqueKeysWithValues: arrays.compactMap { name, value in
                    name.hasPrefix(weightPrefix)
                        ? (String(name.dropFirst(weightPrefix.count)), value) : nil
                })
            try model.update(
                parameters: .unflattened(model.sanitize(weights: weights)), verify: .all)
            for variant in ["global", "cropped"] {
                let key = "\(prefix).\(variant)."
                let tokens = try #require(arrays[key + "tokens"])
                let pixels = try #require(arrays[key + "pixels"])
                let crops = arrays[key + "crops"]
                let grid = try #require(arrays[key + "grid"]).asArray(Int.self)
                let input = LMInput(
                    text: .init(tokens: tokens),
                    image: .init(
                        pixels: pixels,
                        frames: [THW(0, grid[0], grid[1])], crops: crops))
                let embeddings = try model.inputEmbeddings(input)
                #expect(
                    maxAbsDifference(embeddings, try #require(arrays[key + "embeddings"])) < 2e-5)
                let cache = try model.newCache(parameters: nil)
                let output = try model.prepare(
                    input, cache: cache, state: nil, prefill: .init(stepSize: 3))
                guard case .logits(let output) = output else {
                    Issue.record("OCR must prefill media")
                    return
                }
                #expect(
                    maxAbsDifference(
                        output.logits[0, -1], try #require(arrays[key + "logits"])[0, -1]) < 3e-5)
                let decoded = model(MLXArray([5]).reshaped(1, 1), cache: cache)
                #expect(maxAbsDifference(decoded, try #require(arrays[key + "decode"])) < 3e-5)
            }
        }
    }

    @Test(
        "Ragged image and text rows decode like independent sequences", arguments: [false, true],
        [false, true])
    func raggedBatch(ocr2: Bool, quantized: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Self.tiny(ocr2: ocr2, quantized: quantized)
            let media = Self.imageInput(ocr2: ocr2)
            let first = media.text.tokens.asArray(Int.self)
            let second = [6, 7, 8]
            let singles = try [model.newCache(parameters: nil), model.newCache(parameters: nil)]
            let expectedMedia = try model.prepare(
                media, cache: singles[0], state: nil, prefill: .init())
            guard case .logits(let expectedMedia) = expectedMedia else {
                Issue.record("Missing media logits")
                return
            }
            let expectedText = model(MLXArray(second).reshaped(1, -1), cache: singles[1])
            let batch = Array(0 ..< model.kvHeads.count).map { _ in
                BatchKVCache(leftPadding: [0, 0])
            }
            let cache = batch.map { $0 as any KVCache }
            let paddedTokens =
                first + second + Array(repeating: 0, count: first.count - second.count)
            let mask =
                Array(repeating: 1, count: first.count + second.count)
                + Array(repeating: 0, count: first.count - second.count)
            let combined = LMInput(
                text: .init(
                    tokens: MLXArray(paddedTokens).reshaped(2, -1),
                    mask: MLXArray(mask).reshaped(2, -1)), image: media.image)
            let actual = try model.prepare(
                combined, cache: cache, state: nil, prefill: .init(stepSize: 3))
            guard case .logits(let actual) = actual else {
                Issue.record("Missing batched logits")
                return
            }
            #expect(maxAbsDifference(actual.logits[0, -1], expectedMedia.logits[0, -1]) < 2e-4)
            // Right-padding is removed from the cache before subsequent batched decoding.
            #expect(batch[0].batchOffset.asArray(Int.self) == [first.count, second.count])
            _ = expectedText
            for step in 0 ..< 3 {
                let ids = [9 + step, 13 + step]
                let output = model(MLXArray(ids).reshaped(2, 1), cache: cache)
                for row in 0 ..< 2 {
                    let expected = model(MLXArray([ids[row]]).reshaped(1, 1), cache: singles[row])
                    #expect(maxAbsDifference(output[row ..< row + 1], expected) < 2e-4)
                }
            }
            batch.forEach { $0.filter(batchIndices: batchRows(1, 0)) }
            let reordered = model(MLXArray([18, 19]).reshaped(2, 1), cache: cache)
            #expect(
                maxAbsDifference(
                    reordered[0 ..< 1], model(MLXArray([18]).reshaped(1, 1), cache: singles[1]))
                    < 2e-4)
            #expect(
                maxAbsDifference(
                    reordered[1 ..< 2], model(MLXArray([19]).reshaped(1, 1), cache: singles[0]))
                    < 2e-4)
            batch.forEach { $0.filter(batchIndices: batchRows(1)) }
            let extracted = batch.map { $0.extract(0) }
            #expect(
                maxAbsDifference(
                    model(MLXArray([20]).reshaped(1, 1), cache: extracted),
                    model(MLXArray([20]).reshaped(1, 1), cache: singles[0])) < 2e-4)
        }
    }

    @Test("An image prefill can join a running text row", arguments: [false, true])
    func adoption(ocr2: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Self.tiny(ocr2: ocr2)
            let singles = try [model.newCache(parameters: nil), model.newCache(parameters: nil)]
            _ = try model.prepare(
                Self.imageInput(ocr2: ocr2), cache: singles[0], state: nil, prefill: .init())
            eval(model(MLXArray([6, 7, 8, 9]).reshaped(1, -1), cache: singles[1]))
            let batch = try singles[1].map {
                BatchKVCache.fromSingle(try #require($0 as? KVCacheSimple))
            }
            for index in batch.indices {
                batch[index].extend(
                    other: BatchKVCache.fromSingle(
                        try #require(singles[0][index] as? KVCacheSimple)))
            }
            let output = model(
                MLXArray([12, 13]).reshaped(2, 1), cache: batch.map { $0 as any KVCache })
            #expect(
                maxAbsDifference(
                    output[0 ..< 1], model(MLXArray([12]).reshaped(1, 1), cache: singles[1])) < 1e-4
            )
            #expect(
                maxAbsDifference(
                    output[1 ..< 2], model(MLXArray([13]).reshaped(1, 1), cache: singles[0])) < 1e-4
            )
        }
    }

    @Test("Float and quantized decoder projections really become shards", arguments: [false, true])
    func sharding(quantized: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Self.tiny(quantized: quantized)
            let cache = try model.newCache(parameters: nil)
            let tokens = MLXArray([1, 6, 7]).reshaped(1, -1)
            let expected = model(tokens, cache: cache)
            try model.shard(group: MLXDistributed.initialize())
            let output = model(tokens, cache: try model.newCache(parameters: nil))
            #expect(maxAbsDifference(output, expected) < 1e-4)
            let q = model.language.model.layers[0].attention.q
            #expect(quantized ? q is QuantizedAllToShardedLinear : q is AllToShardedLinear)
            #expect(throws: TensorParallelError.alreadySharded) {
                try model.shard(group: MLXDistributed.initialize())
            }
        }
    }

    @Test("Feature count mismatches and missing crops are rejected")
    func malformedMedia() throws {
        let model = Self.tiny()
        let image = try #require(Self.imageInput().image)
        #expect(throws: (any Error).self) {
            try model.inputEmbeddings(
                .init(text: .init(tokens: MLXArray([1, 3, 2]).reshaped(1, -1)), image: image))
        }
        #expect(throws: (any Error).self) {
            try model.inputEmbeddings(
                .init(
                    text: .init(tokens: Self.imageInput().text.tokens),
                    image: .init(pixels: image.pixels, frames: [THW(0, 1, 2)])))
        }
    }

    @Test(
        "Quantized checkpoints can replace vision and projector modules", arguments: [false, true])
    func quantizedVision(ocr2: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            var config = Self.tinyConfiguration(ocr2: ocr2)
            config.sam.width = 32
            config.sam.neckChannels = 32
            config.sam.downsampleChannels = 32
            config.vision.hiddenSize = 32
            config.vision.intermediateSize = 64
            config.vision.encoderDim = 32
            config.vision.encoderHeads = 4
            config.vision.encoderKVHeads = 2
            config.vision.encoderIntermediateSize = 64
            config.projector.inputDim = ocr2 ? 32 : 64
            let model = DeepSeekOCR(config)
            quantize(model: model, groupSize: 32, bits: 4)
            #expect(model.projector.layers[0] is QuantizedLinear)
            #expect(model.sam.blocks[0].attn.qkv is QuantizedLinear)
            if let vision = model.vision as? OCRCLIPVision {
                #expect(vision.transformer.layers[0].mlp.fc1 is QuantizedLinear)
            }
            let result = try model.prepare(
                Self.imageInput(ocr2: ocr2), cache: model.newCache(parameters: nil),
                state: nil, prefill: .init())
            guard case .logits(let output) = result else {
                Issue.record("Missing quantized vision prefill logits")
                return
            }
            #expect(abs(output.logits).max().item(Float.self).isFinite)
        }
    }

    @Test("OCR2 image tokens are bidirectional and query tokens remain causal")
    func visualMask() {
        let mask = OCRQwenEncoder.mask(imageTokens: 2, queries: 2).reshaped(4, 4)
        #expect(mask[0, 1].item(Float.self) == 0)
        #expect(mask[0, 2].item(Float.self) < -1e8)
        #expect(mask[2, 0].item(Float.self) == 0)
        #expect(mask[2, 3].item(Float.self) < -1e8)
        #expect(mask[3, 2].item(Float.self) == 0)
    }

    @Test("Crop selection and padded pixels match the upstream Pillow processor")
    func processorReference() throws {
        try Device.withDefaultDevice(.cpu) {
            let url = try #require(
                Bundle.module.url(
                    forResource: "deepseek-ocr-reference", withExtension: "safetensors"))
            let arrays = try loadArrays(url: url)
            let rgb = try #require(arrays["processor.rgb"])
            let expectedGrid = try #require(arrays["processor.grid"]).asArray(Int.self)
            let grid = DeepSeekOCRProcessor.grid(
                width: 20, height: 12, cropSize: 8, minimum: 2, maximum: 4)
            #expect([grid.rows, grid.columns] == expectedGrid)
            let pixels = rgb.asType(.float32).transposed(2, 0, 1).expandedDimensions(axis: 0) / 255
            let global = DeepSeekOCRProcessor.globalPixels(pixels, size: 16, mean: [0.5, 0.5, 0.5])
            #expect(
                maxAbsDifference(global, try #require(arrays["processor.global"])) <= 1.01 / 255)
            let resized = DeepSeekOCRProcessor.resizedPixels(
                pixels, height: grid.rows * 8, width: grid.columns * 8)
            let crops = (0 ..< grid.columns).map { column in
                resized[0..., 0..., ..<8, column * 8 ..< (column + 1) * 8]
            }
            #expect(
                maxAbsDifference(
                    concatenated(crops, axis: 0), try #require(arrays["processor.crops"])) <= 1.01
                    / 255)
        }
    }
}
