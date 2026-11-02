// Port of Blaizzy/mlx-vlm's DeepSeek OCR vision encoders (MIT).

import Foundation
import MLX
import MLXNN

final class OCRCLIPAttention: Module {
    let heads: Int
    let scale: Float
    @ModuleInfo(key: "qkv_proj") var qkv: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    init(_ config: DeepSeekOCRConfiguration.Vision) {
        heads = config.heads
        scale = pow(Float(config.hiddenSize / heads), -0.5)
        _qkv.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 3)
        _output.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pieces = split(qkv(x), parts: 3, axis: -1)
        let q = pieces[0].reshaped(x.dim(0), x.dim(1), heads, -1).transposed(0, 2, 1, 3)
        let k = pieces[1].reshaped(x.dim(0), x.dim(1), heads, -1).transposed(0, 2, 1, 3)
        let v = pieces[2].reshaped(x.dim(0), x.dim(1), heads, -1).transposed(0, 2, 1, 3)
        return output(
            MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .none
            )
            .transposed(0, 2, 1, 3).reshaped(x.dim(0), x.dim(1), -1))
    }
}

final class OCRCLIPMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    init(_ config: DeepSeekOCRConfiguration.Vision) {
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = fc1(x)
        return fc2(h * sigmoid(1.702 * h))
    }
}

final class OCRCLIPBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: OCRCLIPAttention
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    let mlp: OCRCLIPMLP
    init(_ config: DeepSeekOCRConfiguration.Vision) {
        _attention.wrappedValue = OCRCLIPAttention(config)
        _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        mlp = OCRCLIPMLP(config)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(norm1(x))
        return h + mlp(norm2(h))
    }
}

final class OCRCLIPEmbeddings: Module {
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    init(_ config: DeepSeekOCRConfiguration.Vision) {
        _classEmbedding.wrappedValue = MLXRandom.normal([config.hiddenSize])
        _patchEmbedding.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: config.hiddenSize,
            kernelSize: .init(config.patchSize), stride: .init(config.patchSize), bias: false)
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: (224 / config.patchSize) * (224 / config.patchSize) + 1,
            dimensions: config.hiddenSize)
    }
    func callAsFunction(_ patches: MLXArray) -> MLXArray {
        let batch = patches.dim(0)
        let height = patches.dim(1)
        let width = patches.dim(2)
        let hidden = patches.dim(3)
        let positions = positionEmbedding(MLXArray(0 ..< positionEmbedding.weight.dim(0)))
        let sourceSize = Int(sqrt(Double(positions.dim(0) - 1)))
        let spatialPositions = positions[1..., 0...].reshaped(
            1, sourceSize, sourceSize, hidden)
        let position = concatenated(
            [
                positions[..<1, 0...],
                OCRSpatial.resize(spatialPositions, height: height, width: width).reshaped(
                    height * width, hidden),
            ], axis: 0)
        let cls = broadcast(classEmbedding, to: [batch, 1, hidden]).asType(position.dtype)
        return concatenated([cls, patches.reshaped(batch, height * width, hidden)], axis: 1)
            + position
    }
}

final class OCRCLIPTransformer: Module {
    let layers: [OCRCLIPBlock]
    init(_ config: DeepSeekOCRConfiguration.Vision) {
        layers = (0 ..< config.layers).map { _ in OCRCLIPBlock(config) }
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        layers.reduce(input) { x, layer in layer(x) }
    }
}

final class OCRCLIPVision: Module {
    let embeddings: OCRCLIPEmbeddings
    @ModuleInfo(key: "pre_layrnorm") var preNorm: LayerNorm
    let transformer: OCRCLIPTransformer
    init(_ config: DeepSeekOCRConfiguration.Vision) {
        embeddings = OCRCLIPEmbeddings(config)
        _preNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        transformer = OCRCLIPTransformer(config)
    }
    func callAsFunction(_ sam: MLXArray) -> MLXArray { transformer(preNorm(embeddings(sam))) }
}

final class OCRQwenAttention: Module {
    let config: DeepSeekOCRConfiguration.Vision
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    let rope: RoPE
    init(_ config: DeepSeekOCRConfiguration.Vision) {
        self.config = config
        let dim = config.encoderDim
        let head = dim / config.encoderHeads
        _q.wrappedValue = Linear(dim, dim)
        _k.wrappedValue = Linear(dim, config.encoderKVHeads * head)
        _v.wrappedValue = Linear(dim, config.encoderKVHeads * head)
        _output.wrappedValue = Linear(dim, dim, bias: false)
        rope = RoPE(dimensions: head, traditional: false, base: config.encoderRopeTheta)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let query = rope(
            q(x).reshaped(batch, length, config.encoderHeads, -1).transposed(0, 2, 1, 3), offset: 0)
        let key = rope(
            k(x).reshaped(batch, length, config.encoderKVHeads, -1).transposed(0, 2, 1, 3),
            offset: 0)
        let value = v(x).reshaped(batch, length, config.encoderKVHeads, -1).transposed(0, 2, 1, 3)
        return output(
            MLXFast.scaledDotProductAttention(
                queries: query, keys: key, values: value,
                scale: pow(Float(config.encoderDim / config.encoderHeads), -0.5), mask: .array(mask)
            )
            .transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class OCRQwenBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: OCRQwenAttention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    let mlp: OCRMLP
    init(_ config: DeepSeekOCRConfiguration.Vision) {
        _attention.wrappedValue = OCRQwenAttention(config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.encoderDim, eps: 1e-6)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.encoderDim, eps: 1e-6)
        mlp = OCRMLP(hidden: config.encoderDim, intermediate: config.encoderIntermediateSize)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + attention(inputNorm(x), mask: mask)
        return h + mlp(postNorm(h))
    }
}

final class OCRQwenEncoder: Module {
    @ParameterInfo(key: "query_1024") var globalQueries: MLXArray
    @ParameterInfo(key: "query_768") var localQueries: MLXArray
    let layers: [OCRQwenBlock]
    let norm: RMSNorm
    init(_ config: DeepSeekOCRConfiguration.Vision) {
        _globalQueries.wrappedValue = zeros([config.queryGlobalCount, config.encoderDim])
        _localQueries.wrappedValue = zeros([config.queryLocalCount, config.encoderDim])
        layers = (0 ..< config.encoderLayers).map { _ in OCRQwenBlock(config) }
        norm = RMSNorm(dimensions: config.encoderDim, eps: 1e-6)
    }

    static func mask(imageTokens: Int, queries: Int) -> MLXArray {
        let count = imageTokens + queries
        let rows = MLXArray(0 ..< count)[0..., .newAxis]
        let columns = MLXArray(0 ..< count)[.newAxis, 0...]
        let visible = (columns .< imageTokens) .|| ((rows .>= imageTokens) .&& (rows .>= columns))
        return which(visible, MLXArray(Float(0)), MLXArray(Float(-1e9))).reshaped(
            1, 1, count, count)
    }

    func callAsFunction(_ sam: MLXArray) -> MLXArray {
        let batch = sam.dim(0)
        let hidden = sam.dim(3)
        let image = sam.reshaped(batch, -1, hidden)
        let queries = image.dim(1) == localQueries.dim(0) ? localQueries : globalQueries
        var h = concatenated(
            [image, broadcast(queries, to: [batch, queries.dim(0), hidden])], axis: 1)
        let mask = Self.mask(imageTokens: image.dim(1), queries: queries.dim(0)).asType(h.dtype)
        for layer in layers { h = layer(h, mask: mask) }
        return norm(h)[0..., (h.dim(1) - queries.dim(0))..., 0...]
    }
}

final class OCRVision2: Module {
    @ModuleInfo(key: "qwen2_encoder") var encoder: OCRQwenEncoder
    init(_ config: DeepSeekOCRConfiguration.Vision) {
        _encoder.wrappedValue = OCRQwenEncoder(config)
    }
    func callAsFunction(_ sam: MLXArray) -> MLXArray { encoder(sam) }
}

final class OCRProjector: Module {
    let config: DeepSeekOCRConfiguration.Projector
    @ModuleInfo var layers: [UnaryLayer]
    init(_ config: DeepSeekOCRConfiguration.Projector) {
        self.config = config
        if config.type == "linear" {
            _layers.wrappedValue = [Linear(config.inputDim, config.embedDim)]
        } else {
            var modules: [UnaryLayer] = [
                Linear(
                    config.inputDim * config.downsampleRatio * config.downsampleRatio,
                    config.embedDim * config.mlpRatio)
            ]
            for _ in 1 ..< config.depth - 1 {
                modules += [
                    GELU(),
                    Linear(config.embedDim * config.mlpRatio, config.embedDim * config.mlpRatio),
                ]
            }
            modules += [GELU(), Linear(config.embedDim * config.mlpRatio, config.embedDim)]
            _layers.wrappedValue = modules
        }
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = input
        if config.type != "linear" {
            let side = Int(sqrt(Double(x.dim(1))))
            let stride = config.downsampleRatio
            let padding = (stride - side % stride) % stride
            x = x.reshaped(x.dim(0), side, side, x.dim(-1))
            x = padded(x, widths: [0, .init((0, padding)), .init((0, padding)), 0])
            x = x.reshaped(
                x.dim(0), (side + padding) / stride, stride, (side + padding) / stride, stride,
                x.dim(-1)
            )
            .transposed(0, 1, 3, 5, 2, 4).reshaped(x.dim(0), -1, x.dim(-1) * stride * stride)
        }
        return layers.reduce(x) { x, layer in layer(x) }
    }
}
