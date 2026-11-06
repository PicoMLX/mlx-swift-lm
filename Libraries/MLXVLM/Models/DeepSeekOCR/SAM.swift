// Port of Blaizzy/mlx-vlm/models/deepseekocr/sam.py (MIT), based on Segment Anything.

import Foundation
import MLX
import MLXNN

enum OCRSpatial {
    static func resize(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        if x.dim(1) == height && x.dim(2) == width { return x }
        let rows = EmbeddingGemma2ImageProcessor.bicubicWeights(input: x.dim(1), output: height)
        let columns = EmbeddingGemma2ImageProcessor.bicubicWeights(input: x.dim(2), output: width)
        return matmul(matmul(rows, x.transposed(0, 3, 1, 2).asType(.float32)), columns.T)
            .transposed(0, 2, 3, 1).asType(x.dtype)
    }

    static func relative(_ positions: MLXArray, size: Int) -> MLXArray {
        let target = 2 * size - 1
        var positions = positions
        if positions.dim(0) != target {
            let source = positions.dim(0)
            let indices = maximum(
                (MLXArray(0 ..< target).asType(.float32) + 0.5) * (Float(source) / Float(target))
                    - 0.5, 0)
            let low = floor(indices).asType(.int32)
            let high = minimum(low + 1, source - 1)
            let weight = (indices - low.asType(.float32)).expandedDimensions(axis: -1)
            positions =
                (positions[low].asType(.float32) * (1 - weight) + positions[high].asType(.float32)
                * weight).asType(positions.dtype)
        }
        let axis = MLXArray(0 ..< size)
        return positions[axis[0..., .newAxis] - axis[.newAxis, 0...] + size - 1]
    }

    static func partition(_ x: MLXArray, window: Int) -> (MLXArray, Int, Int) {
        let h = x.dim(1)
        let w = x.dim(2)
        let c = x.dim(3)
        let hp = ((h + window - 1) / window) * window
        let wp = ((w + window - 1) / window) * window
        let paddedX = padded(x, widths: [0, .init((0, hp - h)), .init((0, wp - w)), 0])
        return (
            paddedX.reshaped(x.dim(0), hp / window, window, wp / window, window, c)
                .transposed(0, 1, 3, 2, 4, 5).reshaped(-1, window, window, c), hp, wp
        )
    }

    static func unpartition(
        _ x: MLXArray, window: Int, paddedHeight: Int, paddedWidth: Int, height: Int, width: Int
    ) -> MLXArray {
        let batch = x.dim(0) / (paddedHeight * paddedWidth / window / window)
        return x.reshaped(batch, paddedHeight / window, paddedWidth / window, window, window, -1)
            .transposed(0, 1, 3, 2, 4, 5).reshaped(batch, paddedHeight, paddedWidth, -1)[
                0..., ..<height, ..<width, 0...]
    }
}

final class OCRSAMMLP: Module {
    @ModuleInfo var lin1: Linear
    @ModuleInfo var lin2: Linear
    init(width: Int) {
        _lin1.wrappedValue = Linear(width, width * 4)
        _lin2.wrappedValue = Linear(width * 4, width)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { lin2(gelu(lin1(x))) }
}

final class OCRSAMAttention: Module {
    let heads: Int
    let scale: Float
    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear
    @ParameterInfo(key: "rel_pos_h") var relativeHeight: MLXArray
    @ParameterInfo(key: "rel_pos_w") var relativeWidth: MLXArray
    init(width: Int, heads: Int, size: Int) {
        self.heads = heads
        scale = pow(Float(width / heads), -0.5)
        _qkv.wrappedValue = Linear(width, width * 3)
        _proj.wrappedValue = Linear(width, width)
        _relativeHeight.wrappedValue = zeros([2 * size - 1, width / heads])
        _relativeWidth.wrappedValue = zeros([2 * size - 1, width / heads])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, height, width) = (x.dim(0), x.dim(1), x.dim(2))
        let projected = qkv(x).reshaped(batch, height * width, 3, heads, -1).transposed(
            2, 0, 3, 1, 4)
        let q = projected[0]
        let k = projected[1]
        let v = projected[2]
        let spatialQ = q.reshaped(batch * heads, height, width, -1)
        let rh = OCRSpatial.relative(relativeHeight, size: height)
        let rw = OCRSpatial.relative(relativeWidth, size: width)
        let biasHeight = einsum("bhwc,hkc->bhwk", spatialQ, rh).reshaped(
            batch, heads, height * width, height, 1)
        let biasWidth = einsum("bhwc,wkc->bhwk", spatialQ, rw).reshaped(
            batch, heads, height * width, 1, width)
        let bias = (biasHeight + biasWidth).reshaped(batch, heads, height * width, height * width)
        let result = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .array(bias))
        return proj(result.transposed(0, 2, 1, 3).reshaped(batch, height, width, -1))
    }
}

final class OCRSAMBlock: Module {
    let norm1: LayerNorm
    let norm2: LayerNorm
    let attn: OCRSAMAttention
    let mlp: OCRSAMMLP
    let window: Int
    init(config: DeepSeekOCRConfiguration.SAM, global: Bool) {
        window = global ? 0 : config.windowSize
        norm1 = LayerNorm(dimensions: config.width, eps: 1e-6)
        norm2 = LayerNorm(dimensions: config.width, eps: 1e-6)
        attn = OCRSAMAttention(
            width: config.width, heads: config.heads,
            size: global ? config.imageSize / config.patchSize : config.windowSize)
        mlp = OCRSAMMLP(width: config.width)
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = norm1(input)
        if window > 0 {
            let (windows, hp, wp) = OCRSpatial.partition(x, window: window)
            x = OCRSpatial.unpartition(
                attn(windows), window: window, paddedHeight: hp, paddedWidth: wp, height: x.dim(1),
                width: x.dim(2))
        } else {
            x = attn(x)
        }
        x = input + x
        return x + mlp(norm2(x))
    }
}

final class OCRSAMPatchEmbed: Module {
    let proj: Conv2d
    init(_ config: DeepSeekOCRConfiguration.SAM) {
        proj = Conv2d(
            inputChannels: 3, outputChannels: config.width, kernelSize: .init(config.patchSize),
            stride: .init(config.patchSize))
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

final class OCRSAMEncoder: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: OCRSAMPatchEmbed
    @ParameterInfo(key: "pos_embed") var positionEmbedding: MLXArray
    let blocks: [OCRSAMBlock]
    let neck: [UnaryLayer]
    @ModuleInfo(key: "net_2") var downsample1: Conv2d
    @ModuleInfo(key: "net_3") var downsample2: Conv2d
    init(_ config: DeepSeekOCRConfiguration.SAM, outputChannels: Int) {
        _patchEmbed.wrappedValue = OCRSAMPatchEmbed(config)
        _positionEmbedding.wrappedValue = zeros([
            1, config.imageSize / config.patchSize, config.imageSize / config.patchSize,
            config.width,
        ])
        blocks = (0 ..< config.layers).map {
            OCRSAMBlock(config: config, global: config.globalAttentionLayers.contains($0))
        }
        neck = [
            Conv2d(
                inputChannels: config.width, outputChannels: config.neckChannels, kernelSize: 1,
                bias: false),
            LayerNorm(dimensions: config.neckChannels, eps: 1e-6),
            Conv2d(
                inputChannels: config.neckChannels, outputChannels: config.neckChannels,
                kernelSize: 3, padding: 1, bias: false),
            LayerNorm(dimensions: config.neckChannels, eps: 1e-6),
        ]
        _downsample1.wrappedValue = Conv2d(
            inputChannels: config.neckChannels, outputChannels: config.downsampleChannels,
            kernelSize: 3, stride: 2, padding: 1, bias: false)
        _downsample2.wrappedValue = Conv2d(
            inputChannels: config.downsampleChannels, outputChannels: outputChannels, kernelSize: 3,
            stride: 2, padding: 1, bias: false)
    }
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        var x = patchEmbed(pixels)
        x = x + OCRSpatial.resize(positionEmbedding, height: x.dim(1), width: x.dim(2))
        for block in blocks { x = block(x) }
        for layer in neck { x = layer(x) }
        return downsample2(downsample1(x))
    }
}
