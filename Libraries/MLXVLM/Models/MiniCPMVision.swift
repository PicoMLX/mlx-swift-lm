// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm MiniCPM at 4693cd0 (MIT).
import Foundation
import MLX
import MLXNN

package final class MiniCPMCrossAttention: Module {
    let heads: Int
    let dimension: Int
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    init(_ dimension: Int, heads: Int, keyBias: Bool = true) {
        self.dimension = dimension
        self.heads = heads
        _query.wrappedValue = Linear(dimension, dimension, bias: true)
        _key.wrappedValue = Linear(dimension, dimension, bias: keyBias)
        _value.wrappedValue = Linear(dimension, dimension, bias: true)
        _output.wrappedValue = Linear(dimension, dimension, bias: true)
    }
    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray? = nil)
        -> MLXArray
    {
        let b = q.dim(0)
        let n = q.dim(1)
        let m = k.dim(1)
        let d = dimension / heads
        let queries = query(q).reshaped(b, n, heads, d).transposed(0, 2, 1, 3)
        let keys = key(k).reshaped(b, m, heads, d).transposed(0, 2, 1, 3)
        let values = value(v).reshaped(b, m, heads, d).transposed(0, 2, 1, 3)
        return output(
            MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: pow(Float(d), -0.5), mask: mask.map { .array($0) } ?? .none
            ).transposed(0, 2, 1, 3).reshaped(b, n, dimension))
    }
}

package final class MiniCPMVisionMLP: Module {
    let fc1: Linear, fc2: Linear
    init(_ config: MiniCPMVisionConfiguration) {
        fc1 = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        fc2 = Linear(config.intermediateSize, config.hiddenSize, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = fc1(x)
        // MLX names this tanh formula both GELU(approx="precise") and "tanh".
        return fc2(geluApproximate(h))
    }
}
package final class MiniCPMVisionLayer: Module {
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: MiniCPMCrossAttention
    let mlp: MiniCPMVisionMLP
    init(_ config: MiniCPMVisionConfiguration) {
        _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.eps)
        _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.eps)
        _attention.wrappedValue = MiniCPMCrossAttention(config.hiddenSize, heads: config.heads)
        mlp = .init(config)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = norm1(x)
        let h = x + attention(normalized, normalized, normalized)
        return h + mlp(norm2(h))
    }
}
package final class MiniCPMVisionEmbeddings: Module {
    let config: MiniCPMVisionConfiguration
    @ModuleInfo(key: "patch_embedding") var patch: Conv2d
    @ModuleInfo(key: "position_embedding") var position: Embedding
    init(_ config: MiniCPMVisionConfiguration) {
        self.config = config
        _patch.wrappedValue = Conv2d(
            inputChannels: config.channels, outputChannels: config.hiddenSize,
            kernelSize: .init(config.patchSize), stride: .init(config.patchSize), bias: true)
        let side = config.imageSize / config.patchSize
        _position.wrappedValue = Embedding(
            embeddingCount: side * side, dimensions: config.hiddenSize)
    }
    /// Patch rows have the same row-major layout as the reference's patch-packed convolution.
    func callAsFunction(_ patches: MLXArray, height: Int, width: Int) -> MLXArray {
        let p = config.patchSize
        let side = config.imageSize / p
        let pixels = patches.reshaped(1, height, width, p, p, config.channels)
            .transposed(0, 1, 3, 2, 4, 5).reshaped(1, height * p, width * p, config.channels)
        let x = patch(pixels.asType(patch.weight.dtype)).reshaped(
            1, height * width, config.hiddenSize)
        let ids = (0 ..< height).flatMap { y in
            (0 ..< width).map { x in
                min(y * side / height, side - 1) * side + min(x * side / width, side - 1)
            }
        }
        return x + position(MLXArray(ids))[.newAxis, 0..., 0...]
    }
}
package final class MiniCPMVisionEncoder: Module {
    let layers: [MiniCPMVisionLayer]
    init(_ config: MiniCPMVisionConfiguration) {
        layers = (0 ..< config.layers).map { _ in .init(config) }
    }
}
package final class MiniCPMVision: Module {
    let embeddings: MiniCPMVisionEmbeddings
    let encoder: MiniCPMVisionEncoder
    @ModuleInfo(key: "post_layernorm") var norm: LayerNorm
    init(_ config: MiniCPMVisionConfiguration) {
        embeddings = .init(config)
        encoder = .init(config)
        _norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.eps)
    }
    func callAsFunction(
        _ patches: MLXArray, height: Int, width: Int,
        merger: MiniCPMViTMerger? = nil, insertLayer: Int = 0
    ) -> (MLXArray, Int, Int) {
        var h = height
        var w = width
        var x = embeddings(patches, height: h, width: w)
        for (index, layer) in encoder.layers.enumerated() {
            x = layer(x)
            if index == insertLayer, let merger {
                let result = merger(x[0], height: h, width: w)
                x = result.0[.newAxis, 0..., 0...]
                h = result.1
                w = result.2
            }
        }
        return (norm(x)[0], h, w)
    }
}

/// Windows are flattened in [row, column, within-row, within-column, channel] order.
package func miniCPMWindows(_ x: MLXArray, height: Int, width: Int, group: [Int]) -> MLXArray {
    x.reshaped(height / group[0], group[0], width / group[1], group[1], x.dim(-1))
        .transposed(0, 2, 1, 3, 4).reshaped(
            (height / group[0]) * (width / group[1]), group[0] * group[1], x.dim(-1))
}
package final class MiniCPMViTMerger: Module {
    let group: [Int], tanhGELU: Bool
    @ModuleInfo(key: "pre_norm") var norm: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: MiniCPMCrossAttention
    @ModuleInfo(key: "layer_norm1") var attentionNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(_ config: MiniCPMVisionConfiguration, tanhGELU: Bool) {
        group = config.window
        self.tanhGELU = tanhGELU
        let tokens = group[0] * group[1]
        let dim = config.hiddenSize
        _norm.wrappedValue = LayerNorm(dimensions: dim * tokens, eps: 1e-6)
        _attentionNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _attention.wrappedValue = MiniCPMCrossAttention(dim, heads: config.heads)
        _linear1.wrappedValue = Linear(dim * tokens, config.intermediateSize * tokens, bias: true)
        _linear2.wrappedValue = Linear(config.intermediateSize * tokens, dim, bias: true)
    }
    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> (MLXArray, Int, Int) {
        var windows = miniCPMWindows(x, height: height, width: width, group: group)
        let normalized = attentionNorm(windows)
        windows = windows + attention(normalized, normalized, normalized)
        let hidden = linear1(norm(windows.reshaped(windows.dim(0), -1)))
        return (
            linear2(geluApproximate(hidden)) + mean(windows, axis: 1),
            height / group[0], width / group[1]
        )
    }
}
package final class MiniCPMMergerBlock: Module {
    @ModuleInfo(key: "pre_norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(_ input: Int, output: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: input, eps: 1e-6)
        _linear1.wrappedValue = Linear(input, input, bias: true)
        _linear2.wrappedValue = Linear(input, output, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(geluApproximate(linear1(norm(x)))) }
}
package final class MiniCPMMerger: Module {
    let group: [Int]
    let mlp: [MiniCPMMergerBlock]
    init(_ config: MiniCPMConfiguration) {
        group = config.mergeKernel
        mlp = (0 ..< config.mergeTimes).map {
            .init(
                config.vision.hiddenSize * config.mergeKernel[0] * config.mergeKernel[1],
                output: $0 == config.mergeTimes - 1
                    ? config.geometry.hiddenSize : config.vision.hiddenSize)
        }
    }
    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        var x = x
        var h = height
        var w = width
        for layer in mlp {
            let windows = miniCPMWindows(x, height: h, width: w, group: group)
            x = layer(windows.reshaped(windows.dim(0), -1))
            h /= group[0]
            w /= group[1]
        }
        return x
    }
}
package final class MiniCPMResampler: Module {
    var query: MLXArray
    var proj: MLXArray
    @ModuleInfo(key: "kv_proj") var kvProjection: Linear?
    let attn: MiniCPMCrossAttention
    @ModuleInfo(key: "ln_q") var queryNorm: LayerNorm
    @ModuleInfo(key: "ln_kv") var keyNorm: LayerNorm
    @ModuleInfo(key: "ln_post") var outputNorm: LayerNorm
    let dimension: Int
    init(_ config: MiniCPMConfiguration) {
        dimension = config.geometry.hiddenSize
        query = MLXArray.zeros([config.queryCount, dimension])
        proj = MLXArray.zeros([dimension, dimension])
        _kvProjection.wrappedValue =
            config.vision.hiddenSize == dimension
            ? nil : Linear(config.vision.hiddenSize, dimension, bias: false)
        attn = MiniCPMCrossAttention(dimension, heads: max(1, dimension / 128))
        _queryNorm.wrappedValue = LayerNorm(dimensions: dimension, eps: 1e-6)
        _keyNorm.wrappedValue = LayerNorm(dimensions: dimension, eps: 1e-6)
        _outputNorm.wrappedValue = LayerNorm(dimensions: dimension, eps: 1e-6)
    }
    package static func positions(height: Int, width: Int, dimension: Int) -> MLXArray {
        let quarter = dimension / 4
        var result = [Float]()
        result.reserveCapacity(height * width * dimension)
        for y in 0 ..< height {
            for x in 0 ..< width {
                for coordinate in [x, y] {
                    for cosine in [false, true] {
                        for frequency in 0 ..< quarter {
                            let phase =
                                Double(coordinate) / pow(10000, Double(frequency) / Double(quarter))
                            result.append(Float(cosine ? cos(phase) : sin(phase)))
                        }
                    }
                }
            }
        }
        return MLXArray(result, [1, height * width, dimension])
    }
    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let values = keyNorm(kvProjection.map { $0(x) } ?? x)
        let keys =
            values
            + Self.positions(height: height, width: width, dimension: dimension).asType(
                values.dtype)
        return matmul(outputNorm(attn(queryNorm(query)[.newAxis, 0..., 0...], keys, values)), proj)[
            0]
    }
}
