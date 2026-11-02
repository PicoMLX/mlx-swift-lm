// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm phi4mm at 4693cd0 (MIT).

import Foundation
import MLX
import MLXNN

package enum Phi4MultimodalResampling {
    /// Match the pinned NaFlex path: Gaussian prefilter on downscales,
    /// then half-pixel bilinear sampling (a singleton output samples zero).
    static func weights(input: Int, output: Int) -> MLXArray {
        let scale = Double(input) / Double(output)
        let sigma = (scale - 1) / 2
        let radius = sigma > 0 ? Int(3 * sigma) : 0
        var blur = [Float](repeating: 0, count: input * input)
        if radius > 0 {
            let kernel = (-radius ... radius).map { exp(-Double($0 * $0) / (2 * sigma * sigma)) }
            let total = kernel.reduce(0, +)
            for row in 0 ..< input {
                for (tap, weight) in kernel.enumerated() {
                    let source = min(max(row + tap - radius, 0), input - 1)
                    blur[row * input + source] += Float(weight / total)
                }
            }
        } else {
            for row in 0 ..< input { blur[row * input + row] = 1 }
        }
        var interpolation = [Float](repeating: 0, count: input * output)
        for row in 0 ..< output {
            let position = output == 1 ? 0 : (Double(row) + 0.5) * scale - 0.5
            let floorPosition = Int(floor(position))
            let lower = min(max(floorPosition, 0), input - 1)
            let upper = min(max(floorPosition + 1, 0), input - 1)
            let fraction = Float(position - Double(lower))
            interpolation[row * input + lower] += 1 - fraction
            interpolation[row * input + upper] += fraction
        }
        return matmul(MLXArray(interpolation, [output, input]), MLXArray(blur, [input, input]))
    }

    static func resize(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let rows = weights(input: x.dim(-2), output: height)
        let columns = weights(input: x.dim(-1), output: width)
        return matmul(matmul(rows, x), columns.transposed())
    }
}

package final class Phi4MultimodalVisionAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    let heads: Int
    let scale: Float
    init(_ c: Phi4MultimodalVisionConfiguration) {
        _q.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _k.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _v.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        _output.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
        heads = c.heads
        scale = pow(Float(c.hiddenSize / c.heads), -0.5)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let n = x.dim(1)
        func shape(_ a: MLXArray) -> MLXArray { a.reshaped(b, n, heads, -1).transposed(0, 2, 1, 3) }
        let h = MLXFast.scaledDotProductAttention(
            queries: shape(q(x)), keys: shape(k(x)), values: shape(v(x)), scale: scale, mask: .none)
        return output(h.transposed(0, 2, 1, 3).reshaped(b, n, -1))
    }
}

package final class Phi4MultimodalVisionMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    init(_ c: Phi4MultimodalVisionConfiguration) {
        _fc1.wrappedValue = Linear(c.hiddenSize, c.intermediateSize)
        _fc2.wrappedValue = Linear(c.intermediateSize, c.hiddenSize)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

package final class Phi4MultimodalVisionBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Phi4MultimodalVisionAttention
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    let mlp: Phi4MultimodalVisionMLP
    init(_ c: Phi4MultimodalVisionConfiguration) {
        _attention.wrappedValue = .init(c)
        _norm1.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.eps)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.eps)
        mlp = .init(c)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(norm1(x))
        return h + mlp(norm2(h))
    }
}

package final class Phi4MultimodalVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patch: Linear
    @ModuleInfo(key: "position_embedding") var position: Embedding
    let size: Int
    init(_ c: Phi4MultimodalVisionConfiguration) {
        size = c.imageSize / c.patchSize
        _patch.wrappedValue = Linear(c.patchSize * c.patchSize * c.channels, c.hiddenSize)
        _position.wrappedValue = Embedding(embeddingCount: size * size, dimensions: c.hiddenSize)
    }
    func callAsFunction(_ patches: MLXArray, height: Int, width: Int) -> MLXArray {
        let grid = position.weight.reshaped(size, size, -1).transposed(2, 0, 1)
        let resized = Phi4MultimodalResampling.resize(grid, height: height, width: width)
            .reshaped(-1, height * width).transposed().asType(patches.dtype)
        return patch(patches) + resized
    }
}

package final class Phi4MultimodalVisionEncoder: Module {
    let layers: [Phi4MultimodalVisionBlock]
    init(_ c: Phi4MultimodalVisionConfiguration) {
        layers = (0 ..< c.layers).map { _ in .init(c) }
    }
}

package final class Phi4MultimodalSigLIP: Module {
    let embeddings: Phi4MultimodalVisionEmbeddings
    let encoder: Phi4MultimodalVisionEncoder
    @ModuleInfo(key: "post_layernorm") var norm: LayerNorm
    init(_ c: Phi4MultimodalVisionConfiguration) {
        embeddings = .init(c)
        encoder = .init(c)
        _norm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.eps)
    }
    func callAsFunction(_ patches: MLXArray, height: Int, width: Int) -> MLXArray {
        var x = embeddings(patches, height: height, width: width)
        // The checkpoint uses the hidden state before its final encoder block.
        for layer in encoder.layers.dropLast() { x = layer(x) }
        return x
    }
}

package final class Phi4MultimodalVision: Module {
    @ModuleInfo(key: "vision_tower") var model: Phi4MultimodalSigLIP
    init(_ c: Phi4MultimodalVisionConfiguration) { _model.wrappedValue = .init(c) }
    func callAsFunction(_ patches: MLXArray, height: Int, width: Int) -> MLXArray {
        model(patches, height: height, width: width)
    }
}

package final class Phi4MultimodalProjector: Module {
    @ModuleInfo(key: "proj_0") var first: Linear
    @ModuleInfo(key: "proj_2") var second: Linear
    init(input: Int, output: Int) {
        _first.wrappedValue = Linear(input, output)
        _second.wrappedValue = Linear(output, output)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { second(gelu(first(x))) }
}
