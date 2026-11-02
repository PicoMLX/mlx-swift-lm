// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm phi4mm at 4693cd0 (MIT).

import Foundation
import MLX
import MLXNN

package func phi4Activation(_ x: MLXArray, _ name: String) -> MLXArray {
    switch name.lowercased() {
    case "relu": relu(x)
    case "swish": silu(x)
    case "sigmoid": sigmoid(x)
    case "gelu": gelu(x)
    default: x
    }
}

package final class Phi4AudioGLULinear: Module {
    @ModuleInfo var linear: Linear
    let activation: String
    init(input: Int, output: Int, activation: String, bias: Bool) {
        _linear.wrappedValue = Linear(input, 2 * output, bias: bias)
        self.activation = activation
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let halves = split(linear(x), parts: 2, axis: -1)
        return halves[0] * phi4Activation(halves[1], activation)
    }
}

package final class Phi4AudioFeedForward: Module {
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    @ModuleInfo(key: "net_0") var first: Phi4AudioGLULinear
    @ModuleInfo(key: "net_2") var second: Linear
    init(_ c: Phi4MultimodalAudioConfiguration) {
        _norm.wrappedValue = LayerNorm(dimensions: c.dimension)
        _first.wrappedValue = .init(
            input: c.dimension, output: c.intermediateSize,
            activation: c.activation, bias: c.biasInGLU)
        _second.wrappedValue = Linear(c.intermediateSize, c.dimension)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { second(first(norm(x))) }
}

package final class Phi4AudioPointwiseGLU: Module {
    @ModuleInfo(key: "ext_pw_conv_1d") var linear: Linear
    @ParameterInfo var b1: MLXArray?
    @ParameterInfo var b2: MLXArray?
    let activation: String
    init(_ c: Phi4MultimodalAudioConfiguration) {
        _linear.wrappedValue = Linear(c.dimension, 2 * c.pointwiseChannels)
        activation = c.gluActivation
        if c.biasInGLU {
            _b1.wrappedValue = MLXArray.zeros([c.pointwiseChannels])
            _b2.wrappedValue = MLXArray.zeros([c.pointwiseChannels])
        }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(linear(x), parts: 2, axis: -1)
        return (parts[0] + (b1 ?? MLXArray(0)))
            * phi4Activation(parts[1] + (b2 ?? MLXArray(0)), activation)
    }
}

package final class Phi4AudioDepthwise: Module {
    @ModuleInfo(key: "dw_conv") var depthwise: Conv1d
    @ModuleInfo(key: "pw_conv") var pointwise: Conv1d
    init(_ c: Phi4MultimodalAudioConfiguration) {
        _depthwise.wrappedValue = Conv1d(
            inputChannels: c.dimension,
            outputChannels: c.dimension * c.depthwiseMultiplier, kernelSize: c.kernelSize,
            padding: c.causal ? c.kernelSize - 1 : (c.kernelSize - 1) / 2, groups: c.dimension)
        _pointwise.wrappedValue = Conv1d(
            inputChannels: c.dimension * c.depthwiseMultiplier,
            outputChannels: c.depthwiseChannels, kernelSize: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { pointwise(depthwise(x)) }
}

package final class Phi4AudioConvolution: Module {
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    let glu: Phi4AudioPointwiseGLU
    @ModuleInfo(key: "dw_sep_conv_1d") var convolution: Phi4AudioDepthwise
    @ModuleInfo(key: "ext_pw_conv_1d") var output: Linear
    let causalTrim: Int
    let activation: String
    init(_ c: Phi4MultimodalAudioConfiguration) {
        _norm.wrappedValue = LayerNorm(dimensions: c.dimension)
        glu = .init(c)
        _convolution.wrappedValue = .init(c)
        _output.wrappedValue = Linear(c.dimension, c.pointwiseChannels)
        causalTrim = c.causal ? c.kernelSize - 1 : 0
        activation = c.convActivation
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convolution(glu(norm(x)))
        if causalTrim > 0 { h = h[0..., ..<(h.dim(1) - causalTrim), 0...] }
        return output(phi4Activation(h, activation))
    }
}

package final class Phi4AudioAttention: Module {
    @ModuleInfo(key: "linear_q") var q: Linear
    @ModuleInfo(key: "linear_k") var k: Linear
    @ModuleInfo(key: "linear_v") var v: Linear
    @ModuleInfo(key: "linear_out") var output: Linear
    let heads: Int
    let headDimension: Int
    let scale: Float
    init(_ c: Phi4MultimodalAudioConfiguration) {
        _q.wrappedValue = Linear(c.dimension, c.dimension)
        _k.wrappedValue = Linear(c.dimension, c.dimension)
        _v.wrappedValue = Linear(c.dimension, c.dimension)
        _output.wrappedValue = Linear(c.dimension, c.dimension)
        heads = c.heads
        headDimension = c.dimension / c.heads
        scale = pow(Float(headDimension), -0.5)
    }
    func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let n = x.dim(1)
        func shape(_ a: MLXArray) -> MLXArray {
            a.reshaped(b, n, heads, headDimension).transposed(0, 2, 1, 3)
        }
        let queries = shape(q(x))
        let keys = shape(k(x))
        let values = shape(v(x))
        let scores = matmul(queries * scale, keys.swappedAxes(-1, -2)) + bias
        let h = matmul(softmax(scores, axis: -1), values).transposed(0, 2, 1, 3).reshaped(b, n, -1)
        return output(h)
    }
}

package final class Phi4AudioRelativeBias: Module {
    @ModuleInfo(key: "bias_values") var embedding: Embedding
    let distance: Int
    init(_ c: Phi4MultimodalAudioConfiguration) {
        distance = c.relativeDistance
        _embedding.wrappedValue = Embedding(embeddingCount: 2 * distance, dimensions: c.heads)
    }
    func callAsFunction(length: Int) -> MLXArray {
        let positions = MLXArray(0 ..< length)
        let relative = positions[.newAxis, 0...] - positions[0..., .newAxis]
        let indices = clip(relative, min: -distance, max: distance - 1) + distance
        return embedding(indices).transposed(2, 0, 1)[.newAxis, 0..., 0..., 0...]
    }
}

package final class Phi4AudioNormalization: Module {
    @ParameterInfo(key: "global_mean") var mean: MLXArray
    @ParameterInfo(key: "global_invstd") var inverseStd: MLXArray
    init(_ c: Phi4MultimodalAudioConfiguration) {
        _mean.wrappedValue = MLXArray.zeros([c.inputSize])
        _inverseStd.wrappedValue = MLXArray.ones([c.inputSize])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { (x - mean) * inverseStd }
}

package final class Phi4AudioSubsamplePair: Module {
    let dw: Conv2d
    let pw: Conv2d
    init(channels: Int) {
        dw = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2,
            padding: 1, groups: channels)
        pw = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { relu(pw(dw(x))) }
}

package final class Phi4AudioSubsampling: Module {
    @ModuleInfo(key: "conv_0") var first: Conv2d
    @ModuleInfo(key: "dw_pw_layers") var pairs: [Phi4AudioSubsamplePair]
    @ModuleInfo var out: Linear
    init(_ c: Phi4MultimodalAudioConfiguration) {
        let count = Int(log2(Double(c.reduction)))
        _first.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: c.convChannels,
            kernelSize: 3, stride: 2, padding: 1)
        _pairs.wrappedValue = (0 ..< max(0, count - 1)).map { _ in .init(channels: c.convChannels) }
        var frequency = c.inputSize
        for _ in 0 ..< count { frequency = (frequency + 1) / 2 }
        _out.wrappedValue = Linear(c.convChannels * frequency, c.dimension)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = relu(first(x[0..., 0..., 0..., .newAxis]))
        for pair in pairs { h = pair(h) }
        return out(h.transposed(0, 1, 3, 2).reshaped(h.dim(0), h.dim(1), -1))
    }
}

package final class Phi4AudioConformerBlock: Module {
    @ModuleInfo(key: "feed_forward_in") var feedIn: Phi4AudioFeedForward
    @ModuleInfo(key: "self_attn") var attention: Phi4AudioAttention
    let conv: Phi4AudioConvolution
    @ModuleInfo(key: "feed_forward_out") var feedOut: Phi4AudioFeedForward
    @ModuleInfo(key: "layer_norm_att") var attentionNorm: LayerNorm
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    init(_ c: Phi4MultimodalAudioConfiguration) {
        _feedIn.wrappedValue = .init(c)
        _attention.wrappedValue = .init(c)
        conv = .init(c)
        _feedOut.wrappedValue = .init(c)
        _attentionNorm.wrappedValue = LayerNorm(dimensions: c.dimension)
        _norm.wrappedValue = LayerNorm(dimensions: c.dimension)
    }
    func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        var h = x + 0.5 * feedIn(x)
        h = h + attention(attentionNorm(h), bias: bias)
        h = h + conv(h)
        h = h + 0.5 * feedOut(h)
        return norm(h)
    }
}

package final class Phi4MultimodalAudio: Module {
    @ModuleInfo(key: "encoder_embedding") var normalization: Phi4AudioNormalization
    let embed: Phi4AudioSubsampling
    @ModuleInfo(key: "relative_attention_bias_layer") var relativeBias: Phi4AudioRelativeBias
    let encoders: [Phi4AudioConformerBlock]
    init(_ c: Phi4MultimodalAudioConfiguration) {
        _normalization.wrappedValue = .init(c)
        embed = .init(c)
        _relativeBias.wrappedValue = .init(c)
        encoders = (0 ..< c.layers).map { _ in .init(c) }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = embed(normalization(x))
        let originalLength = h.dim(1)
        let batch = h.dim(0)
        let dimension = h.dim(2)
        let chunks = (originalLength + 499) / 500
        if chunks > 1 {
            let padding = chunks * 500 - originalLength
            if padding > 0 {
                h = concatenated(
                    [h, MLXArray.zeros([batch, padding, dimension], dtype: h.dtype)], axis: 1)
            }
            h = h.reshaped(batch * chunks, 500, dimension)
        }
        let bias = relativeBias(length: h.dim(1))
        for encoder in encoders { h = encoder(h, bias: bias) }
        if chunks > 1 {
            h = h.reshaped(batch, chunks * 500, dimension)[0..., ..<originalLength, 0...]
        }
        return h
    }
}

package final class Phi4AudioProjectionBranch: Module {
    @ModuleInfo(key: "proj_0") var first: Linear
    @ModuleInfo(key: "proj_2") var second: Linear
    init(input: Int, output: Int) {
        _first.wrappedValue = Linear(input, output)
        _second.wrappedValue = Linear(output, output)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { second(gelu(first(x))) }
}

package final class Phi4MultimodalAudioProjection: Module {
    let speech: Phi4AudioProjectionBranch
    let vision: Phi4AudioProjectionBranch
    init(input: Int, output: Int) {
        speech = .init(input: input, output: output)
        vision = .init(input: input, output: output)
    }
}
