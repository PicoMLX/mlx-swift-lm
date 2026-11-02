// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm gemma3n/audio.py at 4693cd0 (MIT).

import Foundation
import MLX
import MLXNN

final class Gemma3nModalityNorm: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray?
    let eps: Float

    init(_ dimensions: Int, eps: Float = 1e-6, scale: Bool = true) {
        self.eps = eps
        if scale { _weight.wrappedValue = MLXArray.ones([dimensions]) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        var y = f * rsqrt(mean(f.square(), axis: -1, keepDims: true) + eps)
        if let weight { y = y * weight }
        return y.asType(x.dtype)
    }
}

final class Gemma3nAudioPosition: Module {
    @ModuleInfo(key: "pos_proj") var projection: Linear
    let channels: Int
    let heads: Int
    let left: Int
    let right: Int

    init(_ config: Gemma3nAudioConfiguration) {
        channels = config.hiddenSize
        heads = config.confNumAttentionHeads
        left = max(0, config.confAttentionContextLeft - 1)
        right = config.confAttentionContextRight
        _projection.wrappedValue = Linear(channels, channels, bias: false)
    }

    func embeddings(dtype: DType) -> MLXArray {
        let half = channels / 2
        let inverse = exp(
            MLXArray(0 ..< half).asType(.float32) * (-log(Float(10_000)) / Float(max(half - 1, 1))))
        let positions = MLXArray(Array(stride(from: left, through: -right, by: -1))).asType(
            .float32)
        let scaled = positions[0..., .newAxis] * inverse[.newAxis, 0...]
        let sinusoid = concatenated([sin(scaled), cos(scaled)], axis: -1).asType(dtype)
        return projection(sinusoid).reshaped(left + right + 1, heads, channels / heads)
    }
}

/// Dense evaluation of the reference's blocked sliding attention. The window and
/// relative-position terms are unchanged; the encoders hold at most 750 frames.
final class Gemma3nAudioAttention: Module {
    @ModuleInfo(key: "relative_position_embedding") var position: Gemma3nAudioPosition
    @ParameterInfo(key: "per_dim_scale") var perDimScale: MLXArray
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    let config: Gemma3nAudioConfiguration

    init(_ config: Gemma3nAudioConfiguration) {
        self.config = config
        _position.wrappedValue = Gemma3nAudioPosition(config)
        _perDimScale.wrappedValue = MLXArray.zeros([
            config.hiddenSize / config.confNumAttentionHeads
        ])
        _q.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        _k.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        _v.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, paddingMask: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let n = config.confNumAttentionHeads
        let d = config.hiddenSize / n
        var queries = q(x).reshaped(b, t, n, d).transposed(0, 2, 1, 3)
        let keys = k(x).reshaped(b, t, n, d).transposed(0, 2, 1, 3)
        let values = v(x).reshaped(b, t, n, d).transposed(0, 2, 1, 3)
        queries =
            queries * logAddExp(perDimScale, MLXArray(0)) * (pow(Float(d), -0.5) / log(Float(2)))
        let relative = position.embeddings(dtype: queries.dtype)
        let relativeLogits = matmul(queries, relative.transposed(1, 2, 0))
        let indices = MLXArray(0 ..< t)
        let difference = indices[0..., .newAxis] - indices[.newAxis, 0...]
        let gather = clip(position.left - difference, min: 0, max: position.left + position.right)
            .asType(.int32).reshaped(1, 1, t, t)
        var logits =
            matmul(queries, keys.transposed(0, 1, 3, 2))
            + takeAlong(relativeLogits, broadcast(gather, to: [b, n, t, t]), axis: -1)
        let cap = config.confAttentionLogitCap
        if cap > 0 { logits = tanh(logits / cap) * cap }
        let window = logicalAnd(difference .<= position.left, difference .>= -position.right)
        let valid = logicalAnd(
            window.reshaped(1, 1, t, t),
            logicalNot(paddingMask).reshaped(b, 1, 1, t))
        logits = MLX.where(valid, logits, MLXArray(config.confAttentionInvalidLogitsValue))
        let probabilities = softmax(logits.asType(.float32), axis: -1).asType(values.dtype)
        let attended = matmul(probabilities, values)
        // The blocked reference includes zero-filled context lanes. A query
        // with no valid keys has a uniform distribution over that whole block,
        // rather than over the dense sequence used above.
        let blockStart =
            indices.floorDivide(config.confAttentionChunkSize) * config.confAttentionChunkSize
        let context = logicalAnd(
            indices[.newAxis, 0...] .>= (blockStart - position.left)[0..., .newAxis],
            indices[.newAxis, 0...]
                .< (blockStart + config.confAttentionChunkSize + position.right)[0..., .newAxis])
        let contextSize = config.confAttentionChunkSize + position.left + position.right
        let emptyAttention = matmul(context.asType(values.dtype), values) / Float(contextSize)
        return MLX.where(any(valid, axis: -1, keepDims: true), attended, emptyAttention)
            .transposed(0, 2, 1, 3)
    }
}

final class Gemma3nCumulativeGroupNorm: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray
    let eps: Float

    init(_ channels: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.ones([channels])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let axes = Array(2 ..< x.ndim)
        let sums = cumsum(sum(f, axes: axes, keepDims: true), axis: 1)
        var countShape = Array(repeating: 1, count: x.ndim)
        countShape[1] = x.dim(1)
        let groupSize = x.shape.dropFirst(2).reduce(1, *)
        let counts =
            MLXArray(1 ... x.dim(1)).asType(.float32).reshaped(countShape) * Float(groupSize)
        let average = sums / counts
        let variance =
            cumsum(sum((f - average).square(), axes: axes, keepDims: true), axis: 1) / counts
        return ((f - average) * rsqrt(variance + eps) * weight).asType(x.dtype)
    }
}

final class Gemma3nAudioConvBlock: Module, UnaryLayer {
    let conv: Conv2d
    let norm: Gemma3nCumulativeGroupNorm
    let kernel: [Int]

    init(_ config: Gemma3nAudioConfiguration, index: Int) {
        kernel = config.sscpConvKernelSize[index]
        let stride = config.sscpConvStrideSize[index]
        let input = index == 0 ? 1 : config.sscpConvChannelSize[index - 1]
        let output = config.sscpConvChannelSize[index]
        conv = Conv2d(
            inputChannels: input, outputChannels: output,
            kernelSize: .init((kernel[0], kernel[1])), stride: .init((stride[0], stride[1])),
            bias: false)
        norm = Gemma3nCumulativeGroupNorm(output, eps: config.sscpConvEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let p = padded(x, widths: [.init(0), .init((0, kernel[0] - 1)), .init((1, 1)), .init(0)])
        return relu(norm(conv(p)))
    }
}

final class Gemma3nAudioSubsampling: Module, UnaryLayer {
    @ModuleInfo(key: "conv_0") var first: Gemma3nAudioConvBlock
    @ModuleInfo(key: "conv_1") var second: Gemma3nAudioConvBlock
    @ModuleInfo(key: "input_proj_linear") var projection: Linear

    init(_ config: Gemma3nAudioConfiguration) {
        _first.wrappedValue = Gemma3nAudioConvBlock(config, index: 0)
        _second.wrappedValue = Gemma3nAudioConvBlock(config, index: 1)
        var frequency = config.inputFeatSize
        for i in 0 ..< 2 {
            frequency =
                (frequency + 2 - config.sscpConvKernelSize[i][1]) / config.sscpConvStrideSize[i][1]
                + 1
        }
        _projection.wrappedValue = Linear(
            frequency * config.sscpConvChannelSize[1], config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = second(first(x.expandedDimensions(axis: -1)))
        return projection(y.reshaped(y.dim(0), y.dim(1), -1))
    }
}

final class Gemma3nAudioFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "pre_layer_norm") var pre: Gemma3nModalityNorm
    @ModuleInfo(key: "ffw_layer_1") var up: Linear
    @ModuleInfo(key: "ffw_layer_2") var down: Linear
    @ModuleInfo(key: "post_layer_norm") var post: Gemma3nModalityNorm
    let clipping: Float
    let scale: Float

    init(_ c: Gemma3nAudioConfiguration) {
        _pre.wrappedValue = Gemma3nModalityNorm(c.hiddenSize)
        _up.wrappedValue = Linear(c.hiddenSize, c.hiddenSize * 4, bias: false)
        _down.wrappedValue = Linear(c.hiddenSize * 4, c.hiddenSize, bias: false)
        _post.wrappedValue = Gemma3nModalityNorm(c.hiddenSize)
        clipping = c.gradientClipping
        scale = c.confResidualWeight
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = down(silu(up(pre(clip(x, min: -clipping, max: clipping)))))
        return x + post(clip(y, min: -clipping, max: clipping)) * scale
    }
}

final class Gemma3nAudioConformerAttention: Module {
    @ModuleInfo(key: "pre_attn_norm") var pre: Gemma3nModalityNorm
    let attn: Gemma3nAudioAttention
    @ModuleInfo var post: Linear
    @ModuleInfo(key: "post_norm") var norm: Gemma3nModalityNorm
    let clipping: Float

    init(_ c: Gemma3nAudioConfiguration) {
        _pre.wrappedValue = Gemma3nModalityNorm(c.hiddenSize)
        attn = Gemma3nAudioAttention(c)
        _post.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _norm.wrappedValue = Gemma3nModalityNorm(c.hiddenSize)
        clipping = c.gradientClipping
    }

    func callAsFunction(_ x: MLXArray, paddingMask: MLXArray) -> MLXArray {
        let features = attn(pre(clip(x, min: -clipping, max: clipping)), paddingMask: paddingMask)
        let y = post(features.reshaped(x.shape))
        return x + norm(clip(y, min: -clipping, max: clipping))
    }
}

final class Gemma3nAudioLightConv: Module, UnaryLayer {
    @ModuleInfo(key: "pre_layer_norm") var pre: Gemma3nModalityNorm
    @ModuleInfo(key: "linear_start") var start: Linear
    @ModuleInfo(key: "depthwise_conv1d") var convolution: Conv1d
    @ModuleInfo(key: "conv_norm") var norm: Gemma3nModalityNorm
    @ModuleInfo(key: "linear_end") var end: Linear
    let padding: Int
    let clipping: Float

    init(_ c: Gemma3nAudioConfiguration) {
        _pre.wrappedValue = Gemma3nModalityNorm(c.hiddenSize, eps: c.rmsNormEps)
        _start.wrappedValue = Linear(c.hiddenSize, c.hiddenSize * 2, bias: false)
        _convolution.wrappedValue = Conv1d(
            inputChannels: c.hiddenSize, outputChannels: c.hiddenSize,
            kernelSize: c.confConvKernelSize, groups: c.hiddenSize, bias: false)
        _norm.wrappedValue = Gemma3nModalityNorm(c.hiddenSize, eps: c.rmsNormEps)
        _end.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        padding = c.confConvKernelSize - 1
        clipping = c.gradientClipping
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = start(pre(x))
        let halves = split(projected, parts: 2, axis: -1)
        let gated = halves[0] * sigmoid(halves[1])
        let input = padded(gated, widths: [.init(0), .init((padding, 0)), .init(0)])
        return x + end(silu(norm(clip(convolution(input), min: -clipping, max: clipping))))
    }
}

final class Gemma3nAudioConformer: Module {
    @ModuleInfo(key: "ffw_layer_start") var first: Gemma3nAudioFeedForward
    let attention: Gemma3nAudioConformerAttention
    let lconv1d: Gemma3nAudioLightConv
    @ModuleInfo(key: "ffw_layer_end") var last: Gemma3nAudioFeedForward
    let norm: Gemma3nModalityNorm
    let clipping: Float

    init(_ c: Gemma3nAudioConfiguration) {
        _first.wrappedValue = Gemma3nAudioFeedForward(c)
        attention = Gemma3nAudioConformerAttention(c)
        lconv1d = Gemma3nAudioLightConv(c)
        _last.wrappedValue = Gemma3nAudioFeedForward(c)
        norm = Gemma3nModalityNorm(c.hiddenSize)
        clipping = c.gradientClipping
    }

    func callAsFunction(_ x: MLXArray, paddingMask: MLXArray) -> MLXArray {
        var y = attention(first(x), paddingMask: paddingMask)
        y = lconv1d(y * logicalNot(paddingMask).expandedDimensions(axis: -1).asType(y.dtype))
        return norm(clip(last(y), min: -clipping, max: clipping))
    }
}

final class Gemma3nAudioModel: Module {
    @ModuleInfo(key: "subsample_conv_projection") var subsampling: Gemma3nAudioSubsampling
    let conformer: [Gemma3nAudioConformer]
    let config: Gemma3nAudioConfiguration

    init(_ config: Gemma3nAudioConfiguration) {
        self.config = config
        _subsampling.wrappedValue = Gemma3nAudioSubsampling(config)
        conformer = (0 ..< config.confNumHiddenLayers).map { _ in Gemma3nAudioConformer(config) }
    }

    func callAsFunction(_ features: MLXArray, paddingMask: MLXArray) -> (MLXArray, MLXArray) {
        var x = subsampling(features)
        let step = config.sscpConvStrideSize.map { $0[0] }.reduce(1, *)
        let indices = minimum(MLXArray(0 ..< x.dim(1)) * step, MLXArray(paddingMask.dim(1) - 1))
        var mask = paddingMask.take(indices.asType(.int32), axis: 1)
        for block in conformer { x = block(x, paddingMask: mask) }
        if config.confReductionFactor > 1 {
            let indices = MLXArray(
                Array(stride(from: 0, to: x.dim(1), by: config.confReductionFactor)))
            x = x.take(indices, axis: 1)
            mask = mask.take(indices, axis: 1)
        }
        return (MLX.where(mask.expandedDimensions(axis: -1), MLXArray(0), x), mask)
    }
}
