// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm gemma3n/vision.py at 4693cd0 (MIT).

import Foundation
import MLX
import MLXNN

final class Gemma3nVisionNorm: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray
    let eps: Float
    let activate: Bool

    init(_ channels: Int, eps: Float = 1e-5, activate: Bool = true) {
        self._weight.wrappedValue = MLXArray.ones([channels])
        self.eps = eps
        self.activate = activate
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let output = x * rsqrt(mean(x * x, axis: -1, keepDims: true) + eps) * weight
        return activate ? gelu(output) : output
    }
}

final class Gemma3nVisionScale: Module, UnaryLayer {
    @ParameterInfo var gamma: MLXArray
    init(_ channels: Int) { self._gamma.wrappedValue = MLXArray.ones([channels]) * 1e-5 }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

final class Gemma3nSameConv2d: Conv2d {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        func padding(_ length: Int, _ kernel: Int, _ stride: Int, _ dilation: Int) -> (Int, Int) {
            let count = max(
                0,
                ((length + stride - 1) / stride - 1) * stride + dilation * (kernel - 1) + 1 - length
            )
            return (count / 2, count - count / 2)
        }
        let h = padding(x.dim(1), weight.dim(1), stride.0, dilation.0)
        let w = padding(x.dim(2), weight.dim(2), stride.1, dilation.1)
        return super.callAsFunction(padded(x, widths: [.init(0), .init(h), .init(w), .init(0)]))
    }
}

final class Gemma3nConvNorm: Module, UnaryLayer {
    @ModuleInfo var conv: Conv2d
    let bn: Gemma3nVisionNorm

    init(
        _ input: Int, _ output: Int, kernel: Int = 1, stride: Int = 1,
        groups: Int = 1, same: Bool = false, padding: Int = 0,
        activate: Bool = true, bias: Bool = false, eps: Float = 1e-5
    ) {
        if same {
            self._conv.wrappedValue = Gemma3nSameConv2d(
                inputChannels: input, outputChannels: output, kernelSize: .init(kernel),
                stride: .init(stride), groups: groups, bias: bias)
        } else {
            self._conv.wrappedValue = Conv2d(
                inputChannels: input, outputChannels: output, kernelSize: .init(kernel),
                stride: .init(stride), padding: .init(padding), groups: groups, bias: bias)
        }
        bn = Gemma3nVisionNorm(output, eps: eps, activate: activate)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { bn(conv(x)) }
}

private func gemma3nDivisible(_ value: Float) -> Int {
    let rounded = max(8, Int(value + 4) / 8 * 8)
    return Float(rounded) < value * 0.9 ? rounded + 8 : rounded
}

final class Gemma3nInvertedResidual: Module, UnaryLayer {
    @ModuleInfo(key: "dw_start") var start: Gemma3nConvNorm?
    @ModuleInfo(key: "pw_exp") var expand: Gemma3nConvNorm
    @ModuleInfo(key: "dw_mid") var middle: Gemma3nConvNorm?
    @ModuleInfo(key: "pw_proj") var project: Gemma3nConvNorm
    @ModuleInfo(key: "layer_scale") var scale: Gemma3nVisionScale?
    let skip: Bool

    init(
        _ input: Int, _ output: Int, startKernel: Int = 0, middleKernel: Int = 0,
        stride: Int = 1, expansion: Float = 4, noSkip: Bool = false, layerScale: Bool = true
    ) {
        skip = input == output && stride == 1 && !noSkip
        if startKernel > 0 {
            _start.wrappedValue = Gemma3nConvNorm(
                input, input, kernel: startKernel,
                stride: middleKernel == 0 ? stride : 1, groups: input,
                padding: (startKernel - 1) / 2, activate: false)
        }
        let width = gemma3nDivisible(Float(input) * expansion)
        _expand.wrappedValue = Gemma3nConvNorm(input, width)
        if middleKernel > 0 {
            _middle.wrappedValue = Gemma3nConvNorm(
                width, width, kernel: middleKernel,
                stride: stride, groups: width, same: true)
        }
        _project.wrappedValue = Gemma3nConvNorm(width, output, activate: false)
        if layerScale { _scale.wrappedValue = Gemma3nVisionScale(output) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = expand(start?(x) ?? x)
        y = project(middle?(y) ?? y)
        y = scale?(y) ?? y
        return skip ? x + y : y
    }
}

final class Gemma3nEdgeResidual: Module, UnaryLayer {
    @ModuleInfo(key: "conv_exp") var expand: Conv2d
    let bn1: Gemma3nVisionNorm
    @ModuleInfo(key: "conv_pwl") var project: Conv2d
    let bn2: Gemma3nVisionNorm
    let skip: Bool

    init(_ input: Int, _ output: Int, stride: Int = 1, expansion: Float = 4) {
        let width = gemma3nDivisible(Float(input) * expansion)
        skip = input == output && stride == 1
        _expand.wrappedValue = Gemma3nSameConv2d(
            inputChannels: input,
            outputChannels: width, kernelSize: 3, stride: .init(stride), bias: false)
        bn1 = Gemma3nVisionNorm(width)
        _project.wrappedValue = Conv2d(
            inputChannels: width, outputChannels: output,
            kernelSize: 1, bias: false)
        bn2 = Gemma3nVisionNorm(output, activate: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = bn2(project(bn1(expand(x))))
        return skip ? x + y : y
    }
}

final class Gemma3nVisionProjection: Module, UnaryLayer {
    @ModuleInfo(key: "down_conv") var down: Conv2d?
    let norm: Gemma3nVisionNorm?
    let proj: Conv2d

    init(_ input: Int, _ output: Int, stride: Int = 1) {
        if stride > 1 {
            _down.wrappedValue = Conv2d(
                inputChannels: input, outputChannels: input,
                kernelSize: 3, stride: .init(stride), padding: 1, groups: input, bias: false)
            norm = Gemma3nVisionNorm(input, eps: 1e-6, activate: false)
        } else {
            norm = nil
        }
        proj = Conv2d(inputChannels: input, outputChannels: output, kernelSize: 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = down?(x) ?? x
        return proj(norm?(y) ?? y)
    }
}

final class Gemma3nVisionAttention: Module, UnaryLayer {
    let query: Gemma3nVisionProjection
    let key: Gemma3nVisionProjection
    let value: Gemma3nVisionProjection
    let output: Gemma3nVisionProjection
    let heads: Int
    let headDim: Int

    init(_ channels: Int, heads: Int, headDim: Int, stride: Int = 1) {
        self.heads = heads
        self.headDim = headDim
        query = Gemma3nVisionProjection(channels, heads * headDim)
        key = Gemma3nVisionProjection(channels, headDim, stride: stride)
        value = Gemma3nVisionProjection(channels, headDim, stride: stride)
        output = Gemma3nVisionProjection(heads * headDim, channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let h = x.dim(1)
        let w = x.dim(2)
        let q = query(x).reshaped(b, h * w, heads, headDim).transposed(0, 2, 1, 3)
        let k = key(x).reshaped(b, -1, headDim).expandedDimensions(axis: 1)
        let v = value(x).reshaped(b, -1, headDim).expandedDimensions(axis: 1)
        let y = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: pow(Float(headDim), -0.5), mask: .none)
        return output(y.transposed(0, 2, 1, 3).reshaped(b, h, w, heads * headDim))
    }
}

final class Gemma3nMobileAttention: Module, UnaryLayer {
    let norm: Gemma3nVisionNorm
    let attn: Gemma3nVisionAttention
    @ModuleInfo(key: "layer_scale") var scale: Gemma3nVisionScale

    init(_ channels: Int, heads: Int, headDim: Int, stride: Int) {
        norm = Gemma3nVisionNorm(channels, activate: false)
        attn = Gemma3nVisionAttention(channels, heads: heads, headDim: headDim, stride: stride)
        _scale.wrappedValue = Gemma3nVisionScale(channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x + scale(attn(norm(x))) }
}

final class Gemma3nMultiscaleFusion: Module {
    let ffn: Gemma3nInvertedResidual
    let norm: Gemma3nVisionNorm
    let resolution: Int

    init(inputChannels: Int = 1920, outputChannels: Int = 2048, resolution: Int = 16) {
        self.resolution = resolution
        ffn = Gemma3nInvertedResidual(
            inputChannels, outputChannels, expansion: 2,
            noSkip: true, layerScale: false)
        norm = Gemma3nVisionNorm(outputChannels, eps: 1e-6, activate: false)
    }

    func callAsFunction(_ inputs: [MLXArray]) -> MLXArray {
        let h = inputs[0].dim(1)
        let w = inputs[0].dim(2)
        let resized = inputs.map { x in
            x.take(MLXArray((0 ..< h).map { Int32($0 * x.dim(1) / h) }), axis: 1)
                .take(MLXArray((0 ..< w).map { Int32($0 * x.dim(2) / w) }), axis: 2)
        }
        var x = ffn(concatenated(resized, axis: -1))
        if h % resolution == 0 && w % resolution == 0 {
            x = AvgPool2d(
                kernelSize: .init((h / resolution, w / resolution)),
                stride: .init((h / resolution, w / resolution)))(x)
        } else {
            // Separable cubic resize also handles nonstandard source resolutions.
            x = gemma3nCubicResize(x, height: resolution, width: resolution)
        }
        // The reference adapter returns pooled W,H order after its NCHW conversions.
        return norm(x.transposed(0, 2, 1, 3))
    }
}

func gemma3nCubicResize(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
    func resize(_ input: MLXArray, axis: Int, length: Int) -> MLXArray {
        let source = input.dim(axis)
        let positions = (0 ..< length).map {
            (Float($0) + 0.5) * Float(source) / Float(length) - 0.5
        }
        let base = positions.map { Int(floor($0)) }
        func cubic(_ x: Float) -> Float {
            let a: Float = -0.75
            let t = abs(x)
            if t <= 1 { return (a + 2) * t * t * t - (a + 3) * t * t + 1 }
            if t < 2 { return a * t * t * t - 5 * a * t * t + 8 * a * t - 4 * a }
            return 0
        }
        var result = MLXArray.zeros(
            input.shape.enumerated().map { $0.offset == axis ? length : $0.element },
            dtype: input.dtype)
        for delta in -1 ... 2 {
            let indices = base.map { Int32(min(max($0 + delta, 0), source - 1)) }
            let weights = positions.enumerated().map {
                cubic($0.element - Float(base[$0.offset] + delta))
            }
            var shape = Array(repeating: 1, count: input.ndim)
            shape[axis] = length
            result =
                result + input.take(MLXArray(indices), axis: axis)
                * MLXArray(weights).reshaped(shape)
        }
        return result
    }
    return resize(resize(x, axis: 1, length: height), axis: 2, length: width)
}

final class Gemma3nVisionTower: Module, UnaryLayer {
    @ModuleInfo(key: "conv_stem") var stem: Gemma3nConvNorm
    @ModuleInfo var blocks: [[Module]]
    let msfa: Gemma3nMultiscaleFusion

    init(_ config: Gemma3nVisionConfiguration) {
        _stem.wrappedValue = Gemma3nConvNorm(3, 64, kernel: 3, stride: 2, same: true, bias: true)
        var stages: [[Module]] = []
        stages.append([
            Gemma3nEdgeResidual(64, 128, stride: 2),
            Gemma3nEdgeResidual(128, 128), Gemma3nEdgeResidual(128, 128),
        ])
        var stage: [Module] = [
            Gemma3nInvertedResidual(
                128, 256, startKernel: 3,
                middleKernel: 5, stride: 2, expansion: 6)
        ]
        for kernel in [5, 3, 5, 3] {
            stage.append(Gemma3nInvertedResidual(256, 256, startKernel: kernel))
        }
        stages.append(stage)
        stage = [
            Gemma3nInvertedResidual(
                256, 640, startKernel: 5,
                middleKernel: 5, stride: 2, expansion: 6)
        ]
        for _ in 0 ..< 7 { stage.append(Gemma3nInvertedResidual(640, 640, startKernel: 5)) }
        stage.append(Gemma3nInvertedResidual(640, 640, expansion: 1))
        for _ in 0 ..< 14 {
            stage.append(Gemma3nMobileAttention(640, heads: 12, headDim: 64, stride: 2))
            stage.append(Gemma3nInvertedResidual(640, 640, expansion: 2))
        }
        stages.append(stage)
        stage = [
            Gemma3nInvertedResidual(
                640, 1280, startKernel: 5,
                middleKernel: 5, stride: 2, expansion: 6)
        ]
        for _ in 0 ..< 19 {
            stage.append(Gemma3nMobileAttention(1280, heads: 16, headDim: 96, stride: 1))
            stage.append(Gemma3nInvertedResidual(1280, 1280, expansion: 2))
        }
        stages.append(stage)
        _blocks.wrappedValue = stages
        msfa = Gemma3nMultiscaleFusion(outputChannels: config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = stem(x)
        var features: [MLXArray] = []
        for (index, stage) in blocks.enumerated() {
            for block in stage { x = (block as! any UnaryLayer)(x) }
            if index >= 2 { features.append(x) }
        }
        return msfa(features)
    }
}

final class Gemma3nVisionModel: Module, UnaryLayer {
    @ModuleInfo(key: "timm_model") var tower: Gemma3nVisionTower
    init(_ config: Gemma3nVisionConfiguration) { _tower.wrappedValue = Gemma3nVisionTower(config) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { tower(x.transposed(0, 2, 3, 1)) }
}
