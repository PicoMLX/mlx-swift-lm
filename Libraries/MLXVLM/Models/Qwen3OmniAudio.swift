// Copyright © 2026 Apple Inc.
// Port of mlx-vlm qwen3_omni_moe/audio.py; see ThirdPartyNotices/Qwen3Omni.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum Qwen3OmniAudio {
    static func outputLength(_ frames: Int, window: Int = 50) -> Int {
        let chunk = window * 2
        return (frames / chunk) * ((chunk + 7) / 8) + ((frames % chunk + 7) / 8)
    }

    final class Attention: Module {
        @ModuleInfo(key: "q_proj") var query: Linear
        @ModuleInfo(key: "k_proj") var key: Linear
        @ModuleInfo(key: "v_proj") var value: Linear
        @ModuleInfo(key: "out_proj") var output: Linear
        let heads: Int
        let headDimensions: Int

        init(_ config: Qwen3OmniConfiguration.AudioConfiguration) {
            heads = config.heads
            headDimensions = config.dimensions / heads
            _query.wrappedValue = Linear(config.dimensions, config.dimensions)
            _key.wrappedValue = Linear(config.dimensions, config.dimensions)
            _value.wrappedValue = Linear(config.dimensions, config.dimensions)
            _output.wrappedValue = Linear(config.dimensions, config.dimensions)
        }

        func callAsFunction(_ x: MLXArray, lengths: [Int]) -> MLXArray {
            let count = x.dim(0)
            let q = query(x).reshaped(count, heads, headDimensions).transposed(1, 0, 2)
            let k = key(x).reshaped(count, heads, headDimensions).transposed(1, 0, 2)
            let v = value(x).reshaped(count, heads, headDimensions).transposed(1, 0, 2)
            var offset = 0
            var outputs: [MLXArray] = []
            for length in lengths {
                let end = offset + length
                let scores =
                    matmul(q[0..., offset ..< end], k[0..., offset ..< end].swappedAxes(-1, -2))
                    * pow(Float(headDimensions), -0.5)
                outputs.append(matmul(softmax(scores, axis: -1), v[0..., offset ..< end]))
                offset = end
            }
            return output(concatenated(outputs, axis: 1).transposed(1, 0, 2).reshaped(count, -1))
        }
    }

    final class Layer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "self_attn_layer_norm") var attentionNorm: LayerNorm
        @ModuleInfo(key: "fc1") var up: Linear
        @ModuleInfo(key: "fc2") var down: Linear
        @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm
        init(_ config: Qwen3OmniConfiguration.AudioConfiguration) {
            _attention.wrappedValue = Attention(config)
            _attentionNorm.wrappedValue = LayerNorm(dimensions: config.dimensions)
            _up.wrappedValue = Linear(config.dimensions, config.intermediateSize)
            _down.wrappedValue = Linear(config.intermediateSize, config.dimensions)
            _finalNorm.wrappedValue = LayerNorm(dimensions: config.dimensions)
        }
        func callAsFunction(_ x: MLXArray, lengths: [Int]) -> MLXArray {
            let hidden = x + attention(attentionNorm(x), lengths: lengths)
            return hidden + down(gelu(up(finalNorm(hidden))))
        }
    }

    final class Model: Module {
        let config: Qwen3OmniConfiguration.AudioConfiguration
        @ModuleInfo(key: "conv2d1") var conv1: Conv2d
        @ModuleInfo(key: "conv2d2") var conv2: Conv2d
        @ModuleInfo(key: "conv2d3") var conv3: Conv2d
        @ModuleInfo(key: "conv_out") var convOutput: Linear
        @ModuleInfo var layers: [Layer]
        @ModuleInfo(key: "ln_post") var norm: LayerNorm
        @ModuleInfo var proj1: Linear
        @ModuleInfo var proj2: Linear

        init(_ config: Qwen3OmniConfiguration.AudioConfiguration) {
            self.config = config
            _conv1.wrappedValue = Conv2d(
                inputChannels: 1, outputChannels: config.downsampleDimensions,
                kernelSize: 3, stride: 2, padding: 1)
            _conv2.wrappedValue = Conv2d(
                inputChannels: config.downsampleDimensions,
                outputChannels: config.downsampleDimensions, kernelSize: 3, stride: 2, padding: 1)
            _conv3.wrappedValue = Conv2d(
                inputChannels: config.downsampleDimensions,
                outputChannels: config.downsampleDimensions, kernelSize: 3, stride: 2, padding: 1)
            _convOutput.wrappedValue = Linear(
                config.downsampleDimensions * ((config.melBins + 7) / 8),
                config.dimensions, bias: false)
            _layers.wrappedValue = (0 ..< config.layers).map { _ in Layer(config) }
            _norm.wrappedValue = LayerNorm(dimensions: config.dimensions)
            _proj1.wrappedValue = Linear(config.dimensions, config.dimensions)
            _proj2.wrappedValue = Linear(config.dimensions, config.outputDimensions)
        }

        /// Packed valid mel frames, [mel, sum(lengths)]. Attention never crosses an audio item.
        func callAsFunction(_ features: MLXArray, lengths: [Int]) throws -> MLXArray {
            guard features.ndim == 2, features.dim(0) == config.melBins,
                !lengths.isEmpty, lengths.allSatisfy({ $0 > 0 }),
                lengths.reduce(0, +) == features.dim(1)
            else {
                throw VLMError.processing(
                    "Qwen3 Omni requires positive valid mel lengths matching its audio features")
            }
            let chunkSize = config.window * 2
            var chunks: [Int] = []
            for length in lengths {
                chunks += Array(repeating: chunkSize, count: length / chunkSize)
                if length % chunkSize > 0 { chunks.append(length % chunkSize) }
            }
            let maximum = chunks.max()!
            let padded = MLXArray.zeros(
                [chunks.count, config.melBins, maximum, 1], dtype: features.dtype)
            var start = 0
            for (index, length) in chunks.enumerated() {
                padded[index, 0..., 0 ..< length, 0] = features[0..., start ..< start + length]
                start += length
            }
            var projected: [MLXArray] = []
            for start in stride(from: 0, to: chunks.count, by: config.convolutionChunkSize) {
                let chunk = padded[start ..< min(start + config.convolutionChunkSize, chunks.count)]
                projected.append(gelu(conv3(gelu(conv2(gelu(conv1(chunk)))))))
            }
            var hidden = concatenated(projected)
            let (batch, mel, time, channels) = (
                hidden.dim(0), hidden.dim(1), hidden.dim(2), hidden.dim(3)
            )
            hidden = convOutput(hidden.transposed(0, 2, 3, 1).reshaped(batch, time, channels * mel))
            guard time <= config.maxPositions else {
                throw VLMError.processing("Qwen3 Omni audio chunk exceeds positional capacity")
            }
            let frequency = exp(
                MLXArray(0 ..< config.dimensions / 2).asType(.float32)
                    * Float(-log(10_000.0) / Double(config.dimensions / 2 - 1)))
            let phase = MLXArray(0 ..< time).asType(.float32)[0..., .newAxis] * frequency[.newAxis]
            hidden = hidden + concatenated([sin(phase), cos(phase)], axis: -1)[.newAxis]
            hidden = concatenated(
                chunks.enumerated().map { index, length in
                    hidden[index, 0 ..< (length + 7) / 8]
                })
            let inferenceWidth = ((maximum + 7) / 8) * (config.inferenceWindow / chunkSize)
            var attentionLengths: [Int] = []
            for length in lengths {
                let output = outputLength(length, window: config.window)
                attentionLengths += Array(repeating: inferenceWidth, count: output / inferenceWidth)
                if output % inferenceWidth > 0 { attentionLengths.append(output % inferenceWidth) }
            }
            for layer in layers { hidden = layer(hidden, lengths: attentionLengths) }
            return proj2(gelu(proj1(norm(hidden))))
        }
    }
}
