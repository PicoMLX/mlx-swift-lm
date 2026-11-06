// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm MiniCPM-o audio understanding at 4693cd0 (MIT).
import Foundation
import MLX
import MLXNN

package final class MiniCPMAudioLayer: Module {
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNorm: LayerNorm
    @ModuleInfo(key: "final_layer_norm") var mlpNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: MiniCPMCrossAttention
    let fc1: Linear, fc2: Linear
    init(_ config: MiniCPMAudioConfiguration) {
        _attentionNorm.wrappedValue = LayerNorm(dimensions: config.dimension, eps: config.eps)
        _mlpNorm.wrappedValue = LayerNorm(dimensions: config.dimension, eps: config.eps)
        _attention.wrappedValue = MiniCPMCrossAttention(
            config.dimension, heads: config.heads, keyBias: false)
        fc1 = Linear(config.dimension, config.intermediateSize, bias: true)
        fc2 = Linear(config.intermediateSize, config.dimension, bias: true)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let normalized = attentionNorm(x)
        let hidden = x + attention(normalized, normalized, normalized, mask: mask)
        // mlx.nn.gelu uses the precise error-function form.
        return hidden + fc2(gelu(fc1(mlpNorm(hidden))))
    }
}
package final class MiniCPMAudio: Module {
    let conv1: Conv1d, conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var positions: Embedding
    let layers: [MiniCPMAudioLayer]
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    init(_ config: MiniCPMAudioConfiguration) {
        conv1 = Conv1d(
            inputChannels: config.melBins, outputChannels: config.dimension, kernelSize: 3,
            padding: 1, bias: true)
        conv2 = Conv1d(
            inputChannels: config.dimension, outputChannels: config.dimension, kernelSize: 3,
            stride: 2, padding: 1, bias: true)
        _positions.wrappedValue = Embedding(
            embeddingCount: config.positions, dimensions: config.dimension)
        layers = (0 ..< config.layers).map { _ in .init(config) }
        _norm.wrappedValue = LayerNorm(dimensions: config.dimension, eps: config.eps)
    }
    /// Features use time-major [batch, frames, mel]; padding is masked after both convolutions.
    func callAsFunction(_ features: MLXArray, length: Int) -> MLXArray {
        var x = gelu(conv2(gelu(conv1(features.asType(conv1.weight.dtype)))))
        x = x + positions(MLXArray(0 ..< x.dim(1)))[.newAxis, 0..., 0...]
        let valid = (length + 1) / 2
        let mask = which(
            MLXArray(0 ..< x.dim(1)) .< valid, MLXArray(Float(0)), MLXArray(Float(-1e9))
        )
        .asType(x.dtype)[.newAxis, .newAxis, .newAxis, 0...]
        for layer in layers { x = layer(x, mask: mask) }
        return norm(x)
    }
}
package final class MiniCPMAudioProjection: Module {
    let linear1: Linear, linear2: Linear
    init(input: Int, output: Int) {
        linear1 = Linear(input, output, bias: true)
        linear2 = Linear(output, output, bias: true)
    }
    func callAsFunction(_ x: MLXArray, length: Int, step: Int) -> MLXArray {
        let projected = linear2(relu(linear1(x)))
        let count = max(1, ((length + 1) / 2) / step)
        return mean(
            projected[0, ..<(count * step), 0...].reshaped(count, step, projected.dim(-1)), axis: 1)
    }
}

/// Whisper's centered 400-point STFT, Slaney mel bank and dynamic-range compression.
package enum MiniCPMAudioFeatures {
    static func filters(count: Int = 80) -> MLXArray {
        func mel(_ hz: Double) -> Double {
            hz < 1000 ? hz / (200.0 / 3) : 15 + log(hz / 1000) / (log(6.4) / 27)
        }
        func hz(_ mel: Double) -> Double {
            mel < 15 ? mel * (200.0 / 3) : 1000 * exp((mel - 15) * (log(6.4) / 27))
        }
        let points = (0 ..< (count + 2)).map { hz(Double($0) * mel(8000) / Double(count + 1)) }
        var matrix = [Float](repeating: 0, count: 201 * count)
        for m in 0 ..< count {
            for f in 0 ..< 201 {
                let frequency = Double(f) * 40
                let weight = max(
                    0,
                    min(
                        (frequency - points[m]) / (points[m + 1] - points[m]),
                        (points[m + 2] - frequency) / (points[m + 2] - points[m + 1])))
                matrix[f * count + m] = Float(weight * 2 / (points[m + 2] - points[m]))
            }
        }
        return MLXArray(matrix, [201, count])
    }
    static func extract(_ samples: MLXArray, paddedSamples: Int? = nil, melBins: Int = 80) throws
        -> MLXArray
    {
        let wave = samples.squeezed()
        guard wave.ndim == 1, wave.size > 200 else {
            throw VLMError.processing(
                "MiniCPM-o audio requires at least 13 ms of mono 16 kHz samples")
        }
        let length = max(paddedSamples ?? wave.size, wave.size)
        let padded =
            length == wave.size ? wave : concatenated([wave, MLXArray.zeros([length - wave.size])])
        // Reflection at the physical waveform boundaries matches torch.stft(center=True).
        let reflected = concatenated([
            padded[MLXArray(stride(from: 200, through: 1, by: -1))], padded,
            padded[MLXArray(stride(from: length - 2, through: length - 201, by: -1))],
        ])
        let count = length / 160 + 1
        let indices =
            MLXArray(0 ..< count)[0..., .newAxis] * 160 + MLXArray(0 ..< 400)[.newAxis, 0...]
        let window = MLXArray(
            (0 ..< 400).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / 400)) })
        let spectrum = abs(MLXFFT.rfft(reflected[indices] * window, n: 400))
        let power = spectrum * spectrum
        var logMel = log10(maximum(matmul(power, filters(count: melBins)), MLXArray(Float(1e-10))))
        // Whisper discards the final centered frame before finding its dynamic range.
        logMel = logMel[..<(count - 1), 0...]
        logMel = maximum(logMel, logMel.max() - 8)
        return (logMel + 4) / 4
    }
}
