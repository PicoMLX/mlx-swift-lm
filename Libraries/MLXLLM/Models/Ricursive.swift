// Port of Ricursive Intelligence's Apache-2.0 Ripple reference implementation.
// See ThirdPartyLicenses/Ricursive.txt for attribution and license.
// https://huggingface.co/ricursiveAI/Ripple-65B-Preview/blob/e27e171792f303d74c048184e8259ba663cfa0d5/modeling_ricursive.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private final class RicursiveRMSNorm: Module, UnaryLayer {
    let eps: Float
    var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self.weight = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let value = x.asType(.float32)
        let normalized = value * rsqrt(mean(value * value, axis: -1, keepDims: true) + eps)
        // The reference rounds the normalized activation before applying the weight.
        return weight * normalized.asType(x.dtype)
    }
}

private final class RicursiveRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dimensions: Int
    private let _inverseFrequencies: MLXArray

    init(dimensions: Int, base: Float) {
        self.dimensions = dimensions
        let indices = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
        self._inverseFrequencies = 1 / MLX.pow(base, indices / Float(dimensions))
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        callAsFunction(x, offset: MLXArray(offset))
    }

    func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        let positions =
            offset.asType(.float32).reshaped(-1, 1)
            + MLXArray(0 ..< x.dim(2)).asType(.float32)
        let frequencies = positions.expandedDimensions(axis: -1) * _inverseFrequencies
        let angles = concatenated([frequencies, frequencies], axis: -1)
            .expandedDimensions(axis: 1)
        // Preserve the reference's low-precision trigonometric values and FP32 rotation.
        let cosine = cos(angles).asType(x.dtype).asType(.float32)
        let sine = sin(angles).asType(x.dtype).asType(.float32)
        let value = x.asType(.float32)
        let rotated = concatenated(
            [-value[.ellipsis, (dimensions / 2)...], value[.ellipsis, ..<(dimensions / 2)]],
            axis: -1)
        return (value * cosine + rotated * sine).asType(x.dtype)
    }
}

private final class RicursiveAttention: Module {
    let args: RicursiveConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "g_proj") var gateProj: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: RicursiveRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RicursiveRMSNorm
    @ParameterInfo(key: "value_res_lambda") var valueResidualLambda: MLXArray?

    let rope: RicursiveRoPE?

    init(_ args: RicursiveConfiguration, layerIndex: Int) {
        self.args = args
        self.scale = pow(Float(args.headDim), -0.5)
        self._qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * args.headDim, bias: false)
        self._kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * args.headDim, bias: false)
        self._vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * args.headDim, bias: false)
        self._oProj.wrappedValue = Linear(
            args.attentionHeads * args.headDim, args.hiddenSize, bias: false)
        self._qNorm.wrappedValue = RicursiveRMSNorm(
            dimensions: args.headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RicursiveRMSNorm(
            dimensions: args.headDim, eps: args.rmsNormEps)
        if args.useGatedAttention {
            self._gateProj.wrappedValue = Linear(
                args.hiddenSize, args.attentionHeads * args.headDim, bias: false)
        }
        if args.useValueResidual {
            self._valueResidualLambda.wrappedValue = MLXArray.zeros([1])
        }
        self.rope =
            args.layerTypes[layerIndex] == "sliding_attention"
            ? RicursiveRoPE(dimensions: args.headDim, base: args.ropeTheta) : nil
    }

    func callAsFunction(
        _ x: MLXArray,
        valueResidual: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var valueInput = x
        if let valueResidual, let valueResidualLambda {
            let blend = sigmoid(valueResidualLambda[0]).asType(x.dtype).asType(.float32)
            let source = valueResidual.asType(x.dtype).asType(.float32)
            let target = x.asType(.float32)
            let difference = target - source
            valueInput = MLX.where(
                blend .< 0.5, source + blend * difference,
                target - difference * (1 - blend)
            ).asType(x.dtype)
        }

        var queries = qNorm(qProj(x).reshaped(batch, length, args.attentionHeads, args.headDim))
            .transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(batch, length, args.kvHeads, args.headDim))
            .transposed(0, 2, 1, 3)
        let values = vProj(valueInput).reshaped(batch, length, args.kvHeads, args.headDim)
            .transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        var output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        if let gateProj {
            let projected = gateProj(x)
            let gate = args.gateFullPrecision ? projected.asType(.float32) : projected
            output = output * sigmoid(gate).asType(output.dtype)
        }
        return oProj(output)
    }
}

private final class RicursiveMLP: Module, UnaryLayer {
    let clampThreshold: Float?

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: RicursiveConfiguration) {
        self.clampThreshold = args.ffnClampThreshold
        self._gateProj.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gate = gateProj(x)
        var up = upProj(x)
        if let threshold = clampThreshold, threshold != 0 {
            gate = clip(gate, max: threshold)
            up = clip(up, min: -threshold, max: threshold)
        }
        let activated = silu(gate.asType(.float32)).asType(gate.dtype)
        return downProj(activated * up)
    }
}

private final class RicursiveTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: RicursiveAttention
    let mlp: RicursiveMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RicursiveRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RicursiveRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RicursiveRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RicursiveRMSNorm

    init(_ args: RicursiveConfiguration, layerIndex: Int) {
        self._attention.wrappedValue = RicursiveAttention(args, layerIndex: layerIndex)
        self.mlp = RicursiveMLP(args)
        self._inputLayerNorm.wrappedValue = RicursiveRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RicursiveRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RicursiveRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RicursiveRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        valueResidual: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attended = attention(
            inputLayerNorm(x), valueResidual: valueResidual, mask: mask, cache: cache)
        let h = x + postAttentionLayerNorm(attended)
        return h + postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(h)))
    }
}

private final class RicursiveModelInner: Module {
    let args: RicursiveConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [RicursiveTransformerBlock]
    let norm: RicursiveRMSNorm
    let firstSlidingLayer: Int?
    let firstFullLayer: Int?

    init(_ args: RicursiveConfiguration) {
        self.args = args
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map {
            RicursiveTransformerBlock(args, layerIndex: $0)
        }
        self.norm = RicursiveRMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.firstSlidingLayer = args.layerTypes.firstIndex(of: "sliding_attention")
        self.firstFullLayer = args.layerTypes.firstIndex(of: "full_attention")
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        precondition(
            cache == nil || cache?.count == layers.count, "Incorrect Ricursive cache count")
        var h = embedTokens(inputs)
        if let scale = args.embedScale {
            h = (h.asType(.float32) * scale).asType(h.dtype)
        }
        let valueResidual = args.useValueResidual ? h : nil
        let slidingMask = firstSlidingLayer.map {
            createAttentionMask(h: h, cache: cache?[$0], windowSize: args.slidingWindow)
        }
        let fullMask = firstFullLayer.map { createAttentionMask(h: h, cache: cache?[$0]) }

        for (index, layer) in layers.enumerated() {
            let mask = args.layerTypes[index] == "sliding_attention" ? slidingMask! : fullMask!
            h = layer(h, valueResidual: valueResidual, mask: mask, cache: cache?[index])
        }
        return norm(h)
    }
}

/// Ripple's dense Transformer with local RoPE, global NoPE, and embedding value residuals.
public final class RicursiveModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let headDim: IntOrPair

    let args: RicursiveConfiguration
    private let model: RicursiveModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: RicursiveConfiguration) {
        self.args = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.headDim = .init(args.headDim)
        self.model = RicursiveModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        filterLMHeadWeights(
            from: weights.filter { !$0.key.contains("rotary_emb.inv_freq") },
            tiedWordEmbeddings: args.tieWordEmbeddings)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try args.layerTypes.map {
            try makeHybridAttentionKVCache(
                parameters: parameters, slidingWindow: args.slidingWindow,
                usesSlidingWindow: $0 == "sliding_attention")
        }
    }
}

extension RicursiveModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}

extension RicursiveModel {
    public var toolCallFormat: ToolCallFormat? { .json }

    public var reasoningConfig: ReasoningConfig? {
        .init(
            startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .none,
            isSpecialToken: false, startPolicy: .responseStart, whitespacePolicy: .singleNewline)
    }
}

/// Configuration shared by Ripple checkpoints using the `ricursive` architecture.
public struct RicursiveConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let vocabularySize: Int
    public let maxPositionEmbeddings: Int
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let embedScale: Float?
    public let ffnClampThreshold: Float?
    public let useValueResidual: Bool
    public let useGatedAttention: Bool
    public let gateFullPrecision: Bool
    public let tieWordEmbeddings: Bool
    public let hiddenActivation: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case embedScale = "embed_scale"
        case ffnClampThreshold = "ffn_clamp_threshold"
        case useValueResidual = "use_valres"
        case useGatedAttention = "use_gated_attention"
        case gateFullPrecision = "gate_full_precision"
        case tieWordEmbeddings = "tie_word_embeddings"
        case hiddenActivation = "hidden_act"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "ricursive"
        self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3072
        self.hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 50
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 24576
        self.attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 48
        self.kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        self.vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 100352
        self.maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 2048
        self.slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        self.embedScale = try c.decodeIfPresent(Float.self, forKey: .embedScale)
        self.ffnClampThreshold = try c.decodeIfPresent(Float.self, forKey: .ffnClampThreshold)
        self.useValueResidual = try c.decodeIfPresent(Bool.self, forKey: .useValueResidual) ?? false
        self.useGatedAttention =
            try c.decodeIfPresent(Bool.self, forKey: .useGatedAttention) ?? true
        self.gateFullPrecision =
            try c.decodeIfPresent(Bool.self, forKey: .gateFullPrecision) ?? true
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.hiddenActivation =
            try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "silu"

        func invalid(_ key: CodingKeys, _ message: String) -> DecodingError {
            .dataCorruptedError(forKey: key, in: c, debugDescription: message)
        }
        guard modelType == "ricursive" else {
            throw invalid(.modelType, "Expected the ricursive model type")
        }
        for (key, value) in [
            (CodingKeys.hiddenSize, hiddenSize), (.hiddenLayers, hiddenLayers),
            (.intermediateSize, intermediateSize), (.attentionHeads, attentionHeads),
            (.kvHeads, kvHeads), (.headDim, headDim), (.vocabularySize, vocabularySize),
            (.maxPositionEmbeddings, maxPositionEmbeddings), (.slidingWindow, slidingWindow),
        ] where value <= 0 {
            throw invalid(key, "Expected a positive dimension")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw invalid(.kvHeads, "Query heads must be divisible by KV heads")
        }
        self.layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< hiddenLayers).map { $0 % 5 == 4 ? "full_attention" : "sliding_attention" }
        guard layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ $0 == "sliding_attention" || $0 == "full_attention" })
        else {
            throw invalid(.layerTypes, "Expected one supported attention type per layer")
        }
        guard headDim.isMultiple(of: 2) || !layerTypes.contains("sliding_attention") else {
            throw invalid(.headDim, "Sliding attention requires an even head dimension")
        }
        guard rmsNormEps.isFinite, rmsNormEps > 0 else {
            throw invalid(.rmsNormEps, "Expected a positive, finite normalization epsilon")
        }
        guard ropeTheta.isFinite, ropeTheta > 0 else {
            throw invalid(.ropeTheta, "Expected a positive, finite RoPE base")
        }
        if let embedScale, !embedScale.isFinite {
            throw invalid(.embedScale, "Expected a finite embedding scale")
        }
        if let ffnClampThreshold, !ffnClampThreshold.isFinite || ffnClampThreshold < 0 {
            throw invalid(.ffnClampThreshold, "Expected a nonnegative, finite FFN clamp")
        }
        guard hiddenActivation == "silu" else {
            throw invalid(.hiddenActivation, "Ricursive supports the silu activation")
        }
    }
}
