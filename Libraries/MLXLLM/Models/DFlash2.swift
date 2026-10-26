// Copyright © 2026 Apple Inc.

// DFlash2 block drafter (https://inco.ai/blog/dflash2/), a port of the reference
// `dflash/model_mlx.py` (https://github.com/z-lab/dflash), which also serves
// DFlash v1 checkpoints.
//
// The drafter embeds an `[anchor, MASK, ...]` block, runs it through a few
// layers that attend to the target's hidden states, and reads the block's
// logits through the target's head. DFlash2 wraps each sublayer in a dynamic
// convolution, keeps the top-K candidates per position, and walks one path
// through them with a pairwise selector. DFlash v1 has neither and drafts the
// argmax at every masked position.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// The `dflash_config` object of a DFlash2 checkpoint.
public struct DFlash2SpecConfiguration: Codable, Sendable {
    public var blockSize: Int = 16
    public var convGroupSize: Int = 16
    public var convKernelSize: Int = 2
    public var maskTokenId: Int = 0
    public var selectorRank: Int = 256
    public var selectorTopK: Int = 16
    public var targetLayerIds: [Int] = []
    public var inputEmbeddingScale: Float = 1.0
    public var outputMultiplier: Float = 1.0
    public var finalLogitSoftcapping: Float?
    /// The mlx-community layout's sliding window: `use_swa` with `swa_window_size`.
    public var useSlidingWindow: Bool?
    public var slidingWindowSize: Int?
    /// Whether the object names a convolution or a selector: DFlash2 rather than v1.
    public var describesDFlash2: Bool = false

    enum CodingKeys: String, CodingKey {
        case blockSize = "block_size"
        case convGroupSize = "conv_group_size"
        case convKernelSize = "conv_kernel_size"
        case maskTokenId = "mask_token_id"
        case selectorRank = "selector_rank"
        case selectorTopK = "selector_top_k"
        case targetLayerIds = "target_layer_ids"
        case inputEmbeddingScale = "input_embedding_scale"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useSlidingWindow = "use_swa"
        case slidingWindowSize = "swa_window_size"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? 16
        convGroupSize = try container.decodeIfPresent(Int.self, forKey: .convGroupSize) ?? 16
        convKernelSize = try container.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 2
        maskTokenId = try container.decodeIfPresent(Int.self, forKey: .maskTokenId) ?? 0
        selectorRank = try container.decodeIfPresent(Int.self, forKey: .selectorRank) ?? 256
        selectorTopK = try container.decodeIfPresent(Int.self, forKey: .selectorTopK) ?? 16
        targetLayerIds =
            try container.decodeIfPresent([Int].self, forKey: .targetLayerIds) ?? []
        inputEmbeddingScale =
            try container.decodeIfPresent(Float.self, forKey: .inputEmbeddingScale) ?? 1.0
        outputMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .outputMultiplier) ?? 1.0
        finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        useSlidingWindow = try container.decodeIfPresent(Bool.self, forKey: .useSlidingWindow)
        slidingWindowSize = try container.decodeIfPresent(Int.self, forKey: .slidingWindowSize)
        describesDFlash2 = [.convKernelSize, .convGroupSize, .selectorRank, .selectorTopK]
            .contains(where: container.contains)
    }
}

/// Configuration of a DFlash2 (`architectures: [DFlash2DraftModel]`) or DFlash
/// v1 (`[DFlashDraftModel]`) draft checkpoint; both report `model_type: qwen3`.
///
/// Reads both published layouts: `layer_types` with `sliding_window` (z-lab,
/// inco.ai), and `dflash_config.use_swa` with a top-level `block_size`
/// (mlx-community). A DFlash2 layer must be non-causal sliding-window
/// attention. A v1 layer may be sliding or full attention; its block attends
/// causally in sliding layers unless `is_causal` says otherwise, as the
/// reference does.
public struct DFlash2Configuration: Decodable, Sendable {
    public var hiddenSize: Int = 5120
    public var hiddenLayers: Int = 5
    public var attentionHeads: Int = 32
    public var kvHeads: Int = 8
    public var headDim: Int = 128
    public var intermediateSize: Int = 17408
    public var vocabularySize: Int = 248320
    public var rmsNormEps: Float = 1e-6
    public var maxPositionEmbeddings: Int = 262144
    public var slidingWindow: Int = 2048
    public var ropeTheta: Float = 10_000_000
    public var ropeScaling: [String: StringOrNumber]? = nil
    /// Depth of the target the drafter was distilled for; mlx-community omits it.
    public var numTargetLayers: Int?
    public var dflash: DFlash2SpecConfiguration = .init()
    /// `true` for DFlash2 (dynamic convolution and candidate selector), `false` for v1.
    public var isDFlash2: Bool = true
    /// `sliding_attention` or `full_attention`, one per layer. Empty means every
    /// layer slides, as a DFlash2 drafter's do.
    public var layerTypes: [String] = []
    /// Whether block positions attend causally; `nil` lets the variant and layer
    /// type decide.
    public var isCausal: Bool?

    enum CodingKeys: String, CodingKey {
        case architectures
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case isCausal = "is_causal"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case numTargetLayers = "num_target_layers"
        case blockSize = "block_size"
        case dflash = "dflash_config"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 5120
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 5
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hiddenSize / attentionHeads)
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 17408
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248320
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
        numTargetLayers = try container.decodeIfPresent(Int.self, forKey: .numTargetLayers)
        dflash =
            try container.decodeIfPresent(DFlash2SpecConfiguration.self, forKey: .dflash)
            ?? .init()
        let nested = try? container.nestedContainer(
            keyedBy: DFlash2SpecConfiguration.CodingKeys.self, forKey: .dflash)
        if try nested?.decodeIfPresent(Int.self, forKey: .blockSize) == nil,
            let blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize)
        {
            dflash.blockSize = blockSize
        }

        var ropeParameters =
            try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
            ?? container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        if ropeParameters?["type"] == nil, let ropeType = ropeParameters?["rope_type"] {
            ropeParameters?["type"] = ropeType
        }
        let topLevelTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = ropeParameters?["rope_theta"]?.asFloat() ?? topLevelTheta ?? 10_000_000
        ropeScaling = ropeParameters

        let architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        if architectures.contains("DFlash2DraftModel") {
            isDFlash2 = true
        } else if architectures.contains("DFlashDraftModel") {
            isDFlash2 = false
        } else {
            isDFlash2 = dflash.describesDFlash2
        }

        if let types = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types
        } else {
            let type = dflash.useSlidingWindow == true ? "sliding_attention" : "full_attention"
            layerTypes = Array(repeating: type, count: hiddenLayers)
        }
        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "layer_types must name one type per layer")
        }
        guard layerTypes.allSatisfy({ $0 == "sliding_attention" || $0 == "full_attention" })
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "unsupported layer type in \(layerTypes)")
        }
        if isDFlash2, layerTypes.contains("full_attention") {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "DFlash2 needs sliding-window attention in every layer")
        }
        if layerTypes.contains("sliding_attention") {
            let window =
                try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
                ?? dflash.slidingWindowSize
            guard let window, window > 1 else {
                throw DecodingError.dataCorruptedError(
                    forKey: .layerTypes, in: container,
                    debugDescription: "sliding_attention layers need a sliding_window above 1")
            }
            slidingWindow = window
        }
        isCausal = try container.decodeIfPresent(Bool.self, forKey: .isCausal)
        // The reference treats a missing `is_causal` as causal for sliding layers.
        if isDFlash2, isCausal != false {
            throw DecodingError.dataCorruptedError(
                forKey: .isCausal, in: container,
                debugDescription: "DFlash2 needs a non-causal (is_causal: false) drafter")
        }
        guard !dflash.targetLayerIds.isEmpty else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription: "dflash_config.target_layer_ids is empty")
        }
    }

    private func slides(_ index: Int) -> Bool {
        layerTypes.isEmpty || layerTypes[index] == "sliding_attention"
    }

    /// Context rows layer `index` can see: the window minus the block's own
    /// anchor, or every row (`Int.max`) for a full-attention layer.
    public func contextWindow(forLayer index: Int) -> Int {
        slides(index) ? slidingWindow - 1 : .max
    }

    /// Context rows the drafter keeps: as many as its widest layer can see.
    public var contextWindow: Int {
        (0 ..< hiddenLayers).map(contextWindow(forLayer:)).max() ?? .max
    }

    /// How far back a block position in layer `index` attends: the sliding
    /// window, or without bound (`Int.max`) for a full-attention layer.
    public func attentionWindow(forLayer index: Int) -> Int {
        slides(index) ? slidingWindow : .max
    }

    /// Whether block positions in layer `index` see only earlier block positions.
    /// A DFlash2 block never does; a v1 sliding layer's does unless `isCausal`
    /// says otherwise, as the reference reads a missing `is_causal`.
    public func attendsCausally(layer index: Int) -> Bool {
        isCausal ?? (!isDFlash2 && slides(index))
    }
}

// MARK: - Two-tap dynamic depthwise convolution

/// `Conv(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}` on `groupSize`-channel groups:
/// a learned base kernel plus a per-group correction projected from the input.
/// `prepare` applies slot 0 and returns the dynamic kernel; `finish` applies slot 1.
final class DFlash2DynamicConv: Module {
    let kernelSize: Int
    let groupSize: Int

    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        precondition(hiddenSize % groupSize == 0, "conv groups must divide hidden size")
        self.kernelSize = kernelSize
        self.groupSize = groupSize
        let groups = hiddenSize / groupSize
        _baseKernel.wrappedValue = MLXArray.zeros([2, kernelSize, hiddenSize])
        _kernelProjection.wrappedValue = Linear(hiddenSize, 2 * kernelSize * groups, bias: false)
        super.init()
    }

    /// hidden `[B, L, H]`, dynamic `[B, L, K, groups]`, base `[K, H]`. The first
    /// position reads zeros before it.
    private static func convolve(
        _ hidden: MLXArray, dynamic: MLXArray, base: MLXArray, groupSize: Int
    ) -> MLXArray {
        let (b, l, h) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        let groups = h / groupSize
        let blocks = hidden.reshaped(b, l, groups, groupSize)
        var output = MLXArray.zeros(like: blocks)
        for tap in 0 ..< base.dim(0) {
            let values: MLXArray
            if tap == 0 {
                values = blocks
            } else {
                values = concatenated(
                    [MLXArray.zeros(like: blocks[0..., ..<tap]), blocks[0..., ..<(l - tap)]],
                    axis: 1)
            }
            let baseKernel = base[tap].asType(hidden.dtype).reshaped(1, 1, groups, groupSize)
            output = output + baseKernel * values
            output = output + dynamic[0..., 0..., tap, 0..., .newAxis] * values
        }
        return output.reshaped(hidden.shape)
    }

    func prepare(_ hidden: MLXArray) -> (convolved: MLXArray, kernel: MLXArray) {
        let groups = hidden.dim(-1) / groupSize
        let dynamic = kernelProjection(hidden).reshaped(
            hidden.dim(0), hidden.dim(1), 2, kernelSize, groups)
        return (
            Self.convolve(
                hidden, dynamic: dynamic[0..., 0..., 0, 0..., 0...],
                base: baseKernel[0], groupSize: groupSize),
            dynamic[0..., 0..., 1, 0..., 0...]
        )
    }

    func finish(_ hidden: MLXArray, kernel: MLXArray) -> MLXArray {
        Self.convolve(hidden, dynamic: kernel, base: baseKernel[1], groupSize: groupSize)
    }
}

// MARK: - Attention

/// Block queries attend to the cached context K/V plus the block's own K/V.
final class DFlash2Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: DFlash2Configuration) {
        numHeads = config.attentionHeads
        numKVHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)
        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// Rotated block queries and keys plus values, heads first.
    func projectBlock(
        _ x: MLXArray, rope: RoPELayer, position: MLXArray
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (b, l) = (x.dim(0), x.dim(1))
        let queries = qNorm(qProj(x).reshaped(b, l, numHeads, headDim)).transposed(0, 2, 1, 3)
        let keys = kNorm(kProj(x).reshaped(b, l, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(b, l, numKVHeads, headDim).transposed(0, 2, 1, 3)
        return (
            applyRotaryPosition(rope, to: queries, offset: .batch(position)),
            applyRotaryPosition(rope, to: keys, offset: .batch(position)),
            values
        )
    }

    /// Rotated context keys plus values for the cache, heads first.
    func projectContext(
        _ x: MLXArray, rope: RoPELayer, position: MLXArray
    ) -> (keys: MLXArray, values: MLXArray) {
        let (b, s) = (x.dim(0), x.dim(1))
        let keys = kNorm(kProj(x).reshaped(b, s, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(b, s, numKVHeads, headDim).transposed(0, 2, 1, 3)
        return (applyRotaryPosition(rope, to: keys, offset: .batch(position)), values)
    }

    func project(_ attention: MLXArray) -> MLXArray {
        let (b, l) = (attention.dim(0), attention.dim(2))
        return oProj(attention.transposed(0, 2, 1, 3).reshaped(b, l, numHeads * headDim))
    }
}

// MARK: - Decoder layer

/// Pre-norm layer, split at the attention so the SDPA over the growing context
/// runs between two traces. DFlash2 wraps each sublayer in a dynamic conv pair;
/// a v1 layer has none.
final class DFlash2DecoderLayer: Module {
    /// Whether block positions see only earlier block positions.
    let causalBlock: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: DFlash2Attention
    @ModuleInfo(key: "mlp") var mlp: DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlash2DynamicConv?
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlash2DynamicConv?

    init(_ config: DFlash2Configuration, index: Int) {
        causalBlock = config.attendsCausally(layer: index)
        _selfAttn.wrappedValue = DFlash2Attention(config)
        _mlp.wrappedValue = DFlash2MLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.isDFlash2 {
            _attentionConv.wrappedValue = DFlash2DynamicConv(
                hiddenSize: config.hiddenSize,
                kernelSize: config.dflash.convKernelSize,
                groupSize: config.dflash.convGroupSize)
            _mlpConv.wrappedValue = DFlash2DynamicConv(
                hiddenSize: config.hiddenSize,
                kernelSize: config.dflash.convKernelSize,
                groupSize: config.dflash.convGroupSize)
        }
        super.init()
    }

    /// Input norm, conv prepare where there is a conv, and the block projections.
    func preAttention(
        x: MLXArray, rope: RoPELayer, position: MLXArray
    ) -> (kernel: MLXArray?, queries: MLXArray, keys: MLXArray, values: MLXArray) {
        var h = inputLayerNorm(x)
        var kernel: MLXArray? = nil
        if let attentionConv {
            (h, kernel) = attentionConv.prepare(h)
        }
        let (queries, keys, values) = selfAttn.projectBlock(h, rope: rope, position: position)
        return (kernel, queries, keys, values)
    }

    /// Output projection, conv finish, both residual adds and the MLP.
    func postAttention(x: MLXArray, attention: MLXArray, kernel: MLXArray?) -> MLXArray {
        var r = selfAttn.project(attention)
        if let attentionConv, let kernel {
            r = attentionConv.finish(r, kernel: kernel)
        }
        let h = x + r
        var m = postAttentionLayerNorm(h)
        guard let mlpConv else { return h + mlp(m) }
        let mlpKernel: MLXArray
        (m, mlpKernel) = mlpConv.prepare(m)
        return h + mlpConv.finish(mlp(m), kernel: mlpKernel)
    }
}

final class DFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: DFlash2Configuration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Candidate selector

/// Scores adjacent pairs of top-K candidates with a low-rank bilinear match
/// gated by the successor's hidden state,
/// `S_t(a, b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>`, then walks the best path.
final class DFlash2CandidateSelector: Module {
    let topK: Int

    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    /// The greedy walk is about 60 small ops; one trace per block width.
    private let compiledGreedy = KeyedCompiledTraces<Int, DFlash2CandidateSelector>()

    init(_ config: DFlash2Configuration) {
        topK = config.dflash.selectorTopK
        _predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.dflash.selectorRank)
        _successorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.dflash.selectorRank)
        _hiddenProjection.wrappedValue = Linear(
            config.hiddenSize, config.dflash.selectorRank, bias: false)
        super.init()
    }

    /// - Parameters:
    ///   - hidden: normed block hidden states past the anchor, `[B, L, H]`.
    ///   - logits: matching draft logits, `[B, L, V]`.
    ///   - anchor: the verified token before the block, `[B]`.
    ///   - temperature: 0 walks greedily; above 0 samples each step from
    ///     `softmax(scores / temperature)` and returns those distributions.
    func select(
        hidden: MLXArray, logits: MLXArray, anchor: MLXArray, temperature: Float
    ) -> DFlash2Proposal {
        guard temperature <= 0 else {
            return walk(hidden: hidden, logits: logits, anchor: anchor, temperature: temperature)
        }
        let trace = compiledGreedy.trace(logits.dim(1)) {
            CompiledTrace<DFlash2CandidateSelector> { selector, args in
                let proposal = selector.walk(
                    hidden: args[0], logits: args[1], anchor: args[2], temperature: 0)
                return [proposal.tokens, proposal.candidates]
            }
        }
        let outputs = trace(self, [hidden, logits, anchor])
        return DFlash2Proposal(tokens: outputs[0], candidates: outputs[1])
    }

    private func walk(
        hidden: MLXArray, logits: MLXArray, anchor: MLXArray, temperature: Float
    ) -> DFlash2Proposal {
        let length = logits.dim(1)
        let kth = logits.dim(-1) - topK
        let candidates = argPartition(logits, kth: kth, axis: -1)[.ellipsis, kth...]
        let unary = takeAlong(logits, candidates, axis: -1)
        let projected = hiddenProjection(hidden)
        let successors = successorCodebook(candidates)
        let predecessors = predecessorCodebook(candidates)
        // Boundary 0 pairs the anchor with position 0; boundary t pairs the
        // candidates of t - 1 with those of t, gated by position t's hidden.
        let anchorEdges =
            ((predecessorCodebook(anchor) * projected[0..., 0])[0..., .newAxis, 0...]
            * successors[0..., 0, 0..., 0...]).sum(axis: -1)
        let gated =
            predecessors[0..., 0 ..< (length - 1), 0..., 0...]
            * projected[0..., 1..., .newAxis, 0...]
        let edges = matmul(gated, successors[0..., 1..., 0..., 0...].transposed(0, 1, 3, 2))

        var path: [MLXArray] = []
        var distributions: [MLXArray] = []
        var previous: MLXArray? = nil
        for position in 0 ..< length {
            var scores = unary[0..., position]
            if let previous {
                scores =
                    scores
                    + takeAlong(
                        edges[0..., position - 1, 0..., 0...],
                        previous[0..., .newAxis, .newAxis], axis: 1
                    )[0..., 0, 0...]
            } else {
                scores = scores + anchorEdges
            }
            let chosen: MLXArray
            if temperature > 0 {
                let distribution = softmax(scores.asType(.float32) / temperature, axis: -1)
                distributions.append(distribution)
                chosen = categorical(log(distribution))
            } else {
                chosen = argMax(scores, axis: -1)
            }
            previous = chosen
            path.append(
                takeAlong(candidates[0..., position], chosen[.newAxis, 0...], axis: -1)[0..., 0])
        }
        return DFlash2Proposal(
            tokens: stacked(path, axis: 1), candidates: candidates,
            probabilities: distributions.isEmpty ? nil : stacked(distributions, axis: 1))
    }
}

// MARK: - Draft model

/// The DFlash2 drafter, which also runs DFlash v1 checkpoints.
///
/// A proposal runs as compiled segments split at each layer's attention. The
/// block position and every accept-dependent value enter as arrays, so one
/// graph serves any accept outcome.
public final class DFlash2DraftModel: Module, DFlash2DrafterModel {
    public let config: DFlash2Configuration

    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [DFlash2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    /// DFlash2 only; a v1 drafter takes the argmax instead.
    @ModuleInfo(key: "candidate_selector") var candidateSelector: DFlash2CandidateSelector?

    let rope: RoPELayer

    /// Test hook: run the segment bodies eagerly instead of through traces.
    var compiledTracesEnabled = true
    /// Test hook, eager mode only: keep each layer's block output and the logits.
    var recordsIntermediates = false
    private(set) var lastLayerOutputs: [MLXArray] = []
    private(set) var lastLogits: MLXArray?

    public init(_ config: DFlash2Configuration) {
        self.config = config
        let concatDim = config.dflash.targetLayerIds.count * config.hiddenSize
        _fc.wrappedValue = Linear(concatDim, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { index in
            DFlash2DecoderLayer(config, index: index)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.isDFlash2 {
            _candidateSelector.wrappedValue = DFlash2CandidateSelector(config)
        }
        rope = initializeRope(
            dims: config.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
        super.init()
    }

    // MARK: DFlash2DrafterModel

    public var blockSize: Int { config.dflash.blockSize }
    public var maskTokenId: Int { config.dflash.maskTokenId }
    public var targetLayerIds: [Int] { config.dflash.targetLayerIds }
    public var contextWindow: Int { config.contextWindow }

    public var targetGeometry: DFlash2Geometry {
        DFlash2Geometry(
            layerCount: config.numTargetLayers, hiddenSize: config.hiddenSize,
            vocabularySize: config.vocabularySize)
    }

    public func makeState() -> DFlash2DrafterState {
        DFlash2DrafterState(
            contextCaches: layers.indices.map {
                DFlash2ContextCache(window: config.contextWindow(forLayer: $0))
            })
    }

    public func propose(
        _ request: DFlash2ProposalRequest, target: any DFlash2TargetModel,
        state: inout DFlash2DrafterState
    ) -> DFlash2Proposal {
        let block = request.block
        precondition(block.dim(0) == 1, "DFlash2 drafts one stream at a time")
        precondition(
            state.contextCaches.count == layers.count, "one context cache per drafter layer")
        let rows = request.targetHidden.dim(1)
        precondition(rows <= contextWindow, "targetHidden exceeds the drafter's context window")
        let width = block.dim(1)
        let contextPosition = MLXArray([Int32(request.contextPosition)])
        let blockPosition = contextPosition + request.validRows.asType(.int32).reshaped([1])

        // Context K/V for the new rows, appended as placeholders.
        let contextKV = projectContext(request.targetHidden, position: contextPosition)
        let positions = (0 ..< rows).map { Int32(request.contextPosition + $0) }
        var layerKV: [(MLXArray, MLXArray)] = []
        for (i, cache) in state.contextCaches.enumerated() {
            layerKV.append(
                cache.append(
                    keys: contextKV[2 * i], values: contextKV[2 * i + 1], positions: positions))
        }
        // The caches move in lockstep; layers of one kind share a mask.
        var masks: [Int: MLXArray] = [:]
        var maskByKind: [String: MLXArray] = [:]
        for index in layers.indices {
            let window = config.attentionWindow(forLayer: index)
            let kind = "\(window):\(layers[index].causalBlock)"
            if maskByKind[kind] == nil {
                maskByKind[kind] = Self.visibilityMask(
                    cache: state.contextCaches[index], appended: rows,
                    validRows: request.validRows, blockPosition: blockPosition, width: width,
                    window: window, causalBlock: layers[index].causalBlock)
            }
            masks[index] = maskByKind[kind]
        }

        // The target's embedding and head run outside the traces.
        var x = target.dflash2Embedding(block) * config.dflash.inputEmbeddingScale
        var attention: MLXArray? = nil
        var kernel: MLXArray? = nil
        if recordsIntermediates { lastLayerOutputs = [] }
        for index in 0 ... layers.count {
            var args = [x]
            if let attention {
                args.append(attention)
                if let kernel { args.append(kernel) }
            }
            args.append(blockPosition)
            let outputs =
                compiledTracesEnabled
                ? segmentTrace(index: index, width: width)(self, args)
                : segmentBody(at: index, args)
            x = outputs[0]
            guard index < layers.count else { break }
            kernel = config.isDFlash2 ? outputs[1] : nil
            let first = config.isDFlash2 ? 2 : 1
            let (contextKeys, contextValues) = layerKV[index]
            attention = MLXFast.scaledDotProductAttention(
                queries: outputs[first],
                keys: concatenated([contextKeys, outputs[first + 1]], axis: 2),
                values: concatenated([contextValues, outputs[first + 2]], axis: 2),
                scale: layers[index].selfAttn.scale, mask: masks[index]!)
        }

        let hidden = x[0..., 1..., 0...]
        var logits = target.dflash2Head?(hidden) ?? target.dflash2Embedding.asLinear(hidden)
        logits = logits * config.dflash.outputMultiplier
        if let cap = config.dflash.finalLogitSoftcapping, cap > 0 {
            logits = tanh(logits / cap) * cap
        }
        if let suppressor = request.suppressedTokens {
            logits = suppressor.process(logits: logits[0]).expandedDimensions(axis: 0)
        }
        if recordsIntermediates { lastLogits = logits }
        guard let candidateSelector else {
            // v1: the argmax at every masked position. A sampled round sees a
            // point mass on each draft, which rejection sampling accepts with
            // the target's probability of it.
            let tokens = argMax(logits, axis: -1).asType(.int32)
            return DFlash2Proposal(
                tokens: tokens, candidates: tokens.expandedDimensions(axis: -1),
                probabilities: request.temperature > 0
                    ? MLXArray.ones(tokens.shape + [1]) : nil)
        }
        return candidateSelector.select(
            hidden: hidden, logits: logits, anchor: block[0..., 0],
            temperature: request.temperature)
    }

    /// Bool mask `[width, cache.count + width]`. Committed context rows within the
    /// window are visible, placeholders never, and the block sees itself in full,
    /// or up to each position when `causalBlock`. `Int.max` is an unbounded window.
    static func visibilityMask(
        cache: DFlash2ContextCache, appended: Int, validRows: MLXArray,
        blockPosition: MLXArray, width: Int, window: Int, causalBlock: Bool = false
    ) -> MLXArray {
        let older = cache.count - appended
        let olderValid = MLXArray(Array(cache.rowValid[0 ..< older]))
        let appendedValid = MLXArray(Int32(0) ..< Int32(appended)) .< validRows.asType(.int32)
        let valid = concatenated([olderValid, appendedValid]).expandedDimensions(axis: 0)
        let contextVisible: MLXArray
        if window >= Int(Int32.max) {
            contextVisible = broadcast(valid, to: [width, cache.count])
        } else {
            let keyPositions = MLXArray(cache.rowPositions)
            let queryPositions =
                blockPosition.asType(.int32) + MLXArray(Int32(0) ..< Int32(width))
            let distance =
                queryPositions.expandedDimensions(axis: 1)
                - keyPositions.expandedDimensions(axis: 0)
            contextVisible = valid & (distance .< Int32(window))
        }
        let offsets = MLXArray(Int32(0) ..< Int32(width))
        let block =
            causalBlock
            ? offsets.expandedDimensions(axis: 0) .<= offsets.expandedDimensions(axis: 1)
            : MLXArray.ones([width, width], dtype: .bool)
        return concatenated([contextVisible, block], axis: 1)
    }

    // MARK: Compiled segments

    private struct SegmentKey: Hashable {
        var index: Int
        var width: Int
    }

    private let segmentTraces = KeyedCompiledTraces<SegmentKey, DFlash2DraftModel>()
    private let contextTraces = KeyedCompiledTraces<Int, DFlash2DraftModel>()

    /// The prompt window projects eagerly; a trace per prompt length would only churn.
    private static let maxTracedContextRows = 16

    /// Live traces, so tests can check that the compiled route ran.
    var compiledTraceCount: Int { segmentTraces.count + contextTraces.count }

    /// Segment `i` closes layer `i - 1` and opens layer `i`; the last segment
    /// applies the final norm. In: `[x, position]` for segment 0, else
    /// `[x, attention, position]` with the conv kernel before the position for
    /// DFlash2. Out: `[x, queries, keys, values]` with the kernel after `x` for
    /// DFlash2, or `[normed]` for the last.
    private func segmentBody(at index: Int, _ args: [MLXArray]) -> [MLXArray] {
        var x = args[0]
        if index > 0 {
            x = layers[index - 1].postAttention(
                x: x, attention: args[1], kernel: config.isDFlash2 ? args[2] : nil)
            if recordsIntermediates, !compiledTracesEnabled { lastLayerOutputs.append(x) }
        }
        guard index < layers.count else { return [norm(x)] }
        let (kernel, queries, keys, values) = layers[index].preAttention(
            x: x, rope: rope, position: args[args.count - 1])
        return [x] + (kernel.map { [$0] } ?? []) + [queries, keys, values]
    }

    private func segmentTrace(index: Int, width: Int) -> CompiledTrace<DFlash2DraftModel> {
        segmentTraces.trace(SegmentKey(index: index, width: width)) {
            CompiledTrace<DFlash2DraftModel> { model, args in model.segmentBody(at: index, args) }
        }
    }

    /// `[targetHidden, position]` to `[keys, values]` per layer.
    private func contextBody(_ args: [MLXArray]) -> [MLXArray] {
        let context = hiddenNorm(fc(args[0]))
        return layers.flatMap { layer in
            let (keys, values) = layer.selfAttn.projectContext(
                context, rope: rope, position: args[1])
            return [keys, values]
        }
    }

    private func projectContext(_ targetHidden: MLXArray, position: MLXArray) -> [MLXArray] {
        let rows = targetHidden.dim(1)
        guard compiledTracesEnabled, rows <= Self.maxTracedContextRows else {
            return contextBody([targetHidden, position])
        }
        let trace = contextTraces.trace(rows) {
            CompiledTrace<DFlash2DraftModel> { model, args in model.contextBody(args) }
        }
        return trace(self, [targetHidden, position])
    }

    /// The checkpoint stores the selector codebooks as bare tensors; `Embedding`
    /// expects a `.weight` suffix.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        for name in ["predecessor_codebook", "successor_codebook"] {
            let bare = "candidate_selector.\(name)"
            if let value = weights.removeValue(forKey: bare) {
                weights["\(bare).weight"] = value
            }
        }
        return weights
    }
}
