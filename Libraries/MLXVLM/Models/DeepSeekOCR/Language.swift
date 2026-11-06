// Port of Blaizzy/mlx-vlm/models/deepseekocr/language.py (MIT).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class OCRAttention: Module {
    let config: DeepSeekOCRConfiguration.Text
    private(set) var heads: Int
    private(set) var kvHeads: Int
    let scale: Float
    let rope: RoPELayer
    @ModuleInfo(key: "q_proj") var q: Linear?
    @ModuleInfo(key: "k_proj") var k: Linear?
    @ModuleInfo(key: "v_proj") var v: Linear?
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_a_proj") var qA: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qB: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvA: Linear?
    @ModuleInfo(key: "kv_a_layernorm") var kvNorm: RMSNorm?
    @ModuleInfo(key: "kv_b_proj") var kvB: Linear?

    init(_ config: DeepSeekOCRConfiguration.Text) {
        self.config = config
        heads = config.heads
        kvHeads = config.usesMLA ? config.heads : config.kvHeads
        let dim = config.usesMLA ? config.qkNopeHeadDim + config.qkRopeHeadDim : config.headDim
        var scale = pow(Float(dim), -0.5)
        if config.usesMLA {
            if let rank = config.qLoraRank {
                _qA.wrappedValue = Linear(config.hiddenSize, rank, bias: config.attentionBias)
                _qNorm.wrappedValue = RMSNorm(dimensions: rank, eps: 1e-6)
                _qB.wrappedValue = Linear(rank, config.heads * dim, bias: false)
            } else {
                _q.wrappedValue = Linear(config.hiddenSize, config.heads * dim, bias: false)
            }
            _kvA.wrappedValue = Linear(
                config.hiddenSize, config.kvLoraRank + config.qkRopeHeadDim,
                bias: config.attentionBias)
            _kvNorm.wrappedValue = RMSNorm(dimensions: config.kvLoraRank, eps: 1e-6)
            _kvB.wrappedValue = Linear(
                config.kvLoraRank, config.heads * (config.qkNopeHeadDim + config.vHeadDim),
                bias: false)
            _output.wrappedValue = Linear(
                config.heads * config.vHeadDim, config.hiddenSize, bias: config.attentionBias)
            if let scaling = config.ropeScaling {
                let factor = scaling["factor"]?.asFloat() ?? 1
                let mscale = scaling["mscale_all_dim"]?.asFloat() ?? 0
                if factor > 1, mscale != 0 {
                    let s = 0.1 * mscale * log(factor) + 1
                    scale *= s * s
                }
            }
        } else {
            _q.wrappedValue = Linear(
                config.hiddenSize, config.heads * dim, bias: config.attentionBias)
            _k.wrappedValue = Linear(
                config.hiddenSize, config.kvHeads * dim, bias: config.attentionBias)
            _v.wrappedValue = Linear(
                config.hiddenSize, config.kvHeads * dim, bias: config.attentionBias)
            _output.wrappedValue = Linear(
                config.heads * dim, config.hiddenSize, bias: config.attentionBias)
        }
        self.scale = scale
        rope = initializeRope(
            dims: config.usesMLA ? config.qkRopeHeadDim : dim, base: config.ropeTheta,
            traditional: config.ropeTraditional, scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let queries: MLXArray
        let keys: MLXArray
        let values: MLXArray
        let offset = cache?.ropeOffset
        if config.usesMLA {
            let query = qB.map { $0(qNorm!(qA!(x))) } ?? q!(x)
            let qParts = split(
                query.reshaped(batch, length, heads, -1).transposed(0, 2, 1, 3),
                indices: [config.qkNopeHeadDim], axis: -1)
            let compressed = split(kvA!(x), indices: [config.kvLoraRank], axis: -1)
            let kv = kvB!(kvNorm!(compressed[0])).reshaped(batch, length, heads, -1).transposed(
                0, 2, 1, 3)
            let kvParts = split(kv, indices: [config.qkNopeHeadDim], axis: -1)
            let qPe = applyRotaryPosition(rope, to: qParts[1], offset: offset)
            let kPe = applyRotaryPosition(
                rope,
                to: compressed[1].reshaped(batch, length, 1, config.qkRopeHeadDim).transposed(
                    0, 2, 1, 3), offset: offset)
            queries = concatenated([qParts[0], qPe], axis: -1)
            keys = concatenated([kvParts[0], repeated(kPe, count: heads, axis: 1)], axis: -1)
            values = kvParts[1]
        } else {
            queries = applyRotaryPosition(
                rope, to: q!(x).reshaped(batch, length, heads, -1).transposed(0, 2, 1, 3),
                offset: offset)
            keys = applyRotaryPosition(
                rope, to: k!(x).reshaped(batch, length, kvHeads, -1).transposed(0, 2, 1, 3),
                offset: offset)
            values = v!(x).reshaped(batch, length, kvHeads, -1).transposed(0, 2, 1, 3)
        }
        return output(
            attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
            ).transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }

    func shard(group: MLXDistributed.Group) throws {
        let size = try group.size
        if config.usesMLA {
            try shardLinears(
                allToSharded: [qB == nil ? "q_proj" : "q_b_proj", "kv_b_proj"],
                shardedToAll: ["o_proj"], group: group)
        } else {
            try shardLinears(
                allToSharded: ["q_proj", "k_proj", "v_proj"], shardedToAll: ["o_proj"], group: group
            )
        }
        heads /= size
        kvHeads /= size
    }
}

final class OCRMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(hidden: Int, intermediate: Int) {
        _gate.wrappedValue = Linear(hidden, intermediate, bias: false)
        _up.wrappedValue = Linear(hidden, intermediate, bias: false)
        _down.wrappedValue = Linear(intermediate, hidden, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class OCRRouter: Module {
    let config: DeepSeekOCRConfiguration.Text
    @ParameterInfo var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correction: MLXArray?
    init(_ config: DeepSeekOCRConfiguration.Text) {
        self.config = config
        _weight.wrappedValue = zeros([config.experts!, config.hiddenSize])
        if config.topKMethod == "noaux_tc" { _correction.wrappedValue = zeros([config.experts!]) }
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = matmul(x, weight.T)
        let scores =
            config.scoringFunction == "sigmoid"
            ? sigmoid(logits) : softmax(logits, axis: -1, precise: true)
        var selected = scores
        if let correction {
            selected = scores + correction.asType(.float32)
            let experts = config.experts!
            let perGroup = experts / config.groups
            let grouped = selected.reshaped(x.dim(0), x.dim(1), config.groups, perGroup)
            let groupScores = sorted(grouped, axis: -1)[0..., 0..., 0..., max(0, perGroup - 2)...]
                .sum(axis: -1)
            if config.topKGroups < config.groups {
                let groupIds = argPartition(
                    groupScores, kth: config.groups - config.topKGroups, axis: -1)[
                        0..., 0..., (config.groups - config.topKGroups)...]
                let axis = MLXArray(0 ..< config.groups).reshaped(1, 1, config.groups, 1)
                let mask = (axis .== groupIds.expandedDimensions(axis: -2)).any(axis: -1)
                selected = which(mask.expandedDimensions(axis: -1), grouped, zeros(like: grouped))
                    .reshaped(scores.shape)
            }
        }
        let count = config.experts!
        let ids = argPartition(selected, kth: count - config.expertsPerToken, axis: -1)[
            0..., 0..., (count - config.expertsPerToken)...]
        return (ids, takeAlong(scores, ids, axis: -1) * config.routingScale)
    }
}

final class OCRMoE: Module, UnaryLayer {
    let gate: OCRRouter
    @ModuleInfo(key: "switch_mlp") var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var shared: OCRMLP?
    private var shardingGroup: MLXDistributed.Group?
    init(_ config: DeepSeekOCRConfiguration.Text) {
        gate = OCRRouter(config)
        _experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize,
            numExperts: config.experts!)
        if let count = config.sharedExperts, count > 0 {
            _shared.wrappedValue = OCRMLP(
                hidden: config.hiddenSize, intermediate: count * config.moeIntermediateSize)
        }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = weightedExpertSum(experts(x, indices), scores)
        if let shared { y = y + shared(x) }
        return shardingGroup.map { MLXDistributed.allSum(y, group: $0) } ?? y
    }
    func shard(group: MLXDistributed.Group) throws {
        try shardInPlace(experts, predicate: ocrExpertSharding, group: group)
        if let shared { try shardInPlace(shared, predicate: ocrExpertSharding, group: group) }
        shardingGroup = group
    }
}

private func ocrExpertSharding(_ path: String, _ weight: MLXArray) -> (
    axis: Int, segments: Segments
)? {
    if path.hasPrefix("down_proj.") {
        return path.hasSuffix(".bias") ? nil : (-1, .count(1))
    }
    return path.hasSuffix(".bias") ? (-1, .count(1)) : (weight.ndim - 2, .count(1))
}

final class OCRDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: OCRAttention
    let mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    init(_ config: DeepSeekOCRConfiguration.Text, index: Int) {
        _attention.wrappedValue = OCRAttention(config)
        if config.experts != nil, index >= config.firstDenseLayers, index % config.moeFrequency == 0
        {
            mlp = OCRMoE(config)
        } else {
            mlp = OCRMLP(hidden: config.hiddenSize, intermediate: config.intermediateSize)
        }
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputNorm(x), mask: mask, cache: cache)
        return h + mlp(postNorm(h))
    }
    func shard(group: MLXDistributed.Group) throws {
        try attention.shard(group: group)
        if let moe = mlp as? OCRMoE {
            try moe.shard(group: group)
        } else {
            try mlp.shardLinears(
                allToSharded: ["gate_proj", "up_proj"], shardedToAll: ["down_proj"], group: group)
        }
    }
}

final class OCRDecoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [OCRDecoderLayer]
    let norm: RMSNorm
    init(_ config: DeepSeekOCRConfiguration.Text) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = (0 ..< config.layers).map { OCRDecoderLayer(config, index: $0) }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }
    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (index, layer) in layers.enumerated() { h = layer(h, mask: mask, cache: cache?[index]) }
        return norm(h)
    }
}

final class OCRLanguageModel: Module {
    let model: OCRDecoder
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    init(_ config: DeepSeekOCRConfiguration.Text) {
        model = OCRDecoder(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }
    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(embeddings, cache: cache))
    }
}
