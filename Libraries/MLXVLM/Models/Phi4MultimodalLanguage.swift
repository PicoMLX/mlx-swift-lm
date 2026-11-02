// Copyright © 2026 Apple Inc.
// Port of Blaizzy/mlx-vlm phi4mm at 4693cd0 (MIT).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The mode belongs to a cached row, never to the shared model's weights.
/// Columns are vision and speech; a combined prompt enables both adapters.
package func phi4Modes(_ cache: (any KVCache)?, batch: Int) -> MLXArray {
    if let list = cache as? CacheList, let modes = (list[1] as? ArraysCache)?[0] {
        return broadcast(modes, to: [batch, 2])
    }
    return broadcast(MLXArray([Float(1), 0]).reshaped(1, 2), to: [batch, 2])
}

package func phi4AttentionCache(_ cache: (any KVCache)?) -> (any KVCache)? {
    (cache as? CacheList).map { $0[0] } ?? cache
}

package final class Phi4MultimodalProjection: Module {
    @ModuleInfo var base: Linear
    @ModuleInfo(key: "vision_a") var visionA: Linear?
    @ModuleInfo(key: "vision_b") var visionB: Linear?
    @ModuleInfo(key: "speech_a") var speechA: Linear?
    @ModuleInfo(key: "speech_b") var speechB: Linear?
    let visionScale: Float
    let speechScale: Float

    init(_ input: Int, _ output: Int, source: String, config: Phi4MultimodalConfiguration) {
        _base.wrappedValue = Linear(input, output, bias: false)
        func active(_ adapter: Phi4MultimodalAdapterConfiguration?) -> Bool {
            guard let adapter, adapter.rank > 0 else { return false }
            return adapter.targets.isEmpty || adapter.targets.contains(source)
        }
        let vision = config.visionAdapter
        let speech = config.speechAdapter
        visionScale = vision.map { $0.alpha / Float(max($0.rank, 1)) } ?? 1
        speechScale = speech.map { $0.alpha / Float(max($0.rank, 1)) } ?? 1
        if active(vision), let vision {
            _visionA.wrappedValue = Linear(input, vision.rank, bias: false)
            _visionB.wrappedValue = Linear(vision.rank, output, bias: false)
        }
        if active(speech), let speech {
            _speechA.wrappedValue = Linear(input, speech.rank, bias: false)
            _speechB.wrappedValue = Linear(speech.rank, output, bias: false)
        }
    }

    func callAsFunction(_ x: MLXArray, modes: MLXArray) -> MLXArray {
        var output = base(x)
        if let a = visionA, let b = visionB {
            output =
                output + b(a(x)) * visionScale * modes[0..., 0 ..< 1, .newAxis].asType(output.dtype)
        }
        if let a = speechA, let b = speechB {
            output =
                output + b(a(x)) * speechScale * modes[0..., 1 ..< 2, .newAxis].asType(output.dtype)
        }
        return output
    }

    func shard(column: Bool, group: MLXDistributed.Group) throws {
        if column {
            var output = ["base"]
            if visionB != nil { output.append("vision_b") }
            if speechB != nil { output.append("speech_b") }
            try shardLinears(allToSharded: output, group: group)
        } else {
            var input = ["base"]
            if visionA != nil { input.append("vision_a") }
            if speechA != nil { input.append("speech_a") }
            try shardLinears(shardedToAll: input, group: group)
        }
    }

    func removeAbsentAdapters(prefix: String, weights: [String: MLXArray]) {
        if weights[prefix + ".vision_a.weight"] == nil {
            visionA = nil
            visionB = nil
        }
        if weights[prefix + ".speech_a.weight"] == nil {
            speechA = nil
            speechB = nil
        }
    }
}

package final class Phi4MultimodalAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Phi4MultimodalProjection
    @ModuleInfo(key: "k_proj") var key: Phi4MultimodalProjection
    @ModuleInfo(key: "v_proj") var value: Phi4MultimodalProjection
    @ModuleInfo(key: "o_proj") var output: Phi4MultimodalProjection
    let headDimension: Int
    let scale: Float
    let rope: any RoPELayer

    init(_ c: Phi4MultimodalConfiguration) {
        _query.wrappedValue = .init(
            c.hiddenSize, c.heads * c.headDimension, source: "qkv_proj", config: c)
        _key.wrappedValue = .init(
            c.hiddenSize, c.kvHeads * c.headDimension, source: "qkv_proj", config: c)
        _value.wrappedValue = .init(
            c.hiddenSize, c.kvHeads * c.headDimension, source: "qkv_proj", config: c)
        _output.wrappedValue = .init(
            c.heads * c.headDimension, c.hiddenSize, source: "o_proj", config: c)
        headDimension = c.headDimension
        scale = pow(Float(headDimension), -0.5)
        let dimensions = Int(Float(headDimension) * c.partialRotaryFactor)
        if let r = c.ropeScaling, r.type == "su" || r.type == "longrope",
            let short = r.shortFactor, let long = r.longFactor
        {
            rope = SuScaledRoPE(
                dimensions: dimensions, base: c.ropeTheta,
                maxPositionEmbeddings: c.maxPositionEmbeddings,
                originalMaxPositionEmbeddings: c.originalMaxPositionEmbeddings,
                shortFactor: short, longFactor: long)
        } else {
            let ropeScale: Float =
                c.ropeScaling?.type == "linear" ? 1 / (c.ropeScaling?.factor ?? 1) : 1
            rope = RoPE(
                dimensions: dimensions, traditional: c.ropeTraditional, base: c.ropeTheta,
                scale: ropeScale)
        }
    }

    func callAsFunction(_ x: MLXArray, cache: (any KVCache)?, modes: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        var q = query(x, modes: modes).reshaped(batch, length, -1, headDimension).transposed(
            0, 2, 1, 3)
        var k = key(x, modes: modes).reshaped(batch, length, -1, headDimension).transposed(
            0, 2, 1, 3)
        let v = value(x, modes: modes).reshaped(batch, length, -1, headDimension).transposed(
            0, 2, 1, 3)
        let offset = cache?.ropeOffset
        q = applyRotaryPosition(rope, to: q, offset: offset)
        k = applyRotaryPosition(rope, to: k, offset: offset)
        let mask = createAttentionMask(h: x, cache: cache)
        let h = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return output(h, modes: modes)
    }
}

package final class Phi4MultimodalMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Phi4MultimodalProjection
    @ModuleInfo(key: "up_proj") var up: Phi4MultimodalProjection
    @ModuleInfo(key: "down_proj") var down: Phi4MultimodalProjection
    init(_ c: Phi4MultimodalConfiguration) {
        _gate.wrappedValue = .init(
            c.hiddenSize, c.intermediateSize, source: "gate_up_proj", config: c)
        _up.wrappedValue = .init(
            c.hiddenSize, c.intermediateSize, source: "gate_up_proj", config: c)
        _down.wrappedValue = .init(c.intermediateSize, c.hiddenSize, source: "down_proj", config: c)
    }
    func callAsFunction(_ x: MLXArray, modes: MLXArray) -> MLXArray {
        down(silu(gate(x, modes: modes)) * up(x, modes: modes), modes: modes)
    }
}

package final class Phi4MultimodalBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Phi4MultimodalAttention
    let mlp: Phi4MultimodalMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    init(_ c: Phi4MultimodalConfiguration) {
        _attention.wrappedValue = .init(c)
        mlp = .init(c)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.eps)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.eps)
    }
    func callAsFunction(_ x: MLXArray, cache: (any KVCache)?, modes: MLXArray) -> MLXArray {
        let h = x + attention(inputNorm(x), cache: cache, modes: modes)
        return h + mlp(postNorm(h), modes: modes)
    }
}

package final class Phi4MultimodalText: Module {
    @ModuleInfo(key: "embed_tokens") var embedding: Embedding
    let layers: [Phi4MultimodalBlock]
    let norm: RMSNorm
    init(_ c: Phi4MultimodalConfiguration) {
        _embedding.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        layers = (0 ..< c.layers).map { _ in .init(c) }
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.eps)
    }
    func callAsFunction(_ embeddings: MLXArray, cache: [any KVCache]?) -> MLXArray {
        var h = embeddings
        for (i, layer) in layers.enumerated() {
            let composite = cache?[i]
            let attention = phi4AttentionCache(composite)
            let modes = phi4Modes(composite, batch: h.dim(0))
            if let list = composite as? CacheList, let state = list[1] as? ArraysCache,
                state[0] == nil
            {
                state[0] = modes
            }
            h = layer(h, cache: attention, modes: modes)
            if let composite = composite as? BaseKVCache, let attention {
                composite.offset = attention.offset
            }
        }
        return norm(h)
    }
}

package final class Phi4MultimodalLanguage: Module {
    let model: Phi4MultimodalText
    @ModuleInfo(key: "lm_head") var head: Linear?
    init(_ c: Phi4MultimodalConfiguration) {
        model = .init(c)
        if !c.tieWordEmbeddings {
            _head.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false)
        }
    }
    func callAsFunction(_ embeddings: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let h = model(embeddings, cache: cache)
        return head.map { $0(h) } ?? model.embedding.asLinear(h)
    }
}
