// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

let group = try MLXDistributed.initialize(backend: .ring, strict: true)

func makeModel(sparse: Bool, quantized: Bool, sharded: Bool) throws -> Gemma3nTextModel {
    let configuration = try JSONDecoder().decode(
        Gemma3nTextConfiguration.self,
        from: Data(
            """
            {"model_type":"gemma3n","hidden_size":32,"num_hidden_layers":4,
             "intermediate_size":[128,128,128,128],"num_attention_heads":4,"head_dim":16,
             "rms_norm_eps":0.000001,"vocab_size":64,"num_key_value_heads":2,
             "num_kv_shared_layers":2,"vocab_size_per_layer_input":48,"sliding_window":4,
             "max_position_embeddings":128,"rope_local_base_freq":10000,"rope_theta":1000000,
             "final_logit_softcapping":30,
             "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
             "activation_sparsity_pattern":\(sparse ? "[0.5,0.5,0,0]" : "[0,0,0,0]"),
             "hidden_size_per_layer_input":8,"altup_num_inputs":2,"altup_coef_clip":120,
             "altup_correct_scale":true,"altup_active_idx":0,"laurel_rank":8}
            """.utf8))
    MLXRandom.seed(719)
    let model = Gemma3nTextModel(config: configuration)
    if quantized {
        quantize(
            model: model, groupSize: 32, bits: 4,
            filter: { _, module in
                if let linear = module as? Linear { return linear.weight.dim(-1) % 32 == 0 }
                if let embedding = module as? Embedding {
                    return embedding.weight.dim(-1) % 32 == 0
                }
                return false
            })
    }
    eval(model)
    if sharded { try model.shard(group: group) }
    return model
}

func mediaPrefill(_ model: Gemma3nTextModel, tokens: MLXArray, cache: [KVCache]) -> MLXArray {
    let embeddings = model.languageModel.embeddings(tokens)
    let flat = embeddings.reshaped(-1, embeddings.dim(-1))
    let positions = tokens.flattened().asArray(Int.self).enumerated().compactMap {
        $0.element == 50 ? Int32($0.offset) : nil
    }
    let features = sin(MLXArray(0 ..< positions.count * embeddings.dim(-1)).asType(.float32) * 0.1)
        .reshaped(positions.count, embeddings.dim(-1))
    flat[MLXArray(positions)] = features
    return model.languageModel(
        inputsEmbeds: flat.reshaped(embeddings.shape),
        cache: cache.map { Optional($0) },
        perLayerInputs: model.languageModel.getPerLayerInputs(tokens))
}

func decode(_ model: Gemma3nTextModel, batched: Bool, media: Bool) throws -> [MLXArray] {
    if !batched {
        let cache = try model.newCache(parameters: nil)
        var tokens: [Int32] = media ? [3, 50, 50, 8, 29, 11, 5] : [3, 17, 42, 8, 29, 11, 5]
        var outputs: [MLXArray] = []
        for step in 0 ..< 9 {
            let input = MLXArray(tokens).reshaped(1, -1)
            let output =
                media && step == 0
                ? mediaPrefill(model, tokens: input, cache: cache)
                : model(input, cache: cache)
            let logits = output[0, -1].asType(.float32)
            eval(logits)
            outputs.append(logits)
            tokens = [Int32(argMax(logits).item(Int.self))]
        }
        return outputs
    }
    let factories = try makeBatchedCacheFactories(for: model.newCache(parameters: nil))
    let batch = factories.map { $0([0, 0]) }
    batch.forEach { $0.prepare(lengths: [3, 7]) }
    let cache = batch.map { $0 as any KVCache }
    let row: [Int32] = media ? [1, 50, 50, 0, 0, 0, 0] : [1, 2, 3, 0, 0, 0, 0]
    let prompt = MLXArray(row + [4, 5, 6, 7, 8, 9, 10]).reshaped(2, 7)
    let prefill =
        media ? mediaPrefill(model, tokens: prompt, cache: cache) : model(prompt, cache: cache)
    batch.forEach { $0.finalize() }
    var logits = stacked([prefill[0, 2], prefill[1, 6]]).asType(.float32)
    var outputs = [logits]
    for _ in 0 ..< 8 {
        let tokens = argMax(logits, axis: -1).reshaped(2, 1)
        logits = model(tokens, cache: cache)[0..., 0, 0...].asType(.float32)
        eval(logits)
        outputs.append(logits)
    }
    return outputs
}

var report: [String: Any] = try ["rank": group.rank, "size": group.size]
try Device.withDefaultDevice(.cpu) {
    for sparse in [false, true] {
        for quantized in [false, true] {
            for batched in [false, true] {
                for media in [false, true] {
                    let whole = try decode(
                        makeModel(sparse: sparse, quantized: quantized, sharded: false),
                        batched: batched, media: media)
                    let model = try makeModel(sparse: sparse, quantized: quantized, sharded: true)
                    let sharded = try decode(model, batched: batched, media: media)
                    let parameters = Dictionary(
                        uniqueKeysWithValues: model.parameters().flattened())
                    let qRows =
                        parameters["language_model.layers.0.self_attn.q_proj.weight"]?.dim(0) ?? 0
                    let gateRows =
                        parameters["language_model.layers.0.mlp.gate_proj.weight"]?.dim(0) ?? 0
                    let scale = max(whole.map { abs($0).max().item(Float.self) }.max() ?? 1, 1e-6)
                    let difference =
                        zip(whole, sharded).map { abs($0 - $1).max().item(Float.self) }.max()
                        ?? .infinity
                    let key =
                        "\(sparse ? "sparse" : "dense"),\(quantized ? "4bit" : "float"),\(batched ? "batch" : "solo"),\(media ? "media" : "text")"
                    report[key] = [
                        "relativeDifference": difference / scale,
                        "kvHeads": model.kvHeads,
                        "tokens": sharded.flatMap {
                            argMax($0, axis: -1).flattened().asArray(Int.self)
                        },
                        "wholeTokens": whole.flatMap {
                            argMax($0, axis: -1).flattened().asArray(Int.self)
                        },
                        "qRows": qRows, "gateRows": gateRows,
                    ]
                }
            }
        }
    }
}
let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
print(String(decoding: data, as: UTF8.self))
