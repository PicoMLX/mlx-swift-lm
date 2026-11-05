// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

let group = try MLXDistributed.initialize(backend: .ring, strict: true)
func makeModel(_ variant: String, quantized: Bool, moe: Bool = false) throws -> MiniCPM {
    var text: [String: Any] = [
        "model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "rms_norm_eps": 1e-6, "vocab_size": 64, "tie_word_embeddings": false,
        "rope_theta": 10_000_000,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
        "rope_parameters": [
            "type": "default", "rope_theta": 10_000_000, "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 2],
        ],
    ]
    if variant == "minicpmo" {
        text["model_type"] = "qwen3_vl_text"
        text["num_hidden_layers"] = 2
    }
    if moe {
        text["model_type"] = "qwen3_5_moe_text"
        text["num_experts"] = 4
        text["num_experts_per_tok"] = 2
        text["moe_intermediate_size"] = 64
        text["shared_expert_intermediate_size"] = 64
    }
    let json: [String: Any] = [
        "model_type": variant, "text_config": text, "query_num": 4, "audio_pool_step": 2,
        "insert_layer_id": 0, "downsample_mode": "16x", "image_start_id": 50, "image_end_id": 51,
        "slice_start_id": 52, "slice_end_id": 53, "newline_id": 54,
        "vision_config": [
            "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2,
            "num_hidden_layers": 2, "patch_size": 2, "image_size": 8,
        ],
        "audio_config": [
            "d_model": 8, "encoder_ffn_dim": 32, "encoder_attention_heads": 2,
            "encoder_layers": 1, "num_mel_bins": 8, "max_source_positions": 32,
        ],
    ]
    MLXRandom.seed(817)
    let config = try JSONDecoder().decode(
        MiniCPMConfiguration.self, from: JSONSerialization.data(withJSONObject: json))
    let model = MiniCPM(config)
    if quantized {
        quantize(
            model: model, groupSize: 32, bits: 4,
            filter: { path, _ in
                path.hasPrefix("language_model.model.layers.") && !path.contains("linear_attn")
            })
    }
    eval(model)
    return model
}
func run(_ model: MiniCPM) throws -> [MLXArray] {
    let batch = try model.newCache(parameters: nil).map {
        try mergeCaches(Array(repeating: $0, count: 2))
    }
    let cache = batch.map { $0 as any KVCache }
    for layer in batch { layer.prepare(lengths: [3, 5]) }
    let prefill = model(
        MLXArray([Int32(1), 2, 3, 0, 0, 4, 5, 6, 7, 8]).reshaped(2, 5), cache: cache)
    var result = [stacked([prefill[0, 2], prefill[1, 4]])]
    for layer in batch { layer.finalize() }
    for _ in 0 ..< 4 {
        let tokens = argMax(result.last!, axis: -1).reshaped(2, 1)
        result.append(model(tokens, cache: cache)[0..., 0, 0...])
    }
    for array in result { eval(array) }
    return result
}
var report: [String: Any] = ["rank": try group.rank, "size": try group.size]
for (variant, moe) in [
    ("minicpmo", false), ("minicpmv4_6", false), ("minicpmv4_7", false), ("minicpmv4_7", true),
] {
    for quantized in [false, true] {
        let model = try makeModel(variant, quantized: quantized, moe: moe)
        let expected = try run(model)
        let layer = variant == "minicpmo" ? 0 : 1
        let key = "language_model.model.layers.\(layer).self_attn.q_proj.weight"
        let wholeRows = model.parameters().flattened().first { $0.0 == key }!.1.dim(0)
        try model.shard(group: group)
        let actual = try run(model)
        let scale = expected.map { abs($0).max().item(Float.self) }.max() ?? 1
        let difference =
            zip(actual, expected).map { abs($0 - $1).max().item(Float.self) }.max() ?? .infinity
        let rows = model.parameters().flattened().first { $0.0 == key }!.1.dim(0)
        let label = variant + (moe ? ".moe" : "") + (quantized ? ".4-bit" : ".float")
        report[label] = [
            "relativeDifference": difference / max(scale, 1e-8), "queryRows": rows,
            "wholeQueryRows": wholeRows,
            "kvHeads": model.kvHeads,
            "tokens": actual.flatMap { argMax($0, axis: -1).asArray(Int.self) },
            "wholeTokens": expected.flatMap { argMax($0, axis: -1).asArray(Int.self) },
        ]
    }
}
print(
    String(
        decoding: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]),
        as: UTF8.self))
