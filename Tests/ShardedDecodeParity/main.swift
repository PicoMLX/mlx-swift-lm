// Copyright © 2026 Apple Inc.

// One rank of `ShardedDecodeParityTests`. Two of these, started with MLX_RANK 0 and 1 and
// the same MLX_HOSTFILE, form a group over MLX's ring backend. Each decodes the same seeded
// Qwen 3.5, whole and sharded, and prints what it found as JSON.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

let group = try MLXDistributed.initialize(backend: .ring, strict: true)

func makeModel(experts: Int, quantized: Bool, sharded: Bool) throws -> Qwen35TextModel {
    MLXRandom.seed(11)
    let model = Qwen35TextModel(
        try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                """
                {
                    "model_type": "qwen3_5_text", "hidden_size": 64, "num_hidden_layers": 4,
                    "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
                    "head_dim": 16, "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                    "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                    "linear_conv_kernel_dim": 4, "vocab_size": 64, "full_attention_interval": 2,
                    "num_experts": \(experts), "num_experts_per_tok": \(min(experts, 2)),
                    "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64
                }
                """.utf8)))
    if quantized {
        // Small enough groups for each rank's half of the sharded-to-all projections.
        quantize(model: model, groupSize: 32, bits: 4)
    }
    eval(model)
    if sharded {
        try model.shard(group: group)
        eval(model)
    }
    return model
}

/// Last-position logits after a prompt and after each of eight decode steps, feeding each
/// step's argmax back.
func decode(_ model: Qwen35TextModel) throws -> [MLXArray] {
    let cache = try model.newCache(parameters: nil)
    var tokens: [Int32] = [3, 17, 42, 8, 29, 51]
    var outputs: [MLXArray] = []
    for _ in 0 ..< 9 {
        let logits = model(MLXArray(tokens).reshaped(1, -1), cache: cache)[0, -1].asType(.float32)
        eval(logits)
        outputs.append(logits)
        tokens = [Int32(argMax(logits).item(Int.self))]
    }
    return outputs
}

var report: [String: Any] = try ["rank": group.rank, "size": group.size]
for experts in [0, 4] {
    for quantized in [false, true] {
        let whole = try decode(makeModel(experts: experts, quantized: quantized, sharded: false))
        let shardedModel = try makeModel(experts: experts, quantized: quantized, sharded: true)
        let sharded = try decode(shardedModel)
        // This rank's rows of the first layer's gate projection: half of the whole.
        let gateKey =
            experts > 0
            ? "model.layers.0.mlp.switch_mlp.gate_proj.weight"
            : "model.layers.0.mlp.gate_proj.weight"
        let gateRows =
            shardedModel.parameters().flattened().first { $0.0 == gateKey }?
            .1.dim(experts > 0 ? 1 : 0) ?? 0
        // Split matmuls sum in another order, so compare relative to the logits' scale.
        let scale = whole.map { abs($0).max().item(Float.self) }.max() ?? 1
        let difference =
            zip(sharded, whole).map { abs($0 - $1).max().item(Float.self) }.max() ?? .infinity
        report["\(experts > 0 ? "moe" : "dense"), \(quantized ? "4-bit" : "float")"] = [
            "relativeDifference": difference / scale,
            "tokens": sharded.map { argMax($0).item(Int.self) },
            "wholeTokens": whole.map { argMax($0).item(Int.self) },
            "gateRows": gateRows,
        ]
    }
}

let json = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
print(String(decoding: json, as: UTF8.self))
