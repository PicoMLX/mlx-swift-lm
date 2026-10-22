// Copyright © 2026 Apple Inc.

// One rank of `Qwen35ShardedDecodeTests`. Two of these, started with MLX_RANK 0 and 1 and the
// same MLX_HOSTFILE, form a group over MLX's ring backend. Each shards the same seeded Qwen 3.5
// and decodes it through the general sharded path and through the traced and compiled paths,
// then prints what it found as JSON.

import Foundation
import MLX
@_spi(Testing) import MLXLLM
import MLXLMCommon
import MLXNN

let configuration = """
    {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "vocab_size": 64,
        "full_attention_interval": 2,
        "num_experts": 0,
        "num_experts_per_tok": 0
    }
    """

let group = try MLXDistributed.initialize(backend: .ring, strict: true)

func makeModel(quantized: Bool) throws -> Qwen35TextModel {
    MLXRandom.seed(11)
    let model = Qwen35TextModel(
        try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(configuration.utf8)))
    if quantized {
        // Small enough groups for each rank's half of the sharded-to-all projections.
        quantize(model: model, groupSize: 32, bits: 4)
    }
    eval(model)
    try model.shard(group: group)
    eval(model)
    return model
}

/// Last-position logits after a prompt and after each of eight decode steps, feeding each
/// step's argmax back.
func decode(quantized: Bool, tracedLayers: Bool, compiledStep: Bool) throws -> [MLXArray] {
    Qwen35TextModelInner.tracesShardedLayers = tracedLayers
    Qwen35TextModelInner.compilesShardedDecodeStep = compiledStep
    let model = try makeModel(quantized: quantized)
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

func maxDifference(_ a: [MLXArray], _ b: [MLXArray]) -> Float {
    zip(a, b).map { abs($0 - $1).max().item(Float.self) }.max() ?? .infinity
}

var report: [String: Any] = ["rank": group.rank, "size": group.size]
for quantized in [false, true] {
    let general = try decode(quantized: quantized, tracedLayers: false, compiledStep: false)
    let traced = try decode(quantized: quantized, tracedLayers: true, compiledStep: false)
    let compiled = try decode(quantized: quantized, tracedLayers: false, compiledStep: true)
    // Every rank ends with the same logits when each matches the mean over all of them.
    let mean = MLXDistributed.allSum(general.last!, group: group, stream: .cpu) / Float(group.size)
    eval(mean)
    // This rank's rows of the first MLP projection: half of the whole when sharded.
    let gateRows =
        try makeModel(quantized: quantized).parameters().flattened()
        .first { $0.0 == "model.layers.0.mlp.gate_proj.weight" }?.1.dim(0) ?? 0
    report[quantized ? "4-bit" : "float"] = [
        "tracedLayers": maxDifference(traced, general),
        "compiledStep": maxDifference(compiled, general),
        "acrossRanks": abs(general.last! - mean).max().item(Float.self),
        "tokens": general.map { argMax($0).item(Int.self) },
        "gateRows": gateRows,
    ]
}

let json = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
print(String(decoding: json, as: UTF8.self))
