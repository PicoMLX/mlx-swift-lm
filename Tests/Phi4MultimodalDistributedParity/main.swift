// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

let group = try MLXDistributed.initialize(backend: .ring, strict: true)

func makeModel(quantized: Bool) throws -> Phi4Multimodal {
    let json = """
        {"model_type":"phi4mm","hidden_size":64,"vocab_size":64,"num_hidden_layers":2,
         "intermediate_size":128,"num_attention_heads":4,"num_key_value_heads":2,"partial_rotary_factor":0.5,
         "max_position_embeddings":128,"original_max_position_embeddings":128,
         "vision_lora":{"r":4,"lora_alpha":4},"speech_lora":{"r":4,"lora_alpha":4},
         "vision_config":{"hidden_size":16,"intermediate_size":32,"num_attention_heads":2,"num_hidden_layers":2,"patch_size":2,"image_size":4},
         "audio_processor":{"config":{"attention_dim":8,"attention_heads":2,"num_blocks":1,"linear_units":16,"input_size":8,"time_reduction":2,"conv_channels":4,"ext_pw_out_channel":8,"depthwise_seperable_out_channel":8,"t5_bias_max_distance":3}}}
        """
    MLXRandom.seed(817)
    let configuration = try JSONDecoder().decode(
        Phi4MultimodalConfiguration.self, from: Data(json.utf8))
    let model = Phi4Multimodal(configuration)
    if quantized {
        quantize(
            model: model, groupSize: 32, bits: 4, filter: { path, _ in path.hasSuffix(".base") })
    }
    eval(model)
    return model
}

func decode(_ model: Phi4Multimodal, modes: [Float], media: Bool = false) throws -> [MLXArray] {
    let emptyRows = try (0 ..< 2).map { _ in try model.newCache(parameters: nil) }
    let batch = try emptyRows[0].indices.map { layer in
        try mergeCaches(emptyRows.map { $0[layer] })
    }
    let caches = batch.map { $0 as any KVCache }
    for cache in caches {
        guard let composite = cache as? CacheList, let state = composite[1] as? ArraysCache else {
            fatalError("Missing mode cache")
        }
        state[0] = MLXArray(modes, [2, 2])
    }
    let prefill: MLXArray
    let lengths: [Int]
    if media {
        lengths = [6, 5]
        let tokens = MLXArray([
            1, model.config.imageToken, model.config.imageToken,
            model.config.imageToken, model.config.imageToken, 2,
            3, model.config.audioToken, model.config.audioToken, model.config.audioToken, 4, 0,
        ]).reshaped(2, 6)
        let mask = MLXArray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]).reshaped(2, 6)
        let input = LMInput(
            text: .init(tokens: tokens, mask: mask),
            image: .init(
                pixels: sin(MLXArray(0 ..< 48).asType(.float32) * 0.13).reshaped(1, 4, 12),
                frames: [THW(1, 2, 2)]),
            audio: .init(
                features: cos(MLXArray(0 ..< 48).asType(.float32) * 0.17).reshaped(1, 6, 8)))
        guard
            case .logits(let output) = try model.prepare(
                input, cache: caches, state: nil,
                prefill: .init(chunking: .unchunked))
        else { fatalError("Expected media prefill logits") }
        prefill = output.logits
    } else {
        lengths = [3, 5]
        for cache in batch { cache.prepare(lengths: lengths) }
        prefill = model(
            MLXArray([Int32(1), 2, 3, 0, 0, 4, 5, 6, 7, 8]).reshaped(2, 5), cache: caches)
    }
    var result = [stacked([prefill[0, lengths[0] - 1], prefill[1, lengths[1] - 1]])]
    for cache in batch { cache.finalize() }
    for _ in 0 ..< 4 {
        let tokens = argMax(result.last!, axis: -1).reshaped(2, 1)
        result.append(model(tokens, cache: caches)[0..., 0, 0...])
    }
    for array in result { eval(array) }
    return result
}

var invalidConfiguration = try makeModel(quantized: false).config
invalidConfiguration.hiddenSize = 48
invalidConfiguration.heads = 3
invalidConfiguration.kvHeads = 3
invalidConfiguration.intermediateSize = 96
let invalidModel = Phi4Multimodal(invalidConfiguration)
let invalidTokens = MLXArray([Int32(1), 2]).reshaped(1, 2)
let beforeRefusal = invalidModel(invalidTokens, cache: nil)
eval(beforeRefusal)
var refused = false
do { try invalidModel.shard(group: group) } catch { refused = true }
precondition(refused, "An indivisible decoder must fail before any mutation")
precondition(
    abs(invalidModel(invalidTokens, cache: nil) - beforeRefusal).max().item(Float.self) == 0)

var report: [String: Any] = try ["rank": group.rank, "size": group.size]
for quantized in [false, true] {
    for media in [false, true] {
        let model = try makeModel(quantized: quantized)
        let expected = try decode(model, modes: [1, 0, 0, 1], media: media)
        try model.shard(group: group)
        let actual = try decode(model, modes: [1, 0, 0, 1], media: media)
        let scale = expected.map { abs($0).max().item(Float.self) }.max() ?? 1
        let difference =
            zip(actual, expected).map { abs($0 - $1).max().item(Float.self) }.max() ?? .infinity
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let rows = parameters["language_model.model.layers.0.mlp.gate_proj.base.weight"]!.dim(0)
        let keyRows = parameters["language_model.model.layers.0.self_attn.k_proj.base.weight"]!.dim(
            0)
        report["\(quantized ? "4-bit" : "float"),\(media ? "media" : "text")"] = [
            "relativeDifference": difference / max(scale, 1e-8),
            "tokens": actual.flatMap { argMax($0, axis: -1).asArray(Int.self) },
            "wholeTokens": expected.flatMap { argMax($0, axis: -1).asArray(Int.self) },
            "gateRows": rows, "keyRows": keyRows, "kvHeads": model.kvHeads,
        ]
    }
}
print(
    String(
        decoding: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]),
        as: UTF8.self))
