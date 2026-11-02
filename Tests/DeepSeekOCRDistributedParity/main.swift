import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

let group = try MLXDistributed.initialize(backend: .ring, strict: true)

func makeModel(ocr2: Bool, quantized: Bool, sharded: Bool) throws -> DeepSeekOCR {
    MLXRandom.seed(81)
    let json = """
        {
          "model_type": "\(ocr2 ? "deepseekocr_2" : "deepseekocr")", "image_token_index": 3,
          "text_config": {"hidden_size":64,"vocab_size":64,"num_hidden_layers":2,
            "num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":128,
            "n_routed_experts":4,"n_shared_experts":1,"num_experts_per_tok":2,"moe_intermediate_size":64},
          "vision_config": {"hidden_size":8,"layers":1,"num_attention_heads":2,"intermediate_size":16,
            "query_global_count":4,"query_local_count":1,
            "params":{"qwen2":{"dim":8,"layers":1,"heads":2,"kv_heads":1,"intermediate_size":16}}},
          "sam_config":{"image_size":16,"patch_size":2,"width":8,"layers":1,"heads":2,
            "window_size":4,"global_attn_indexes":[],"neck_channels":8,"downsample_channels":8},
          "projector_config":{"input_dim":\(ocr2 ? 8 : 16),"n_embed":64,"projector_type":"linear"}
        }
        """
    let config = try JSONDecoder().decode(DeepSeekOCRConfiguration.self, from: Data(json.utf8))
    try config.validateModelConfiguration()
    let model = DeepSeekOCR(config)
    if quantized {
        quantize(
            model: model, groupSize: 32, bits: 4,
            filter: { path, _ in path.hasPrefix("language_model.") })
    }
    eval(model)
    if sharded {
        try model.shard(group: group)
        eval(model)
    }
    return model
}

func outputs(_ model: DeepSeekOCR, ocr2: Bool) throws -> [MLXArray] {
    let imageCount = ocr2 ? 5 : 7
    let input = LMInput(
        text: .init(
            tokens: MLXArray([1] + Array(repeating: 3, count: imageCount) + [2, 4]).reshaped(1, -1)),
        image: .init(
            pixels: MLXArray(0 ..< 768).asType(.float32).reshaped(1, 3, 16, 16) / 768,
            frames: [THW(0, 0, 0)]))
    let mediaCache = try model.newCache(parameters: nil)
    let prepared = try model.prepare(
        input, cache: mediaCache, state: nil, prefill: .init(stepSize: 3))
    guard case .logits(let prepared) = prepared else { fatalError("Expected OCR media prefill") }
    var results = [prepared.logits[0, -1].asType(.float32)]
    let textCache = try model.newCache(parameters: nil)
    results.append(
        model(MLXArray([6, 7, 8]).reshaped(1, -1), cache: textCache)[0, -1].asType(.float32))
    let batch = mediaCache.map { BatchKVCache.fromSingle($0 as! KVCacheSimple) }
    for index in batch.indices {
        batch[index].extend(other: BatchKVCache.fromSingle(textCache[index] as! KVCacheSimple))
    }
    for step in 0 ..< 4 {
        let output = model(
            MLXArray([9 + step, 17 + step]).reshaped(2, 1), cache: batch.map { $0 as any KVCache }
        ).asType(.float32)
        results.append(output.reshaped(-1))
        eval(output)
    }
    batch.forEach { $0.filter(batchIndices: MLXArray([1, 0] as [Int32])) }
    results.append(
        model(MLXArray([23, 24]).reshaped(2, 1), cache: batch.map { $0 as any KVCache }).asType(
            .float32
        ).reshaped(-1))
    eval(results)
    return results
}

var report: [String: Any] = try ["rank": group.rank, "size": group.size]
for ocr2 in [false, true] {
    for quantized in [false, true] {
        let whole = try outputs(
            makeModel(ocr2: ocr2, quantized: quantized, sharded: false), ocr2: ocr2)
        let shardedModel = try makeModel(ocr2: ocr2, quantized: quantized, sharded: true)
        let sharded = try outputs(shardedModel, ocr2: ocr2)
        let parameters = Dictionary(uniqueKeysWithValues: shardedModel.parameters().flattened())
        let qRows = parameters["language_model.model.layers.0.self_attn.q_proj.weight"]?.dim(0) ?? 0
        let expertRows =
            parameters["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.dim(1) ?? 0
        let scale = max(whole.map { abs($0).max().item(Float.self) }.max() ?? 1, 1e-8)
        let error =
            zip(whole, sharded).map { abs($0 - $1).max().item(Float.self) }.max() ?? .infinity
        report["\(ocr2 ? "ocr2" : "ocr1"), \(quantized ? "4-bit" : "float")"] = [
            "relativeDifference": error / scale,
            "tokens": sharded.map { argMax($0).item(Int.self) },
            "wholeTokens": whole.map { argMax($0).item(Int.self) },
            "qRows": qRows, "expertRows": expertRows, "kvHeads": shardedModel.kvHeads,
        ]
    }
}
let json = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
print(String(decoding: json, as: UTF8.self))
