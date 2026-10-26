// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXVLM
import Testing

@testable import MLXLMCommon

/// An image row and a text row prefilled separately, then decoded as one batch, must each
/// get the logits they get on their own.
@Suite(.serialized)
struct VLMRaggedDecodeTests {

    @Test("FastVLM decodes an image row and a text row with per-row positions")
    func fastVLM() throws {
        try Device.withDefaultDevice(.cpu) {
            let configuration = try JSONDecoder().decode(
                FastVLMConfiguration.self,
                from: Data(
                    """
                    {
                      "model_type": "fastvlm", "hidden_size": 16, "num_hidden_layers": 1,
                      "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
                      "vocab_size": 16, "eos_token_id": 15, "mm_projector_type": "mlp2x_gelu",
                      "mm_hidden_size": 16, "tokenizer_model_max_length": 1024,
                      "tokenizer_padding_side": "right",
                      "vision_config": {
                        "cls_ratio": 1, "down_patch_size": 3, "down_stride": 2,
                        "downsamples": [false, false], "embed_dims": [16, 16],
                        "hidden_size": 16, "image_size": 128, "intermediate_size": 32,
                        "layers": [1, 1], "layer_scale_init_value": 0.00001,
                        "mlp_ratios": [2, 2], "num_classes": 16, "patch_size": 99,
                        "pos_embs_shapes": [null, null], "projection_dim": 16,
                        "repmixer_kernel_size": 3, "token_mixers": ["repmixer", "repmixer"]
                      }
                    }
                    """.utf8))
            let model = withRandomState(MLXRandom.RandomState(seed: 501)) {
                let model = FastVLM(configuration)
                eval(model)
                return model
            }
            let inputs = [
                LMInput(
                    text: .init(tokens: MLXArray([1, -200, 2]).reshaped([1, 3])),
                    image: .init(pixels: MLXArray.ones([1, 3, 16, 16]), frames: [])),
                LMInput(tokens: MLXArray([1, 2, 3]).reshaped([1, 3])),
            ]
            try expectRaggedDecode(model, inputs: inputs, positions: [18, 3])
        }
    }

    @Test("Gemma 3 decodes an image row and a text row across its sliding window")
    func gemma3() throws {
        try Device.withDefaultDevice(.cpu) {
            let configuration = try JSONDecoder().decode(
                Gemma3Configuration.self,
                from: Data(
                    """
                    {
                      "model_type": "gemma3", "mm_tokens_per_image": 4,
                      "text_config": {
                        "model_type": "gemma3_text", "hidden_size": 16, "num_hidden_layers": 6,
                        "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
                        "head_dim": 8, "query_pre_attn_scalar": 8, "sliding_window": 8
                      },
                      "vision_config": {
                        "model_type": "siglip_vision_model", "num_hidden_layers": 1,
                        "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2,
                        "patch_size": 2, "image_size": 8
                      }
                    }
                    """.utf8))
            let model = withRandomState(MLXRandom.RandomState(seed: 502)) {
                let model = Gemma3(configuration)
                eval(model)
                return model
            }
            let inputs = [
                LMInput(
                    text: .init(
                        tokens: MLXArray([1, 262144, 262144, 262144, 262144, 2])
                            .reshaped([1, 6])),
                    image: .init(pixels: MLXArray.ones([1, 3, 8, 8]), frames: [])),
                LMInput(tokens: MLXArray([1, 2, 3]).reshaped([1, 3])),
            ]
            try expectRaggedDecode(model, inputs: inputs, positions: [6, 3])
        }
    }
}

/// Prefills each input alone, batches the caches, decodes three tokens and drops row 0,
/// comparing logits and retained keys and values with the separate runs.
private func expectRaggedDecode(
    _ model: any LanguageModel, inputs: [LMInput], positions: [Int]
) throws {
    func prefill(_ input: LMInput) throws -> [any KVCache] {
        let cache = try model.newCache(parameters: nil)
        switch try model.prepare(input, cache: cache, state: nil, prefill: .init()) {
        case .logits(let output): eval(output.logits)
        case .tokens: Issue.record("The model must complete its own prefill")
        }
        eval(cache)
        return cache
    }
    func batched(_ cache: any KVCache) throws -> any BatchedCache {
        if let rotating = cache as? RotatingKVCache {
            return try #require(BatchRotatingKVCache.fromSingle(rotating))
        }
        return BatchKVCache.fromSingle(try #require(cache as? KVCacheSimple))
    }

    let singles = try inputs.map(prefill)
    let batch = try singles[0].map(batched)
    for layer in batch.indices {
        batch[layer].extend(other: try batched(singles[1][layer]))
        let positioned = try #require(batch[layer] as? any BatchPositionedKVCache)
        #expect(positioned.batchOffset.asArray(Int.self) == positions)
    }
    let caches = batch.map { $0 as any KVCache }
    try expectRetainedHistory(batch, singles)

    for step in 0 ..< 3 {
        let token = Int32(4 + step)
        let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: caches)
        for row in 0 ..< 2 {
            let expected = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
            #expect(
                maxAbsDifference(output[row ..< row + 1], expected) < 1e-4,
                "row \(row), step \(step)")
        }
        try expectRetainedHistory(batch, singles)
    }

    batch.forEach { $0.filter(batchIndices: batchRows(1)) }
    let actual = model(MLXArray([Int32(8)]).reshaped([1, 1]), cache: caches)
    let expected = model(MLXArray([Int32(8)]).reshaped([1, 1]), cache: singles[1])
    #expect(maxAbsDifference(actual, expected) < 1e-4, "after filtering")
    try expectRetainedHistory(batch, [singles[1]])
}

private func expectRetainedHistory(_ batch: [any BatchedCache], _ singles: [[any KVCache]])
    throws
{
    func retained(_ cache: any KVCache) throws -> [MLXArray] {
        guard let rotating = cache as? RotatingKVCache else { return cache.state }
        let window = try #require(rotating.maxSize)
        let view = try #require(rotating.logicalView(tail: window))
        return [view.0, view.1]
    }
    for (row, caches) in singles.enumerated() {
        try #require(caches.count == batch.count)
        for layer in batch.indices {
            let actual = batch[layer].extract(row)
            #expect(actual.offset == caches[layer].offset)
            let a = try retained(actual)
            let b = try retained(caches[layer])
            try #require(a.count == b.count)
            for (actualArray, expectedArray) in zip(a, b) {
                try #require(actualArray.shape == expectedArray.shape)
                #expect(
                    maxAbsDifference(actualArray, expectedArray) < 1e-4,
                    "row \(row), layer \(layer)")
            }
        }
    }
}
