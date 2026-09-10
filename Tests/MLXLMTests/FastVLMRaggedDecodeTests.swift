import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

@Suite(.serialized)
struct FastVLMRaggedDecodeTests {
    @Test("FastVLM ragged adopted decode matches separately prefilled rows")
    func fastVLMRaggedDecode() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try tinyFastVLMConfiguration()
            let model = withRandomState(MLXRandom.RandomState(seed: 501)) {
                let model = FastVLM(config)
                eval(model)
                return model
            }
            let inputs = [
                LMInput(
                    text: .init(tokens: MLXArray([1, -200, 2]).reshaped([1, 3])),
                    image: .init(pixels: MLXArray.ones([1, 3, 16, 16]), frames: [])),
                LMInput(tokens: MLXArray([1, 2, 3]).reshaped([1, 3])),
            ]
            func prefill(_ input: LMInput) throws -> [KVCacheSimple] {
                let cache = try model.newCache(parameters: nil)
                switch try model.prepare(input, cache: cache, state: nil, prefill: .init()) {
                case .logits(let output): eval(output.logits)
                case .tokens: Issue.record("FastVLM must complete its own prefill")
                }
                eval(cache)
                return try cache.map { try #require($0 as? KVCacheSimple) }
            }
            let singles = try inputs.map(prefill)
            let batched = singles[0].map { BatchKVCache.fromSingle($0) }
            for layer in batched.indices {
                batched[layer].extend(other: BatchKVCache.fromSingle(singles[1][layer]))
            }
            #expect(batched[0].batchOffsets.asArray(Int.self) == [18, 3])
            try expectCaches(batched, match: singles)
            for step in 0 ..< 3 {
                let token = Int32(4 + step)
                let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: batched)
                eval(output)
                for row in 0 ..< 2 {
                    let expected = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
                    eval(expected)
                    let error = abs(output[row ..< row + 1] - expected).max().item(Float.self)
                    #expect(
                        error < 0.0001,
                        "row \(row), step \(step), maximum logit difference \(error)")
                    #expect(
                        output[row ..< row + 1].argMax(axis: -1).item(Int.self)
                            == expected.argMax(axis: -1).item(Int.self))
                }
                try expectCaches(batched, match: singles)
            }
            batched.forEach { $0.filter(batchIndices: [1]) }
            let actual = model(MLXArray([Int32(8)]).reshaped([1, 1]), cache: batched)
            let expected = model(MLXArray([Int32(8)]).reshaped([1, 1]), cache: singles[1])
            let error = abs(actual - expected).max().item(Float.self)
            #expect(error < 0.0001, "after filtering: maximum logit difference \(error)")
            #expect(
                actual.argMax(axis: -1).item(Int.self) == expected.argMax(axis: -1).item(Int.self))
            try expectCaches(batched, match: [singles[1]])
        }
    }
}

private func expectCaches(_ batched: [BatchKVCache], match singles: [[KVCacheSimple]]) throws {
    for (row, caches) in singles.enumerated() {
        #expect(caches.count == batched.count)
        for layer in batched.indices {
            let actual = batched[layer].extract(idx: row)
            let expected = caches[layer]
            #expect(actual.offset == expected.offset)
            try #require(actual.state.count == expected.state.count)
            for (a, b) in zip(actual.state, expected.state) {
                try #require(a.shape == b.shape)
                #expect(
                    abs(a - b).max().item(Float.self) < 0.0001,
                    "row \(row), layer \(layer): retained K/V differs")
            }
        }
    }
}

private func tinyFastVLMConfiguration() throws -> FastVLMConfiguration {
    try JSONDecoder().decode(
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
}
