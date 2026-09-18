import Foundation
import MLX
import MLXVLM
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct Gemma3RaggedDecodeTests {
    @Test("Gemma3 image/text decode preserves each row across sliding-window wrap")
    func gemma3RaggedDecode() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try JSONDecoder().decode(
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
                let model = Gemma3(config)
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
            func prefill(_ input: LMInput) throws -> [any KVCache] {
                let cache = try model.newCache(parameters: nil)
                switch try model.prepare(input, cache: cache, state: nil, prefill: .init()) {
                case .logits(let output): eval(output.logits)
                case .tokens: Issue.record("Gemma3 must complete its own prefill")
                }
                eval(cache)
                return cache
            }
            func convert(_ cache: any KVCache) throws -> any BatchedCache {
                if let rotating = cache as? RotatingKVCache {
                    return BatchRotatingKVCache.fromSingle(rotating)
                }
                return BatchKVCache.fromSingle(try #require(cache as? KVCacheSimple))
            }
            let singles = try inputs.map(prefill)
            let batched = try singles[0].map(convert)
            #expect(batched.contains { $0 is BatchKVCache })
            #expect(batched.contains { $0 is BatchRotatingKVCache })
            for layer in batched.indices {
                batched[layer].extendBatched(try convert(singles[1][layer]))
                let positions = try #require(batched[layer] as? any BatchPositionedKVCache)
                #expect(positions.batchOffset.asArray(Int.self) == [6, 3])
            }
            try expectGemmaCaches(batched, match: singles)
            for step in 0 ..< 3 {
                let token = Int32(4 + step)
                let output = model(MLXArray([token, token]).reshaped([2, 1]), cache: batched)
                eval(output)
                for row in 0 ..< 2 {
                    let expected = model(MLXArray([token]).reshaped([1, 1]), cache: singles[row])
                    eval(expected)
                    let error = abs(output[row ..< row + 1] - expected).max().item(Float.self)
                    #expect(error < 0.0001, "row \(row), step \(step), logit difference \(error)")
                    #expect(
                        output[row ..< row + 1].argMax(axis: -1).item(Int.self)
                            == expected.argMax(axis: -1).item(Int.self))
                }
                try expectGemmaCaches(batched, match: singles)
            }
            batched.forEach { $0.filterBatched(batchIndices: MLXArray([Int32(1)])) }
            let actual = model(MLXArray([Int32(8)]).reshaped([1, 1]), cache: batched)
            let expected = model(MLXArray([Int32(8)]).reshaped([1, 1]), cache: singles[1])
            let error = abs(actual - expected).max().item(Float.self)
            #expect(error < 0.0001, "after filtering: logit difference \(error)")
            #expect(
                actual.argMax(axis: -1).item(Int.self) == expected.argMax(axis: -1).item(Int.self))
            try expectGemmaCaches(batched, match: [singles[1]])
        }
    }
}

private func expectGemmaCaches(_ batched: [any BatchedCache], match singles: [[any KVCache]]) throws
{
    func retainedState(_ cache: any KVCache) throws -> [MLXArray] {
        guard let rotating = cache as? RotatingKVCache else { return cache.state }
        let window = try #require(rotating.maxSize)
        let view = try #require(rotating.logicalView(tail: window))
        return [view.0, view.1]
    }
    for (row, caches) in singles.enumerated() {
        try #require(caches.count == batched.count)
        for layer in batched.indices {
            let actual = batched[layer].extractBatched(row)
            let expected = caches[layer]
            #expect(actual.offset == expected.offset)
            let a = try retainedState(actual)
            let b = try retainedState(expected)
            try #require(a.count == b.count)
            for (actualArray, expectedArray) in zip(a, b) {
                try #require(actualArray.shape == expectedArray.shape)
                #expect(
                    abs(actualArray - expectedArray).max().item(Float.self) < 0.0001,
                    "row \(row), layer \(layer): retained K/V differs")
            }
        }
    }
}
