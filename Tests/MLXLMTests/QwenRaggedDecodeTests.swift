import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

@Suite(.serialized)
struct QwenRaggedDecodeTests {
    enum Family: CaseIterable, Sendable { case dense, moe, hybrid, hybridMoE }

    static let cases: [(Family, Bool)] = Family.allCases.flatMap { family in
        (family == .hybridMoE ? [true] : [false, true]).map { (family, $0) }
    }

    @Test(arguments: Family.allCases, [4, 16])
    func rankOneColdPrefillMatchesBatchedInput(family: Family, stepSize: Int) throws {
        try Device.withDefaultDevice(.gpu) {
            try withRandomState(MLXRandom.RandomState(seed: 913)) {
                let model = try makeModel(family)
                eval(model)
                for image in [false, true] {
                    let tokens =
                        image
                        ? MLXArray([Int32(1), 502, 500, 500, 500, 500, 2, 3])
                        : MLXArray([Int32(4), 5, 6])
                    let media: LMInput.ProcessedImage? =
                        image
                        ? .init(
                            pixels: MLXArray.ones([16, 3 * 2 * 16 * 16]), frames: [THW(1, 4, 4)])
                        : nil
                    let singleCache = try model.newCache(parameters: nil)
                    let batchCache = try model.newCache(parameters: nil)
                    let prefill = PrefillParameters(stepSize: stepSize)
                    guard
                        case .logits(let single) = try model.prepare(
                            .init(text: .init(tokens: tokens), image: media),
                            cache: singleCache, state: nil, prefill: prefill),
                        case .logits(let batch) = try model.prepare(
                            .init(
                                text: .init(tokens: tokens.reshaped(1, tokens.size)), image: media),
                            cache: batchCache, state: nil, prefill: prefill)
                    else {
                        Issue.record("Expected native logits")
                        return
                    }
                    #expect(single.logits.shape == batch.logits.shape)
                    #expect(abs(single.logits - batch.logits).max().item(Float.self) < 0.0001)
                    for (a, b) in zip(singleCache, batchCache) {
                        #expect(a.offset == b.offset)
                        for (x, y) in zip(a.state, b.state) {
                            #expect(x.shape == y.shape)
                            #expect(abs(x - y).max().item(Float.self) < 0.0001)
                        }
                    }
                }
            }
        }
    }

    @Test(arguments: cases)
    func raggedDecodeMatchesIndependentRows(family: Family, gpu: Bool) throws {
        try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
            try withRandomState(MLXRandom.RandomState(seed: 911)) {
                let model = try makeModel(family)
                eval(model)
                let inputs = [
                    LMInput(
                        text: .init(
                            tokens: MLXArray([Int32(1), 502, 500, 500, 500, 500, 2, 3]).reshaped(
                                1, 8)),
                        image: .init(
                            pixels: MLXRandom.normal([16, 3 * 2 * 16 * 16]), frames: [THW(1, 4, 4)])
                    ),
                    LMInput(tokens: MLXArray([Int32(4), 5, 6]).reshaped(1, 3)),
                    LMInput(tokens: MLXArray(Array(Int32(7) ... Int32(19))).reshaped(1, 13)),
                ]
                func prepare(_ input: LMInput) throws -> ([any KVCache], LMOutput.State) {
                    let cache = try model.newCache(parameters: nil)
                    guard
                        case .logits(let output) = try model.prepare(
                            input, cache: cache, state: nil, prefill: .init(stepSize: 4))
                    else {
                        Issue.record("Qwen must complete native prefill")
                        throw CancellationError()
                    }
                    let state = try #require(output.state)
                    eval(output.logits, cache, model.generationStateArrays(state))
                    return (cache, state)
                }
                // Separate oracle caches: extending recurrent caches mutates their original objects.
                let references = try inputs.map(prepare)
                let adopted = try inputs.map(prepare)
                func convert(_ caches: [any KVCache]) throws -> [any BatchedCache] {
                    try caches.map { cache in
                        if let simple = cache as? KVCacheSimple {
                            return BatchKVCache.fromSingle(simple)
                        }
                        return try #require(cache as? any BatchedCache)
                    }
                }
                let batch = try convert(adopted[0].0)
                func extend(_ row: Int) throws {
                    let next = try convert(adopted[row].0)
                    for layer in batch.indices { batch[layer].extendBatched(next[layer]) }
                }
                try extend(1)
                var state = model.mergeGenerationStates([adopted[0].1, adopted[1].1])
                var referenceStates = references.map { $0.1 }
                var rows = [0, 1]
                func check(_ step: Int) throws {
                    let token = Int32(30 + step)
                    let actual = model(
                        .init(
                            tokens: MLXArray(Array(repeating: token, count: rows.count)).reshaped(
                                rows.count, 1)), cache: batch, state: state)
                    state = try #require(actual.state)
                    eval(actual.logits, batch, model.generationStateArrays(state))
                    for (index, row) in rows.enumerated() {
                        let expected = model(
                            .init(tokens: MLXArray([token]).reshaped(1, 1)),
                            cache: references[row].0, state: referenceStates[row])
                        referenceStates[row] = try #require(expected.state)
                        eval(expected.logits, references[row].0)
                        let error = abs(actual.logits[index ..< index + 1] - expected.logits).max()
                            .item(Float.self)
                        #expect(
                            error < 0.001,
                            "\(family), row \(row), step \(step), logit error \(error)")
                        let actualState = model.selectGenerationState(state, batchIndices: [index])
                        for (a, b) in zip(
                            model.generationStateArrays(actualState),
                            model.generationStateArrays(referenceStates[row]))
                        {
                            #expect(a.asArray(Int32.self) == b.asArray(Int32.self))
                        }
                        for layer in batch.indices {
                            let extracted = batch[layer].extractBatched(index)
                            let expectedCache = references[row].0[layer]
                            #expect(extracted.offset == expectedCache.offset)
                            try #require(extracted.state.count == expectedCache.state.count)
                            for (a, b) in zip(extracted.state, expectedCache.state) {
                                try #require(a.shape == b.shape)
                                #expect(
                                    abs(a - b).max().item(Float.self) < 0.001,
                                    "\(family), row \(row), layer \(layer): retained cache differs")
                            }
                        }
                    }
                }
                try check(0)
                try check(1)
                try extend(2)
                state = model.mergeGenerationStates([state, adopted[2].1])
                rows.append(2)
                try check(2)
                batch.forEach { $0.filterBatched(batchIndices: MLXArray([Int32(2), 0])) }
                state = model.selectGenerationState(state, batchIndices: [2, 0])
                rows = [2, 0]
                try check(3)
                batch.forEach { $0.filterBatched(batchIndices: MLXArray([Int32(1)])) }
                state = model.selectGenerationState(state, batchIndices: [1])
                rows = [0]
                try check(4)
            }
        }
    }
}

private func makeModel(_ family: QwenRaggedDecodeTests.Family) throws
    -> any BatchedGenerationStateProviding
{
    switch family {
    case .dense:
        return Qwen3VL(
            try JSONDecoder().decode(
                Qwen3VLConfiguration.self,
                from: Data(
                    """
                    {
                      "model_type": "qwen3_vl",
                      "image_token_id": 500,
                      "video_token_id": 501,
                      "vision_start_token_id": 502,
                      "vision_end_token_id": 503,
                      "vocab_size": 512,
                      "text_config": {
                        "model_type": "qwen3_vl",
                        "hidden_size": 64,
                        "num_hidden_layers": 4,
                        "intermediate_size": 128,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "head_dim": 16,
                        "vocab_size": 512,
                        "max_position_embeddings": 4096,
                        "rms_norm_eps": 1e-06,
                        "rope_theta": 100000.0,
                        "rope_scaling": {
                          "type": "default",
                          "mrope_section": [
                            4,
                            2,
                            2
                          ]
                        }
                      },
                      "vision_config": {
                        "model_type": "qwen3_vl",
                        "depth": 2,
                        "hidden_size": 32,
                        "intermediate_size": 64,
                        "out_hidden_size": 64,
                        "num_heads": 2,
                        "patch_size": 16,
                        "spatial_merge_size": 2,
                        "temporal_patch_size": 2,
                        "num_position_embeddings": 64,
                        "deepstack_visual_indexes": [
                          0,
                          1
                        ]
                      }
                    }
                    """.utf8)))
    case .moe:
        return Qwen3VLMoE(
            try JSONDecoder().decode(
                Qwen3VLMoEConfiguration.self,
                from: Data(
                    """
                    {
                      "model_type": "qwen3_vl_moe",
                      "image_token_id": 500,
                      "video_token_id": 501,
                      "vision_start_token_id": 502,
                      "vision_end_token_id": 503,
                      "vocab_size": 512,
                      "text_config": {
                        "model_type": "qwen3_vl_moe",
                        "hidden_size": 64,
                        "num_hidden_layers": 4,
                        "intermediate_size": 128,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "head_dim": 16,
                        "vocab_size": 512,
                        "max_position_embeddings": 4096,
                        "rms_norm_eps": 1e-06,
                        "rope_theta": 100000.0,
                        "rope_scaling": {
                          "type": "default",
                          "mrope_section": [
                            4,
                            2,
                            2
                          ]
                        },
                        "num_experts": 2,
                        "num_experts_per_tok": 1,
                        "decoder_sparse_step": 1,
                        "mlp_only_layers": [],
                        "moe_intermediate_size": 128,
                        "tie_word_embeddings": false
                      },
                      "vision_config": {
                        "model_type": "qwen3_vl",
                        "depth": 2,
                        "hidden_size": 32,
                        "intermediate_size": 64,
                        "out_hidden_size": 64,
                        "num_heads": 2,
                        "patch_size": 16,
                        "spatial_merge_size": 2,
                        "temporal_patch_size": 2,
                        "num_position_embeddings": 64,
                        "deepstack_visual_indexes": [
                          0,
                          1
                        ]
                      }
                    }
                    """.utf8)))
    case .hybrid:
        return Qwen35(
            try JSONDecoder().decode(
                Qwen35Configuration.self,
                from: Data(
                    """
                    {
                      "model_type": "qwen3_5_vl",
                      "image_token_id": 500,
                      "video_token_id": 501,
                      "vision_start_token_id": 502,
                      "vision_end_token_id": 503,
                      "vocab_size": 512,
                      "text_config": {
                        "model_type": "qwen3_5",
                        "hidden_size": 64,
                        "num_hidden_layers": 4,
                        "intermediate_size": 128,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "head_dim": 32,
                        "vocab_size": 512,
                        "full_attention_interval": 2,
                        "linear_num_value_heads": 4,
                        "linear_num_key_heads": 2,
                        "linear_key_head_dim": 32,
                        "linear_value_head_dim": 32,
                        "linear_conv_kernel_dim": 4,
                        "max_position_embeddings": 4096,
                        "rope_parameters": {
                          "type": "default",
                          "mrope_section": [
                            8,
                            4,
                            4
                          ],
                          "rope_theta": 100000.0,
                          "partial_rotary_factor": 1.0
                        }
                      },
                      "vision_config": {
                        "model_type": "qwen3_vl",
                        "depth": 2,
                        "hidden_size": 32,
                        "intermediate_size": 64,
                        "out_hidden_size": 64,
                        "num_heads": 2,
                        "patch_size": 16,
                        "spatial_merge_size": 2,
                        "temporal_patch_size": 2,
                        "num_position_embeddings": 64
                      }
                    }
                    """.utf8)))
    case .hybridMoE:
        return Qwen35MoE(
            try JSONDecoder().decode(
                Qwen35Configuration.self,
                from: Data(
                    """
                    {
                      "model_type": "qwen3_5_moe",
                      "image_token_id": 500,
                      "video_token_id": 501,
                      "vision_start_token_id": 502,
                      "vision_end_token_id": 503,
                      "vocab_size": 512,
                      "text_config": {
                        "model_type": "qwen3_5_moe",
                        "hidden_size": 64,
                        "num_hidden_layers": 4,
                        "intermediate_size": 128,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "head_dim": 32,
                        "vocab_size": 512,
                        "full_attention_interval": 2,
                        "linear_num_value_heads": 4,
                        "linear_num_key_heads": 2,
                        "linear_key_head_dim": 32,
                        "linear_value_head_dim": 32,
                        "linear_conv_kernel_dim": 4,
                        "max_position_embeddings": 4096,
                        "rope_parameters": {
                          "type": "default",
                          "mrope_section": [
                            8,
                            4,
                            4
                          ],
                          "rope_theta": 100000.0,
                          "partial_rotary_factor": 1.0
                        },
                        "num_experts": 2,
                        "num_experts_per_tok": 1,
                        "moe_intermediate_size": 128,
                        "shared_expert_intermediate_size": 128
                      },
                      "vision_config": {
                        "model_type": "qwen3_vl",
                        "depth": 2,
                        "hidden_size": 32,
                        "intermediate_size": 64,
                        "out_hidden_size": 64,
                        "num_heads": 2,
                        "patch_size": 16,
                        "spatial_merge_size": 2,
                        "temporal_patch_size": 2,
                        "num_position_embeddings": 64
                      }
                    }
                    """.utf8)))
    }
}
