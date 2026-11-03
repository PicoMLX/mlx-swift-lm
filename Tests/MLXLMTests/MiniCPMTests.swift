// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

package func tinyMiniCPMConfiguration(_ variant: String = "minicpmv4_7", moe: Bool = false) throws
    -> MiniCPMConfiguration
{
    var text: [String: Any] = [
        "model_type": "qwen3_5_text", "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "rms_norm_eps": 1e-6, "vocab_size": 64, "tie_word_embeddings": false,
        "rope_theta": 10_000_000,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
        "rope_parameters": [
            "type": "default", "rope_theta": 10_000_000, "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
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
        text["moe_intermediate_size"] = 32
        text["shared_expert_intermediate_size"] = 32
    }
    let json: [String: Any] = [
        "model_type": variant, "text_config": text, "query_num": 4, "audio_pool_step": 2,
        "insert_layer_id": 0, "downsample_mode": "16x", "image_start_id": 50, "image_end_id": 51,
        "slice_start_id": 52, "slice_end_id": 53, "newline_id": 54,
        "vision_config": [
            "model_type": "siglip_vision_model", "hidden_size": 16, "intermediate_size": 32,
            "num_attention_heads": 2, "num_hidden_layers": 2, "patch_size": 2, "image_size": 8,
        ],
        "audio_config": [
            "d_model": 8, "encoder_ffn_dim": 32, "encoder_attention_heads": 2, "encoder_layers": 1,
            "num_mel_bins": 8, "max_source_positions": 32,
        ],
    ]
    let config = try JSONDecoder().decode(
        MiniCPMConfiguration.self, from: JSONSerialization.data(withJSONObject: json))
    try config.validateModelConfiguration()
    return config
}
package func miniCPMPrepared(
    _ model: MiniCPM, _ input: LMInput, cache: [any KVCache], chunk: Int? = nil
) throws -> MLXArray {
    switch try model.prepare(
        input, cache: cache, state: nil,
        prefill: .init(stepSize: chunk, chunking: chunk == nil ? .unchunked : .balanced))
    {
    case .logits(let output): return output.logits
    case .tokens: throw VLMError.processing("MiniCPM must return prepared media logits")
    }
}

@Suite(.serialized)
struct MiniCPMTests {
    @Test(
        "All MiniCPM decoders preserve ragged row state through chunked prefill, filtering, extraction and extension",
        arguments: ["minicpmo", "minicpmv4_6", "minicpmv4_7"])
    func ragged(_ variant: String) throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(817)
            let model = MiniCPM(try tinyMiniCPMConfiguration(variant))
            let prompts: [[Int32]] = [[1, 2, 3], [4, 5, 6, 7, 8], [9, 10]]
            let singles = try prompts.map { _ in try model.newCache(parameters: nil) }
            let expected = try prompts.enumerated().map { row, prompt in
                try miniCPMPrepared(
                    model, LMInput(tokens: MLXArray(prompt).reshaped(1, prompt.count)),
                    cache: singles[row])[0, -1]
            }
            let factories = try makeBatchedCacheFactories(for: model.newCache(parameters: nil))
            let batch = factories.map { $0([0, 0, 0]) }
            let cache = batch.map { $0 as any KVCache }
            let padded = prompts.flatMap { $0 + Array(repeating: 0, count: 5 - $0.count) }
            for layer in batch { layer.prepare(lengths: prompts.map(\.count)) }
            let first = model(MLXArray(padded).reshaped(3, 5)[0..., ..<2], cache: cache)
            let rest = model(MLXArray(padded).reshaped(3, 5)[0..., 2...], cache: cache)
            let actual = concatenated([first, rest], axis: 1)
            for layer in batch { layer.finalize() }
            for row in prompts.indices {
                #expect(maxAbsDifference(actual[row, prompts[row].count - 1], expected[row]) < 3e-4)
            }
            for token: Int32 in [11, 12, 13] {
                let actual = model(MLXArray([token, token, token]).reshaped(3, 1), cache: cache)
                for row in prompts.indices {
                    #expect(
                        maxAbsDifference(
                            actual[row ..< row + 1],
                            model(MLXArray([token]).reshaped(1, 1), cache: singles[row])) < 3e-4)
                }
            }
            for layer in batch { layer.filter(batchIndices: MLXArray([Int32(2), 0])) }
            let filtered = model(MLXArray([Int32(14), 14]).reshaped(2, 1), cache: cache)
            for (row, original) in [2, 0].enumerated() {
                #expect(
                    maxAbsDifference(
                        filtered[row ..< row + 1],
                        model(MLXArray([Int32(14)]).reshaped(1, 1), cache: singles[original]))
                        < 3e-4)
            }
            let extracted = batch.map { $0.extract(0) }
            let next = MLXArray([Int32(15)]).reshaped(1, 1)
            #expect(
                maxAbsDifference(model(next, cache: extracted), model(next, cache: singles[2]))
                    < 3e-4)
            let added = factories.map { $0([0]) }
            let addedCache = added.map { $0 as any KVCache }
            _ = try miniCPMPrepared(
                model, LMInput(tokens: MLXArray([Int32(2), 3]).reshaped(1, 2)), cache: addedCache)
            let clone = added.map { $0.copy() }
            for index in batch.indices { batch[index].extend(other: added[index]) }
            let extended = model(MLXArray([Int32(16), 16, 16]).reshaped(3, 1), cache: cache)
            #expect(
                maxAbsDifference(
                    extended[2 ..< 3], model(MLXArray([Int32(16)]).reshaped(1, 1), cache: clone))
                    < 3e-4)
        }
    }
    @Test(
        "Prepared media rows can join a live batch without sharing canvas or audio state",
        arguments: ["minicpmo", "minicpmv4_6", "minicpmv4_7"])
    func mediaAdoption(_ variant: String) throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(818)
            let model = MiniCPM(try tinyMiniCPMConfiguration(variant))
            let count = variant == "minicpmo" ? 4 : 1
            let pixels = sin(MLXArray(0 ..< 192).asType(.float32)).reshaped(1, 16, 12)
            var inputs = [LMInput(tokens: MLXArray([Int32(1), 2]).reshaped(1, 2))]
            for video in [false, true] {
                let sentinel = video ? MiniCPM.videoToken : MiniCPM.imageToken
                let tokens = [1, 50] + Array(repeating: sentinel, count: count) + [51, 2]
                inputs.append(
                    LMInput(
                        text: .init(tokens: MLXArray(tokens).reshaped(1, tokens.count)),
                        image: video ? nil : .init(pixels: pixels, frames: [THW(1, 4, 4)]),
                        video: video ? .init(pixels: pixels, frames: [THW(1, 4, 4)]) : nil))
            }
            if variant != "minicpmo" {
                inputs.append(
                    LMInput(
                        text: .init(
                            tokens: MLXArray([Int32(1), 50, -200, -200, -200, -200, 51, 2])
                                .reshaped(1, 8)),
                        image: .init(pixels: pixels, frames: [THW(1, 4, 4)])))
            }
            if variant == "minicpmo" {
                inputs.append(
                    LMInput(
                        text: .init(
                            tokens: MLXArray([Int32(1), -202, -202, -202, 2]).reshaped(1, 5)),
                        audio: .init(
                            features: sin(MLXArray(0 ..< 96).asType(.float32)).reshaped(1, 12, 8))))
            }
            let factories = try makeBatchedCacheFactories(for: model.newCache(parameters: nil))
            var prepared: [[any BatchedCache]] = []
            var reference: [[any KVCache]] = []
            for input in inputs {
                let row = factories.map { $0([0]) }
                let cache = row.map { $0 as any KVCache }
                _ = try miniCPMPrepared(model, input, cache: cache, chunk: 2)
                prepared.append(row)
                reference.append(row.map { $0.copy() })
            }
            let batch = prepared[0]
            let cache = batch.map { $0 as any KVCache }
            for row in prepared.dropFirst() {
                for index in batch.indices { batch[index].extend(other: row[index]) }
            }
            for token: Int32 in [11, 12, 13] {
                let actual = model(
                    MLXArray(Array(repeating: token, count: inputs.count)).reshaped(
                        inputs.count, 1), cache: cache)
                for row in inputs.indices {
                    #expect(
                        maxAbsDifference(
                            actual[row ..< row + 1],
                            model(MLXArray([token]).reshaped(1, 1), cache: reference[row])) < 4e-4)
                }
            }
            for layer in batch { layer.filter(batchIndices: MLXArray([Int32(2), 0])) }
            let row = batch.map { $0.extract(0) }
            let next = MLXArray([Int32(14)]).reshaped(1, 1)
            #expect(
                maxAbsDifference(model(next, cache: row), model(next, cache: reference[2])) < 4e-4)
        }
    }
    @Test("MiniCPM visual and audio bounds reject missing or surplus features")
    func bounds() throws {
        let model = MiniCPM(try tinyMiniCPMConfiguration())
        #expect(throws: (any Error).self) {
            try model.inputEmbeddings(LMInput(tokens: MLXArray([Int32(-200)]).reshaped(1, 1)))
        }
        let image = LMInput.ProcessedImage(
            pixels: MLXArray.zeros([1, 16, 12]), frames: [THW(1, 4, 4)])
        #expect(throws: (any Error).self) {
            try model.inputEmbeddings(
                LMInput(text: .init(tokens: MLXArray([Int32(1)]).reshaped(1, 1)), image: image))
        }
    }
    @Test("Image slicing preserves Python banker rounding, crop order and compression geometry")
    func slicing() {
        #expect(MiniCPMImageProcessing.divide(5, 2) == 4)
        #expect(MiniCPMImageProcessing.divide(7, 2) == 8)
        #expect(
            MiniCPMImageProcessing.sliceGrid(width: 1600, height: 800, scale: 448, maxSlices: 9)?.0
                == 4)
        var config = MiniCPMProcessorConfiguration()
        config.patchSize = 2
        config.scale = 16
        config.maxSlices = 4
        let crops = MiniCPMImageProcessing.crops(
            MLXArray.zeros([16, 48, 3]), config: config, omni: false)
        #expect(crops.layout?.0 == 3)
        #expect(crops.grids.count == 4)
        #expect(crops.grids.allSatisfy { $0.h % 4 == 0 && $0.w % 4 == 0 })
    }
    @Test("Fused MoE experts retain their quantization auxiliaries")
    func experts() throws {
        let model = MiniCPM(try tinyMiniCPMConfiguration(moe: true))
        let prefix = "model.language_model.layers.0.mlp.experts"
        let rows = MLXArray(0 ..< 64).asType(.float32).reshaped(1, 64, 1)
        let result = try model.sanitize(weights: [
            prefix + ".gate_up_proj": broadcast(rows, to: [4, 64, 32]),
            prefix + ".gate_up_proj_scales": broadcast(rows, to: [4, 64, 1]),
            prefix + ".gate_up_proj_biases": broadcast(rows, to: [4, 64, 1]),
        ])
        #expect(
            result["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [
                4, 32, 32,
            ])
        #expect(
            result["language_model.model.layers.0.mlp.switch_mlp.up_proj.scales"]?[0, 0, 0].item(
                Float.self) == 32)
    }
    @Test("Single-rank sharding preserves float and 4-bit logits and reports local cache heads")
    func singleRank() throws {
        try Device.withDefaultDevice(.gpu) {
            for variant in ["minicpmo", "minicpmv4_6", "minicpmv4_7"] {
                for quantized in [false, true] {
                    MLXRandom.seed(819)
                    let model = MiniCPM(try tinyMiniCPMConfiguration(variant))
                    if quantized {
                        quantize(
                            model: model, groupSize: 32, bits: 4,
                            filter: { path, _ in
                                path.hasPrefix("language_model.model.layers.")
                                    && !path.contains("linear_attn")
                            })
                    }
                    let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
                    let expected = model(tokens, cache: try model.newCache(parameters: nil))
                    eval(expected)
                    let group = try MLXDistributed.initialize()
                    try model.shard(group: group)
                    let actual = model(tokens, cache: try model.newCache(parameters: nil))
                    #expect(maxAbsDifference(actual, expected) < 4e-4)
                    #expect(model.kvHeads.allSatisfy { $0 == 2 })
                    #expect(throws: TensorParallelError.alreadySharded) {
                        try model.shard(group: group)
                    }
                }
            }
        }
    }

}
