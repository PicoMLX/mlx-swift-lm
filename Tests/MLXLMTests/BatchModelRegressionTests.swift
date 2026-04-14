// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@preconcurrency @testable import MLXLMCommon
@testable import MLXLLM
import Testing

@Suite(
    "Batch Model Regressions",
    .enabled(if: MLXMetalGuard.isAvailable, MLXMetalGuard.unavailableComment),
    .serialized
)
struct BatchModelRegressionTests {
    private let prompts: [[Int32]] = [
        [11, 12, 13, 14, 15],
        [21, 22, 23],
    ]

    private let decodeTokens: [Int32] = [31, 32]

    @Test("Phi3 batch prefill matches single prefill")
    func phi3BatchPrefillMatchesSingle() throws {
        let model = try makePhi3Model(seed: 100)
        try assertPrefillMatchesSingle(model: model, prompts: prompts)
    }

    @Test("Phi3 batch decode matches single decode")
    func phi3BatchDecodeMatchesSingle() throws {
        let model = try makePhi3Model(seed: 101)
        try assertDecodeMatchesSingle(
            model: model,
            prompts: prompts,
            decodeTokens: decodeTokens
        )
    }

    @Test("Gemma4Text batch decode matches single decode")
    func gemma4TextBatchDecodeMatchesSingle() throws {
        let model = try makeGemma4TextModel(seed: 150)
        try assertDecodeMatchesSingle(
            model: model,
            prompts: prompts,
            decodeTokens: decodeTokens
        )
    }

    @Test("Gemma4 batch decode matches single decode")
    func gemma4BatchDecodeMatchesSingle() throws {
        let model = try makeGemma4Model(seed: 152)
        try assertDecodeMatchesSingle(
            model: model,
            prompts: prompts,
            decodeTokens: decodeTokens
        )
    }

    @Test("Gemma4Text remains scheduler batch-compatible")
    func gemma4TextRemainsBatchCompatible() throws {
        let model = try makeGemma4TextModel(seed: 151)
        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))

        #expect(
            InferenceScheduler.isBatchCompatible(
                input: input,
                parameters: GenerateParameters(maxTokens: 1, temperature: 0),
                cache: nil,
                model: model
            )
        )
    }

    @Test("Gemma4 remains scheduler batch-compatible")
    func gemma4RemainsBatchCompatible() throws {
        let model = try makeGemma4Model(seed: 153)
        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))

        #expect(
            InferenceScheduler.isBatchCompatible(
                input: input,
                parameters: GenerateParameters(maxTokens: 1, temperature: 0),
                cache: nil,
                model: model
            )
        )
    }

    @Test("Phi3 remains scheduler batch-compatible")
    func phi3RemainsBatchCompatible() throws {
        let model = try makePhi3Model(seed: 102)
        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))

        #expect(
            InferenceScheduler.isBatchCompatible(
                input: input,
                parameters: GenerateParameters(maxTokens: 1, temperature: 0),
                cache: nil,
                model: model
            )
        )
    }

    @Test("FalconH1 remains scheduler batch-incompatible")
    func falconH1RemainsBatchIncompatible() throws {
        let model = try makeFalconH1Model(seed: 200)
        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))

        #expect(
            InferenceScheduler.isBatchCompatible(
                input: input,
                parameters: GenerateParameters(maxTokens: 1, temperature: 0),
                cache: nil,
                model: model
            ) == false
        )
    }

    @Test("FalconH1 attention batched decode matches merged single decode")
    func falconAttentionDecodeMatchesMergedSingles() throws {
        let configuration = try makeFalconH1Configuration()
        let attention = withRandomState(MLXRandom.RandomState(seed: 201)) {
            let attention = FalconH1Attention(configuration)
            eval(attention)
            return attention
        }

        try assertFalconAttentionDecodeMatchesMergedSingles(
            attention: attention,
            hiddenSize: 16,
            promptLengths: prompts.map(\.count)
        )
    }

    private func makePhi3Model(seed: UInt64) throws -> Phi3Model {
        let configuration: Phi3Configuration = try decodeConfiguration(
            """
            {
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "intermediate_size": 32,
              "num_attention_heads": 4,
              "rms_norm_eps": 0.00001,
              "vocab_size": 64,
              "num_key_value_heads": 2,
              "rope_theta": 10000.0,
              "rope_traditional": false,
              "partial_rotary_factor": 1.0,
              "max_position_embeddings": 128,
              "original_max_position_embeddings": 128,
              "tie_word_embeddings": false
            }
            """
        )

        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = Phi3Model(configuration)
            eval(model)
            return model
        }
    }

    private func makeFalconH1Configuration() throws -> FalconH1Configuration {
        try decodeConfiguration(
            """
            {
              "model_type": "falcon_h1",
              "hidden_size": 16,
              "vocab_size": 64,
              "num_hidden_layers": 2,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 4,
              "max_position_embeddings": 128,
              "intermediate_size": 32,
              "mamba_d_ssm": 8,
              "mamba_d_state": 4,
              "mamba_n_heads": 2,
              "mamba_d_head": 4,
              "mamba_d_conv": 4,
              "rope_theta": 10000.0,
              "rope_traditional": false
            }
            """
        )
    }

    private func makeGemma4TextModel(seed: UInt64) throws -> Gemma4TextModel {
        let configuration: Gemma4TextConfiguration = try decodeConfiguration(
            """
            {
              "model_type": "gemma4_text",
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "intermediate_size": 32,
              "num_attention_heads": 4,
              "head_dim": 4,
              "global_head_dim": 4,
              "global_partial_rotary_factor": 1.0,
              "rms_norm_eps": 0.00001,
              "vocab_size": 64,
              "vocab_size_per_layer_input": 64,
              "num_key_value_heads": 2,
              "num_kv_shared_layers": 0,
              "hidden_size_per_layer_input": 0,
              "sliding_window": 8,
              "sliding_window_pattern": 2,
              "max_position_embeddings": 128,
              "attention_k_eq_v": false,
              "final_logit_softcapping": 30.0,
              "use_double_wide_mlp": false,
              "layer_types": ["full_attention", "full_attention"],
              "tie_word_embeddings": false,
              "rope_parameters": {
                "full_attention": {
                  "rope_theta": 10000.0,
                  "partial_rotary_factor": 1.0
                },
                "sliding_attention": {
                  "rope_theta": 10000.0
                }
              }
            }
            """
        )

        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = Gemma4TextModel(configuration)
            eval(model)
            return model
        }
    }

    private func makeGemma4Model(seed: UInt64) throws -> Gemma4Model {
        let configuration: Gemma4Configuration = try decodeConfiguration(
            """
            {
              "model_type": "gemma4",
              "vocab_size": 64,
              "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "head_dim": 4,
                "global_head_dim": 4,
                "global_partial_rotary_factor": 1.0,
                "rms_norm_eps": 0.00001,
                "vocab_size": 64,
                "vocab_size_per_layer_input": 64,
                "num_key_value_heads": 2,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "sliding_window": 8,
                "sliding_window_pattern": 2,
                "max_position_embeddings": 128,
                "attention_k_eq_v": false,
                "final_logit_softcapping": 30.0,
                "use_double_wide_mlp": false,
                "layer_types": ["full_attention", "full_attention"],
                "tie_word_embeddings": false,
                "rope_parameters": {
                  "full_attention": {
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0
                  },
                  "sliding_attention": {
                    "rope_theta": 10000.0
                  }
                }
              }
            }
            """
        )

        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = Gemma4Model(configuration)
            eval(model)
            return model
        }
    }

    private func makeFalconH1Model(seed: UInt64) throws -> FalconH1Model {
        let configuration = try makeFalconH1Configuration()

        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = FalconH1Model(configuration)
            eval(model)
            return model
        }
    }

    private func decodeConfiguration<T: Decodable>(_ json: String) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(json.utf8))
    }

    private func assertPrefillMatchesSingle<M: LanguageModel>(
        model: M,
        prompts: [[Int32]]
    ) throws {
        let singleResults = prompts.map { prompt in
            prefillSingle(model: model, prompt: prompt)
        }
        let batched = prefillBatch(model: model, prompts: prompts)

        for (index, prompt) in prompts.enumerated() {
            let pad = batched.leftPadding[index]
            let batchValid = batched.logits[index ..< (index + 1), pad..., 0...].asType(.float32)
            let single = singleResults[index].logits.asType(.float32)

            #expect(batchValid.shape == single.shape)
            #expect(
                maxAbsDifference(batchValid, single) <= 0.01,
                "Prefill logits diverged for prompt \(prompt)"
            )
        }
    }

    private func assertDecodeMatchesSingle<M: LanguageModel>(
        model: M,
        prompts: [[Int32]],
        decodeTokens: [Int32]
    ) throws {
        let singleResults = prompts.enumerated().map { index, prompt in
            var result = prefillSingle(model: model, prompt: prompt)
            let decodeInput = MLXArray([decodeTokens[index]])[.newAxis, .ellipsis]
            let decodeLogits = model.callAsFunction(decodeInput, cache: result.cache)
            materialize(arrays: [decodeLogits], cache: result.cache)
            result.logits = decodeLogits
            return result
        }

        var batched = prefillBatch(model: model, prompts: prompts)
        let batchedDecodeInput = MLXArray(decodeTokens, [decodeTokens.count, 1])
        let batchedDecodeLogits = model.callAsFunction(batchedDecodeInput, cache: batched.cache)
        materialize(arrays: [batchedDecodeLogits], cache: batched.cache)
        batched.logits = batchedDecodeLogits

        for index in prompts.indices {
            let batchRow = batched.logits[index ..< (index + 1), 0..., 0...].asType(.float32)
            let single = singleResults[index].logits.asType(.float32)

            #expect(batchRow.shape == single.shape)
            #expect(
                maxAbsDifference(batchRow, single) <= 0.01,
                "Decode logits diverged for prompt index \(index)"
            )
        }
    }

    private func assertFalconAttentionDecodeMatchesMergedSingles(
        attention: FalconH1Attention,
        hiddenSize: Int,
        promptLengths: [Int]
    ) throws {
        let singleCaches: [KVCacheSimple] = promptLengths.enumerated().map { index, length in
            let cache = KVCacheSimple()
            let hidden = makeHiddenStates(
                length: length,
                hiddenSize: hiddenSize,
                base: Float(index + 1)
            )
            let mask = createAttentionMask(h: hidden, cache: cache)
            let output = attention(hidden, mask: mask, cache: cache)
            materialize(arrays: [output], cache: [cache])
            return cache
        }

        let batchCache = BatchKVCache.merge(singleCaches.map { $0 as KVCache })
        let decodeInputs = promptLengths.indices.map { index in
            makeHiddenStates(length: 1, hiddenSize: hiddenSize, base: Float(100 + index))
        }

        let singleOutputs = decodeInputs.enumerated().map { index, decodeInput in
            let mask = createAttentionMask(h: decodeInput, cache: singleCaches[index])
            let output = attention(decodeInput, mask: mask, cache: singleCaches[index])
            materialize(arrays: [output], cache: [singleCaches[index]])
            return output
        }

        let batchedDecodeInput = concatenated(decodeInputs, axis: 0)
        let batchedMask = createAttentionMask(h: batchedDecodeInput, cache: batchCache)
        let batchedOutput = attention(batchedDecodeInput, mask: batchedMask, cache: batchCache)
        materialize(arrays: [batchedOutput], cache: [batchCache])

        for index in promptLengths.indices {
            let batchRow = batchedOutput[index ..< (index + 1), 0..., 0...].asType(.float32)
            let single = singleOutputs[index].asType(.float32)

            #expect(batchRow.shape == single.shape)
            #expect(
                maxAbsDifference(batchRow, single) <= 0.01,
                "FalconH1 attention decode diverged for prompt index \(index)"
            )
        }
    }

    private func prefillSingle<M: LanguageModel>(
        model: M,
        prompt: [Int32]
    ) -> (logits: MLXArray, cache: [KVCache]) {
        let cache = model.newCache(parameters: nil)
        let input = MLXArray(prompt)[.newAxis, .ellipsis]
        let logits = model.callAsFunction(input, cache: cache)
        materialize(arrays: [logits], cache: cache)
        return (logits, cache)
    }

    private func prefillBatch<M: LanguageModel>(
        model: M,
        prompts: [[Int32]]
    ) -> (logits: MLXArray, cache: [KVCache], leftPadding: [Int]) {
        let maxLength = prompts.map(\.count).max() ?? 0
        let leftPadding = prompts.map { maxLength - $0.count }

        let flat = zip(prompts, leftPadding).flatMap { prompt, pad in
            Array(repeating: Int32(0), count: pad) + prompt
        }
        let input = MLXArray(flat, [prompts.count, maxLength])
        let cache: [KVCache] = model.newCache(parameters: nil).map { layerCache in
            if let rotatingCache = layerCache as? RotatingKVCache {
                return BatchRotatingKVCache(
                    maxSize: rotatingCache.maxSize ?? 0,
                    leftPadding: leftPadding,
                    keep: rotatingCache.keep
                )
            } else {
                return BatchKVCache(leftPadding: leftPadding)
            }
        }
        let logits = model.callAsFunction(input, cache: cache)
        materialize(arrays: [logits], cache: cache)
        return (logits, cache, leftPadding)
    }

    private func makeHiddenStates(length: Int, hiddenSize: Int, base: Float) -> MLXArray {
        let values = (0 ..< (length * hiddenSize)).map { index in
            base + Float(index) / 100
        }
        return MLXArray(values, [1, length, hiddenSize])
    }

    private func materialize(arrays: [MLXArray], cache: [KVCache]) {
        if !arrays.isEmpty {
            eval(arrays)
        }
        let cacheState = cache.flatMap { $0.state }
        if !cacheState.isEmpty {
            eval(cacheState)
        }
    }

    private func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs - rhs).max().item(Float.self)
    }
}
