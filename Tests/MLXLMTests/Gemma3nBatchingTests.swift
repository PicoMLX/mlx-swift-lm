// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Gemma3n sliding-window mask regressions, single-stream and batched.
///
/// Gemma3n is the only model in the repo that rewrites its layer masks
/// (the sliding-mask surgery in `Gemma3nDecoderLayer`), so it needs its own
/// coverage on top of `BatchModelRegressionTests`. These tests pin three
/// stacked bugs:
///
/// 1. The surgery filled masked-out positions via `MLXArray.maskFill`,
///    which traps on the BOOLEAN masks every batched cache produces (and
///    single-stream `RotatingKVCache` produces past the window).
/// 2. `Gemma3nAttention` cast boolean masks to the query dtype, turning
///    select masks into additive +1/+0 no-ops (nothing masked).
/// 3. The surgery unconditionally sliced masks to `max(n, W)` columns,
///    truncating chunked-prefill masks that already match the post-update
///    key width -- an SDPA shape mismatch on chunk 2+.
///
/// Same conventions as `BatchModelRegressionTests`: in-process random-weight
/// model from a JSON config, no downloads, greedy/deterministic, and no
/// Metal-availability guard (macOS / Metal CI runs these end to end).
@Suite("Gemma3n batching regressions", .serialized)
struct Gemma3nBatchingTests {
    private let prompts: [[Int32]] = [
        [11, 12, 13, 14, 15],
        [21, 22, 23],
    ]

    private let decodeTokens: [Int32] = [31, 32]

    @Test("Gemma3n batch prefill matches single prefill")
    func gemma3nBatchPrefillMatchesSingle() throws {
        let model = try makeGemma3nTextModel(seed: 300)
        try assertPrefillMatchesSingle(model: model, prompts: prompts)
    }

    @Test("Gemma3n batch decode matches single decode across the window wrap")
    func gemma3nBatchDecodeMatchesSingleAcrossWrap() throws {
        let model = try makeGemma3nTextModel(seed: 301)

        let singles = prompts.map { prefillSingle(model: model, prompt: $0) }
        let batched = prefillBatch(model: model, prompts: prompts)

        // 8 decode steps advance the prompt offsets 5/3 to 13/11, crossing
        // sliding_window 8: the batched rotating cache wraps and rolls its
        // boolean decode masks while single-stream decodes with `.none`.
        // Feeding fixed tokens keeps both runs on identical streams so
        // per-step logits must agree row for row.
        for step in 0 ..< 8 {
            let stepTokens = decodeTokens.map { $0 + Int32(step) }

            let singleLogits = prompts.indices.map { index in
                let decodeInput = MLXArray([stepTokens[index]])[.newAxis, .ellipsis]
                let logits = model.callAsFunction(decodeInput, cache: singles[index].cache)
                materialize(arrays: [logits], cache: singles[index].cache)
                return logits
            }

            let batchedInput = MLXArray(stepTokens, [stepTokens.count, 1])
            let batchedLogits = model.callAsFunction(batchedInput, cache: batched.cache)
            materialize(arrays: [batchedLogits], cache: batched.cache)

            for index in prompts.indices {
                let batchRow = batchedLogits[index ..< (index + 1), 0..., 0...].asType(.float32)
                let single = singleLogits[index].asType(.float32)

                #expect(batchRow.shape == single.shape)
                #expect(
                    maxAbsDifference(batchRow, single) <= 0.01,
                    "Decode logits diverged at step \(step) for prompt index \(index)"
                )
            }
        }
    }

    @Test("Gemma3n chunked prefill past the sliding window matches single-pass")
    func gemma3nChunkedPrefillMatchesSinglePass() throws {
        let model = try makeGemma3nTextModel(seed: 302)
        let promptTokens = (0 ..< 24).map { Int32(($0 * 7 + 3) % 63) }

        // Chunk size == sliding_window 8: chunk 2 starts at cache offset 8,
        // where RotatingKVCache.makeMask returns a BOOLEAN array mask of
        // width min(maxSize - 1, offset) + n == 15 that already matches the
        // post-update keys exactly. This is the engine-style chunked prefill
        // (PrefillBatch / EngineDriver drive callAsFunction in prefill-sized
        // chunks); it used to trap on the bool fill and, once past that,
        // slice the mask to 8 columns against 15 keys.
        let chunkSize = 8
        let chunkedCache = model.newCache(parameters: nil)
        var chunkedLogits: MLXArray?
        var index = 0
        while index < promptTokens.count {
            let n = min(chunkSize, promptTokens.count - index)
            let chunk = MLXArray(Array(promptTokens[index ..< (index + n)]))[.newAxis, .ellipsis]
            let logits = model.callAsFunction(chunk, cache: chunkedCache)
            materialize(arrays: [logits], cache: chunkedCache)
            chunkedLogits = logits
            index += n
        }

        let singleCache = model.newCache(parameters: nil)
        let singleInput = MLXArray(promptTokens)[.newAxis, .ellipsis]
        let singleLogits = model.callAsFunction(singleInput, cache: singleCache)
        materialize(arrays: [singleLogits], cache: singleCache)

        #expect(chunkedCache.map { $0.offset } == singleCache.map { $0.offset })
        #expect(chunkedCache.allSatisfy { $0.offset == promptTokens.count })

        let chunkedFinal = try #require(chunkedLogits)
        let chunkedLast = chunkedFinal[0..., -1, 0...].asType(.float32)
        let singleLast = singleLogits[0..., -1, 0...].asType(.float32)
        #expect(chunkedLast.shape == singleLast.shape)
        #expect(
            maxAbsDifference(chunkedLast, singleLast) <= 0.01,
            "Chunked and single-pass prefill diverged past the sliding window"
        )
    }

    @Test("Gemma3n TokenIterator completes a prompt longer than the window")
    func gemma3nTokenIteratorLongPromptCompletes() throws {
        let model = try makeGemma3nTextModel(seed: 303)

        // 24 tokens against sliding_window 8: the prefill mask is a BOOLEAN
        // array (RotatingKVCache past the window) and decode continues past
        // the window. The production single-stream pipeline must complete
        // without trapping.
        let promptTokens = (0 ..< 24).map { Int32(($0 * 5 + 2) % 63) }
        let parameters = GenerateParameters(
            maxTokens: 4, temperature: 0, prefillStepSize: 8)
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(promptTokens)),
            model: model,
            parameters: parameters
        )

        var generated: [Int] = []
        while let token = iterator.next() {
            generated.append(token)
        }

        #expect(generated.count == 4)
        #expect(generated.allSatisfy { (0 ..< 64).contains($0) })
    }

    /// Tiny Gemma3n text model: 2 layers (one sliding, one full) with a
    /// sliding window of 8 so tests cross the window in a handful of tokens.
    private func makeGemma3nTextModel(seed: UInt64) throws -> Gemma3nTextModel {
        let configuration: Gemma3nTextConfiguration = try decodeConfiguration(
            """
            {
              "model_type": "gemma3n",
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "intermediate_size": 32,
              "num_attention_heads": 4,
              "head_dim": 4,
              "rms_norm_eps": 0.000001,
              "vocab_size": 64,
              "num_key_value_heads": 2,
              "num_kv_shared_layers": 0,
              "vocab_size_per_layer_input": 64,
              "sliding_window": 8,
              "max_position_embeddings": 128,
              "rope_local_base_freq": 10000.0,
              "rope_theta": 1000000.0,
              "final_logit_softcapping": 30.0,
              "layer_types": ["sliding_attention", "full_attention"],
              "hidden_size_per_layer_input": 8,
              "altup_num_inputs": 2,
              "altup_correct_scale": true,
              "altup_active_idx": 0,
              "laurel_rank": 4
            }
            """
        )

        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = Gemma3nTextModel(config: configuration)
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
                // RotatingKVCache.keep is private; read it via metaState[0] (= keep),
                // matching the pattern used in the batched cache factories.
                let keep = Int(rotatingCache.metaState.first ?? "0") ?? 0
                return BatchRotatingKVCache(
                    maxSize: rotatingCache.maxSize ?? 0,
                    leftPadding: leftPadding,
                    keep: keep
                )
            } else {
                return BatchKVCache(leftPadding: leftPadding)
            }
        }
        let logits = model.callAsFunction(input, cache: cache)
        materialize(arrays: [logits], cache: cache)
        return (logits, cache, leftPadding)
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
