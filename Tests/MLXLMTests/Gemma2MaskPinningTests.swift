// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Pins the Gemma2 attention-mask numerics introduced by
/// `applyGemma2AttentionMask` (Libraries/MLXLLM/Models/Gemma2.swift).
///
/// The legacy path built a BOOLEAN causal mask with the array helper and then
/// ADDED it to the softcapped attention logits — a 0/1 nudge instead of
/// exclusion — so multi-token prefill attended to future positions. The fixed
/// path applies boolean masks with `where` and a most-negative-FINITE fill
/// (`MLXArray.maskFill`), which both restores causality and keeps fully-masked
/// (left-padded) query rows finite in reduced precision.
///
/// These tests pin, with a tiny random-weight Gemma2 built from a JSON config
/// (no downloads):
/// 1. prefill causality — trailing-token changes cannot alter earlier logits
///    (fails under the additive-bool mask),
/// 2. decode/prefill parity — incremental decode matches the corresponding
///    full-prefill position (pins RoPE/cache offset handling),
/// 3. softcapping under extreme weight scales — logits stay finite and bounded
///    by `final_logit_softcapping`,
/// 4. batched (left-padded) decode/prefill parity in float32 — pins the
///    boolean `.array` mask blend, including the 5-D GQA head-group expansion,
/// 5. float16 left-padded prefill stays finite — pins the finite `maskFill`
///    choice (a `-1e9` fill overflows to `-inf` in fp16 and an all-masked
///    padded row would softmax to NaN and poison later layers).
///
/// No Metal-availability guard, matching BatchModelRegressionTests: the real
/// gate is the macOS/Metal CI, which runs these end to end.
@Suite("Gemma2 mask pinning", .serialized)
struct Gemma2MaskPinningTests {

    // MARK: - Fixture

    private func makeModel(seed: UInt64) throws -> Gemma2Model {
        let configuration: Gemma2Configuration = try decodeConfiguration(
            """
            {
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "intermediate_size": 32,
              "num_attention_heads": 4,
              "head_dim": 4,
              "rms_norm_eps": 0.000001,
              "vocab_size": 64,
              "num_key_value_heads": 2,
              "rope_theta": 10000.0,
              "rope_traditional": false,
              "attn_logit_softcapping": 50.0,
              "final_logit_softcapping": 30.0,
              "query_pre_attn_scalar": 16.0
            }
            """
        )

        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = Gemma2Model(configuration)
            eval(model)
            return model
        }
    }

    private let prompt: [Int32] = [3, 9, 27, 17, 40, 5, 61, 33, 12, 48]

    // MARK: - 1. Prefill causality

    @Test("Changing trailing prompt tokens leaves earlier prefill logits unchanged")
    func prefillIsCausal() throws {
        let model = try makeModel(seed: 400)

        var altered = prompt
        altered[8] = 55
        altered[9] = 21
        let sharedPrefix = 8

        let base = prefillSingle(model: model, prompt: prompt).logits
        let changed = prefillSingle(model: model, prompt: altered).logits

        let basePrefix = base[0..., ..<sharedPrefix, 0...].asType(.float32)
        let changedPrefix = changed[0..., ..<sharedPrefix, 0...].asType(.float32)
        #expect(basePrefix.shape == changedPrefix.shape)
        // Under the legacy additive-bool mask nothing excluded future
        // positions during multi-token prefill, so the trailing-token edit
        // leaked into every earlier position's logits.
        #expect(
            maxAbsDifference(basePrefix, changedPrefix) <= 1e-5,
            "Prefill attended to future positions: earlier logits changed with trailing tokens"
        )

        // Sanity that the comparison is not vacuous: the edited positions
        // themselves must produce different logits.
        let baseSuffix = base[0..., sharedPrefix..., 0...].asType(.float32)
        let changedSuffix = changed[0..., sharedPrefix..., 0...].asType(.float32)
        #expect(maxAbsDifference(baseSuffix, changedSuffix) > 1e-3)
    }

    // MARK: - 2. Decode step parity (offset handling)

    @Test("Incremental decode matches the corresponding full-prefill position")
    func decodeMatchesFullPrefill() throws {
        let model = try makeModel(seed: 401)

        let full = prefillSingle(model: model, prompt: prompt).logits
        let fullLast = full[0..., (prompt.count - 1)..., 0...].asType(.float32)

        let partial = prefillSingle(model: model, prompt: Array(prompt.dropLast()))
        let decodeInput = MLXArray([prompt[prompt.count - 1]])[.newAxis, .ellipsis]
        let decodeLogits = model.callAsFunction(decodeInput, cache: partial.cache)
        materialize(arrays: [decodeLogits], cache: partial.cache)

        let decoded = decodeLogits.asType(.float32)
        #expect(decoded.shape == fullLast.shape)
        #expect(
            maxAbsDifference(decoded, fullLast) <= 0.01,
            "n=1 decode diverged from the same position of a full prefill"
        )
    }

    // MARK: - 3. Softcapping under extreme scales

    @Test("Softcapping keeps extreme-scale logits finite and bounded")
    func softcappingBoundsExtremeScales() throws {
        let model = try makeModel(seed: 402)
        // Inflate every weight so raw attention scores far exceed the attn
        // softcap (50) and raw output logits far exceed the final softcap
        // (30); tanh-based capping must keep everything finite and bounded.
        model.apply { $0 * 8 }
        eval(model)

        let logits = prefillSingle(model: model, prompt: prompt).logits.asType(.float32)
        let maxMagnitude = abs(logits).max().item(Float.self)

        #expect(maxMagnitude.isFinite, "Extreme weight scales produced non-finite logits")
        // tanh(x / 30) * 30 bounds every logit by the final softcap.
        #expect(maxMagnitude <= 30.001)
        #expect(maxMagnitude > 0)
    }

    // MARK: - 4. Batched (left-padded) parity, float32

    @Test("Left-padded batched prefill matches single-stream prefill")
    func batchedPrefillMatchesSingle() throws {
        let model = try makeModel(seed: 403)
        let prompts: [[Int32]] = [
            [11, 12, 13, 14, 15],
            [21, 22, 23],
        ]

        let singles = prompts.map { prefillSingle(model: model, prompt: $0) }
        let batched = prefillBatch(model: model, prompts: prompts)

        for (index, prompt) in prompts.enumerated() {
            let pad = batched.leftPadding[index]
            let batchValid = batched.logits[index ..< (index + 1), pad..., 0...].asType(.float32)
            let single = singles[index].logits.asType(.float32)

            #expect(batchValid.shape == single.shape)
            #expect(
                maxAbsDifference(batchValid, single) <= 0.01,
                "Batched prefill diverged for prompt \(prompt)"
            )
        }
    }

    @Test("Left-padded batched decode matches single-stream decode")
    func batchedDecodeMatchesSingle() throws {
        let model = try makeModel(seed: 404)
        let prompts: [[Int32]] = [
            [11, 12, 13, 14, 15],
            [21, 22, 23],
        ]
        let decodeTokens: [Int32] = [31, 32]

        let singles = prompts.enumerated().map { index, prompt in
            let result = prefillSingle(model: model, prompt: prompt)
            let decodeInput = MLXArray([decodeTokens[index]])[.newAxis, .ellipsis]
            let decodeLogits = model.callAsFunction(decodeInput, cache: result.cache)
            materialize(arrays: [decodeLogits], cache: result.cache)
            return decodeLogits
        }

        let batched = prefillBatch(model: model, prompts: prompts)
        let batchedInput = MLXArray(decodeTokens, [decodeTokens.count, 1])
        let batchedLogits = model.callAsFunction(batchedInput, cache: batched.cache)
        materialize(arrays: [batchedLogits], cache: batched.cache)

        for index in prompts.indices {
            let batchRow = batchedLogits[index ..< (index + 1), 0..., 0...].asType(.float32)
            let single = singles[index].asType(.float32)

            #expect(batchRow.shape == single.shape)
            // With 4 heads over 2 KV heads the scores are 5-D here, so this
            // also pins the [B, 1, L, S] → head-group mask expansion inside
            // applyGemma2AttentionMask.
            #expect(
                maxAbsDifference(batchRow, single) <= 0.01,
                "Batched decode diverged for prompt index \(index)"
            )
        }
    }

    // MARK: - 5. float16 finite fill (maskFill)

    @Test("float16 left-padded batched prefill stays finite and matches single")
    func float16PaddedPrefillStaysFinite() throws {
        let model = try makeModel(seed: 405)
        // Match real-world deployment dtype. This is the case the finite
        // `maskFill` exists for: a -1e9 fill cast to float16 overflows to
        // -inf, an all-masked (fully left-padded) query row softmaxes to NaN,
        // and the NaN poisons every later layer's K/V for that row.
        model.apply {
            if $0.dtype == .float32 {
                $0.asType(.float16)
            } else {
                $0
            }
        }
        eval(model)

        let prompts: [[Int32]] = [
            [11, 12, 13, 14, 15],
            [21, 22, 23],
        ]

        let singles = prompts.map { prefillSingle(model: model, prompt: $0) }
        let batched = prefillBatch(model: model, prompts: prompts)

        for (index, prompt) in prompts.enumerated() {
            let pad = batched.leftPadding[index]
            let batchValid = batched.logits[index ..< (index + 1), pad..., 0...].asType(.float32)
            let single = singles[index].logits.asType(.float32)

            let maxMagnitude = abs(batchValid).max().item(Float.self)
            #expect(
                maxMagnitude.isFinite,
                "float16 padded prefill produced non-finite logits for prompt \(prompt)"
            )
            #expect(batchValid.shape == single.shape)
            #expect(
                maxAbsDifference(batchValid, single) <= 0.05,
                "float16 batched prefill diverged for prompt \(prompt)"
            )
        }
    }

    // MARK: - Helpers

    private func decodeConfiguration<T: Decodable>(_ json: String) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(json.utf8))
    }

    private func prefillSingle(
        model: Gemma2Model, prompt: [Int32]
    ) -> (logits: MLXArray, cache: [KVCache]) {
        let cache = model.newCache(parameters: nil)
        let input = MLXArray(prompt)[.newAxis, .ellipsis]
        let logits = model.callAsFunction(input, cache: cache)
        materialize(arrays: [logits], cache: cache)
        return (logits, cache)
    }

    /// Left-padded ragged batch prefill over hand-built `BatchKVCache`s
    /// (Gemma2's per-layer caches are all `KVCacheSimple`), mirroring the
    /// batched cache factories used by the engine.
    private func prefillBatch(
        model: Gemma2Model, prompts: [[Int32]]
    ) -> (logits: MLXArray, cache: [KVCache], leftPadding: [Int]) {
        let maxLength = prompts.map(\.count).max() ?? 0
        let leftPadding = prompts.map { maxLength - $0.count }

        let flat = zip(prompts, leftPadding).flatMap { prompt, pad in
            Array(repeating: Int32(0), count: pad) + prompt
        }
        let input = MLXArray(flat, [prompts.count, maxLength])
        let cache: [KVCache] = model.newCache(parameters: nil).map { _ in
            BatchKVCache(leftPadding: leftPadding)
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
