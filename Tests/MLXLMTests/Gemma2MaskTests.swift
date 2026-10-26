// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct Gemma2MaskTests {

    @Test("A prefill position does not attend to later tokens")
    func prefillIsCausal() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyGemma2()
            let tokens: [Int32] = [3, 7, 1, 12, 5, 9]
            let full = model(
                MLXArray(tokens).reshaped([1, tokens.count]),
                cache: try model.newCache(parameters: nil))
            for end in 1 ..< tokens.count {
                let prefix = model(
                    MLXArray(Array(tokens[..<end])).reshaped([1, end]),
                    cache: try model.newCache(parameters: nil))
                #expect(
                    maxAbsDifference(full[0, end - 1], prefix[0, end - 1]) < 1e-4,
                    "position \(end - 1)")
            }
        }
    }

    @Test("Fully masked left-padding rows stay finite in float16")
    func float16PaddingRowsStayFinite() throws {
        let attention = Gemma2Attention(try tinyGemma2Configuration())
        attention.update(parameters: attention.parameters().mapValues { $0.asType(.float16) })

        // Row 0's first two query positions can only attend its left padding.
        let cache = BatchKVCache(leftPadding: [2, 0])
        let mask = cache.makeMask(n: 4, windowSize: nil, returnArray: false)
        let output = attention(
            MLXArray.ones([2, 4, 16]).asType(.float16), mask: mask, cache: cache)
        eval(output)
        #expect(isNaN(output).any().item(Bool.self) == false)
        #expect(isInf(output).any().item(Bool.self) == false)
    }
}

func tinyGemma2Configuration() throws -> Gemma2Configuration {
    try JSONDecoder().decode(
        Gemma2Configuration.self,
        from: Data(
            """
            {
              "hidden_size": 16, "num_hidden_layers": 2, "intermediate_size": 32,
              "num_attention_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-6,
              "vocab_size": 32, "num_key_value_heads": 1, "attn_logit_softcapping": 50.0,
              "final_logit_softcapping": 30.0, "query_pre_attn_scalar": 144.0
            }
            """.utf8))
}

func tinyGemma2() throws -> Gemma2Model {
    let configuration = try tinyGemma2Configuration()
    return withRandomState(MLXRandom.RandomState(seed: 13)) {
        let model = Gemma2Model(configuration)
        eval(model)
        return model
    }
}
