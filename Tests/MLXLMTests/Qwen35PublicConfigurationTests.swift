// Copyright © 2026 Apple Inc.

import Foundation
import MLXLLM
import Testing

/// Deliberately uses a normal import: applications need these dimensions
/// before running either the target or the drafter.
@Suite struct Qwen35PublicConfigurationTests {
    @Test(arguments: [false, true])
    func targetAndDrafterExposeReadOnlyGeometry(moe: Bool) throws {
        let json = """
            {"model_type":"qwen3_5","text_config":{
              "hidden_size":16,"num_hidden_layers":2,"intermediate_size":32,
              "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
              "linear_num_value_heads":2,"linear_num_key_heads":1,"linear_key_head_dim":32,
              "linear_value_head_dim":8,"linear_conv_kernel_dim":2,"vocab_size":64,
              "full_attention_interval":2,"mtp_num_hidden_layers":1,
              "num_experts":\(moe ? 4 : 0),"num_experts_per_tok":\(moe ? 2 : 0),
              "shared_expert_intermediate_size":16,"moe_intermediate_size":24}}
            """
        let configuration = try JSONDecoder().decode(
            Qwen35Configuration.self, from: Data(json.utf8))
        let target: Qwen35Model = moe ? Qwen35MoEModel(configuration) : Qwen35Model(configuration)
        let textTarget = Qwen35TextModel(target.configuration)
        let drafter = Qwen35MTPDraftModel(target.configuration)
        for geometry in [target.configuration, textTarget.configuration, drafter.configuration] {
            #expect(geometry.hiddenSize == 16)
            #expect(geometry.hiddenLayers == 2)
            #expect(geometry.intermediateSize == 32)
            #expect(geometry.attentionHeads == 2)
            #expect(geometry.kvHeads == 1)
            #expect(geometry.headDim == 8)
            #expect(geometry.linearNumValueHeads == 2)
            #expect(geometry.linearNumKeyHeads == 1)
            #expect(geometry.linearKeyHeadDim == 32)
            #expect(geometry.linearValueHeadDim == 8)
            #expect(geometry.linearConvKernelDim == 2)
            #expect(geometry.vocabularySize == 64)
            #expect(geometry.fullAttentionInterval == 2)
            #expect(geometry.mtpNumHiddenLayers == 1)
            #expect(geometry.numExperts == (moe ? 4 : 0))
            #expect(geometry.numExpertsPerTok == (moe ? 2 : 0))
            #expect(geometry.sharedExpertIntermediateSize == 16)
            #expect(geometry.moeIntermediateSize == 24)
        }
    }
}
