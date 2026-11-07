import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class RicursiveFactoryTests: XCTestCase {
    func testPublished65BConfiguration() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "ripple-65b-config", withExtension: "json"))
        let config = try JSONDecoder().decode(
            RicursiveConfiguration.self, from: Data(contentsOf: url))
        XCTAssertEqual(config.hiddenSize, 5120)
        XCTAssertEqual(config.hiddenLayers, 80)
        XCTAssertEqual(config.attentionHeads, 80)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.vocabularySize, 100_352)
        XCTAssertEqual(config.slidingWindow, 1024)
        XCTAssertEqual(config.layerTypes.filter { $0 == "sliding_attention" }.count, 64)
        XCTAssertEqual(config.layerTypes.filter { $0 == "full_attention" }.count, 16)
        XCTAssertTrue(config.useValueResidual)
        XCTAssertTrue(config.useGatedAttention)
        XCTAssertEqual(config.ffnClampThreshold, 20)
    }

    func testLoadPreservesChatConventionsAndUsesRippleTemplate() async throws {
        let data = Data(
            """
            {
                "model_type": "ricursive", "hidden_size": 8, "num_hidden_layers": 2,
                "intermediate_size": 16, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 8, "vocab_size": 16, "sliding_window": 4,
                "layer_types": ["sliding_attention", "full_attention"],
                "use_valres": true, "eos_token_id": [100265, 100257]
            }
            """.utf8)
        let config = try JSONDecoder().decode(RicursiveConfiguration.self, from: data)
        let model = withRandomState(MLXRandom.RandomState(seed: 17)) { RicursiveModel(config) }
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "RicursiveFactoryTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try data.write(to: directory.appending(component: "config.json"))
        try save(
            arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
            url: directory.appending(component: "model.safetensors"))

        let context = try await LLMModelFactory.shared._load(
            configuration: .init(directory: directory), tokenizerLoader: RawTokenizerLoader())
        XCTAssertTrue(context.model is RicursiveModel)
        XCTAssertEqual(context.configuration.eosTokenIds, [100265, 100257])
        XCTAssertEqual(context.configuration.toolCallFormat, .json)
        XCTAssertEqual(context.configuration.reasoningConfig?.startDelimiter, "<think>")
        XCTAssertEqual(
            context.configuration.reasoningConfig?.promptStrategy, ReasoningPromptStrategy.none)
        XCTAssertEqual(context.configuration.reasoningConfig?.isSpecialToken, false)

        let input = UserInput(
            chat: [.system("Use tools."), .user("What time is it?")],
            tools: [["type": "function", "function": ["name": "clock"]]])
        let prepared = try await context.processor.prepare(input: input)
        let rendered = context.tokenizer.decode(tokenIds: prepared.text.tokens.asArray(Int.self))
        XCTAssertTrue(rendered.contains("<tools>"))
        XCTAssertTrue(rendered.contains("\"clock\""))
        XCTAssertTrue(rendered.contains("<|im_start|>user\nWhat time is it?<|im_end|>\n"))
        XCTAssertTrue(rendered.hasSuffix("<|im_start|>assistant\n"))
    }

    func testRegistryExposesQuantizedCheckpoint() {
        XCTAssertTrue(LLMRegistry.shared.contains(id: LLMRegistry.ripple_65b_preview_4bit.name))
    }
}

private struct RawTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { RawTokenizer() }
}

// The factory must supply Ripple's template even when the upstream loader has only basic ChatML.
private struct RawTokenizer: Tokenizer {
    var bosToken: String? { "<|endoftext|>" }
    var eosToken: String? { "<|im_end|>" }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(String.UnicodeScalarView(tokenIds.compactMap { UnicodeScalar($0) }))
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == eosToken ? 100265 : nil
    }

    func convertIdToToken(_ id: Int) -> String? { id == 100265 ? eosToken : nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        XCTFail("The factory should use Ripple's full template")
        return []
    }
}
