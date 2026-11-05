// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import Testing
import Tokenizers

@testable import MLXLLM
@testable import MLXLMCommon

private let ricursiveTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()
private let ricursiveDownloader: any Downloader = #hubDownloader()

struct RicursiveTokenizerTests {
    private static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "Tests/MLXLMTests/Resources/Ricursive")

    @Test("Ripple's real Swift tokenizer matches the offline HF token and template fixtures")
    func offlineTokenizerParity() async throws {
        let upstream = try await ricursiveTokenizerLoader.load(from: Self.fixtures)
        try Self.check(tokenizer: RicursiveTokenizer(upstream))
    }

    @Test("Ripple's complete pinned production tokenizer matches the same HF reference")
    func completeTokenizerParity() async throws {
        let reference = try Self.reference()
        let directory: URL
        if let local = ProcessInfo.processInfo.environment["RIPPLE_TOKENIZER_DIRECTORY"] {
            directory = URL(fileURLWithPath: local)
        } else {
            directory = try await ricursiveDownloader.download(
                id: reference.repository, revision: reference.revision,
                matching: ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
                useLatest: false, progressHandler: { _ in })
        }
        let upstream = try await ricursiveTokenizerLoader.load(from: directory)
        try Self.check(tokenizer: RicursiveTokenizer(upstream))
    }

    @Test("The production LLM factory selects Ripple's full template and both EOS tokens")
    func factoryTemplateSelection() async throws {
        let reference = try Self.reference()
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "RicursiveTokenizerTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
            try FileManager.default.copyItem(
                at: Self.fixtures.appending(path: name), to: directory.appending(path: name))
        }
        var config = try JSONDecoder().decode(
            [String: JSONValue].self,
            from: Data(contentsOf: Self.fixtures.appending(path: "ricursive-config.json")))
        config["eos_token_id"] = .array(reference.eosTokenIds.map(JSONValue.int))
        try JSONEncoder().encode(config).write(to: directory.appending(path: "config.json"))
        try FileManager.default.copyItem(
            at: Self.fixtures.appending(path: "ricursive-weights.safetensors"),
            to: directory.appending(path: "model.safetensors"))

        let context = try await LLMModelFactory.shared.load(
            from: directory, using: ricursiveTokenizerLoader)
        #expect(context.configuration.eosTokenIds == Set(reference.eosTokenIds))
        for fixture in reference.cases {
            let input = UserInput(
                messages: fixture.messages.map { $0.mapValues(\.sendableValue) },
                tools: fixture.tools?.map { $0.mapValues(\.sendableValue) },
                additionalContext: fixture.context?.mapValues(\.sendableValue))
            let prepared = try await context.processor.prepare(input: input)
            #expect(prepared.text.tokens.asArray(Int32.self).map(Int.init) == fixture.ids)
        }
    }

    private static func check(tokenizer: any MLXLMCommon.Tokenizer) throws {
        let reference = try reference()
        for (token, id) in [
            "<|endoftext|>": 100_257, "<|im_start|>": 100_264,
            "<|im_end|>": 100_265, "<|pad|>": 100_277,
        ] {
            #expect(tokenizer.convertTokenToId(token) == id)
            #expect(tokenizer.encode(text: token, addSpecialTokens: false) == [id])
        }
        #expect(tokenizer.eosTokenId == 100_265)
        for probe in reference.probes {
            #expect(tokenizer.encode(text: probe.text, addSpecialTokens: false) == probe.ids)
            #expect(tokenizer.decode(tokenIds: probe.ids, skipSpecialTokens: false) == probe.text)
        }
        for fixture in reference.cases {
            let ids = try tokenizer.applyChatTemplate(
                messages: fixture.messages.map { $0.mapValues(\.sendableValue) },
                tools: fixture.tools?.map { $0.mapValues(\.sendableValue) },
                additionalContext: fixture.context?.mapValues(\.sendableValue))
            #expect(ids == fixture.ids)
            #expect(tokenizer.decode(tokenIds: ids, skipSpecialTokens: false) == fixture.rendered)
        }
    }

    private static func reference() throws -> Reference {
        try JSONDecoder().decode(
            Reference.self,
            from: Data(contentsOf: fixtures.appending(path: "tokenizer-reference.json")))
    }

    private struct Reference: Decodable {
        let repository: String
        let revision: String
        let eosTokenIds: [Int]
        let probes: [Probe]
        let cases: [Case]

        enum CodingKeys: String, CodingKey {
            case repository, revision, probes, cases
            case eosTokenIds = "eos_token_ids"
        }

        struct Probe: Decodable {
            let text: String
            let ids: [Int]
        }

        struct Case: Decodable {
            let name: String
            let messages: [[String: JSONValue]]
            let tools: [[String: JSONValue]]?
            let context: [String: JSONValue]?
            let rendered: String
            let ids: [Int]
        }
    }
}
