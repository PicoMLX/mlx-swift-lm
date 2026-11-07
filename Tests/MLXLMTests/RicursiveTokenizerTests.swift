// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

struct RicursiveTokenizerTests {
    @Test(
        "Ripple renders the official full template",
        arguments: [
            "system_user", "code_whitespace_unicode", "reasoning_history", "existing_thinking",
            "clear_thinking", "tools_round_trip", "tool_only_history", "empty_tool_content",
            "later_system",
        ])
    func officialTemplate(name: String) throws {
        let url = try #require(
            Bundle.module.url(forResource: "tokenizer-reference", withExtension: "json"))
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
        let fixture = try #require(reference.cases.first { $0.name == name })
        let rendered = try RicursiveTokenizer.render(
            messages: fixture.messages.map { $0.mapValues(\.sendableValue) },
            tools: fixture.tools?.map { $0.mapValues(\.sendableValue) },
            additionalContext: fixture.context?.mapValues(\.sendableValue))
        #expect(rendered == fixture.rendered)
    }

    private struct Reference: Decodable {
        let cases: [Case]

        struct Case: Decodable {
            let name: String
            let messages: [[String: JSONValue]]
            let tools: [[String: JSONValue]]?
            let context: [String: JSONValue]?
            let rendered: String
        }
    }
}
