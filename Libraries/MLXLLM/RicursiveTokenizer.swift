// Adapted from Ricursive Intelligence's Apache-2.0 Ripple chat template.
// See ThirdPartyLicenses/Ricursive.txt for attribution and license.

import Foundation
import MLXLMCommon

/// Ripple's full chat format; its tokenizer_config.json contains only basic ChatML.
struct RicursiveTokenizer: Tokenizer {
    private let upstream: any Tokenizer

    init(_ upstream: any Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    func makeStreamingDetokenizer() -> any StreamingDetokenizer {
        upstream.makeStreamingDetokenizer()
    }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let prompt = try Self.render(
            messages: messages, tools: tools, additionalContext: additionalContext)
        return upstream.encode(text: prompt, addSpecialTokens: false)
    }

    static func render(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> String {
        let system =
            messages.first?["role"] as? String == "system"
            ? messages.first?["content"] as? String : nil
        var prompt = ""
        if let tools, tools.isEmpty == false {
            prompt += "<|im_start|>system\n"
            if let system { prompt += system + "\n\n" }
            prompt +=
                "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                + "You are provided with function signatures within <tools> XML tags:\n<tools>"
            for tool in tools { prompt += "\n" + (try json(tool)) }
            prompt +=
                "\n</tools>\n\nFor each function call, return exactly one valid JSON object with "
                + "function name and arguments inside XML tool-call tags. The arguments field must "
                + "be a JSON object, not a string, and strings must use valid JSON escaping:\n"
                + "<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n"
                + "</tool_call><|im_end|>\n"
        } else if let system {
            prompt += "<|im_start|>system\n" + system + "<|im_end|>\n"
        }

        let clearThinking = additionalContext?["clear_thinking"] as? Bool ?? false
        for (index, message) in messages.enumerated() {
            let role = message["role"] as? String ?? ""
            var content = message["content"] as? String ?? ""
            if role == "assistant", let reasoning = message["reasoning_content"] as? String,
                reasoning.isEmpty == false, clearThinking == false,
                content.drop(while: \.isWhitespace).hasPrefix("<think>") == false
            {
                content = "<think>\n" + reasoning + "\n</think>\n" + content
            }
            let calls = message["tool_calls"] as? [[String: any Sendable]] ?? []
            if role == "user" || (role == "system" && index != 0)
                || (role == "assistant" && calls.isEmpty)
            {
                prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n"
            } else if role == "assistant" {
                prompt += "<|im_start|>assistant"
                if content.isEmpty == false { prompt += "\n" + content }
                for call in calls {
                    let function = call["function"] as? [String: any Sendable] ?? call
                    prompt +=
                        "\n<tool_call>\n{\"name\": "
                        + (try json(function["name"] ?? NSNull()))
                        + ", \"arguments\": " + (try json(function["arguments"] ?? NSNull()))
                        + "}\n</tool_call>"
                }
                prompt += "<|im_end|>\n"
            } else if role == "tool" {
                if index == 0 || messages[index - 1]["role"] as? String != "tool" {
                    prompt += "<|im_start|>user"
                }
                prompt += "\n<tool_response>\n" + content + "\n</tool_response>"
                if index == messages.count - 1 || messages[index + 1]["role"] as? String != "tool" {
                    prompt += "<|im_end|>\n"
                }
            }
        }
        return prompt + "<|im_start|>assistant\n"
    }

    private static func json(_ value: any Sendable) throws -> String {
        // Swift dictionaries have no insertion order. Sort keys and match Jinja's JSON spacing.
        let data = try JSONSerialization.data(
            withJSONObject: value,
            options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes])
        var result = ""
        var quoted = false
        var escaped = false
        for character in String(decoding: data, as: UTF8.self) {
            result.append(character)
            if escaped {
                escaped = false
            } else if character == "\\" && quoted {
                escaped = true
            } else if character == "\"" {
                quoted.toggle()
            } else if quoted == false && (character == "," || character == ":") {
                result.append(" ")
            }
        }
        return result
    }
}
