// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXLMCommon

struct RicursiveReasoningTests {
    private static let config = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .none,
        startPolicy: .responseStart, whitespacePolicy: .singleNewline)

    struct Fixture: Sendable {
        let text: String
        let reasoning: String
        let response: String
        let closed: Bool
    }

    // Expected strings follow ripple_format.py's startswith/partition and one-newline removal.
    static let fixtures = [
        Fixture(
            text: "<think>\n  reason\n\n</think>\n    def f():\n",
            reasoning: "  reason\n", response: "    def f():\n", closed: true),
        Fixture(
            text: "<think>\n\nr\n\n</think>\n\nA",
            reasoning: "\nr\n", response: "\nA", closed: true),
        Fixture(text: "<think>\n</think>\nA", reasoning: "", response: "A", closed: true),
        Fixture(text: "<think>  r  </think>  A", reasoning: "  r  ", response: "  A", closed: true),
        Fixture(
            text: "<think>\r\nr\r\n</think>\r\nA",
            reasoning: "\r\nr\r", response: "\r\nA", closed: true),
        Fixture(
            text: "<think>\ntruncated\n", reasoning: "truncated\n", response: "", closed: false),
        Fixture(
            text: "<think>\npartial\n</thi", reasoning: "partial\n</thi", response: "",
            closed: false),
        Fixture(text: "<thi", reasoning: "", response: "<thi", closed: false),
        Fixture(text: "<think>\n", reasoning: "", response: "", closed: false),
        Fixture(
            text: "Use <think>notes</think> tags.", reasoning: "",
            response: "Use <think>notes</think> tags.", closed: false),
        Fixture(
            text: " <think>notes</think>", reasoning: "",
            response: " <think>notes</think>", closed: false),
        Fixture(
            text: "<thinking>literal<think>x</think>", reasoning: "",
            response: "<thinking>literal<think>x</think>", closed: false),
        Fixture(
            text: "<think>x</think>Use <think>notes</think>", reasoning: "x",
            response: "Use <think>notes</think>", closed: true),
        Fixture(
            text: "<think>\n🙂\n</think>\n café", reasoning: "🙂", response: " café", closed: true),
    ]

    @Test("Ripple reasoning preserves exact content at every stream split", arguments: fixtures)
    func everySplit(fixture: Fixture) {
        let boundaries = Array(fixture.text.unicodeScalars.indices) + [fixture.text.endIndex]
        for boundary in boundaries {
            check(
                [String(fixture.text[..<boundary]), String(fixture.text[boundary...])],
                fixture: fixture)
        }
        check(fixture.text.unicodeScalars.map(String.init), fixture: fixture)
    }

    @Test("A primed reasoning span preserves generated whitespace and cannot reopen")
    func primedSpan() {
        let fixtures = [
            Fixture(
                text: "\n  thought\n</think>\n    code<think>literal</think>",
                reasoning: "\n  thought", response: "    code<think>literal</think>", closed: true),
            Fixture(
                text: "  unfinished\n</thi", reasoning: "  unfinished\n</thi",
                response: "", closed: false),
        ]
        for fixture in fixtures {
            let boundaries = Array(fixture.text.unicodeScalars.indices) + [fixture.text.endIndex]
            for boundary in boundaries {
                check(
                    [String(fixture.text[..<boundary]), String(fixture.text[boundary...])],
                    fixture: fixture, primedInside: true)
            }
        }
    }

    @Test("Existing reasoning policies remain the defaults")
    func defaultsRemainUnchanged() {
        let config = ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .none)
        #expect(config.startPolicy == .anywhere)
        #expect(config.whitespacePolicy == .trim)
        var emitter = ReasoningEventEmitter(config: config, primedInside: false)
        let segments = emitter.process("prefix<think>\n  thought  \n</think>\n  answer")
        #expect(segments == [.response("prefix"), .reasoning("thought"), .response("answer")])
    }

    @Test("Historical literal delimiters cannot prime Ripple's assistant response")
    func promptPrimingRequiresFinalOpeningDelimiter() {
        #expect(
            ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: "<|im_start|>assistant\n<think>\n", config: Self.config))
        for tail in [
            "<|im_start|>user\nExplain <think> tags.<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>assistant\nA literal <think> tag.<|im_end|>\n<|im_start|>assistant\n",
            "<think>already supplied content",
        ] {
            #expect(
                ReasoningEventEmitter.promptEndsInsideReasoning(
                    renderedPromptTail: tail, config: Self.config) == false)
        }
    }

    private func check(_ chunks: [String], fixture: Fixture, primedInside: Bool = false) {
        var emitter = ReasoningEventEmitter(config: Self.config, primedInside: primedInside)
        var segments = chunks.flatMap { emitter.process($0) }
        segments += emitter.finalize()
        let reasoning = segments.compactMap {
            if case .reasoning(let text) = $0 { text } else { nil }
        }.joined()
        let response = segments.compactMap {
            if case .response(let text) = $0 { text } else { nil }
        }.joined()
        #expect(reasoning == fixture.reasoning)
        #expect(response == fixture.response)
        #expect(emitter.hasClosedReasoning == fixture.closed)
        #expect(emitter.finalize().isEmpty)
    }
}
