import Foundation
import MLXLMCommon
import Testing

struct ToolCallIdTests {
    @Test("Tool message round-trips tool_call_id through DefaultMessageGenerator")
    func testToolMessageRoundTripsId() throws {
        let generator = DefaultMessageGenerator()
        let message = Chat.Message.tool("ok", id: "call_1")

        let dict = generator.generate(message: message)

        #expect(dict["role"] as? String == "tool")
        #expect(dict["content"] as? String == "ok")
        #expect(dict["tool_call_id"] as? String == "call_1")
    }

    @Test("Assistant message round-trips tool_calls through DefaultMessageGenerator")
    func testAssistantMessageRoundTripsToolCalls() throws {
        let tc1 = ToolCall(
            id: "call_a",
            function: .init(
                name: "get_weather",
                arguments: ["location": .string("Paris")]
            )
        )
        let tc2 = ToolCall(
            id: "call_b",
            function: .init(
                name: "get_time",
                arguments: ["timezone": .string("UTC")]
            )
        )

        let generator = DefaultMessageGenerator()
        let message = Chat.Message.assistant("", toolCalls: [tc1, tc2])

        let dict = generator.generate(message: message)

        #expect(dict["role"] as? String == "assistant")

        let calls = try #require(dict["tool_calls"] as? [[String: any Sendable]])
        #expect(calls.count == 2)

        let first = calls[0]
        #expect(first["id"] as? String == "call_a")
        #expect(first["type"] as? String == "function")
        let firstFn = try #require(first["function"] as? [String: any Sendable])
        #expect(firstFn["name"] as? String == "get_weather")
        let firstArgs = try #require(firstFn["arguments"] as? [String: any Sendable])
        #expect(firstArgs["location"] as? String == "Paris")

        let second = calls[1]
        #expect(second["id"] as? String == "call_b")
        #expect(second["type"] as? String == "function")
        let secondFn = try #require(second["function"] as? [String: any Sendable])
        #expect(secondFn["name"] as? String == "get_time")
        let secondArgs = try #require(secondFn["arguments"] as? [String: any Sendable])
        #expect(secondArgs["timezone"] as? String == "UTC")
    }

    @Test("Plain user message does not emit tool_call_id or tool_calls keys")
    func testPlainMessageDoesNotEmitToolKeys() throws {
        let generator = DefaultMessageGenerator()
        let message = Chat.Message.user("hi")

        let dict = generator.generate(message: message)

        #expect(dict["tool_call_id"] == nil)
        #expect(dict["tool_calls"] == nil)
    }

    @Test("ToolCallProcessor assigns an id when parser does not provide one")
    func testToolCallProcessorAssignsIdFallback() throws {
        let processor = ToolCallProcessor(format: .json)
        let content = "<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>"

        _ = processor.processChunk(content)
        processor.processEOS()

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "x")
        let id = try #require(toolCall.id)
        #expect(!id.isEmpty)
        #expect(id.hasPrefix("call_"))
    }

    @Test("ToolCallProcessor assigns unique ids across multiple tool calls")
    func testToolCallProcessorUniqueIds() throws {
        let processor = ToolCallProcessor(format: .json)
        let content =
            "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>"
            + "<tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>"

        _ = processor.processChunk(content)
        processor.processEOS()

        #expect(processor.toolCalls.count == 2)
        let first = try #require(processor.toolCalls.first?.id)
        let second = try #require(processor.toolCalls.last?.id)
        #expect(first != second)
    }

    @Test("ToolCall initialized without id stays nil (source compatibility)")
    func testToolCallDefaultIdIsNil() throws {
        let tc = ToolCall(function: .init(name: "noop", arguments: [:]))
        #expect(tc.id == nil)
    }
}
