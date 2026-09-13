// Copyright © 2026 Apple Inc.

import Dispatch
import Foundation
import Testing
import os

@testable import MLXLMCommon

// Exercise the public raw-token task so the actual worker, stop handling,
// iterator finalization and completion telemetry remain covered together.
private struct MockMTPIterator: GenerationFinalizingTokenIterator, MTPStatsCollecting {
    let tokensToYield: [Int]
    let perTokenProposed: Int
    let perTokenAccepted: Int
    let finalPassthroughReason: String?
    let onNext: @Sendable () -> Void
    let onFinalize: @Sendable () -> Void

    var index = 0
    public var tokenCount = 0
    public let maxTokens: Int? = nil
    public let promptPrefillTime: TimeInterval = 0
    public private(set) var proposedDraftTokens: Int = 0
    public private(set) var acceptedDraftTokens: Int = 0
    public private(set) var passthroughReason: String?

    init(
        tokensToYield: [Int],
        perTokenProposed: Int,
        perTokenAccepted: Int,
        finalPassthroughReason: String?,
        onNext: @escaping @Sendable () -> Void = {},
        onFinalize: @escaping @Sendable () -> Void = {}
    ) {
        self.tokensToYield = tokensToYield
        self.perTokenProposed = perTokenProposed
        self.perTokenAccepted = perTokenAccepted
        self.finalPassthroughReason = finalPassthroughReason
        self.onNext = onNext
        self.onFinalize = onFinalize
    }

    mutating func next() -> Int? {
        guard index < tokensToYield.count else {
            passthroughReason = finalPassthroughReason
            return nil
        }
        onNext()
        let token = tokensToYield[index]
        index += 1
        tokenCount += 1
        proposedDraftTokens += perTokenProposed
        acceptedDraftTokens += perTokenAccepted
        return token
    }

    mutating func finalizeGeneration() { onFinalize() }
}

private func consumeIteratorAndBuildInfo<I: TokenIteratorProtocol>(
    _ iterator: consuming I,
    configuration: ModelConfiguration = .init(id: "test"),
    includeStopToken: Bool = false
) async throws -> (tokens: [Int], info: GenerateCompletionInfo) {
    let (stream, task) = generateTokenTask(
        promptTokenCount: 3, modelConfiguration: configuration,
        tokenizer: RawTaskTokenizer(), iterator: iterator, includeStopToken: includeStopToken)
    var tokens: [Int] = []
    var completion: GenerateCompletionInfo?
    for await event in stream {
        switch event {
        case .token(let token): tokens.append(token)
        case .info(let info): completion = info
        }
    }
    await task.value
    return (tokens, try #require(completion))
}

@Suite
struct MTPGenerateLoopTaskContractTests {

    // A value-type iterator's counters must reflect the worker's mutations.
    @Test
    func iteratorCountersOnOuterBindingReflectLoopMutations() async throws {
        let mock = MockMTPIterator(
            tokensToYield: [11, 22, 33, 44],
            perTokenProposed: 3,
            perTokenAccepted: 2,
            finalPassthroughReason: nil
        )

        let (tokens, info) = try await consumeIteratorAndBuildInfo(mock)

        #expect(tokens == [11, 22, 33, 44])
        #expect(info.generationTokenCount == 4)
        #expect(
            info.proposedDraftTokens == 12,
            "outer-binding proposedDraftTokens stuck at \(info.proposedDraftTokens ?? -1); regression of the for-in copy-semantics bug fixed in Phase 4"
        )
        #expect(
            info.acceptedDraftTokens == 8,
            "outer-binding acceptedDraftTokens stuck at \(info.acceptedDraftTokens ?? -1); regression of the for-in copy-semantics bug fixed in Phase 4"
        )
        #expect(info.passthroughReason == nil)
    }

    // The worker must observe state set by the final next() call.
    @Test
    func passthroughReasonObservedOnOuterBinding() async throws {
        let mock = MockMTPIterator(
            tokensToYield: [7, 8],
            perTokenProposed: 0,
            perTokenAccepted: 0,
            finalPassthroughReason: "main model did not emit drafter state"
        )

        let (tokens, info) = try await consumeIteratorAndBuildInfo(mock)

        #expect(tokens == [7, 8])
        #expect(info.proposedDraftTokens == 0)
        #expect(info.acceptedDraftTokens == 0)
        #expect(
            info.passthroughReason == "main model did not emit drafter state",
            "outer-binding passthroughReason was \(info.passthroughReason ?? "nil"); regression of the for-in copy-semantics bug fixed in Phase 4"
        )
    }

    @Test(
        "Raw stop inclusion preserves EOS, extra EOS and unknown token IDs",
        arguments: [101, 102, 103, 104], [false, true])
    func stopTokens(stop: Int, include: Bool) async throws {
        let finalized = OSAllocatedUnfairLock(initialState: false)
        let iterator = MockMTPIterator(
            tokensToYield: [11, 22, stop, 44], perTokenProposed: 3,
            perTokenAccepted: 2, finalPassthroughReason: nil,
            onFinalize: { finalized.withLock { $0 = true } })
        let (tokens, info) = try await consumeIteratorAndBuildInfo(
            iterator, configuration: .init(id: "test", extraEOSTokens: ["103"], eosTokenIds: [104]),
            includeStopToken: include)
        #expect(tokens == (include ? [11, 22, stop] : [11, 22]))
        #expect(info.generationTokenCount == tokens.count)
        #expect(info.promptTokenCount == 3)
        #expect(info.stopReason == .stop)
        #expect(info.proposedDraftTokens == 9)
        #expect(info.acceptedDraftTokens == 6)
        #expect(finalized.withLock { $0 })
    }

    @Test(
        "Raw producer completion waits for finalization after cancellation",
        arguments: [false, true])
    func cancellation(throughConsumer: Bool) async {
        let entered = AsyncStream<Void>.makeStream()
        let resume = DispatchSemaphore(value: 0)
        let finalized = OSAllocatedUnfairLock(initialState: false)
        let calls = OSAllocatedUnfairLock(initialState: 0)
        let iterator = MockMTPIterator(
            tokensToYield: [11, 22, 33], perTokenProposed: 3,
            perTokenAccepted: 2, finalPassthroughReason: nil,
            onNext: {
                calls.withLock { $0 += 1 }
                entered.continuation.yield(())
                #expect(resume.wait(timeout: .now() + 5) == .success)
            },
            onFinalize: { finalized.withLock { $0 = true } })
        let (stream, producer) = generateTokenTask(
            promptTokenCount: 3, modelConfiguration: .init(id: "test"),
            tokenizer: RawTaskTokenizer(), iterator: iterator)
        let consumer = Task { for await _ in stream {} }
        for await _ in entered.stream { break }
        if throughConsumer { consumer.cancel() } else { producer.cancel() }
        resume.signal()
        await consumer.value
        await producer.value
        entered.continuation.finish()
        #expect(calls.withLock { $0 } == 1)
        #expect(finalized.withLock { $0 })
    }

}

private struct RawTaskTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { "101" }
    var unknownToken: String? { "102" }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.split(separator: " ").compactMap { Int($0) }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }
    func convertTokenToId(_ token: String) -> Int? { Int(token) }
    func convertIdToToken(_ id: Int) -> String? { String(id) }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { throw TokenizerError.missingChatTemplate }
}
