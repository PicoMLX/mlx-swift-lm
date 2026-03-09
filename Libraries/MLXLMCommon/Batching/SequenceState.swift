// Copyright © 2025 Apple Inc.

import Foundation
import Tokenizers

final class SequenceState: @unchecked Sendable {
    let uid: Int
    let continuation: AsyncStream<Generation>.Continuation
    let promptTokenCount: Int
    let prefillStartTime: Date

    private var outputProcessor: StreamingGenerationProcessor

    var generatedTokens: [Int] = []
    var decodeStartTime: Date?
    var promptTime: TimeInterval = 0
    var stopReason: GenerateStopReason?
    var isFinished = false

    init(
        uid: Int,
        promptTokenCount: Int,
        tokenizer: Tokenizer,
        modelConfiguration: ModelConfiguration,
        continuation: AsyncStream<Generation>.Continuation
    ) {
        self.uid = uid
        self.promptTokenCount = promptTokenCount
        self.prefillStartTime = .now
        self.outputProcessor = StreamingGenerationProcessor(
            tokenizer: tokenizer,
            format: modelConfiguration.toolCallFormat ?? .json
        )
        self.continuation = continuation
    }

    func recordGeneratedToken(_ token: Int) {
        if decodeStartTime == nil {
            decodeStartTime = .now
            promptTime = decodeStartTime!.timeIntervalSince(prefillStartTime)
        }
        generatedTokens.append(token)
    }

    @discardableResult
    func emitToken(_ token: Int) -> Bool {
        outputProcessor.onToken(token, emit: continuation.yield)
    }

    func finish(stopReason: GenerateStopReason) {
        guard !isFinished else { return }
        isFinished = true
        self.stopReason = stopReason

        if decodeStartTime == nil {
            promptTime = Date.now.timeIntervalSince(prefillStartTime)
        }

        let generationTime = decodeStartTime.map { Date.now.timeIntervalSince($0) } ?? 0
        continuation.yield(
            .info(
                GenerateCompletionInfo(
                    promptTokenCount: promptTokenCount,
                    generationTokenCount: generatedTokens.count,
                    promptTime: promptTime,
                    generationTime: generationTime,
                    stopReason: stopReason
                )
            )
        )
        continuation.finish()
    }
}
