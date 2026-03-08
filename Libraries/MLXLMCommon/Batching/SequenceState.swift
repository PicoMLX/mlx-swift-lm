// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Tokenizers

/// Per-request state for a sequence being decoded in batch mode.
///
/// Tracks the state of a single sequence within a batched inference job,
/// including the generated tokens, the async stream continuation to deliver
/// results to the caller, and per-sequence generation parameters.
final class SequenceState: @unchecked Sendable {
    /// Unique identifier within the scheduler
    let uid: Int

    /// Continuation for delivering ``Generation`` events to the caller
    let continuation: AsyncStream<Generation>.Continuation

    /// All tokens generated so far (not including the prompt)
    var generatedTokens: [Int] = []

    /// Current token being held (the most recently sampled, about to be decoded/emitted)
    var currentToken: Int

    /// Number of tokens generated (same as generatedTokens.count for tracking)
    var numTokens: Int = 0

    /// Maximum number of tokens to generate (nil = unlimited)
    let maxTokens: Int?

    /// Sampler for this sequence
    let sampler: any LogitSampler

    /// Optional logit processor for this sequence
    var processor: (any LogitProcessor)?

    /// Tokenizer for decoding tokens to text
    let tokenizer: Tokenizer

    /// Optional generation deadline
    let deadline: Date?

    /// EOS token ID to detect stop condition
    let eosTokenId: Int?

    /// The prompt tokens (for processor initialization)
    let promptTokens: [Int]

    /// Time when prefill started (for timing stats)
    var prefillStartTime: Date = .now

    /// Time when first decode token was generated
    var decodeStartTime: Date?

    /// Number of prompt tokens
    var promptTokenCount: Int

    init(
        uid: Int,
        currentToken: Int,
        promptTokens: [Int],
        parameters: GenerateParameters,
        tokenizer: Tokenizer,
        continuation: AsyncStream<Generation>.Continuation,
        deadline: Date? = nil
    ) {
        self.uid = uid
        self.currentToken = currentToken
        self.promptTokens = promptTokens
        self.promptTokenCount = promptTokens.count
        self.maxTokens = parameters.maxTokens
        self.sampler = parameters.sampler()
        var proc = parameters.processor()
        proc?.prompt(MLXArray(promptTokens.map { Int32($0) }))
        self.processor = proc
        self.tokenizer = tokenizer
        self.continuation = continuation
        self.deadline = deadline
        self.eosTokenId = tokenizer.eosTokenId
    }

    /// Whether this sequence has reached its stop condition.
    var isDone: Bool {
        if let maxTokens, numTokens >= maxTokens {
            return true
        }
        if let eosTokenId, currentToken == eosTokenId {
            return true
        }
        if let deadline, Date.now > deadline {
            return true
        }
        return false
    }

    /// The stop reason for a completed sequence.
    var stopReason: GenerateStopReason {
        if let eosTokenId, currentToken == eosTokenId {
            return .stop
        }
        if let maxTokens, numTokens >= maxTokens {
            return .length
        }
        return .cancelled
    }

    /// Record the start of decode (first decode token time).
    func markDecodeStart() {
        if decodeStartTime == nil {
            decodeStartTime = .now
        }
    }
}
