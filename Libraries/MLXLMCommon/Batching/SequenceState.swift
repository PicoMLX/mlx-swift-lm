// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Internal state for a single sequence being processed by InferenceScheduler.
///
/// This tracks all per-request state including:
/// - Input/output tokens
/// - KV cache slot allocation
/// - Prefill progress
/// - Streaming continuation
/// - Cancellation/completion status
///
/// NOTE: This type is @unchecked Sendable but must only be accessed
/// from within the InferenceScheduler actor.
final class SequenceState: @unchecked Sendable {
    
    // MARK: - Identity
    
    /// Unique identifier for this sequence
    let id: UUID
    
    // MARK: - Tokens
    
    /// Original input tokens (prompt)
    let inputTokens: [Int]
    
    /// Tokens generated so far
    var generatedTokens: [Int]
    
    // MARK: - Cache & Position
    
    /// Index into BatchKVCache for this sequence's KV data
    var kvSlot: Int
    
    /// Current decode position (RoPE offset)
    var position: Int
    
    /// How many input tokens have been prefilled
    var prefillIndex: Int
    
    /// For future prefix cache support
    var inheritedPrefixLength: Int
    
    var ropeOffset: Int {
        inheritedPrefixLength + prefillIndex + generatedTokens.count
    }
    
    // MARK: - Limits
    
    /// Maximum tokens to generate
    let maxTokens: Int
    
    /// Custom stop tokens for this request (in addition to model EOS)
    let stopTokens: Set<Int>?
    
    /// Generation parameters (temperature, top-p, etc.)
    let params: GenerateParameters
    
    /// Optional deadline for this request
    let deadline: Date?
    
    /// When this request was created
    let createdAt: Date
    
    // MARK: - Timing
    
    /// When prefill started
    var prefillStartTime: Date?
    
    /// Time spent in prefill phase
    var prefillTime: TimeInterval = 0
    
    /// When decode started (after prefill completes)
    var decodeStartTime: Date?
    
    // MARK: - Streaming
    
    /// Continuation to yield Generation events back to the client
    let continuation: AsyncStream<Generation>.Continuation
    
    // MARK: - Status
    
    /// Whether the client has cancelled this request
    var isCancelled: Bool
    
    /// Whether generation has completed (stop token, length limit, etc.)
    var isFinished: Bool
    
    // MARK: - Computed Properties
    
    /// Whether prefill is complete for this sequence
    var isPrefillComplete: Bool {
        prefillIndex >= inputTokens.count
    }
    
    /// Number of tokens remaining to prefill
    var remainingPrefillTokens: Int {
        max(0, inputTokens.count - prefillIndex)
    }
    
    /// Number of tokens generated
    var generatedCount: Int {
        generatedTokens.count
    }
    
    /// Whether this sequence has exceeded its deadline
    var isExpired: Bool {
        guard let deadline = deadline else { return false }
        return Date() >= deadline
    }
    
    // MARK: - Initialization
    
    init(
        id: UUID,
        inputTokens: [Int],
        kvSlot: Int,
        maxTokens: Int,
        stopTokens: Set<Int>?,
        params: GenerateParameters,
        deadline: Date?,
        createdAt: Date,
        continuation: AsyncStream<Generation>.Continuation
    ) {
        self.id = id
        self.inputTokens = inputTokens
        self.generatedTokens = []
        self.kvSlot = kvSlot
        self.position = 0
        self.prefillIndex = 0
        self.inheritedPrefixLength = 0
        self.maxTokens = maxTokens
        self.stopTokens = stopTokens
        self.params = params
        self.deadline = deadline
        self.createdAt = createdAt
        self.continuation = continuation
        self.isCancelled = false
        self.isFinished = false
    }
    
    // MARK: - Methods
    
    /// Record a generated token
    func addGeneratedToken(_ token: Int) {
        generatedTokens.append(token)
        position += 1
    }
    
    /// Advance prefill progress
    func advancePrefill(by count: Int) {
        prefillIndex += count
        position += count
    }
    
    /// Mark as cancelled and finish the stream
    func cancel() {
        guard !isFinished else { return }
        isCancelled = true
        isFinished = true
        let info = buildCompletionInfo(finishReason: .cancelled)
        continuation.yield(.info(info))
        continuation.finish()
    }
    
    /// Mark as timed out and finish the stream
    func timeout() {
        guard !isFinished else { return }
        isFinished = true
        let info = buildCompletionInfo(finishReason: .timeout)
        continuation.yield(.info(info))
        continuation.finish()
    }
    
    /// Mark as completed with a reason and finish the stream
    func complete(reason: FinishReason) {
        guard !isFinished else { return }
        isFinished = true
        let info = buildCompletionInfo(finishReason: reason)
        continuation.yield(.info(info))
        continuation.finish()
    }
    
    /// Emit an error and finish the stream (Generation doesn't have an error case)
    func fail(error: SchedulerError) {
        guard !isFinished else { return }
        isFinished = true
        let info = buildCompletionInfo(finishReason: .error)
        continuation.yield(.info(info))
        continuation.finish()
    }
    
    /// Emit a text chunk
    func emitChunk(_ text: String) {
        continuation.yield(.chunk(text))
    }
    
    /// Build completion info with timing stats
    private func buildCompletionInfo(finishReason: FinishReason) -> GenerateCompletionInfo {
        let generateTime: TimeInterval
        if let decodeStart = decodeStartTime {
            generateTime = Date.timeIntervalSinceReferenceDate - decodeStart.timeIntervalSinceReferenceDate
        } else {
            generateTime = 0
        }
        
        return GenerateCompletionInfo(
            promptTokenCount: inputTokens.count,
            generationTokenCount: generatedTokens.count,
            promptTime: prefillTime,
            generationTime: generateTime,
            finishReason: finishReason
        )
    }
}
