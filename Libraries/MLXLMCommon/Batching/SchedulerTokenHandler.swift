// Copyright © 2026 Apple Inc.

import Foundation
import os

// MARK: - SchedulerTokenHandler

/// Type-erased per-request output adapter for the scheduler.
///
/// The driver/scheduler calls `handler.processToken(token)` without knowing
/// whether the consumer wants decoded text (`AsyncStream<Generation>`) or raw
/// token IDs (`AsyncStream<TokenGeneration>`). Two factory methods build a
/// handler for each mode.
///
/// The struct holds only `@Sendable` closures and a `Sendable` mode enum, so it
/// is plainly `Sendable` — **no `@unchecked`**. Per-request mutable streaming
/// state (the detokenizer and tool-call processor) lives behind an
/// `OSAllocatedUnfairLock` inside the text-mode factory's state box, which keeps
/// that box `Sendable` as well. In practice a single executor drives one
/// request's token loop, so the lock is uncontended (it exists only to satisfy
/// the `Sendable` checker without an `@unchecked` conformance).
struct SchedulerTokenHandler: Sendable {

    /// The output mode this handler was created for.
    enum OutputMode: Sendable {
        case decoded
        case rawTokens(includeStopToken: Bool)
    }

    /// Which output mode this handler serves.
    let mode: OutputMode

    /// Process a generated token. Returns `false` if the consumer cancelled.
    let processToken: @Sendable (Int) -> Bool

    /// Process a stop token. Only meaningful for `.rawTokens(includeStopToken: true)`.
    /// Returns `false` if the consumer cancelled.
    let processStopToken: @Sendable (Int) -> Bool

    /// Flush buffered state at end-of-sequence (e.g. pending tool calls for text mode).
    let processEndOfSequence: @Sendable () -> Void

    /// Yield completion info.
    let yieldInfo: @Sendable (GenerateCompletionInfo) -> Void

    /// Close the stream.
    let finish: @Sendable () -> Void

    /// Register a cancellation callback on the stream's continuation.
    let onCancellation: @Sendable (@Sendable @escaping () -> Void) -> Void
}

// MARK: - Factory: Text Mode

extension SchedulerTokenHandler {

    /// Mutable streaming state for the text-mode handler, held behind an
    /// `OSAllocatedUnfairLock` so this box is `Sendable` with no `@unchecked`.
    private struct TextMutableState {
        var detokenizer: NaiveStreamingDetokenizer
        let toolCallProcessor: ToolCallProcessor
    }

    /// `Sendable` state box for the text-mode handler.
    ///
    /// The continuation is `Sendable` and stored directly; the detokenizer and
    /// tool-call processor are non-`Sendable` and guarded by the lock. Access is
    /// single-threaded by design (one executor drives one request's decode
    /// loop), so the lock is uncontended.
    private final class TextState: Sendable {
        let continuation: AsyncStream<Generation>.Continuation
        private let mutableState: OSAllocatedUnfairLock<TextMutableState>

        init(
            tokenizer: Tokenizer,
            toolCallFormat: ToolCallFormat,
            continuation: AsyncStream<Generation>.Continuation
        ) {
            self.continuation = continuation
            self.mutableState = OSAllocatedUnfairLock(
                uncheckedState: TextMutableState(
                    detokenizer: NaiveStreamingDetokenizer(tokenizer: tokenizer),
                    toolCallProcessor: ToolCallProcessor(format: toolCallFormat)
                ))
        }

        func withMutableState<R>(_ body: (inout TextMutableState) -> R) -> R {
            mutableState.withLockUnchecked { body(&$0) }
        }
    }

    /// Create a handler that detokenizes tokens and yields `.chunk` / `.toolCall` events.
    static func text(
        continuation: AsyncStream<Generation>.Continuation,
        tokenizer: Tokenizer,
        toolCallFormat: ToolCallFormat
    ) -> SchedulerTokenHandler {
        let box = TextState(
            tokenizer: tokenizer,
            toolCallFormat: toolCallFormat,
            continuation: continuation
        )

        return SchedulerTokenHandler(
            mode: .decoded,
            processToken: { token in
                box.withMutableState { state -> Bool in
                    state.detokenizer.append(token: token)
                    guard let chunk = state.detokenizer.next() else { return true }
                    if let textToYield = state.toolCallProcessor.processChunk(chunk) {
                        if case .terminated = box.continuation.yield(.chunk(textToYield)) {
                            return false
                        }
                    }
                    // Drain ALL accumulated tool calls in FIFO order (popLast
                    // would emit only the last one, in LIFO order, and delay the
                    // rest to end-of-sequence).
                    if !state.toolCallProcessor.toolCalls.isEmpty {
                        let pending = state.toolCallProcessor.toolCalls
                        state.toolCallProcessor.toolCalls.removeAll()
                        for toolCall in pending {
                            if case .terminated = box.continuation.yield(.toolCall(toolCall)) {
                                return false
                            }
                        }
                    }
                    return true
                }
            },
            processStopToken: { _ in
                // Decoded mode never emits stop tokens.
                true
            },
            processEndOfSequence: {
                box.withMutableState { state in
                    // Flush any text the ToolCallProcessor buffered while
                    // deciding whether it was a tool call. Without
                    // `returnBufferedText: true` a trailing non-tool fragment
                    // (e.g. an unmatched `{...` at the end of ordinary text) is
                    // silently dropped — parity with `generateTask`'s
                    // `TextToolTokenLoopHandler.onGenerationEnd`.
                    if let buffered = state.toolCallProcessor.processEOS(
                        returnBufferedText: true), !buffered.isEmpty
                    {
                        if case .terminated = box.continuation.yield(.chunk(buffered)) {
                            return
                        }
                    }
                    for toolCall in state.toolCallProcessor.toolCalls {
                        if case .terminated = box.continuation.yield(.toolCall(toolCall)) {
                            break
                        }
                    }
                }
            },
            yieldInfo: { info in
                _ = box.continuation.yield(.info(info))
            },
            finish: {
                box.continuation.finish()
            },
            onCancellation: { callback in
                box.continuation.onTermination = { termination in
                    if case .cancelled = termination {
                        callback()
                    }
                }
            }
        )
    }
}

// MARK: - Factory: Raw Token Mode

extension SchedulerTokenHandler {

    /// Create a handler that yields raw `.token(Int)` events.
    static func rawToken(
        continuation: AsyncStream<TokenGeneration>.Continuation,
        includeStopToken: Bool
    ) -> SchedulerTokenHandler {
        SchedulerTokenHandler(
            mode: .rawTokens(includeStopToken: includeStopToken),
            processToken: { token in
                if case .terminated = continuation.yield(.token(token)) {
                    return false
                }
                return true
            },
            processStopToken: { token in
                guard includeStopToken else { return true }
                if case .terminated = continuation.yield(.token(token)) {
                    return false
                }
                return true
            },
            processEndOfSequence: {
                // No-op for raw token mode.
            },
            yieldInfo: { info in
                _ = continuation.yield(.info(info))
            },
            finish: {
                continuation.finish()
            },
            onCancellation: { callback in
                continuation.onTermination = { termination in
                    if case .cancelled = termination {
                        callback()
                    }
                }
            }
        )
    }
}
