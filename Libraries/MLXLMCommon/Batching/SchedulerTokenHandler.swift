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

    /// What the caller must do after handing the handler one token.
    enum TokenDisposition: Sendable, Equatable {
        /// Keep generating.
        case more
        /// A semantic stop completed in the decoded text (a configured stop
        /// string fully matched). The text before the stop was already
        /// emitted and the stop text suppressed; the caller must finish the
        /// request with `.stop`, exactly like a stop token.
        case stop
        /// The consumer dropped the stream.
        case cancelled
    }

    /// Which output mode this handler serves.
    let mode: OutputMode

    /// Process a generated token.
    let processToken: @Sendable (Int) -> TokenDisposition

    /// Process a stop token. Only meaningful for `.rawTokens(includeStopToken: true)`.
    /// Never returns `.stop` (the caller is already finishing the request).
    let processStopToken: @Sendable (Int) -> TokenDisposition

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
        var stopStringFilter: StopStringFilter
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
            stopStrings: Set<String>,
            toolCallFormat: ToolCallFormat,
            continuation: AsyncStream<Generation>.Continuation
        ) {
            self.continuation = continuation
            self.mutableState = OSAllocatedUnfairLock(
                uncheckedState: TextMutableState(
                    detokenizer: NaiveStreamingDetokenizer(tokenizer: tokenizer),
                    stopStringFilter: StopStringFilter(stopStrings: stopStrings),
                    toolCallProcessor: ToolCallProcessor(format: toolCallFormat)
                ))
        }

        func withMutableState<R>(_ body: (inout TextMutableState) -> R) -> R {
            mutableState.withLockUnchecked { body(&$0) }
        }
    }

    /// Create a handler that detokenizes tokens and yields `.chunk` / `.toolCall` events.
    ///
    /// `stopStrings` mirrors `generateTask`'s `TextToolTokenLoopHandler`: the
    /// decoded text is run through a ``StopStringFilter`` BEFORE tool-call
    /// parsing (text ahead of a stop is emitted, the stop text and everything
    /// after are suppressed), and a completed stop string returns `.stop` so
    /// the caller finishes the request like a stop token.
    static func text(
        continuation: AsyncStream<Generation>.Continuation,
        tokenizer: Tokenizer,
        stopStrings: Set<String> = [],
        toolCallFormat: ToolCallFormat
    ) -> SchedulerTokenHandler {
        let box = TextState(
            tokenizer: tokenizer,
            stopStrings: stopStrings,
            toolCallFormat: toolCallFormat,
            continuation: continuation
        )

        // Shared emit path: route filtered text through the tool-call
        // processor and drain any completed tool calls in FIFO order (popLast
        // would emit only the last one, in LIFO order, and delay the rest to
        // end-of-sequence). Returns false when the consumer cancelled.
        @Sendable func emitText(_ text: String, _ state: inout TextMutableState) -> Bool {
            if let textToYield = state.toolCallProcessor.processChunk(text) {
                if case .terminated = box.continuation.yield(.chunk(textToYield)) {
                    return false
                }
            }
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

        return SchedulerTokenHandler(
            mode: .decoded,
            processToken: { token in
                box.withMutableState { state -> TokenDisposition in
                    state.detokenizer.append(token: token)
                    guard let chunk = state.detokenizer.next() else { return .more }
                    let result = state.stopStringFilter.process(chunk)
                    if let text = result.text, !emitText(text, &state) {
                        return .cancelled
                    }
                    return result.stopped ? .stop : .more
                }
            },
            processStopToken: { _ in
                // Decoded mode never emits stop tokens.
                .more
            },
            processEndOfSequence: {
                box.withMutableState { state in
                    // Release text the stop-string filter held back as a
                    // potential stop prefix that never completed — parity
                    // with `TextToolTokenLoopHandler.onGenerationEnd`.
                    if let held = state.stopStringFilter.finish(), !emitText(held, &state) {
                        return
                    }
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
            // No stop-string filtering in raw mode: stop strings are a
            // text-level concept and the historical raw-token path does not
            // detokenize, matching `generateTokenTask`.
            processToken: { token in
                if case .terminated = continuation.yield(.token(token)) {
                    return .cancelled
                }
                return .more
            },
            processStopToken: { token in
                guard includeStopToken else { return .more }
                if case .terminated = continuation.yield(.token(token)) {
                    return .cancelled
                }
                return .more
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
