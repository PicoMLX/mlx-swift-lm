// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Completion information for one row of a fixed batch.
public struct BatchCompletionInfo: Sendable {
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let stopReason: GenerateStopReason
    /// Time until this row's first model token, including the batch's sequential prefills.
    /// Nil when the row was cancelled before its first token.
    public let timeToFirstToken: TimeInterval?
    /// Time from starting the batch worker until this row finished.
    public let elapsedTime: TimeInterval
}

/// An output or final completion event, attributed to its zero-based prompt index.
///
/// Completion is reported by the outer `info` case, not an embedded `Generation.info` or
/// `TokenGeneration.info`. Batch timings do not represent isolated per-row prefill costs.
public enum BatchGenerationEvent<Output: Sendable>: Sendable {
    case output(uid: Int, Output)
    case info(uid: Int, BatchCompletionInfo)

    public var uid: Int {
        switch self {
        case .output(let uid, _), .info(let uid, _): uid
        }
    }
}

public enum BatchGenerationError: Error, LocalizedError, Equatable {
    case emptyPrompt(index: Int)
    case cacheCountMismatch
    case invalidTokenLimit

    public var errorDescription: String? {
        switch self {
        case .emptyPrompt(let index): "Prompt \(index) must contain at least one token."
        case .cacheCountMismatch: "Supply one cache entry per prompt."
        case .invalidTokenLimit: "The token limit must be positive."
        }
    }
}

/// Streams raw token events from a fixed batch. For cancellation and deterministic MLX
/// cleanup, use ``batchGenerateTokensTask(model:prompts:tokenizer:modelConfiguration:caches:maxTokens:stopTokens:sampler:processor:prefill:includeStopToken:wiredMemoryTicket:cacheEvaluationPolicy:)``.
///
/// Events are buffered without dropping tokens. Consume the stream to completion or cancel
/// its consumer; a plain `break` does not reliably terminate an `AsyncThrowingStream`.
public func batchGenerateTokens(
    model: any LanguageModel, prompts: [[Int]], tokenizer: Tokenizer,
    modelConfiguration: ModelConfiguration, caches: [[KVCache]?]? = nil,
    maxTokens: Int = 128, stopTokens: Set<Int> = [], sampler: LogitSampler = ArgMaxSampler(),
    processor: LogitProcessor? = nil, prefill: PrefillParameters = .init(),
    includeStopToken: Bool = false, wiredMemoryTicket: WiredMemoryTicket? = nil,
    cacheEvaluationPolicy: KVCacheEvaluationPolicy = .allState
) throws -> AsyncThrowingStream<BatchGenerationEvent<TokenGeneration>, Error> {
    try batchGenerateTokensTask(
        model: model, prompts: prompts, tokenizer: tokenizer,
        modelConfiguration: modelConfiguration, caches: caches, maxTokens: maxTokens,
        stopTokens: stopTokens, sampler: sampler, processor: processor, prefill: prefill,
        includeStopToken: includeStopToken, wiredMemoryTicket: wiredMemoryTicket,
        cacheEvaluationPolicy: cacheEvaluationPolicy
    ).0
}

/// Streams raw tokens and per-row completion information using one batch iterator.
///
/// EOS and unknown tokens end their own row. `includeStopToken` controls whether the terminal
/// token is emitted and counted. A cancelled task completes unfinished rows as `.cancelled`;
/// model/cache errors terminate the stream with that error. Await `task.value` after cancelling
/// to wait for GPU work and cache cleanup. The model and supplied caches must not be used
/// concurrently until the task completes. Admission, prefix reuse and budgets remain caller-owned.
/// ``KVCacheEvaluationPolicy/attentionDependencies`` requires the caller's model to consume
/// its attention-cache updates in the token computation.
public func batchGenerateTokensTask(
    model: any LanguageModel, prompts: [[Int]], tokenizer: Tokenizer,
    modelConfiguration: ModelConfiguration, caches: [[KVCache]?]? = nil,
    maxTokens: Int = 128, stopTokens: Set<Int> = [], sampler: LogitSampler = ArgMaxSampler(),
    processor: LogitProcessor? = nil, prefill: PrefillParameters = .init(),
    includeStopToken: Bool = false, wiredMemoryTicket: WiredMemoryTicket? = nil,
    cacheEvaluationPolicy: KVCacheEvaluationPolicy = .allState
) throws -> (
    AsyncThrowingStream<BatchGenerationEvent<TokenGeneration>, Error>, Task<Void, Never>
) {
    try batchGenerationTask(
        model: model, prompts: prompts, tokenizer: tokenizer,
        modelConfiguration: modelConfiguration, caches: caches, maxTokens: maxTokens,
        stopTokens: stopTokens, sampler: sampler, processor: processor, prefill: prefill,
        includeStopToken: includeStopToken, wiredMemoryTicket: wiredMemoryTicket,
        cacheEvaluationPolicy: cacheEvaluationPolicy,
        handlers: prompts.map { _ in RawTokenLoopHandler() })
}

/// Streams decoded text, reasoning and tool-call events from a fixed batch.
///
/// Each row has independent decoder state. Output is lossless and buffered; use the task
/// variant when a consumer may stop before the whole batch finishes.
public func batchGenerateText(
    model: any LanguageModel, prompts: [[Int]], tokenizer: Tokenizer,
    modelConfiguration: ModelConfiguration, caches: [[KVCache]?]? = nil,
    maxTokens: Int = 128, stopTokens: Set<Int> = [], sampler: LogitSampler = ArgMaxSampler(),
    processor: LogitProcessor? = nil, prefill: PrefillParameters = .init(),
    tools: [[String: any Sendable]]? = nil, toolCallPolicy: ToolCallPolicy = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    cacheEvaluationPolicy: KVCacheEvaluationPolicy = .allState
) throws -> AsyncThrowingStream<BatchGenerationEvent<Generation>, Error> {
    try batchGenerateTextTask(
        model: model, prompts: prompts, tokenizer: tokenizer,
        modelConfiguration: modelConfiguration, caches: caches, maxTokens: maxTokens,
        stopTokens: stopTokens, sampler: sampler, processor: processor, prefill: prefill,
        tools: tools, toolCallPolicy: toolCallPolicy, wiredMemoryTicket: wiredMemoryTicket,
        cacheEvaluationPolicy: cacheEvaluationPolicy
    ).0
}

/// Runs one batch with an independent text/tool/reasoning handler for every prompt.
///
/// A stop string or protocol boundary removes only the matching row. Cancelling the task
/// cancels the whole fixed batch. Await the task to observe completion of MLX cleanup.
/// Shared output is buffered without dropping events; this API supplies no serving scheduler.
public func batchGenerateTextTask(
    model: any LanguageModel, prompts: [[Int]], tokenizer: Tokenizer,
    modelConfiguration: ModelConfiguration, caches: [[KVCache]?]? = nil,
    maxTokens: Int = 128, stopTokens: Set<Int> = [], sampler: LogitSampler = ArgMaxSampler(),
    processor: LogitProcessor? = nil, prefill: PrefillParameters = .init(),
    tools: [[String: any Sendable]]? = nil, toolCallPolicy: ToolCallPolicy = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    cacheEvaluationPolicy: KVCacheEvaluationPolicy = .allState
) throws -> (AsyncThrowingStream<BatchGenerationEvent<Generation>, Error>, Task<Void, Never>) {
    let handlers = prompts.map { prompt in
        TextToolTokenLoopHandler(
            tokenizer: tokenizer, stopStrings: modelConfiguration.effectiveStopStrings,
            format: modelConfiguration.toolCallFormat ?? .json, tools: tools,
            toolCallPolicy: toolCallPolicy,
            reasoning: modelConfiguration.reasoningConfig.map {
                (
                    config: $0,
                    primedInside: ReasoningEventEmitter.promptEndsInsideReasoning(
                        promptTokens: prompt, config: $0, tokenizer: tokenizer)
                )
            })
    }
    return try batchGenerationTask(
        model: model, prompts: prompts, tokenizer: tokenizer,
        modelConfiguration: modelConfiguration, caches: caches, maxTokens: maxTokens,
        stopTokens: stopTokens, sampler: sampler, processor: processor, prefill: prefill,
        includeStopToken: false, wiredMemoryTicket: wiredMemoryTicket,
        cacheEvaluationPolicy: cacheEvaluationPolicy, handlers: handlers)
}

private struct BatchStreamRow {
    var tokenCount = 0
    var firstTokenTime: TimeInterval?
    var finished = false
}

private func batchGenerationTask<Handler: TokenLoopHandler>(
    model: any LanguageModel, prompts: [[Int]], tokenizer: Tokenizer,
    modelConfiguration: ModelConfiguration, caches: [[KVCache]?]?, maxTokens: Int,
    stopTokens: Set<Int>, sampler: LogitSampler, processor: LogitProcessor?,
    prefill: PrefillParameters, includeStopToken: Bool, wiredMemoryTicket: WiredMemoryTicket?,
    cacheEvaluationPolicy: KVCacheEvaluationPolicy,
    handlers: consuming [Handler]
) throws -> (AsyncThrowingStream<BatchGenerationEvent<Handler.Output>, Error>, Task<Void, Never>) {
    if let index = prompts.firstIndex(where: { $0.isEmpty }) {
        throw BatchGenerationError.emptyPrompt(index: index)
    }
    guard caches.map({ $0.count == prompts.count }) ?? true else {
        throw BatchGenerationError.cacheCountMismatch
    }
    guard maxTokens > 0 else { throw BatchGenerationError.invalidTokenLimit }

    var terminalTokens = buildStopTokenIds(
        modelConfiguration: modelConfiguration, tokenizer: tokenizer)
    terminalTokens.formUnion(stopTokens)
    let handlerStopTokens = terminalTokens
    if let unknown = tokenizer.unknownTokenId { terminalTokens.insert(unknown) }
    let iteratorStopTokens = terminalTokens
    let payload = SendableBox((model, caches, sampler, processor, handlers))
    let (stream, continuation) = AsyncThrowingStream<BatchGenerationEvent<Handler.Output>, Error>
        .makeStream()
    let task = Task {
        let worker = GenerationWorker()
        let performIteration = { @Sendable in
            let clock = ContinuousClock()
            let start = clock.now
            let (model, caches, sampler, processor, initialHandlers) = payload.consume()
            var handlers = initialHandlers
            var iterator = BatchTokenIterator(
                model: model, sampler: sampler, processor: processor,
                stopTokens: iteratorStopTokens,
                maxTokens: maxTokens, prefill: prefill, cacheEvaluationPolicy: cacheEvaluationPolicy
            )
            if !Task.isCancelled { _ = iterator.insert(prompts: prompts, caches: caches) }
            var rows = prompts.map { _ in BatchStreamRow() }
            func elapsed() -> TimeInterval {
                let duration = start.duration(to: clock.now).components
                return Double(duration.seconds) + Double(duration.attoseconds) / 1e18
            }
            func emit(_ uid: Int, _ output: sending Handler.Output) -> Bool {
                if case .terminated = continuation.yield(.output(uid: uid, output)) { return false }
                return true
            }
            func finish(_ uid: Int, reason: GenerateStopReason) {
                guard !rows[uid].finished else { return }
                rows[uid].finished = true
                var reason = reason
                switch handlers[uid].onGenerationEnd(emit: { emit(uid, $0) }) {
                case .more: break
                case .stop: if reason != .cancelled { reason = .stop }
                case .cancelled: reason = .cancelled
                }
                _ = continuation.yield(
                    .info(
                        uid: uid,
                        BatchCompletionInfo(
                            promptTokenCount: prompts[uid].count,
                            generationTokenCount: rows[uid].tokenCount, stopReason: reason,
                            timeToFirstToken: rows[uid].firstTokenTime, elapsedTime: elapsed())))
            }

            var failure: Error?
            var terminated = false
            do {
                // Cancellation is checked before next(), which pipelines more MLX work.
                while !Task.isCancelled, !terminated {
                    let responses = try iterator.next()
                    if responses.isEmpty { break }
                    let responseTime = elapsed()
                    var semanticStops: [Int] = []
                    for response in responses {
                        let uid = response.uid
                        if rows[uid].firstTokenTime == nil {
                            rows[uid].firstTokenTime = responseTime
                        }
                        var reason: GenerateStopReason?
                        let semanticStop = handlers[uid].additionalStopTokenIDs.contains(
                            response.token)
                        let isStop = response.finishReason == .stop || semanticStop
                        let disposition: TokenLoopDisposition
                        if isStop {
                            if includeStopToken { rows[uid].tokenCount += 1 }
                            if includeStopToken
                                || (handlers[uid].receivesStopTokens
                                    && (semanticStop || handlerStopTokens.contains(response.token)))
                            {
                                disposition = handlers[uid].onStopToken(
                                    response.token, logProbabilities: nil, emit: { emit(uid, $0) })
                            } else {
                                disposition = .more
                            }
                            reason = .stop
                        } else {
                            rows[uid].tokenCount += 1
                            disposition = handlers[uid].onToken(
                                response.token, logProbabilities: nil, emit: { emit(uid, $0) })
                            if response.finishReason == .length { reason = .length }
                        }
                        switch disposition {
                        case .more: break
                        case .stop: reason = .stop
                        case .cancelled:
                            reason = .cancelled
                            terminated = true
                        }
                        if let reason {
                            if response.finishReason == nil { semanticStops.append(uid) }
                            finish(uid, reason: reason)
                        }
                        if terminated { break }
                    }
                    // Natural stops are already filtered by next(); semantic stops share one gather.
                    if !semanticStops.isEmpty { iterator.remove(semanticStops) }
                }
                for uid in rows.indices where !rows[uid].finished {
                    finish(uid, reason: .cancelled)
                }
            } catch is CancellationError where Task.isCancelled {
                for uid in rows.indices where !rows[uid].finished {
                    finish(uid, reason: .cancelled)
                }
            } catch {
                failure = error
                for uid in rows.indices where !rows[uid].finished {
                    _ = handlers[uid].onGenerationEnd(emit: { emit(uid, $0) })
                }
            }
            iterator.remove(iterator.uids)
            // Cache owners and consumers may proceed only after queued device work settles.
            Stream.defaultStream.synchronize()
            continuation.finish(throwing: failure)
        }
        if let ticket = wiredMemoryTicket {
            await WiredMemoryTicket.withWiredLimit(ticket) { await worker.run(performIteration) }
        } else {
            await worker.run(performIteration)
        }
    }
    continuation.onTermination = { termination in
        if case .cancelled = termination { task.cancel() }
    }
    return (stream, task)
}
