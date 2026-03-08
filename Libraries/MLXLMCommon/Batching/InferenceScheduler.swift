// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN
import Tokenizers

/// An inference request submitted to the ``InferenceScheduler``.
///
/// `LMInput` contains `MLXArray` which is not `Sendable` in the strict sense,
/// but we need to transfer ownership from the caller to the scheduler actor.
/// The `@unchecked` suppression is safe because the caller transfers ownership
/// at the `submit` call site and does not retain a reference afterwards.
public struct InferenceRequest: @unchecked Sendable {
    public let input: LMInput
    public let parameters: GenerateParameters

    public init(input: LMInput, parameters: GenerateParameters) {
        self.input = input
        self.parameters = parameters
    }
}

/// Actor that manages a queue of inference requests and dispatches them through
/// batched prefill + decode.
///
/// ## Overview
///
/// `InferenceScheduler` is the top-level entry point for concurrent LLM inference
/// on a single model. Callers submit ``InferenceRequest`` values and receive
/// `AsyncStream<Generation>` back. The scheduler:
///
/// 1. Runs requests immediately if capacity allows (no artificial wait time).
/// 2. Batches new requests with any currently decoding sequences when they arrive
///    mid-generation (the _single-to-batch upgrade_).
/// 3. Removes completed sequences from the batch and continues.
///
/// ## Configuration
///
/// ```swift
/// let scheduler = InferenceScheduler(
///     context: modelContext,
///     config: .init(completionBatchSize: 4)
/// )
/// let stream = await scheduler.submit(InferenceRequest(input: input, parameters: params))
/// ```
///
/// ## Thread Safety
///
/// `InferenceScheduler` is an `actor`. Submit calls from any context; the scheduler
/// serializes access internally.
public actor InferenceScheduler {

    // MARK: - Config

    /// Configuration for the inference scheduler.
    public struct Config: Sendable {
        /// Maximum number of sequences decoded simultaneously.
        ///
        /// Requests beyond this limit queue until space is available.
        /// Default is 4, suitable for local/LAN use.
        public var completionBatchSize: Int = 4

        /// Maximum number of sequences that can be prefilled in one go.
        ///
        /// Setting this lower than `completionBatchSize` staggers prefills to
        /// reduce peak memory during the fill phase.
        public var prefillBatchSize: Int = 4

        /// Maximum number of requests waiting in the queue.
        ///
        /// Submissions beyond this limit throw ``SchedulerError/queueFull``.
        public var maxQueueSize: Int = 32

        public init(
            completionBatchSize: Int = 4,
            prefillBatchSize: Int = 4,
            maxQueueSize: Int = 32
        ) {
            self.completionBatchSize = completionBatchSize
            self.prefillBatchSize = prefillBatchSize
            self.maxQueueSize = maxQueueSize
        }
    }

    // MARK: - Errors

    public enum SchedulerError: Error, Sendable {
        case queueFull
        case modelNotReady
    }

    // MARK: - State

    private let context: ModelContext
    public let config: Config

    /// Pending requests waiting to be prefilled and admitted.
    private var pendingQueue: [PendingEntry] = []

    /// The active batch iterator (nil when idle).
    private var iterator: BatchTokenIterator?

    /// The decode loop task (nil when idle).
    private var decodeTask: Task<Void, Never>?

    /// Monotonically increasing UID counter.
    private var nextUID: Int = 0

    private struct PendingEntry {
        let uid: Int
        let request: InferenceRequest
        let continuation: AsyncStream<Generation>.Continuation
        let submitted: Date
    }

    // MARK: - Init

    /// Create a scheduler bound to the given model context.
    ///
    /// - Parameters:
    ///   - context: The loaded model context. The scheduler holds a strong reference.
    ///   - config: Scheduling configuration.
    public init(context: ModelContext, config: Config = Config()) {
        self.context = context
        self.config = config
    }

    // MARK: - Public API

    /// Submit an inference request and return a streaming generation result.
    ///
    /// - Parameter request: The request to enqueue.
    /// - Returns: An `AsyncStream<Generation>` that emits text chunks, tool calls, and
    ///   completion info as the model generates tokens.
    /// - Throws: ``SchedulerError/queueFull`` if the pending queue is at capacity.
    public func submit(_ request: InferenceRequest) async throws -> AsyncStream<Generation> {
        guard pendingQueue.count < config.maxQueueSize else {
            throw SchedulerError.queueFull
        }

        var continuation: AsyncStream<Generation>.Continuation!
        let stream = AsyncStream<Generation> { continuation = $0 }

        let uid = nextUID
        nextUID += 1

        pendingQueue.append(PendingEntry(
            uid: uid,
            request: request,
            continuation: continuation,
            submitted: .now
        ))

        // Start the decode loop if it's not already running
        ensureDecodeLoopRunning()

        return stream
    }

    // MARK: - Internal

    private func ensureDecodeLoopRunning() {
        guard decodeTask == nil || decodeTask!.isCancelled else { return }
        decodeTask = Task { [weak self] in
            await self?.runDecodeLoop()
        }
    }

    /// The main decode loop: runs until both the pending queue and active batch are empty.
    private func runDecodeLoop() async {
        while !pendingQueue.isEmpty || (iterator?.batchSize ?? 0) > 0 {
            // Admit pending requests into the active batch
            admitPending()

            guard let iter = iterator, iter.batchSize > 0 else { break }

            // Run one decode step (this is the hot path)
            let stepResults = iter.step()

            // Deliver tokens and handle completions
            processStepResults(stepResults)

            // Yield to allow new submissions to be processed
            await Task.yield()
        }

        // Cleanup
        iterator = nil
        decodeTask = nil
    }

    /// Admit pending requests into the active batch (up to `completionBatchSize`).
    ///
    /// Prefill runs synchronously on the actor's executor. MLX operations are
    /// non-blocking (graph construction is fast; GPU evaluation is async via
    /// `asyncEval`), so this does not stall other actor methods for long.
    private func admitPending() {
        let currentSize = iterator?.batchSize ?? 0
        let available = config.completionBatchSize - currentSize
        guard available > 0, !pendingQueue.isEmpty else { return }

        let toAdmit = Array(pendingQueue.prefix(min(available, config.prefillBatchSize)))
        pendingQueue.removeFirst(toAdmit.count)

        // Prefill each pending sequence individually on the actor's executor
        var prefilled: [(sequence: SequenceState, caches: [KVCacheSimple])] = []

        for entry in toAdmit {
            do {
                let promptTokens = entry.request.input.text.tokens.asArray(Int32.self).map { Int($0) }
                let params = entry.request.parameters

                let (caches, firstToken) = try BatchTokenIterator.prefillSingle(
                    promptTokens: promptTokens,
                    model: context.model,
                    parameters: params
                )

                let seq = SequenceState(
                    uid: entry.uid,
                    currentToken: firstToken,
                    promptTokens: promptTokens,
                    parameters: params,
                    tokenizer: context.tokenizer,
                    continuation: entry.continuation
                )

                prefilled.append((seq, caches))

            } catch {
                // Prefill failed — deliver the error and skip this sequence
                entry.continuation.finish()
            }
        }

        guard !prefilled.isEmpty else { return }

        // Merge into the active batch iterator
        if let existingIter = iterator {
            // Single-to-batch upgrade: add each new sequence to the existing batch
            for (seq, caches) in prefilled {
                existingIter.addSequence(seq, perLayerCache: caches)
            }
        } else {
            // Create a new iterator from the freshly prefilled sequences
            let sequences = prefilled.map { $0.sequence }
            let caches = prefilled.map { $0.caches }
            iterator = BatchTokenIterator(
                sequences: sequences,
                perSequenceCaches: caches,
                model: context.model
            )
        }
    }

    /// Process decode step results: deliver tokens, finish completed sequences.
    private func processStepResults(_ results: [BatchStepResult]) {
        guard let iter = iterator else { return }

        var finishedIndices: [Int] = []

        for (idx, result) in results.enumerated() {
            let seq = iter.activeSequences[idx]

            // Decode the token to text
            let text = context.tokenizer.decode(tokens: [result.token])

            // Check if it's actually an EOS or just an empty string
            let isEOS = result.token == context.tokenizer.eosTokenId

            if !isEOS && !text.isEmpty {
                seq.continuation.yield(.chunk(text))
            }

            if result.isDone {
                let promptTime = seq.decodeStartTime.map { $0.timeIntervalSince(seq.prefillStartTime) } ?? 0
                let decodeTime = seq.decodeStartTime.map { Date.now.timeIntervalSince($0) } ?? 0

                let info = GenerateCompletionInfo(
                    promptTokenCount: seq.promptTokenCount,
                    generationTokenCount: seq.numTokens,
                    promptTime: promptTime,
                    generationTime: decodeTime,
                    stopReason: seq.stopReason
                )
                seq.continuation.yield(.info(info))
                seq.continuation.finish()
                finishedIndices.append(idx)
            }
        }

        // Remove finished sequences (in reverse order to preserve indices)
        if !finishedIndices.isEmpty {
            iter.removeSequences(at: finishedIndices.reversed())
        }

        // If the batch is now empty, clear the iterator
        if iter.batchSize == 0 {
            iterator = nil
        }
    }

    // MARK: - Cancellation

    /// Cancel all pending and active requests.
    public func cancelAll() {
        // Finish all pending continuations
        for entry in pendingQueue {
            entry.continuation.finish()
        }
        pendingQueue.removeAll()

        // The decode loop will see an empty batch and exit naturally
        decodeTask?.cancel()
    }
}
