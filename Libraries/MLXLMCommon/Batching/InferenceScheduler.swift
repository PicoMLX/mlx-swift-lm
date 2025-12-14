// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import Tokenizers

/// Thread-safe accumulator for prefill statistics from BatchTokenIterator hooks.
private final class PrefillStatsAccumulator: @unchecked Sendable {
    private let lock = NSLock()
    private var _totalTime: TimeInterval = 0
    private var _tokenCount: Int = 0
    private var _chunkCount: Int = 0
    
    func recordPrefill(duration: TimeInterval, tokenCount: Int) {
        lock.lock()
        defer { lock.unlock() }
        _totalTime += duration
        _tokenCount += tokenCount
        _chunkCount += 1
    }
    
    func harvest() -> (totalTime: TimeInterval, tokenCount: Int, chunkCount: Int) {
        lock.lock()
        defer { lock.unlock() }
        let result = (_totalTime, _tokenCount, _chunkCount)
        _totalTime = 0
        _tokenCount = 0
        _chunkCount = 0
        return result
    }
}

/// Scheduler for continuous batching of LLM inference requests.
///
/// Each `InferenceScheduler` is bound to a single model and manages:
/// - Request queuing and admission
/// - Prefill/decode scheduling with fairness
/// - Per-request streaming via `AsyncStream<Generation>`
/// - Graceful shutdown and cancellation
///
/// Usage:
/// ```swift
/// let scheduler = InferenceScheduler(model: model, tokenizer: tokenizer)
/// let stream = try await scheduler.enqueue(request)
/// for await event in stream {
///     switch event {
///     case .chunk(let text):
///         print(text, terminator: "")
///     case .info(let info):
///         print("\nFinished: \(info.finishReason ?? .stop)")
///     case .toolCall(let toolCall):
///         print("Tool call: \(toolCall)")
///     }
/// }
/// ```
public actor InferenceScheduler {
    
    // MARK: - Dependencies
    
    private let model: any LanguageModel
    private let tokenizer: Tokenizer
    private let config: SchedulerConfig
    
    // MARK: - State
    
    private var queuedRequests: [(InferenceRequest, AsyncStream<Generation>.Continuation)] = []
    private var activeSequences: [UUID: SequenceState] = [:]
    private var isShuttingDown = false
    private var loopTask: Task<Void, Never>?
    
    // MARK: - Batch Infrastructure
    
    private var batchIterator: BatchTokenIterator?
    private var uidToRequestID: [Int: UUID] = [:]
    private var requestIDToUID: [UUID: Int] = [:]
    private let prefillAccumulator = PrefillStatsAccumulator()
    
    // MARK: - Statistics
    
    private var totalDecodeTime: TimeInterval = 0
    private var decodeTickCount = 0
    private var totalPrefillTime: TimeInterval = 0
    private var prefillTokenCount = 0
    private var prefillChunkCount = 0
    private var totalRequestsProcessed = 0
    private var outcomeSuccess = 0
    private var outcomeCancelled = 0
    private var outcomeTimeout = 0
    private var outcomeError = 0
    
    // MARK: - Stop Tokens
    
    private let modelStopTokens: Set<Int>
    
    // MARK: - Initialization
    
    public init(
        model: any LanguageModel,
        tokenizer: Tokenizer,
        config: SchedulerConfig = SchedulerConfig(),
        extraEOSTokens: [String] = []
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        // Build stop tokens from tokenizer
        var stopTokens = Set<Int>()
        if let eos = tokenizer.eosTokenId {
            stopTokens.insert(eos)
        }
        for token in extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                stopTokens.insert(id)
            }
        }
        self.modelStopTokens = stopTokens
    }
    
    // MARK: - Public API
    
    /// Enqueue a request for processing.
    ///
    /// Returns an `AsyncStream` that yields `Generation` events as tokens are generated.
    /// The stream completes with `.info` when generation finishes.
    ///
    /// - Throws: `SchedulerError.shutdownInProgress` if the scheduler is shutting down.
    /// - Throws: `SchedulerError.queueFull` if the request queue is at capacity.
    public func enqueue(_ request: InferenceRequest) throws -> AsyncStream<Generation> {
        // Reject if shutting down
        guard !isShuttingDown else {
            throw SchedulerError.shutdownInProgress
        }
        
        // Reject if queue full
        guard queuedRequests.count < config.maxQueuedRequests else {
            throw SchedulerError.queueFull
        }
        
        return AsyncStream { continuation in
            self.handleEnqueue(request, continuation: continuation)
        }
    }
    
    /// Get statistics for a specific request.
    public func stats(for requestID: UUID) -> RequestStats? {
        guard let seq = activeSequences[requestID] else { return nil }
        return RequestStats(
            requestID: requestID,
            promptTokens: seq.inputTokens.count,
            generatedTokens: seq.generatedTokens.count,
            promptTime: 0, // TODO: track per-request timing
            generationTime: 0
        )
    }
    
    /// Get aggregated statistics across all requests.
    public func aggregatedStats() -> AggregatedStats {
        AggregatedStats(
            currentBatchSize: activeSequences.count,
            queueLength: queuedRequests.count,
            prefillPending: batchIterator?.pendingPrefillCount ?? 0,
            avgDecodeTickLatency: decodeTickCount > 0 ? totalDecodeTime / Double(decodeTickCount) : 0,
            avgPrefillChunkLatency: prefillChunkCount > 0 ? totalPrefillTime / Double(prefillChunkCount) : 0,
            totalPrefillTime: totalPrefillTime,
            prefillTokenCount: prefillTokenCount,
            totalRequestsProcessed: totalRequestsProcessed,
            outcomeSuccess: outcomeSuccess,
            outcomeCancelled: outcomeCancelled,
            outcomeTimeout: outcomeTimeout,
            outcomeError: outcomeError
        )
    }
    
    /// Gracefully shutdown the scheduler.
    ///
    /// Stops accepting new requests and waits for in-flight requests to complete
    /// or times out after the grace period.
    public func shutdown() async {
        isShuttingDown = true
        
        // Reject all queued requests (finish streams without yielding - error was thrown at enqueue)
        for (_, continuation) in queuedRequests {
            continuation.finish()
        }
        queuedRequests.removeAll()
        
        // Wait for grace period or completion
        let deadline = Date().addingTimeInterval(config.shutdownGracePeriod)
        while !activeSequences.isEmpty && Date() < deadline {
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        }
        
        // Cancel remaining after grace period (guard against double-counting)
        for (_, seq) in activeSequences {
            if !seq.isFinished {
                seq.timeout()
                outcomeTimeout += 1
            }
        }
        activeSequences.removeAll()
        
        // Cancel the loop task
        loopTask?.cancel()
        loopTask = nil
        batchIterator = nil
        uidToRequestID.removeAll()
        requestIDToUID.removeAll()
    }
    
    // MARK: - Private: Enqueue Handling
    
    private func handleEnqueue(
        _ request: InferenceRequest,
        continuation: AsyncStream<Generation>.Continuation
    ) {
        // Note: shutdownInProgress and queueFull checks are done in enqueue() which throws
        
        // Set up cancellation handler
        continuation.onTermination = { @Sendable [weak self] _ in
            Task { [weak self] in
                await self?.handleCancellation(requestID: request.id)
            }
        }
        
        // Add to queue
        queuedRequests.append((request, continuation))
        
        // Start loop if not running
        if loopTask == nil {
            loopTask = Task { await schedulerLoop() }
        }
    }
    
    private func handleCancellation(requestID: UUID) {
        // Remove from queue if still there
        queuedRequests.removeAll { $0.0.id == requestID }
        
        // Mark active sequence as cancelled
        if let seq = activeSequences[requestID] {
            seq.cancel()
            outcomeCancelled += 1
            // Will be cleaned up in next tick
        }
    }
    
    // MARK: - Private: Scheduler Loop
    
    private func schedulerLoop() async {
        while !isShuttingDown || !activeSequences.isEmpty {
            // Admit new requests from queue
            admitFromQueue()
            
            // Run decode tick if we have active sequences
            // Note: Prefill/decode decisions are delegated to BatchTokenIterator
            if !activeSequences.isEmpty {
                await runDecodeTick()
                cleanupFinished()
            }
            
            // Check deadlines
            checkDeadlines()
            
            // Exit if shutdown and nothing left
            if isShuttingDown && activeSequences.isEmpty && queuedRequests.isEmpty {
                break
            }
            
            // Brief yield if nothing to do
            if activeSequences.isEmpty && queuedRequests.isEmpty {
                try? await Task.sleep(nanoseconds: 1_000_000) // 1ms
            }
        }
        
        loopTask = nil
    }
    
    // MARK: - Private: Admission
    
    private func admitFromQueue() {
        while !queuedRequests.isEmpty && activeSequences.count < config.maxBatchSize {
            let (request, continuation) = queuedRequests.removeFirst()
            
            // Create sequence state
            let seq = SequenceState(
                id: request.id,
                inputTokens: request.tokens,
                kvSlot: 0, // Will be assigned by BatchTokenIterator
                maxTokens: request.maxTokens,
                stopTokens: request.stopTokens,
                params: request.params,
                deadline: request.deadline,
                createdAt: request.createdAt,
                continuation: continuation
            )
            
            activeSequences[request.id] = seq
            
            // Add to batch iterator
            insertIntoBatch(seq)
            
            totalRequestsProcessed += 1
        }
    }
    
    private func insertIntoBatch(_ seq: SequenceState) {
        // Create iterator if needed
        if batchIterator == nil {
            // Use neutral defaults for batch-wide params; per-request params go via samplers
            let params = BatchGenerateParameters(
                maxTokens: config.maxBatchSize * 1000, // High limit, we control via per-request
                completionBatchSize: config.maxBatchSize,
                prefillBatchSize: config.prefillBatchSize,
                prefillStepSize: config.prefillStepSize,
                generation: GenerateParameters(), // Neutral defaults
                returnLogProbs: config.returnLogProbs
            )
            var iterator = BatchTokenIterator(
                model: model,
                parameters: params,
                stopTokens: modelStopTokens,
                unknownTokenId: tokenizer.unknownTokenId
            )
            
            // Set up prefill hooks for statistics
            let accumulator = prefillAccumulator
            iterator.onPrefillComplete = { duration, tokenCount in
                accumulator.recordPrefill(duration: duration, tokenCount: tokenCount)
            }
            
            batchIterator = iterator
        }
        
        // TODO: Per-request stop tokens require BatchTokenIterator extension (v2)
        // For now, only model-level stop tokens are used.
        
        // Insert into iterator with per-request sampler
        let sampler = seq.params.sampler()
        let processor = seq.params.processor()
        
        let uids = batchIterator!.insert(
            prompts: [seq.inputTokens],
            maxTokens: [seq.maxTokens],
            samplers: [sampler],
            processors: [processor]
        )
        
        if let uid = uids.first {
            uidToRequestID[uid] = seq.id
            requestIDToUID[seq.id] = uid
        }
    }
    
    // MARK: - Private: Decode Tick
    
    private func runDecodeTick() async {
        guard let iterator = batchIterator else { return }
        
        let start = Date.timeIntervalSinceReferenceDate
        
        // Serialize via DeviceEngine to prevent concurrent MLX evals across models.
        // BatchTokenIterator is @unchecked Sendable, so this is Swift 6 compliant.
        // We capture by value and return the mutated iterator to avoid "mutation of captured var" warning.
        let (responses, updatedIterator) = await DeviceEngine.shared.runForward {
            var localIterator = iterator
            let result = localIterator.next()
            return (result, localIterator)
        }
        
        // Update with the mutated iterator from the closure
        batchIterator = updatedIterator
        
        let elapsed = Date.timeIntervalSinceReferenceDate - start
        totalDecodeTime += elapsed
        decodeTickCount += 1
        
        // Harvest any prefill stats accumulated during this tick
        let prefillStats = prefillAccumulator.harvest()
        totalPrefillTime += prefillStats.totalTime
        prefillTokenCount += prefillStats.tokenCount
        prefillChunkCount += prefillStats.chunkCount
        
        // Process responses
        guard let responses = responses else { return }
        
        for response in responses {
            guard let requestID = uidToRequestID[response.uid],
                  let seq = activeSequences[requestID] else {
                continue
            }
            
            // Skip if cancelled
            if seq.isCancelled || seq.isFinished { continue }
            
            // Record token
            seq.addGeneratedToken(response.token)
            
            // Decode and emit text chunk
            let text = tokenizer.decode(tokens: [response.token])
            if !text.isEmpty {
                seq.emitChunk(text)
            }
            
            // Handle finish
            if let finishReason = response.finishReason {
                let reason: FinishReason = finishReason == .stop ? .stop : .length
                seq.complete(reason: reason)
                outcomeSuccess += 1
            }
        }
    }
    
    // MARK: - Private: Cleanup
    
    private func cleanupFinished() {
        // Remove finished sequences
        let finishedIDs = activeSequences.filter { $0.value.isFinished }.map { $0.key }
        
        for id in finishedIDs {
            activeSequences.removeValue(forKey: id)
            
            // Remove from batch iterator
            if let uid = requestIDToUID[id] {
                batchIterator?.remove(uids: [uid])
                uidToRequestID.removeValue(forKey: uid)
                requestIDToUID.removeValue(forKey: id)
            }
        }
        
        // Clear iterator if no active sequences
        if activeSequences.isEmpty {
            batchIterator = nil
            uidToRequestID.removeAll()
            requestIDToUID.removeAll()
        }
    }
    
    private func checkDeadlines() {
        let now = Date()
        for (_, seq) in activeSequences {
            if let deadline = seq.deadline, now >= deadline && !seq.isFinished {
                seq.timeout()
                outcomeTimeout += 1
            }
        }
    }
}
