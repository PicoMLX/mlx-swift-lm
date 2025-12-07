// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// MARK: - Requests

/// Request at the ModelRouter level (includes modelID for routing).
///
/// Once routed to a `ModelContainer`, the request is converted to `InferenceRequest`
/// which no longer needs the modelID.
public struct RoutableRequest: Sendable, Identifiable {
    public let id: UUID
    public let modelID: String
    public let tokens: [Int]
    public let params: GenerateParameters
    public let maxTokens: Int
    public let stopTokens: Set<Int>?
    public let deadline: Date?
    public let createdAt: Date
    
    public init(
        id: UUID = UUID(),
        modelID: String,
        tokens: [Int],
        params: GenerateParameters = GenerateParameters(),
        maxTokens: Int,
        stopTokens: Set<Int>? = nil,
        deadline: Date? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.modelID = modelID
        self.tokens = tokens
        self.params = params
        self.maxTokens = maxTokens
        self.stopTokens = stopTokens
        self.deadline = deadline
        self.createdAt = createdAt
    }
    
    /// Convert to model-bound InferenceRequest (strips modelID)
    public func toInferenceRequest() -> InferenceRequest {
        InferenceRequest(
            id: id,
            tokens: tokens,
            params: params,
            maxTokens: maxTokens,
            stopTokens: stopTokens,
            deadline: deadline,
            createdAt: createdAt
        )
    }
}

/// Request inside InferenceScheduler (model-bound, no modelID needed).
public struct InferenceRequest: Sendable, Identifiable {
    public let id: UUID
    public let tokens: [Int]
    public let params: GenerateParameters
    public let maxTokens: Int
    public let stopTokens: Set<Int>?
    public let deadline: Date?
    public let createdAt: Date
    
    public init(
        id: UUID = UUID(),
        tokens: [Int],
        params: GenerateParameters = GenerateParameters(),
        maxTokens: Int,
        stopTokens: Set<Int>? = nil,
        deadline: Date? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.tokens = tokens
        self.params = params
        self.maxTokens = maxTokens
        self.stopTokens = stopTokens
        self.deadline = deadline
        self.createdAt = createdAt
    }
}

// MARK: - Events

/// Events emitted by the scheduler for each token generated.
public enum TokenEvent: Sendable {
    /// A token was generated
    case token(Int, textDelta: String?, logProbs: MLXArray?)
    
    /// Generation completed
    case done(finishReason: FinishReason)
    
    /// An error occurred
    case error(SchedulerError)
}

/// Reason why generation finished.
public enum FinishReason: String, Sendable, Equatable {
    /// Hit a stop token (EOS or custom)
    case stop
    
    /// Hit maxTokens limit
    case length
    
    /// Request was cancelled by client
    case cancelled
    
    /// Request exceeded its deadline
    case timeout
    
    /// An error occurred during generation
    case error
}

// MARK: - Errors

/// Errors that can occur in the scheduler.
public enum SchedulerError: Error, Sendable, Equatable {
    /// The scheduler is shutting down and not accepting new requests
    case shutdownInProgress
    
    /// The request queue is full
    case queueFull
    
    /// Model inference failed
    case modelFailed(String, error: MLXError?)
    
    /// Cache allocation failed
    case cacheAllocationFailed
    
    /// Request was cancelled
    case requestCancelled
    
    /// Request timed out
    case requestTimeout
    
    /// Invalid request parameters
    case invalidRequest(String)
}

// MARK: - Configuration

/// Configuration for the InferenceScheduler.
public struct SchedulerConfig: Sendable {
    /// Maximum number of sequences in decode batch
    public var maxBatchSize: Int
    
    /// Maximum requests waiting in queue
    public var maxQueuedRequests: Int
    
    /// Number of prompts to prefill at a time
    public var prefillBatchSize: Int
    
    /// Number of prompt tokens per prefill chunk
    public var prefillStepSize: Int
    
    /// Number of decode ticks between prefill opportunities (fairness)
    public var decodeTicksBetweenPrefill: Int
    
    /// Whether to compute and return log probabilities
    public var returnLogProbs: Bool
    
    /// Grace period for shutdown before cancelling remaining requests
    public var shutdownGracePeriod: TimeInterval
    
    public init(
        maxBatchSize: Int = 16,
        maxQueuedRequests: Int = 64,
        prefillBatchSize: Int = 8,
        prefillStepSize: Int = 2048,
        decodeTicksBetweenPrefill: Int = 3,
        returnLogProbs: Bool = false,
        shutdownGracePeriod: TimeInterval = 30
    ) {
        self.maxBatchSize = maxBatchSize
        self.maxQueuedRequests = maxQueuedRequests
        self.prefillBatchSize = prefillBatchSize
        self.prefillStepSize = prefillStepSize
        self.decodeTicksBetweenPrefill = decodeTicksBetweenPrefill
        self.returnLogProbs = returnLogProbs
        self.shutdownGracePeriod = shutdownGracePeriod
    }
}

// MARK: - Statistics

/// Per-request statistics.
public struct RequestStats: Sendable {
    public let requestID: UUID
    public var promptTokens: Int
    public var generatedTokens: Int
    public var promptTime: TimeInterval
    public var generationTime: TimeInterval
    
    public var tokensPerSecond: Double {
        guard generationTime > 0 else { return 0 }
        return Double(generatedTokens) / generationTime
    }
    
    public init(
        requestID: UUID,
        promptTokens: Int = 0,
        generatedTokens: Int = 0,
        promptTime: TimeInterval = 0,
        generationTime: TimeInterval = 0
    ) {
        self.requestID = requestID
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.promptTime = promptTime
        self.generationTime = generationTime
    }
}

/// Aggregated statistics across all requests.
public struct AggregatedStats: Sendable {
    /// Active decode sequences
    public let currentBatchSize: Int
    
    /// Pending requests in queue
    public let queueLength: Int
    
    /// Sequences awaiting prefill completion
    public let prefillPending: Int
    
    /// Average time per decode tick
    public let avgDecodeTickLatency: TimeInterval
    
    /// Average time per prefill chunk
    public let avgPrefillChunkLatency: TimeInterval
    
    /// Total requests processed since scheduler start
    public let totalRequestsProcessed: Int
    
    /// Requests that completed successfully
    public let outcomeSuccess: Int
    
    /// Requests that were cancelled
    public let outcomeCancelled: Int
    
    /// Requests that timed out
    public let outcomeTimeout: Int
    
    /// Requests that failed with errors
    public let outcomeError: Int
    
    public init(
        currentBatchSize: Int = 0,
        queueLength: Int = 0,
        prefillPending: Int = 0,
        avgDecodeTickLatency: TimeInterval = 0,
        avgPrefillChunkLatency: TimeInterval = 0,
        totalRequestsProcessed: Int = 0,
        outcomeSuccess: Int = 0,
        outcomeCancelled: Int = 0,
        outcomeTimeout: Int = 0,
        outcomeError: Int = 0
    ) {
        self.currentBatchSize = currentBatchSize
        self.queueLength = queueLength
        self.prefillPending = prefillPending
        self.avgDecodeTickLatency = avgDecodeTickLatency
        self.avgPrefillChunkLatency = avgPrefillChunkLatency
        self.totalRequestsProcessed = totalRequestsProcessed
        self.outcomeSuccess = outcomeSuccess
        self.outcomeCancelled = outcomeCancelled
        self.outcomeTimeout = outcomeTimeout
        self.outcomeError = outcomeError
    }
}
