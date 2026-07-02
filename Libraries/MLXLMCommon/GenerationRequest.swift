// Copyright © 2026 Apple Inc.

import MLX

/// A prepared generation request for APIs that accept more than one request at
/// a time (the manual ``ModelContainer/generateBatched(_:)`` API and the
/// auto-upgrading ``InferenceScheduler``).
///
/// ``input`` is kept as a full ``LMInput`` (not reduced to `[Int]`) so the
/// image/video tensors a VLM prefill needs are carried end-to-end; batched VLM
/// prefill can then be added as a `PrefillBatch` extension without reworking
/// this request type or the engine signatures.
///
/// `promptCache` is typed as `(any PromptCaching)?` (a `Sendable` protocol
/// bound) rather than the concrete `LRUPromptCache` so a future block/paged
/// cache lands without churning this public surface. Disk persistence is an
/// app-level concern and is not part of the protocol.
///
/// Not `Sendable`: ``input`` carries non-`Sendable` `MLXArray` tensors. The
/// batched API transfers requests across isolation with `sending` (the same
/// way ``ModelContainer/generate(input:parameters:wiredMemoryTicket:)`` already
/// transfers an ``LMInput``).
public struct GenerationRequest {
    /// The prepared model input (text and, for VLMs, image/video tensors).
    public var input: LMInput

    /// Generation parameters (sampling, penalties, max tokens, …).
    public var parameters: GenerateParameters

    /// Optional wired-memory ticket for policy-based coordination.
    public var wiredMemoryTicket: WiredMemoryTicket?

    /// Optional cross-request prompt cache for prefix KV reuse.
    public var promptCache: (any PromptCaching)?

    /// Per-agent isolation salt for prompt-cache lookups.
    public var promptCacheSalt: UInt64

    public init(
        input: LMInput,
        parameters: GenerateParameters = .init(),
        wiredMemoryTicket: WiredMemoryTicket? = nil,
        promptCache: (any PromptCaching)? = nil,
        promptCacheSalt: UInt64 = 0
    ) {
        self.input = input
        self.parameters = parameters
        self.wiredMemoryTicket = wiredMemoryTicket
        self.promptCache = promptCache
        self.promptCacheSalt = promptCacheSalt
    }
}

/// Errors thrown by explicit batched generation APIs.
public enum BatchedGenerationError: Error, Equatable {
    /// `generateBatched` was given fewer than the minimum supported requests.
    case batchTooSmall

    /// The number of requests did not match the number of expected outputs.
    case mismatchedRequestCounts

    /// One or more requests cannot be batched together (indices listed).
    case incompatibleRequests([Int])

    /// The scheduler was busy with an incompatible workload.
    case schedulerBusy
}
