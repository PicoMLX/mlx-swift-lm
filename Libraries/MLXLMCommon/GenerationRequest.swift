// Copyright © 2026 Apple Inc.

import MLX

/// A prepared generation request for APIs that accept more than one request at a time.
public struct GenerationRequest {
    public var input: LMInput
    public var parameters: GenerateParameters
    public var wiredMemoryTicket: WiredMemoryTicket?
    public var promptCache: LRUPromptCache?
    public var promptCacheSalt: UInt64

    public init(
        input: LMInput,
        parameters: GenerateParameters = .init(),
        wiredMemoryTicket: WiredMemoryTicket? = nil,
        promptCache: LRUPromptCache? = nil,
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
    case batchTooSmall
    case mismatchedRequestCounts
    case incompatibleRequests([Int])
    case cannotShareBatch([Int])
    case schedulerBusy
}
