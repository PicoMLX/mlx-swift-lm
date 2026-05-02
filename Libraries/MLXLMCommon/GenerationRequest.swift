// Copyright © 2026 Apple Inc.

import MLX

/// A prepared generation request for APIs that accept more than one request at a time.
public struct GenerationRequest {
    public var input: LMInput
    public var parameters: GenerateParameters
    public var wiredMemoryTicket: WiredMemoryTicket?

    public init(
        input: LMInput,
        parameters: GenerateParameters = .init(),
        wiredMemoryTicket: WiredMemoryTicket? = nil
    ) {
        self.input = input
        self.parameters = parameters
        self.wiredMemoryTicket = wiredMemoryTicket
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
