// Copyright © 2026 Apple Inc.

import MLX

/// Internal carrier threaded from the scheduler into the ``EngineDriver``.
///
/// This bundles everything the driver needs to admit a request to the engine
/// and to write its final KV state back to the prompt cache when it finishes.
///
/// Not `Sendable`: ``input`` carries non-`Sendable` `MLXArray` tensors, and
/// `cachedKVState` carries non-`Sendable` `KVCache` values. It is transferred
/// across the actor boundary with a Swift 6 `sending` parameter (region-based
/// isolation), never `@unchecked Sendable`.
struct SchedulerRequest {
    /// The prepared model input.
    var input: LMInput

    /// Generation parameters.
    var parameters: GenerateParameters

    /// The token IDs of `input.text` (precomputed so the driver does not have
    /// to materialize them again).
    var inputTokens: [Int]

    /// Model identifier for prompt-cache isolation / write-back keys.
    var modelName: String

    /// Optional cross-request prompt cache.
    var promptCache: (any PromptCaching)?

    /// Per-agent prompt-cache isolation salt.
    var promptCacheSalt: UInt64

    /// A KV cache prefix recovered from the prompt cache (covering a leading
    /// run of `inputTokens`), or `nil`. When present, only the remaining tokens
    /// need prefilling. Carried for a future single-path cache-reuse hookup;
    /// the batched engine prefills from scratch today.
    var cachedKVState: [KVCache]?

    /// The tokens not covered by `cachedKVState` (the prefill remainder).
    var cachedPromptRemainder: [Int]?

    /// Optional wired-memory ticket.
    var wiredMemoryTicket: WiredMemoryTicket?

    init(
        input: LMInput,
        parameters: GenerateParameters,
        inputTokens: [Int],
        modelName: String,
        promptCache: (any PromptCaching)? = nil,
        promptCacheSalt: UInt64 = 0,
        cachedKVState: [KVCache]? = nil,
        cachedPromptRemainder: [Int]? = nil,
        wiredMemoryTicket: WiredMemoryTicket? = nil
    ) {
        self.input = input
        self.parameters = parameters
        self.inputTokens = inputTokens
        self.modelName = modelName
        self.promptCache = promptCache
        self.promptCacheSalt = promptCacheSalt
        self.cachedKVState = cachedKVState
        self.cachedPromptRemainder = cachedPromptRemainder
        self.wiredMemoryTicket = wiredMemoryTicket
    }
}
