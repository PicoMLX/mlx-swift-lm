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

    /// Optional wired-memory ticket.
    var wiredMemoryTicket: WiredMemoryTicket?

    /// Whether ``PromptCachePolicy`` permits prompt-cache use for this
    /// request (`LRUPromptCache.canUsePromptCache`), decided ONCE by the
    /// scheduler at submit and threaded through so the driver's prefix fetch
    /// AND its write-back flags both respect policy — no path re-derives (or
    /// silently ignores) it. Defaults to `true` for callers that construct
    /// requests directly (tests), preserving the opt-in-by-`promptCache`
    /// behavior; the scheduler always passes its computed decision.
    var cacheEligible: Bool

    init(
        input: LMInput,
        parameters: GenerateParameters,
        inputTokens: [Int],
        modelName: String,
        promptCache: (any PromptCaching)? = nil,
        promptCacheSalt: UInt64 = 0,
        wiredMemoryTicket: WiredMemoryTicket? = nil,
        cacheEligible: Bool = true
    ) {
        self.input = input
        self.parameters = parameters
        self.inputTokens = inputTokens
        self.modelName = modelName
        self.promptCache = promptCache
        self.promptCacheSalt = promptCacheSalt
        self.wiredMemoryTicket = wiredMemoryTicket
        self.cacheEligible = cacheEligible
    }
}
