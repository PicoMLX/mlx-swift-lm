// Copyright © 2026 Apple Inc.

import MLX

struct SchedulerRequest: @unchecked Sendable {
    var input: LMInput
    var parameters: GenerateParameters
    var cache: [KVCache]?
    var tokenizer: Tokenizer
    var configuration: ModelConfiguration
    var cachedKVState: [KVCache]?
    var promptCache: LRUPromptCache?
    var promptCacheModelName: String?
    var inputTokens: [Int]?
    var wiredMemoryTicket: WiredMemoryTicket?

    init(
        input: LMInput,
        parameters: GenerateParameters,
        cache: [KVCache]?,
        tokenizer: Tokenizer,
        configuration: ModelConfiguration,
        cachedKVState: [KVCache]? = nil,
        promptCache: LRUPromptCache? = nil,
        promptCacheModelName: String? = nil,
        inputTokens: [Int]? = nil,
        wiredMemoryTicket: WiredMemoryTicket? = nil
    ) {
        self.input = input
        self.parameters = parameters
        self.cache = cache
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.cachedKVState = cachedKVState
        self.promptCache = promptCache
        self.promptCacheModelName = promptCacheModelName
        self.inputTokens = inputTokens
        self.wiredMemoryTicket = wiredMemoryTicket
    }
}
