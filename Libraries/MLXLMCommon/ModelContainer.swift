// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Container for models that guarantees single threaded access.
///
/// Wrap models used by e.g. the UI in a ModelContainer. Callers can access
/// the model and/or tokenizer (any values from the ``ModelContext``):
///
/// ```swift
/// let messages = [["role": "user", "content": prompt]]
/// let promptTokens = try await modelContainer.perform { context in
///     try context.tokenizer.applyChatTemplate(messages: messages)
/// }
/// ```
///
/// or:
///
/// ```swift
/// let userInput: UserInput
/// let result = await modelContainer.perform { context in
///     let input = try await context.processor.prepare(input: userInput)
///     return generate(
///         input: input, parameters: generateParameters, context: context
///     ) { tokens in
///     ...
///     }
/// }
/// ```
public final class ModelContainer: Sendable {
    private let context: SerialAccessContainer<ModelContext>

    /// Optional inference scheduler for concurrent/batched request handling.
    ///
    /// When set, ``generate(input:parameters:wiredMemoryTicket:)`` routes requests
    /// through the scheduler instead of running single-sequence inference directly.
    /// Multiple concurrent callers benefit from automatic batching.
    ///
    /// Set this during initialization via ``init(context:schedulerConfig:)`` or
    /// assign it directly after creation.
    public let scheduler: InferenceScheduler?

    public var configuration: ModelConfiguration {
        get async {
            await context.read { $0.configuration }
        }
    }

    public var processor: UserInputProcessor {
        get async {
            await context.read { $0.processor }
        }
    }

    public var tokenizer: Tokenizer {
        get async {
            await context.read { $0.tokenizer }
        }
    }

    /// Create a `ModelContainer` without a scheduler (standard single-sequence mode).
    public init(context: consuming ModelContext) {
        self.context = .init(context)
        self.scheduler = nil
    }

    /// Create a `ModelContainer` with an ``InferenceScheduler`` for batched inference.
    ///
    /// When a scheduler is provided, concurrent ``generate`` calls are automatically
    /// batched together, improving throughput for multi-user or multi-request scenarios.
    ///
    /// - Parameters:
    ///   - context: The loaded model context.
    ///   - schedulerConfig: Configuration for the inference scheduler.
    public init(context: ModelContext, schedulerConfig: InferenceScheduler.Config) {
        self.scheduler = InferenceScheduler(context: context, config: schedulerConfig)
        self.context = .init(context)
    }

    /// Perform an action on the model and/or tokenizer. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(_:) that uses a ModelContext")
    public func perform<R: Sendable>(
        _ action: @Sendable (any LanguageModel, Tokenizer) throws -> sending R
    )
        async rethrows
        -> sending R
    {
        try await context.read {
            try action($0.model, $0.tokenizer)
        }
    }

    /// Perform an action on the model and/or tokenizer with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(values:_:) that uses a ModelContext")
    public func perform<V: Sendable, R: Sendable>(
        values: V, _ action: @Sendable (any LanguageModel, Tokenizer, V) throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try action($0.model, $0.tokenizer, values)
        }
    }

    /// Perform an action on the ``ModelContext``. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    ///
    /// - Note: The closure receives `ModelContext` which is not `Sendable`. This is intentional -
    ///   the closure runs within the actor's isolation, ensuring thread-safe access to the model.
    /// - Note: The `sending` keyword indicates the return value is transferred (not shared) across
    ///   isolation boundaries, allowing non-Sendable types to be safely returned.
    public func perform<R: Sendable>(
        _ action: @Sendable (ModelContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0)
        }
    }

    /// Perform an action on the ``ModelContext`` with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V: Sendable, R: Sendable>(
        values: V, _ action: @Sendable (ModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0, values)
        }
    }

    /// Perform an action on the ``ModelContext`` with additional (non `Sendable`) context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V, R: Sendable>(
        nonSendable values: consuming V, _ action: @Sendable (ModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        let values = SendableBox(values)
        return try await context.read {
            try await action($0, values.consume())
        }
    }

    /// Update the owned `ModelContext`.
    /// - Parameter action: update action
    public func update(_ action: @Sendable (inout ModelContext) -> Void) async {
        await context.update {
            action(&$0)
        }
    }

    // MARK: - Thread-safe convenience methods

    /// Prepare user input for generation.
    ///
    /// This method safely prepares input within the actor's isolation,
    /// avoiding the need for closure-based `perform` calls.
    ///
    /// - Parameter input: The user input to prepare
    /// - Returns: Prepared language model input (transferred via `sending`)
    /// - Note: The `sending` keyword indicates the return value is transferred (not shared),
    ///   allowing non-Sendable types like `LMInput` to safely cross isolation boundaries.
    public func prepare(input: consuming sending UserInput) async throws -> sending LMInput {
        let processor = await self.processor
        return try await processor.prepare(input: input)
    }

    /// Generate tokens from prepared input, returning an AsyncStream.
    ///
    /// This method provides a thread-safe way to generate tokens without
    /// needing to use closure-based `perform` calls.
    ///
    /// Example:
    /// ```swift
    /// let input = try await modelContainer.prepare(input: userInput)
    /// let stream = try modelContainer.generate(input: input, parameters: parameters)
    /// for await generation in stream {
    ///     switch generation {
    ///     case .chunk(let text): print(text)
    ///     case .info(let info): print(info.tokensPerSecond)
    ///     case .toolCall(let call): handleToolCall(call)
    ///     }
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - input: Prepared language model input (transferred via `sending`)
    ///   - parameters: Generation parameters
    ///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination
    /// - Returns: An AsyncStream of generation events
    /// - Note: The `sending` parameter indicates the input is transferred (not shared),
    ///   allowing non-Sendable types like `LMInput` to safely cross isolation boundaries.
    public func generate(
        input: consuming sending LMInput,
        parameters: GenerateParameters,
        wiredMemoryTicket: WiredMemoryTicket? = nil
    ) async throws -> AsyncStream<Generation> {
        try await generate(
            input: input,
            cache: nil,
            parameters: parameters,
            wiredMemoryTicket: wiredMemoryTicket
        )
    }

    public func generate(
        input: consuming sending LMInput,
        cache: consuming [KVCache]? = nil,
        parameters: GenerateParameters,
        wiredMemoryTicket: WiredMemoryTicket? = nil
    ) async throws -> AsyncStream<Generation> {
        let input = SendableBox(input)
        let cache = SendableBox(cache)
        return try await context.read { context in
            let input = input.consume()
            let cache = cache.consume()

            if let scheduler,
                InferenceScheduler.isBatchCompatible(
                    input: input,
                    promptCache: cache,
                    parameters: parameters,
                    context: context
                )
            {
                return try await scheduler.submit(
                    InferenceRequest(input: input, promptCache: cache, parameters: parameters)
                )
            }

            return try MLXLMCommon.generate(
                input: input,
                cache: cache,
                parameters: parameters,
                context: context,
                wiredMemoryTicket: wiredMemoryTicket
            )
        }
    }

    public func generateTokens(
        input: consuming sending LMInput,
        cache: consuming [KVCache]? = nil,
        parameters: GenerateParameters,
        includeStopToken: Bool = false,
        wiredMemoryTicket: WiredMemoryTicket? = nil
    ) async throws -> AsyncStream<TokenGeneration> {
        let input = SendableBox(input)
        let cache = SendableBox(cache)
        return try await context.read { context in
            try MLXLMCommon.generateTokens(
                input: input.consume(),
                cache: cache.consume(),
                parameters: parameters,
                context: context,
                includeStopToken: includeStopToken,
                wiredMemoryTicket: wiredMemoryTicket
            )
        }
    }

    /// Decode token IDs to a string.
    ///
    /// - Parameter tokens: Array of token IDs
    /// - Returns: Decoded string
    public func decode(tokens: [Int]) async -> String {
        let tokenizer = await self.tokenizer
        return tokenizer.decode(tokens: tokens)
    }

    /// Encode a string to token IDs.
    ///
    /// - Parameter text: Text to encode
    /// - Returns: Array of token IDs
    public func encode(_ text: String) async -> [Int] {
        let tokenizer = await self.tokenizer
        return tokenizer.encode(text: text)
    }

    /// Apply chat template to messages and return token IDs.
    ///
    /// - Parameter messages: Array of message dictionaries with "role" and "content" keys
    /// - Returns: Array of token IDs
    public func applyChatTemplate(messages: [[String: String]]) async throws -> [Int] {
        let tokenizer = await self.tokenizer
        return try tokenizer.applyChatTemplate(messages: messages)
    }
}
