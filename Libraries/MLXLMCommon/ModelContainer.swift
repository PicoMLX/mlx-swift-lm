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
public actor ModelContainer {
    var context: ModelContext
    public var configuration: ModelConfiguration { context.configuration }
    /// Optional batched inference scheduler for this model.
    public var scheduler: InferenceScheduler?
    
    public init(context: ModelContext, schedulerConfig: SchedulerConfig? = nil) {
        self.context = context
        
        if let schedulerConfig {
            self.scheduler = InferenceScheduler(
                model: context.model,
                tokenizer: context.tokenizer,
                config: schedulerConfig
            )
        } else {
            self.scheduler = nil
        }
    }

    /// Perform an action on the model and/or tokenizer. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(_:) that uses a ModelContext")
    public func perform<R>(_ action: @Sendable (any LanguageModel, Tokenizer) throws -> R) rethrows
        -> R
    {
        try action(context.model, context.tokenizer)
    }

    /// Perform an action on the model and/or tokenizer with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(values:_:) that uses a ModelContext")
    public func perform<V, R>(
        values: V, _ action: @Sendable (any LanguageModel, Tokenizer, V) throws -> R
    ) rethrows -> R {
        try action(context.model, context.tokenizer, values)
    }

    /// Perform an action on the ``ModelContext``. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<R>(_ action: @Sendable (ModelContext) async throws -> R) async rethrows -> R
    {
        try await action(context)
    }

    /// Perform an action on the ``ModelContext`` with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V, R>(
        values: V, _ action: @Sendable (ModelContext, V) async throws -> R
    ) async rethrows -> R {
        try await action(context, values)
    }

    /// Update the owned `ModelContext`.
    /// - Parameter action: update action
    public func update(_ action: @Sendable (inout ModelContext) -> Void) {
        action(&context)
    }
}

extension ModelContainer {
    public struct ChatGenerationRequest {
        public let prompt: String
        public let maxTokens: Int
        public let temperature: Float
        public let topP: Float
        public let topK: Int
        public let stopSequences: [String]?
        public let timeout: TimeInterval?

        public init(
            prompt: String,
            maxTokens: Int = 256,
            temperature: Float = 0.7,
            topP: Float = 0.9,
            topK: Int = 40,
            stopSequences: [String]? = nil,
            timeout: TimeInterval? = nil
        ) {
            self.prompt = prompt
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topP = topP
            self.topK = topK
            self.stopSequences = stopSequences
            self.timeout = timeout
        }
    }

    /// Convenience batched generation API using the optional scheduler.
    /// If `scheduler` is nil, this will `fatalError` or you can choose to fall back
    /// to the non-batched path.
    public func generateStream(
        _ request: ChatGenerationRequest
    ) async throws -> AsyncStream<Generation> {
        guard let scheduler else {
            fatalError("ModelContainer.scheduler is not configured")
        }

        // Tokenization happens under ModelContainer actor isolation.
        let tokens = try await perform { context in
            try context.tokenizer.encode(text: request.prompt)
        }

        let params = GenerateParameters(
            temperature: request.temperature,
            topP: request.topP
        )

        let now = Date()
        let deadline = request.timeout.map { now.addingTimeInterval($0) }

        let inferenceReq = InferenceRequest(
            id: UUID(),
            tokens: tokens,
            params: params,
            maxTokens: request.maxTokens,
            stopTokens: nil,   // or map stopSequences → IDs if you want
            deadline: deadline,
            createdAt: now
        )

        return try await scheduler.enqueue(inferenceReq)
    }
}
