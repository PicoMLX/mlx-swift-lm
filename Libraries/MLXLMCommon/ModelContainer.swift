// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

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

    /// Optional continuous-batching scheduler. When set, the streaming
    /// `generate(input:parameters:wiredMemoryTicket:)` transparently routes
    /// through it (auto-upgrading single→batch as concurrent requests arrive)
    /// for non-VLM models. When `nil`, generation uses the existing
    /// single-stream path with **identical** behavior — strictly opt-in.
    private let scheduler: InferenceScheduler?

    /// Optional cross-request prompt cache (prefix KV reuse), shared with the
    /// scheduler. Typed as the protocol so a future block/paged cache lands
    /// without churning this surface; disk persistence is an app-level
    /// concern.
    public let promptCache: (any PromptCaching)?

    /// Whether a continuous-batching scheduler is installed. Consumers (e.g.
    /// ``ChatSession``) use this to decide whether to route turns through the
    /// transparent batched ``generate(input:parameters:wiredMemoryTicket:)``
    /// path (which reuses KV state via ``promptCache``) instead of holding a
    /// session-local `[KVCache]`.
    var usesScheduler: Bool { scheduler != nil }

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

    public init(context: consuming ModelContext) {
        self.context = .init(context)
        self.scheduler = nil
        self.promptCache = nil
    }

    /// Create a container with an optional continuous-batching scheduler and/or
    /// cross-request prompt cache.
    ///
    /// Passing `scheduler: nil` (the default) yields a container behaviorally
    /// identical to ``init(context:)`` — the single-stream path, unchanged.
    /// Passing a scheduler enables transparent batched routing for non-VLM
    /// models; VLM contexts (`loadedAsVLM == true`) always use the single path.
    public init(
        context: consuming ModelContext,
        scheduler: InferenceScheduler?,
        promptCache: (any PromptCaching)? = nil
    ) {
        self.context = .init(context)
        self.scheduler = scheduler
        self.promptCache = promptCache
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

    /// The resolved local model directory for the loaded container.
    public var modelDirectory: URL {
        get async throws {
            try (await configuration).modelDirectory
        }
    }

    /// The resolved local tokenizer directory for the loaded container.
    public var tokenizerDirectory: URL {
        get async throws {
            try (await configuration).tokenizerDirectory
        }
    }

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
        let input = SendableBox(input)

        // Transparent batched routing: only when a scheduler is installed AND
        // the model is not a VLM (image/video prefill is not batched yet). When
        // `scheduler == nil` this branch is skipped entirely and behavior is
        // identical to the historical single-stream path below.
        if let scheduler {
            let loadedAsVLM = await context.read { $0.loadedAsVLM }
            if !loadedAsVLM {
                try await attachSchedulerIfNeeded(scheduler)
                let request = GenerationRequest(
                    input: input.consume(),
                    parameters: parameters,
                    wiredMemoryTicket: wiredMemoryTicket,
                    promptCache: promptCache
                )
                return try await scheduler.submit(request)
            }
        }

        // Note: this is only visiting the model exclusively
        // for the pre-fill time.  Beyond that there is no
        // shared mutable state.
        //
        // This means that there may be concurrent access to the
        // model weights themselves (but they are already evaluated).

        return try await context.read { context in
            try MLXLMCommon.generate(
                input: input.consume(),
                parameters: parameters,
                context: context,
                wiredMemoryTicket: wiredMemoryTicket
            )
        }
    }

    /// Bind the scheduler to this container's context on first use (idempotent).
    /// The context is transferred to the scheduler in a `SendableBox` — the
    /// repo's existing non-`Sendable` hand-off box; no new `@unchecked` is
    /// introduced. The scheduler shares the (already-evaluated) model weights.
    private func attachSchedulerIfNeeded(_ scheduler: InferenceScheduler) async {
        if await scheduler.isAttached { return }
        let cache = promptCache
        let box: SendableBox<ModelContext> = await context.read { context in
            // Copy the context struct (shares the evaluated model reference) and
            // box it for transfer to the scheduler actor.
            let copy = context
            return SendableBox(copy)
        }
        await scheduler.attach(context: box, promptCache: cache)
    }

    /// Explicitly batched generation: submit several requests at once and get a
    /// per-request `AsyncStream<Generation>` back, in input order.
    ///
    /// Requests route through the scheduler's normal state machine: the first
    /// starts on the single-stream path and is upgraded into the batch when the
    /// second is routed, with the rest admitted to the engine directly. (Models
    /// whose caches cannot migrate single→batch — SSM/hybrid topologies — run
    /// the requests sequentially instead; admitting fresh batches directly to
    /// the engine for those models is a planned scheduler improvement.)
    /// Requires a scheduler to be installed; throws otherwise.
    ///
    /// Requests are transferred with `sending` (they carry non-`Sendable`
    /// `LMInput`).
    public func generateBatched(
        _ requests: sending [GenerationRequest]
    ) async throws -> [AsyncStream<Generation>] {
        guard let scheduler else {
            throw BatchedGenerationError.schedulerBusy
        }
        guard !requests.isEmpty else {
            throw BatchedGenerationError.batchTooSmall
        }
        // VLM inputs carry image/video tensors the text-only batch prefill does
        // not handle; reject them rather than silently dropping the media.
        let loadedAsVLM = await context.read { $0.loadedAsVLM }
        guard !loadedAsVLM else {
            throw BatchedGenerationError.incompatibleRequests(Array(requests.indices))
        }
        try await attachSchedulerIfNeeded(scheduler)
        // The whole array is transferred to the scheduler actor, which owns it
        // and routes each request internally (no per-element boundary crossing).
        return try await scheduler.submitBatch(requests)
    }

    /// Adjust the runtime batch admission cap on the installed scheduler
    /// (queue-when-full, clamp `< 1 → 1`). No-op if no scheduler is installed.
    public func setMaxBatchSize(_ size: Int?) async {
        await scheduler?.setMaxBatchSize(size)
    }

    /// Decode token IDs to a string.
    ///
    /// - Parameter tokenIds: Array of token IDs
    /// - Returns: Decoded string
    public func decode(tokenIds: [Int]) async -> String {
        let tokenizer = await self.tokenizer
        return tokenizer.decode(tokenIds: tokenIds)
    }

    @available(*, deprecated, renamed: "decode(tokenIds:)")
    public func decode(tokens: [Int]) async -> String {
        await decode(tokenIds: tokens)
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
    @available(*, deprecated, message: "Use applyChatTemplate directly on tokenizer")
    public func applyChatTemplate(messages: [[String: String]]) async throws -> [Int] {
        let tokenizer = await self.tokenizer
        return try tokenizer.applyChatTemplate(messages: messages)
    }
}
