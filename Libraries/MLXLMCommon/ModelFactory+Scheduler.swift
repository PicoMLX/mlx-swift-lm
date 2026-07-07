// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - Scheduler-aware loading (the batching "front door")
//
// Convenience overloads that load a model and wrap it in a ``ModelContainer``
// with an ``InferenceScheduler`` (and optional cross-request prompt cache)
// installed — the opt-in for continuous batching:
//
// ```swift
// let downloader: any Downloader = ...           // e.g. a Hub downloader
// let tokenizerLoader: any TokenizerLoader = ... // e.g. a Hub tokenizer loader
// let container = try await loadModelContainer(
//     from: downloader,
//     using: tokenizerLoader,
//     configuration: .init(id: "mlx-community/Qwen3-4B-4bit"),
//     scheduler: InferenceScheduler(),
//     promptCache: LRUPromptCache(maxBytes: 2 << 30)
// )
// ```
//
// Design notes:
// - `scheduler` has NO default value, so these overloads never compete with
//   the plain loaders during overload resolution: existing call sites keep
//   resolving to the original functions and keep today's scheduler-free,
//   zero-overhead container.
// - Implemented purely by composition — load a ``ModelContext`` through the
//   existing (trampoline-dispatched) paths, then wrap it with the
//   scheduler-aware ``ModelContainer`` initializer. No existing declaration
//   changes, and the `ModelFactory` protocol surface is untouched.
// - A scheduler on a VLM container is inert: ``ModelContainer/generate``
//   routes VLM contexts to the single-stream path and `generateBatched`
//   rejects them, so passing one is harmless (it simply never engages).

/// Load a model from a ``Downloader`` and ``ModelConfiguration``, producing a
/// ``ModelContainer`` with the given ``InferenceScheduler`` installed.
///
/// The scheduler enables transparent continuous batching: the first request
/// runs on the zero-overhead single-stream path and concurrent requests are
/// upgraded into a shared batch. See ``InferenceScheduler`` for details.
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - configuration: the model configuration to resolve and load
///   - useLatest: when true, always checks the provider for the latest version
///   - scheduler: the scheduler to install (opts this container into batching)
///   - promptCache: optional cross-request prompt cache shared by the
///     scheduler's single and batched paths
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContainer`` that routes generation through `scheduler`
public func loadModelContainer(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: ModelConfiguration,
    useLatest: Bool = false,
    scheduler: InferenceScheduler,
    promptCache: (any PromptCaching)? = nil,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    let context = try await loadModel(
        from: downloader, using: tokenizerLoader, configuration: configuration,
        useLatest: useLatest, progressHandler: progressHandler)
    return ModelContainer(context: context, scheduler: scheduler, promptCache: promptCache)
}

/// Load a model given a model identifier, producing a ``ModelContainer`` with
/// the given ``InferenceScheduler`` installed.
///
/// See ``loadModelContainer(from:using:configuration:useLatest:scheduler:promptCache:progressHandler:)``.
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - id: model identifier, e.g "mlx-community/Qwen3-4B-4bit"
///   - revision: revision to download (defaults to "main")
///   - useLatest: when true, always checks the provider for the latest version
///   - scheduler: the scheduler to install (opts this container into batching)
///   - promptCache: optional cross-request prompt cache
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContainer`` that routes generation through `scheduler`
public func loadModelContainer(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    id: String,
    revision: String = "main",
    useLatest: Bool = false,
    scheduler: InferenceScheduler,
    promptCache: (any PromptCaching)? = nil,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    let context = try await loadModel(
        from: downloader, using: tokenizerLoader, id: id, revision: revision,
        useLatest: useLatest, progressHandler: progressHandler)
    return ModelContainer(context: context, scheduler: scheduler, promptCache: promptCache)
}

/// Load a model from a local directory, producing a ``ModelContainer`` with
/// the given ``InferenceScheduler`` installed.
///
/// See ``loadModelContainer(from:using:configuration:useLatest:scheduler:promptCache:progressHandler:)``.
///
/// - Parameters:
///   - directory: directory of configuration and weights
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - scheduler: the scheduler to install (opts this container into batching)
///   - promptCache: optional cross-request prompt cache
/// - Returns: a ``ModelContainer`` that routes generation through `scheduler`
public func loadModelContainer(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader,
    scheduler: InferenceScheduler,
    promptCache: (any PromptCaching)? = nil
) async throws -> sending ModelContainer {
    let context = try await loadModel(from: directory, using: tokenizerLoader)
    return ModelContainer(context: context, scheduler: scheduler, promptCache: promptCache)
}

extension GenericModelFactory where ContextType == ModelContext, ContainerType == ModelContainer {

    /// Load a model from a ``Downloader`` and ``ModelConfiguration``,
    /// producing a ``ModelContainer`` with the given ``InferenceScheduler``
    /// installed (the opt-in for continuous batching).
    public func loadContainer(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration,
        useLatest: Bool = false,
        scheduler: InferenceScheduler,
        promptCache: (any PromptCaching)? = nil,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        let context = try await load(
            from: downloader, using: tokenizerLoader, configuration: configuration,
            useLatest: useLatest, progressHandler: progressHandler)
        return ModelContainer(context: context, scheduler: scheduler, promptCache: promptCache)
    }

    /// Load a model from a local directory, producing a ``ModelContainer``
    /// with the given ``InferenceScheduler`` installed (the opt-in for
    /// continuous batching).
    public func loadContainer(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader,
        scheduler: InferenceScheduler,
        promptCache: (any PromptCaching)? = nil
    ) async throws -> ModelContainer {
        let context = try await load(from: directory, using: tokenizerLoader)
        return ModelContainer(context: context, scheduler: scheduler, promptCache: promptCache)
    }
}
