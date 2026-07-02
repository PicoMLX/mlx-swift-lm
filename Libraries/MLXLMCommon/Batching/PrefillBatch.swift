// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Prefill-phase batch over a shared `[any BatchedCache]`.
///
/// Right-pads ragged prompts to a uniform `[B, max_length]` tensor, runs
/// chunked prefill through the model, then rolls each row into right-aligned
/// position via per-cache finalization. Transition into a ``DecodeBatch``
/// via `generate(lastTokensOf:)`.
///
/// This is mutable engine state owned by ``BatchGenerationEngine`` and is
/// intentionally **not** `Sendable`. Do not share a prefill batch across
/// concurrent tasks.
public final class PrefillBatch {

    public let model: any LanguageModel
    public private(set) var uids: [Int]
    public private(set) var promptCache: [any BatchedCache]
    public private(set) var tokens: [[Int]]
    public private(set) var maxTokens: [Int]

    public let prefillStepSize: Int

    public private(set) var samplers: [RowSampler?]
    public let fallbackSampler: RowSampler

    /// Whether `fallbackSampler` is the deterministic ``greedySampler``.
    /// Forwarded to the ``DecodeBatch`` built in `generate(lastTokensOf:)` so
    /// it can gate its greedy `argMax` fast path correctly.
    public let fallbackIsGreedy: Bool

    /// Per-row ``LogitProcessor`` (penalties), forwarded to the
    /// ``DecodeBatch`` on `generate(lastTokensOf:)`. `nil` for a row with no
    /// penalties. Built (non-`Sendable`) processors held directly because
    /// `PrefillBatch` is actor-confined and never crosses an actor boundary.
    public private(set) var processors: [LogitProcessor?]
    public private(set) var stateMachines: [StopSequenceMatcher]

    public init(
        model: any LanguageModel,
        uids: [Int],
        promptCache: [any BatchedCache],
        tokens: [[Int]],
        maxTokens: [Int],
        prefillStepSize: Int = 2048,
        samplers: [RowSampler?]? = nil,
        fallbackSampler: @escaping RowSampler = greedySampler,
        fallbackIsGreedy: Bool = true,
        processors: [LogitProcessor?]? = nil,
        stateMachines: [StopSequenceMatcher]? = nil
    ) {
        precondition(uids.count == tokens.count, "uids/tokens count mismatch")
        precondition(uids.count == maxTokens.count, "uids/max_tokens count mismatch")
        if let samplers {
            precondition(uids.count == samplers.count, "uids/samplers count mismatch")
        }
        if let processors {
            precondition(uids.count == processors.count, "uids/processors count mismatch")
        }
        if let stateMachines {
            precondition(uids.count == stateMachines.count, "uids/stateMachines count mismatch")
        }
        self.model = model
        self.uids = uids
        self.promptCache = promptCache
        self.tokens = tokens
        self.maxTokens = maxTokens
        self.prefillStepSize = prefillStepSize
        self.samplers = samplers ?? Array(repeating: nil, count: uids.count)
        self.fallbackSampler = fallbackSampler
        self.fallbackIsGreedy = fallbackIsGreedy
        self.processors = processors ?? Array(repeating: nil, count: uids.count)
        self.stateMachines =
            stateMachines
            ?? Array(
                repeating: StopSequenceMatcher(),
                count: uids.count
            )
    }

    public static func empty(
        model: any LanguageModel,
        prefillStepSize: Int = 2048,
        fallbackSampler: @escaping RowSampler = greedySampler,
        fallbackIsGreedy: Bool = true
    ) -> PrefillBatch {
        PrefillBatch(
            model: model,
            uids: [],
            promptCache: [],
            tokens: [],
            maxTokens: [],
            prefillStepSize: prefillStepSize,
            samplers: [],
            fallbackSampler: fallbackSampler,
            fallbackIsGreedy: fallbackIsGreedy,
            processors: [],
            stateMachines: []
        )
    }

    public var batchSize: Int { uids.count }
    public var isEmpty: Bool { uids.isEmpty }

    /// Run the model over `promptTokens` to populate the cache. Ragged
    /// rows are right-padded to `max(lengths)`; the BatchKVCache records
    /// the right-padding for `finalize()` to roll out afterwards.
    public func prompt(_ promptTokens: [[Int]]) {
        precondition(
            promptTokens.count == uids.count,
            "PrefillBatch.prompt: token list length \(promptTokens.count) "
                + "does not match batch size \(uids.count)"
        )
        if promptTokens.isEmpty { return }

        for (i, t) in promptTokens.enumerated() {
            tokens[i].append(contentsOf: t)
        }

        let lengths = promptTokens.map { $0.count }
        let maxLength = lengths.max() ?? 0
        let padding = lengths.map { maxLength - $0 }
        let maxPadding = padding.max() ?? 0

        var inputs: MLXArray
        if maxPadding > 0 {
            var padded: [[Int]] = []
            padded.reserveCapacity(promptTokens.count)
            for t in promptTokens {
                padded.append(t + Array(repeating: 0, count: maxLength - t.count))
            }
            let flat: [UInt32] = padded.flatMap { $0.map { UInt32($0) } }
            inputs = MLXArray(flat).reshaped([promptTokens.count, maxLength])

            for cache in promptCache {
                cache.prepareBatched(leftPadding: nil, lengths: lengths, rightPadding: padding)
            }
        } else {
            let flat: [UInt32] = promptTokens.flatMap { $0.map { UInt32($0) } }
            inputs = MLXArray(flat).reshaped([promptTokens.count, maxLength])
        }

        var remaining = inputs
        while remaining.dim(1) > 0 {
            let n = min(prefillStepSize, remaining.dim(1))
            let chunk = remaining[0..., ..<n]
            _ = model.callAsFunction(
                chunk,
                cache: promptCache.map { $0 as any KVCache }
            )
            // One combined eval per chunk (not per cache): each cache's
            // state is independent, and a single eval turns L GPU sync
            // points into 1 -- mirroring the single path's `eval(cache)`
            // over the whole layer array.
            eval(promptCache.flatMap { $0.innerState() })
            for cache in promptCache {
                cache.advanceBatched(n)
            }
            if n == remaining.dim(1) {
                break
            }
            remaining = remaining[0..., n...]
        }

        if maxPadding > 0 {
            // Queue every cache's (lazy) roll first, then force them all
            // with one combined eval instead of one sync per cache.
            for cache in promptCache {
                cache.finalizeBatched()
            }
            eval(promptCache.flatMap { $0.innerState() })
        }
    }

    /// Move from prefill into decode. The last token of each row's prompt
    /// becomes the seed input for the ``DecodeBatch``; any prefix of length
    /// > 1 is run through `prompt(...)` first.
    ///
    /// Ownership of the cache and per-row state transfers to the returned
    /// ``DecodeBatch``.
    public func generate(lastTokensOf inputTokens: [[Int]]) -> DecodeBatch {
        precondition(
            inputTokens.count == uids.count,
            "PrefillBatch.generate: token list length \(inputTokens.count) "
                + "does not match batch size \(uids.count)"
        )
        precondition(
            !inputTokens.contains(where: { $0.isEmpty }),
            "PrefillBatch.generate requires non-empty token rows; "
                + "BatchGenerationEngine.insert validates this public invariant"
        )
        if inputTokens.contains(where: { $0.count > 1 }) {
            let prefixes = inputTokens.map { Array($0.dropLast()) }
            prompt(prefixes)
        }

        let lastTokens = inputTokens.map { UInt32($0.last!) }
        let seed = MLXArray(lastTokens)

        let gen = DecodeBatch(
            model: model,
            uids: uids,
            seedTokens: seed,
            promptCache: promptCache,
            tokens: tokens,
            maxTokens: maxTokens,
            samplers: samplers,
            fallbackSampler: fallbackSampler,
            fallbackIsGreedy: fallbackIsGreedy,
            processors: processors,
            stateMachines: stateMachines
        )

        uids = []
        promptCache = []
        tokens = []
        samplers = []
        processors = []
        maxTokens = []
        stateMachines = []

        return gen
    }

    public func filter(keep: [Int]) {
        let keepArr = keep.isEmpty ? nil : MLXArray(keep.map { Int32($0) })

        if let keepArr {
            for cache in promptCache {
                cache.filterBatched(batchIndices: keepArr)
            }
        } else {
            promptCache.removeAll()
        }

        uids = keep.map { uids[$0] }
        tokens = keep.map { tokens[$0] }
        samplers = keep.map { samplers[$0] }
        processors = keep.map { processors[$0] }
        maxTokens = keep.map { maxTokens[$0] }
        stateMachines = keep.map { stateMachines[$0] }
    }

    /// Append `other`'s rows. Currently restricted to merging two batches
    /// before either has run prefill (i.e. both caches empty); merging
    /// non-empty prefill caches is not yet implemented.
    public func extend(_ other: PrefillBatch) {
        precondition(
            uids.isEmpty && other.uids.isEmpty,
            "PrefillBatch.extend currently only supports merging two empty batches"
        )
        uids.append(contentsOf: other.uids)
        tokens.append(contentsOf: other.tokens)
        samplers.append(contentsOf: other.samplers)
        processors.append(contentsOf: other.processors)
        maxTokens.append(contentsOf: other.maxTokens)
        stateMachines.append(contentsOf: other.stateMachines)
    }
}
