// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Tokens for a batch of prompts, decoded together, with rows joining and leaving at any step.
///
/// The batched form of ``TokenIterator``: port of mlx-vlm's `BatchGenerator`
/// (`mlx_vlm/generate/ar.py`) and mlx-lm's (`mlx_lm/generate.py`), without their prompt
/// scheduling. Prompts queued with ``insert(prompts:maxTokens:caches:samplers:processors:)``
/// are prefilled on the next call to ``next()``, one at a time through the model's own
/// prefill, then their caches are merged into the decoding batch. Every ``next()`` decodes
/// one token for every row; a row leaves when it emits a stop token or reaches its limit, or
/// when it is removed.
///
/// One sampler draws the whole `[B, V]` batch when every row shares one; otherwise each row
/// is sampled on its own, so a row with a seeded sampler draws the tokens it would draw
/// alone. Logit processors run per row.
public struct BatchTokenIterator {

    /// One decoded token of one row.
    public struct Response: Sendable, Equatable {
        public let uid: Int
        public let token: Int
        /// Set on the row's last token: the stop token itself, or the last one within the limit.
        public let finishReason: FinishReason?
    }

    public enum FinishReason: Sendable, Equatable {
        case stop
        case length
    }

    private let model: any LanguageModel
    private let sampler: LogitSampler
    private let processor: LogitProcessor?
    private let stopTokens: Set<Int>
    private let maxTokens: Int
    private let prefill: PrefillParameters

    private struct Row {
        let uid: Int
        var sampler: LogitSampler
        var processor: LogitProcessor?
        let maxTokens: Int
        var generated = 0
    }

    private struct Pending {
        let row: Row
        let prompt: [Int]
        let cache: [KVCache]?
    }

    private var pending: [Pending] = []
    private var uidCount = 0

    /// The rows being decoded.
    private var rows: [Row] = []
    /// Their caches, one batched cache per layer.
    private var cache: [KVCache] = []
    /// The tokens the next step feeds: the last prompt token at first, then the sampled one.
    private var nextTokens: MLXArray?

    /// - Parameters:
    ///   - model: the model
    ///   - sampler: the sampler for rows inserted without one
    ///   - processor: the processor rows inserted without one get a copy of
    ///   - stopTokens: tokens that end a row
    ///   - maxTokens: the token limit for rows inserted without one
    ///   - prefill: how prompts are prefilled
    public init(
        model: any LanguageModel, sampler: LogitSampler = ArgMaxSampler(),
        processor: LogitProcessor? = nil, stopTokens: Set<Int> = [], maxTokens: Int = 128,
        prefill: PrefillParameters = .init()
    ) {
        self.model = model
        self.sampler = sampler
        self.processor = processor
        self.stopTokens = stopTokens
        self.maxTokens = maxTokens
        self.prefill = prefill
    }

    /// The uids of every row decoding or queued, in insertion order.
    public var uids: [Int] { rows.map(\.uid) + pending.map(\.row.uid) }

    /// Queues prompts; they are prefilled on the next ``next()``.
    ///
    /// - Parameters:
    ///   - prompts: the token prompts, each at least one token
    ///   - maxTokens: a token limit per prompt, or the iterator's for all
    ///   - caches: a prompt cache per prompt to continue from, as ``LanguageModel/newCache(parameters:)``
    ///     makes it, or `nil` for a fresh one. The batch keeps them; do not use them elsewhere.
    ///   - samplers: a sampler per prompt, or the iterator's
    ///   - processors: a processor per prompt, or a copy of the iterator's
    /// - Returns: one uid per prompt, in order
    public mutating func insert(
        prompts: [[Int]], maxTokens: [Int]? = nil, caches: [[KVCache]?]? = nil,
        samplers: [LogitSampler?]? = nil, processors: [LogitProcessor?]? = nil
    ) -> [Int] {
        let count = prompts.count
        precondition(maxTokens.map { $0.count == count } ?? true, "one limit per prompt")
        precondition(caches.map { $0.count == count } ?? true, "one cache per prompt")
        precondition(samplers.map { $0.count == count } ?? true, "one sampler per prompt")
        precondition(processors.map { $0.count == count } ?? true, "one processor per prompt")
        var uids: [Int] = []
        for (index, prompt) in prompts.enumerated() {
            precondition(!prompt.isEmpty, "a prompt needs at least one token")
            let limit = maxTokens?[index] ?? self.maxTokens
            precondition(limit > 0, "a row's token limit must be positive")
            var processor = processors?[index] ?? self.processor?.copy()
            processor?.prompt(MLXArray(prompt.map { Int32($0) }))
            let row = Row(
                uid: uidCount, sampler: samplers?[index] ?? sampler, processor: processor,
                maxTokens: limit)
            pending.append(Pending(row: row, prompt: prompt, cache: caches?[index]))
            uids.append(uidCount)
            uidCount += 1
        }
        return uids
    }

    /// Drops rows, queued or decoding.
    public mutating func remove(_ uids: [Int]) {
        let removed = Set(uids)
        pending.removeAll { removed.contains($0.row.uid) }
        let keep = rows.indices.filter { !removed.contains(rows[$0].uid) }
        if keep.count < rows.count {
            filter(keeping: keep)
        }
    }

    /// Prefills the queued prompts and decodes one token for every row.
    ///
    /// - Returns: a token per decoding row, in batch order; empty once nothing is left
    /// - Throws: ``KVCacheError`` when a queued prompt's cache has no batched form
    public mutating func next() throws -> [Response] {
        if !pending.isEmpty {
            try admit()
        }
        guard !rows.isEmpty else { return [] }

        // Drain the forward's autoreleased temporaries every step, as TokenIterator does.
        let tokens = autoreleasepool { step() }
        var responses: [Response] = []
        var keep: [Int] = []
        for (index, token) in tokens.enumerated() {
            rows[index].generated += 1
            let reason: FinishReason? =
                stopTokens.contains(token)
                ? .stop : rows[index].generated >= rows[index].maxTokens ? .length : nil
            responses.append(Response(uid: rows[index].uid, token: token, finishReason: reason))
            if reason == nil {
                keep.append(index)
            }
        }
        if keep.count < rows.count {
            filter(keeping: keep)
        }
        return responses
    }

    /// Prefills the queued prompts as one right-padded batch and joins them to the rows.
    private mutating func admit() throws {
        let admitted = pending
        pending = []

        // Everything but each prompt's last token goes into its own cache first, through the
        // model's prefill (chunked; the cache is evaluated, the logits never are), and the
        // rows merge after. Rows are not prefilled together right-padded: the recurrent
        // models' scans do not honour the prepared lengths yet. The last token is the first
        // decode step's input, as in mlx-lm, so its logits give the first sample.
        let caches = try admitted.map { pending -> [KVCache] in
            let cache = try pending.cache ?? model.newCache(parameters: nil)
            let body = pending.prompt.dropLast().map { Int32($0) }
            guard !body.isEmpty else { return cache }
            let prepared = try model.prepare(
                LMInput(tokens: MLXArray(body)), cache: cache, state: nil, prefill: prefill)
            if case .tokens(let remainder) = prepared {
                _ = model(remainder.tokens[.newAxis], cache: cache)
            }
            eval(cache)
            return cache
        }
        let merged = try caches[0].indices.map { layer in
            try mergeCaches(caches.map { $0[layer] })
        }

        // The new rows take their first step alone: the decoding rows' next tokens are
        // sampled already and must not be fed again.
        var batch = BatchTokenIterator(
            model: model, sampler: sampler, processor: processor, stopTokens: stopTokens,
            maxTokens: maxTokens, prefill: prefill)
        batch.rows = admitted.map(\.row)
        batch.cache = merged
        batch.nextTokens = MLXArray(admitted.map { Int32($0.prompt.last!) })
        _ = batch.step()

        if rows.isEmpty {
            rows = batch.rows
            cache = batch.cache
            nextTokens = batch.nextTokens
        } else {
            for (layer, added) in zip(cache, batch.cache) {
                layer.extend(other: added)
            }
            rows += batch.rows
            nextTokens = concatenated([nextTokens!, batch.nextTokens!])
        }
    }

    /// Feeds the next tokens, samples their successors and returns the fed tokens.
    private mutating func step() -> [Int] {
        let current = nextTokens!
        var logits = model(current[0..., .newAxis], cache: cache)[0..., -1, 0...]
        if rows.contains(where: { $0.processor != nil }) {
            logits = concatenated(
                rows.indices.map { index in
                    let row = logits[index ..< index + 1]
                    return rows[index].processor?.process(logits: row) ?? row
                })
        }
        // Int32 in the graph, so reading the tokens back next step converts nothing.
        let sampled = sample(logits).asType(.int32)
        for index in rows.indices {
            rows[index].processor?.didSample(token: sampled[index ..< index + 1])
        }
        // The cache state settles with the token, as TokenIterator does, so an unevaluated
        // chain of updates never keeps earlier steps alive.
        asyncEval([sampled] + cache.flatMap { $0.state })
        let tokens = current.asArray(Int32.self).map { Int($0) }
        nextTokens = sampled
        return tokens
    }

    /// One batched draw when every row shares a sampler, otherwise one draw per row from
    /// log-probabilities normalized once.
    private func sample(_ logits: MLXArray) -> MLXArray {
        let shared = rows.allSatisfy { $0.sampler.sharesState(with: rows[0].sampler) }
        if shared, rows[0].sampler is ArgMaxSampler {
            return argMax(logits, axis: -1)
        }
        let logProbabilities = logSoftmax(logits.asType(.float32), axis: -1)
        if shared {
            return rows[0].sampler.sample(normalized: logProbabilities)
        }
        return concatenated(
            rows.indices.map { index in
                rows[index].sampler.sample(normalized: logProbabilities[index ..< index + 1])
            })
    }

    private mutating func filter(keeping keep: [Int]) {
        rows = keep.map { rows[$0] }
        guard !keep.isEmpty else {
            cache = []
            nextTokens = nil
            return
        }
        let indices = MLXArray(keep.map { Int32($0) })
        for layer in cache {
            layer.filter(batchIndices: indices)
        }
        nextTokens = nextTokens?[indices]
    }
}

extension LogitSampler {
    /// Whether `other` draws from this sampler's random state, so one call can sample a batch.
    fileprivate func sharesState(with other: LogitSampler) -> Bool {
        switch (self, other) {
        case (is ArgMaxSampler, is ArgMaxSampler):
            return true
        case (let a as TopPSampler, let b as TopPSampler):
            return a.randomState === b.randomState
        case (let a as CategoricalSampler, let b as CategoricalSampler):
            return a.randomState === b.randomState
        default:
            return false
        }
    }

    /// Samples from log-probabilities, without normalizing them again.
    fileprivate func sample(normalized logProbabilities: MLXArray) -> MLXArray {
        if let sampler = self as? TopPSampler {
            return sampler.sample(logProbabilities: logProbabilities)
        }
        return sample(logits: logProbabilities)
    }
}

/// Generates up to `maxTokens` tokens for every prompt as one batch. Port of mlx-lm's
/// `batch_generate`: the thin loop over ``BatchTokenIterator``.
///
/// - Parameters:
///   - model: the model
///   - prompts: the token prompts
///   - caches: a prompt cache per prompt to continue from, or `nil`
///   - maxTokens: the token limit per prompt
///   - stopTokens: tokens that end a prompt's generation; a stop token is not returned
///   - sampler: the sampler for every prompt
///   - processor: the processor every prompt gets a copy of
///   - prefill: how prompts are prefilled
/// - Returns: the generated tokens per prompt, in order
public func batchGenerate(
    model: any LanguageModel, prompts: [[Int]], caches: [[KVCache]?]? = nil,
    maxTokens: Int = 128, stopTokens: Set<Int> = [], sampler: LogitSampler = ArgMaxSampler(),
    processor: LogitProcessor? = nil, prefill: PrefillParameters = .init()
) throws -> [[Int]] {
    var iterator = BatchTokenIterator(
        model: model, sampler: sampler, processor: processor, stopTokens: stopTokens,
        maxTokens: maxTokens, prefill: prefill)
    let uids = iterator.insert(prompts: prompts, caches: caches)
    var results = [Int: [Int]](uniqueKeysWithValues: uids.map { ($0, []) })
    while true {
        let responses = try iterator.next()
        if responses.isEmpty { break }
        for response in responses where response.finishReason != .stop {
            results[response.uid, default: []].append(response.token)
        }
    }
    return uids.map { results[$0] ?? [] }
}
