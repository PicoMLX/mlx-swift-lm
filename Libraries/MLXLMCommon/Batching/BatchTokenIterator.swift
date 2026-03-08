// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN
import Tokenizers

/// Result of a single batch decode step for one sequence.
public struct BatchStepResult: Sendable {
    /// Unique sequence ID within the scheduler.
    public let uid: Int
    /// The newly generated token.
    public let token: Int
    /// Whether this sequence has finished generating.
    public let isDone: Bool
}

/// Manages batched prefill and decode for multiple concurrent sequences.
///
/// `BatchTokenIterator` owns the ``BatchKVCache`` for all active sequences and
/// drives the batch forward pass each decode step.
///
/// ## Lifecycle
///
/// 1. Create with a set of already-prefilled sequences via ``init(sequences:model:context:)``.
/// 2. Call ``step()`` repeatedly to decode one token per sequence.
/// 3. When sequences complete, call ``removeCompleted(_:)`` to shrink the batch.
/// 4. When the batch is empty, discard the iterator.
///
/// ## Thread Safety
///
/// This type is **not** thread-safe. All calls must come from the same actor
/// (``InferenceScheduler``). MLX graph construction and evaluation are serialized
/// through ``DeviceEngine``.
final class BatchTokenIterator {

    // MARK: - State

    /// Model to run inference on.
    private let model: any LanguageModel

    /// Per-layer batch KV caches (one ``BatchKVCache`` per transformer layer).
    private var batchCache: [BatchKVCache]

    /// Current token for each sequence: shape `[B]`.
    private var tokens: [Int]

    /// The active sequences, in the same order as `tokens` and `batchCache`.
    private var sequences: [SequenceState]

    /// Number of active sequences.
    var batchSize: Int { sequences.count }

    // MARK: - Init

    /// Create a `BatchTokenIterator` from sequences that have already been prefilled.
    ///
    /// This is the primary entry point. Each sequence must have:
    /// - A completed prefill (KV cache populated)
    /// - A `currentToken` set to the first decode token (sampled from the prefill logits)
    ///
    /// Sequences with different prompt lengths are automatically left-padded so all
    /// share the same cache offset.
    ///
    /// - Parameters:
    ///   - sequences: Active sequences (already prefilled, with `currentToken` set).
    ///   - perSequenceCaches: `caches[seqIdx]` = per-layer `[KVCacheSimple]` for that sequence.
    ///   - model: The language model.
    init(
        sequences: [SequenceState],
        perSequenceCaches: [[KVCacheSimple]],
        model: any LanguageModel
    ) {
        self.model = model
        self.sequences = sequences
        self.tokens = sequences.map { $0.currentToken }
        self.batchCache = BatchKVCache.fromSingle(perSequenceCaches: perSequenceCaches)
    }

    // MARK: - Prefill

    /// Prefill a single sequence and return its first decode token.
    ///
    /// This runs the full prompt through the model using the standard chunked
    /// prefill path, populating the per-layer ``KVCacheSimple`` caches.
    ///
    /// - Parameters:
    ///   - promptTokens: Raw token IDs for the prompt.
    ///   - model: The language model.
    ///   - parameters: Generation parameters (controls prefill step size).
    /// - Returns: `(caches, firstToken)` where `caches` is the populated per-layer cache
    ///   and `firstToken` is the sampled first decode token.
    static func prefillSingle(
        promptTokens: [Int],
        model: any LanguageModel,
        parameters: GenerateParameters
    ) throws -> (caches: [KVCacheSimple], firstToken: Int) {
        let cache: [KVCache] = model.newCache(parameters: nil).map { _ in KVCacheSimple() }
        let simpleCaches = cache.compactMap { $0 as? KVCacheSimple }

        let input = LMInput(tokens: MLXArray(promptTokens.map { Int32($0) }))

        let logits: MLXArray
        switch try model.prepare(input, cache: cache, windowSize: parameters.prefillStepSize) {
        case .tokens(let remaining):
            let out = model(remaining[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: nil)
            eval(out.logits)
            logits = out.logits
        case .logits(let out):
            eval(out.logits)
            logits = out.logits
        }

        // Sample the first decode token from the last position
        let lastLogits = logits[0..., -1, 0...]  // [1, vocab_size] or [vocab_size]
        let sampler = parameters.sampler()
        let tokenArray = sampler.sample(logits: lastLogits.squeezed())
        eval(tokenArray)
        let firstToken = tokenArray.item(Int.self)

        return (simpleCaches, firstToken)
    }

    // MARK: - Decode Step

    /// Run one batch decode step and return per-sequence results.
    ///
    /// This method:
    /// 1. Stacks the current tokens into `[B, 1]`
    /// 2. Runs the model forward pass
    /// 3. Samples one token per sequence
    /// 4. Updates each ``SequenceState``
    ///
    /// - Returns: One ``BatchStepResult`` per active sequence.
    @discardableResult
    func step() -> [BatchStepResult] {
        guard !sequences.isEmpty else { return [] }

        let B = sequences.count

        // Build [B, 1] input tokens
        let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped(B, 1)

        // Run batch forward pass through the model
        // batchCache is [BatchKVCache], satisfying [KVCache] protocol
        let logits = model(tokenArray, cache: batchCache.isEmpty ? nil : batchCache)
        // logits: [B, 1, vocab_size]

        asyncEval(logits)

        // Sample one token per sequence
        var results: [BatchStepResult] = []
        var newTokens: [Int] = []

        for (i, seq) in sequences.enumerated() {
            // Extract logits for this sequence: [vocab_size]
            let seqLogits = logits[i, 0, 0...]  // [vocab_size]

            // Apply processor if any
            var processedLogits = seqLogits
            if let proc = seq.processor {
                processedLogits = proc.process(logits: seqLogits)
            }

            // Sample
            let tokenArr = seq.sampler.sample(logits: processedLogits)
            eval(tokenArr)
            let newToken = tokenArr.item(Int.self)

            // Update processor
            if var proc = seq.processor {
                proc.didSample(token: tokenArr)
                seq.processor = proc
            }

            // Update sequence state
            seq.currentToken = newToken
            seq.numTokens += 1
            seq.generatedTokens.append(newToken)
            seq.markDecodeStart()

            newTokens.append(newToken)
            results.append(BatchStepResult(uid: seq.uid, token: newToken, isDone: seq.isDone))
        }

        tokens = newTokens
        return results
    }

    // MARK: - Batch Management

    /// Remove sequences at the given indices and compact the batch.
    ///
    /// Call this after identifying finished sequences from ``step()`` results.
    ///
    /// - Parameter finishedIndices: Sorted list of indices to remove.
    func removeSequences(at finishedIndices: [Int]) {
        guard !finishedIndices.isEmpty else { return }

        let finishedSet = Set(finishedIndices)
        let keepIndices = (0 ..< sequences.count).filter { !finishedSet.contains($0) }

        sequences = keepIndices.map { sequences[$0] }
        tokens = keepIndices.map { tokens[$0] }

        // Compact the batch cache for each layer
        batchCache = batchCache.map { $0.filter(keepIndices: keepIndices) }
    }

    /// Add a new sequence to the batch that has already been prefilled.
    ///
    /// The new sequence's cache is merged into the existing batch via left-padding.
    ///
    /// - Parameters:
    ///   - sequence: The new sequence (already prefilled, `currentToken` set).
    ///   - perLayerCache: The prefilled per-layer caches for the new sequence.
    func addSequence(_ sequence: SequenceState, perLayerCache: [KVCacheSimple]) {
        let existingB = sequences.count
        let newB = existingB + 1

        // Compute padding: new sequence may be shorter than existing batch offset
        let batchOffset = batchCache.first?.offset ?? 0
        let newSeqOffset = perLayerCache.first?.offset ?? 0
        let newPadding = max(0, batchOffset - newSeqOffset)

        // Merge caches layer by layer
        var mergedCaches: [BatchKVCache] = []
        for (layerIdx, existingLayer) in batchCache.enumerated() {
            let newLayerCache = perLayerCache[layerIdx]

            guard let existingK = existingLayer.batchKeys,
                  let existingV = existingLayer.batchValues
            else {
                // Existing cache is empty — shouldn't happen in practice
                mergedCaches.append(existingLayer)
                continue
            }

            let maxOffset = max(batchOffset, newSeqOffset + newPadding)

            // Pad new sequence's cache on the left
            var newK: MLXArray
            var newV: MLXArray
            let newState = newLayerCache.state
            if newState.count >= 2 {
                newK = newState[0]  // [1, nHeads, newSeqOffset, headDim]
                newV = newState[1]
                if newPadding > 0 {
                    let kPad = MLXArray.zeros([1, newK.dim(1), newPadding, newK.dim(3)], dtype: newK.dtype)
                    let vPad = MLXArray.zeros([1, newV.dim(1), newPadding, newV.dim(3)], dtype: newV.dtype)
                    newK = concatenated([kPad, newK], axis: 2)
                    newV = concatenated([vPad, newV], axis: 2)
                }
            } else {
                // New sequence has empty cache (shouldn't happen)
                newK = MLXArray.zeros([1, existingK.dim(1), maxOffset, existingK.dim(3)], dtype: existingK.dtype)
                newV = MLXArray.zeros([1, existingV.dim(1), maxOffset, existingV.dim(3)], dtype: existingV.dtype)
            }

            // Pad existing sequences if the new one is longer (shouldn't happen often)
            var paddedExistingK = existingK
            var paddedExistingV = existingV
            if maxOffset > batchOffset {
                let extraPad = maxOffset - batchOffset
                let ep = MLXArray.zeros([existingB, existingK.dim(1), extraPad, existingK.dim(3)], dtype: existingK.dtype)
                let epV = MLXArray.zeros([existingB, existingV.dim(1), extraPad, existingV.dim(3)], dtype: existingV.dtype)
                paddedExistingK = concatenated([ep, paddedExistingK], axis: 2)
                paddedExistingV = concatenated([epV, paddedExistingV], axis: 2)
            }

            // Stack existing + new
            let mergedK = concatenated([paddedExistingK, newK], axis: 0)  // [B+1, ...]
            let mergedV = concatenated([paddedExistingV, newV], axis: 0)

            // Compute updated padding array
            let existingPadding = existingLayer.leftPadding
            let allPadding = existingPadding + [newPadding]

            let merged = BatchKVCache(batchSize: newB, leftPadding: allPadding)
            merged.batchKeys = mergedK
            merged.batchValues = mergedV
            merged.offset = maxOffset
            mergedCaches.append(merged)
        }

        batchCache = mergedCaches
        sequences.append(sequence)
        tokens.append(sequence.currentToken)
    }

    /// Current sequence states (for external inspection).
    var activeSequences: [SequenceState] { sequences }
}
