// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Batched KV cache for multi-sequence inference.
///
/// Stores keys and values for all sequences in a batch in a single contiguous
/// `[B, nHeads, seqLen, headDim]` tensor. Sequences with shorter prompts are
/// left-padded with zeros so all sequences share the same `offset`.
///
/// ## Design Notes
///
/// - **Left-padding**: Shorter sequences are padded on the left so that all
///   sequences in the batch are aligned at the right (most-recent token).
///   This means `cache.offset` is the same for all sequences.
///
/// - **RoPE position approximation**: During batch decode, `cache.offset`
///   (an `Int`) is used by models for all sequences. For sequences merged via
///   the single-to-batch upgrade path (``fromSingle(_:leftPadding:)``), shorter
///   sequences will have their decode-step RoPE position overestimated by
///   `leftPadding[i]` positions. For simultaneous batch (all sequences prefilled
///   together with the same padded length), RoPE positions are exact.
///
/// - **Attention mask**: The mask returned by ``makeMask(n:windowSize:returnArray:)``
///   accounts for left-padding. However, many models use the deprecated
///   `createAttentionMask(h:cache:)` helper which returns `nil` for single-token
///   decode steps. In that case, padding positions are attended to (mild quality
///   degradation). This will be fully resolved once models adopt the new
///   `createAttentionMask(h:cache:windowSize:returnArray:)` API.
///
/// ## Usage
///
/// ```swift
/// // Create from simultaneously-prefilled single-sequence caches
/// let batchCaches = BatchKVCache.fromSingle(
///     perSequenceCaches: [[cacheA_layer0, cacheA_layer1], [cacheB_layer0, cacheB_layer1]],
///     leftPadding: [0, 20]   // B already padded 20 tokens on the left
/// )
/// ```
public class BatchKVCache: BaseKVCache {

    // MARK: - Storage

    /// Batched keys: `[B, nHeads, seqLen, headDim]`
    internal var batchKeys: MLXArray?

    /// Batched values: `[B, nHeads, seqLen, headDim]`
    internal var batchValues: MLXArray?

    /// Number of sequences in this batch.
    public let batchSize: Int

    /// Per-sequence left-padding amounts.
    ///
    /// `leftPadding[i]` is the number of zero-value padding tokens prepended to
    /// the start of sequence `i`'s key/value arrays.
    public let leftPadding: [Int]

    /// Allocation step size (keys/values grow in multiples of this).
    public var step: Int = 256

    // MARK: - Init

    /// Create an empty BatchKVCache for `batchSize` sequences.
    ///
    /// - Parameters:
    ///   - batchSize: Number of sequences in the batch.
    ///   - leftPadding: Per-sequence left-padding lengths. Must have exactly `batchSize` elements.
    public init(batchSize: Int, leftPadding: [Int]) {
        precondition(leftPadding.count == batchSize,
                     "leftPadding must have exactly batchSize elements")
        self.batchSize = batchSize
        self.leftPadding = leftPadding
        super.init()
    }

    // MARK: - KVCache Protocol

    public override func innerState() -> [MLXArray] {
        [batchKeys, batchValues].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset
        let incoming = keys.dim(2)  // number of new tokens

        // Grow allocation if needed
        let needsResize = batchKeys == nil || (previous + incoming) > batchKeys!.dim(2)
        if needsResize {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + incoming - 1) / step
            let newK = MLXArray.zeros([B, kvHeads, nSteps * step, kHeadDim], dtype: keys.dtype)
            let newV = MLXArray.zeros([B, kvHeads, nSteps * step, vHeadDim], dtype: values.dtype)

            if var cur = batchKeys, var curV = batchValues {
                if previous % step != 0 {
                    cur = cur[.ellipsis, ..<previous, 0...]
                    curV = curV[.ellipsis, ..<previous, 0...]
                }
                batchKeys = concatenated([cur, newK], axis: 2)
                batchValues = concatenated([curV, newV], axis: 2)
            } else {
                batchKeys = newK
                batchValues = newV
            }
        }

        self.offset += incoming

        batchKeys?[.ellipsis, previous..<self.offset, 0...] = keys
        batchValues?[.ellipsis, previous..<self.offset, 0...] = values

        let retK = batchKeys![.ellipsis, ..<self.offset, 0...]
        let retV = batchValues![.ellipsis, ..<self.offset, 0...]
        return (retK, retV)
    }

    /// Create a batch-aware attention mask that excludes left-padded positions.
    ///
    /// Returns a `[B, 1, n, offset+n]` additive float mask where:
    /// - `-1e9` = masked out (causal violation OR left-padding)
    /// - `0.0`  = attend normally
    ///
    /// For decode (n=1) with no padding this returns `.none`.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let total = offset + n

        // Fast path: single decode token with no left-padding
        if n == 1 && leftPadding.allSatisfy({ $0 == 0 }) {
            return .none
        }

        // Causal part: [1, n, total]
        // Query positions run from `offset` to `offset+n-1`.
        let queryPositions = (offset != 0
            ? MLXArray(Int32(offset) ..< Int32(offset + n))
            : MLXArray(Int32(0) ..< Int32(n))
        ).reshaped(1, -1, 1)  // [1, n, 1]

        let keyPositions = MLXArray(Int32(0) ..< Int32(total)).reshaped(1, 1, -1)  // [1, 1, total]

        // causal[0, i, j] = (query_pos[i] >= key_pos[j])
        var causal = queryPositions .>= keyPositions  // [1, n, total]

        // Optional window constraint
        if let windowSize {
            let windowed = queryPositions .< keyPositions + Int32(windowSize)
            causal = causal & windowed
        }

        // Per-sequence left-padding mask: key_pos[j] >= leftPadding[b]
        if leftPadding.contains(where: { $0 > 0 }) {
            let padArr = MLXArray(leftPadding.map { Int32($0) }).reshaped(-1, 1, 1)  // [B, 1, 1]
            let noPad = keyPositions .>= padArr  // [B, 1, total]
            causal = causal & noPad  // broadcasts to [B, n, total]
        }

        // Convert bool to additive float mask and add head dim → [B, 1, n, total]
        let floatMask = causal.asType(.float32)
        let additive = (1.0 - floatMask) * Float(-1e9)
        return .array(additive.reshaped(batchSize, 1, n, total))
    }

    public override var state: [MLXArray] {
        get {
            guard let k = batchKeys, let v = batchValues else { return [] }
            if offset == k.dim(2) {
                return [k, v]
            }
            return [k[.ellipsis, ..<offset, 0...], v[.ellipsis, ..<offset, 0...]]
        }
        set {
            guard newValue.count == 2 else {
                fatalError("BatchKVCache state must have exactly 2 arrays (keys, values)")
            }
            batchKeys = newValue[0]
            batchValues = newValue[1]
            self.offset = batchKeys!.dim(2)
        }
    }

    public override var metaState: [String] {
        get { ["batch:\(batchSize)"] }
        set {}
    }

    public override var isTrimmable: Bool { false }

    // MARK: - Factory

    /// Create a per-layer array of ``BatchKVCache`` from independent single-sequence caches.
    ///
    /// This is the primary entry point for merging individually prefilled sequences into
    /// a batch for the subsequent decode loop.
    ///
    /// Shorter sequences are left-padded with zeros to match the longest sequence.
    /// The returned `leftPadding` values can be used to create the appropriate attention
    /// mask during decode.
    ///
    /// - Parameters:
    ///   - perSequenceCaches: `caches[seqIdx][layerIdx]` — one inner `[KVCache]` per sequence.
    ///   - leftPadding: Per-sequence left-padding amounts. If `nil`, computed automatically
    ///     so all sequences have the same length.
    /// - Returns: `[BatchKVCache]` indexed by layer (replaces a single sequence's `[KVCache]`).
    public static func fromSingle(
        perSequenceCaches: [[KVCacheSimple]],
        leftPadding: [Int]? = nil
    ) -> [BatchKVCache] {
        let batchSize = perSequenceCaches.count
        guard batchSize > 0 else { return [] }

        let nLayers = perSequenceCaches[0].count
        let offsets = perSequenceCaches.map { $0.first?.offset ?? 0 }
        let maxOffset = offsets.max() ?? 0

        // Compute left-padding for each sequence so all reach maxOffset
        let padding = leftPadding ?? offsets.map { maxOffset - $0 }

        return (0 ..< nLayers).map { layerIdx in
            let layerCaches = perSequenceCaches.map { $0[layerIdx] }
            return makeBatchLayer(
                layerCaches: layerCaches,
                padding: padding,
                maxOffset: maxOffset,
                batchSize: batchSize
            )
        }
    }

    // MARK: - Helpers

    private static func makeBatchLayer(
        layerCaches: [KVCacheSimple],
        padding: [Int],
        maxOffset: Int,
        batchSize: Int
    ) -> BatchKVCache {
        let batchCache = BatchKVCache(batchSize: batchSize, leftPadding: padding)
        batchCache.offset = maxOffset

        guard maxOffset > 0 else { return batchCache }

        // Stack keys and values for each sequence, with left-padding
        var keysList: [MLXArray] = []
        var valuesList: [MLXArray] = []

        for (cache, pad) in zip(layerCaches, padding) {
            let state = cache.state
            guard state.count >= 2 else {
                // Empty cache: create all-zeros for this sequence
                // We'll fill in shape below once we know the dims
                keysList.append(MLXArray.zeros([1, 1, maxOffset, 1]))
                valuesList.append(MLXArray.zeros([1, 1, maxOffset, 1]))
                continue
            }

            var k = state[0]  // [1, nHeads, seqLen, headDim]
            var v = state[1]  // [1, nHeads, seqLen, headDim]

            if pad > 0 {
                let kPad = MLXArray.zeros([1, k.dim(1), pad, k.dim(3)], dtype: k.dtype)
                let vPad = MLXArray.zeros([1, v.dim(1), pad, v.dim(3)], dtype: v.dtype)
                k = concatenated([kPad, k], axis: 2)
                v = concatenated([vPad, v], axis: 2)
            }

            keysList.append(k)
            valuesList.append(v)
        }

        // Fix any placeholder zeros using the shape from real caches
        let realIdx = keysList.firstIndex(where: { $0.dim(1) > 1 }) ?? 0
        if realIdx < keysList.count {
            let refK = keysList[realIdx]
            let refV = valuesList[realIdx]
            for i in 0 ..< keysList.count {
                if keysList[i].dim(1) == 1 && refK.dim(1) > 1 {
                    keysList[i] = MLXArray.zeros(
                        [1, refK.dim(1), maxOffset, refK.dim(3)], dtype: refK.dtype)
                    valuesList[i] = MLXArray.zeros(
                        [1, refV.dim(1), maxOffset, refV.dim(3)], dtype: refV.dtype)
                }
            }
        }

        batchCache.batchKeys = concatenated(keysList, axis: 0)    // [B, nHeads, maxOffset, headDim]
        batchCache.batchValues = concatenated(valuesList, axis: 0)

        return batchCache
    }

    /// Remove sequences at specific indices, returning a new cache with only the kept sequences.
    ///
    /// Use this when sequences complete and you want to continue decoding the remainder.
    /// If only one sequence remains, consider using ``extract(index:)`` to get a
    /// ``KVCacheSimple`` for the standard single-sequence path.
    ///
    /// - Parameter keepIndices: Indices (in `0 ..< batchSize`) of sequences to keep.
    /// - Returns: A new ``BatchKVCache`` containing only the kept sequences.
    public func filter(keepIndices: [Int]) -> BatchKVCache {
        guard let k = batchKeys, let v = batchValues else {
            return BatchKVCache(batchSize: keepIndices.count, leftPadding: keepIndices.map { leftPadding[$0] })
        }

        let keptK = concatenated(keepIndices.map { k[($0)...($0), .ellipsis] }, axis: 0)
        let keptV = concatenated(keepIndices.map { v[($0)...($0), .ellipsis] }, axis: 0)
        let keptPadding = keepIndices.map { leftPadding[$0] }

        let newCache = BatchKVCache(batchSize: keepIndices.count, leftPadding: keptPadding)
        newCache.batchKeys = keptK
        newCache.batchValues = keptV
        newCache.offset = self.offset
        return newCache
    }

    /// Extract a single sequence's cache as a ``KVCacheSimple``.
    ///
    /// Use this when the batch drops to a single sequence and you want to switch
    /// back to the standard single-sequence code path.
    ///
    /// - Parameter index: Index of the sequence to extract.
    /// - Returns: A ``KVCacheSimple`` containing only the real (non-padded) tokens.
    public func extract(index: Int) -> KVCacheSimple {
        let simple = KVCacheSimple()
        guard let k = batchKeys, let v = batchValues else { return simple }

        let pad = leftPadding[index]
        let realStart = pad
        let realEnd = offset

        if realEnd > realStart {
            let extractedK = k[index...(index), .ellipsis, realStart..<realEnd, 0...]
            let extractedV = v[index...(index), .ellipsis, realStart..<realEnd, 0...]
            simple.state = [extractedK, extractedV]
        }
        return simple
    }
}
