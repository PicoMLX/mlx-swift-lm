// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// MARK: - BatchedCache

/// A ``KVCache`` that supports the continuous-batching primitives used by the
/// batched inference engine: in-place row filtering, concatenation of admitted
/// rows, ragged-prefill prepare/finalize, single-row extraction, and chunk-local
/// advancement.
///
/// Both full-attention caches (``BatchKVCache``, ``BatchRotatingKVCache``) and
/// SSM-style caches (``ArraysCache`` / ``MambaCache``) conform, so the engine can
/// treat a heterogeneous per-layer cache list uniformly.
///
/// Conformers are **not** thread-safe; the engine mutates them from a single
/// serial executor.
public protocol BatchedCache: KVCache {
    /// In-place keep only the rows at the given batch indices.
    func filterBatched(batchIndices: MLXArray)

    /// In-place append `other`'s rows. The runtime types must match.
    func extendBatched(_ other: any BatchedCache)

    /// Prepare cache metadata before a ragged prompt prefill.
    func prepareBatched(leftPadding: [Int]?, lengths: [Int]?, rightPadding: [Int]?)

    /// Finalize cache metadata after a ragged prompt prefill.
    func finalizeBatched()

    /// Extract one row as its corresponding single-request cache.
    func extractBatched(_ idx: Int) -> any KVCache

    /// Advance chunk-local metadata after a chunked prefill step.
    func advanceBatched(_ n: Int)
}

// MARK: - BatchedCacheList

/// Batched wrapper for composite per-layer caches. Some hybrid models keep
/// multiple cache objects per logical layer, so batching has to preserve that
/// nested topology instead of treating the composite as full attention.
public final class BatchedCacheList: CacheList, BatchedCache {

    private let batchedCaches: [any BatchedCache]

    internal init(caches: [any BatchedCache]) {
        self.batchedCaches = caches
        super.init(caches: caches.map { $0 as any KVCache })
    }

    public func filterBatched(batchIndices: MLXArray) {
        for cache in batchedCaches {
            cache.filterBatched(batchIndices: batchIndices)
        }
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? BatchedCacheList else {
            preconditionFailure("BatchedCacheList.extendBatched requires another BatchedCacheList")
        }
        precondition(
            batchedCaches.count == other.batchedCaches.count,
            "Cannot extend BatchedCacheList with different child count"
        )

        for (a, b) in zip(batchedCaches, other.batchedCaches) {
            a.extendBatched(b)
        }
    }

    public func prepareBatched(leftPadding: [Int]?, lengths: [Int]?, rightPadding: [Int]?) {
        for cache in batchedCaches {
            cache.prepareBatched(
                leftPadding: leftPadding,
                lengths: lengths,
                rightPadding: rightPadding
            )
        }
    }

    public func finalizeBatched() {
        for cache in batchedCaches {
            cache.finalizeBatched()
        }
    }

    public func extractBatched(_ idx: Int) -> any KVCache {
        CacheList(caches: batchedCaches.map { $0.extractBatched(idx) })
    }

    public func advanceBatched(_ n: Int) {
        for cache in batchedCaches {
            cache.advanceBatched(n)
        }
    }

    /// The attention child that carries the per-row left-padding mask, if any.
    ///
    /// Composite per-layer caches (e.g. BaichuanM1's `CacheList(MambaCache,
    /// KVCacheSimple/RotatingKVCache)`) pair a recurrent/SSM child with a
    /// full-attention child. Only the full-attention child (a
    /// `BatchPositionedKVCache`) holds the per-row `leftPadding` needed to build
    /// a correct batched attention mask; the SSM child produces no attention mask.
    private var maskingChild: (any BatchPositionedKVCache)? {
        batchedCaches.lazy.compactMap { $0 as? any BatchPositionedKVCache }.first
    }

    /// Delegate masking to the attention child so composite caches honour per-row
    /// left padding.
    ///
    /// `BaichuanM1ModelInner` builds its attention mask via
    /// `createAttentionMask(h:cache: cache?.first)` where `cache?.first` is this
    /// composite (a `CacheList`/`BatchedCacheList`). Without this override the
    /// inherited `BaseKVCache.makeMask` returns `.causal`/`.none` and never masks
    /// the left-padded KV slots under continuous batching. Delegating to the
    /// `BatchPositionedKVCache` child reuses its batched `createCausalMask`
    /// (which honours `leftPadding`).
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if let child = maskingChild {
            return child.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
        }
        return super.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    /// Report the attention child's offset so callers that derive masks from
    /// `cache.offset` (e.g. the array-based `createAttentionMask(h:cache:)`
    /// overload) see the batched progress rather than the inherited scalar `0`.
    public override var offset: Int {
        get { maskingChild?.offset ?? super.offset }
        set { super.offset = newValue }
    }

    public override var maxSize: Int? {
        maskingChild?.maxSize ?? super.maxSize
    }

    /// Surface the attention child's per-row RoPE offset so composite caches use
    /// the batched positions rather than the inherited scalar offset.
    public override var ropeOffset: RoPEOffset {
        maskingChild?.ropeOffset ?? super.ropeOffset
    }

    /// Deep-copy as a `BatchedCacheList`, preserving the `BatchedCache` wrapper.
    ///
    /// `CacheList.copy()` rebuilds a plain `CacheList` from copied children, which
    /// would strip the continuous-batching protocol from a snapshot taken after the
    /// factory builds a composite batched cache. Each batched child's `copy()`
    /// returns the same concrete batched type, so it re-conforms to `BatchedCache`.
    public override func copy() -> any KVCache {
        BatchedCacheList(
            caches: batchedCaches.map { child in
                guard let copied = child.copy() as? any BatchedCache else {
                    preconditionFailure(
                        "BatchedCache.copy() must return a BatchedCache for "
                            + "BatchedCacheList snapshots"
                    )
                }
                return copied
            }
        )
    }
}

// MARK: - SSM cache conformance

extension ArraysCache: BatchedCache {
    public func filterBatched(batchIndices: MLXArray) {
        filter(batchIndices: batchIndices)
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? ArraysCache else {
            preconditionFailure("ArraysCache.extendBatched requires another ArraysCache")
        }
        extend(other: other)
    }

    public func prepareBatched(leftPadding _: [Int]?, lengths: [Int]?, rightPadding _: [Int]?) {
        prepare(lengths: lengths)
    }

    public func finalizeBatched() {
        finalize()
    }

    public func extractBatched(_ idx: Int) -> any KVCache {
        extract(idx)
    }

    public func advanceBatched(_ n: Int) {
        advance(n)
    }
}

// MARK: - Factory

/// A closure that allocates one batched cache for `leftPadding.count` rows.
public typealias BatchedCacheFactory = (_ leftPadding: [Int]) -> any BatchedCache

/// Error thrown when a model's cache topology cannot be batched.
public enum BatchedCacheError: Error, CustomStringConvertible, Equatable {
    case unsupportedCacheTopology(layer: Int, path: String, cacheType: String, reason: String)

    public var description: String {
        switch self {
        case .unsupportedCacheTopology(let layer, let path, let cacheType, let reason):
            return "Unsupported cache topology at layer \(layer), \(path): "
                + "\(cacheType). \(reason)"
        }
    }
}

/// Build one ``BatchedCacheFactory`` per layer from a probe of single-stream
/// caches (typically `model.newCache(parameters:)`).
///
/// Validating the topology once at engine-construction time means per-admission
/// allocation is a cheap closure call that cannot fail.
///
/// - Throws: ``BatchedCacheError`` if any layer's cache type has no batched
///   implementation (e.g. quantized or chunked caches).
public func makeBatchedCacheFactories(for probe: [any KVCache]) throws -> [BatchedCacheFactory] {
    try probe.enumerated().map { layer, cache in
        try makeBatchedCacheFactory(for: cache, layer: layer, path: "layer")
    }
}

/// Convenience: build a fresh set of batched caches for `leftPadding.count` rows
/// directly from a probe of single-stream caches.
public func makeBatchedCache(
    for probe: [any KVCache], leftPadding: [Int]
) throws -> [any BatchedCache] {
    try makeBatchedCacheFactories(for: probe).map { $0(leftPadding) }
}

private func makeBatchedCacheFactory(
    for cache: any KVCache,
    layer: Int,
    path: String
) throws -> BatchedCacheFactory {
    let cacheType = String(describing: Swift.type(of: cache))

    func unsupported(_ reason: String) -> BatchedCacheError {
        .unsupportedCacheTopology(
            layer: layer,
            path: path,
            cacheType: cacheType,
            reason: reason
        )
    }

    if cache is QuantizedKVCache {
        throw unsupported("Quantized KV caches are not supported by continuous batching.")
    }

    if cache is ChunkedKVCache {
        throw unsupported("Chunked KV caches are not supported by continuous batching.")
    }

    if let cacheList = cache as? CacheList {
        let childFactories = try cacheList.children.enumerated().map { childIndex, child in
            try makeBatchedCacheFactory(
                for: child,
                layer: layer,
                path: "\(path).children[\(childIndex)]"
            )
        }
        return { leftPadding in
            BatchedCacheList(caches: childFactories.map { $0(leftPadding) })
        }
    }

    // Exact-type matches avoid misclassifying subclasses such as
    // MambaCache : ArraysCache and ChunkedKVCache : KVCacheSimple.
    //
    // SSM caches (MambaCache / ArraysCache) only mask left-padded / right-padded
    // positions when the *model* threads `createSSMMask(...)` into its conv/SSM
    // mixers. Several in-repo Mamba users never do this — their SSM mask is
    // hard-coded `nil` (e.g. `GraniteMoeHybrid`'s `createSSMMask` returns `nil`
    // and is passed at its mixer call sites; `Jamba`, `LFM2`, and `BaichuanM1`
    // call their conv/SSM mixers without any mask argument). For those models a
    // ragged or right-padded continuous batch would feed pad-token embeddings
    // through the recurrent state instead of zeroing them, silently corrupting
    // every later token in the row.
    //
    // The factory only sees the cache *instance*, not the model, so it cannot
    // distinguish the safe SSM users (NemotronH, Qwen3Next, Qwen35, LFM2MoE,
    // FalconH1 — all route a real `createSSMMask`) from the unsafe ones. Per the
    // conservative policy, keep SSM topologies OUT of continuous batching until a
    // model-level guarantee that the SSM mask is honoured exists; reject them
    // here rather than risk corrupting recurrent state. The batched `MambaCache`
    // / `ArraysCache` machinery still exists and is exercised directly by the
    // SSM lifecycle tests; only the topology-probing factory rejects it.
    if Swift.type(of: cache) == MambaCache.self {
        throw unsupported(
            "SSM (Mamba) caches are not supported by continuous batching: some "
                + "in-repo models do not thread createSSMMask into their conv/SSM "
                + "mixers, so ragged batches would corrupt recurrent state.")
    }

    if Swift.type(of: cache) == ArraysCache.self {
        throw unsupported(
            "SSM (ArraysCache) caches are not supported by continuous batching: "
                + "some in-repo models do not thread createSSMMask into their "
                + "conv/SSM mixers, so ragged batches would corrupt recurrent state.")
    }

    if let rotating = cache as? RotatingKVCache {
        guard let maxSize = rotating.maxSize else {
            throw unsupported("RotatingKVCache must have a non-nil maxSize.")
        }

        // RotatingKVCache.keep is private; metaState layout is
        // [keep, maxCacheSize, step, offset, idx].
        let keep = Int(rotating.metaState.first ?? "0") ?? 0

        // keep > 0 cannot currently be combined with per-row left padding: at
        // the rotation wrap, `BatchRotatingKVCache` rolls a padded row's pads
        // to the END of the buffer to protect the keep prefix, but the
        // prefix-only `leftPadding` mask cannot express trailing garbage, so
        // those zero-K/V slots would be attended until overwritten. Until the
        // mask model supports it, keep-prefix topologies fall back to
        // single-stream (in-repo models all use keep == 0; keep == 4 arises
        // only via `GenerateParameters.maxKVSize`).
        guard keep == 0 else {
            throw unsupported(
                "RotatingKVCache with keep > 0 is not supported by continuous batching."
            )
        }

        return { leftPadding in
            BatchRotatingKVCache(maxSize: maxSize, leftPadding: leftPadding, keep: keep)
        }
    }

    if Swift.type(of: cache) == KVCacheSimple.self {
        return { leftPadding in BatchKVCache(leftPadding: leftPadding) }
    }

    throw unsupported("No batched cache implementation exists for this cache type.")
}
