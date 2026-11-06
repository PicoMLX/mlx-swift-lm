// Copyright © 2026 Apple Inc.

import MLX

/// Controls cache evaluation during ``TokenIterator`` and batched generation decode.
///
/// Speculative iterators do not use this policy. Prompt preparation is unchanged.
public enum KVCacheEvaluationPolicy: Sendable, Equatable {
    /// Explicitly evaluate all cache state with each sampled token. The default.
    case allState

    /// Omit attention buffers that the caller guarantees are consumed by the logits.
    ///
    /// This is an expert opt-in: cache types alone cannot establish that dependency.
    /// A model that updates attention state without consuming it in the logits must use
    /// ``allState``. Otherwise unevaluated updates can retain an unbounded lazy graph.
    /// Recurrent and custom cache state, and sliding-batch metadata, remain explicit roots.
    /// Only exact built-in full, rotating and affine-quantized attention caches are optimized.
    /// Dynamic cache conversion keeps full-state evaluation until its plan completes.
    /// Raw model logits remain evaluation roots when processors or samplers replace them.
    case attentionDependencies

    func evaluationState(for caches: [KVCache]) -> [MLXArray] {
        guard self == .attentionDependencies else { return caches.flatMap { $0.state } }
        return caches.flatMap { cache in
            if let rotating = cache as? BatchRotatingKVCache {
                return [rotating.batchOffset, rotating.leftPadding]
            }
            let cacheType = type(of: cache)
            if cacheType == CacheList.self, let list = cache as? CacheList {
                return evaluationState(for: list.children)
            }
            if cacheType == KVCacheSimple.self || cacheType == RotatingKVCache.self
                || cacheType == QuantizedKVCache.self || cacheType == BatchKVCache.self
                || cacheType == BatchQuantizedKVCache.self
            {
                return []
            }
            return cache.state
        }
    }
}
