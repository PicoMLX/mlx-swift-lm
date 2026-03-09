// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Cache capability for batched caches that track per-sequence positions.
public protocol BatchPositionedKVCache: KVCache {
    var offsets: MLXArray { get }
}

extension BatchKVCache: BatchPositionedKVCache {}

extension BatchRotatingKVCache: BatchPositionedKVCache {}

public func batchedOffsets(for cache: KVCache?) -> MLXArray? {
    (cache as? any BatchPositionedKVCache)?.offsets
}

public func applyRotaryPosition(
    _ rope: any OffsetLayer,
    to x: MLXArray,
    cache: KVCache?
) -> MLXArray {
    if let offsets = batchedOffsets(for: cache),
        let arrayOffsetRope = rope as? any ArrayOffsetLayer
    {
        return arrayOffsetRope.callAsFunction(x, offset: offsets)
    }
    return rope.callAsFunction(x, offset: cache?.offset ?? 0)
}

public func applyRotaryPosition(
    _ x: MLXArray,
    cache: KVCache?,
    dimensions: Int,
    traditional: Bool,
    base: Float?,
    scale: Float,
    freqs: MLXArray? = nil
) -> MLXArray {
    if let offsets = batchedOffsets(for: cache) {
        return MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: traditional,
            base: base,
            scale: scale,
            offset: offsets,
            freqs: freqs
        )
    }

    return MLXFast.RoPE(
        x,
        dimensions: dimensions,
        traditional: traditional,
        base: base,
        scale: scale,
        offset: cache?.offset ?? 0,
        freqs: freqs
    )
}
