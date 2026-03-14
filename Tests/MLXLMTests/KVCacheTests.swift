import Foundation
import MLX
import MLXLMCommon
import Testing

@Test(
    .enabled(
        if: MLXMetalGuard.isAvailable,
        "Requires MLX Metal library (unavailable in SPM debug builds)"),
    .serialized,
    arguments: [
        ({ KVCacheSimple() } as @Sendable () -> any KVCache),
        ({ RotatingKVCache(maxSize: 32) } as @Sendable () -> any KVCache),
        ({ QuantizedKVCache() } as @Sendable () -> any KVCache),
        ({ ChunkedKVCache(chunkSize: 16) } as @Sendable () -> any KVCache),
        ({ ArraysCache(size: 2) } as @Sendable () -> any KVCache),
        ({ MambaCache() } as @Sendable () -> any KVCache),
    ])
func testCacheSerialization(creator: @Sendable () -> any KVCache) async throws {
    let cache = (0 ..< 10).map { _ in creator() }
    let keys = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    for item in cache {
        switch item {
        case let arrays as ArraysCache:
            arrays[0] = keys
            arrays[1] = values
        case let quantized as QuantizedKVCache:
            _ = quantized.updateQuantized(keys: keys, values: values)
        default:
            _ = item.update(keys: keys, values: values)
        }
    }

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try savePromptCache(url: url, cache: cache, metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    #expect(cache.count == loadedCache.count)
    for (lhs, rhs) in zip(cache, loadedCache) {
        #expect(type(of: lhs) == type(of: rhs))
        #expect(lhs.metaState == rhs.metaState)
        #expect(lhs.state.count == rhs.state.count)
    }
}
