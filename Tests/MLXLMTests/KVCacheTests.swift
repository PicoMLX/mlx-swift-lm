import Foundation
import MLX
import MLXLMCommon
import Testing

enum CacheFactory: String, CaseIterable, Sendable {
    case simple
    case rotating
    case quantized
    case chunked
    case arrays
    case mamba

    func make() -> any KVCache {
        switch self {
        case .simple:
            return KVCacheSimple()
        case .rotating:
            return RotatingKVCache(maxSize: 32)
        case .quantized:
            return QuantizedKVCache()
        case .chunked:
            return ChunkedKVCache(chunkSize: 16)
        case .arrays:
            return ArraysCache(size: 2)
        case .mamba:
            return MambaCache()
        }
    }
}

@Test(
    .serialized,
    arguments: CacheFactory.allCases
)
func testCacheSerialization(factory: CacheFactory) async throws {
    let cache = (0 ..< 10).map { _ in factory.make() }
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
