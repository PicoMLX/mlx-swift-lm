# Batched Inference (InferenceScheduler)

`InferenceScheduler` enables concurrent multi-request inference by batching multiple sequences through a single model forward pass. This is useful for local servers, API gateways, or any scenario where more than one request may be in flight at the same time.

## Basic Setup

```swift
// Load a model as usual
let context = try await LLMModelFactory.shared.loadContext(
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

// Wrap it in a ModelContainer with a scheduler
let container = ModelContainer(
    context: context,
    schedulerConfig: InferenceScheduler.Config(
        completionBatchSize: 4   // up to 4 sequences decoded together
    )
)

// Use generate() exactly as before — the scheduler handles batching transparently
let stream = try await container.generate(input: input, parameters: parameters)
for await event in stream {
    if case .chunk(let text) = event { print(text, terminator: "") }
}
```

Multiple concurrent callers each get their own `AsyncStream<Generation>` and are automatically batched together whenever they overlap in time.

## Configuration

```swift
InferenceScheduler.Config(
    completionBatchSize: 4,   // max concurrent decode sequences (default: 4)
    prefillBatchSize:    4,   // max sequences admitted per prefill round (default: 4)
    maxQueueSize:        32   // max requests waiting in the queue (default: 32)
)
```

Default values are tuned for local/LAN use where memory is the primary constraint. Increase `completionBatchSize` for higher-throughput scenarios if memory allows.

## How It Works

1. **Prefill** — each request's prompt is processed independently with its own KV cache, using the same chunked prefill path as single-sequence inference.
2. **Batch decode** — once two or more sequences are ready, the scheduler stacks their current tokens into `[B, 1]` and runs a single batched forward pass per decode step.
3. **Single-to-batch upgrade** — if a second request arrives while the first is already mid-generation, the scheduler merges the two KV caches and continues with a batch of size 2. No restart required.
4. **Completion** — finished sequences are removed from the batch; the remainder continue decoding.

## Architecture

```
Client Request
    ↓
InferenceScheduler (actor — manages queue + single-to-batch upgrade)
    ↓
BatchTokenIterator (batched prefill + decode)
    ↓
BatchKVCache (left-padded, [B, heads, seq_len, head_dim])
    ↓
LanguageModel forward pass ([B, 1] tokens → [B, 1, vocab] logits)
```

**`BatchKVCache`** stores all sequences' keys and values in a single `[B, nHeads, seqLen, headDim]` tensor. Shorter sequences are left-padded so that all sequences share the same `offset` and can be processed in one pass. The `fromSingle(_:leftPadding:)` factory method converts per-sequence `KVCacheSimple` instances into a merged batch cache.

**`BatchTokenIterator`** runs prefill for each new sequence independently (preserving correct per-token RoPE positions in the cache), then merges the caches and runs all decode steps as a batch.

**`InferenceScheduler`** is an `actor` whose decode loop yields between steps (via `Task.yield()`) so that new `submit()` calls can be processed promptly without artificial batching delays.

**`DeviceEngine`** is a `@globalActor` that serializes all MLX forward passes onto a single executor, preventing concurrent GPU access from different tasks.

## BatchKVCache API

### Creating

```swift
// From scratch (e.g. unit tests or custom prefill)
let cache = BatchKVCache(batchSize: 2, leftPadding: [0, 0])

// From existing per-sequence KVCacheSimple instances (single-to-batch upgrade)
let batchCaches = BatchKVCache.fromSingle(perSequenceCaches: [seqACaches, seqBCaches])
// Returns [[BatchKVCache]] — one [BatchKVCache] per layer
```

### Querying

```swift
cache.batchSize    // number of sequences
cache.offset       // shared sequence position (aligned to longest sequence)
cache.leftPadding  // [Int] per-sequence left-pad amounts
```

### Filtering and Extraction

```swift
// Remove completed sequences (keepIndices is an array of surviving indices)
let reduced = cache.filter(keepIndices: [0, 2])

// Extract single sequence back to KVCacheSimple (downgrade path)
let single = cache.extract(index: 1)  // returns KVCacheSimple with padding removed
```

### Masking

`makeMask(n:windowSize:returnArray:)` returns a `MaskMode`:
- `.none` — for n=1 decode with no left-padding (most common case)
- `.array(MLXArray)` — `[B, 1, n, total]` additive float mask when sequences have different lengths

## Known Limitations

### RoPE Position Approximation During Decode

During a decode step, all sequences in the batch use the same `cache.offset` integer for RoPE. For sequences with different prompt lengths (which are left-padded to match), shorter sequences' decode tokens will have their RoPE position overestimated by `leftPadding[i]` steps. This is exact when all sequences have the same prompt length (simultaneous batch), and mild for small length differences.

### Attention Mask on Single-Token Decode

Many models use the deprecated `createAttentionMask(h:cache:)` helper which returns no mask for single-token decode steps. With left-padding, this means padding positions are not explicitly excluded during decode, producing slight quality degradation proportional to the padding amount. Updating models to use `createAttentionMask(h:cache:windowSize:returnArray:)` will resolve this.

Both limitations only affect the single-to-batch upgrade path. Pure simultaneous batching (all requests start at the same time, padded to the same prefill length) is unaffected.

## Scheduling Policy

The scheduler is **no-wait**: requests are batched immediately if capacity allows. There is no artificial delay to accumulate requests. Batching only happens when a second request arrives while the first is already in flight.

This differs from server-oriented Python implementations which use wait timers to batch many short requests together — appropriate for LAN/Internet traffic but adds unnecessary latency for local use.
