# MLXLMCommon

# Documentation

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon) -- common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm) -- large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm) -- vision language model example implementations

# Quick Start

Using LLMs and VLMs from MLXLMCommon is as easy as:

```swift
let model = try await loadModel(id: "mlx-community/Qwen3-4B-4bit")
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?")
print(try await session.respond(to: "How about a great place to eat?")
```

For more information see 
[Evaluation](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/evaluation)
or [Using Models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using-model)
for more advanced API.

# Contents

MLXLMCommon contains types and code that is generic across many types
of language models, from LLMs to VLMs:

- Evaluation
- KVCache
- Loading
- UserInput

## Loading a Model

A model is typically loaded by using a `ModelFactory` and a `ModelConfiguration`:

```swift
// e.g. VLMModelFactory.shared
let modelFactory: ModelFactory

// e.g. VLMRegistry.paligemma3bMix4488bit
let modelConfiguration: ModelConfiguration

let container = try await modelFactory.loadContainer(configuration: modelConfiguration)
```

The `container` provides an isolation context (an `actor`) to run inference in the model.

Predefined `ModelConfiguration` instances are provided as static variables
on the `ModelRegistry` types or they can be created:

```swift
let modelConfiguration = ModelConfiguration(id: "mlx-community/paligemma-3b-mix-448-8bit")
```

The flow inside the `ModelFactory` goes like this:

```swift
public class VLMModelFactory: ModelFactory {

    public func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContext {
        // download the weight and config using HubApi
        // load the base configuration
        // using the typeRegistry create a model (random weights)
        // load the weights, apply quantization as needed, update the model
            // calls model.sanitize() for weight preparation
        // load the tokenizer
        // (vlm) load the processor configuration, create the processor
    }
}
```

Callers with specialized requirements can use these individual components to manually
load models, if needed.

## Evaluation Flow

- Load the Model
- UserInput
- LMInput
- generate()
    - NaiveStreamingDetokenizer
    - TokenIterator

## Using a Model

Once a model is loaded you can evaluate a prompt or series of
messages. Minimally you need to prepare the user input:

```swift
let prompt = "Describe the image in English"
var input = UserInput(prompt: prompt, images: image.map { .url($0) })
input.processing.resize = .init(width: 256, height: 256)
```

This example shows adding some images and processing instructions -- if
model accepts text only then these parts can be omitted. The inference
calls are the same.

Assuming you are using a `ModelContainer` (an actor that holds
a `ModelContext`, which is the bundled set of types that implement a
model), the first step is to convert the `UserInput` into the
`LMInput` (LanguageModel Input):

```swift
let generateParameters: GenerateParameters
let input: UserInput

let result = try await modelContainer.perform { [input] context in
    let input = try context.processor.prepare(input: input)

```

Given that `input` we can call `generate()` to produce a stream
of tokens. In this example we use a `NaiveStreamingDetokenizer`
to assist in converting a stream of tokens into text and print it.
The stream is stopped after we hit a maximum number of tokens:

```
    var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)

    return try MLXLMCommon.generate(
        input: input, parameters: generateParameters, context: context
    ) { tokens in

        if let last = tokens.last {
            detokenizer.append(token: last)
        }

        if let new = detokenizer.next() {
            print(new, terminator: "")
            fflush(stdout)
        }

        if tokens.count >= maxTokens {
            return .stop
        } else {
            return .more
        }
    }
}
```

### Wired Memory (Optional)

Use the policy-based API to coordinate a single global wired limit across tasks.
`WiredMemoryManager` and `WiredMemoryTicket` are provided by MLX, while
MLXLMCommon adds LLM-oriented policies (like `WiredFixedPolicy` or capped sum).
Policy-only admission is enabled by default on unsupported backends so the
same ticket logic applies on CPU (no OS limit changes are attempted).

```swift
let policy = WiredSumPolicy()
let ticket = policy.ticket(size: estimatedBytes)

let stream = try MLXLMCommon.generate(
    input: input,
    parameters: generateParameters,
    context: context,
    wiredMemoryTicket: ticket
)
```

Tickets are cheap handles into a shared manager that serializes updates and
restores the baseline when the last ticket completes.

For long-lived model weights, consider using a reservation ticket by passing
`kind: .reservation` when creating the ticket. Reservation tickets influence
admission and desired limits but do not keep the wired limit elevated unless
there is at least one active (inference) ticket.

#### Policies and Tickets

`WiredMemoryPolicy` is pure: it computes a desired limit from the baseline and
the active ticket sizes. The library includes a few policies:

- `WiredSumPolicy`: `baseline + sum(activeSizes)` with an optional cap.
- `WiredMaxPolicy`: `max(baseline, max(activeSizes))`.
- `WiredFixedPolicy`: fixed limit while any ticket is active.

Tickets are safe to start/end multiple times (extra ends are ignored). For
structured usage, wrap work with `WiredMemoryTicket.withWiredLimit` to ensure
start/end pairing and cancellation safety:

```swift
let policy = WiredSumPolicy()
let ticket = policy.ticket(size: estimatedBytes)

try await WiredMemoryTicket.withWiredLimit(ticket) {
    // run inference
}
```

#### Admission Control (Optional)

Policies can also gate concurrency by overriding `canAdmit`. If admission is
denied, `start()` suspends until capacity is available and resumes when tickets
end. This helps prevent over-commit when many inferences launch at once.

#### Debug Event Stream

Use `WiredMemoryManager.events()` to observe policy stacking and limit changes
in DEBUG builds. The stream is empty in release builds, so event logging is a
no-op in production.

## Batched Inference (InferenceScheduler)

`InferenceScheduler` enables concurrent multi-request inference by batching
multiple sequences through a single model forward pass. This is useful for
local servers, API gateways, or any scenario where more than one request may
be in flight at the same time.

### Basic Setup

```swift
// Load a model as usual
let context = try await LLMModelFactory.shared.loadContext(configuration: .init(id: "mlx-community/Qwen3-4B-4bit"))

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

Multiple concurrent callers get their own `AsyncStream<Generation>` and are
automatically batched together whenever they overlap in time.

### How It Works

1. **Prefill** — each request's prompt is processed independently with its own
   KV cache, using the same chunked prefill path as single-sequence inference.
2. **Batch decode** — once two or more sequences are ready, the scheduler stacks
   their current tokens into `[B, 1]` and runs a single batched forward pass per
   decode step.
3. **Single-to-batch upgrade** — if a second request arrives while the first is
   already mid-generation, the scheduler merges the two KV caches and continues
   with a batch of size 2. No restart required.
4. **Completion** — finished sequences are removed from the batch; the remainder
   continue decoding.

### Configuration

```swift
InferenceScheduler.Config(
    completionBatchSize: 4,   // max concurrent decode sequences (default: 4)
    prefillBatchSize:    4,   // max sequences admitted per prefill round (default: 4)
    maxQueueSize:        32   // max requests waiting in the queue (default: 32)
)
```

Default values are tuned for local/LAN use where memory is the primary
constraint. Increase `completionBatchSize` for higher-throughput scenarios if
memory allows.

### Architecture

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

**`BatchKVCache`** stores all sequences' keys and values in a single
`[B, nHeads, seqLen, headDim]` tensor. Shorter sequences are left-padded so
that all sequences share the same `offset` and can be processed in one pass.
The `fromSingle(_:leftPadding:)` factory method converts per-sequence
`KVCacheSimple` instances into a merged batch cache.

**`BatchTokenIterator`** runs prefill for each new sequence independently
(preserving correct per-token RoPE positions in the cache), then merges the
caches and runs all decode steps as a batch.

**`InferenceScheduler`** is an `actor` whose decode loop yields between steps
(via `Task.yield()`) so that new `submit()` calls can be processed promptly
without artificial batching delays.

### Known Limitations

- **RoPE position approximation during decode**: During a decode step, all
  sequences in the batch use the same `cache.offset` integer for RoPE. For
  sequences with different prompt lengths (which are left-padded to match),
  shorter sequences' decode tokens will have their RoPE position overestimated
  by `leftPadding[i]` steps. This is exact when all sequences have the same
  prompt length (simultaneous batch), and mild for small length differences.

- **Attention mask on single-token decode**: Many models use the deprecated
  `createAttentionMask(h:cache:)` helper which returns no mask for single-token
  decode steps (since there is normally no constraint needed). With left-padding
  this means padding positions are not explicitly excluded during decode,
  producing slight quality degradation proportional to the padding amount.
  Updating models to use `createAttentionMask(h:cache:windowSize:returnArray:)`
  will resolve this.

Both limitations only affect the single-to-batch upgrade path. Pure simultaneous
batching (all requests start at the same time, padded to the same prefill length)
is unaffected.
