# MLX Swift LM

MLX Swift LM is a Swift package to build tools and applications with large language models (LLMs) and vision language models (VLMs) in [MLX Swift](https://github.com/ml-explore/mlx-swift).

> [!IMPORTANT]
> The `main` branch is a _new_ major version number: 3.x.  In order
> to decouple from tokenizer and downloader packages some breaking
> changes were introduced. See [upgrading documentation](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/upgrade) for detailed instructions on upgrading.

Some key features include:

- Model loading with integrations for a variety of tokenizer and model downloading packages.
- Low-rank (LoRA) and full model fine-tuning with support for quantized models.
- Many model architectures for both LLMs and VLMs.

For some example applications and tools that use MLX Swift LM, check out [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples).

## Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [Techniques for developing in mlx-swift-lm](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/developing)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon): Common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm): Large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm): Vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders): Popular encoders and embedding models example implementations

## Usage

This package integrates with a variety of tokenizer and downloader packages through protocol conformance. Users can pick from three ways to integrate with these packages, which offer different tradeoffs between freedom and convenience.

See documentation on [how to integrate mlx-swift-lm and downloaders/tokenizers](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using).

### Installation

Add the core package to your `Package.swift`:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
```

Then chose one of the methods below to select download and tokenizer implementations.

### Method 1: Integration Packages

Then add your preferred tokenizer and downloader integrations, see [how to integrate mlx-swift-lm and downloaders/tokenizers](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using#Integration-Packages):

```swift
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx", from: "0.2.0", traits: ["Swift"]),
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx", from: "0.2.0"),
```

And add the libraries to your target:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
        .product(name: "MLXLMHFAPI", package: "swift-hf-api-mlx"),
    ]),
```

### Method 2: Macros

This preserves parity with mlx-swift-lm 2.x.  Simply reference the huggingface packages and use the `MLXHuggingFace` macros to adapt the APIs.  [Read more here](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using#MLXHuggingFace-Macros).

Add these to your dependencies:

```swift
.package(url: "https://github.com/huggingface/swift-huggingface", .upToNextMajor(from: "0.9.0")),
.package(url: "https://github.com/huggingface/swift-transformers", .upToNextMajor(from: "1.3.0")),
```

And add the libraries to your target:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
        .product(name: "HuggingFace", package: "swift-huggingface"),
        .product(name: "Tokenizers", package: "swift-transformers"),
    ]),
```

## Quick Start

You can get started with a wide variety of open-weights LLMs and VLMs using this simplified API (for more details, see  [MLXLMCommon](Libraries/MLXLMCommon)):

If using the [integration macros](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using#MLXHuggingFace-Macros), you can get started with code like this:

```swift
import MLXLLM
import MLXLMCommon
import MLXHuggingFace

import HuggingFace
import Tokenizers

let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit

let model = try await #huggingFaceLoadModelContainer(
    configuration: modelConfiguration
)

let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

Using the [adapter packages](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using#Integration-Packages) you would have similar code -- replace the imports and the load line.

For example, loading from a local directory using the [DePasqualeOrg/swift-tokenizers-mlx](https://github.com/DePasqualeOrg/swift-tokenizers-mlx):

```swift
import MLXLLM
import MLXLMTokenizers

let modelDirectory = URL(filePath: "/path/to/model")
let container = try await loadModelContainer(
    from: modelDirectory,
    using: TokenizersLoader()
)
```

Use a custom Hugging Face client:

```swift
import MLXLLM
import MLXLMHuggingFace
import MLXLMTokenizers

let hub = HubClient(token: "hf_...")
let container = try await loadModelContainer(
    from: hub,
    using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
```

Use a custom downloader:

```swift
import MLXLLM
import MLXLMCommon
import MLXLMTokenizers

struct S3Downloader: Downloader {
    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        // Download files and return a local directory URL.
        return URL(filePath: "/tmp/model")
    }
}

let container = try await loadModelContainer(
    from: S3Downloader(),
    using: TokenizersLoader(),
    id: "my-bucket/my-model"
)
```

Or use the underlying API to control every aspect of the evaluation.

## Migrating to Version 3

Version 3 of MLX Swift LM decouples the tokenizer and downloader implementations. See the [integrations](#Tokenizer-and-Downloader-Integrations) section for details.

### New dependencies

Add your preferred tokenizer and downloader adapters:

```swift
// Before (2.x) – single dependency
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", from: "2.30.0"),

// After (3.x) – core + adapters
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", from: "3.0.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx/", from: "0.1.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx/", from: "0.1.0"),
```

And add their products to your target:

```swift
.product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
.product(name: "MLXLMHFAPI", package: "swift-hf-api-mlx"),

// If you use MLXEmbedders:
.product(name: "MLXEmbeddersTokenizers", package: "swift-tokenizers-mlx"),
.product(name: "MLXEmbeddersHFAPI", package: "swift-hf-api-mlx"),
```

### New imports

```swift
// Before (2.x)
import MLXLLM

// After (3.x)
import MLXLLM
import MLXLMHFAPI      // Downloader adapter
import MLXLMTokenizers // Tokenizer adapter
```

If you use MLXEmbedders:

```swift
import MLXEmbedders
import MLXEmbeddersHFAPI      // Downloader adapter
import MLXEmbeddersTokenizers // Tokenizer adapter
```

### Loading API changes

The core APIs now include a `from:` parameter of type `URL` or `any Downloader` as well as a `using:` parameter for the tokenizer loader. Tokenizer integration packages may supply convenience methods with a default tokenizer loader, allowing you to omit the `using:` parameter.

The most visible call-site changes are:

- `hub:` → `from:`: Models are now loaded from a directory `URL` or  `Downloader`.
- `HubApi` → `HubClient`: A new implementation of the Hugging Face Hub client is used.

Example when downloading from Hugging Face:

```swift
// Before (2.x) – hub defaulted to HubApi()
let container = try await loadModelContainer(
    id: "mlx-community/Qwen3-4B-4bit"
)

// After (3.x) – Using Swift Hugging Face + Swift Tokenizers
let container = try await loadModelContainer(
    from: HubClient.default,
    id: "mlx-community/Qwen3-4B-4bit"
)
```

At the lower-level core API, you can still pass any `Downloader` and any `TokenizerLoader` explicitly.

Loading from a local directory:

```swift
// Before (2.x)
let container = try await loadModelContainer(directory: modelDirectory)

// After (3.x)
let container = try await loadModelContainer(from: modelDirectory)
```

Loading with a model factory:

```swift
let container = try await LLMModelFactory.shared.loadContainer(
    from: HubClient.default,
    configuration: modelConfiguration
)
```

Loading an embedder:

```swift
import MLXEmbedders
import MLXEmbeddersHFAPI
import MLXEmbeddersTokenizers

let container = try await loadModelContainer(
    from: HubClient.default,
    configuration: .configuration(id: "sentence-transformers/all-MiniLM-L6-v2")
)
```

### Renamed methods

`decode(tokens:)` is renamed to `decode(tokenIds:)` to align with the `transformers` library in Python:

```swift
// Before (2.x)
let text = tokenizer.decode(tokens: ids)

// After (3.0)
let text = tokenizer.decode(tokenIds: ids)
```

## Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon): Common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm): Large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm): Vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders): Popular encoders and embedding models example implementations

## Breaking Changes

### Loading API

The `hub` parameter (previously `HubApi`) has been replaced with `from` (any `Downloader` or `URL` for a local directory). Functions that previously defaulted to `defaultHubApi` no longer have a default – callers must either pass a `Downloader` explicitly or use the convenience methods in `MLXLMHuggingFace` / `MLXEmbeddersHuggingFace`, which default to `HubClient.default`.

For most users who were using the default Hub client, adding `import MLXLMHuggingFace` or `import MLXEmbeddersHuggingFace` and using the convenience overloads is sufficient.

Users who were passing a custom `HubApi` instance should create a `HubClient` instead and pass it as the `from` parameter. `HubClient` conforms to `Downloader` via `MLXLMHuggingFace`.

### `ModelConfiguration`

- `tokenizerId` and `overrideTokenizer` have been replaced by `tokenizerSource: TokenizerSource?`, which supports `.id(String)` for remote sources and `.directory(URL)` for local paths.
- `preparePrompt` has been removed. This shouldn't be used anyway, since support for chat templates is available.
- `modelDirectory(hub:)` has been removed. For local directories, pass the `URL` directly to the loading functions. For remote models, the `Downloader` protocol handles resolution.

### Tokenizer loading

`loadTokenizer(configuration:hub:)` has been removed. Tokenizer loading now uses `AutoTokenizer.from(directory:)` from Swift Tokenizers directly.

`replacementTokenizers` (the `TokenizerReplacementRegistry`) has been removed. Use `AutoTokenizer.register(_:for:)` from Swift Tokenizers instead.

### `defaultHubApi`

The `defaultHubApi` global has been removed. Hugging Face Hub access is now provided by `HubClient.default` from the `HuggingFace` module.

### Low-level APIs

- `downloadModel(hub:configuration:progressHandler:)` → `Downloader.download(id:revision:matching:useLatest:progressHandler:)`
- `loadTokenizerConfig(configuration:hub:)` → `AutoTokenizer.from(directory:)`
- `ModelFactory._load(hub:configuration:progressHandler:)` → `_load(configuration: ResolvedModelConfiguration)`
- `ModelFactory._loadContainer`: removed (base `loadContainer` now builds the container from `_load`)

