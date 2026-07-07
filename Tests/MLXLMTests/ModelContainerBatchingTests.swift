// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@preconcurrency @testable import MLXLMCommon
import MLXNN
import Testing

// Metal-free behavioral tests for the PR4 ModelContainer integration. These
// assert the routing contract (scheduler == nil is a perfect no-op; the manual
// API requires a scheduler; admission-cap passthroughs are safe) WITHOUT
// running a real decode (which needs a GPU). No MLXMetalGuard.

@Suite("ModelContainer batching API")
struct ModelContainerBatchingTests {

    @Test("No scheduler installed → generateBatched throws schedulerUnavailable")
    func generateBatchedRequiresScheduler() async throws {
        let container = ModelContainer(context: makeContext())

        await #expect(throws: BatchedGenerationError.self) {
            // Built inside the closure: GenerationRequest is non-Sendable, so
            // capturing it from the test body cannot satisfy the async
            // #expect closure's sendability.
            let request = GenerationRequest(
                input: LMInput(tokens: MLXArray([Int32(1), 2, 3])))
            _ = try await container.generateBatched([request])
        }
    }

    @Test("Empty request array throws batchTooSmall when a scheduler is present")
    func generateBatchedRejectsEmpty() async throws {
        let container = ModelContainer(
            context: makeContext(),
            scheduler: InferenceScheduler()
        )
        await #expect(throws: BatchedGenerationError.batchTooSmall) {
            _ = try await container.generateBatched([])
        }
    }

    @Test("setMaxBatchSize is a no-op without a scheduler")
    func setMaxBatchSizeNoOpWithoutScheduler() async {
        let container = ModelContainer(context: makeContext())
        // Must not crash or trap; simply does nothing.
        await container.setMaxBatchSize(4)
        await container.setMaxBatchSize(nil)
    }

    @Test("promptCache property defaults to nil and round-trips when provided")
    func promptCachePropertyExposed() {
        let plain = ModelContainer(context: makeContext())
        #expect(plain.promptCache == nil)

        let cache = LRUPromptCache()
        let withCache = ModelContainer(
            context: makeContext(),
            scheduler: InferenceScheduler(),
            promptCache: cache
        )
        #expect(withCache.promptCache != nil)
    }

    @Test("usesScheduler reflects installation")
    func usesSchedulerReflectsInstall() {
        #expect(ModelContainer(context: makeContext()).usesScheduler == false)
        #expect(
            ModelContainer(context: makeContext(), scheduler: InferenceScheduler())
                .usesScheduler == true)
    }
}

// MARK: - Metal-free fixtures

/// Minimal context whose model is never actually run by these tests (they only
/// exercise routing guards that fail/short-circuit before any decode).
private func makeContext() -> ModelContext {
    ModelContext(
        configuration: ModelConfiguration(id: "test/model"),
        model: NoopLanguageModel(),
        processor: StandInUserInputProcessor(),
        tokenizer: TestTokenizer(vocabularySize: 16)
    )
}

/// A `LanguageModel` that conforms structurally but is never invoked by the
/// routing-guard tests. `callAsFunction` traps to make accidental use obvious.
private final class NoopLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize = 16
    var kvHeads: [Int] { [1] }

    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        fatalError("NoopLanguageModel must not be run by routing-guard tests")
    }
}
