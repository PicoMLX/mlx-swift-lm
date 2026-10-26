// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@Suite struct TensorParallelRefusalTests {
    private struct LoaderReached: Error {}

    // The model cache is keyed by model ID alone, so a sharded container must never enter it.
    @Test("A configuration naming a distributed group is refused before anything loads")
    func tensorParallelConfigurationIsRefused() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        var configuration = ModelConfiguration(id: "org/sharded-\(UUID().uuidString)")
        configuration.tensorParallelGroup = try MLXDistributed.initialize()
        let model = MLXLanguageModel(
            configuration: configuration,
            capabilities: [],
            weightsLocation: { _ in URL(fileURLWithPath: "/no/such/path") },
            load: { _, _ in throw LoaderReached() })

        await #expect(throws: ModelFactoryError.self) {
            _ = try await model.loadContainer()
        }
    }
}

#endif
