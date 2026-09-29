// Copyright © 2026 Apple Inc.

import MLX

/// Model-owned decoder state that can follow rows through continuous batching.
/// Prefill each request independently before merging its state. These values,
/// like KV caches, must remain on one serial executor.
public protocol BatchedGenerationStateProviding: LanguageModel {
    /// Logical retained decoder-state bytes for one prepared row.
    var generationStateBytesPerRow: Int { get }

    /// Combine states in row order, discarding state used only during prefill.
    func mergeGenerationStates(_ states: [LMOutput.State]) -> LMOutput.State

    /// Keep the selected rows in the same order as the corresponding caches.
    func selectGenerationState(_ state: LMOutput.State, batchIndices: [Int]) -> LMOutput.State

    /// Arrays that must be evaluated before releasing the state's allocation coverage.
    func generationStateArrays(_ state: LMOutput.State) -> [MLXArray]
}
