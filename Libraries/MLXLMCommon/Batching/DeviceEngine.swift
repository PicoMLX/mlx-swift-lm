// Copyright © 2025 Apple Inc.

import Foundation
import MLX

/// Global actor that serializes MLX forward passes during batch inference.
///
/// All batched model forward passes are dispatched through this actor to prevent
/// concurrent MLX graph construction, which is not thread-safe.
///
/// Usage:
/// ```swift
/// let logits = await DeviceEngine.shared.run {
///     model(tokens, cache: cache)
/// }
/// ```
@globalActor
public actor DeviceEngine {
    public static let shared = DeviceEngine()

    /// Run work exclusively on the device engine's executor.
    ///
    /// Use this to serialize MLX forward passes and ensure only one batch
    /// decode step runs at a time.
    @discardableResult
    public func run<T: Sendable>(_ work: @Sendable () throws -> T) throws -> T {
        try work()
    }

    /// Run async work exclusively on the device engine's executor.
    @discardableResult
    public func run<T: Sendable>(_ work: @Sendable () async throws -> T) async throws -> T {
        try await work()
    }
}
