// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Global singleton that serializes all MLX forward passes across models.
///
/// MLX is not generally thread-safe for concurrent evaluations on the same device.
/// `DeviceEngine` ensures exactly one MLX eval runs at a time, even with multiple
/// models and schedulers active.
///
/// Usage:
/// ```swift
/// let result = try await DeviceEngine.shared.runForward {
///     model(tokens, cache: cache)
/// }
/// ```
///
/// All model containers and schedulers should route MLX operations through this actor.
@globalActor
public actor DeviceEngine {
    
    /// Shared singleton instance
    public static let shared = DeviceEngine()
    
    private init() {}
    
    /// Serialize an MLX forward pass.
    ///
    /// The actor guarantees that only one `work` closure executes at a time,
    /// preventing concurrent MLX evaluations that could cause undefined behavior.
    ///
    /// - Parameter work: The MLX operation to perform (e.g., model forward pass)
    /// - Returns: The result of the work closure
    /// - Throws: Rethrows any error from the work closure
    public func runForward<T: Sendable>(
        _ work: @Sendable () async throws -> T
    ) async rethrows -> T {
        try await work()
    }
    
    /// Synchronous variant for non-async contexts.
    ///
    /// - Parameter work: The MLX operation to perform
    /// - Returns: The result of the work closure
    /// - Throws: Rethrows any error from the work closure
    public func runForwardSync<T: Sendable>(
        _ work: @Sendable () throws -> T
    ) rethrows -> T {
        try work()
    }
}
