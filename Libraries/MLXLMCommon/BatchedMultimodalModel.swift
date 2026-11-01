// Copyright © 2026 Apple Inc.

import Foundation

/// Media inputs whose prepared caches a model can safely adopt into ragged decoding.
public struct BatchedMediaCapabilities: OptionSet, Sendable {
    public let rawValue: Int

    public init(rawValue: Int) { self.rawValue = rawValue }

    public static let image = Self(rawValue: 1 << 0)
    public static let video = Self(rawValue: 1 << 1)
    public static let audio = Self(rawValue: 1 << 2)
}

/// A model validated for ragged decoding after independently preparing media requests.
///
/// Conforming models keep request-specific continuation data in their caches. Their
/// media `prepare` returns logits without a separate `LMOutput.State`, and the caches
/// support `makeBatchedCacheFactories(for:)`, row filtering, extraction and extension.
/// Media encoders may run separately before those caches join a running decode batch.
public protocol BatchedMultimodalModel: LanguageModel {
    var batchedMediaCapabilities: BatchedMediaCapabilities { get }

    /// Negative placeholders introduced by this model's processor, replaced before
    /// embedding lookup. Admission can validate the remaining token IDs normally.
    var batchTokenPlaceholderIDs: Set<Int> { get }
}

extension BatchedMultimodalModel {
    public var batchTokenPlaceholderIDs: Set<Int> { [] }
}
