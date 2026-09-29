// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Tracks the streams used by serial inference across task-local scopes.
/// Call `capture()` before submitting work and `synchronize()` after the producer stops.
public final class GenerationStreamTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var streams: [MLX.Stream] = []

    public init() {
        capture()
    }

    public func capture() {
        let current = [MLX.Stream.defaultStream, MLX.Stream.gpu, MLX.Stream.cpu]
        lock.lock()
        defer { lock.unlock() }
        for stream in current where !streams.contains(stream) {
            streams.append(stream)
        }
    }

    var trackedStreams: [MLX.Stream] {
        lock.lock()
        defer { lock.unlock() }
        return streams
    }

    /// The owner must serialize this with inference submission.
    public func synchronize() {
        for stream in trackedStreams {
            stream.synchronize()
        }
    }
}
