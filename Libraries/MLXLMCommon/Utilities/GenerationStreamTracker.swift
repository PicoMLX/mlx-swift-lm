// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The streams a generation submitted work to, across task-local stream scopes.
///
/// Call `capture()` where work is submitted and `synchronize()` once it stops.
package final class GenerationStreamTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var streams: [MLX.Stream] = []

    package init() {
        capture()
    }

    package func capture() {
        let current = [MLX.Stream.defaultStream, MLX.Stream.gpu, MLX.Stream.cpu]
        lock.withLock {
            for stream in current where !streams.contains(stream) {
                streams.append(stream)
            }
        }
    }

    var trackedStreams: [MLX.Stream] {
        lock.withLock { streams }
    }

    /// The caller serializes this with the work it waits for.
    package func synchronize() {
        for stream in trackedStreams {
            stream.synchronize()
        }
    }
}
