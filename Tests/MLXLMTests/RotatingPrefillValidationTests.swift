// Copyright © 2026 Apple Inc.

import MLXLMCommon
import Testing

@Suite struct RotatingPrefillValidationTests {
    @Test func rightPaddingRequiresLengths() async {
        await #expect(processExitsWith: .failure) {
            let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 0])
            cache.prepare(rightPadding: [0, 1])
        }
    }

    @Test func batchedEntryPointRequiresLengths() async {
        await #expect(processExitsWith: .failure) {
            let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 0])
            cache.prepareBatched(leftPadding: [0, 0], lengths: nil, rightPadding: [1, 0])
        }
    }

    @Test func zeroRightPaddingDoesNotRequireLengths() {
        let cache = BatchRotatingKVCache(maxSize: 8, leftPadding: [0, 0])
        cache.prepare(rightPadding: [0, 0])
        cache.finalize()
        #expect(cache.offset == 0)
    }
}
