import MLXLMCommon
import Testing
import os

struct PrefillCancellationTests {
    @Test("Prefill defaults to the driving task's cancellation")
    func defaultCancellation() async {
        await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            var chunks: [Range<Int>] = []
            #expect(throws: CancellationError.self) {
                try PrefillParameters(stepSize: 2).forEachChunk(total: 8) { chunks.append($0) }
            }
            #expect(chunks.isEmpty)
        }.value
    }

    @Test("A live request can prefill on a cancelled driving task")
    func independentRequest() async throws {
        var prefill = PrefillParameters(stepSize: 2)
        prefill.cancellationCheck = {}
        let parameters = prefill
        let chunks = try await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            var chunks: [Range<Int>] = []
            let processed = try parameters.forEachChunk(total: 8) { chunks.append($0) }
            #expect(processed == 7)
            return chunks
        }.value
        #expect(chunks == [0 ..< 2, 2 ..< 4, 4 ..< 6, 6 ..< 7])
    }

    @Test("Request cancellation stops between chunks without cancelling the driver")
    func requestCancellation() throws {
        let cancelled = OSAllocatedUnfairLock(initialState: false)
        var prefill = PrefillParameters(
            stepSize: 2,
            progress: { _, _ in
                cancelled.withLock { $0 = true }
            })
        prefill.cancellationCheck = {
            if cancelled.withLock({ $0 }) { throw CancellationError() }
        }
        var chunks: [Range<Int>] = []
        #expect(throws: CancellationError.self) {
            try prefill.forEachChunk(total: 8) { chunks.append($0) }
        }
        #expect(chunks == [0 ..< 2])
        #expect(!Task.isCancelled)
    }
}
