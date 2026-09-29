import MLX
import Testing

@testable import MLXLMCommon

struct GenerationStreamTests {
    @Test
    func tracksActualStreamsAcrossTaskScopes() {
        let tracker = GenerationStreamTracker()
        let first = Stream.withNewDefaultStream(device: .gpu) {
            tracker.capture()
            let stream = Stream.defaultStream
            asyncEval(MLXArray.ones([32, 32]) + 1)
            return stream
        }
        let second = Stream.withNewDefaultStream(device: .cpu) {
            tracker.capture()
            let stream = Stream.defaultStream
            asyncEval(MLXArray.ones([32, 32]) + 2)
            return stream
        }
        #expect(first != second)
        #expect(first != Stream.defaultStream)
        #expect(tracker.trackedStreams.contains(first))
        #expect(tracker.trackedStreams.contains(second))
        let count = tracker.trackedStreams.count
        tracker.capture()
        #expect(tracker.trackedStreams.count == count)
        tracker.synchronize()
    }
}
