import MLX
import MLXLMCommon
import Testing

struct MediaPlaceholderPenaltyTests {
    @Test(
        "Image placeholders do not penalize vocabulary entries", arguments: [0, 1, 2], [0, 1, 2])
    func placeholderPenalty(kind: Int, promptCase: Int) {
        Device.withDefaultDevice(.cpu) {
            var actual: any LogitProcessor
            var expected: any LogitProcessor
            switch kind {
            case 0:
                actual = RepetitionContext(repetitionPenalty: 2, repetitionContextSize: 8)
                expected = RepetitionContext(repetitionPenalty: 2, repetitionContextSize: 8)
            case 1:
                actual = PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 8)
                expected = PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 8)
            default:
                actual = FrequencyPenaltyContext(frequencyPenalty: 0.5, frequencyContextSize: 8)
                expected = FrequencyPenaltyContext(frequencyPenalty: 0.5, frequencyContextSize: 8)
            }
            let prompt: [Int]
            let text: [Int]
            switch promptCase {
            case 0:
                prompt = [1, -200, 2, 1]
                text = [1, 2, 1]
            case 1:
                prompt = [-200, -200]
                text = []
            default:
                prompt = [0, 1, 2, 3, -200, 4, 5, 6, 7, -200, 8, 9]
                text = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            }
            actual.prompt(MLXArray(prompt).reshaped([1, prompt.count]))
            expected.prompt(MLXArray(text))
            let logits = MLXArray((0 ..< 16).map { Float($0) - 8 }).reshaped([1, 16])
            for token in [0, 1, 5, 7, 2, 8, 9, 10, 11] {
                #expect(
                    actual.process(logits: logits).asArray(Float.self)
                        == expected.process(logits: logits).asArray(Float.self))
                actual.didSample(token: MLXArray(token))
                expected.didSample(token: MLXArray(token))
            }
        }
    }
}
