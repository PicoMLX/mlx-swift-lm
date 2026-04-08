// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite(
    "Model Container Batching",
    .enabled(if: MLXMetalGuard.isAvailable, MLXMetalGuard.unavailableComment),
    .serialized
)
struct ModelContainerBatchingTests {

    private func makeContainer(
        scheduler: InferenceScheduler? = nil,
        promptCache: LRUPromptCache? = nil,
        loadedAsVLM: Bool = false,
        configurationID: String = "test-model",
        callDelay: TimeInterval = 0
    ) -> (container: ModelContainer, model: CallTrackingModel, configuration: ModelConfiguration) {
        let model = CallTrackingModel(vocabSize: 32, numLayers: 1, callDelay: callDelay)
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: configurationID)
        let processor = ContainerInputProcessor(
            tokenizer: tokenizer,
            configuration: configuration
        )

        let context = ModelContext(
            configuration: configuration,
            model: model,
            processor: processor,
            tokenizer: tokenizer,
            loadedAsVLM: loadedAsVLM
        )

        let container = ModelContainer(context: context)
        container.scheduler = scheduler
        container.promptCache = promptCache

        return (container, model, configuration)
    }

    @Test("Without a scheduler the container stays on the direct path")
    func directPathWithoutSchedulerProducesOutput() async throws {
        let (container, _, _) = makeContainer()
        #expect(container.scheduler == nil)

        let stream = try await container.generate(
            input: LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)])),
            parameters: GenerateParameters(maxTokens: 3, temperature: 0)
        )

        let collected = await collectGenerationOutput(stream)
        let info = try #require(collected.info)
        #expect(collected.text.isEmpty == false)
        #expect(info.generationTokenCount == 3)
    }

    @Test("Scheduler-backed containers route through the scheduler")
    func schedulerBackedContainerRoutesThroughScheduler() async throws {
        let scheduler = InferenceScheduler()
        let (container, _, _) = makeContainer(scheduler: scheduler, callDelay: 0.002)

        let stream = try await container.generate(
            input: LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)])),
            parameters: GenerateParameters(maxTokens: 4, temperature: 0)
        )

        #expect(await waitForSchedulerState(scheduler, equals: "single"))

        let collected = await collectGenerationOutput(stream)
        #expect(collected.text.isEmpty == false)
    }

    @Test("VLM-loaded containers bypass the scheduler even when one is configured")
    func vlmLoadedContainersBypassScheduler() async throws {
        let scheduler = InferenceScheduler()
        let (container, _, _) = makeContainer(
            scheduler: scheduler,
            loadedAsVLM: true
        )

        let stream = try await container.generate(
            input: LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)])),
            parameters: GenerateParameters(maxTokens: 3, temperature: 0)
        )

        let collected = await collectGenerationOutput(stream)
        #expect(collected.text.isEmpty == false)
        #expect(await scheduler.currentState == "idle")
    }

    @Test("Staggered concurrent requests keep independent streams in batch mode")
    func staggeredConcurrentRequestsKeepIndependentStreams() async throws {
        let scheduler = InferenceScheduler()
        let (container, _, _) = makeContainer(scheduler: scheduler, callDelay: 0.002)

        let firstTask = Task {
            let stream = try await container.generate(
                input: LMInput(tokens: MLXArray([Int32(1), Int32(2)])),
                parameters: GenerateParameters(maxTokens: 2, temperature: 0)
            )
            return await collectGenerationOutput(stream)
        }

        let secondTask = Task {
            try? await Task.sleep(nanoseconds: 10_000_000)
            let stream = try await container.generate(
                input: LMInput(tokens: MLXArray([Int32(8), Int32(9)])),
                parameters: GenerateParameters(maxTokens: 5, temperature: 0)
            )
            return await collectGenerationOutput(stream)
        }

        let results = try await (firstTask.value, secondTask.value)
        let info1 = try #require(results.0.info)
        let info2 = try #require(results.1.info)

        #expect(results.0.text.isEmpty == false)
        #expect(results.1.text.isEmpty == false)
        #expect(info1.generationTokenCount == 2)
        #expect(info2.generationTokenCount == 5)
    }

    @Test("Cancelling one scheduler-backed stream does not stop the other request")
    func cancellingOneStreamDoesNotStopTheOther() async throws {
        let scheduler = InferenceScheduler()
        let (container, _, _) = makeContainer(scheduler: scheduler, callDelay: 0.002)

        let stream1 = try await container.generateTokens(
            input: LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)])),
            parameters: GenerateParameters(maxTokens: 8, temperature: 0)
        )
        let stream2 = try await container.generateTokens(
            input: LMInput(tokens: MLXArray([Int32(9), Int32(10)])),
            parameters: GenerateParameters(maxTokens: 8, temperature: 0)
        )

        #expect(await waitForSchedulerState(scheduler, equals: "batched"))

        let firstTask = Task {
            await collectTokenOutput(stream1)
        }

        let sawSecondToken = AsyncFlagContainer()
        let secondTask = Task { () -> [Int] in
            var tokens = [Int]()
            for await generation in stream2 {
                if case .token(let token) = generation {
                    tokens.append(token)
                    await sawSecondToken.set()
                }
            }
            return tokens
        }

        for _ in 0 ..< 50 {
            if await sawSecondToken.get() {
                break
            }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }

        secondTask.cancel()
        let firstResult = await firstTask.value
        let secondTokens = await secondTask.value

        let firstInfo = try #require(firstResult.info)
        #expect(firstInfo.generationTokenCount == 8)
        #expect(secondTokens.count < 8)
    }

    @Test("Prompt cache wiring reduces work on repeated scheduler requests")
    func promptCacheIsWiredIntoSchedulerPath() async throws {
        let scheduler = InferenceScheduler()
        let promptCache = LRUPromptCache(maxSize: 10)
        let (container, model, _) = makeContainer(
            scheduler: scheduler,
            promptCache: promptCache
        )
        let prompt = [1, 2, 3, 4, 5]

        let firstStream = try await container.generate(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))),
            parameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )
        _ = await collectGenerationOutput(firstStream)
        let firstTotal = model.totalTokensProcessed

        model.resetCounters()

        let secondStream = try await container.generate(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))),
            parameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )
        _ = await collectGenerationOutput(secondStream)
        let secondTotal = model.totalTokensProcessed

        #expect(model.sawPreloadedCache)
        #expect(secondTotal < firstTotal)
    }

    @Test("Scheduler and prompt cache properties remain assignable on the container")
    func schedulerAndPromptCachePropertiesRemainAssignable() {
        let (container, _, _) = makeContainer()
        let scheduler = InferenceScheduler()
        let promptCache = LRUPromptCache(maxSize: 4)

        container.scheduler = scheduler
        container.promptCache = promptCache

        #expect(container.scheduler != nil)
        #expect(container.promptCache === promptCache)
    }

    @Test("ChatSession keeps working on the scheduler-backed path")
    func chatSessionUsesSchedulerAcrossTurns() async throws {
        let scheduler = InferenceScheduler()
        let (container, _, _) = makeContainer(scheduler: scheduler, callDelay: 0.001)
        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )

        let first = try await session.respond(to: "First message")
        let second = try await session.respond(to: "Second message")

        #expect(first.isEmpty == false)
        #expect(second.isEmpty == false)
    }
}

private func appendContainerSyntheticKV(
    to caches: [KVCache]?,
    inputTokens: MLXArray,
    defaultHeads: Int = 4,
    defaultHeadDim: Int = 8
) {
    guard let caches else { return }

    let batchSize = inputTokens.dim(0)
    let seqLen = inputTokens.dim(1)

    for (layerIndex, cache) in caches.enumerated() {
        let state = cache.innerState()
        let existingKeys = state.first
        let existingValues = state.count > 1 ? state[1] : nil

        let heads = existingKeys?.dim(1) ?? defaultHeads
        let keyDim = existingKeys?.dim(3) ?? defaultHeadDim
        let valueDim = existingValues?.dim(3) ?? keyDim

        let baseValue = Float(layerIndex + 1)
        let keys = MLXArray.ones([batchSize, heads, seqLen, keyDim]) * baseValue
        let values = MLXArray.ones([batchSize, heads, seqLen, valueDim]) * (baseValue + 1)
        _ = cache.update(keys: keys, values: values)
    }
}

private struct ContainerInputProcessor: UserInputProcessor {
    let tokenizer: Tokenizer
    let configuration: ModelConfiguration

    var messageGenerator: MessageGenerator { DefaultMessageGenerator() }

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )
        return LMInput(tokens: MLXArray(promptTokens))
    }
}

private final class CallTrackingModel: Module, LanguageModel, KVCacheDimensionProvider,
    @unchecked Sendable
{
    let vocabSize: Int
    let numLayers: Int
    let callDelay: TimeInterval

    var kvHeads: [Int] { Array(repeating: 4, count: numLayers) }

    var callCount = 0
    var totalTokensProcessed = 0
    var inputShapes = [[Int]]()
    var sawPreloadedCache = false

    init(vocabSize: Int = 32, numLayers: Int = 1, callDelay: TimeInterval = 0) {
        self.vocabSize = vocabSize
        self.numLayers = numLayers
        self.callDelay = callDelay
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let cachedLength = cache.first?.offset ?? 0
        let promptLength = input.text.tokens.size

        if cachedLength >= promptLength, promptLength > 0 {
            _ = trimPromptCache(cache, numTokens: 1)
            return .tokens(input.text[(promptLength - 1)...])
        }

        if cachedLength > 0 {
            return .tokens(input.text[cachedLength...])
        }

        return .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        if callDelay > 0 {
            Thread.sleep(forTimeInterval: callDelay)
        }

        callCount += 1
        inputShapes.append(input.tokens.shape)
        totalTokensProcessed += input.tokens.dim(0) * input.tokens.dim(1)

        if let cache {
            let hasPreloadedKeys = cache.contains { layer in
                layer.innerState().first != nil
            }
            sawPreloadedCache = sawPreloadedCache || hasPreloadedKeys
        }

        appendContainerSyntheticKV(to: cache, inputTokens: input.tokens)

        let tokens = input.tokens
        let batchSize = tokens.dim(0)
        let steps = tokens.dim(1)
        var logitsFlat = [Float]()
        logitsFlat.reserveCapacity(batchSize * steps * vocabSize)

        for batch in 0 ..< batchSize {
            for step in 0 ..< steps {
                let token = Int(tokens[batch, step].item(Int32.self))
                let predictedToken = (token + 1) % vocabSize
                var row = [Float](repeating: -100, count: vocabSize)
                row[predictedToken] = 0
                logitsFlat.append(contentsOf: row)
            }
        }

        return LMOutput(logits: MLXArray(logitsFlat, [batchSize, steps, vocabSize]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< numLayers).map { _ in KVCacheSimple() }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    func resetCounters() {
        callCount = 0
        totalTokensProcessed = 0
        inputShapes = []
        sawPreloadedCache = false
    }
}

private struct CollectedGenerationOutput {
    let text: String
    let info: GenerateCompletionInfo?
}

private struct CollectedTokenOutput {
    let tokens: [Int]
    let info: GenerateCompletionInfo?
}

private actor AsyncFlagContainer {
    private var value = false

    func set() {
        value = true
    }

    func get() -> Bool {
        value
    }
}

private func collectGenerationOutput(_ stream: AsyncStream<Generation>) async
    -> CollectedGenerationOutput
{
    var text = ""
    var info: GenerateCompletionInfo?

    for await generation in stream {
        switch generation {
        case .chunk(let chunk):
            text += chunk
        case .info(let completionInfo):
            info = completionInfo
        case .toolCall:
            break
        }
    }

    return CollectedGenerationOutput(text: text, info: info)
}

private func collectTokenOutput(_ stream: AsyncStream<TokenGeneration>) async -> CollectedTokenOutput
{
    var tokens = [Int]()
    var info: GenerateCompletionInfo?

    for await generation in stream {
        switch generation {
        case .token(let token):
            tokens.append(token)
        case .info(let completionInfo):
            info = completionInfo
        }
    }

    return CollectedTokenOutput(tokens: tokens, info: info)
}

private func waitForSchedulerState(
    _ scheduler: InferenceScheduler,
    equals expected: String,
    attempts: Int = 100,
    sleepNanoseconds: UInt64 = 10_000_000
) async -> Bool {
    for _ in 0 ..< attempts {
        if await scheduler.currentState == expected {
            return true
        }
        try? await Task.sleep(nanoseconds: sleepNanoseconds)
    }
    return false
}
