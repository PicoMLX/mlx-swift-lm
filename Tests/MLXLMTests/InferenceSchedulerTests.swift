// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@preconcurrency @testable import MLXLMCommon
import MLXNN
import Testing

@Suite("Scheduler Token Handler")
struct SchedulerTokenHandlerTests {

    @Test("Raw token handler emits tokens, stop token, and info")
    func rawTokenHandlerEmitsTokensAndInfo() async throws {
        let (stream, continuation) = AsyncStream<TokenGeneration>.makeStream()
        let handler = SchedulerTokenHandler.rawToken(
            continuation: continuation,
            includeStopToken: true
        )

        #expect(handler.processToken(4))
        #expect(handler.processStopToken(8))
        handler.yieldInfo(
            GenerateCompletionInfo(
                promptTokenCount: 2,
                generationTokenCount: 2,
                promptTime: 0.1,
                generationTime: 0.2,
                stopReason: .stop
            )
        )
        handler.finish()

        let collected = await collectTokenGenerations(stream)
        let info = try #require(collected.info)
        #expect(collected.tokens == [4, 8])
        #expect(info.generationTokenCount == 2)
        #expect(info.stopReason == .stop)
    }

    @Test("Text handler emits decoded chunks and completion info")
    func textHandlerEmitsChunksAndInfo() async throws {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let handler = SchedulerTokenHandler.text(
            continuation: continuation,
            tokenizer: TestTokenizer(vocabularySize: 20),
            toolCallFormat: .json
        )

        #expect(handler.processToken(1))
        handler.processEndOfSequence()
        handler.yieldInfo(
            GenerateCompletionInfo(
                promptTokenCount: 1,
                generationTokenCount: 1,
                promptTime: 0.05,
                generationTime: 0.05,
                stopReason: .length
            )
        )
        handler.finish()

        let collected = await collectGenerations(stream)
        let info = try #require(collected.info)
        #expect(collected.chunks.isEmpty == false)
        #expect(collected.chunks.joined().isEmpty == false)
        #expect(info.stopReason == .length)
    }
}

@Suite(
    "Inference Scheduler",
    .enabled(if: MLXMetalGuard.isAvailable, MLXMetalGuard.unavailableComment),
    .serialized
)
struct InferenceSchedulerTests {

    private func submitRaw(
        scheduler: InferenceScheduler,
        prompt: [Int],
        parameters: GenerateParameters,
        model: any LanguageModel,
        tokenizer: Tokenizer,
        configuration: ModelConfiguration,
        promptCache: LRUPromptCache? = nil,
        wiredMemoryTicket: WiredMemoryTicket? = nil
    ) async throws -> AsyncStream<TokenGeneration> {
        try await scheduler.submitTokens(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))),
            parameters: parameters,
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: configuration,
            promptCache: promptCache,
            promptCacheModelName: configuration.name,
            inputTokens: prompt,
            wiredMemoryTicket: wiredMemoryTicket
        )
    }

    @Test("Single request uses the single path and reports completion info")
    func singleRequestUsesSinglePathAndReportsInfo() async throws {
        let scheduler = InferenceScheduler()
        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "scheduler-single")

        let stream = try await scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)])),
            parameters: GenerateParameters(maxTokens: 3, temperature: 0),
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: configuration
        )

        #expect(await scheduler.currentState == "single")

        let collected = await collectGenerations(stream)
        let info = try #require(collected.info)
        #expect(collected.chunks.isEmpty == false)
        #expect(info.promptTokenCount == 3)
        #expect(info.generationTokenCount == 3)
        #expect(info.stopReason == .length)

        _ = await waitForState(scheduler, equals: "idle")
        #expect(await scheduler.currentState == "idle")
    }

    @Test("Compatible second request upgrades to batch and preserves per-request metadata")
    func compatibleSecondRequestUpgradesToBatchAndPreservesMetadata() async throws {
        let scheduler = InferenceScheduler(
            configuration: .init(
                completionBatchSize: 4,
                prefillBatchSize: 2,
                prefillStepSize: 32,
                cacheClearInterval: 4
            )
        )
        let model = SchedulerMockModel(callDelay: 0.002)
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "scheduler-batch")

        let stream1 = try await scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)])),
            parameters: GenerateParameters(maxTokens: 6, temperature: 0),
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: configuration
        )

        let stream2 = try await scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(4), Int32(5)])),
            parameters: GenerateParameters(maxTokens: 4, temperature: 0),
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: configuration
        )

        #expect(await waitForState(scheduler, equals: "batched"))

        let iteratorConfig = try #require(await scheduler.activeBatchIteratorConfiguration)
        #expect(iteratorConfig.completionBatchSize == 4)
        #expect(iteratorConfig.prefillBatchSize == 2)
        #expect(iteratorConfig.prefillStepSize == 32)

        async let result1 = collectGenerations(stream1)
        async let result2 = collectGenerations(stream2)
        let (collected1, collected2) = await (result1, result2)

        let info1 = try #require(collected1.info)
        let info2 = try #require(collected2.info)
        #expect(info1.promptTokenCount == 3)
        #expect(info2.promptTokenCount == 2)
        #expect(info1.promptTime > 0)
        #expect(info2.promptTime > 0)
    }

    @Test("Requests with different maxKVSize do not share a batch")
    func differentMaxKVSizeRequestsDoNotBatch() async throws {
        let scheduler = InferenceScheduler()
        let model = SchedulerMockModel(callDelay: 0.004)
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "scheduler-max-kv")

        let stream1 = try await scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)])),
            parameters: GenerateParameters(maxTokens: 20, maxKVSize: 4, temperature: 0),
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: configuration
        )

        let stream2 = try await scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(9), Int32(10)])),
            parameters: GenerateParameters(maxTokens: 3, maxKVSize: 8, temperature: 0),
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: configuration
        )

        try? await Task.sleep(nanoseconds: 50_000_000)
        #expect(await scheduler.activeBatchIteratorConfiguration == nil)

        async let result1 = collectGenerations(stream1)
        async let result2 = collectGenerations(stream2)
        _ = await (result1, result2)
    }

    @Test("Batch compatibility rejects unsupported inputs and cache types")
    func batchCompatibilityRejectsUnsupportedInputsAndCacheTypes() {
        let standardModel = SchedulerMockModel()
        let ssmModel = SSMSchedulerMockModel()
        let textInput = LMInput(tokens: MLXArray([Int32(1), Int32(2)]))
        let imageInput = LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            image: .init(pixels: MLXArray.zeros([1, 3, 224, 224]))
        )
        let videoInput = LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            video: .init(pixels: MLXArray.zeros([1, 3, 8, 224, 224]))
        )

        #expect(
            InferenceScheduler.isBatchCompatible(
                input: textInput,
                parameters: GenerateParameters(temperature: 0),
                cache: nil,
                model: standardModel
            )
        )
        #expect(
            InferenceScheduler.isBatchCompatible(
                input: imageInput,
                parameters: GenerateParameters(temperature: 0),
                cache: nil,
                model: standardModel
            ) == false
        )
        #expect(
            InferenceScheduler.isBatchCompatible(
                input: videoInput,
                parameters: GenerateParameters(temperature: 0),
                cache: nil,
                model: standardModel
            ) == false
        )
        #expect(
            InferenceScheduler.isBatchCompatible(
                input: textInput,
                parameters: GenerateParameters(kvBits: 4, temperature: 0),
                cache: nil,
                model: standardModel
            ) == false
        )
        #expect(
            InferenceScheduler.isBatchCompatible(
                input: textInput,
                parameters: GenerateParameters(temperature: 0),
                cache: [CacheList(KVCacheSimple(), KVCacheSimple())],
                model: standardModel
            ) == false
        )
        #expect(
            InferenceScheduler.isBatchCompatible(
                input: textInput,
                parameters: GenerateParameters(temperature: 0),
                cache: nil,
                model: ssmModel
            ) == false
        )
    }

    @Test("Single-path prompt cache write-back stores the generated sequence")
    func singlePathWritesBackToPromptCache() async throws {
        let scheduler = InferenceScheduler()
        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "single-writeback")
        let promptCache = LRUPromptCache(maxSize: 10)
        let prompt = [1, 2, 3]

        let stream = try await submitRaw(
            scheduler: scheduler,
            prompt: prompt,
            parameters: GenerateParameters(maxTokens: 2, temperature: 0),
            model: model,
            tokenizer: tokenizer,
            configuration: configuration,
            promptCache: promptCache
        )

        let collected = await collectTokenGenerations(stream)
        let fullSequence = prompt + collected.tokens
        let (cached, remainder) = promptCache.fetchNearestCache(
            model: configuration.name,
            tokens: fullSequence
        )

        let firstLayer = try #require(cached?.first)
        #expect(remainder.isEmpty)
        #expect(firstLayer.offset == fullSequence.count)
    }

    @Test("Single path uses cached prefix remainder instead of replaying the full prompt")
    func singlePathUsesCachedPrefixRemainder() async throws {
        let scheduler = InferenceScheduler()
        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "single-cache-remainder")
        let promptCache = LRUPromptCache(maxSize: 10)
        let cachedPrefix = [1, 2, 3]
        let prompt = [1, 2, 3, 4, 5]

        promptCache.insertCache(
            model: configuration.name,
            tokens: cachedPrefix,
            promptCache: makeSchedulerPromptCache(seqLen: cachedPrefix.count)
        )

        let (cached, remainder) = promptCache.fetchNearestCache(
            model: configuration.name,
            tokens: prompt
        )
        let cachedState = try #require(cached)
        #expect(remainder == [4, 5])

        let stream = try await scheduler.submitTokens(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))),
            parameters: GenerateParameters(maxTokens: 1, temperature: 0),
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: configuration,
            cachedKVState: cachedState,
            cachedPromptRemainder: remainder,
            promptCache: promptCache,
            promptCacheModelName: configuration.name,
            inputTokens: prompt
        )

        _ = await collectTokenGenerations(stream)

        #expect(model.inputShapes.first == [1, 2])
        #expect(model.inputShapes.contains([1, prompt.count]) == false)
    }

    @Test("Text-mode stop-token write-back includes the stop token in the cache key")
    func textModeStopTokenWriteBackIncludesStopToken() async throws {
        let scheduler = InferenceScheduler()
        let model = SchedulerMockModel(vocabSize: 128)
        let tokenizer = TestTokenizer(vocabularySize: 128)
        var configuration = ModelConfiguration(id: "text-stop-writeback")
        let promptCache = LRUPromptCache(maxSize: 10)
        let prompt = [100]
        let stopToken = try #require(tokenizer.eosTokenId)
        configuration.eosTokenIds = [stopToken]

        let stream = try await scheduler.submit(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))),
            parameters: GenerateParameters(maxTokens: 4, temperature: 0),
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: configuration,
            promptCache: promptCache,
            promptCacheModelName: configuration.name,
            inputTokens: prompt
        )

        let collected = await collectGenerations(stream)
        let info = try #require(collected.info)
        let expectedKey = prompt + [stopToken]
        let (cached, remainder) = promptCache.fetchNearestCache(
            model: configuration.name,
            tokens: expectedKey
        )
        let firstLayer = try #require(cached?.first)

        #expect(info.generationTokenCount == 0)
        #expect(remainder.isEmpty)
        #expect(firstLayer.offset == expectedKey.count)
    }

    @Test("Raw-token stop-token write-back remains aligned when stop token is emitted")
    func rawTokenStopTokenWriteBackIncludesEmittedStopToken() async throws {
        let scheduler = InferenceScheduler()
        let model = SchedulerMockModel(vocabSize: 128)
        let tokenizer = TestTokenizer(vocabularySize: 128)
        var configuration = ModelConfiguration(id: "raw-stop-writeback")
        let promptCache = LRUPromptCache(maxSize: 10)
        let prompt = [100]
        let stopToken = try #require(tokenizer.eosTokenId)
        configuration.eosTokenIds = [stopToken]

        let stream = try await scheduler.submitTokens(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))),
            parameters: GenerateParameters(maxTokens: 4, temperature: 0),
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: configuration,
            includeStopToken: true,
            promptCache: promptCache,
            promptCacheModelName: configuration.name,
            inputTokens: prompt
        )

        let collected = await collectTokenGenerations(stream)
        let info = try #require(collected.info)
        let expectedKey = prompt + [stopToken]
        let (cached, remainder) = promptCache.fetchNearestCache(
            model: configuration.name,
            tokens: expectedKey
        )
        let firstLayer = try #require(cached?.first)

        #expect(collected.tokens == [stopToken])
        #expect(info.generationTokenCount == 1)
        #expect(remainder.isEmpty)
        #expect(firstLayer.offset == expectedKey.count)
    }

    @Test("Cached prefixes are ignored for incompatible single-path fallbacks")
    func cachedPrefixesAreIgnoredForIncompatibleFallbacks() async throws {
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "incompatible-cache-fallback")
        let prompt = [1, 2, 3]

        do {
            let scheduler = InferenceScheduler()
            let model = SchedulerMockModel(syntheticHeadDim: 32)
            let stream = try await scheduler.submitTokens(
                input: LMInput(tokens: MLXArray(prompt.map(Int32.init))),
                parameters: GenerateParameters(
                    maxTokens: 0,
                    kvBits: 4,
                    kvGroupSize: 32,
                    temperature: 0
                ),
                model: model,
                cache: nil,
                tokenizer: tokenizer,
                configuration: configuration,
                cachedKVState: makeSchedulerPromptCache(seqLen: 2),
                cachedPromptRemainder: [3],
                inputTokens: prompt
            )

            _ = await collectTokenGenerations(stream)
            #expect(model.inputShapes.first == [1, prompt.count])
        }

        do {
            let scheduler = InferenceScheduler()
            let model = SchedulerMockModel()
            let imageInput = LMInput(
                text: .init(tokens: MLXArray(prompt.map(Int32.init))),
                image: .init(pixels: MLXArray.zeros([1, 3, 8, 8]))
            )
            let stream = try await scheduler.submitTokens(
                input: imageInput,
                parameters: GenerateParameters(maxTokens: 1, temperature: 0),
                model: model,
                cache: nil,
                tokenizer: tokenizer,
                configuration: configuration,
                cachedKVState: makeSchedulerPromptCache(seqLen: 2),
                cachedPromptRemainder: [3],
                inputTokens: prompt
            )

            _ = await collectTokenGenerations(stream)
            #expect(model.inputShapes.first == [1, prompt.count])
        }

        do {
            let scheduler = InferenceScheduler()
            let model = SchedulerMockModel()
            let quantizedCache = QuantizedKVCache()
            quantizedCache.offset = 2
            let stream = try await scheduler.submitTokens(
                input: LMInput(tokens: MLXArray(prompt.map(Int32.init))),
                parameters: GenerateParameters(maxTokens: 1, temperature: 0),
                model: model,
                cache: nil,
                tokenizer: tokenizer,
                configuration: configuration,
                cachedKVState: [quantizedCache],
                cachedPromptRemainder: [3],
                inputTokens: prompt
            )

            _ = await collectTokenGenerations(stream)
            #expect(model.inputShapes.first == [1, prompt.count])
        }
    }

    @Test("Batch-path prompt cache write-back stores both generated sequences")
    func batchPathWritesBackToPromptCache() async throws {
        let scheduler = InferenceScheduler()
        let model = SchedulerMockModel(callDelay: 0.002)
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "batch-writeback")
        let promptCache = LRUPromptCache(maxSize: 10)
        let promptA = [1, 2, 3]
        let promptB = [10, 11]

        let streamA = try await submitRaw(
            scheduler: scheduler,
            prompt: promptA,
            parameters: GenerateParameters(maxTokens: 3, temperature: 0),
            model: model,
            tokenizer: tokenizer,
            configuration: configuration,
            promptCache: promptCache
        )

        let streamB = try await submitRaw(
            scheduler: scheduler,
            prompt: promptB,
            parameters: GenerateParameters(maxTokens: 2, temperature: 0),
            model: model,
            tokenizer: tokenizer,
            configuration: configuration,
            promptCache: promptCache
        )

        #expect(await waitForState(scheduler, equals: "batched"))

        async let resultA = collectTokenGenerations(streamA)
        async let resultB = collectTokenGenerations(streamB)
        let (collectedA, collectedB) = await (resultA, resultB)

        let fullA = promptA + collectedA.tokens
        let fullB = promptB + collectedB.tokens

        let (cachedA, remainderA) = promptCache.fetchNearestCache(
            model: configuration.name,
            tokens: fullA
        )
        let (cachedB, remainderB) = promptCache.fetchNearestCache(
            model: configuration.name,
            tokens: fullB
        )
        let firstLayerA = try #require(cachedA?.first)
        let firstLayerB = try #require(cachedB?.first)

        #expect(remainderA.isEmpty)
        #expect(remainderB.isEmpty)
        #expect(firstLayerA.offset == fullA.count)
        #expect(firstLayerB.offset == fullB.count)
    }

    @Test("Single-path wired-memory tickets start and end exactly once")
    func singlePathStartsAndEndsWiredMemoryTicket() async throws {
        let manager = makeWiredMemoryManager()
        let (recorder, recorderTask) = startRecording(manager: manager)
        defer { recorderTask.cancel() }

        let policy = WiredSumPolicy(cap: 200)
        let ticket = policy.ticket(size: 40, manager: manager)
        let scheduler = InferenceScheduler()
        let model = SchedulerMockModel()
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "wired-single")

        let stream = try await scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)])),
            parameters: GenerateParameters(maxTokens: 4, temperature: 0),
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: configuration,
            wiredMemoryTicket: ticket
        )

        _ = await collectGenerations(stream)
        await settleEvents()

        let events = await recorder.snapshot()
        #expect(ticketEvents(events, ticket: ticket, kind: .ticketStarted).count == 1)
        #expect(ticketEvents(events, ticket: ticket, kind: .ticketEnded).count == 1)
    }

    @Test("Waiting second ticket does not stall the active request")
    func waitingSecondTicketDoesNotStallActiveRequest() async throws {
        let manager = makeWiredMemoryManager(baseline: 100)
        let (recorder, recorderTask) = startRecording(manager: manager)
        defer { recorderTask.cancel() }

        let policy = WiredSumPolicy(cap: 140)
        let blockerTicket = policy.ticket(size: 30, manager: manager)
        let firstTicket = policy.ticket(size: 10, manager: manager)
        let secondTicket = policy.ticket(size: 20, manager: manager)

        let scheduler = InferenceScheduler()
        let model = SchedulerMockModel(callDelay: 0.002)
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "wired-wait")

        var blockerReleased = false
        _ = await blockerTicket.start()
        defer {
            if !blockerReleased {
                Task { _ = await blockerTicket.end() }
            }
        }

        let stream1 = try await scheduler.submit(
            input: LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)])),
            parameters: GenerateParameters(maxTokens: 20, temperature: 0),
            model: model,
            cache: nil,
            tokenizer: tokenizer,
            configuration: configuration,
            wiredMemoryTicket: firstTicket
        )

        let secondReturned = AsyncFlag()
        let secondTask = Task<Void, Error> {
            let stream2 = try await scheduler.submit(
                input: LMInput(tokens: MLXArray([Int32(11), Int32(12)])),
                parameters: GenerateParameters(maxTokens: 4, temperature: 0),
                model: model,
                cache: nil,
                tokenizer: tokenizer,
                configuration: configuration,
                wiredMemoryTicket: secondTicket
            )
            await secondReturned.set()
            _ = await collectGenerations(stream2)
        }
        defer { secondTask.cancel() }

        try? await Task.sleep(nanoseconds: 50_000_000)
        #expect(await secondReturned.get() == false)

        let firstChunkSeen = AsyncFlag()
        let firstConsumer = Task {
            for await generation in stream1 {
                if case .chunk = generation {
                    await firstChunkSeen.set()
                }
            }
        }
        defer { firstConsumer.cancel() }

        var sawChunk = false
        for _ in 0 ..< 50 {
            if await firstChunkSeen.get() {
                sawChunk = true
                break
            }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        #expect(sawChunk)

        _ = await firstConsumer.value
        _ = await blockerTicket.end()
        blockerReleased = true
        _ = try await secondTask.value
        await settleEvents()

        let events = await recorder.snapshot()
        #expect(ticketEvents(events, ticket: secondTicket, kind: .admissionWait).isEmpty == false)

        let firstEnd = try #require(ticketEvents(events, ticket: firstTicket, kind: .ticketEnded).first)
        let secondStart = try #require(
            ticketEvents(events, ticket: secondTicket, kind: .ticketStarted).first)
        #expect(firstEnd.sequence < secondStart.sequence)
    }

    @Test("Cancelling a joined batched request leaves the other request running")
    func cancellingJoinedBatchRequestLeavesOtherRequestRunning() async throws {
        let scheduler = InferenceScheduler()
        let model = SchedulerMockModel(callDelay: 0.002)
        let tokenizer = TestTokenizer()
        let configuration = ModelConfiguration(id: "scheduler-cancel")

        let stream1 = try await submitRaw(
            scheduler: scheduler,
            prompt: [1, 2, 3],
            parameters: GenerateParameters(maxTokens: 8, temperature: 0),
            model: model,
            tokenizer: tokenizer,
            configuration: configuration
        )
        let stream2 = try await submitRaw(
            scheduler: scheduler,
            prompt: [9, 10],
            parameters: GenerateParameters(maxTokens: 8, temperature: 0),
            model: model,
            tokenizer: tokenizer,
            configuration: configuration
        )

        #expect(await waitForState(scheduler, equals: "batched"))

        let firstResultTask = Task {
            await collectTokenGenerations(stream1)
        }

        let sawSecondToken = AsyncFlag()
        let secondResultTask = Task { () -> [Int] in
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

        secondResultTask.cancel()
        let firstResult = await firstResultTask.value
        let secondTokens = await secondResultTask.value

        let firstInfo = try #require(firstResult.info)
        #expect(firstInfo.generationTokenCount == 8)
        #expect(secondTokens.count < 8)

        _ = await waitForState(scheduler, equals: "idle")
        #expect(await scheduler.currentState == "idle")
    }
}

private func appendSchedulerSyntheticKV(
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

private func makeSchedulerPromptCache(
    layers: Int = 1,
    seqLen: Int,
    heads: Int = 4,
    headDim: Int = 8
) -> [KVCache] {
    (0 ..< layers).map { layer in
        let cache = KVCacheSimple()
        let keys = MLXArray.ones([1, heads, seqLen, headDim]) * Float(layer + 1)
        let values = MLXArray.ones([1, heads, seqLen, headDim]) * Float(layer + 2)
        _ = cache.update(keys: keys, values: values)
        return cache
    }
}

private final class SchedulerMockModel: Module, LanguageModel, KVCacheDimensionProvider,
    @unchecked Sendable
{
    let vocabSize: Int
    let numLayers: Int
    let callDelay: TimeInterval
    let syntheticHeadDim: Int

    var kvHeads: [Int] { Array(repeating: 4, count: numLayers) }
    var inputShapes = [[Int]]()
    var totalTokensProcessed = 0

    init(
        vocabSize: Int = 32,
        numLayers: Int = 1,
        callDelay: TimeInterval = 0,
        syntheticHeadDim: Int = 8
    ) {
        self.vocabSize = vocabSize
        self.numLayers = numLayers
        self.callDelay = callDelay
        self.syntheticHeadDim = syntheticHeadDim
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        if callDelay > 0 {
            Thread.sleep(forTimeInterval: callDelay)
        }

        appendSchedulerSyntheticKV(
            to: cache,
            inputTokens: input.tokens,
            defaultHeadDim: syntheticHeadDim
        )

        let tokens = input.tokens
        let batchSize = tokens.dim(0)
        let steps = tokens.dim(1)
        inputShapes.append(tokens.shape)
        totalTokensProcessed += batchSize * steps
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
        inputShapes = []
        totalTokensProcessed = 0
    }
}

private final class SSMSchedulerMockModel: Module, LanguageModel, @unchecked Sendable {
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        LMOutput(logits: MLXArray.zeros([input.tokens.dim(0), input.tokens.dim(1), 32]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [MambaCache()]
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

private actor AsyncFlag {
    private var value = false

    func set() {
        value = true
    }

    func get() -> Bool {
        value
    }
}

private actor WiredMemoryEventRecorder {
    private var events = [WiredMemoryEvent]()

    func append(_ event: WiredMemoryEvent) {
        events.append(event)
    }

    func snapshot() -> [WiredMemoryEvent] {
        events
    }
}

private struct CollectedGeneration {
    let chunks: [String]
    let info: GenerateCompletionInfo?
}

private struct CollectedTokenGeneration {
    let tokens: [Int]
    let info: GenerateCompletionInfo?
}

private func collectGenerations(_ stream: AsyncStream<Generation>) async -> CollectedGeneration {
    var chunks = [String]()
    var info: GenerateCompletionInfo?

    for await generation in stream {
        switch generation {
        case .chunk(let chunk):
            chunks.append(chunk)
        case .info(let completionInfo):
            info = completionInfo
        case .toolCall:
            break
        }
    }

    return CollectedGeneration(chunks: chunks, info: info)
}

private func collectTokenGenerations(_ stream: AsyncStream<TokenGeneration>) async
    -> CollectedTokenGeneration
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

    return CollectedTokenGeneration(tokens: tokens, info: info)
}

private func waitForState(
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

private func makeWiredMemoryManager(baseline: Int = 100) -> WiredMemoryManager {
    WiredMemoryManager.makeForTesting(
        configuration: .init(
            policyOnlyWhenUnsupported: true,
            baselineOverride: baseline,
            useRecommendedWorkingSetWhenUnsupported: false
        )
    )
}

private func startRecording(
    manager: WiredMemoryManager
) -> (WiredMemoryEventRecorder, Task<Void, Never>) {
    let recorder = WiredMemoryEventRecorder()
    let task = Task {
        for await event in await manager.events() {
            await recorder.append(event)
        }
    }
    return (recorder, task)
}

private func ticketEvents(
    _ events: [WiredMemoryEvent],
    ticket: WiredMemoryTicket,
    kind: WiredMemoryEvent.Kind? = nil
) -> [WiredMemoryEvent] {
    events.filter { event in
        event.ticketID == ticket.id && (kind == nil || event.kind == kind)
    }
}

private func settleEvents() async {
    try? await Task.sleep(nanoseconds: 20_000_000)
}
