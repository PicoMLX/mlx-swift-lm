// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Tokenizers

public struct InferenceRequest: @unchecked Sendable {
    public let input: LMInput
    public let parameters: GenerateParameters

    public init(input: LMInput, parameters: GenerateParameters) {
        self.input = input
        self.parameters = parameters
    }
}

public actor InferenceScheduler {

    public struct Config: Sendable {
        public var completionBatchSize: Int = 4
        public var prefillBatchSize: Int = 4
        public var maxQueueSize: Int = 32

        public init(
            completionBatchSize: Int = 4,
            prefillBatchSize: Int = 4,
            maxQueueSize: Int = 32
        ) {
            self.completionBatchSize = completionBatchSize
            self.prefillBatchSize = prefillBatchSize
            self.maxQueueSize = maxQueueSize
        }
    }

    public enum SchedulerError: Error, Sendable {
        case queueFull
        case modelNotReady
    }

    private struct PendingEntry {
        let uid: Int
        let promptTokens: [Int]
        let parameters: GenerateParameters
        let sampler: any LogitSampler
        let processor: BatchLogitProcessorBox?
        let batchOptions: BatchExecutionOptions

        var maxTokens: Int {
            parameters.maxTokens ?? .max
        }
    }

    private let context: ModelContext
    public let config: Config
    private let stopTokenIDs: Set<Int>
    private let unknownTokenId: Int?

    private var pendingQueue: [PendingEntry] = []
    private var sequences: [Int: SequenceState] = [:]
    private var iterator: BatchTokenIterator?
    private var decodeTask: Task<Void, Never>?
    private var nextUID = 0
    private var cancelledUIDs = Set<Int>()

    public init(context: ModelContext, config: Config = Config()) {
        self.context = context
        self.config = config
        self.stopTokenIDs = buildStopTokenIDs(
            modelConfiguration: context.configuration,
            tokenizer: context.tokenizer
        )
        self.unknownTokenId = context.tokenizer.unknownTokenId
    }

    static func isBatchCompatible(
        input: LMInput,
        parameters: GenerateParameters,
        context: ModelContext
    ) -> Bool {
        guard input.image == nil, input.video == nil, input.text.mask == nil else {
            return false
        }

        // Batched KV quantization is not implemented for batch cache types.
        guard parameters.kvBits == nil else {
            return false
        }

        return supportsBatchCaches(context.model.newCache(parameters: parameters))
    }

    private static func supportsBatchCaches(_ caches: [KVCache]) -> Bool {
        caches.allSatisfy { cache in
            switch cache {
            case is KVCacheSimple:
                return true
            case let rotating as RotatingKVCache:
                return rotating.keepTokens == 0 && rotating.maxSize != nil
            case is ArraysCache:
                return true
            case let list as CacheList:
                return supportsBatchCaches((0 ..< list.count).map { list[$0] })
            default:
                return false
            }
        }
    }

    public func submit(_ request: InferenceRequest) async throws -> AsyncStream<Generation> {
        guard sequences.count < config.maxQueueSize + config.completionBatchSize else {
            throw SchedulerError.queueFull
        }

        let uid = nextUID
        nextUID += 1

        let promptTokens = request.input.text.tokens.asArray(Int32.self).map(Int.init)
        let (stream, continuation) = AsyncStream<Generation>.makeStream()

        sequences[uid] = SequenceState(
            uid: uid,
            promptTokenCount: promptTokens.count,
            tokenizer: context.tokenizer,
            modelConfiguration: context.configuration,
            continuation: continuation
        )

        let processor = request.parameters.processor().map(BatchLogitProcessorBox.init)
        pendingQueue.append(
            PendingEntry(
                uid: uid,
                promptTokens: promptTokens,
                parameters: request.parameters,
                sampler: request.parameters.sampler(),
                processor: processor,
                batchOptions: BatchExecutionOptions(parameters: request.parameters)
            )
        )

        continuation.onTermination = { [weak self] _ in
            Task { await self?.cancelSequence(uid: uid) }
        }

        ensureDecodeLoopRunning()
        return stream
    }

    public func cancelAll() {
        let outstandingUIDs = Array(sequences.keys)
        for uid in outstandingUIDs {
            cancelSequenceInternal(uid: uid)
        }

        pendingQueue.removeAll()
        cancelledUIDs.removeAll()
        iterator = nil
        decodeTask?.cancel()
        decodeTask = nil
    }

    private func ensureDecodeLoopRunning() {
        guard decodeTask == nil else { return }
        decodeTask = Task { [weak self] in
            await self?.runDecodeLoop()
        }
    }

    private func runDecodeLoop() async {
        defer { decodeTask = nil }

        while !Task.isCancelled {
            applyCancelledSequences()
            admitPending()

            guard let localIterator = iterator else {
                if pendingQueue.isEmpty {
                    break
                }
                await Task.yield()
                continue
            }

            let snapshot = localIterator
            let result = try? await DeviceEngine.shared.run {
                var iterator = snapshot
                let responses = iterator.next()
                return (responses, iterator)
            }
            guard let (responses, updatedIterator) = result else {
                break
            }
            iterator = updatedIterator

            applyCancelledSequences()
            if let responses {
                processStepResults(responses)
            }

            if iterator?.hasWork == false {
                iterator = nil
            }

            if pendingQueue.isEmpty && sequences.isEmpty && iterator == nil {
                break
            }

            await Task.yield()
        }

        applyCancelledSequences()
        if Task.isCancelled {
            iterator = nil
        }
    }

    private func admitPending() {
        guard !pendingQueue.isEmpty else { return }

        if iterator == nil, let first = pendingQueue.first {
            iterator = BatchTokenIterator(
                model: context.model,
                configuration: BatchIteratorConfiguration(
                    completionBatchSize: config.completionBatchSize,
                    prefillBatchSize: config.prefillBatchSize,
                    generation: first.parameters
                ),
                stopTokens: stopTokenIDs,
                unknownTokenId: unknownTokenId
            )
        }

        guard let iterator else { return }
        let targetOptions = iterator.batchOptions

        var accepted: [PendingEntry] = []
        var deferred: [PendingEntry] = []
        for entry in pendingQueue {
            if entry.batchOptions == targetOptions {
                accepted.append(entry)
            } else {
                deferred.append(entry)
            }
        }

        pendingQueue = deferred
        guard !accepted.isEmpty else { return }

        var updatedIterator = iterator
        updatedIterator.insert(
            uids: accepted.map(\.uid),
            prompts: accepted.map(\.promptTokens),
            maxTokens: accepted.map(\.maxTokens),
            samplers: accepted.map(\.sampler),
            processors: accepted.map(\.processor)
        )
        self.iterator = updatedIterator
    }

    private func processStepResults(_ responses: [BatchTokenIterator.Response]) {
        var emittedCancellationUIDs: [Int] = []

        for response in responses {
            guard let sequence = sequences[response.uid] else { continue }

            let isStopToken = response.token == unknownTokenId || stopTokenIDs.contains(response.token)

            if !isStopToken {
                sequence.recordGeneratedToken(response.token)
                if !sequence.emitToken(response.token) {
                    emittedCancellationUIDs.append(response.uid)
                    continue
                }
            }

            if let stopReason = response.stopReason {
                sequences.removeValue(forKey: response.uid)
                sequence.finish(stopReason: stopReason)
            }
        }

        for uid in emittedCancellationUIDs {
            cancelSequenceInternal(uid: uid)
        }
    }

    private func cancelSequence(uid: Int) {
        cancelSequenceInternal(uid: uid)
    }

    private func cancelSequenceInternal(uid: Int) {
        pendingQueue.removeAll { $0.uid == uid }
        cancelledUIDs.insert(uid)

        guard let sequence = sequences.removeValue(forKey: uid) else { return }
        sequence.finish(stopReason: .cancelled)
    }

    private func applyCancelledSequences() {
        guard !cancelledUIDs.isEmpty else { return }

        pendingQueue.removeAll { cancelledUIDs.contains($0.uid) }

        if var iterator {
            iterator.remove(uids: cancelledUIDs)
            self.iterator = iterator.hasWork ? iterator : nil
        }

        cancelledUIDs.removeAll()
    }
}
