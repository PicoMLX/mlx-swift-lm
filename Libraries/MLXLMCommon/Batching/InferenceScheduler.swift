// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Tokenizers

public struct InferenceRequest: @unchecked Sendable {
    public let input: LMInput
    public let promptCache: [KVCache]?
    public let parameters: GenerateParameters

    public init(input: LMInput, promptCache: [KVCache]? = nil, parameters: GenerateParameters) {
        self.input = input
        self.promptCache = promptCache
        self.parameters = parameters
    }
}

public actor InferenceScheduler {

    public struct Config: Sendable {
        public var completionBatchSize: Int = 4
        public var prefillBatchSize: Int = 4
        public var maxQueueSize: Int = 32
        public var maxConcurrentLanes: Int = 2

        public init(
            completionBatchSize: Int = 4,
            prefillBatchSize: Int = 4,
            maxQueueSize: Int = 32,
            maxConcurrentLanes: Int = 2
        ) {
            self.completionBatchSize = completionBatchSize
            self.prefillBatchSize = prefillBatchSize
            self.maxQueueSize = maxQueueSize
            self.maxConcurrentLanes = maxConcurrentLanes
        }
    }

    public enum SchedulerError: Error, Sendable {
        case queueFull
    }

    private struct PendingEntry {
        let uid: Int
        let promptTokens: [Int]
        let promptCache: [KVCache]?
        let parameters: GenerateParameters
        let sampler: any LogitSampler
        let processor: BatchLogitProcessorBox?
        let batchOptions: BatchExecutionOptions

        var maxTokens: Int {
            parameters.maxTokens ?? .max
        }
    }

    private struct LaneState {
        var pendingQueue: [PendingEntry] = []
        var iterator: BatchTokenIterator?
        var cancelledUIDs = Set<Int>()

        var hasWork: Bool {
            iterator != nil || !pendingQueue.isEmpty || !cancelledUIDs.isEmpty
        }
    }

    private let context: ModelContext
    public let config: Config
    private let stopTokenIDs: Set<Int>
    private let unknownTokenId: Int?

    private var lanes: [BatchExecutionOptions: LaneState] = [:]
    private var activeLaneOrder: [BatchExecutionOptions] = []
    private var deferredQueues: [BatchExecutionOptions: [PendingEntry]] = [:]
    private var deferredLaneOrder: [BatchExecutionOptions] = []
    private var sequences: [Int: SequenceState] = [:]
    private var sequenceLaneKeys: [Int: BatchExecutionOptions] = [:]
    private var runLoopTask: Task<Void, Never>?
    private var nextLaneIndex = 0
    private var nextUID = 0

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
        promptCache: [KVCache]? = nil,
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

        let caches = promptCache ?? context.model.newCache(parameters: parameters)
        return supportsBatchCaches(caches)
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
        let maxOutstanding = config.maxQueueSize + config.completionBatchSize * max(config.maxConcurrentLanes, 1)
        guard sequences.count < maxOutstanding else {
            throw SchedulerError.queueFull
        }

        let uid = nextUID
        nextUID += 1

        let promptTokens = request.input.text.tokens.asArray(Int32.self).map(Int.init)
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let batchOptions = BatchExecutionOptions(parameters: request.parameters)

        sequences[uid] = SequenceState(
            uid: uid,
            promptTokenCount: promptTokens.count,
            tokenizer: context.tokenizer,
            modelConfiguration: context.configuration,
            continuation: continuation
        )
        sequenceLaneKeys[uid] = batchOptions

        let processor = request.parameters.processor().map(BatchLogitProcessorBox.init)
        enqueue(
            PendingEntry(
                uid: uid,
                promptTokens: promptTokens,
                promptCache: request.promptCache,
                parameters: request.parameters,
                sampler: request.parameters.sampler(),
                processor: processor,
                batchOptions: batchOptions
            )
        )

        continuation.onTermination = { [weak self] _ in
            Task { await self?.cancelSequence(uid: uid) }
        }

        return stream
    }

    public func cancelAll() {
        runLoopTask?.cancel()
        runLoopTask = nil

        for sequence in sequences.values {
            sequence.finish(stopReason: .cancelled)
        }

        lanes.removeAll()
        activeLaneOrder.removeAll()
        deferredQueues.removeAll()
        deferredLaneOrder.removeAll()
        sequences.removeAll()
        sequenceLaneKeys.removeAll()
        nextLaneIndex = 0
    }

    private func enqueue(_ entry: PendingEntry) {
        let laneKey = entry.batchOptions

        if var lane = lanes[laneKey] {
            lane.pendingQueue.append(entry)
            lanes[laneKey] = lane
            ensureRunLoopRunning()
            return
        }

        if lanes.count < max(config.maxConcurrentLanes, 1) {
            var lane = LaneState()
            lane.pendingQueue = takeDeferredEntries(for: laneKey)
            lane.pendingQueue.append(entry)
            activateLane(lane, for: laneKey)
            ensureRunLoopRunning()
            return
        }

        deferredQueues[laneKey, default: []].append(entry)
        if !deferredLaneOrder.contains(laneKey) {
            deferredLaneOrder.append(laneKey)
        }
        ensureRunLoopRunning()
    }

    private func takeDeferredEntries(for laneKey: BatchExecutionOptions) -> [PendingEntry] {
        let entries = deferredQueues.removeValue(forKey: laneKey) ?? []
        deferredLaneOrder.removeAll { $0 == laneKey }
        return entries
    }

    private func activateLane(_ lane: LaneState, for laneKey: BatchExecutionOptions) {
        lanes[laneKey] = lane
        if !activeLaneOrder.contains(laneKey) {
            activeLaneOrder.append(laneKey)
        }
    }

    private func ensureRunLoopRunning() {
        guard runLoopTask == nil else { return }
        runLoopTask = Task { [weak self] in
            await self?.runLoop()
        }
    }

    private func runLoop() async {
        defer { finishRunLoop() }

        while !Task.isCancelled {
            activateDeferredLanesIfPossible()

            guard let laneKey = nextRunnableLane() else {
                break
            }

            await processNextStep(in: laneKey)
            await Task.yield()
        }
    }

    private func finishRunLoop() {
        runLoopTask = nil
        compactIdleLanes()
        activateDeferredLanesIfPossible()

        if hasOutstandingWork {
            ensureRunLoopRunning()
        }
    }

    private var hasOutstandingWork: Bool {
        !lanes.isEmpty || deferredLaneOrder.contains { deferredQueues[$0]?.isEmpty == false }
    }

    private func nextRunnableLane() -> BatchExecutionOptions? {
        compactIdleLanes()
        guard !activeLaneOrder.isEmpty else { return nil }

        let laneCount = activeLaneOrder.count
        for offset in 0 ..< laneCount {
            let index = (nextLaneIndex + offset) % laneCount
            let laneKey = activeLaneOrder[index]
            guard let lane = lanes[laneKey], lane.hasWork else { continue }
            nextLaneIndex = (index + 1) % laneCount
            return laneKey
        }
        return nil
    }

    private func compactIdleLanes() {
        let idleLaneKeys = activeLaneOrder.filter { !(lanes[$0]?.hasWork ?? false) }
        for laneKey in idleLaneKeys {
            removeLane(for: laneKey)
        }
    }

    private func removeLane(for laneKey: BatchExecutionOptions) {
        lanes.removeValue(forKey: laneKey)

        guard let index = activeLaneOrder.firstIndex(of: laneKey) else {
            return
        }

        activeLaneOrder.remove(at: index)

        if activeLaneOrder.isEmpty {
            nextLaneIndex = 0
        } else {
            if index < nextLaneIndex {
                nextLaneIndex -= 1
            }
            nextLaneIndex %= activeLaneOrder.count
        }
    }

    private func processNextStep(in laneKey: BatchExecutionOptions) async {
        applyCancelledSequences(in: laneKey)
        admitPending(in: laneKey)

        guard let lane = lanes[laneKey] else {
            return
        }

        guard let localIterator = lane.iterator else {
            if !lane.hasWork {
                removeLane(for: laneKey)
                activateDeferredLanesIfPossible()
            }
            return
        }

        let snapshot = localIterator
        let result = try? await DeviceEngine.shared.run {
            var iterator = snapshot
            let responses = iterator.next()
            return (responses, iterator)
        }

        guard let (responses, updatedIterator) = result else {
            removeLane(for: laneKey)
            activateDeferredLanesIfPossible()
            return
        }

        guard var refreshedLane = lanes[laneKey] else {
            return
        }
        refreshedLane.iterator = updatedIterator
        lanes[laneKey] = refreshedLane

        applyCancelledSequences(in: laneKey)
        if let responses {
            processStepResults(responses)
        }

        if var latestLane = lanes[laneKey], latestLane.iterator?.hasWork == false {
            latestLane.iterator = nil
            lanes[laneKey] = latestLane
        }

        if let latestLane = lanes[laneKey], !latestLane.hasWork {
            removeLane(for: laneKey)
            activateDeferredLanesIfPossible()
        }
    }

    private func activateDeferredLanesIfPossible() {
        while lanes.count < max(config.maxConcurrentLanes, 1), let laneKey = deferredLaneOrder.first {
            let entries = takeDeferredEntries(for: laneKey)
            guard !entries.isEmpty else { continue }

            var lane = LaneState()
            lane.pendingQueue = entries
            activateLane(lane, for: laneKey)
        }
    }

    private func admitPending(in laneKey: BatchExecutionOptions) {
        guard var lane = lanes[laneKey], !lane.pendingQueue.isEmpty else { return }

        if lane.iterator == nil, let first = lane.pendingQueue.first {
            lane.iterator = BatchTokenIterator(
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

        guard var iterator = lane.iterator else {
            lanes[laneKey] = lane
            return
        }

        let accepted = lane.pendingQueue
        lane.pendingQueue.removeAll(keepingCapacity: true)
        iterator.insert(
            uids: accepted.map(\.uid),
            prompts: accepted.map(\.promptTokens),
            promptCaches: accepted.map(\.promptCache),
            maxTokens: accepted.map(\.maxTokens),
            samplers: accepted.map(\.sampler),
            processors: accepted.map(\.processor)
        )

        lane.iterator = iterator
        lanes[laneKey] = lane
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
                sequenceLaneKeys.removeValue(forKey: response.uid)
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
        guard let laneKey = sequenceLaneKeys.removeValue(forKey: uid) else { return }

        if var lane = lanes[laneKey] {
            lane.pendingQueue.removeAll { $0.uid == uid }
            lane.cancelledUIDs.insert(uid)
            lanes[laneKey] = lane
        } else if var deferred = deferredQueues[laneKey] {
            deferred.removeAll { $0.uid == uid }
            if deferred.isEmpty {
                deferredQueues.removeValue(forKey: laneKey)
                deferredLaneOrder.removeAll { $0 == laneKey }
            } else {
                deferredQueues[laneKey] = deferred
            }
        }

        guard let sequence = sequences.removeValue(forKey: uid) else { return }
        sequence.finish(stopReason: .cancelled)
    }

    private func applyCancelledSequences(in laneKey: BatchExecutionOptions) {
        guard var lane = lanes[laneKey], !lane.cancelledUIDs.isEmpty else { return }

        lane.pendingQueue.removeAll { lane.cancelledUIDs.contains($0.uid) }

        if var iterator = lane.iterator {
            iterator.remove(uids: lane.cancelledUIDs)
            lane.iterator = iterator.hasWork ? iterator : nil
        }

        lane.cancelledUIDs.removeAll()
        lanes[laneKey] = lane
    }
}
