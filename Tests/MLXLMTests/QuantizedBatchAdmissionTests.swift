// Copyright © 2026 Apple Inc.

import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct QuantizedBatchAdmissionTests {

    @Test(
        "Rejected quantized admission preserves every active layer, queued prefix and sampler stream",
        arguments: QuantizedAdmissionMismatch.allCases, [false, true])
    func rejectedAdmission(mismatch: QuantizedAdmissionMismatch, keepQueued: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = QuantizedAdmissionModel()
            let activePrompt = [1, 3, 5]
            let queuedPrompt = [2, 4]
            let prefix = mismatch.caches(incoming: true)
            _ = model(
                LMInput.Text(tokens: MLXArray([Int32(9), 10])[.newAxis]),
                cache: prefix, state: nil)
            let prefixState = AdmissionCacheSnapshot(prefix)
            let expectedActive = try admissionTokens(
                model, prompt: activePrompt, caches: mismatch.caches(incoming: false),
                seed: 17, maxTokens: 6)
            let expectedQueued = try admissionTokens(
                model, prompt: queuedPrompt, caches: prefix.map { $0.copy() },
                seed: 18, maxTokens: 4)

            var iterator = BatchTokenIterator(model: model)
            let active = iterator.insert(
                prompts: [activePrompt], maxTokens: [6],
                caches: [mismatch.caches(incoming: false)],
                samplers: [TopPSampler(temperature: 0.8, topP: 0.9, seed: 17)])
            let first = try iterator.next().map(\.token)
            #expect(first == Array(expectedActive.prefix(1)))
            let activeCache = try #require(model.lastCache)
            if mismatch == .groupSize {
                #expect(try #require(activeCache[1] as? BatchQuantizedKVCache).groupSize == 32)
                #expect(try #require(prefix[1] as? QuantizedKVCache).groupSize == 64)
            }
            let activeState = AdmissionCacheSnapshot(activeCache)
            let queuedSampler = TopPSampler(temperature: 0.8, topP: 0.9, seed: 18)
            let queued = iterator.insert(
                prompts: [queuedPrompt], maxTokens: [4], caches: [prefix],
                samplers: [queuedSampler])

            for _ in 0 ..< 2 {
                #expect(throws: KVCacheError.self) { _ = try iterator.next() }
                #expect(iterator.uids == active + queued)
                try activeState.expectUnchanged(activeCache)
                try prefixState.expectUnchanged(prefix)
            }
            if keepQueued {
                iterator.remove(active)
                #expect(try collectAdmission(&iterator, uids: queued) == [expectedQueued])
            } else {
                iterator.remove(queued)
                let remainder = try collectAdmission(&iterator, uids: active)
                #expect(first + remainder[0] == expectedActive)
                let retried = iterator.insert(
                    prompts: [queuedPrompt], maxTokens: [4], caches: [prefix],
                    samplers: [queuedSampler])
                #expect(try collectAdmission(&iterator, uids: retried) == [expectedQueued])
            }
            try prefixState.expectUnchanged(prefix)
        }
    }

    @Test(
        "Matching quantized rows join without changing either seeded token stream",
        arguments: [4, 8])
    func compatibleAdmission(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = QuantizedAdmissionModel()
            func caches() -> [KVCache] {
                [KVCacheSimple(), QuantizedKVCache(groupSize: 64, bits: bits)]
            }
            let prompts = [[1, 3, 5], [2, 4]]
            let expected = try prompts.indices.map { index in
                try admissionTokens(
                    model, prompt: prompts[index], caches: caches(), seed: UInt64(17 + index),
                    maxTokens: 6)
            }
            var iterator = BatchTokenIterator(model: model)
            let active = iterator.insert(
                prompts: [prompts[0]], maxTokens: [6], caches: [caches()],
                samplers: [TopPSampler(temperature: 0.8, topP: 0.9, seed: 17)])
            let first = try iterator.next().map(\.token)
            let queued = iterator.insert(
                prompts: [prompts[1]], maxTokens: [6], caches: [caches()],
                samplers: [TopPSampler(temperature: 0.8, topP: 0.9, seed: 18)])
            let remainder = try collectAdmission(&iterator, uids: active + queued)
            #expect(first + remainder[0] == expected[0])
            #expect(remainder[1] == expected[1])
        }
    }
}

enum QuantizedAdmissionMismatch: CaseIterable, Sendable {
    case precision
    case groupSize
    case floatToQuantized
    case quantizedToFloat
    case compositeChild

    fileprivate func caches(incoming: Bool) -> [KVCache] {
        let first = QuantizedKVCache(groupSize: 64, bits: 4)
        switch self {
        case .precision:
            return [first, QuantizedKVCache(groupSize: 64, bits: incoming ? 8 : 4)]
        case .groupSize:
            return [first, QuantizedKVCache(groupSize: incoming ? 64 : 32, bits: 4)]
        case .floatToQuantized:
            return [first, incoming ? QuantizedKVCache(bits: 4) : KVCacheSimple()]
        case .quantizedToFloat:
            return [first, incoming ? KVCacheSimple() : QuantizedKVCache(bits: 4)]
        case .compositeChild:
            return [
                first,
                CacheList(caches: [
                    KVCacheSimple(), QuantizedKVCache(groupSize: 64, bits: incoming ? 8 : 4),
                ]),
            ]
        }
    }
}

/// All cache writes use 64-wide heads, so group sizes 32 and 64 remain distinct.
private final class QuantizedAdmissionModel: Module, LanguageModel {
    private(set) var lastCache: [KVCache]?

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let tokens = input.tokens
        let keys =
            tokens.asType(.float32)[0..., .newAxis, 0..., .newAxis]
            + MLXArray.arange(64).asType(.float32) * 0.01
        for layer in cache ?? [] { update(layer, keys: keys) }
        lastCache = cache
        let target = tokens[0..., -1, .newAxis, .newAxis].asType(.float32)
        let logits = -abs(MLXArray.arange(32).asType(.float32)[.newAxis, .newAxis] - target) * 0.07
        return LMOutput(logits: logits)
    }

    private func update(_ cache: KVCache, keys: MLXArray) {
        if let composite = cache as? CacheList {
            for child in composite.children { update(child, keys: keys) }
        } else if let quantized = cache as? any QuantizedKVCacheProtocol {
            _ = quantized.updateQuantized(keys: keys, values: keys + 0.2)
        } else {
            _ = cache.update(keys: keys, values: keys + 0.2)
        }
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }
}

private struct AdmissionCacheSnapshot {
    let offsets: [Int]
    let states: [[MLXArray]]

    init(_ caches: [KVCache]) {
        offsets = caches.map(\.offset)
        states = caches.map { $0.state.map { $0[.ellipsis] } }
        eval(states.flatMap { $0 })
    }

    func expectUnchanged(_ caches: [KVCache], sourceLocation: SourceLocation = #_sourceLocation)
        throws
    {
        #expect(caches.map(\.offset) == offsets, sourceLocation: sourceLocation)
        try #require(caches.count == states.count, sourceLocation: sourceLocation)
        for (cache, expected) in zip(caches, states) {
            let actual = cache.state
            try #require(actual.count == expected.count, sourceLocation: sourceLocation)
            for (lhs, rhs) in zip(actual, expected) {
                try #require(lhs.shape == rhs.shape, sourceLocation: sourceLocation)
                #expect(lhs.dtype == rhs.dtype, sourceLocation: sourceLocation)
                #expect(all(lhs .== rhs).item(Bool.self), sourceLocation: sourceLocation)
            }
        }
    }
}

private func admissionTokens(
    _ model: any LanguageModel, prompt: [Int], caches: [KVCache], seed: UInt64, maxTokens: Int
) throws -> [Int] {
    var iterator = BatchTokenIterator(model: model)
    let uids = iterator.insert(
        prompts: [prompt], maxTokens: [maxTokens], caches: [caches],
        samplers: [TopPSampler(temperature: 0.8, topP: 0.9, seed: seed)])
    return try collectAdmission(&iterator, uids: uids)[0]
}

private func collectAdmission(_ iterator: inout BatchTokenIterator, uids: [Int]) throws -> [[Int]] {
    var output = [Int: [Int]]()
    while true {
        let responses = try iterator.next()
        if responses.isEmpty { break }
        for response in responses { output[response.uid, default: []].append(response.token) }
    }
    return uids.map { output[$0] ?? [] }
}
