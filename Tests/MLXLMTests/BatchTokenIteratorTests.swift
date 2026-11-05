// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
import Testing

@testable import MLXLMCommon

/// Batched generation must give every prompt the tokens it gets generated alone.
@Suite(.serialized)
struct BatchTokenIteratorTests {

    static let prompts: [[Int]] = [[1, 2, 3], [4, 5, 6, 7, 8], [9]]

    @Test("Greedy batched generation matches each prompt generated alone")
    func greedy() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            let expected = try Self.prompts.map { prompt in
                try generateAlone(model, prompt: prompt, sampler: ArgMaxSampler(), maxTokens: 8)
            }
            let actual = try batchGenerate(model: model, prompts: Self.prompts, maxTokens: 8)
            #expect(actual == expected)
        }
    }

    @Test("Rows with their own seeded samplers draw the tokens they draw alone")
    func seededPerRow() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            func sampler(_ row: Int) -> TopPSampler {
                TopPSampler(temperature: 0.8, topP: 0.9, seed: UInt64(7 + row))
            }
            let expected = try Self.prompts.indices.map { row in
                try generateAlone(
                    model, prompt: Self.prompts[row], sampler: sampler(row), maxTokens: 8)
            }

            var iterator = BatchTokenIterator(model: model, maxTokens: 8)
            let uids = iterator.insert(
                prompts: Self.prompts, samplers: Self.prompts.indices.map { sampler($0) })
            let actual = try collect(&iterator, uids: uids)
            #expect(actual == expected)
        }
    }

    @Test("One shared seeded sampler draws the whole batch, reproducibly")
    func sharedSampler() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            func run() throws -> [[Int]] {
                try batchGenerate(
                    model: model, prompts: Self.prompts, maxTokens: 6,
                    sampler: TopPSampler(temperature: 0.8, topP: 0.9, seed: 3))
            }
            let first = try run()
            #expect(first.allSatisfy { $0.count == 6 })
            #expect(try run() == first)
        }
    }

    @Test("Rows leave on a stop token or at their limit, and rows join while others decode")
    func rowsJoinAndLeave() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            let alone = try Self.prompts.map { prompt in
                try generateAlone(model, prompt: prompt, sampler: ArgMaxSampler(), maxTokens: 6)
            }
            // The third token of the first prompt ends that row.
            let stop = alone[0][2]
            var iterator = BatchTokenIterator(model: model, stopTokens: [stop], maxTokens: 6)
            let first = iterator.insert(prompts: [Self.prompts[0], Self.prompts[1]])
            var steps: [[BatchTokenIterator.Response]] = []
            steps.append(try iterator.next())
            steps.append(try iterator.next())
            let third = iterator.insert(prompts: [Self.prompts[2]], maxTokens: [3])
            #expect(iterator.uids == first + third)
            while true {
                let responses = try iterator.next()
                if responses.isEmpty { break }
                steps.append(responses)
            }

            func tokens(_ uid: Int) -> [Int] {
                steps.flatMap { $0.filter { $0.uid == uid } }.map(\.token)
            }
            func reasons(_ uid: Int) -> [BatchTokenIterator.FinishReason?] {
                steps.flatMap { $0.filter { $0.uid == uid } }.map(\.finishReason)
            }
            #expect(tokens(first[0]) == Array(alone[0].prefix(3)))
            #expect(reasons(first[0]) == [nil, nil, .stop])
            #expect(tokens(first[1]) == alone[1])
            #expect(reasons(first[1]) == [nil, nil, nil, nil, nil, .length])
            #expect(tokens(third[0]) == Array(alone[2].prefix(3)))
            #expect(reasons(third[0]) == [nil, nil, .length])
            #expect(steps[2].map(\.uid) == first + third)
            #expect(steps[3].map(\.uid) == [first[1], third[0]])
            #expect(iterator.uids.isEmpty)
        }
    }

    @Test("A removed row leaves the batch, queued or decoding")
    func remove() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            var iterator = BatchTokenIterator(model: model, maxTokens: 4)
            let uids = iterator.insert(prompts: Self.prompts)
            iterator.remove([uids[1]])
            #expect(iterator.uids == [uids[0], uids[2]])
            _ = try iterator.next()
            iterator.remove([uids[0]])
            #expect(iterator.uids == [uids[2]])
            let responses = try iterator.next()
            #expect(responses.map(\.uid) == [uids[2]])
        }
    }

    @Test("A row continues from its prompt cache")
    func promptCache() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            let prefix: [Int] = [1, 2, 3, 4]
            let suffix: [Int] = [5, 6]
            let alone = try model.newCache(parameters: nil)
            eval(model(MLXArray(prefix.map { Int32($0) })[.newAxis], cache: alone))
            let expected = try generateAlone(
                model, prompt: suffix, cache: alone, sampler: ArgMaxSampler(), maxTokens: 6)

            let cache = try model.newCache(parameters: nil)
            eval(model(MLXArray(prefix.map { Int32($0) })[.newAxis], cache: cache))
            let actual = try batchGenerate(
                model: model, prompts: [suffix, Self.prompts[1]], caches: [cache, nil],
                maxTokens: 6)
            #expect(actual[0] == expected)
            let second = try generateAlone(
                model, prompt: Self.prompts[1], sampler: ArgMaxSampler(), maxTokens: 6)
            #expect(actual[1] == second)
        }
    }

    @Test("Logit processors run per row")
    func processors() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            let parameters = GenerateParameters(repetitionPenalty: 1.5, repetitionContextSize: 8)
            let expected = try Self.prompts.map { prompt in
                try generateAlone(
                    model, prompt: prompt, processor: parameters.processor(),
                    sampler: ArgMaxSampler(), maxTokens: 8)
            }
            let actual = try batchGenerate(
                model: model, prompts: Self.prompts, maxTokens: 8,
                processor: parameters.processor())
            #expect(actual == expected)
        }
    }

    @Test("Top-p keeps the top token when its threshold would remove every token")
    func topPKeepsTopToken() {
        // 1 - 1e-9 rounds to 1 in float32, which no cumulative probability exceeds.
        let sampler = TopPSampler(temperature: 1, topP: 1e-9, seed: 1)
        let logits = MLXArray([Float(-1000), -1000, 0, -1000])[.newAxis]
        for _ in 0 ..< 4 {
            #expect(sampler.sample(logits: logits).item(Int.self) == 2)
        }
        let batch = MLXArray([Float(-1000), 0, -1000, 0, -1000, -1000]).reshaped([2, 3])
        #expect(
            sampler.sample(logProbabilities: logSoftmax(batch, axis: -1)).asArray(Int.self)
                == [1, 0])
    }

    @Test("The row methods and merge dispatch through the KVCache type")
    func protocolSurface() throws {
        let rows: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
        let batch = try mergeCaches(rows)
        #expect(batch is BatchKVCache)
        let (keys, values) = makeDistinctKV(batchSize: 2, heads: 2, seqLen: 3, headDim: 4)
        _ = batch.update(keys: keys, values: values)
        batch.filter(batchIndices: batchRows(1))
        #expect((batch as? BatchKVCache)?.batchSize == 1)
        let added = try mergeCaches([KVCacheSimple()])
        _ = added.update(keys: keys[0 ..< 1], values: values[0 ..< 1])
        batch.extend(other: added)
        #expect((batch as? BatchKVCache)?.batchSize == 2)
        let row = try #require(batch.extract(0) as? KVCacheSimple)
        let rowKeys = try #require(row.state.first)
        #expect(maxAbsDifference(rowKeys, MLXArray.ones(rowKeys.shape) * 2) == 0)

        #expect(try mergeCaches([RotatingKVCache(maxSize: 8)]) is BatchRotatingKVCache)
        #expect(try mergeCaches([MambaCache()]) is MambaCache)
        #expect(try mergeCaches([CacheList(KVCacheSimple(), MambaCache())]) is CacheList)
        #expect(throws: KVCacheError.self) { _ = try mergeCaches([QuantizedKVCache()]) }
        #expect(throws: KVCacheError.self) { _ = try mergeCaches([ChunkedKVCache(chunkSize: 4)]) }
        #expect(throws: KVCacheError.self) { _ = try QuantizedKVCache.merge([QuantizedKVCache()]) }
    }
}

/// The tokens `TokenIterator` generates for `prompt`.
private func generateAlone(
    _ model: any LanguageModel, prompt: [Int], cache: [KVCache]? = nil,
    processor: LogitProcessor? = nil, sampler: LogitSampler, maxTokens: Int
) throws -> [Int] {
    let iterator = try TokenIterator(
        input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), model: model, cache: cache,
        processor: processor, sampler: sampler, maxTokens: maxTokens)
    return Array(IteratorSequence(iterator))
}

/// Every row's tokens, in uid order, until the batch is empty.
private func collect(_ iterator: inout BatchTokenIterator, uids: [Int]) throws -> [[Int]] {
    var results = [Int: [Int]]()
    while true {
        let responses = try iterator.next()
        if responses.isEmpty { break }
        for response in responses {
            results[response.uid, default: []].append(response.token)
        }
    }
    return uids.map { results[$0] ?? [] }
}
