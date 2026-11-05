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

    @Test("A ragged batch of a recurrent model starts every row like the prompt alone")
    func recurrentRows() throws {
        // On the default device: the state-space scan is a Metal kernel. The first token of
        // each row comes from its own prefill, so it must match exactly; later steps run a
        // two-row scan whose arithmetic differs from the one-row scan.
        let model = try tinyMamba2()
        let prompts: [[Int]] = [[1, 2, 3], [4, 5, 6, 7, 8]]
        let expected = try prompts.map { prompt in
            try generateAlone(model, prompt: prompt, sampler: ArgMaxSampler(), maxTokens: 4)
        }
        let actual = try batchGenerate(model: model, prompts: prompts, maxTokens: 4)
        #expect(actual.map(\.count) == [4, 4])
        #expect(actual.map { $0[0] } == expected.map { $0[0] })
    }

    @Test("Ragged rows of a hybrid attention-recurrent model join while others decode")
    func hybridRowsJoinMidDecode() throws {
        // On the default device, like `recurrentRows`: each row's first token comes from its
        // own prefill and must match the prompt alone; later steps run a wider scan.
        let model = try tinyFalconH1()
        let prompts: [[Int]] = [[1, 2, 3, 4, 5, 6], [7, 8], [9, 10, 11]]
        let expected = try prompts.map { prompt in
            try generateAlone(model, prompt: prompt, sampler: ArgMaxSampler(), maxTokens: 5)
        }

        var iterator = BatchTokenIterator(model: model, maxTokens: 5)
        var uids = iterator.insert(prompts: Array(prompts[0 ..< 2]))
        var tokens = [Int: [Int]]()
        for _ in 0 ..< 2 {
            for response in try iterator.next() {
                tokens[response.uid, default: []].append(response.token)
            }
        }
        uids += iterator.insert(prompts: [prompts[2]])
        while true {
            let responses = try iterator.next()
            if responses.isEmpty { break }
            for response in responses {
                tokens[response.uid, default: []].append(response.token)
            }
        }
        let actual = uids.map { tokens[$0] ?? [] }
        #expect(actual.map(\.count) == [5, 5, 5])
        #expect(actual.map { $0[0] } == expected.map { $0[0] })
    }

    @Test("A sampler from outside the library gets each row's raw logits")
    func customSamplerGetsRawLogits() throws {
        // The first token with a positive logit: all log-probabilities are negative, so a
        // sampler handed those instead would always pick token zero.
        struct FirstPositiveSampler: LogitSampler {
            func sample(logits: MLXArray) -> MLXArray {
                argMax(logits .> 0, axis: -1)
            }
        }
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            let sampler = FirstPositiveSampler()
            let expected = try Self.prompts.map { prompt in
                try generateAlone(model, prompt: prompt, sampler: sampler, maxTokens: 6)
            }
            #expect(expected.contains { $0.contains { $0 != 0 } }, "the model has positive logits")
            // One sampler shared by every row, and one per row.
            #expect(
                try batchGenerate(
                    model: model, prompts: Self.prompts, maxTokens: 6, sampler: sampler)
                    == expected)
            var iterator = BatchTokenIterator(model: model, maxTokens: 6)
            let uids = iterator.insert(
                prompts: Self.prompts, samplers: Self.prompts.map { _ in FirstPositiveSampler() })
            #expect(try collect(&iterator, uids: uids) == expected)
        }
    }

    @Test("A one-token prompt joins a longer one on a sliding-window model")
    func mixedLengthRotatingRows() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyGemma3()
            let prompts: [[Int]] = [[1], [2, 3]]
            let expected = try prompts.map { prompt in
                try generateAlone(model, prompt: prompt, sampler: ArgMaxSampler(), maxTokens: 4)
            }
            let actual = try batchGenerate(model: model, prompts: prompts, maxTokens: 4)
            #expect(actual == expected)
        }
    }

    @Test("Prefill progress ends at (total, total) for short and chunked prompts alike")
    func prefillProgressCompletes() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyLlama()
            final class Log: @unchecked Sendable { var events: [[Int]] = [] }
            let log = Log()
            var prefill = PrefillParameters(stepSize: 8)
            prefill.progress = { log.events.append([$0, $1]) }
            let prompts: [[Int]] = [
                [1, 2, 3], Array(repeating: [4, 5, 6, 7], count: 5).flatMap { $0 },
            ]

            _ = try batchGenerate(model: model, prompts: prompts, maxTokens: 1, prefill: prefill)

            // Prompts prefill in order: the short one reports only its completion, the
            // chunked one its chunks and then its completion, all against its own total.
            let short = log.events.prefix { $0[1] == 3 }
            let long = log.events.dropFirst(short.count)
            #expect(Array(short) == [[3, 3]])
            #expect(long.count > 1)
            #expect(long.allSatisfy { $0[1] == 20 })
            #expect(long.last == [20, 20])
            #expect(long.map { $0[0] } == long.map { $0[0] }.sorted())
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

private func tinyMamba2() throws -> Mamba2Model {
    let configuration = try JSONDecoder().decode(
        Mamba2Configuration.self,
        from: Data(
            """
            {
              "model_type": "mamba2", "num_heads": 4, "head_dim": 4, "vocab_size": 32,
              "hidden_size": 16, "state_size": 8, "num_hidden_layers": 2,
              "layer_norm_epsilon": 1e-5, "conv_kernel": 4, "n_groups": 1, "use_bias": false,
              "use_conv_bias": true, "tie_word_embeddings": false, "time_step_limit": [0.0, 100.0]
            }
            """.utf8))
    return withRandomState(MLXRandom.RandomState(seed: 601)) {
        let model = Mamba2Model(configuration)
        eval(model)
        return model
    }
}

private func tinyFalconH1() throws -> FalconH1Model {
    let configuration = try JSONDecoder().decode(
        FalconH1Configuration.self,
        from: Data(
            """
            {
              "model_type": "falcon_h1", "hidden_size": 32, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
              "vocab_size": 100, "mamba_d_ssm": 16, "mamba_d_state": 8, "mamba_d_head": 8,
              "mamba_n_heads": 2, "mamba_n_groups": 1, "mamba_d_conv": 4, "mamba_chunk_size": 64,
              "attention_in_multiplier": 2.0, "key_multiplier": 0.5
            }
            """.utf8))
    return withRandomState(MLXRandom.RandomState(seed: 602)) {
        let model = FalconH1Model(configuration)
        eval(model)
        return model
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
