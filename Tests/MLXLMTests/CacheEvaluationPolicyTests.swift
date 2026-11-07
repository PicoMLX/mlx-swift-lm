// Copyright © 2026 Apple Inc.

import Cmlx
import MLX
import MLXLLM
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct CacheEvaluationPolicyTests {
    @Test(
        "Forced processors preserve the model's attention dependencies", arguments: [false, true])
    func forcedProcessorsPreserveAttentionRoots(batched: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            for policy in [KVCacheEvaluationPolicy.allState, .attentionDependencies] {
                for kind in EvaluationForcedProcessorKind.allCases {
                    let processor = try kind.makeProcessor()
                    try expectSeveredAttentionSettles(
                        batched: batched, policy: policy, processor: processor,
                        sampler: ArgMaxSampler(), expectedToken: kind == .thinkingBudget ? 2 : 0)
                }
            }
        }
    }

    @Test(
        "A constant custom sampler preserves the model's attention dependencies",
        arguments: [false, true])
    func constantSamplerPreservesAttentionRoots(batched: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            for policy in [KVCacheEvaluationPolicy.allState, .attentionDependencies] {
                try expectSeveredAttentionSettles(
                    batched: batched, policy: policy, processor: nil,
                    sampler: EvaluationConstantSampler(), expectedToken: 0)
            }
        }
    }

    @Test("A final decode settles newly converted affine cache state", arguments: [4, 8])
    func delayedAffineConversionSettles(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            for policy in [KVCacheEvaluationPolicy.allState, .attentionDependencies] {
                let model = EvaluationAttentionModel(bits: 0)
                let parameters = GenerateParameters(
                    maxTokens: 1, kvBits: bits, kvGroupSize: 64, quantizedKVStart: 3,
                    temperature: 0, cacheEvaluationPolicy: policy)
                var iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray([Int32(1), 2, 3])), model: model,
                    parameters: parameters)
                #expect(iterator.cache.allSatisfy { type(of: $0) == KVCacheSimple.self })
                #expect(iterator.next() != nil)
                Stream.defaultStream.synchronize()
                #expect(iterator.next() == nil)
                for layer in iterator.cache {
                    let quantized = try #require(layer as? QuantizedKVCache)
                    #expect(quantized.bits == bits)
                    expectEvaluationAvailable(quantized.innerState())
                }
            }
        }
    }

    @Test("Default iterators settle attention writes even when logits do not consume them")
    func defaultSettlesIndependentAttention() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = EvaluationAuxiliaryModel(kind: .attention)
            var single = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(1), 2])), model: model,
                processor: nil, sampler: ArgMaxSampler(), maxTokens: 8)
            for _ in 0 ..< 4 {
                #expect(single.next() != nil)
                expectEvaluationAvailable(model.observed)
            }
            var batch = BatchTokenIterator(model: model, maxTokens: 8)
            _ = batch.insert(prompts: [[1, 2], [3, 4, 5]])
            for _ in 0 ..< 4 {
                let responses = try batch.next()
                #expect(responses.count == 2)
                expectEvaluationAvailable(model.observed)
            }
        }
    }

    @Test("Both policies settle independent recurrent, nested and custom-subclass state")
    func independentStateSettles() throws {
        try Device.withDefaultDevice(.cpu) {
            for policy in [KVCacheEvaluationPolicy.allState, .attentionDependencies] {
                for kind in [EvaluationAuxiliaryModel.Kind.recurrent, .nested, .custom] {
                    let model = EvaluationAuxiliaryModel(kind: kind)
                    var single = try TokenIterator(
                        input: LMInput(tokens: MLXArray([Int32(1), 2])), model: model,
                        processor: nil, sampler: ArgMaxSampler(), maxTokens: 8,
                        cacheEvaluationPolicy: policy)
                    for _ in 0 ..< 4 {
                        #expect(single.next() != nil)
                        expectEvaluationAvailable(model.observed)
                    }
                    var batch = BatchTokenIterator(
                        model: model, maxTokens: 8, cacheEvaluationPolicy: policy)
                    // The custom subclass implements its own B1 merge; ordinary recurrent
                    // caches also exercise merging two independently prepared rows.
                    let prompts = kind == .custom ? [[1, 2]] : [[1, 2], [3, 4, 5]]
                    _ = batch.insert(prompts: prompts)
                    for _ in 0 ..< 4 {
                        let responses = try batch.next()
                        #expect(responses.count == prompts.count)
                        expectEvaluationAvailable(model.observed)
                    }
                }
            }
        }
    }

    @Test("Opt-in rotating metadata settles across wraps, row removal and admission")
    func rotatingMetadataLifecycle() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = EvaluationAttentionModel(bits: 0, window: 8)
            var iterator = BatchTokenIterator(
                model: model, maxTokens: 64, cacheEvaluationPolicy: .attentionDependencies)
            let ids = iterator.insert(prompts: [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10]])
            var positions = [ids[0]: 3, ids[1]: 7]
            for step in 0 ..< 24 {
                if step == 10 {
                    iterator.remove([ids[0]])
                    positions.removeValue(forKey: ids[0])
                    let added = iterator.insert(prompts: [[11, 12]])
                    positions[added[0]] = 2
                }
                let responses = try iterator.next()
                for response in responses { positions[response.uid]! += 1 }
                #expect(responses.count == 2)
                for layer in model.lastCache {
                    let cache = try #require(layer as? BatchRotatingKVCache)
                    expectEvaluationAvailable([cache.batchOffset, cache.leftPadding])
                    #expect(
                        cache.batchOffset.asArray(Int32.self)
                            == iterator.uids.map { Int32(positions[$0]!) })
                    #expect(cache.leftPadding.asArray(Int32.self).allSatisfy { $0 >= 0 })
                    #expect(cache.innerState().allSatisfy { $0.dim(2) <= 8 })
                }
            }
        }
    }

    @Test(
        "Attention dependencies preserve greedy and seeded full/affine generation",
        arguments: [0, 4, 8])
    func attentionTokenParity(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = EvaluationAttentionModel(bits: bits)
            for seeded in [false, true] {
                func generate(_ policy: KVCacheEvaluationPolicy) throws -> [[Int]] {
                    var iterator = BatchTokenIterator(
                        model: model, maxTokens: 8, cacheEvaluationPolicy: policy)
                    func sampler(_ row: Int) -> any LogitSampler {
                        if seeded {
                            return TopPSampler(
                                temperature: 0.7, topP: 0.9, topK: 8, seed: UInt64(7 + row))
                        }
                        return ArgMaxSampler()
                    }
                    let ids = iterator.insert(
                        prompts: [[1, 2, 3], [4, 5, 6, 7, 8]], samplers: [sampler(0), sampler(1)])
                    var tokens = [[Int]](repeating: [], count: 2)
                    while true {
                        let responses = try iterator.next()
                        if responses.isEmpty { break }
                        for response in responses {
                            tokens[ids.firstIndex(of: response.uid)!].append(response.token)
                        }
                    }
                    Stream.defaultStream.synchronize()
                    return tokens
                }
                let reference = try generate(.allState)
                let actual = try generate(.attentionDependencies)
                #expect(actual == reference)
                #expect(actual.map(\.count) == [8, 8])
            }
        }
    }

    @Test("Parameter-based opt-in preserves deferred chosen and top log probabilities")
    func deferredLogProbabilityParity() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = EvaluationAttentionModel(bits: 0)
            func generate(_ policy: KVCacheEvaluationPolicy) throws -> (
                [Int], [GenerateTokenLogProbabilities]
            ) {
                let parameters = GenerateParameters(
                    maxTokens: 5, temperature: 0.7, topP: 0.9, logProbabilities: 3, seed: 41,
                    cacheEvaluationPolicy: policy)
                var iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray([Int32(1), 2, 3])), model: model,
                    parameters: parameters)
                var tokens: [Int] = []
                var probabilities: [GenerateTokenLogProbabilities] = []
                var first: DeferredTokenLogProbabilities?
                while let token = iterator.next() {
                    let deferred = try #require(iterator.lastLogProbabilities)
                    if first == nil { first = deferred }
                    let values = deferred.materialize()
                    #expect(values.chosen.token == token)
                    #expect(values.topLogProbabilities.count == 3)
                    tokens.append(token)
                    probabilities.append(values)
                }
                #expect(try #require(first).materialize() == probabilities.first)
                return (tokens, probabilities)
            }
            let reference = try generate(.allState)
            let actual = try generate(.attentionDependencies)
            #expect(actual.0 == reference.0)
            #expect(actual.1 == reference.1)
            #expect(actual.0.count == 5)
        }
    }
}

private func expectEvaluationAvailable(
    _ arrays: [MLXArray], sourceLocation: SourceLocation = #_sourceLocation
) {
    Stream.defaultStream.synchronize()
    #expect(!arrays.isEmpty, sourceLocation: sourceLocation)
    for array in arrays {
        var available = false
        let status = _mlx_array_is_available(&available, array.ctx)
        #expect(status == 0, sourceLocation: sourceLocation)
        #expect(
            available, "the independent state must already be materialized",
            sourceLocation: sourceLocation)
    }
}

private func expectSeveredAttentionSettles(
    batched: Bool, policy: KVCacheEvaluationPolicy, processor: (any LogitProcessor)?,
    sampler: any LogitSampler, expectedToken: Int
) throws {
    let model = EvaluationAttentionModel(bits: 0)
    if batched {
        var iterator = BatchTokenIterator(
            model: model, sampler: sampler, processor: processor, maxTokens: 3,
            cacheEvaluationPolicy: policy)
        _ = iterator.insert(prompts: [[5, 1], [6, 7, 1]])
        let responses = try iterator.next()
        #expect(responses.map(\.token) == [expectedToken, expectedToken])
    } else {
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(5), 1])), model: model,
            processor: processor, sampler: sampler, maxTokens: 3,
            cacheEvaluationPolicy: policy)
        #expect(iterator.next() == expectedToken)
    }
    // Captured during the original forward, before processing or sampling can replace
    // its logits. Checking these descriptors must not construct or evaluate new views.
    expectEvaluationAvailable(model.lastAttentionBuffers)
}

private enum EvaluationForcedProcessorKind: CaseIterable {
    case replacement, inPlace, thinkingBudget

    func makeProcessor() throws -> any LogitProcessor {
        switch self {
        case .replacement: EvaluationForcedProcessor(mutateInput: false)
        case .inPlace: EvaluationForcedProcessor(mutateInput: true)
        case .thinkingBudget:
            try ThinkingBudgetProcessor(
                configuration: ThinkingBudgetConfiguration(
                    maximumTokenCount: 0, minimumAnswerTokenCount: 0),
                reasoning: ReasoningConfig(
                    startDelimiter: "A", endDelimiter: "BC", promptStrategy: .none,
                    budgetTransition: .immediate),
                tokenizer: EvaluationBudgetTokenizer())
        }
    }
}

private struct EvaluationForcedProcessor: LogitProcessor {
    let mutateInput: Bool
    mutating func prompt(_ prompt: MLXArray) {}
    mutating func didSample(token: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        let forced = MLXArray.full(logits.shape, values: MLXArray(Float(0)), dtype: logits.dtype)
        if mutateInput {
            logits[.ellipsis] = forced
            return logits
        }
        return forced
    }
}

private struct EvaluationConstantSampler: LogitSampler {
    func sample(logits: MLXArray) -> MLXArray {
        MLXArray.zeros([logits.dim(0)], dtype: .int32)
    }
}

/// Alphabet tokens keep the real two-token forced transition inside the tiny vocabulary.
private struct EvaluationBudgetTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map { Int($0) - 64 } }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map { UInt8($0 + 64) }, as: UTF8.self)
    }
    func convertTokenToId(_ token: String) -> Int? {
        token.utf8.count == 1 ? token.utf8.first.map { Int($0) - 64 } : nil
    }
    func convertIdToToken(_ id: Int) -> String? { decode(tokenIds: [id], skipSpecialTokens: false) }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// Its extra state is deliberately absent from the token's dependency graph. It is its own
/// one-row batch, so `mergeCaches` merges it through its own class.
private final class EvaluationCustomCache: KVCacheSimple, BatchableKVCache {
    var auxiliary = MLXArray.zeros([1, 4])

    override var state: [MLXArray] {
        get { super.state + [auxiliary] }
        set {
            auxiliary = newValue.last!
            if newValue.count > 1 { super.state = Array(newValue.dropLast()) }
        }
    }

    override func copy() -> any KVCache {
        let copy = EvaluationCustomCache()
        copy.keys = keys?[.ellipsis]
        copy.values = values?[.ellipsis]
        copy.offset = offset
        copy.auxiliary = auxiliary[.ellipsis]
        return copy
    }

    static func merge(_ caches: [any KVCache]) throws -> any BatchableKVCache {
        guard caches.count == 1, let cache = caches.first as? EvaluationCustomCache,
            let copy = cache.copy() as? EvaluationCustomCache
        else {
            throw KVCacheError(message: "The test custom cache supports one row")
        }
        return copy
    }

    func filter(batchIndices: MLXArray) {
        precondition(
            batchIndices.asArray(Int32.self) == [0], "The test custom cache holds one row")
    }

    func extend(other: any KVCache) {
        preconditionFailure("The test custom cache holds one row")
    }

    func validateExtension(other: any KVCache) throws {
        throw KVCacheError(message: "The test custom cache holds one row")
    }

    func extract(_ index: Int) -> any KVCache {
        precondition(index == 0, "The test custom cache holds one row")
        return copy()
    }
}

private final class EvaluationAuxiliaryModel: Module, LanguageModel {
    enum Kind { case attention, recurrent, nested, custom }
    let kind: Kind
    var observed: [MLXArray] = []

    init(kind: Kind) {
        self.kind = kind
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let tokens = input.tokens
        observed = []
        func update(_ cache: KVCache) {
            if let list = cache as? CacheList {
                list.children.forEach(update)
            } else if let arrays = cache as? ArraysCache {
                for slot in 0 ..< arrays.slotCount {
                    let previous = arrays[slot] ?? MLXArray.zeros([tokens.dim(0), 4])
                    let next = previous + Float(slot + 1)
                    arrays[slot] = next
                    observed.append(next)
                }
                arrays.offset += tokens.dim(1)
            } else {
                let keys = tokens.asType(.float32).reshaped([tokens.dim(0), 1, tokens.dim(1), 1])
                _ = cache.update(keys: keys, values: keys)
                if let custom = cache as? EvaluationCustomCache {
                    custom.auxiliary = custom.auxiliary + 1
                    observed.append(custom.auxiliary)
                } else {
                    observed += cache.innerState()
                }
            }
        }
        cache?.forEach(update)
        let target = (tokens[0..., -1] + Int32(1)) % Int32(16)
        let logits = MLX.where(
            MLXArray.arange(16)[.newAxis, .newAxis] .== target[0..., .newAxis, .newAxis],
            MLXArray(Float(5)), MLXArray(Float(-5)))
        return LMOutput(logits: logits)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        switch kind {
        case .attention: [KVCacheSimple()]
        case .recurrent: [MambaCache()]
        case .nested: [CacheList(ArraysCache(size: 2), CacheList(MambaCache()))]
        case .custom: [EvaluationCustomCache()]
        }
    }
}

private final class EvaluationAttentionModel: Module, LanguageModel {
    let model: LlamaModel
    let bits: Int
    let window: Int?
    var lastCache: [KVCache] = []
    var lastAttentionBuffers: [MLXArray] = []

    init(bits: Int, window: Int? = nil) {
        self.bits = bits
        self.window = window
        model = withRandomState(MLXRandom.RandomState(seed: 804)) {
            LlamaModel(
                LlamaConfiguration(
                    hiddenSize: 128, hiddenLayers: 2, intermediateSize: 128, attentionHeads: 2,
                    rmsNormEps: 1e-5, vocabularySize: 32, kvHeads: 1))
        }
        super.init()
        eval(model)
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        lastCache = cache ?? []
        let logits = model(input.tokens, cache: cache)
        lastAttentionBuffers = lastCache.flatMap { $0.innerState() }
        return LMOutput(logits: logits)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< 2).map { _ in
            if let window { return RotatingKVCache(maxSize: window, step: 4) }
            if bits != 0 { return QuantizedKVCache(groupSize: 64, bits: bits) }
            return KVCacheSimple()
        }
    }
}
