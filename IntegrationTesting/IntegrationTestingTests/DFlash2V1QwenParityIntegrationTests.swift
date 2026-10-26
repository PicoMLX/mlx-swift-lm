// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
import Tokenizers

// MARK: - DFlash v1 drafter against Qwen3.5-4B
//
// Loads z-lab's DFlash v1 drafter for Qwen3.5-4B, checks that every tensor
// of the checkpoint lands in the model, and compares greedy DFlash decoding
// against plain `TokenIterator` decoding on three prompts at every block
// size a caller may ask for.

private let targetModelId = "mlx-community/Qwen3.5-4B-bf16"
private let targetRevision = "491fdc7c087ba7fb48adcb1253f8e76d011db783"
private let drafterModelId = "z-lab/Qwen3.5-4B-DFlash"
private let drafterRevision = "1905b5a05f974e82798be3ea0fd6baf2e4805a1c"

/// The pair is ~10GB. As the MTP suites do, run only when both snapshots are
/// already in the HF cache so a cold run does not spend its budget downloading.
private let hasPair: Bool =
    hfSnapshotDir(modelId: targetModelId, revision: targetRevision) != nil
    && hfSnapshotDir(modelId: drafterModelId, revision: drafterRevision) != nil

private let downloader: any Downloader = #hubDownloader()

private struct LoadedPair {
    let target: ModelContext
    let drafter: any DFlash2DrafterModel
}

nonisolated(unsafe) private var sharedPair: Result<LoadedPair, Error>?

/// One load for the whole serialized suite.
private func loadPair() async throws -> LoadedPair {
    if let sharedPair {
        return try sharedPair.get()
    }
    do {
        let target = try await LLMModelFactory.shared.load(
            from: downloader, using: #huggingFaceTokenizerLoader(),
            configuration: .init(id: targetModelId, revision: targetRevision))
        await DFlash2Registration.register()
        let drafter = try await DFlash2DrafterModelFactory.shared.load(
            from: downloader, using: #huggingFaceTokenizerLoader(),
            configuration: .init(id: drafterModelId, revision: drafterRevision))
        let pair = LoadedPair(target: target, drafter: drafter.model)
        sharedPair = .success(pair)
        return pair
    } catch {
        sharedPair = .failure(error)
        throw error
    }
}

private let prompts: [(name: String, text: String)] = [
    ("math", "Compute 23 * 47 step by step, then check the result by dividing it by 47."),
    (
        "code",
        "Write a Swift function that returns the n-th Fibonacci number iteratively, "
            + "with a short doc comment."
    ),
    ("chat", "Why is the sky blue? Explain briefly."),
]

private let blockSizes = [4, 5, 8, 16]

private func decode(_ iterator: inout some TokenIteratorProtocol) -> [Int] {
    var tokens: [Int] = []
    while let token = iterator.next() {
        tokens.append(token)
    }
    return tokens
}

/// The target's own next-token distribution after `prompt + prefix`: the log
/// probability gap between `plain` and `speculative`, and `speculative`'s rank.
private func nearTie(
    target: any DFlash2TargetModel, prompt: MLXArray, prefix: [Int], plain: Int,
    speculative: Int
) throws -> (gap: Float, rank: Int) {
    let tokens = concatenated([prompt.asType(.int32), MLXArray(prefix.map { Int32($0) })])
        .expandedDimensions(axis: 0)
    let cache = try target.newCache(parameters: nil)
    let logits = target.dflash2Prefill(tokens, cache: cache, captureLayers: []).logits
    let logProbs = logSoftmax(logits[0, -1, 0...].asType(.float32), axis: -1)
    let gap = (logProbs[plain] - logProbs[speculative]).item(Float.self)
    let rank = (logProbs .> logProbs[speculative]).sum().item(Int.self)
    return (gap, rank)
}

@Suite(.serialized)
struct DFlash2V1QwenParityIntegrationTests {

    /// Every tensor of the checkpoint is a parameter of the loaded drafter.
    @Test(.enabled(if: hasPair))
    func drafterLoadsEveryTensor() async throws {
        let pair = try await loadPair()
        let drafter = try #require(pair.drafter as? DFlash2DraftModel)
        #expect(drafter.config.isDFlash2 == false)
        #expect(drafter.blockSize == 16)
        #expect(drafter.maskTokenId == 248077)
        #expect(drafter.targetLayerIds == [1, 5, 9, 13, 17, 21, 25, 29])
        #expect(drafter.contextWindow == .max)

        let directory = try #require(
            hfSnapshotDir(modelId: drafterModelId, revision: drafterRevision))
        let checkpoint = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        let parameters = Dictionary(uniqueKeysWithValues: drafter.parameters().flattened())
        #expect(Set(checkpoint.keys) == Set(parameters.keys))
        #expect(checkpoint.count == 69)
        for (name, value) in checkpoint {
            #expect(parameters[name]?.shape == value.shape, Comment(rawValue: name))
        }
    }

    /// Greedy DFlash decoding matches plain decoding token for token, or
    /// diverges only at a near-tie of the target's own distribution, and every
    /// run accepts drafts.
    @Test(.enabled(if: hasPair), arguments: blockSizes)
    func greedyParityWithPlainDecoding(blockSize: Int) async throws {
        let pair = try await loadPair()
        let target = try #require(pair.target.model as? DFlash2TargetModel)
        let parameters = GenerateParameters(maxTokens: 128, temperature: 0)
        for prompt in prompts {
            let input = try await pair.target.processor.prepare(
                input: UserInput(chat: [.user(prompt.text)]))

            var plain = try TokenIterator(
                input: input, model: pair.target.model, parameters: parameters)
            let expected = decode(&plain)

            var speculative = try DFlash2SpeculativeTokenIterator(
                input: input, mainModel: pair.target.model, drafter: pair.drafter,
                parameters: parameters, blockSize: blockSize)
            let produced = decode(&speculative)
            speculative.finalizeGeneration()

            let telemetry = try #require(speculative.speculativeDecodingTelemetry)
            let rounds = telemetry.roundCount
            let accepted = telemetry.acceptedDraftTokenCount
            let meanAcceptance = Double(accepted + rounds) / Double(max(rounds, 1))
            print(
                "DFlash v1 \(prompt.name) block \(blockSize): \(produced.count) tokens in "
                    + "\(rounds) rounds, \(accepted)/\(telemetry.draftTokenCount) drafts "
                    + "accepted, mean acceptance length "
                    + String(format: "%.2f", meanAcceptance))
            #expect(produced.count == expected.count, "\(prompt.name) block \(blockSize)")
            #expect(accepted > 0, "\(prompt.name) block \(blockSize) accepted no draft")
            if prompt.name == "math" {
                #expect(meanAcceptance >= 2.0, "math block \(blockSize)")
            }

            guard
                let index = zip(produced, expected).enumerated().first(where: { $0.1.0 != $0.1.1 })?
                    .offset
            else {
                #expect(produced == expected, "\(prompt.name) block \(blockSize)")
                continue
            }
            // A verify pass sums in another order than a one-token step, so the
            // streams may part where the target's top two nearly tie.
            let tie = try nearTie(
                target: target, prompt: input.text.tokens, prefix: Array(expected.prefix(index)),
                plain: expected[index], speculative: produced[index])
            print(
                "DFlash v1 \(prompt.name) block \(blockSize): first divergence at token "
                    + "\(index), plain \(expected[index]) vs DFlash \(produced[index]), "
                    + "log-probability gap \(tie.gap), DFlash token rank \(tie.rank)")
            #expect(tie.rank <= 1, "\(prompt.name) block \(blockSize): not a near-tie")
            #expect(abs(tie.gap) < 0.5, "\(prompt.name) block \(blockSize): gap \(tie.gap)")
        }
    }
}
