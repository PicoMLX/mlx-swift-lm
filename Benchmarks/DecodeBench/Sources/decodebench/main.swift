// Copyright © 2026 Apple Inc.

// Greedy decode throughput on a local model, as a median over rounds: the single-sequence
// `TokenIterator`, and `BatchTokenIterator` at one and four rows. Decode is timed after the
// prefill and the first token. The single and the one-row batch must decode the same tokens,
// and every round must repeat them.
//
//   swift run -c release --package-path Benchmarks/DecodeBench decodebench <model directory> \
//       [decode tokens: 128] [rounds: 5] [prompt tokens: 145]

import BenchmarkHelpers
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

let arguments = CommandLine.arguments
guard arguments.count > 1 else {
    print("usage: decodebench <model directory> [decode tokens] [rounds] [prompt tokens]")
    exit(1)
}
let modelDirectory = URL(fileURLWithPath: arguments[1])
let decodeTokens = arguments.count > 2 ? Int(arguments[2]) ?? 128 : 128
let rounds = arguments.count > 3 ? Int(arguments[3]) ?? 5 : 5
let promptLength = arguments.count > 4 ? Int(arguments[4]) ?? 145 : 145

let context = try await LLMModelFactory.shared.load(
    from: modelDirectory, using: NoOpTokenizerLoader())
let model = context.model
print("model: \(modelDirectory.lastPathComponent)")
print("prompt tokens: \(promptLength), decode tokens: \(decodeTokens), rounds: \(rounds)")

/// A fixed pseudo-random prompt, so every round and every checkout decodes the same tokens.
@MainActor func prompt(seed: UInt64) -> [Int32] {
    var state = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return (0 ..< promptLength).map { _ in
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int32(1000 + (state >> 33) % 20000)
    }
}

func seconds(since start: ContinuousClock.Instant) -> Double {
    let elapsed = ContinuousClock.now - start
    return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
}

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count % 2 == 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
}

/// Prefills the prompt, then times `decodeTokens` greedy steps of `TokenIterator`.
@MainActor func runSingle(prompt: [Int32]) throws -> (tokensPerSecond: Double, tokens: [Int]) {
    var iterator = try TokenIterator(
        input: LMInput(tokens: MLXArray(prompt)), model: model, processor: nil,
        sampler: ArgMaxSampler(), maxTokens: decodeTokens + 1)
    var tokens: [Int] = []
    // The first token comes from the prefill; the steps after it are the decode.
    if let first = iterator.next() { tokens.append(first) }
    let start = ContinuousClock.now
    while let token = iterator.next() { tokens.append(token) }
    return (Double(decodeTokens) / seconds(since: start), tokens)
}

/// Queues the prompts in a `BatchTokenIterator`, takes the first step (the prefill and the
/// first token) and times `decodeTokens` further steps. Tokens per second count one row.
@MainActor func runBatched(prompts: [[Int32]]) throws -> (tokensPerSecond: Double, tokens: [[Int]])
{
    var iterator = BatchTokenIterator(model: model, maxTokens: decodeTokens + 1)
    let uids = iterator.insert(prompts: prompts.map { $0.map { Int($0) } })
    var tokens = [[Int]](repeating: [], count: prompts.count)
    func record(_ responses: [BatchTokenIterator.Response]) {
        for response in responses {
            tokens[uids.firstIndex(of: response.uid)!].append(response.token)
        }
    }
    record(try iterator.next())
    let start = ContinuousClock.now
    while true {
        let responses = try iterator.next()
        if responses.isEmpty { break }
        record(responses)
    }
    return (Double(decodeTokens) / seconds(since: start), tokens.map { Array($0.dropFirst()) })
}

let singlePrompt = prompt(seed: 1)
let fourPrompts = (1 ... 4).map { prompt(seed: UInt64($0)) }

// Warm up every path once so kernel compilation is not timed.
_ = try runSingle(prompt: singlePrompt)
_ = try runBatched(prompts: [singlePrompt])
_ = try runBatched(prompts: fourPrompts)

var single: [Double] = []
var batchOfOne: [Double] = []
var batchOfFour: [Double] = []
var reference: [Int]? = nil
var fourReference: [[Int]]? = nil
var mismatchedRounds: [Int] = []
for round in 0 ..< rounds {
    // Alternate the order so a thermal or background-load slope lands on both paths.
    let batchedFirst = round % 2 == 1
    let batched = batchedFirst ? try runBatched(prompts: [singlePrompt]) : nil
    let plain = try runSingle(prompt: singlePrompt)
    let one = try batched ?? runBatched(prompts: [singlePrompt])
    let four = try runBatched(prompts: fourPrompts)

    single.append(plain.tokensPerSecond)
    batchOfOne.append(one.tokensPerSecond)
    batchOfFour.append(four.tokensPerSecond)
    if reference == nil { reference = plain.tokens }
    if fourReference == nil { fourReference = four.tokens }
    let expected = Array(plain.tokens.dropFirst())
    if plain.tokens != reference || one.tokens[0] != expected || four.tokens[0] != expected
        || four.tokens != fourReference
    {
        mismatchedRounds.append(round + 1)
    }
    print(
        String(
            format:
                "round %d  single %.2f tok/s   batch(1) %.2f tok/s   batch(4) %.2f tok/s per row, %.2f total   (%@ first)",
            round + 1, plain.tokensPerSecond, one.tokensPerSecond, four.tokensPerSecond,
            4 * four.tokensPerSecond, batchedFirst ? "batched" : "single"))
    Memory.clearCache()
}
print(
    String(
        format:
            "median  single %.2f tok/s   batch(1) %.2f tok/s   batch(4) %.2f tok/s per row, %.2f total",
        median(single), median(batchOfOne), median(batchOfFour), 4 * median(batchOfFour)))
print(
    "tokens identical across paths and rounds: \(mismatchedRounds.isEmpty)"
        + (mismatchedRounds.isEmpty ? "" : ", not in rounds \(mismatchedRounds)"))
