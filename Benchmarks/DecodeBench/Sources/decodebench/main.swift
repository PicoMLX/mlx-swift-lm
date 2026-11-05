// Copyright © 2026 Apple Inc.

// Decode throughput and time to first token on a local model, as medians over rounds: the
// single-sequence `TokenIterator`, and `BatchTokenIterator` at one and four rows. Decode is
// timed after the prefill and the first token; the time to first token covers the prefill.
// Every path must repeat its tokens in every round; how far the paths agree with each other
// is reported.
//
//   swift run -c release --package-path Benchmarks/DecodeBench decodebench <model directory> \
//       [decode tokens: 128] [rounds: 5] [prompt tokens: 145] [sampler: greedy|topp|categorical]
//
// Built with `-Xswiftc -DSINGLE_ONLY` it times `TokenIterator` alone, so it also builds
// against a checkout without `BatchTokenIterator`.

import BenchmarkHelpers
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

let arguments = CommandLine.arguments
guard arguments.count > 1 else {
    print("usage: decodebench <model directory> [decode tokens] [rounds] [prompt tokens] [sampler]")
    exit(1)
}
let modelDirectory = URL(fileURLWithPath: arguments[1])
let decodeTokens = arguments.count > 2 ? Int(arguments[2]) ?? 128 : 128
let rounds = arguments.count > 3 ? Int(arguments[3]) ?? 5 : 5
let promptLength = arguments.count > 4 ? Int(arguments[4]) ?? 145 : 145
let samplerName = arguments.count > 5 ? arguments[5] : "greedy"

let context = try await LLMModelFactory.shared.load(
    from: modelDirectory, using: NoOpTokenizerLoader())
let model = context.model
print("model: \(modelDirectory.lastPathComponent)")
print(
    "prompt tokens: \(promptLength), decode tokens: \(decodeTokens), rounds: \(rounds), "
        + "sampler: \(samplerName)")

/// A fresh sampler per run, seeded, so every run of a path draws the same tokens.
@MainActor func makeSampler() -> LogitSampler {
    switch samplerName {
    case "topp": return TopPSampler(temperature: 0.8, topP: 0.9, seed: 7)
    case "categorical": return CategoricalSampler(temperature: 0.8, seed: 7)
    case "greedy": return ArgMaxSampler()
    default:
        print("unknown sampler \(samplerName): greedy, topp or categorical")
        exit(1)
    }
}

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

struct Run {
    /// Decode tokens per second, counting one row.
    let tokensPerSecond: Double
    /// Seconds from the start of the run to the first token of every row: the prefill.
    let firstToken: Double
    /// Every row's tokens, the first one included.
    let tokens: [[Int]]
}

/// Prefills the prompt, then times `decodeTokens` steps of `TokenIterator`.
@MainActor func runSingle(prompt: [Int32]) throws -> Run {
    let start = ContinuousClock.now
    var iterator = try TokenIterator(
        input: LMInput(tokens: MLXArray(prompt)), model: model, processor: nil,
        sampler: makeSampler(), maxTokens: decodeTokens + 1)
    var tokens: [Int] = []
    // The first token comes from the prefill; the steps after it are the decode.
    if let first = iterator.next() { tokens.append(first) }
    let firstToken = seconds(since: start)
    let decodeStart = ContinuousClock.now
    while let token = iterator.next() { tokens.append(token) }
    return Run(
        tokensPerSecond: Double(decodeTokens) / seconds(since: decodeStart),
        firstToken: firstToken, tokens: [tokens])
}

#if !SINGLE_ONLY
/// Queues the prompts in a `BatchTokenIterator`, takes the first step (the prefill and
/// the first token) and times `decodeTokens` further steps.
@MainActor func runBatched(prompts: [[Int32]]) throws -> Run {
    let start = ContinuousClock.now
    var iterator = BatchTokenIterator(
        model: model, sampler: makeSampler(), maxTokens: decodeTokens + 1)
    let uids = iterator.insert(prompts: prompts.map { $0.map { Int($0) } })
    var tokens = [[Int]](repeating: [], count: prompts.count)
    func record(_ responses: [BatchTokenIterator.Response]) {
        for response in responses {
            tokens[uids.firstIndex(of: response.uid)!].append(response.token)
        }
    }
    record(try iterator.next())
    let firstToken = seconds(since: start)
    let decodeStart = ContinuousClock.now
    while true {
        let responses = try iterator.next()
        if responses.isEmpty { break }
        record(responses)
    }
    return Run(
        tokensPerSecond: Double(decodeTokens) / seconds(since: decodeStart),
        firstToken: firstToken, tokens: tokens)
}
#endif

let singlePrompt = prompt(seed: 1)
let fourPrompts = (1 ... 4).map { prompt(seed: UInt64($0)) }

// Warm up every path once so kernel compilation is not timed.
_ = try runSingle(prompt: singlePrompt)
#if !SINGLE_ONLY
_ = try runBatched(prompts: [singlePrompt])
_ = try runBatched(prompts: fourPrompts)
#endif

/// The tokens of a path in its first round, to check every later round against.
var references: [String: [[Int]]] = [:]
var mismatchedRounds: [Int] = []
@MainActor func check(_ path: String, _ run: Run, round: Int) {
    if let reference = references[path] {
        if run.tokens != reference, !mismatchedRounds.contains(round) {
            mismatchedRounds.append(round)
        }
    } else {
        references[path] = run.tokens
    }
}
@MainActor func format(_ run: Run, rows: Int) -> String {
    let perRow = String(format: "%.2f tok/s", run.tokensPerSecond)
    let rate =
        rows == 1
        ? perRow
        : perRow + String(format: " per row, %.2f total", Double(rows) * run.tokensPerSecond)
    return rate + String(format: " (ttft %.0f ms)", 1000 * run.firstToken)
}

var single: [Double] = []
var singleFirst: [Double] = []
var batchOfOne: [Double] = []
var batchOfOneFirst: [Double] = []
var batchOfFour: [Double] = []
var batchOfFourFirst: [Double] = []
var agreement: [Int] = []
var rowAgreement: [Int] = []
for round in 0 ..< rounds {
    #if SINGLE_ONLY
    let plain = try runSingle(prompt: singlePrompt)
    single.append(plain.tokensPerSecond)
    singleFirst.append(plain.firstToken)
    check("single", plain, round: round + 1)
    print("round \(round + 1)  single \(format(plain, rows: 1))")
    #else
    // Alternate the order so a thermal or background-load slope lands on both paths.
    let batchedFirst = round % 2 == 1
    let batched = batchedFirst ? try runBatched(prompts: [singlePrompt]) : nil
    let plain = try runSingle(prompt: singlePrompt)
    let one = try batched ?? runBatched(prompts: [singlePrompt])
    let four = try runBatched(prompts: fourPrompts)

    single.append(plain.tokensPerSecond)
    singleFirst.append(plain.firstToken)
    batchOfOne.append(one.tokensPerSecond)
    batchOfOneFirst.append(one.firstToken)
    batchOfFour.append(four.tokensPerSecond)
    batchOfFourFirst.append(four.firstToken)
    // Each path must repeat itself across rounds exactly. Between paths the arithmetic
    // can differ (four rows run other matmul shapes than one, and a sampler draws a
    // batch from one random stream), so their streams can part ways; how far they agree
    // is reported, not asserted.
    check("single", plain, round: round + 1)
    check("batch(1)", one, round: round + 1)
    check("batch(4)", four, round: round + 1)
    agreement.append(zip(plain.tokens[0], one.tokens[0]).prefix { $0 == $1 }.count)
    rowAgreement.append(zip(one.tokens[0], four.tokens[0]).prefix { $0 == $1 }.count)
    print(
        "round \(round + 1)  single \(format(plain, rows: 1))   batch(1) \(format(one, rows: 1))"
            + "   batch(4) \(format(four, rows: 4))   (\(batchedFirst ? "batched" : "single") first)"
    )
    #endif
    Memory.clearCache()
}
#if SINGLE_ONLY
print(
    String(
        format: "median  single %.2f tok/s (ttft %.0f ms)", median(single),
        1000 * median(singleFirst)))
#else
print(
    String(
        format:
            "median  single %.2f tok/s (ttft %.0f ms)   batch(1) %.2f tok/s (ttft %.0f ms)   "
            + "batch(4) %.2f tok/s per row, %.2f total (ttft %.0f ms)",
        median(single), 1000 * median(singleFirst), median(batchOfOne),
        1000 * median(batchOfOneFirst), median(batchOfFour), 4 * median(batchOfFour),
        1000 * median(batchOfFourFirst)))
print(
    "single and batch(1) agree on the first \(agreement.min() ?? 0) of \(decodeTokens + 1) tokens; "
        + "batch(1) and row 0 of batch(4) on the first \(rowAgreement.min() ?? 0)")
#endif
print(
    "each path repeats its tokens in every round: \(mismatchedRounds.isEmpty)"
        + (mismatchedRounds.isEmpty ? "" : ", not in rounds \(mismatchedRounds)"))
