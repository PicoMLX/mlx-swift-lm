// Copyright © 2026 Apple Inc.

// Decode throughput and time to first token on a local model, as medians over rounds: the
// single-sequence `TokenIterator`, and `BatchTokenIterator` at every batch size asked for.
// Decode is timed after the prefill and the first token; the time to first token covers the
// prefill. Every path must repeat its tokens in every round; how far each batch row agrees
// with a single-sequence run of the same prompt is reported.
//
//   swift run -c release --package-path Benchmarks/DecodeBench decodebench <model directory> \
//       [decode tokens: 128] [rounds: 5] [prompt tokens: 145] [sampler: greedy|topp|categorical] \
//       [batch sizes: 1,4,8] [prompt lengths: equal|mixed]
//
// With `mixed` prompt lengths, the rows of a batch get prompts that shrink evenly from the full
// length to a sixth of it, so every row but the first is left-padded. With the greedy sampler,
// a row whose positions or padding mask are wrong parts ways with its single-sequence run at
// once, while correct rows agree until the different matmul shapes of a batch round a near tie
// the other way.
//
// Built with `-Xswiftc -DSINGLE_ONLY` it times `TokenIterator` alone, so it also builds
// against a checkout without `BatchTokenIterator`.

import BenchmarkHelpers
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

let usage =
    "usage: decodebench <model directory> [decode tokens] [rounds] [prompt tokens] [sampler] "
    + "[batch sizes] [prompt lengths]"
let arguments = CommandLine.arguments
guard arguments.count > 1 else {
    print(usage)
    exit(1)
}
let modelDirectory = URL(fileURLWithPath: arguments[1])
let decodeTokens = arguments.count > 2 ? Int(arguments[2]) ?? 128 : 128
let rounds = max(1, arguments.count > 3 ? Int(arguments[3]) ?? 5 : 5)
let promptLength = arguments.count > 4 ? Int(arguments[4]) ?? 145 : 145
let samplerName = arguments.count > 5 ? arguments[5] : "greedy"
let batchSizes = Set(
    (arguments.count > 6 ? arguments[6] : "1,4,8").split(separator: ",").compactMap { Int($0) }
        .filter { $0 > 0 }
).sorted()
let lengthsName = arguments.count > 7 ? arguments[7] : "equal"
guard !batchSizes.isEmpty, ["equal", "mixed"].contains(lengthsName) else {
    print(usage)
    print("batch sizes are a comma-separated list such as 1,4,8; prompt lengths equal or mixed")
    exit(1)
}

let context = try await LLMModelFactory.shared.load(
    from: modelDirectory, using: NoOpTokenizerLoader())
let model = context.model
print("model: \(modelDirectory.lastPathComponent)")
#if SINGLE_ONLY
print(
    "prompt tokens: \(promptLength), decode tokens: \(decodeTokens), rounds: \(rounds), "
        + "sampler: \(samplerName)")
#else
print(
    "prompt tokens: \(promptLength), decode tokens: \(decodeTokens), rounds: \(rounds), "
        + "sampler: \(samplerName), batch sizes: \(batchSizes), prompt lengths: \(lengthsName)")
#endif

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
@MainActor func prompt(seed: UInt64, length: Int = promptLength) -> [Int32] {
    var state = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return (0 ..< length).map { _ in
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
    defer { Stream.defaultStream.synchronize() }
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
    defer { Stream.defaultStream.synchronize() }
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

/// The prompts of a batch: row 0 is the single-sequence prompt and every other row has its
/// own. With mixed lengths the rows shrink evenly to a sixth of the full length.
@MainActor func prompts(rows: Int) -> [[Int32]] {
    (0 ..< rows).map { row in
        guard lengthsName == "mixed", rows > 1 else { return prompt(seed: UInt64(row + 1)) }
        let shortest = max(1, promptLength / 6)
        let length = promptLength - (promptLength - shortest) * row / (rows - 1)
        return prompt(seed: UInt64(row + 1), length: length)
    }
}
#endif

let singlePrompt = prompt(seed: 1)

// Warm up every path once so kernel compilation is not timed.
_ = try runSingle(prompt: singlePrompt)
#if !SINGLE_ONLY
let batches = batchSizes.map { (rows: $0, prompts: prompts(rows: $0)) }
for batch in batches {
    _ = try runBatched(prompts: batch.prompts)
    if lengthsName == "mixed", batch.rows > 1 {
        print("batch(\(batch.rows)) prompt lengths: \(batch.prompts.map(\.count))")
    }
}

/// Every row's prompt run alone, untimed, to compare the batch rows with.
var singleTokens: [[Int32]: [Int]] = [:]
for batch in batches {
    for prompt in batch.prompts where singleTokens[prompt] == nil {
        singleTokens[prompt] = try runSingle(prompt: prompt).tokens[0]
    }
}
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
#if !SINGLE_ONLY
var batchRates: [Int: [Double]] = [:]
var batchFirst: [Int: [Double]] = [:]
/// Per batch size, the fewest leading tokens each row shares with its single-sequence run.
var rowAgreement: [Int: [Int]] = [:]
#endif
for round in 0 ..< rounds {
    #if SINGLE_ONLY
    let plain = try runSingle(prompt: singlePrompt)
    single.append(plain.tokensPerSecond)
    singleFirst.append(plain.firstToken)
    check("single", plain, round: round + 1)
    print("round \(round + 1)  single \(format(plain, rows: 1))")
    #else
    // Alternate the order so a thermal or background-load slope lands on every path.
    let batchedFirst = round % 2 == 1
    var runs: [Run] = []
    if batchedFirst { runs = try batches.map { try runBatched(prompts: $0.prompts) } }
    let plain = try runSingle(prompt: singlePrompt)
    if !batchedFirst { runs = try batches.map { try runBatched(prompts: $0.prompts) } }

    single.append(plain.tokensPerSecond)
    singleFirst.append(plain.firstToken)
    check("single", plain, round: round + 1)
    var line = "round \(round + 1)  single \(format(plain, rows: 1))"
    for (batch, run) in zip(batches, runs) {
        batchRates[batch.rows, default: []].append(run.tokensPerSecond)
        batchFirst[batch.rows, default: []].append(run.firstToken)
        // Each path must repeat itself across rounds exactly. Against a row's single run the
        // arithmetic can differ (a batch runs other matmul shapes, and a sampler draws a batch
        // from one random stream), so their streams can part ways; how far they agree is
        // reported, not asserted.
        check("batch(\(batch.rows))", run, round: round + 1)
        let agreement = zip(batch.prompts, run.tokens).map { prompt, tokens in
            zip(singleTokens[prompt]!, tokens).prefix { $0 == $1 }.count
        }
        rowAgreement[batch.rows] =
            rowAgreement[batch.rows].map { zip($0, agreement).map { min($0, $1) } } ?? agreement
        line += "   batch(\(batch.rows)) \(format(run, rows: batch.rows))"
    }
    print(line + "   (\(batchedFirst ? "batched" : "single") first)")
    #endif
    Memory.clearCache()
}
print(
    "median  single "
        + format(
            Run(tokensPerSecond: median(single), firstToken: median(singleFirst), tokens: []),
            rows: 1))
#if !SINGLE_ONLY
for batch in batches {
    let run = Run(
        tokensPerSecond: median(batchRates[batch.rows]!),
        firstToken: median(batchFirst[batch.rows]!), tokens: [])
    print("median  batch(\(batch.rows)) " + format(run, rows: batch.rows))
}
for batch in batches {
    let agreement = rowAgreement[batch.rows]!
    print(
        "batch(\(batch.rows)) rows agree with their single runs on the first "
            + "\(agreement.min()!) of \(decodeTokens + 1) tokens"
            + (batch.rows == 1 ? "" : "; per row \(agreement)"))
}
if samplerName != "greedy" {
    print("(a batch draws every row from one random stream, so rows can part ways at once)")
}
#endif
print(
    "each path repeats its tokens in every round: \(mismatchedRounds.isEmpty)"
        + (mismatchedRounds.isEmpty ? "" : ", not in rounds \(mismatchedRounds)"))
