// Copyright © 2026 Apple Inc.
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

struct SmokeFixture: Decodable, Sendable {
    struct Template: Decodable, Sendable {
        let tokens: [Int]
        let text: String
    }
    let image: String
    let prompts: [String]
    let encodings: [String: [Int]]
    let templates: [String: Template]
    let specialTokens: [String: Int]
}
/// Native Hugging Face tokenizer outputs for the finite prompts used by this smoke test.
/// The Python wrapper owns tokenizer execution and generation decoding.
struct NativePromptTokenizer: Tokenizer, TokenizerLoader {
    let fixture: SmokeFixture
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        if let value = fixture.specialTokens[text] { return [value] }
        guard let value = fixture.encodings[text], !addSpecialTokens else {
            preconditionFailure("Missing native tokenization: \(text.debugDescription)")
        }
        return value
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        guard let template = fixture.templates.values.first(where: { $0.tokens == tokenIds }) else {
            preconditionFailure("Decode generated tokens with the native Python tokenizer")
        }
        return template.text
    }
    func convertTokenToId(_ token: String) -> Int? { fixture.specialTokens[token] }
    func convertIdToToken(_ id: Int) -> String? {
        fixture.specialTokens.first { $0.value == id }?.key
    }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        guard messages.count == 1, let content = messages[0]["content"] as? String,
            let value = fixture.templates[content]
        else { throw TokenizerError.missingChatTemplate }
        return value.tokens
    }
    func load(from directory: URL) async throws -> any Tokenizer { self }
}
func lastLogits(_ result: PrepareResult) throws -> MLXArray {
    guard case .logits(let output) = result else {
        throw VLMError.processing("MiniCPM did not return prepared logits")
    }
    let value = output.logits[0, -1].asType(.float32)
    guard abs(value).max().item(Float.self).isFinite else {
        throw VLMError.processing("Non-finite native checkpoint logits")
    }
    return value
}
let arguments = CommandLine.arguments
precondition(arguments.count == 3, "Usage: MiniCPMCheckpointSmoke checkpoint-directory input.json")
let directory = URL(fileURLWithPath: arguments[1], isDirectory: true)
let fixture = try JSONDecoder().decode(
    SmokeFixture.self, from: Data(contentsOf: URL(fileURLWithPath: arguments[2])))
let context = try await VLMModelFactory.shared.load(
    from: directory, using: NativePromptTokenizer(fixture: fixture))
let float32Compute = ProcessInfo.processInfo.environment["MLX_SMOKE_FLOAT32"] == "1"
if float32Compute {
    try context.model.update(
        parameters: ModuleParameters.unflattened(
            context.model.parameters().flattened().map { name, value in
                (name, value.dtype.isFloatingPoint ? value.asType(.float32) : value)
            }), verify: [.all])
    try context.model.prepare()
    eval(context.model)
}
let media = try await context.processor.prepare(
    input: .init(
        prompt: fixture.prompts[0], images: [.url(URL(fileURLWithPath: fixture.image))],
        additionalContext: ["enable_thinking": false]))
let text = try await context.processor.prepare(
    input: .init(prompt: fixture.prompts[1], additionalContext: ["enable_thinking": false]))
guard let image = media.image, image.pixels.ndim == 3, image.frames?.count == 1 else {
    throw VLMError.processing("Native MiniCPM slicing did not produce a thumbnail grid")
}
let model = context.model
let singles = try [model.newCache(parameters: nil), model.newCache(parameters: nil)]
let first = try lastLogits(model.prepare(media, cache: singles[0], state: nil, prefill: .init()))
let second = try lastLogits(model.prepare(text, cache: singles[1], state: nil, prefill: .init()))
var next = [argMax(first).item(Int.self), argMax(second).item(Int.self)]
var generated = next.map { [$0] }
guard let batch = makeBatchedCaches(from: singles[0]),
    let added = makeBatchedCaches(from: singles[1])
else {
    throw VLMError.processing("Native MiniCPM caches could not be adopted through the shared API")
}
for index in batch.indices { batch[index].extend(other: added[index]) }
var maximumDifference: Float = 0
var stepDifferences: [[Float]] = []
var firstLayerStateDifferences: [[[Float]]] = []
var logitScales: [[Float]] = []
for _ in 0 ..< 7 {
    let batched = model(MLXArray(next).reshaped(2, 1), cache: batch.map { $0 as any KVCache })
    var nextTokens: [Int] = []
    var differences: [Float] = []
    var stateDifferences: [[Float]] = []
    var scales: [Float] = []
    for row in 0 ..< 2 {
        let independent = model(MLXArray([next[row]]).reshaped(1, 1), cache: singles[row])
        let difference = abs(batched[row] - independent[0]).max().item(Float.self)
        maximumDifference = max(maximumDifference, difference)
        differences.append(difference)
        scales.append(abs(independent[0]).max().item(Float.self))
        if let b = batch[0] as? MambaCache, let s = singles[row][0] as? MambaCache {
            stateDifferences.append(
                (0 ..< 2).map { abs(b[$0]![row] - s[$0]![0]).max().item(Float.self) })
        }
        let actual = argMax(batched[row, -1]).item(Int.self)
        guard actual == argMax(independent[0, -1]).item(Int.self) else {
            throw VLMError.processing(
                "Native MiniCPM batch tokens differ from independent decoding")
        }
        nextTokens.append(actual)
        generated[row].append(actual)
    }
    next = nextTokens
    stepDifferences.append(differences)
    firstLayerStateDifferences.append(stateDifferences)
    logitScales.append(scales)
}
let report: [String: Any] = [
    "model": directory.lastPathComponent, "float32Compute": float32Compute,
    "sourceShape": [64, 64], "pixelShape": image.pixels.shape,
    "cropGrids": image.frames!.map { [$0.t, $0.h, $0.w] },
    "prefillLengths": [media.text.tokens.size, text.text.tokens.size],
    "generatedTokens": generated, "batchMaxAbsDifference": maximumDifference,
    "batchRelativeDifference": maximumDifference / max(logitScales.flatMap { $0 }.max() ?? 1, 1e-6),
    "stepDifferences": stepDifferences, "firstLayerStateDifferences": firstLayerStateDifferences,
    "logitScales": logitScales,
]
print(
    String(
        decoding: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]),
        as: UTF8.self))
