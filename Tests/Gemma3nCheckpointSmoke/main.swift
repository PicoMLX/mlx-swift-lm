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
        guard messages.count == 1 else { throw TokenizerError.missingChatTemplate }
        let key: String
        if let text = messages[0]["content"] as? String {
            key = text.contains("<|image_") ? "image" : text.contains("<|audio_") ? "audio" : "text"
        } else if let parts = messages[0]["content"] as? [[String: any Sendable]] {
            key =
                parts.contains { $0["type"] as? String == "image" }
                ? "image"
                : parts.contains { $0["type"] as? String == "audio" } ? "audio" : "text"
        } else {
            throw TokenizerError.missingChatTemplate
        }
        guard let value = fixture.templates[key] else { throw TokenizerError.missingChatTemplate }
        return value.tokens
    }
    func load(from directory: URL) async throws -> any Tokenizer { self }
}

func prepare(_ input: LMInput, model: any LanguageModel, cache: [any KVCache]) throws -> MLXArray {
    let result = try model.prepare(input, cache: cache, state: nil, prefill: .init())
    let logits: MLXArray
    switch result {
    case .logits(let output): logits = output.logits
    case .tokens(let text): logits = model(text.tokens, cache: cache)
    }
    let last = logits[0, -1].asType(.float32)
    guard all(isFinite(last)).item(Bool.self) else {
        throw VLMError.processing("Non-finite native checkpoint logits")
    }
    return last
}
func adopted(_ source: [any KVCache]) throws -> [any BatchedCache] {
    guard let converted = makeBatchedCaches(from: source) else {
        throw VLMError.processing("Native checkpoint cache cannot enter continuous batching")
    }
    return converted
}
let arguments = CommandLine.arguments
precondition(arguments.count == 3, "Usage: checkpoint-smoke checkpoint-directory input.json")
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
    eval(context.model)
}
let image = try await context.processor.prepare(
    input: .init(prompt: fixture.prompts[0], images: [.url(URL(fileURLWithPath: fixture.image))]))
let audio = try await context.processor.prepare(
    input: .init(
        prompt: fixture.prompts[1],
        audios: [
            .url(
                URL(fileURLWithPath: fixture.image).deletingLastPathComponent()
                    .appendingPathComponent("audio.wav"))
        ]))
let text = try await context.processor.prepare(input: .init(prompt: fixture.prompts[2]))
let inputs = [image, audio, text]
let model = context.model
let singles = try inputs.map { _ in try model.newCache(parameters: nil) }
var next = try inputs.enumerated().map { row, input in
    argMax(try prepare(input, model: model, cache: singles[row])).item(Int.self)
}
var generated = next.map { [$0] }
let batch = try adopted(singles[0])
for row in 1 ..< singles.count {
    let added = try adopted(singles[row])
    for layer in batch.indices { batch[layer].extend(other: added[layer]) }
}
var difference: Float = 0
var scale: Float = 0
for _ in 0 ..< 7 {
    let output = model(MLXArray(next).reshaped(3, 1), cache: batch.map { $0 as any KVCache })
    var tokens: [Int] = []
    for row in singles.indices {
        let independent = model(MLXArray([next[row]]).reshaped(1, 1), cache: singles[row])
        difference = max(difference, abs(output[row] - independent[0]).max().item(Float.self))
        scale = max(scale, abs(independent[0]).max().item(Float.self))
        let token = argMax(output[row, -1]).item(Int.self)
        guard token == argMax(independent[0, -1]).item(Int.self) else {
            throw VLMError.processing(
                "Native checkpoint batch token differs from independent decode")
        }
        tokens.append(token)
        generated[row].append(token)
    }
    next = tokens
}
let report: [String: Any] = [
    "model": directory.lastPathComponent,
    "float32Compute": float32Compute,
    "pixelShape": image.image!.pixels.shape, "audioShape": audio.audio!.features.shape,
    "prefillLengths": inputs.map { $0.text.tokens.size }, "generatedTokens": generated,
    "batchMaxAbsDifference": difference, "batchRelativeDifference": difference / max(scale, 1e-6),
]
print(
    String(
        decoding: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]),
        as: UTF8.self))
