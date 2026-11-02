import Foundation
import MLX
import MLXLMCommon
import MLXVLM

struct SmokeFixture: Decodable, Sendable {
    let image: String
    let prompts: [String]
    let encodings: [String: [Int]]
    let imageTokenId: Int
    let specialTokens: [String: Int]
}

/// Replays IDs produced by the checkpoint's native tokenizer for these exact prompts.
/// The Python wrapper decodes generated IDs with that same tokenizer.
struct NativePromptTokenizer: Tokenizer, TokenizerLoader {
    let fixture: SmokeFixture
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        if let token = fixture.specialTokens[text] { return [token] }
        guard let tokens = fixture.encodings[text], !addSpecialTokens else {
            preconditionFailure(
                "Missing native prompt IDs for \(text.debugDescription), special=\(addSpecialTokens)."
            )
        }
        return tokens
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        preconditionFailure("Decode generated IDs with the native Python tokenizer.")
    }
    func convertTokenToId(_ token: String) -> Int? {
        fixture.specialTokens[token]
    }
    func convertIdToToken(_ id: Int) -> String? {
        id == fixture.imageTokenId ? "<image>" : nil
    }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
    func load(from directory: URL) async throws -> any Tokenizer { self }
}

func lastLogits(_ result: PrepareResult) throws -> MLXArray {
    guard case .logits(let output) = result else {
        throw VLMError.processing("The OCR checkpoint did not produce prefill logits.")
    }
    let logits = output.logits[0, -1].asType(.float32)
    guard abs(logits).max().item(Float.self).isFinite else {
        throw VLMError.processing("Non-finite checkpoint logits.")
    }
    return logits
}

let arguments = CommandLine.arguments
guard arguments.count == 3 else {
    fatalError("Usage: DeepSeekOCRCheckpointSmoke checkpoint-directory input.json")
}
let directory = URL(fileURLWithPath: arguments[1], isDirectory: true)
let fixture = try JSONDecoder().decode(
    SmokeFixture.self, from: Data(contentsOf: URL(fileURLWithPath: arguments[2])))
let context = try await VLMModelFactory.shared.load(
    from: directory, using: NativePromptTokenizer(fixture: fixture))
context.model.update(
    parameters: context.model.parameters().mapValues { value in
        value.dtype == .float16 || value.dtype == .bfloat16 ? value.asType(.float32) : value
    })
eval(context.model)
let media = try await context.processor.prepare(
    input: .init(prompt: fixture.prompts[0], images: [.url(URL(fileURLWithPath: fixture.image))]))
guard let pixels = media.image?.pixels, pixels.shape == [1, 3, 1024, 1024] else {
    throw VLMError.processing("The 64px source must pass through native 1024px preprocessing.")
}
let text = try await context.processor.prepare(input: .init(prompt: fixture.prompts[1]))
let model = context.model
let singles = try [model.newCache(parameters: nil), model.newCache(parameters: nil)]
let first = try lastLogits(model.prepare(media, cache: singles[0], state: nil, prefill: .init()))
let second = try lastLogits(model.prepare(text, cache: singles[1], state: nil, prefill: .init()))
var next = [argMax(first).item(Int.self), argMax(second).item(Int.self)]
var generated = next.map { [$0] }
let batch = try singles[0].enumerated().map { index, cache -> BatchKVCache in
    guard let first = cache as? KVCacheSimple, let second = singles[1][index] as? KVCacheSimple
    else {
        throw VLMError.processing("Unexpected OCR attention cache type.")
    }
    let combined = BatchKVCache.fromSingle(first.copy() as! KVCacheSimple)
    combined.extend(other: BatchKVCache.fromSingle(second.copy() as! KVCacheSimple))
    return combined
}
var maximumDifference: Float = 0
for _ in 0 ..< 7 {
    let batched = model(MLXArray(next).reshaped(2, 1), cache: batch.map { $0 as any KVCache })
    var nextTokens = [Int]()
    for row in 0 ..< 2 {
        let independent = model(MLXArray([next[row]]).reshaped(1, 1), cache: singles[row])
        maximumDifference = max(
            maximumDifference,
            abs(batched[row] - independent[0]).max().item(Float.self))
        let actual = argMax(batched[row, -1]).item(Int.self)
        guard actual == argMax(independent[0, -1]).item(Int.self) else {
            throw VLMError.processing(
                "Native checkpoint batch tokens differ from independent decoding.")
        }
        nextTokens.append(actual)
        generated[row].append(actual)
    }
    next = nextTokens
}
let report: [String: Any] = [
    "model": directory.lastPathComponent,
    "sourceShape": [64, 64], "pixelShape": pixels.shape,
    "prefillLengths": [media.text.tokens.size, text.text.tokens.size],
    "generatedTokens": generated, "batchMaxAbsDifference": maximumDifference,
    "floatingParameterType": "float32",
]
guard maximumDifference < 2e-4 else {
    throw VLMError.processing("Checkpoint batch logits differ by \(maximumDifference).")
}
let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
print(String(decoding: data, as: UTF8.self))
