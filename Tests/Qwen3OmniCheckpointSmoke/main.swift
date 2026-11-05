import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

struct SmokeFixture: Decodable, Sendable {
    struct Input: Decodable, Sendable {
        let label: String, prompt: String, signature: String
        let tokens: [Int]
    }
    let image: String, audio: String, video: String
    let cases: [Input]
    let specialTokens: [String: Int]
}

/// Native Python tokenizer/template output is replayed only after verifying the
/// Swift processor supplied those exact messages. No approximate tokenizer is used.
struct NativeTokenizer: Tokenizer, TokenizerLoader {
    let fixture: SmokeFixture
    var bosToken: String? { nil }
    var eosToken: String? { "<|im_end|>" }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        guard let token = fixture.specialTokens[text], !addSpecialTokens else {
            preconditionFailure("Unexpected non-native tokenizer request")
        }
        return [token]
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        preconditionFailure("Decode with the checkpoint's native Python tokenizer")
    }
    func convertTokenToId(_ token: String) -> Int? { fixture.specialTokens[token] }
    func convertIdToToken(_ id: Int) -> String? {
        fixture.specialTokens.first { $0.value == id }?.key
    }
    func load(from directory: URL) async throws -> any Tokenizer { self }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let signature = String(
            decoding: try JSONSerialization.data(withJSONObject: messages, options: [.sortedKeys]),
            as: UTF8.self)
        guard tools == nil, additionalContext?["enable_thinking"] as? Bool == true,
            let item = fixture.cases.first(where: { $0.signature == signature })
        else {
            throw VLMError.processing(
                "Swift messages differ from native checkpoint chat-template inputs: \(signature)")
        }
        return item.tokens
    }
}

@MainActor
func makeVideo(image: URL, audio: URL, destination: URL) async throws {
    let source = AVURLAsset(url: audio)
    guard let track = try await source.loadTracks(withMediaType: .audio).first else {
        throw VLMError.processing("Missing fixture audio")
    }
    let reader = try AVAssetReader(asset: source)
    let audioOutput = AVAssetReaderTrackOutput(
        track: track,
        outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16000, AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false,
        ])
    reader.add(audioOutput)
    let writer = try AVAssetWriter(outputURL: destination, fileType: .mov)
    let videoInput = AVAssetWriterInput(
        mediaType: .video,
        outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 64, AVVideoHeightKey: 64,
        ])
    let audioInput = AVAssetWriterInput(
        mediaType: .audio,
        outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 16000, AVNumberOfChannelsKey: 1, AVEncoderBitRateKey: 32000,
        ])
    writer.add(videoInput)
    writer.add(audioInput)
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: videoInput,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
            kCVPixelBufferWidthKey as String: 64, kCVPixelBufferHeightKey as String: 64,
        ])
    guard writer.startWriting(), reader.startReading() else {
        throw VLMError.processing("Unable to start native media fixture")
    }
    writer.startSession(atSourceTime: .zero)
    let picture = try UserInput.Image.url(image).asCIImage()
    let context = CIContext()
    for index in 0 ..< 4 {
        var waits = 0
        while !videoInput.isReadyForMoreMediaData && waits < 500 {
            try await Task.sleep(for: .milliseconds(5))
            waits += 1
        }
        guard videoInput.isReadyForMoreMediaData, let pool = adaptor.pixelBufferPool else {
            throw VLMError.processing("Video writer did not become ready")
        }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        guard let buffer else { throw VLMError.processing("Video fixture allocation failed") }
        context.render(picture, to: buffer)
        guard
            adaptor.append(buffer, withPresentationTime: CMTime(value: Int64(index), timescale: 2))
        else { throw VLMError.processing("Video append failed") }
    }
    videoInput.markAsFinished()
    while let buffer = audioOutput.copyNextSampleBuffer() {
        var waits = 0
        while !audioInput.isReadyForMoreMediaData && waits < 500 {
            try await Task.sleep(for: .milliseconds(5))
            waits += 1
        }
        guard audioInput.isReadyForMoreMediaData, audioInput.append(buffer) else {
            throw VLMError.processing("Audio append failed")
        }
    }
    audioInput.markAsFinished()
    await writer.finishWriting()
    guard writer.status == .completed, reader.status == .completed else {
        throw VLMError.processing("Native video/audio fixture failed")
    }
}

func lastLogits(_ result: PrepareResult, model: any LanguageModel, cache: [KVCache]) throws
    -> MLXArray
{
    let output: MLXArray
    switch result {
    case .logits(let value): output = value.logits
    case .tokens(let text): output = model(text, cache: cache, state: nil).logits
    }
    let last = output[0, -1].asType(.float32)
    guard abs(last).max().item(Float.self).isFinite else {
        throw VLMError.processing("Non-finite native checkpoint logits")
    }
    return last
}

func adopt(_ source: [KVCache]) throws -> [any BatchedCache] {
    guard let batch = makeBatchedCaches(from: source) else {
        throw VLMError.processing(
            "Native Qwen3 Omni caches are not accepted by the production adoption gate")
    }
    return batch
}

let arguments = CommandLine.arguments
precondition(
    arguments.count == 3, "Usage: Qwen3OmniCheckpointSmoke checkpoint-directory input.json")
let directory = URL(fileURLWithPath: arguments[1], isDirectory: true)
let fixture = try JSONDecoder().decode(
    SmokeFixture.self, from: Data(contentsOf: URL(fileURLWithPath: arguments[2])))
try await makeVideo(
    image: URL(fileURLWithPath: fixture.image), audio: URL(fileURLWithPath: fixture.audio),
    destination: URL(fileURLWithPath: fixture.video))
let context = try await VLMModelFactory.shared.load(
    from: directory, using: NativeTokenizer(fixture: fixture))
let model = context.model
let float32Compute = ProcessInfo.processInfo.environment["MLX_SMOKE_FLOAT32"] == "1"
if float32Compute {
    try model.update(
        parameters: ModuleParameters.unflattened(
            model.parameters().flattened().map { name, array in
                (name, array.dtype.isFloatingPoint ? array.asType(.float32) : array)
            }), verify: [.all])
    eval(model)
}
var singles: [[KVCache]] = []
var initial: [MLXArray] = []
var mediaShapes: [[String: Any]] = []
for item in fixture.cases {
    var input = UserInput(
        prompt: item.prompt,
        images: item.label == "image" ? [.url(URL(fileURLWithPath: fixture.image))] : [],
        videos: item.label == "video_audio" ? [.url(URL(fileURLWithPath: fixture.video))] : [],
        audios: item.label == "audio" ? [.url(URL(fileURLWithPath: fixture.audio))] : [],
        additionalContext: [
            "enable_thinking": true, "use_audio_in_video": item.label == "video_audio",
        ])
    input.processing.maxPixels = 64 * 64
    let prepared = try await context.processor.prepare(input: input)
    let cache = try model.newCache(parameters: nil)
    initial.append(
        try lastLogits(
            model.prepare(prepared, cache: cache, state: nil, prefill: .init()), model: model,
            cache: cache))
    singles.append(cache)
    mediaShapes.append([
        "label": item.label, "tokens": prepared.text.tokens.size,
        "image": prepared.image?.pixels.shape ?? [], "video": prepared.video?.pixels.shape ?? [],
        "audio": prepared.audio?.features.shape ?? [],
    ])
}
var batch = try adopt(singles[0])
for source in singles.dropFirst() {
    let added = try adopt(source)
    for index in batch.indices { batch[index].extend(other: added[index]) }
}
var next = initial.map { argMax($0).item(Int.self) }
var generated = next.map { [$0] }
var maximumDifference: Float = 0
var maximumScale: Float = 1
for _ in 0 ..< 7 {
    let actual = model(
        LMInput.Text(tokens: MLXArray(next).reshaped(next.count, 1)),
        cache: batch.map { $0 as KVCache }, state: nil
    ).logits
    var following: [Int] = []
    for row in next.indices {
        let expected = model(
            LMInput.Text(tokens: MLXArray([next[row]]).reshaped(1, 1)), cache: singles[row],
            state: nil
        ).logits
        maximumDifference = max(
            maximumDifference, abs(actual[row] - expected[0]).max().item(Float.self))
        maximumScale = max(maximumScale, abs(expected).max().item(Float.self))
        let token = argMax(actual[row, -1]).item(Int.self)
        guard token == argMax(expected[0, -1]).item(Int.self) else {
            throw VLMError.processing(
                "Native checkpoint batch tokens differ from independent decoding")
        }
        following.append(token)
        generated[row].append(token)
    }
    next = following
}
let report: [String: Any] = [
    "labels": fixture.cases.map(\.label), "mediaShapes": mediaShapes,
    "generatedTokens": generated, "batchMaxAbsDifference": maximumDifference,
    "batchRelativeDifference": maximumDifference / maximumScale, "float32Compute": float32Compute,
    "tf32Disabled": ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0",
]
print(
    String(
        decoding: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]),
        as: UTF8.self))
