// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

let fixtureURL = URL(fileURLWithPath: CommandLine.arguments[1])
let fixture = try loadArrays(url: fixtureURL)
let configuration = try JSONDecoder().decode(
    Qwen3OmniConfiguration.self,
    from: Data(contentsOf: fixtureURL.deletingPathExtension().appendingPathExtension("json")))
let group = try MLXDistributed.initialize(backend: .ring, strict: true)

@MainActor
func model(quantized: Bool, sharded: Bool) throws -> Qwen3Omni {
    let model = Qwen3Omni(configuration)
    var weights: [String: MLXArray] = [:]
    for (name, value) in fixture {
        for (prefix, destination) in [
            ("audio.weights.", "audio_tower."),
            ("vision.weights.", "vision_tower."), ("language.weights.", "language_model."),
        ] {
            if name.hasPrefix(prefix) {
                weights[destination + name.dropFirst(prefix.count)] = value
            }
        }
    }
    try model.update(
        parameters: ModuleParameters.unflattened(model.sanitize(weights: weights)), verify: [.all])
    if quantized {
        quantize(
            model: model, groupSize: 32, bits: 4,
            filter: { path, _ in path.hasPrefix("language_model.") })
    }
    eval(model)
    if sharded { try model.shard(group: group) }
    return model
}

@MainActor
func decode(_ model: Qwen3Omni, batched: Bool, media: Bool) throws -> [MLXArray] {
    let first = media ? fixture["multimodal.tokens"]!.asArray(Int32.self) : [3, 4, 5, 6, 7]
    let length = first.count
    let second: [Int32] = [8, 9, 10]
    let tokens =
        batched
        ? MLXArray(first + second + Array(repeating: 0, count: length - second.count)).reshaped(
            2, length)
        : MLXArray(first).reshaped(1, length)
    let caches: [KVCache]
    let batch: [any BatchedCache]
    if batched {
        batch = try makeBatchedCacheFactories(for: model.newCache(parameters: nil)).map {
            $0([0, 0])
        }
        for cache in batch { cache.prepare(lengths: [length, second.count]) }
        caches = batch.map { $0 as KVCache }
    } else {
        batch = []
        caches = try model.newCache(parameters: nil)
    }
    let output: MLXArray
    if media {
        let mask =
            batched
            ? MLXArray(
                Array(repeating: 1, count: length) + Array(repeating: 1, count: second.count)
                    + Array(repeating: 0, count: length - second.count)
            ).reshaped(2, length) : nil
        let input = LMInput(
            text: .init(tokens: tokens, mask: mask),
            image: .init(pixels: fixture["vision.input"]![..<16], frames: [THW(1, 4, 4)]),
            video: .init(pixels: fixture["vision.input"]![16...], frames: [THW(2, 4, 4)]),
            audio: .init(
                features: fixture["audio.input"]![0..., ..<101].expandedDimensions(axis: 0)))
        guard
            case .logits(let result) = try model.prepare(
                input, cache: caches, state: nil,
                prefill: .init(chunking: .unchunked))
        else { fatalError("Expected media logits") }
        output = result.logits
    } else {
        output = model(.init(tokens: tokens), cache: caches, state: nil).logits
    }
    for cache in batch { cache.finalize() }
    var logits =
        batched ? stacked([output[0, length - 1], output[1, second.count - 1]]) : output[0..., -1]
    var result = [logits.asType(.float32)]
    for _ in 0 ..< 5 {
        let next = argMax(logits, axis: -1).reshaped(batched ? 2 : 1, 1)
        logits = model(.init(tokens: next), cache: caches, state: nil).logits[0..., 0]
        eval(logits)
        result.append(logits.asType(.float32))
    }
    return result
}

var report: [String: Any] = try ["rank": group.rank, "size": group.size]
for quantized in [false, true] {
    for batched in [false, true] {
        for media in [false, true] {
            let expected = try decode(
                model(quantized: quantized, sharded: false), batched: batched, media: media)
            let sharded = try model(quantized: quantized, sharded: true)
            let actual = try decode(sharded, batched: batched, media: media)
            let parameters = Dictionary(uniqueKeysWithValues: sharded.parameters().flattened())
            let scale = max(expected.map { abs($0).max().item(Float.self) }.max()!, 1e-6)
            let difference = zip(expected, actual).map { abs($0 - $1).max().item(Float.self) }
                .max()!
            let key =
                "\(quantized ? "4bit" : "float"),\(batched ? "batch" : "solo"),\(media ? "media" : "text")"
            report[key] = [
                "relativeDifference": difference / scale,
                "tokens": actual.flatMap { argMax($0, axis: -1).asArray(Int.self) },
                "wholeTokens": expected.flatMap { argMax($0, axis: -1).asArray(Int.self) },
                "qRows": parameters["language_model.model.layers.0.self_attn.q_proj.weight"]!.dim(
                    0),
                "kvHeads": sharded.kvHeads,
                "keyRows": parameters["language_model.model.layers.0.self_attn.k_proj.weight"]!.dim(
                    0),
                "expertRows": parameters[
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]!.dim(1),
            ]
        }
    }
}
print(
    String(
        decoding: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]),
        as: UTF8.self))
