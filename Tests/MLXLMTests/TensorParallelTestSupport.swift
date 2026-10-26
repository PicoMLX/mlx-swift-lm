// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

/// Checks every tensor-parallel model shares. One test process can only form a group of
/// one, where every shard is the whole weight, so these cover the sharded code path and its
/// checks; splitting weights across processes needs a multi-process run.
///
/// `makeModel(quantized)` builds a seeded model with evaluated random weights.
func assertShardingKeepsTheModel<Model: LanguageModel & TensorParallelModel>(
    _ makeModel: (Bool) throws -> Model, file: StaticString = #filePath, line: UInt = #line
) throws {
    for quantized in [false, true] {
        let model = try makeModel(quantized)
        let whole = try shardingLogits(model)
        try model.shard(group: MLXDistributed.initialize())
        assertClose(
            try shardingLogits(model), whole, tolerance: 1e-4,
            "\(quantized ? "4-bit" : "float") sharded path", file: file, line: line)
    }

    // Sharding again would split the shards a second time.
    let sharded = try makeModel(false)
    let group = try MLXDistributed.initialize()
    try sharded.shard(group: group)
    let shardedLogits = try shardingLogits(sharded)
    XCTAssertThrowsError(try sharded.shard(group: group), file: file, line: line) { error in
        XCTAssertEqual(error as? TensorParallelError, .alreadySharded, file: file, line: line)
    }
    assertClose(
        try shardingLogits(sharded), shardedLogits, tolerance: 0, "after a refused call",
        file: file, line: line)
}

/// Last-position logits after a prompt and after each of three decode steps.
func shardingLogits(_ model: any LanguageModel) throws -> [MLXArray] {
    let cache = try model.newCache(parameters: nil)
    return [[3, 17, 42, 8, 29, 51], [7], [23], [60]].map { (tokens: [Int32]) in
        let out = model(MLXArray(tokens).reshaped(1, -1), cache: cache)[0, -1]
        eval(out)
        return out
    }
}

func assertClose(
    _ actual: [MLXArray], _ expected: [MLXArray], tolerance: Float, _ label: String,
    file: StaticString = #filePath, line: UInt = #line
) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    for (step, (a, e)) in zip(actual, expected).enumerated() {
        XCTAssertLessThanOrEqual(
            abs(a - e).max().item(Float.self), tolerance, "\(label), step \(step)",
            file: file, line: line)
    }
}

/// A model of `type` decoded from `json`, with seeded random weights, quantized if asked.
func makeShardingModel<Model: Module, Configuration: Decodable>(
    _ json: String, configuration: Configuration.Type, quantized: Bool,
    _ makeModel: (Configuration) -> Model
) throws -> Model {
    MLXRandom.seed(11)
    let model = makeModel(try JSONDecoder().decode(configuration, from: Data(json.utf8)))
    if quantized {
        quantize(model: model, groupSize: 64, bits: 4)
    }
    eval(model)
    return model
}
