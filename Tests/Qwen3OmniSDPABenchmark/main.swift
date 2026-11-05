// Bounded standalone probe of Qwen3's D72 SDPA and inherited outer-padding path.
// Build Release; run without other GPU workloads. No pretrained weights needed.
import Foundation
import MLX
import MLXNN
import Metal

let length = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1])! : 1024
precondition(length >= 256 && length <= 4096)
let heads = 16
let dimensions = 72
let iterations = 20
let warmups = 3
MLXRandom.seed(314159)
var reports: [[String: Any]] = []
for dtype in [DType.float16, .bfloat16] {
    let q = (MLXRandom.normal([1, heads, length, dimensions]) * 0.1).asType(dtype)
    let k = (MLXRandom.normal([1, heads, length, dimensions]) * 0.1).asType(dtype)
    let v = MLXRandom.normal([1, heads, length, dimensions]).asType(dtype)
    eval(q, k, v)
    var reference: MLXArray?
    for width in [72, 80, 96] {
        func pad(_ x: MLXArray) -> MLXArray {
            width == dimensions
                ? x
                : concatenated(
                    [x, MLXArray.zeros([1, heads, length, width - dimensions], dtype: dtype)],
                    axis: -1)
        }
        let queries = pad(q)
        let keys = pad(k)
        let values = pad(v)
        eval(queries, keys, values)
        func attention() -> MLXArray {
            MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: pow(Float(dimensions), -0.5), mask: .none)[0..., 0..., 0..., ..<dimensions]
        }
        for _ in 0 ..< warmups { eval(attention()) }
        var milliseconds: [Double] = []
        var output = attention()
        for _ in 0 ..< iterations {
            let start = DispatchTime.now().uptimeNanoseconds
            output = attention()
            eval(output)
            milliseconds.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
        }
        let maximumDifference =
            reference.map {
                abs(output.asType(.float32) - $0.asType(.float32)).max().item(Float.self)
            } ?? 0
        if reference == nil { reference = output }
        precondition(maximumDifference < 0.01, "Padding changed SDPA results")
        let sorted = milliseconds.sorted()
        reports.append([
            "dtype": String(describing: dtype), "headWidth": width,
            "path": width == 72
                ? "direct, core internal padding"
                : (width == 80 ? "inherited outer padding" : "explicit outer padding"),
            "medianMilliseconds": sorted[sorted.count / 2], "minimumMilliseconds": sorted[0],
            "maximumAbsDifference": maximumDifference,
        ])
    }
}
let report: [String: Any] = [
    "gpu": MTLCreateSystemDefaultDevice()?.name ?? "unknown",
    "sequenceLength": length, "heads": heads, "originalHeadWidth": dimensions,
    "iterations": iterations, "warmups": warmups, "results": reports,
    "scope":
        "Release standalone SDPA only; excludes projection, encoders, model loading and generation",
]
print(
    String(
        decoding: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]),
        as: UTF8.self))
