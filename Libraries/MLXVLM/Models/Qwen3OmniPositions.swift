// Copyright © 2026 Apple Inc.
// Omni time-aware MRoPE, following Hugging Face Transformers' Apache-2.0 implementation.

import Foundation
import MLX
import MLXLMCommon

enum Qwen3OmniPositions {
    struct Parameters: Sendable {
        var imageToken = 151_655, videoToken = 151_656, audioToken = 151_675
        var visionStartToken = 151_652, audioStartToken = 151_669
        var visionEndToken = 151_653, audioEndToken = 151_670
        var positionsPerSecond: Double = 13
        var merge = 2, audioWindow = 50
        init() {}
        init(_ config: Qwen3OmniConfiguration.ThinkerConfiguration) {
            imageToken = config.imageToken
            videoToken = config.videoToken
            audioToken = config.audioToken
            visionStartToken = config.visionStartToken
            audioStartToken = config.audioStartToken
            visionEndToken = config.visionEndToken
            audioEndToken = config.audioEndToken
            positionsPerSecond = config.positionsPerSecond
            merge = config.vision.spatialMergeSize
            audioWindow = config.audio.window
        }
    }
    /// Positions remain floating point: temporal video coordinates need not be integer.
    static func make(
        tokens: MLXArray, mask: MLXArray?, images: [THW], videos: [THW],
        secondsPerVideoGrid: [Double], audioLengths: [Int],
        config: Parameters, videoTimes: [[Double]]? = nil
    ) throws -> (positions: MLXArray, deltas: MLXArray) {
        let batch = tokens.dim(0)
        let length = tokens.dim(1)
        let ids = tokens.asArray(Int.self)
        let valid =
            mask?.asType(.bool).asArray(Bool.self) ?? Array(repeating: true, count: ids.count)
        guard valid.count == ids.count else {
            throw VLMError.processing("Omni text mask shape does not match tokens")
        }
        var output = Array(repeating: Float(0), count: 3 * ids.count)
        var deltas = Array(repeating: Float(0), count: batch)
        var imageIndex = 0
        var videoIndex = 0
        var audioIndex = 0
        let merge = config.merge
        func gridPositions(_ grid: THW, seconds: Double, timestamps: [Double]?) throws -> [[Float]]
        {
            guard grid.t > 0, grid.h > 0, grid.w > 0, grid.h % merge == 0, grid.w % merge == 0,
                seconds.isFinite, seconds > 0
            else {
                throw VLMError.processing("Invalid Qwen3 Omni visual grid")
            }
            guard
                timestamps == nil
                    || (timestamps?.count == grid.t
                        && timestamps!.allSatisfy({ $0.isFinite && $0 >= 0 }))
            else { throw VLMError.processing("Omni video timestamps do not match temporal grid") }
            var result: [[Float]] = []
            for t in 0 ..< grid.t {
                for h in 0 ..< grid.h / merge {
                    for w in 0 ..< grid.w / merge {
                        result.append([
                            Float(
                                (timestamps?[t] ?? Double(t) * seconds) * config.positionsPerSecond),
                            Float(h), Float(w),
                        ])
                    }
                }
            }
            return result
        }
        for row in 0 ..< batch {
            let columns = (0 ..< length).filter { valid[row * length + $0] }
            guard !columns.isEmpty else {
                throw VLMError.processing(
                    "Omni media batch rows must contain at least one valid token")
            }
            let sequence = columns.map { ids[row * length + $0] }
            var cursor = 0
            var next: Float = 0
            func write(_ positions: [[Float]]) {
                for p in positions {
                    let column = columns[cursor]
                    for axis in 0 ..< 3 { output[(axis * batch + row) * length + column] = p[axis] }
                    cursor += 1
                }
                if let maximum = positions.flatMap({ $0 }).max() { next = max(next, maximum + 1) }
            }
            while cursor < sequence.count {
                let token = sequence[cursor]
                if token == config.visionStartToken, cursor + 1 < sequence.count {
                    let kind = sequence[cursor + 1]
                    if kind == config.imageToken || kind == config.videoToken
                        || kind == config.audioStartToken
                    {
                        write([[next, next, next]])
                        let interleaved = kind == config.audioStartToken
                        if interleaved { write([[next, next, next]]) }
                        let origin = next
                        let grid: THW
                        let seconds: Double
                        let timestamps: [Double]?
                        if kind == config.imageToken {
                            guard imageIndex < images.count else {
                                throw VLMError.processing("Missing Omni image grid")
                            }
                            grid = images[imageIndex]
                            imageIndex += 1
                            seconds = 1
                            timestamps = nil
                        } else {
                            guard videoIndex < videos.count, videoIndex < secondsPerVideoGrid.count
                            else {
                                throw VLMError.processing("Missing Omni video grid/timing")
                            }
                            grid = videos[videoIndex]
                            seconds = secondsPerVideoGrid[videoIndex]
                            timestamps = videoTimes?[videoIndex]
                            videoIndex += 1
                        }
                        let visual = try gridPositions(
                            grid, seconds: seconds, timestamps: timestamps)
                        var positions: [[Float]] = []
                        var expected: [Int] = []
                        if interleaved {
                            guard audioIndex < audioLengths.count else {
                                throw VLMError.processing(
                                    "Missing audio track for Omni synchronized video")
                            }
                            let audioCount = Qwen3OmniAudio.outputLength(
                                audioLengths[audioIndex], window: config.audioWindow)
                            audioIndex += 1
                            var v = 0
                            var a = 0
                            while v < visual.count || a < audioCount {
                                if v < visual.count && (a == audioCount || visual[v][0] <= Float(a))
                                {
                                    positions.append(visual[v].map { $0 + origin })
                                    expected.append(config.videoToken)
                                    v += 1
                                } else {
                                    positions.append(Array(repeating: origin + Float(a), count: 3))
                                    expected.append(config.audioToken)
                                    a += 1
                                }
                            }
                        } else {
                            positions = visual.map { $0.map { $0 + origin } }
                            expected = Array(repeating: kind, count: visual.count)
                        }
                        guard cursor + expected.count <= sequence.count,
                            Array(sequence[cursor ..< cursor + expected.count]) == expected
                        else {
                            throw VLMError.processing(
                                "Omni visual/audio placeholders do not match grids and synchronized timing"
                            )
                        }
                        write(positions)
                        continue
                    }
                }
                if token == config.audioStartToken {
                    guard audioIndex < audioLengths.count else {
                        throw VLMError.processing("Missing Omni audio length")
                    }
                    let count = Qwen3OmniAudio.outputLength(
                        audioLengths[audioIndex], window: config.audioWindow)
                    audioIndex += 1
                    guard cursor + count < sequence.count,
                        sequence[(cursor + 1) ..< (cursor + count + 1)].allSatisfy({
                            $0 == config.audioToken
                        })
                    else {
                        throw VLMError.processing(
                            "Omni audio token count does not match valid mel frames")
                    }
                }
                write([[next, next, next]])
            }
            deltas[row] = next - Float(columns.count)
        }
        guard imageIndex == images.count, videoIndex == videos.count,
            audioIndex == audioLengths.count
        else {
            throw VLMError.processing(
                "Unused Omni media: message placeholders and media must match")
        }
        return (MLXArray(output, [3, batch, length]), MLXArray(deltas, [batch, 1]))
    }
}
