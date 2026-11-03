// Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
// Licensed under the Apache License, Version 2.0. See MiniCPM-Canvas-LICENSE.
// Swift adaptation of mlx-vlm/minicpmv4_7/rope.py at 4693cd0.
import Foundation
import MLX
import MLXLMCommon

package struct MiniCPMCanvas {
    struct Markers {
        let imageStart: Int, imageEnd: Int, sliceStart: Int, sliceEnd: Int, newline: Int
        var all: Set<Int> { [imageStart, imageEnd, sliceStart, sliceEnd, newline] }
    }
    struct Frame {
        var modality: Int, thumbnail: Range<Int>, end: Int
        var slices: [Range<Int>] = []
    }
    struct Group { var start: Int, end: Int, frames: [Frame] }
    static func groups(_ ids: [Int], markers: Markers) throws -> [Group] {
        var groups: [Group] = []
        var previousEnd = 0
        var cursor = 0
        while cursor < ids.count {
            guard ids[cursor] == MiniCPM.imageToken || ids[cursor] == MiniCPM.videoToken else {
                cursor += 1
                continue
            }
            let start = cursor
            let modality = ids[cursor]
            while cursor < ids.count && ids[cursor] == modality { cursor += 1 }
            guard start > 0 else {
                throw VLMError.processing("MiniCPM canvas crops require opening markers")
            }
            let crop = start ..< cursor
            if !groups.isEmpty, ids[start - 1] == markers.sliceStart {
                groups[groups.count - 1].frames[groups[groups.count - 1].frames.count - 1].slices
                    .append(crop)
            } else {
                let frame = Frame(modality: modality, thumbnail: crop, end: cursor)
                let adjacent =
                    !groups.isEmpty
                    && ids[previousEnd ..< (start - 1)].allSatisfy { markers.all.contains($0) }
                if adjacent {
                    groups[groups.count - 1].frames.append(frame)
                } else {
                    groups.append(Group(start: start - 1, end: cursor, frames: [frame]))
                }
            }
            previousEnd = cursor
        }
        for g in groups.indices {
            for f in groups[g].frames.indices {
                var end =
                    groups[g].frames[f].slices.last?.upperBound
                    ?? groups[g].frames[f].thumbnail.upperBound
                while end < ids.count && markers.all.contains(ids[end]) { end += 1 }
                groups[g].frames[f].end = end
            }
            groups[g].end = groups[g].frames.last!.end
        }
        return groups
    }
    static func positions(
        tokens: [Int], lengths: [Int], width: Int, images: [THW], videos: [THW],
        factor: Int, rowFactors: [Int]? = nil, markers: Markers
    ) throws -> (MLXArray, MLXArray) {
        let batch = lengths.count
        var output = [Int32](repeating: 0, count: 3 * batch * width)
        var deltas = [Int32]()
        var cropIndex = [MiniCPM.imageToken: 0, MiniCPM.videoToken: 0]
        func grid(_ modality: Int) throws -> THW {
            let grids = modality == MiniCPM.imageToken ? images : videos
            let index = cropIndex[modality, default: 0]
            guard index < grids.count else {
                throw VLMError.processing("MiniCPM canvas has more crop spans than grids")
            }
            cropIndex[modality] = index + 1
            return grids[index]
        }
        for row in 0 ..< batch {
            let factor = rowFactors?[row] ?? factor
            let length = lengths[row]
            let ids = Array(tokens[(row * width) ..< (row * width + length)])
            var p = (0 ..< 3).map { _ in Array(0 ..< length) }
            func text(_ range: Range<Int>, from start: Int) {
                for axis in 0 ..< 3 {
                    for (offset, index) in range.enumerated() { p[axis][index] = start + offset }
                }
            }
            func vision(
                _ range: Range<Int>, _ size: THW, origin: Int, canvasH: Int = 0, canvasW: Int = 0,
                offsetH: Int = 0, offsetW: Int = 0
            ) throws {
                let h = size.h / factor
                let w = size.w / factor
                guard h * w == range.count else {
                    throw VLMError.processing("MiniCPM canvas grid does not match its feature span")
                }
                let ch = canvasH > 0 ? canvasH : h
                let cw = canvasW > 0 ? canvasW : w
                for y in 0 ..< h {
                    for x in 0 ..< w {
                        let index = range.lowerBound + y * w + x
                        let yy =
                            h == 1
                            ? 0
                            : Int(
                                (Double(y) * Double(ch - 1) / Double(h - 1)).rounded(
                                    .toNearestOrEven))
                        let xx =
                            w == 1
                            ? 0
                            : Int(
                                (Double(x) * Double(cw - 1) / Double(w - 1)).rounded(
                                    .toNearestOrEven))
                        p[0][index] = origin
                        p[1][index] = origin + offsetH + yy
                        p[2][index] = origin + offsetW + xx
                    }
                }
            }
            var current = 0
            var index = 0
            for group in try groups(ids, markers: markers) {
                if group.start > index {
                    text(index ..< group.start, from: current)
                    current += group.start - index
                }
                var frameCursor = group.start
                for frame in group.frames {
                    let thumbGrid = try grid(frame.modality)
                    let start = frame.thumbnail.lowerBound - 1
                    let end = min(frame.end, group.end)
                    if start > frameCursor {
                        text(frameCursor ..< start, from: current)
                        current += start - frameCursor + 1
                    }
                    let origin = current
                    var canvasH = thumbGrid.h / factor
                    var canvasW = thumbGrid.w / factor
                    var columns = 0
                    var sliceH = 0
                    var sliceW = 0
                    var firstSlice: THW?
                    if !frame.slices.isEmpty {
                        columns = frame.slices.count
                        for k in 0 ..< (frame.slices.count - 1)
                        where frame.slices[k + 1].lowerBound - frame.slices[k].upperBound > 2 {
                            columns = k + 1
                            break
                        }
                        var rows = frame.slices.count / columns
                        if rows * columns != frame.slices.count {
                            rows = 1
                            columns = frame.slices.count
                        }
                        firstSlice = try grid(frame.modality)
                        sliceH = firstSlice!.h / factor
                        sliceW = firstSlice!.w / factor
                        canvasH = rows * sliceH
                        canvasW = columns * sliceW
                    }
                    for axis in 0 ..< 3 { for j in start ..< end { p[axis][j] = origin } }
                    p[1][start] = max(origin - 1, 0)
                    p[2][start] = max(origin - 1, 0)
                    if frame.thumbnail.upperBound < end {
                        p[1][frame.thumbnail.upperBound] = origin + canvasH
                        p[2][frame.thumbnail.upperBound] = origin + canvasW
                    }
                    try vision(
                        frame.thumbnail, thumbGrid, origin: origin, canvasH: canvasH,
                        canvasW: canvasW)
                    for (k, slice) in frame.slices.enumerated() {
                        let offsetH = k / columns * sliceH
                        let offsetW = k % columns * sliceW
                        let size = k == 0 ? firstSlice! : try grid(frame.modality)
                        p[1][slice.lowerBound - 1] = origin + offsetH
                        p[2][slice.lowerBound - 1] = origin + offsetW
                        if slice.upperBound < end {
                            p[1][slice.upperBound] = origin + offsetH + size.h / factor - 1
                            p[2][slice.upperBound] = origin + offsetW + size.w / factor - 1
                        }
                        try vision(slice, size, origin: origin, offsetH: offsetH, offsetW: offsetW)
                    }
                    if frame.slices.count > 1 {
                        for k in 0 ..< (frame.slices.count - 1) {
                            let gapStart = frame.slices[k].upperBound
                            let gapEnd = frame.slices[k + 1].lowerBound
                            if gapEnd - gapStart > 2 {
                                for newline in (gapStart + 1) ..< (gapEnd - 1) {
                                    p[1][newline] = origin + (k / columns + 1) * sliceH - 1
                                    p[2][newline] = origin + columns * sliceW
                                }
                            }
                        }
                    }
                    current = origin + max(canvasH, canvasW) + 1
                    frameCursor = end
                }
                if frameCursor < group.end {
                    text(frameCursor ..< group.end, from: current)
                    current += group.end - frameCursor
                }
                index = group.end
            }
            if index < length { text(index ..< length, from: current) }
            let maxPosition = p.flatMap { $0 }.max() ?? -1
            deltas.append(Int32(maxPosition + 1 - length))
            for axis in 0 ..< 3 {
                for j in 0 ..< length {
                    output[(axis * batch + row) * width + j] = Int32(p[axis][j])
                }
            }
        }
        guard cropIndex[MiniCPM.imageToken, default: 0] == images.count,
            cropIndex[MiniCPM.videoToken, default: 0] == videos.count
        else {
            throw VLMError.processing("MiniCPM canvas received unused crop grids")
        }
        return (MLXArray(output, [3, batch, width]), MLXArray(deltas, [batch, 1]))
    }
}
