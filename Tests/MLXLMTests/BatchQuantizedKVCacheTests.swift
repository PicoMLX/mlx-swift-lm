// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXVLM
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

@Suite(.serialized)
struct BatchQuantizedKVCacheTests {

    @Test("Quantized Gemma 4 shared layers use each owner's per-row positions", arguments: [4, 8])
    func gemma4SharedPositions(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyGemma4VLMTextForBatchPositions(numKVSharedLayers: 2)
            let prompts = [[Int32(1), 2, 3], [Int32(4), 5, 6, 7, 8, 9, 10]]
            var singles: [[KVCache]] = prompts.map { prompt in
                let cache: [KVCache] = (0 ..< 2).map { _ in
                    QuantizedKVCache(groupSize: 64, bits: bits)
                }
                eval(model(MLXArray(prompt)[.newAxis], cache: cache).logits)
                return cache
            }
            let batch = try (0 ..< 2).map { layer in
                try mergeCaches(singles.map { $0[layer] })
            }
            for layer in batch {
                #expect(try #require(layer as? BatchQuantizedKVCache).groupSize == 64)
            }
            func compare(_ tokens: MLXArray) {
                let actual = model(tokens, cache: batch)
                #expect(actual.state == nil)
                for row in singles.indices {
                    let expected = model(tokens[row ..< row + 1], cache: singles[row]).logits
                    #expect(maxAbsDifference(actual.logits[row ..< row + 1], expected) < 1e-3)
                }
            }
            for length in [1, 2, 1] {
                let tokens = MLXArray((0 ..< 2 * length).map { Int32(11 + $0) })
                    .reshaped([2, length])
                compare(tokens)
            }
            batch.forEach { $0.filter(batchIndices: MLXArray([Int32(1)])) }
            singles = [singles[1]]
            compare(MLXArray([Int32(21)])[.newAxis])
        }
    }

    @Test("Quantized Idefics3 rows retain their own rotary positions", arguments: [4, 8])
    func idefics3RaggedPositions(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyIdefics3ForBatchPositions()
            func prefill(_ prompt: [Int32]) -> [KVCache] {
                let cache: [KVCache] = (0 ..< 2).map { _ in
                    QuantizedKVCache(groupSize: 64, bits: bits)
                }
                eval(model(MLXArray(prompt)[.newAxis], cache: cache))
                return cache
            }
            var singles = [[Int32(1), 2, 3], [4, 5, 6, 7, 8, 9, 10]].map(prefill)
            let batch = try singles[0].indices.map { layer in
                try mergeCaches(singles.map { $0[layer].copy() })
            }
            let first = try #require(batch.first as? BatchQuantizedKVCache)
            #expect(first.batchOffset.asArray(Int32.self) == [3, 7])

            func decode(
                _ tokens: [[Int32]], sourceLocation: SourceLocation = #_sourceLocation
            ) throws {
                let output = model(
                    MLXArray(tokens.flatMap { $0 }).reshaped([tokens.count, tokens[0].count]),
                    cache: batch)
                eval(output)
                for row in tokens.indices {
                    let expected = model(MLXArray(tokens[row])[.newAxis], cache: singles[row])
                    #expect(
                        maxAbsDifference(output[row ..< row + 1], expected) < 1e-3,
                        "row \(row), \(bits)-bit cache, chunk length \(tokens[row].count)",
                        sourceLocation: sourceLocation)
                }
                for (layer, cache) in batch.enumerated() {
                    let packed = try #require(
                        cache as? BatchQuantizedKVCache, sourceLocation: sourceLocation)
                    #expect(packed.groupSize == 64, sourceLocation: sourceLocation)
                    #expect(packed.bits == bits, sourceLocation: sourceLocation)
                    for row in singles.indices {
                        let single = try #require(
                            singles[row][layer] as? QuantizedKVCache, sourceLocation: sourceLocation
                        )
                        #expect(single.groupSize == 64, sourceLocation: sourceLocation)
                    }
                }
            }
            try decode([[11], [12]])
            try decode([[13, 14], [15, 16]])

            let added = prefill([17, 18])
            let incoming = try added.map { try mergeCaches([$0.copy()]) }
            for layer in batch.indices {
                try batch[layer].validateExtension(other: incoming[layer])
            }
            for layer in batch.indices {
                batch[layer].extend(other: incoming[layer])
            }
            singles.append(added)
            try decode([[19, 20], [21, 22], [23, 24]])

            batch.forEach { $0.filter(batchIndices: batchRows(2, 0)) }
            singles = [singles[2], singles[0]]
            try decode([[25], [26]])
            batch.forEach { $0.filter(batchIndices: batchRows(1)) }
            singles = [singles[1]]
            try decode([[27]])
        }
    }

    @Test("Quantized Mistral rows use their own attention-scaling positions", arguments: [4, 8])
    func mistralPositionScaling(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = withRandomState(MLXRandom.RandomState(seed: 705)) {
                Mistral3TextModel(
                    Mistral3TextConfiguration(
                        hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64,
                        attentionHeads: 4, rmsNormEps: 1e-5, vocabularySize: 64,
                        headDimensions: 64, kvHeads: 2,
                        ropeParameters: [
                            "llama_4_scaling_beta": .float(1),
                            "original_max_position_embeddings": .int(4),
                        ]))
            }
            eval(model)
            let prompts = [[Int32(1), 2, 3, 4, 5, 6], [Int32(7), 8]]
            let rows: [[KVCache]] = prompts.map { prompt in
                let cache: [KVCache] = (0 ..< 2).map { _ in
                    QuantizedKVCache(groupSize: 64, bits: bits)
                }
                eval(model(MLXArray(prompt)[.newAxis], cache: cache))
                return cache
            }
            let batch = try (0 ..< 2).map { layer in
                try mergeCaches(rows.map { $0[layer] })
            }
            for layer in batch {
                #expect(try #require(layer as? BatchQuantizedKVCache).groupSize == 64)
            }
            // Cross different scaling bands, including a multi-token cached forward.
            for count in [1, 2, 1] {
                let tokens = MLXArray((0 ..< 2 * count).map { Int32(11 + $0) })
                    .reshaped([2, count])
                let actual = model(tokens, cache: batch)
                let expected = rows.indices.map { row in
                    model(tokens[row ..< row + 1], cache: rows[row])
                }
                eval(actual)
                for row in rows.indices {
                    #expect(maxAbsDifference(actual[row ..< row + 1], expected[row]) < 1e-4)
                }
            }
        }
    }

    @Test(
        "Empty merges preserve row count through state restoration and the first write",
        arguments: [4, 8])
    func emptyMergeFirstUpdate(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let batch = try QuantizedKVCache.merge([
                QuantizedKVCache(bits: bits), QuantizedKVCache(bits: bits),
            ])
            #expect(batch.batchSize == 2)
            #expect(batch.offset == 0)
            #expect(batch.state.count == 2)
            let restored = BatchQuantizedKVCache(leftPadding: [], bits: bits)
            restored.state = batch.state
            #expect(restored.batchOffset.asArray(Int32.self) == [0, 0])

            let raw = quantizedTestKV(positions: 0 ..< 3, salt: 1, batchSize: 2)
            for cache in [batch, restored] {
                _ = cache.updateQuantized(keys: raw.0, values: raw.1)
                #expect(cache.batchOffset.asArray(Int32.self) == [3, 3])
                for row in 0 ..< 2 {
                    let expected = QuantizedKVCache(bits: bits)
                    _ = expected.updateQuantized(
                        keys: raw.0[row ..< row + 1], values: raw.1[row ..< row + 1])
                    try expectPackedQuantizedRow(cache.extract(row), equals: expected)
                }
            }
        }
    }

    @Test("Ragged merges preserve every packed word, scale and bias", arguments: [4, 8])
    func mergePreservesPackedRows(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let rows = [
                quantizedTestRow(count: 3, salt: 1, bits: bits),
                quantizedTestRow(count: 7, salt: 2, bits: bits),
                QuantizedKVCache(bits: bits),
            ]
            let batch = try QuantizedKVCache.merge(rows)
            #expect(batch.batchSize == 3)
            #expect(batch.leftPadding.asArray(Int32.self) == [4, 0, 7])
            #expect(batch.batchOffset.asArray(Int32.self) == [3, 7, 0])
            #expect(batch.mode == .affine)
            for index in rows.indices {
                try expectPackedQuantizedRow(batch.extract(index), equals: rows[index])
            }

            let next = quantizedTestKV(positions: 7 ..< 8, salt: 3, batchSize: 3)
            _ = batch.updateQuantized(keys: next.0, values: next.1)
            #expect(rows.map(\.offset) == [3, 7, 0])
            try expectPackedQuantizedRow(
                rows[0], equals: quantizedTestRow(count: 3, salt: 1, bits: bits))
        }
    }

    @Test(
        "Packed rows survive allocation growth, reordering, trimming and admission",
        arguments: [4, 8])
    func growthAndRowLifecycle(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let rows = [
                quantizedTestRow(count: 253, salt: 1, bits: bits),
                quantizedTestRow(count: 255, salt: 2, bits: bits),
            ]
            let batch = try QuantizedKVCache.merge(rows)
            let next = [
                quantizedTestKV(positions: 253 ..< 256, salt: 1),
                quantizedTestKV(positions: 255 ..< 258, salt: 2),
            ]
            _ = batch.updateQuantized(
                keys: concatenated(next.map { $0.0 }), values: concatenated(next.map { $0.1 }))
            for index in rows.indices {
                _ = rows[index].updateQuantized(keys: next[index].0, values: next[index].1)
                try expectPackedQuantizedRow(batch.extract(index), equals: rows[index])
            }
            #expect(batch.batchOffset.asArray(Int32.self) == [256, 258])

            batch.filter(batchIndices: batchRows(1, 0))
            try expectPackedQuantizedRow(batch.extract(0), equals: rows[1])
            try expectPackedQuantizedRow(batch.extract(1), equals: rows[0])
            #expect(batch.trim(2) == 2)
            _ = rows[0].trim(2)
            _ = rows[1].trim(2)
            try expectPackedQuantizedRow(batch.extract(0), equals: rows[1])
            try expectPackedQuantizedRow(batch.extract(1), equals: rows[0])

            batch.filter(batchIndices: batchRows(0))
            let admitted = quantizedTestRow(count: 5, salt: 3, bits: bits)
            let added = try QuantizedKVCache.merge([admitted])
            batch.extend(other: added)
            #expect(batch.batchOffset.asArray(Int32.self) == [256, 5])
            try expectPackedQuantizedRow(batch.extract(0), equals: rows[1])
            try expectPackedQuantizedRow(batch.extract(1), equals: admitted)
            #expect(added.batchSize == 1)
            try expectPackedQuantizedRow(added.extract(0), equals: admitted)

            batch.filter(batchIndices: batchRows())
            #expect(batch.batchSize == 0)
            batch.extend(other: added)
            #expect(batch.batchSize == 1)
            try expectPackedQuantizedRow(batch.extract(0), equals: admitted)
        }
    }

    @Test("Right-padded chunked prefill extracts only each row's real tokens", arguments: [4, 8])
    func raggedPrefill(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let batch = BatchQuantizedKVCache(leftPadding: [0, 0], bits: bits)
            let rows = [
                quantizedTestRow(count: 2, salt: 1, bits: bits),
                quantizedTestRow(count: 7, salt: 2, bits: bits),
            ]
            batch.prepare(lengths: [2, 7])
            for range in [0 ..< 3, 3 ..< 7] {
                let rawRows = [
                    quantizedTestKV(positions: range, salt: 1),
                    quantizedTestKV(positions: range, salt: 2),
                ]
                _ = batch.updateQuantized(
                    keys: concatenated(rawRows.map { $0.0 }),
                    values: concatenated(rawRows.map { $0.1 }))
            }
            batch.finalize()
            #expect(batch.leftPadding.asArray(Int32.self) == [5, 0])
            #expect(batch.batchOffset.asArray(Int32.self) == [2, 7])
            try expectPackedQuantizedRow(batch.extract(0), equals: rows[0])
            try expectPackedQuantizedRow(batch.extract(1), equals: rows[1])

            guard case .array(let mask) = batch.makeMask(n: 1, windowSize: nil, returnArray: false)
            else {
                Issue.record("a padded row must retain its attention mask")
                return
            }
            #expect(mask.shape == [2, 1, 1, 8])
            #expect(mask[0, 0, 0, 4].item(Bool.self) == false)
            #expect(mask[0, 0, 0, 5].item(Bool.self) == true)
            #expect(mask[1, 0, 0, 0].item(Bool.self) == true)
        }
    }

    @Test(
        "Filtering a prepared batch preserves the surviving row's real prefix", arguments: [4, 8])
    func filterDuringPrefill(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let batch = BatchQuantizedKVCache(leftPadding: [0, 0], bits: bits)
            batch.prepare(lengths: [2, 7])
            let first = [
                quantizedTestKV(positions: 0 ..< 3, salt: 1),
                quantizedTestKV(positions: 0 ..< 3, salt: 2),
            ]
            _ = batch.updateQuantized(
                keys: concatenated(first.map { $0.0 }), values: concatenated(first.map { $0.1 }))
            batch.filter(batchIndices: batchRows(0))
            let tail = quantizedTestKV(positions: 3 ..< 7, salt: 1)
            _ = batch.updateQuantized(keys: tail.0, values: tail.1)
            batch.finalize()
            try expectPackedQuantizedRow(
                batch.extract(0), equals: quantizedTestRow(count: 2, salt: 1, bits: bits))
            #expect(batch.batchOffset.asArray(Int32.self) == [2])
        }
    }

    @Test(
        "Filtering an all-padding prefix never removes more positions than were written",
        arguments: [4, 8])
    func filterBeforePaddingIsConsumed(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let batch = BatchQuantizedKVCache(leftPadding: [10, 0], bits: bits)
            let prefix = quantizedTestKV(positions: 0 ..< 3, salt: 1, batchSize: 2)
            _ = batch.updateQuantized(keys: prefix.0, values: prefix.1)
            try expectPackedQuantizedRow(batch.extract(0), equals: QuantizedKVCache(bits: bits))
            batch.filter(batchIndices: batchRows(0))
            #expect(batch.offset == 0)
            #expect(batch.leftPadding.asArray(Int32.self) == [7])
            #expect(batch.batchOffset.asArray(Int32.self) == [-7])
            try expectPackedQuantizedRow(batch.extract(0), equals: QuantizedKVCache(bits: bits))

            let tail = quantizedTestKV(positions: 3 ..< 11, salt: 1)
            _ = batch.updateQuantized(keys: tail.0, values: tail.1)
            let expected = QuantizedKVCache(bits: bits)
            let realToken = quantizedTestKV(positions: 10 ..< 11, salt: 1)
            _ = expected.updateQuantized(keys: realToken.0, values: realToken.1)
            #expect(batch.batchOffset.asArray(Int32.self) == [1])
            try expectPackedQuantizedRow(batch.extract(0), equals: expected)
        }
    }

    @Test(
        "Packed staged rounds refuse writes before every row consumes its padding",
        arguments: [4, 8])
    func paddedRoundIsRefused(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            for written in [0, 2] {
                let cache = BatchQuantizedKVCache(leftPadding: [3, 0], bits: bits)
                if written > 0 {
                    let prefix = quantizedTestKV(positions: 0 ..< written, salt: 1, batchSize: 2)
                    _ = cache.updateQuantized(keys: prefix.0, values: prefix.1)
                }
                let state = cache.state.map { $0[.ellipsis] }
                let storage = KVCacheStorage([cache], plan: .disabled)
                #expect(cache.isTrimmable)
                #expect(!cache.isTrimmable(after: -1))
                for width in [1, 2, 5] {
                    #expect(!cache.isTrimmable(after: width))
                    let refused = storage.beginRound(maximumPositions: width)
                    #expect(refused == nil)
                    // Old code admits the round. Write nothing so its rollback is still safe.
                    if let refused { storage.rollback(refused) }
                }
                #expect(cache.offset == written)
                #expect(storage.processedTokenCount == written)
                #expect(storage.nativeAttentionOffsetsAreAligned)
                try expectQuantizedRoundState(cache.state, equals: state)

                #expect(cache.isTrimmable(after: 0))
                let empty = try #require(storage.beginRound(maximumPositions: 0))
                storage.rollback(empty)
                #expect(storage.nativeAttentionOffsetsAreAligned)
                try expectQuantizedRoundState(cache.state, equals: state)
            }
        }
    }

    @Test("Zero-row packed batches allow only empty staged rounds", arguments: [4, 8])
    func zeroRowRoundIsRefused(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = BatchQuantizedKVCache(leftPadding: [], bits: bits)
            let state = cache.state.map { $0[.ellipsis] }
            let storage = KVCacheStorage([cache], plan: .disabled)
            #expect(!cache.isTrimmable(after: -1))
            for width in [1, 2] {
                #expect(!cache.isTrimmable(after: width))
                let refused = storage.beginRound(maximumPositions: width)
                #expect(refused == nil)
                if let refused { storage.rollback(refused) }
            }
            #expect(cache.isTrimmable(after: 0))
            let empty = try #require(storage.beginRound(maximumPositions: 0))
            storage.rollback(empty)
            #expect(cache.offset == 0)
            #expect(cache.batchSize == 0)
            #expect(storage.processedTokenCount == 0)
            #expect(storage.nativeAttentionOffsetsAreAligned)
            try expectQuantizedRoundState(cache.state, equals: state)
        }
    }

    @Test(
        "Packed staged rounds commit exactly once every row consumes its padding",
        arguments: [4, 8], [0, 1])
    func roundAfterPaddingCommitsExactly(bits: Int, retaining: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = BatchQuantizedKVCache(leftPadding: [3, 0], bits: bits)
            let prefix = quantizedTestKV(positions: 0 ..< 3, salt: 1, batchSize: 2)
            _ = cache.updateQuantized(keys: prefix.0, values: prefix.1)
            let storage = KVCacheStorage([cache], plan: .disabled)
            #expect(cache.isTrimmable(after: 2))
            #expect(!cache.isTrimmable(after: -1))
            let round = try #require(storage.beginRound(maximumPositions: 2))
            let staged = try #require(round.caches[0] as? BatchQuantizedKVCache)
            let proposed = quantizedTestKV(positions: 3 ..< 5, salt: 1, batchSize: 2)
            _ = staged.updateQuantized(keys: proposed.0, values: proposed.1)
            let result = storage.commit(round, retaining: retaining)

            #expect(result.committedPositions == retaining)
            #expect(result.discardedPositions == 2 - retaining)
            #expect(cache.offset == 3 + retaining)
            #expect(cache.leftPadding.asArray(Int32.self) == [3, 0])
            #expect(
                cache.batchOffset.asArray(Int32.self) == [Int32(retaining), Int32(3 + retaining)])
            #expect(storage.processedTokenCount == 3 + retaining)
            #expect(storage.nativeAttentionOffsetsAreAligned)
            let expected = QuantizedKVCache(bits: bits)
            let kept = quantizedTestKV(positions: 0 ..< 3 + retaining, salt: 1, batchSize: 2)
            _ = expected.updateQuantized(keys: kept.0, values: kept.1)
            try expectQuantizedRoundState(Array(cache.state.dropLast(2)), equals: expected.state)
        }
    }

    @Test(
        "Copy, state round trip and extracted rows do not share mutable packed storage",
        arguments: [4, 8])
    func copyAndStateIsolation(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let row = quantizedTestRow(count: 6, salt: 1, bits: bits)
            let batch = try QuantizedKVCache.merge([row])
            let copy = try #require(batch.copy() as? BatchQuantizedKVCache)
            let restored = BatchQuantizedKVCache(leftPadding: [], bits: bits)
            restored.state = batch.state.map { $0[.ellipsis] }
            #expect(restored.batchOffset.asArray(Int32.self) == [6])
            #expect(restored.leftPadding.asArray(Int32.self) == [0])
            let extracted = try #require(batch.extract(0) as? QuantizedKVCache)

            let next = quantizedTestKV(positions: 6 ..< 7, salt: 1)
            _ = batch.updateQuantized(keys: next.0, values: next.1)
            try expectPackedQuantizedRow(copy.extract(0), equals: row)
            try expectPackedQuantizedRow(restored.extract(0), equals: row)
            try expectPackedQuantizedRow(extracted, equals: row)
            _ = extracted.updateQuantized(keys: -next.0, values: -next.1)
            let batchRow = try #require(batch.extract(0) as? QuantizedKVCache)
            #expect(
                maxAbsDifference(
                    extracted.toUnquantized().state[0], batchRow.toUnquantized().state[0]) > 0.1)
            #expect(row.offset == 6)

            let empty = BatchQuantizedKVCache(leftPadding: [0, 0], bits: bits)
            let emptyRestored = BatchQuantizedKVCache(leftPadding: [], bits: bits)
            emptyRestored.state = empty.state
            #expect(emptyRestored.batchSize == 2)
            #expect(emptyRestored.offset == 0)
        }
    }

    @Test("Effective group size follows both key and value head dimensions", arguments: [4, 8])
    func effectiveGroupSize(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let batch = BatchQuantizedKVCache(leftPadding: [0], groupSize: 64, bits: bits)
            let single = QuantizedKVCache(groupSize: 64, bits: bits)
            let raw = quantizedTestKV(positions: 0 ..< 5, salt: 1, keyDim: 32, valueDim: 96)
            _ = batch.updateQuantized(keys: raw.0, values: raw.1)
            _ = single.updateQuantized(keys: raw.0, values: raw.1)
            #expect(batch.groupSize == 32)
            #expect(single.groupSize == 32)
            try expectPackedQuantizedRow(batch.extract(0), equals: single)
            let copy = try #require(single.copy() as? QuantizedKVCache)
            let merged = try QuantizedKVCache.merge([single, copy])
            #expect(merged.groupSize == 32)
            #expect(merged.batchSize == 2)
        }
    }

    @Test("Merge rejects mismatched precision, group size, mode and exact cache types")
    func mergeRefusals() throws {
        #expect(throws: KVCacheError.self) {
            _ = try QuantizedKVCache.merge([QuantizedKVCache(bits: 4), QuantizedKVCache(bits: 8)])
        }
        #expect(throws: KVCacheError.self) {
            _ = try QuantizedKVCache.merge([
                QuantizedKVCache(groupSize: 32), QuantizedKVCache(groupSize: 64),
            ])
        }
        #expect(throws: KVCacheError.self) {
            _ = try QuantizedKVCache.merge([QuantizedKVCache(groupSize: 32, bits: 4, mode: .mxfp4)])
        }
        #expect(throws: KVCacheError.self) {
            _ = try mergeCaches([QuantizedCacheSubclass()])
        }
        #expect(throws: KVCacheError.self) {
            _ = try mergeCaches([QuantizedKVCache(), KVCacheSimple()])
        }
        #expect(throws: KVCacheError.self) { _ = try QuantizedKVCache.merge([]) }
    }

    @Test("A quantized row takes no row operations; its packed batch does")
    func batchableConformance() throws {
        #expect(!(QuantizedKVCache() as any KVCache is any BatchableKVCache))
        let batch = try BatchQuantizedKVCache.merge([QuantizedKVCache()])
        #expect(batch is BatchQuantizedKVCache)
        #expect(throws: KVCacheError.self) { _ = try BatchQuantizedKVCache.merge([any KVCache]()) }
        #expect(throws: KVCacheError.self) {
            _ = try BatchQuantizedKVCache.merge([QuantizedCacheSubclass()])
        }
        #expect(throws: KVCacheError.self) { _ = try mergeCaches([batch]) }
    }

    @Test("Extension validation accepts empty rows and packed ragged batches", arguments: [4, 8])
    func extensionValidationAcceptsCompatibleRows(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let batch = try QuantizedKVCache.merge([
                quantizedTestRow(count: 3, salt: 1, bits: bits), QuantizedKVCache(bits: bits),
            ])
            let added = try QuantizedKVCache.merge([
                quantizedTestRow(count: 7, salt: 2, bits: bits)
            ])
            let original = try #require(batch.copy() as? BatchQuantizedKVCache)
            try batch.validateExtension(other: added)
            #expect(batch.batchSize == 2)
            #expect(batch.offset == original.offset)
            for row in 0 ..< batch.batchSize {
                try expectPackedQuantizedRow(
                    batch.extract(row),
                    equals: try #require(original.extract(row) as? QuantizedKVCache))
            }
            let empty = BatchQuantizedKVCache(leftPadding: [0, 0], bits: bits)
            try empty.validateExtension(other: added)
            try added.validateExtension(other: empty)
            try empty.validateExtension(other: BatchQuantizedKVCache(leftPadding: [], bits: bits))
            empty.filter(batchIndices: batchRows())
            try empty.validateExtension(other: added)
            try added.validateExtension(other: empty)
        }
    }

    @Test("Extension validation rejects precision, layout, metadata and prepared-state mismatches")
    func extensionValidationRefusals() throws {
        try Device.withDefaultDevice(.cpu) {
            let batch = try QuantizedKVCache.merge([quantizedTestRow(count: 3, salt: 1, bits: 4)])
            #expect(throws: KVCacheError.self) {
                try batch.validateExtension(other: BatchQuantizedKVCache(leftPadding: [0], bits: 8))
            }
            #expect(throws: KVCacheError.self) {
                try batch.validateExtension(
                    other: BatchQuantizedKVCache(leftPadding: [0], groupSize: 32, bits: 4))
            }
            #expect(throws: KVCacheError.self) {
                try batch.validateExtension(other: BatchKVCache(leftPadding: [0]))
            }
            let prepared = BatchQuantizedKVCache(leftPadding: [0, 0], bits: 4)
            prepared.prepare(lengths: [1, 2])
            #expect(throws: KVCacheError.self) { try batch.validateExtension(other: prepared) }
            #expect(throws: KVCacheError.self) { try prepared.validateExtension(other: batch) }

            let differentHeads = BatchQuantizedKVCache(leftPadding: [0], bits: 4)
            let raw = quantizedTestKV(positions: 0 ..< 2, salt: 1, heads: 1)
            _ = differentHeads.updateQuantized(keys: raw.0, values: raw.1)
            #expect(throws: KVCacheError.self) {
                try batch.validateExtension(other: differentHeads)
            }
            let differentDtype = try #require(batch.copy() as? BatchQuantizedKVCache)
            var halfState = differentDtype.state
            for index in [1, 2, 4, 5] { halfState[index] = halfState[index].asType(.float16) }
            differentDtype.state = halfState
            #expect(throws: KVCacheError.self) {
                try batch.validateExtension(other: differentDtype)
            }
            for malformedIndex in [0, 1, 2] {
                let malformed = try #require(batch.copy() as? BatchQuantizedKVCache)
                var state = malformed.state
                switch malformedIndex {
                case 0: state[0] = state[0].asType(.int32)
                case 1: state[1] = state[1][0, .ellipsis]
                default: state[2] = state[2][.ellipsis, ..<0]
                }
                malformed.state = state
                #expect(throws: KVCacheError.self) {
                    try batch.validateExtension(other: malformed)
                }
            }
            let invalidOffset = try #require(batch.copy() as? BatchQuantizedKVCache)
            invalidOffset.offset += 1
            #expect(throws: KVCacheError.self) { try batch.validateExtension(other: invalidOffset) }
        }
    }

    @Test(
        "Packed state remains compressed and runtime reports affine compatibility",
        arguments: [4, 8])
    func packedMemoryAndRuntime(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let rows = [
                quantizedTestRow(count: 16, salt: 1, bits: bits),
                quantizedTestRow(count: 16, salt: 2, bits: bits),
            ]
            let batch = try QuantizedKVCache.merge(rows)
            let packed = batch.state
            try #require(packed.count == 8)
            #expect(packed[0].dtype == .uint32)
            #expect(packed[3].dtype == .uint32)
            #expect(packed[0].shape == [2, 2, 16, 2 * bits])
            #expect(packed[3].shape == [2, 2, 16, 2 * bits])
            let logicalPackedBytes = packed.prefix(6).reduce(0) { $0 + $1.nbytes }
            let float32KVBytes = 2 * 2 * 16 * 64 * 4 * 2
            #expect(logicalPackedBytes < float32KVBytes / 2)

            let affine = KVCacheConfiguration(
                strategy: .affine(bits == 4 ? .fourBit : .eightBit))
            let status = KVCacheStatus(cache: [batch], configuration: affine)
            #expect(status.compressedLayerCount == 1)
            #expect(status.memoryBytes == batch.innerState().reduce(0) { $0 + $1.nbytes })
            let report = kvCacheRuntimeReport(cache: [batch], configuration: affine)
            try #require(report.layers.count == 1)
            #expect(report.layers[0].state == .active)
            #expect(report.layers[0].resolvedStrategy == .affine)
            try validateKVCacheCompatibility([batch], configuration: affine)
            #expect(throws: KVCacheConfigurationError.self) {
                try validateKVCacheCompatibility([batch], configuration: KVCacheConfiguration())
            }
            let leaf = try #require(KVCacheTree.leaves(in: [batch]).first)
            #expect(!leaf.participatesInTurboQuantBoundaryProtection)
        }
    }

    @Test(
        "Grouped-query attention over ragged packed rows matches separate quantized caches",
        arguments: [4, 8])
    func attentionParity(bits: Int) throws {
        let singles = [
            quantizedTestRow(count: 3, salt: 1, bits: bits),
            quantizedTestRow(count: 7, salt: 2, bits: bits),
        ]
        let batch = try QuantizedKVCache.merge(singles)
        for step in 0 ..< 3 {
            let raw = [
                quantizedTestKV(positions: (3 + step) ..< (4 + step), salt: 1),
                quantizedTestKV(positions: (7 + step) ..< (8 + step), salt: 2),
            ]
            let queries = quantizedTestKV(
                positions: step ..< step + 1, salt: 9, batchSize: 2, heads: 4
            ).0
            let actual = attentionWithCacheUpdate(
                queries: queries, keys: concatenated(raw.map { $0.0 }),
                values: concatenated(raw.map { $0.1 }), cache: batch,
                scale: 0.125, mask: batch.makeMask(n: 1, windowSize: nil, returnArray: false))
            for row in singles.indices {
                let expected = attentionWithCacheUpdate(
                    queries: queries[row ..< row + 1], keys: raw[row].0,
                    values: raw[row].1, cache: singles[row], scale: 0.125)
                #expect(
                    maxAbsDifference(actual[row ..< row + 1], expected)
                        < MatmulPrecision.splitTolerance)
                try expectPackedQuantizedRow(batch.extract(row), equals: singles[row])
            }
        }
        batch.filter(batchIndices: batchRows(1))
        let raw = quantizedTestKV(positions: 10 ..< 11, salt: 2)
        let query = quantizedTestKV(positions: 3 ..< 4, salt: 9, heads: 4).0
        let actual = attentionWithCacheUpdate(
            queries: query, keys: raw.0, values: raw.1, cache: batch,
            scale: 0.125, mask: batch.makeMask(n: 1, windowSize: nil, returnArray: false))
        let expected = attentionWithCacheUpdate(
            queries: query, keys: raw.0, values: raw.1, cache: singles[1], scale: 0.125)
        #expect(maxAbsDifference(actual, expected) < MatmulPrecision.splitTolerance)
    }
}

private final class QuantizedCacheSubclass: QuantizedKVCache {}

private func quantizedTestRow(count: Int, salt: Int, bits: Int) -> QuantizedKVCache {
    let cache = QuantizedKVCache(bits: bits)
    let raw = quantizedTestKV(positions: 0 ..< count, salt: salt)
    _ = cache.updateQuantized(keys: raw.0, values: raw.1)
    return cache
}

private func quantizedTestKV(
    positions: Range<Int>, salt: Int, batchSize: Int = 1, heads: Int = 2,
    keyDim: Int = 64, valueDim: Int = 64
) -> (MLXArray, MLXArray) {
    func tensor(dim: Int, phase: Float) -> MLXArray {
        var data: [Float] = []
        data.reserveCapacity(batchSize * heads * positions.count * dim)
        for row in 0 ..< batchSize {
            for head in 0 ..< heads {
                let bias = Float(salt + row) * 0.2 + Float(head) * 0.03
                for position in positions {
                    let angle = Float(position) * 0.31 + phase
                    for component in 0 ..< dim {
                        data.append(sin(Float(component) * 0.17 + angle) + bias)
                    }
                }
            }
        }
        return MLXArray(data).reshaped([batchSize, heads, positions.count, dim])
    }
    return (tensor(dim: keyDim, phase: 0), tensor(dim: valueDim, phase: 0.7))
}

private func expectQuantizedRoundState(
    _ actual: [MLXArray], equals expected: [MLXArray],
    sourceLocation: SourceLocation = #_sourceLocation
) throws {
    try #require(actual.count == expected.count, sourceLocation: sourceLocation)
    for (actual, expected) in zip(actual, expected) {
        try #require(actual.shape == expected.shape, sourceLocation: sourceLocation)
        #expect(actual.dtype == expected.dtype, sourceLocation: sourceLocation)
        #expect(all(actual .== expected).item(Bool.self), sourceLocation: sourceLocation)
    }
}

private func expectPackedQuantizedRow(_ actual: any KVCache, equals expected: QuantizedKVCache)
    throws
{
    let actual = try #require(actual as? QuantizedKVCache)
    #expect(actual.groupSize == expected.groupSize)
    #expect(actual.bits == expected.bits)
    #expect(actual.mode == expected.mode)
    #expect(actual.offset == expected.offset)
    let actualState = actual.state
    let expectedState = expected.state
    try #require(actualState.count == expectedState.count)
    for (actualArray, expectedArray) in zip(actualState, expectedState) {
        try #require(actualArray.shape == expectedArray.shape)
        #expect(actualArray.dtype == expectedArray.dtype)
        #expect(all(actualArray .== expectedArray).item(Bool.self))
    }
}

@Suite(.serialized)
struct QuantizedBatchAdmissionTests {

    @Test(
        "Rejected quantized admission preserves every active layer, queued prefix and sampler stream",
        arguments: QuantizedAdmissionMismatch.allCases, [false, true])
    func rejectedAdmission(mismatch: QuantizedAdmissionMismatch, keepQueued: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = QuantizedAdmissionModel()
            let activePrompt = [1, 3, 5]
            let queuedPrompt = [2, 4]
            let prefix = mismatch.caches(incoming: true)
            _ = model(
                LMInput.Text(tokens: MLXArray([Int32(9), 10])[.newAxis]),
                cache: prefix, state: nil)
            let prefixState = AdmissionCacheSnapshot(prefix)
            let expectedActive = try admissionTokens(
                model, prompt: activePrompt, caches: mismatch.caches(incoming: false),
                seed: 17, maxTokens: 6)
            let expectedQueued = try admissionTokens(
                model, prompt: queuedPrompt, caches: prefix.map { $0.copy() },
                seed: 18, maxTokens: 4)

            var iterator = BatchTokenIterator(model: model)
            let active = iterator.insert(
                prompts: [activePrompt], maxTokens: [6],
                caches: [mismatch.caches(incoming: false)],
                samplers: [TopPSampler(temperature: 0.8, topP: 0.9, seed: 17)])
            let first = try iterator.next().map(\.token)
            #expect(first == Array(expectedActive.prefix(1)))
            let activeCache = try #require(model.lastCache)
            if mismatch == .groupSize {
                #expect(try #require(activeCache[1] as? BatchQuantizedKVCache).groupSize == 32)
                #expect(try #require(prefix[1] as? QuantizedKVCache).groupSize == 64)
            }
            let activeState = AdmissionCacheSnapshot(activeCache)
            let queuedSampler = TopPSampler(temperature: 0.8, topP: 0.9, seed: 18)
            let queued = iterator.insert(
                prompts: [queuedPrompt], maxTokens: [4], caches: [prefix],
                samplers: [queuedSampler])

            for _ in 0 ..< 2 {
                #expect(throws: KVCacheError.self) { _ = try iterator.next() }
                #expect(iterator.uids == active + queued)
                try activeState.expectUnchanged(activeCache)
                try prefixState.expectUnchanged(prefix)
            }
            if keepQueued {
                iterator.remove(active)
                #expect(try collectAdmission(&iterator, uids: queued) == [expectedQueued])
            } else {
                iterator.remove(queued)
                let remainder = try collectAdmission(&iterator, uids: active)
                #expect(first + remainder[0] == expectedActive)
                let retried = iterator.insert(
                    prompts: [queuedPrompt], maxTokens: [4], caches: [prefix],
                    samplers: [queuedSampler])
                #expect(try collectAdmission(&iterator, uids: retried) == [expectedQueued])
            }
            try prefixState.expectUnchanged(prefix)
        }
    }

    @Test(
        "Matching quantized rows join without changing either seeded token stream",
        arguments: [4, 8])
    func compatibleAdmission(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let model = QuantizedAdmissionModel()
            func caches() -> [KVCache] {
                [KVCacheSimple(), QuantizedKVCache(groupSize: 64, bits: bits)]
            }
            let prompts = [[1, 3, 5], [2, 4]]
            let expected = try prompts.indices.map { index in
                try admissionTokens(
                    model, prompt: prompts[index], caches: caches(), seed: UInt64(17 + index),
                    maxTokens: 6)
            }
            var iterator = BatchTokenIterator(model: model)
            let active = iterator.insert(
                prompts: [prompts[0]], maxTokens: [6], caches: [caches()],
                samplers: [TopPSampler(temperature: 0.8, topP: 0.9, seed: 17)])
            let first = try iterator.next().map(\.token)
            let queued = iterator.insert(
                prompts: [prompts[1]], maxTokens: [6], caches: [caches()],
                samplers: [TopPSampler(temperature: 0.8, topP: 0.9, seed: 18)])
            let remainder = try collectAdmission(&iterator, uids: active + queued)
            #expect(first + remainder[0] == expected[0])
            #expect(remainder[1] == expected[1])
        }
    }
}

enum QuantizedAdmissionMismatch: CaseIterable, Sendable {
    case precision
    case groupSize
    case floatToQuantized
    case quantizedToFloat
    case compositeChild

    fileprivate func caches(incoming: Bool) -> [KVCache] {
        let first = QuantizedKVCache(groupSize: 64, bits: 4)
        switch self {
        case .precision:
            return [first, QuantizedKVCache(groupSize: 64, bits: incoming ? 8 : 4)]
        case .groupSize:
            return [first, QuantizedKVCache(groupSize: incoming ? 64 : 32, bits: 4)]
        case .floatToQuantized:
            return [first, incoming ? QuantizedKVCache(bits: 4) : KVCacheSimple()]
        case .quantizedToFloat:
            return [first, incoming ? KVCacheSimple() : QuantizedKVCache(bits: 4)]
        case .compositeChild:
            return [
                first,
                CacheList(caches: [
                    KVCacheSimple(), QuantizedKVCache(groupSize: 64, bits: incoming ? 8 : 4),
                ]),
            ]
        }
    }
}

/// All cache writes use 64-wide heads, so group sizes 32 and 64 remain distinct.
private final class QuantizedAdmissionModel: Module, LanguageModel {
    private(set) var lastCache: [KVCache]?

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let tokens = input.tokens
        let keys =
            tokens.asType(.float32)[0..., .newAxis, 0..., .newAxis]
            + MLXArray.arange(64).asType(.float32) * 0.01
        for layer in cache ?? [] { update(layer, keys: keys) }
        lastCache = cache
        let target = tokens[0..., -1, .newAxis, .newAxis].asType(.float32)
        let logits = -abs(MLXArray.arange(32).asType(.float32)[.newAxis, .newAxis] - target) * 0.07
        return LMOutput(logits: logits)
    }

    private func update(_ cache: KVCache, keys: MLXArray) {
        if let composite = cache as? CacheList {
            for child in composite.children { update(child, keys: keys) }
        } else if let quantized = cache as? any QuantizedKVCacheProtocol {
            _ = quantized.updateQuantized(keys: keys, values: keys + 0.2)
        } else {
            _ = cache.update(keys: keys, values: keys + 0.2)
        }
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }
}

private struct AdmissionCacheSnapshot {
    let offsets: [Int]
    let states: [[MLXArray]]

    init(_ caches: [KVCache]) {
        offsets = caches.map(\.offset)
        states = caches.map { $0.state.map { $0[.ellipsis] } }
        eval(states.flatMap { $0 })
    }

    func expectUnchanged(_ caches: [KVCache], sourceLocation: SourceLocation = #_sourceLocation)
        throws
    {
        #expect(caches.map(\.offset) == offsets, sourceLocation: sourceLocation)
        try #require(caches.count == states.count, sourceLocation: sourceLocation)
        for (cache, expected) in zip(caches, states) {
            let actual = cache.state
            try #require(actual.count == expected.count, sourceLocation: sourceLocation)
            for (lhs, rhs) in zip(actual, expected) {
                try #require(lhs.shape == rhs.shape, sourceLocation: sourceLocation)
                #expect(lhs.dtype == rhs.dtype, sourceLocation: sourceLocation)
                #expect(all(lhs .== rhs).item(Bool.self), sourceLocation: sourceLocation)
            }
        }
    }
}

private func admissionTokens(
    _ model: any LanguageModel, prompt: [Int], caches: [KVCache], seed: UInt64, maxTokens: Int
) throws -> [Int] {
    var iterator = BatchTokenIterator(model: model)
    let uids = iterator.insert(
        prompts: [prompt], maxTokens: [maxTokens], caches: [caches],
        samplers: [TopPSampler(temperature: 0.8, topP: 0.9, seed: seed)])
    return try collectAdmission(&iterator, uids: uids)[0]
}

private func collectAdmission(_ iterator: inout BatchTokenIterator, uids: [Int]) throws -> [[Int]] {
    var output = [Int: [Int]]()
    while true {
        let responses = try iterator.next()
        if responses.isEmpty { break }
        for response in responses { output[response.uid, default: []].append(response.token) }
    }
    return uids.map { output[$0] ?? [] }
}
