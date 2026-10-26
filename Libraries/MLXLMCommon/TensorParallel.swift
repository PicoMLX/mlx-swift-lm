// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// A model that can keep only its share of the weights, so that a group of processes
/// runs it together with tensor parallelism.
///
/// ``loadShardedWeights(modelDirectory:model:group:quantization:perLayerQuantization:weightFileSelection:)``
/// shards a model through this protocol before it reads the weights. Port of the `shard`
/// methods of mlx-lm's models.
public protocol TensorParallelModel {
    /// Keep this process' share of the model.
    ///
    /// Every process in `group` holds the same weights and calls this before its first
    /// forward pass; they must then run the same forward passes in the same order. A
    /// group of one keeps every weight. Throws when a dimension does not divide by the
    /// group size; discard the model then.
    func shard(group: MLXDistributed.Group) throws
}

/// Why a model cannot be sharded.
public enum TensorParallelError: Error, Equatable {
    /// The model holds unfused LoRA or DoRA layers, which sharding would drop. Fuse the
    /// adapters into the model first.
    case unfusedAdapters

    /// The model already holds shards.
    case alreadySharded
}

/// Throws when sharding `model` would drop its adapters or split its shards again.
package func validateShardable(_ model: Module) throws {
    let modules = model.modules()
    let sharded = modules.contains {
        $0 is AllToShardedLinear || $0 is ShardedToAllLinear
            || $0 is QuantizedAllToShardedLinear || $0 is QuantizedShardedToAllLinear
    }
    guard !sharded else { throw TensorParallelError.alreadySharded }
    guard !modules.contains(where: { $0 is LoRALayer }) else {
        throw TensorParallelError.unfusedAdapters
    }
}

/// Throws when a named dimension does not divide by the group `size`, before any layer
/// is sharded.
package func requireDivisible(_ dimensions: KeyValuePairs<String, Int>, by size: Int) throws {
    for (name, count) in dimensions where count % size != 0 {
        throw ShardingError.indivisible(dimension: name, of: count, across: size)
    }
}

extension Module {
    /// Replaces `Linear` children with this process' shards. The `allToSharded` keys
    /// split their outputs; the `shardedToAll` keys split their inputs and sum the
    /// results across `group`.
    package func shardLinears(
        allToSharded: [String] = [], shardedToAll: [String] = [], group: MLXDistributed.Group
    ) throws {
        let children = children()
        var sharded = [String: NestedItem<String, Module>]()
        for (keys, sharding) in [
            (allToSharded, ShardingType.allToSharded), (shardedToAll, .shardedToAll),
        ] {
            for key in keys {
                guard case .value(let layer as Linear) = children[key] else {
                    preconditionFailure("\(key) is not a Linear child of \(type(of: self))")
                }
                sharded[key] = .value(try shardLinear(layer, sharding: sharding, group: group))
            }
        }
        try update(modules: ModuleChildren(values: sharded), verify: [])
    }
}
