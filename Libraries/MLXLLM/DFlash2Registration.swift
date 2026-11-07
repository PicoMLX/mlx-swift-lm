// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers the DFlash2 drafter model type, which also loads DFlash v1 checkpoints.
///
/// Call once before loading a drafter through `DFlash2DrafterModelFactory`.
/// DFlash checkpoints report `model_type: qwen3`; the `dflash_config` object or
/// the `DFlashDraftModel` / `DFlash2DraftModel` architecture tells them apart
/// from a Qwen3 language model.
public enum DFlash2Registration {
    public static func register() async {
        await DFlash2DrafterTypeRegistry.shared.registerModelType(
            "qwen3",
            matches: isDFlash2Configuration,
            creator: { data in
                let config = try JSONDecoder.json5().decode(DFlash2Configuration.self, from: data)
                return DFlash2DraftModel(config)
            }
        )
    }
}

private func isDFlash2Configuration(_ data: Data) -> Bool {
    guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        return false
    }
    if root["dflash_config"] is [String: Any] { return true }
    let architectures = root["architectures"] as? [String] ?? []
    return architectures.contains("DFlashDraftModel") || architectures.contains("DFlash2DraftModel")
}
