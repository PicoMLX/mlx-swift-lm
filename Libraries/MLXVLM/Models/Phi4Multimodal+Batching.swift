// Copyright © 2026 Apple Inc.

import MLXLMCommon

extension Phi4Multimodal: BatchedMultimodalModel {
    public var batchedMediaCapabilities: BatchedMediaCapabilities { [.image, .audio] }
    public var batchTokenPlaceholderIDs: Set<Int> { [config.imageToken] }
}
