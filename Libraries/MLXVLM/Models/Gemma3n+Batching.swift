// Copyright © 2026 Apple Inc.

import MLXLMCommon

extension Gemma3n: BatchedMultimodalModel {
    public var batchedMediaCapabilities: BatchedMediaCapabilities { [.image, .audio] }
}
