#!/usr/bin/env python3
"""Regenerate tiny Phi4MM fixtures using unmodified network modules at the pinned SHA.

Usage: python generate_phi4mm_reference.py /path/to/mlx-vlm
Requires MLX and NumPy. No checkpoint downloads or remote Python code are used.
"""
import ast
import dataclasses
import importlib
import inspect
import json
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

REFERENCE_SHA = "4693cd0eaf022ae7ef61c6fdb55e17e6399879a7"
# CPU f32 avoids Metal's shape-dependent merged/unfused GEMM rounding.
# The network remains the unmodified pinned, merged-LoRA reference.
mx.set_default_device(mx.cpu)
root = Path(sys.argv[1]).resolve()
for name, path in [("mlx_vlm", root / "mlx_vlm"), ("mlx_vlm.models", root / "mlx_vlm/models"),
                   ("mlx_vlm.models.phi4mm", root / "mlx_vlm/models/phi4mm")]:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module

# Keep the reference's dataclasses/mask function while avoiding its unrelated server,
# quantization and Transformers import graph. Numerical network code is unchanged.
base_path = root / "mlx_vlm/models/base.py"
base = types.ModuleType("mlx_vlm.models.base")
base.__dict__.update(mx=mx, nn=nn, inspect=inspect, dataclass=dataclasses.dataclass,
                     Any=Any, Dict=Dict, List=List, Optional=Optional)
cache_module = importlib.import_module("mlx_vlm.models.cache")
base.create_causal_mask = cache_module.create_causal_mask
selected = {"BaseModelConfig", "LanguageModelOutput", "InputEmbeddingsFeatures", "create_attention_mask"}
tree = ast.parse(base_path.read_text())
tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected]
sys.modules[base.__name__] = base
exec(compile(tree, str(base_path), "exec"), base.__dict__)

config_module = importlib.import_module("mlx_vlm.models.phi4mm.config")
model_module = importlib.import_module("mlx_vlm.models.phi4mm.phi4mm")
audio_config = dict(attention_dim=8, attention_heads=2, num_blocks=1, linear_units=16,
                    input_size=8, time_reduction=2, conv_channels=4,
                    ext_pw_out_channel=8, depthwise_seperable_out_channel=8, t5_bias_max_distance=3)
config = dict(model_type="phi4mm", hidden_size=32, vocab_size=64, num_hidden_layers=2,
              intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
              partial_rotary_factor=0.5, mm_hidden_size=16, max_position_embeddings=128,
              original_max_position_embeddings=128, audio_token_index=60,
              min_num_patches=1, max_num_patches=16,
              vision_config=dict(hidden_size=16, intermediate_size=32, num_attention_heads=2,
                                 num_hidden_layers=2, patch_size=2, image_size=4),
              audio_processor=dict(config=audio_config),
              vision_lora=dict(r=2, lora_alpha=2), speech_lora=dict(r=2, lora_alpha=2))
c = config_module.ModelConfig.from_dict(config)
model = model_module.Model(c)
mx.random.seed(314159)
weights = {}
for index, (name, value) in enumerate(sorted(tree_flatten(model.parameters()))):
    # Seeded full-rank Gaussian matrices avoid rank-two sin(arange) weights,
    # whose nearly singular hidden states amplify backend roundoff.
    data = mx.random.normal(value.shape, dtype=mx.float32).reshape(-1) * 0.035
    if name.endswith("weight") and value.ndim == 1:
        data = data + 1
    if name.endswith("global_invstd"):
        data = data + 1
    weights[name] = data.reshape(value.shape)
model.update(tree_unflatten(list(weights.items())))
mx.eval(model.parameters())

# Raw checkpoint-shaped fused decoder weights plus separate modality adapter tensors.
checkpoint = dict(weights)
for name, value in list(weights.items()):
    if not any(f".{key}.weight" in name for key in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")):
        continue
    del checkpoint[name]
    checkpoint[name.replace(".weight", ".base_layer.weight")] = value
    for mode, phase in (("vision", 0.7), ("speech", 1.3)):
        a = mx.sin(mx.arange(2 * value.shape[1], dtype=mx.float32) * 0.23 + phase).reshape(2, value.shape[1]) * 0.06
        b = mx.cos(mx.arange(value.shape[0] * 2, dtype=mx.float32) * 0.19 + phase).reshape(value.shape[0], 2) * 0.06
        checkpoint[name.replace(".weight", f".lora_A.{mode}.weight")] = a
        checkpoint[name.replace(".weight", f".lora_B.{mode}.weight")] = b

fixture = {"weights." + name: value for name, value in checkpoint.items()}
tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
fixture["input.tokens"] = tokens
for mode, adapters in (("base", []), ("vision", ["vision"]), ("speech", ["speech"]), ("both", ["vision", "speech"])):
    updated = dict(weights)
    for name, value in list(weights.items()):
        for adapter in adapters:
            a = checkpoint.get(name.replace(".weight", f".lora_A.{adapter}.weight"))
            b = checkpoint.get(name.replace(".weight", f".lora_B.{adapter}.weight"))
            if a is not None:
                updated[name] = updated[name] + b @ a
    model.update(tree_unflatten(list(updated.items())))
    cache = [cache_module.KVCache() for _ in range(c.num_hidden_layers)]
    fixture[f"expected.{mode}.prefill"] = model.language_model(tokens, cache=cache).logits
    fixture[f"expected.{mode}.decode"] = model.language_model(mx.array([[4]], dtype=mx.int32), cache=cache).logits
# Keep modality effects substantial: disabling adapters must be easy to detect.
for mode in ("vision", "speech", "both"):
    effect = mx.max(mx.abs(fixture[f"expected.{mode}.prefill"] - fixture["expected.base.prefill"]))
    assert float(effect.item()) > 0.01, (mode, effect)
model.update(tree_unflatten(list(weights.items())))
# A projection-level independent check separates unfused adapter arithmetic
# from its mathematically equivalent checkpoint merge.
projection_input = model.language_model.model.layers[0].input_layernorm(
    model.language_model.model.embed_tokens(tokens))
name = "language_model.model.layers.0.self_attn.qkv_proj.weight"
base_weight = weights[name]
a = checkpoint[name.replace(".weight", ".lora_A.vision.weight")]
b = checkpoint[name.replace(".weight", ".lora_B.vision.weight")]
fixture["probe.projection_input"] = projection_input
fixture["probe.projection_merged"] = projection_input @ (base_weight + b @ a).T
fixture["probe.projection_unfused"] = projection_input @ base_weight.T + (projection_input @ a.T) @ b.T
# Validate the per-row unfused adapter projection on its own Metal backend too.
mx.set_default_device(mx.gpu)
fixture["probe.projection_unfused_gpu"] = projection_input @ base_weight.T + (projection_input @ a.T) @ b.T
mx.eval(fixture["probe.projection_unfused_gpu"])
mx.set_default_device(mx.cpu)
patches = (mx.arange(4 * 12, dtype=mx.float32).reshape(1, 4, 12) - 20) / 31
fixture["input.patches"] = patches
fixture["expected.vision"] = model.vision_tower(patches, spatial_shapes=mx.array([[2, 2]]))[0]
fixture["expected.projector"] = model.apply_mm_projector(fixture["expected.vision"])
audio = mx.sin(mx.arange(6 * 8, dtype=mx.float32) * 0.27).reshape(1, 6, 8)
fixture["input.audio"] = audio
fixture["expected.audio"] = model.audio_encoder(audio)[0]
fixture["expected.audio_projector"] = model.audio_projection(fixture["expected.audio"])
# Exercise NaFlex interpolation in a non-square, downsampled positional grid.
fixture["expected.vision_rectangular"] = model.vision_tower(patches[:, :2], spatial_shapes=mx.array([[1, 2]]))[0]
# Extract the reference's audio frontend without importing optional Transformers/SciPy.
processing = root / "mlx_vlm/models/phi4mm/processing_phi4mm.py"
frontend_tree = ast.parse(processing.read_text())
frontend_tree.body = [node for node in frontend_tree.body if getattr(node, "name", None) in {"speechlib_mel", "Phi4MMAudioFeatureExtractor"}]
frontend = dict(mx=mx, np=np)
exec(compile(frontend_tree, str(processing), "exec"), frontend)
samples = (np.sin(np.arange(1600, dtype=np.float32) * 0.17) * 0.1
           + np.random.default_rng(314159).normal(0, 0.001, 1600).astype(np.float32))
fixture["input.samples"] = mx.array(samples)
fixture["expected.log_mel"] = mx.array(frontend["Phi4MMAudioFeatureExtractor"]()._extract_features(samples, 16000))
mx.eval(fixture)
output = Path(__file__).with_name("phi4mm-reference.safetensors")
mx.save_safetensors(str(output), fixture, metadata={"reference_sha": REFERENCE_SHA, "format": "mlx", "purpose": "tiny deterministic parity fixture", "seed": "314159", "weights": "full-rank normal(0,0.035), unit-centered norm weights", "waveform": "sine plus normal(0,0.001), NumPy seed 314159", "network_backend": "mlx.cpu float32", "projection_gpu_backend": "mlx.gpu float32 unfused LoRA", "frontend_backend": "NumPy CPU FFT"})
output.with_suffix(".json").write_text(json.dumps(config, indent=2) + "\n")
print(output)
