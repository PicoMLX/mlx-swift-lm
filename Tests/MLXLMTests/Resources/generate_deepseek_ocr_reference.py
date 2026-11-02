"""Generate numerical fixtures from pinned, unmodified upstream model classes.

Usage: python generate_deepseek_ocr_reference.py /path/to/mlx-vlm
Requires mlx and numpy. The package shims avoid importing optional server/training
dependencies; model equations come from the upstream source files themselves.
Source: Blaizzy/mlx-vlm @ 4693cd0eaf022ae7ef61c6fdb55e17e6399879a7 (MIT).
"""

import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
from dataclasses import dataclass
import inspect
import ast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
from PIL import Image, ImageOps

source = Path(sys.argv[1]).resolve()
mx.set_default_device(mx.cpu)


def package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


package("mlx_vlm", source / "mlx_vlm")
package("mlx_vlm.models", source / "mlx_vlm/models")
for name in ("deepseekocr", "deepseekocr_2"):
    package("mlx_vlm.models." + name, source / "mlx_vlm/models" / name)


class BaseModelConfig:
    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})


@dataclass
class LanguageModelOutput:
    logits: object


@dataclass
class InputEmbeddingsFeatures:
    inputs_embeds: object


def mask(h, cache):
    length = h.shape[1]
    if length == 1:
        return None
    offset = cache.offset if cache else 0
    return mx.arange(offset + length)[None, :] <= (mx.arange(length)[:, None] + offset)


base = types.ModuleType("mlx_vlm.models.base")
base.BaseModelConfig = BaseModelConfig
base.LanguageModelOutput = LanguageModelOutput
base.InputEmbeddingsFeatures = InputEmbeddingsFeatures
base.create_attention_mask = mask
base.scaled_dot_product_attention = lambda q, k, v, cache, scale, mask: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
base.check_array_shape = lambda x: x.ndim == 4 and x.shape[1] == x.shape[2]
sys.modules[base.__name__] = base
linear = types.ModuleType("mlx_vlm.models.linear")
linear.DECODE_BLOCK_SIZE = 0
sys.modules[linear.__name__] = linear
kernels = importlib.import_module("mlx_vlm.models.kernels")
kernels._HAS_METAL = False

config1 = importlib.import_module("mlx_vlm.models.deepseekocr.config")
config2 = importlib.import_module("mlx_vlm.models.deepseekocr_2.config")
original1 = importlib.import_module("mlx_vlm.models.deepseekocr.deepseekocr")
original2 = importlib.import_module("mlx_vlm.models.deepseekocr_2.deepseekocr_2")
sam_module = importlib.import_module("mlx_vlm.models.deepseekocr.sam")


def sam_factory(**kwargs):
    # Shrink channel counts and spatial grids while keeping the upstream SAM equations.
    model = sam_module.SAMEncoder(img_size=16, patch_size=2, embed_dim=8,
        depth=1, num_heads=2, out_chans=8, window_size=4,
        global_attn_indexes=(), final_out_chans=8)
    model.net_2 = nn.Conv2d(8, 8, 3, stride=2, padding=1, bias=False)
    model.net_3 = nn.Conv2d(8, 8, 3, stride=2, padding=1, bias=False)
    return model


original1.SAMEncoder = sam_factory
original2.SAMEncoder = sam_factory


class Cache:
    def __init__(self):
        self.offset = 0
        self.keys = None
        self.values = None

    def update_and_fetch(self, keys, values):
        if self.keys is not None:
            keys = mx.concatenate([self.keys, keys], axis=2)
            values = mx.concatenate([self.values, values], axis=2)
        self.keys, self.values = keys, values
        self.offset = keys.shape[2]
        return keys, values


def deterministic(model):
    weights = {}
    for name, old in tree_flatten(model.parameters()):
        phase = int.from_bytes(hashlib.sha256(name.encode()).digest()[:2], "little") / 100
        values = 0.05 * np.sin(np.arange(old.size, dtype=np.float32) * 0.37 + phase)
        if old.ndim == 1 and name.endswith("weight"):
            values += 1
        weights[name] = mx.array(values.reshape(old.shape))
    model.load_weights(list(weights.items()), strict=True)
    return weights


arrays, metadata = {}, {"source": "Blaizzy/mlx-vlm@4693cd0eaf022ae7ef61c6fdb55e17e6399879a7"}
for version in (1, 2):
    prefix = f"ocr{version}"
    text = dict(hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        moe_intermediate_size=64, n_routed_experts=4, n_shared_experts=1,
        num_experts_per_tok=2, max_position_embeddings=2048)
    vision = dict(model_type="vision", hidden_size=8, layers=1,
        intermediate_size=16, num_attention_heads=2, patch_size=14)
    if version == 2:
        vision["params"] = {"qwen2": dict(dim=8, layers=1, heads=2, kv_heads=1,
            intermediate_size=16, rms_norm_eps=1e-6, rope_theta=1000000)}
    projector = dict(projector_type="linear", input_dim=16 if version == 1 else 8, n_embed=64)
    module = config1 if version == 1 else config2
    config = module.ModelConfig(text_config=module.TextConfig(**text),
        vision_config=module.VisionConfig(**vision),
        projector_config=module.ProjectorConfig(**projector),
        model_type="deepseekocr" if version == 1 else "deepseekocr_2", image_token_index=3)
    model = (original1 if version == 1 else original2).Model(config)
    weights = deterministic(model)
    arrays.update({prefix + ".weights." + k: v for k, v in weights.items()})
    swift_config = dict(model_type=config.model_type, image_token_index=3,
        text_config=text, vision_config=vision, projector_config=projector,
        sam_config=dict(image_size=16, patch_size=2, width=8, layers=1, heads=2,
            window_size=4, global_attn_indexes=[], neck_channels=8, downsample_channels=8))
    metadata[prefix + ".configuration"] = json.dumps(swift_config, sort_keys=True)
    for variant in ("global", "cropped"):
        key = prefix + "." + variant + "."
        image_size = 16 if version == 1 else 128
        pixels = mx.array(np.linspace(0.01, 0.9, 3 * image_size**2, dtype=np.float32).reshape(1, 3, image_size, image_size))
        crops = None
        crop_count = 0
        if variant == "cropped":
            crop_count = 2 if version == 1 else 1
            crop_size = 16 if version == 1 else 96
            crops = mx.array(np.linspace(0.1, 0.8, crop_count * 3 * crop_size**2, dtype=np.float32).reshape(crop_count, 3, crop_size, crop_size))
        dummy = mx.zeros((1, 3, image_size, image_size))
        if version == 1:
            feature_count = 7 + (10 if crop_count else 0)
            grid = [2, 1] if crop_count else [1, 1]
        else:
            feature_count = 257 + 144 * crop_count
            grid = [1, 1] if crop_count else [0, 0]
        tokens = mx.array([[1] + [3] * feature_count + [2, 4]], dtype=mx.int32)
        image_mask = tokens == 3
        spatial = mx.array([grid], dtype=mx.int32)
        pixels_arg = [crops if crops is not None else dummy, pixels]
        embeddings = model.get_input_embeddings(tokens, pixels_arg, spatial, image_mask).inputs_embeds
        caches = [Cache() for _ in range(2)]
        logits = model.language_model(tokens, inputs_embeds=embeddings, cache=caches).logits
        decode = model.language_model(mx.array([[5]], dtype=mx.int32), cache=caches).logits
        arrays.update({key + "tokens": tokens, key + "pixels": pixels,
            key + "grid": mx.array([0, 0] if crop_count == 0 else [1, 2] if version == 1 else grid),
            key + "embeddings": embeddings, key + "logits": logits, key + "decode": decode})
        if crops is not None:
            arrays[key + "crops"] = crops
        mx.eval(embeddings, logits, decode)

# Exercise the source crop selector and Pillow's real resize/padding independently.
processor_source = (source / "mlx_vlm/models/deepseekocr/processing_deepseekocr.py").read_text()
nodes = ast.parse(processor_source).body
crop_nodes = [node for node in nodes if isinstance(node, ast.FunctionDef)
    and node.name in {"find_closest_aspect_ratio", "dynamic_preprocess"}]
namespace = {}
exec(compile(ast.Module(body=crop_nodes, type_ignores=[]), "upstream-crops", "exec"), namespace)
rgb = (np.arange(12 * 20 * 3).reshape(12, 20, 3) % 256).astype(np.uint8)
image = Image.fromarray(rgb)
patches, grid = namespace["dynamic_preprocess"](image, min_num=2, max_num=4, image_size=8)
global_image = ImageOps.pad(image, (16, 16), color=(127, 127, 127))
arrays["processor.rgb"] = mx.array(rgb)
arrays["processor.grid"] = mx.array([grid[1], grid[0]], dtype=mx.int32)
arrays["processor.global"] = mx.array(np.asarray(global_image).transpose(2, 0, 1)[None]).astype(mx.float32) / 255
arrays["processor.crops"] = mx.array(np.stack([np.asarray(patch).transpose(2, 0, 1) for patch in patches])).astype(mx.float32) / 255

destination = Path(__file__).with_name("deepseek-ocr-reference.safetensors")
mx.save_safetensors(str(destination), arrays, metadata)
print(f"Wrote {destination}: {len(arrays)} tensors")
