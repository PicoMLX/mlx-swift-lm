"""Generate tiny numerical fixtures from mlx-vlm 4693cd0, without model downloads.

Usage: python generate_gemma3n_fixture.py /path/to/mlx-vlm /path/to/output.safetensors
Requires mlx, numpy and transformers with Gemma3n support. Reference modules are
loaded directly so this script does not depend on mlx-vlm server/training extras.
"""

import ast
import importlib.util
from pathlib import Path
import sys
import types
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np


root = Path(sys.argv[1]).resolve()
models = root / "mlx_vlm" / "models"
for name, path in (("mlx_vlm", root / "mlx_vlm"), ("mlx_vlm.models", models),
                   ("mlx_vlm.models.gemma3n", models / "gemma3n")):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module

base = types.ModuleType("mlx_vlm.models.base")
base.BaseModelConfig = type("BaseModelConfig", (), {})
base.check_array_shape = lambda _: True
sys.modules[base.__name__] = base


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference_class(path, name, namespace):
    parsed = ast.parse(path.read_text())
    definition = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == name)
    code = compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    return namespace[name]


language = types.ModuleType("mlx_vlm.models.gemma3n.language")
language.Gemma3nRMSNorm = reference_class(models / "gemma3n/language.py", "Gemma3nRMSNorm", {"nn": nn, "mx": mx})
sys.modules[language.__name__] = language
config = load("mlx_vlm.models.gemma3n.config", models / "gemma3n/config.py")
load("mlx_vlm.models.kernels", models / "kernels.py")
audio = load("mlx_vlm.models.gemma3n.audio", models / "gemma3n/audio.py")
vision = load("mlx_vlm.models.gemma3n.vision", models / "gemma3n/vision.py")

mx.random.seed(314159)
arrays = {}


def keep(name, model, inputs, output):
    for key, value in tree_flatten(model.parameters()):
        arrays[f"{name}.weights.{key}"] = value
    for key, value in inputs.items():
        arrays[f"{name}.{key}"] = value
    if isinstance(output, tuple):
        for index, value in enumerate(output):
            arrays[f"{name}.output{index}"] = value
    else:
        arrays[f"{name}.output"] = output


c = config.AudioConfig(input_feat_size=8, hidden_size=8, conf_num_attention_heads=2,
    conf_num_hidden_layers=2, conf_attention_chunk_size=4, conf_attention_context_left=3,
    conf_attention_context_right=1, conf_reduction_factor=2,
    sscp_conv_channel_size=(2, 2))
model = audio.AudioModel(c)
x = mx.random.normal((2, 19, 8)) * 0.1
mask = mx.array([[False] * 19, [False] * 13 + [True] * 6])
keep("audio", model, {"input": x, "mask": mask}, model(x, mask))
arrays["audio.subsampled"] = model.subsample_conv_projection(x)
stage = arrays["audio.subsampled"]
stage_mask = mask[:, mx.arange(stage.shape[1]) * 4]
for i, block in enumerate(model.conformer):
    stage = block(stage, stage_mask)
    arrays[f"audio.block{i}"] = stage

model = audio.Gemma3nAudioAttention(c)
x = mx.random.normal((2, 9, 8)) * 0.1
mask = mx.array([[False] * 9, [False] * 6 + [True] * 3])
keep("attention", model, {"input": x, "mask": mask}, model(x, mask))

model = vision.UniversalInvertedResidual(8, 16, dw_kernel_size_start=3,
    dw_kernel_size_mid=5, stride=2, exp_ratio=2)
x = mx.random.normal((2, 5, 7, 8)) * 0.1
keep("inverted", model, {"input": x}, model(x))

model = vision.MobileAttention(8, 8, num_heads=2, key_dim=4, value_dim=4, kv_stride=2)
x = mx.random.normal((2, 4, 6, 8)) * 0.1
keep("vision_attention", model, {"input": x}, model(x))

model = vision.MobileNetV5MultiScaleFusionAdapter([8, 8], 16, (2, 2))
first = mx.random.normal((1, 4, 4, 8)) * 0.1
second = mx.random.normal((1, 2, 2, 8)) * 0.1
keep("fusion", model, {"first": first, "second": second}, model([first, second]))

Embedder = reference_class(models / "gemma3n/gemma3n.py", "Gemma3nMultimodalEmbedder",
    {"nn": nn, "mx": mx, "Gemma3nRMSNorm": language.Gemma3nRMSNorm,
     "ModelConfig": config.ModelConfig, "TextConfig": config.TextConfig})
model = Embedder(types.SimpleNamespace(hidden_size=8, rms_norm_eps=1e-6,
    vocab_offset=32, vocab_size=8), types.SimpleNamespace(hidden_size=16))
ids = mx.array([[32, 34, 39]])
soft = mx.random.normal((1, 3, 8)) * 0.1
keep("embedder", model, {"ids": ids, "soft": soft}, model(input_ids=ids))
arrays["embedder.soft_output"] = model(inputs_embeds=soft)

from transformers.models.gemma3n.feature_extraction_gemma3n import Gemma3nAudioFeatureExtractor
extractor = Gemma3nAudioFeatureExtractor(feature_size=8)
wave = (np.sin(np.arange(1600, dtype=np.float32) * 0.04) * 0.1
        + np.random.default_rng(314159).normal(0, 0.01, 1600)).astype(np.float32)
features, _ = extractor._extract_spectrogram(wave[None, :], np.ones(wave.shape, dtype=np.int64))
arrays["features.input"] = mx.array(wave)
arrays["features.output"] = mx.array(features)

from PIL import Image
image = ((np.arange(13 * 19 * 3).reshape(13, 19, 3) * 29 + 17) % 256).astype(np.uint8)
resized = np.asarray(Image.fromarray(image).resize((8, 8), Image.Resampling.BILINEAR))
arrays["image.input"] = mx.array(image.astype(np.float32)[None] / 255)
arrays["image.output"] = mx.array(resized.astype(np.float32)[None] / 255)

mx.eval(arrays)
mx.save_safetensors(sys.argv[2], arrays, metadata={"source": "Blaizzy/mlx-vlm@4693cd0eaf022ae7ef61c6fdb55e17e6399879a7"})
print(f"Wrote {len(arrays)} arrays to {sys.argv[2]}")
