"""Numerical fixtures from mlx-vlm 4693cd0, no checkpoint downloads.

Usage: python generate_qwen3_omni_fixture.py /path/to/reference-root output.safetensors
Needs mlx, mlx-lm, numpy, transformers and the ordinary mlx-vlm dependencies.
Reference package paths select the pinned local sources while permitting shared
utilities from an installed mlx-vlm package. No speech output modules are loaded.
"""

import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

root = Path(sys.argv[1]).resolve()
installed = Path(importlib.util.find_spec("mlx_vlm").origin).parent
for name, paths in (
    ("mlx_vlm", [root / "mlx_vlm", installed]),
    ("mlx_vlm.models", [root / "mlx_vlm/models", installed / "models"]),
    ("mlx_vlm.models.qwen3_omni_moe", [root / "mlx_vlm/models/qwen3_omni_moe"]),
):
    module = types.ModuleType(name)
    module.__path__ = [str(p) for p in paths]
    sys.modules[name] = module

c = importlib.import_module("mlx_vlm.models.qwen3_omni_moe.config")
a = importlib.import_module("mlx_vlm.models.qwen3_omni_moe.audio")
v = importlib.import_module("mlx_vlm.models.qwen3_omni_moe.vision")
l = importlib.import_module("mlx_vlm.models.qwen3_omni_moe.language")
# Use the precise CPU backend for cross-language golden equations; GPU GEMM
# shape-dependent roundoff is covered separately by actual distributed parity.
mx.set_default_device(mx.cpu)
importlib.import_module("mlx_vlm.models.rope_utils")._HAS_METAL = False
mx.random.seed(271828)
arrays = {}


def keep_weights(prefix, model):
    for key, value in tree_flatten(model.parameters()):
        arrays[f"{prefix}.weights.{key}"] = value


text = dict(num_hidden_layers=2, hidden_size=64, intermediate_size=128,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
    mlp_only_layers=[], moe_intermediate_size=64, rms_norm_eps=1e-6,
    vocab_size=128, rope_theta=1e6, max_position_embeddings=2048,
    rope_scaling={"type": "default", "mrope_section": [3, 3, 2]},
    tie_word_embeddings=False)
vision = dict(model_type="qwen3_omni_moe_vision_encoder", depth=2, hidden_size=32, intermediate_size=64, out_hidden_size=64,
    num_heads=4, patch_size=2, spatial_merge_size=2, temporal_patch_size=2,
    num_position_embeddings=16, deepstack_visual_indexes=[0, 1])
audio = dict(d_model=16, encoder_layers=2, encoder_attention_heads=2,
    encoder_ffn_dim=32, num_mel_bins=8, output_dim=64, downsample_hidden_size=4,
    n_window=50, n_window_infer=800, max_source_positions=128)
thinker = dict(text_config=text, vision_config=vision, audio_config=audio,
    image_token_id=60, video_token_id=61, audio_token_id=62,
    vision_start_token_id=56, vision_end_token_id=57,
    audio_start_token_id=58, audio_end_token_id=59, position_id_per_seconds=13)
configuration = {"model_type": "qwen3_omni_moe", "thinker_config": thinker}
Path(sys.argv[2]).with_suffix(".json").write_text(json.dumps(configuration, indent=2))

audio_model = a.AudioModel(c.AudioConfig(**audio))
audio_input = mx.random.normal((8, 238)) * 0.1
arrays["audio.input"] = audio_input
arrays["audio.lengths"] = mx.array([101, 137], mx.int32)
arrays["audio.output"] = audio_model(audio_input, feature_lens=arrays["audio.lengths"])
keep_weights("audio", audio_model)

vision_model = v.VisionModel(c.VisionConfig(**vision))
# The pinned Omni port ignores hidden_act and hardcodes exact GELU here.
# Match the declared published checkpoint config and authoritative Transformers.
for block in vision_model.blocks:
    block.mlp.act_fn = nn.GELU(approx="tanh")
pixels = mx.random.normal((48, 24)) * 0.1
grids = mx.array([[1, 4, 4], [2, 4, 4]], mx.int32)
arrays["vision.input"] = pixels
arrays["vision.grids"] = grids
features, residuals = vision_model(pixels, grids)
arrays["vision.output"] = features
for index, residual in enumerate(residuals):
    arrays[f"vision.deepstack{index}"] = residual
keep_weights("vision", vision_model)

language = l.LanguageModel(c.TextConfig(**text))
tokens = mx.array([[3, 4, 5, 6, 7], [9, 10, 11, 12, 13]], mx.int32)
positions = mx.array([[[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]],
    [[0, 1, 0, 1, 2], [2, 3, 1, 2, 3]],
    [[0, 1, 1, 0, 2], [2, 3, 2, 1, 3]]], mx.float32)
deepstack = mx.random.normal((2, 5, 2, 64)) * 0.01
arrays["language.tokens"] = tokens
arrays["language.positions"] = positions
arrays["language.deepstack"] = deepstack
hidden = language.model(tokens, position_ids=positions, deepstack_visual_embeds=deepstack)
arrays["language.output"] = language.lm_head(hidden)
keep_weights("language", language)

# Full image + two-frame video + 101-frame audio prompt. Explicit positions are
# the HF Omni rules: text/boundaries scalar, visual t in seconds * 13, h/w grid.
ids = [3, 56] + [60] * 4 + [57, 4, 56] + [61] * 8 + [57, 5, 58] + [62] * 14 + [59, 6]
position_list = []
next_position = 0


def scalar(n):
    global next_position
    for _ in range(n):
        position_list.append([next_position] * 3)
        next_position += 1


def visual_positions(frames):
    global next_position
    start = next_position
    for t in range(frames):
        for h in range(2):
            for w in range(2):
                position_list.append([start + t * 13, start + h, start + w])
    next_position = max(max(x) for x in position_list) + 1


scalar(2); visual_positions(1); scalar(3); visual_positions(2); scalar(3 + 14 + 2)
ids = mx.array([ids], mx.int32)
positions = mx.array(np.array(position_list, dtype=np.float32).T[:, None, :])
embeddings = language.model.embed_tokens(ids)
residual_tensor = mx.zeros((1, ids.shape[1], 2, 64))
for token, feature in ((60, features[:4]), (61, features[4:]),
    (62, audio_model(audio_input[:, :101], feature_lens=mx.array([101])))):
    indices = mx.array(np.flatnonzero(np.array(ids).reshape(-1) == token), mx.uint32)
    flat = embeddings.reshape(-1, 64)
    flat[indices] = feature
    embeddings = flat.reshape(embeddings.shape)
    if token != 62:
        for index, residual in enumerate(residuals):
            residual_tensor[0, indices, index] = residual[:4] if token == 60 else residual[4:]
hidden = language.model(ids, inputs_embeds=embeddings, position_ids=positions,
    deepstack_visual_embeds=residual_tensor)
arrays["multimodal.tokens"] = ids
arrays["multimodal.positions"] = positions
arrays["multimodal.output"] = language.lm_head(hidden)

from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
wave = (np.sin(np.arange(3200, dtype=np.float32) * 0.07) * 0.1
        + np.random.default_rng(314159).normal(0, 0.01, 3200)).astype(np.float32)
extractor = WhisperFeatureExtractor(feature_size=8, sampling_rate=16000,
    n_fft=400, hop_length=160, return_attention_mask=True)
arrays["whisper.samples"] = mx.array(wave)
arrays["whisper.output"] = mx.array(extractor(wave, sampling_rate=16000,
    padding="longest", truncation=False, return_tensors="np")["input_features"][0])
mx.eval(*arrays.values())
mx.save_safetensors(sys.argv[2], arrays, metadata={
    "mlx-vlm-commit": "4693cd0eaf022ae7ef61c6fdb55e17e6399879a7",
    "purpose": "Qwen3 Omni Thinker audio, vision, interleaved MRoPE, MoE/deepstack and full multimodal logits",
    "network_backend": "mlx.cpu float32 eager MRoPE", "seed": "271828",
    "reference_correction": "vision block MLP follows declared gelu_pytorch_tanh, as Transformers; pinned MLX hardcodes exact GELU",
})
