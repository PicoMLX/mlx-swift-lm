#!/usr/bin/env python3
"""Generate small deterministic MiniCPM fixtures from the pinned Python network code.

Usage: python generate_minicpm_reference.py /path/to/mlx-vlm
Requires MLX, NumPy, Pillow and Transformers. No checkpoints are downloaded.
"""
import ast
import dataclasses
import importlib
import inspect
import json
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
from PIL import Image

REFERENCE_SHA = "4693cd0eaf022ae7ef61c6fdb55e17e6399879a7"
root = Path(sys.argv[1]).resolve()
# Keep the installed package's auxiliary speculative helpers, while selecting the
# downloaded, pinned network files. No network class or numerical method is patched.
import mlx_vlm
import mlx_vlm.models
mlx_vlm.__path__.insert(0, str(root / "mlx_vlm"))
mlx_vlm.models.__path__.insert(0, str(root / "mlx_vlm/models"))
for package in ("minicpmo", "minicpmv4_6", "minicpmv4_7", "qwen3_vl", "qwen3_5", "qwen3_5_moe"):
    name = "mlx_vlm.models." + package
    module = types.ModuleType(name)
    module.__path__ = [str(root / "mlx_vlm/models" / package)]
    sys.modules[name] = module

# Load native class definitions without package __init__ side effects that
# initialize optional TTS/vocoder output paths (outside this port's input scope).
def selected(path, names, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace

vision_namespace = selected(root / "mlx_vlm/models/minicpmo/vision.py",
    {"SiglipAttention", "SiglipMLP", "SiglipEncoderLayer", "SiglipEncoder", "SiglipVisionEmbeddings", "VisionModel"},
    dict(mx=mx, nn=nn, np=np, Optional=Optional, VisionConfig=Any))
merger_namespace = selected(root / "mlx_vlm/models/minicpmv4_6/minicpmv4_6.py",
    {"CrossAttention", "VitMerger", "MergerBlock", "Merger"},dict(mx=mx, nn=nn, Optional=Optional))
resampler_namespace = selected(root / "mlx_vlm/models/minicpmo/minicpmo.py",
    {"CrossAttention", "Resampler", "get_2d_sincos_pos_embed", "get_2d_sincos_pos_embed_from_grid", "get_1d_sincos_pos_embed_from_grid"},
    dict(mx=mx, nn=nn, np=np, Optional=Optional))
audio_namespace = selected(root / "mlx_vlm/models/minicpmo/audio.py",
    {"AudioAttention", "AudioEncoderLayer", "AudioProjector", "AudioModel"},
    dict(mx=mx, nn=nn, Optional=Optional, AudioConfig=Any))
canvas_namespace = selected(root / "mlx_vlm/models/minicpmv4_7/rope.py", {"CanvasRoPE"},
    dict(np=np, itertools=importlib.import_module("itertools")))

text_base = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6,
    vocab_size=64, head_dim=8, tie_word_embeddings=False, rope_theta=10000000,
    max_position_embeddings=256, attention_bias=False, hidden_act="silu")
hybrid = dict(model_type="qwen3_5_text", linear_num_value_heads=4, linear_num_key_heads=2,
    linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
    full_attention_interval=2, rope_parameters=dict(type="default", rope_theta=10000000,
    partial_rotary_factor=0.5, mrope_section=[1,1,0]))
vision_config = dict(model_type="siglip_vision_model", hidden_size=16, intermediate_size=32,
    num_hidden_layers=2, num_attention_heads=2, num_channels=3, image_size=8,
    patch_size=2, layer_norm_eps=1e-6, hidden_act="gelu_pytorch_tanh", window_kernel_size=[2,2])
audio_config = dict(model_type="whisper", d_model=8, encoder_attention_heads=2,
    encoder_layers=1, encoder_ffn_dim=32, num_mel_bins=8, max_source_positions=32, layer_norm_eps=1e-5)

def stamp(model):
    weights = {}
    for index,(name,tensor) in enumerate(sorted(tree_flatten(model.parameters()))):
        data = mx.sin(mx.arange(tensor.size,dtype=mx.float32)*0.17 + index*0.31)*0.035
        if tensor.ndim == 1 and name.endswith("weight"):
            data = data+1
        if name.endswith("A_log"):
            data = data+0.2
        weights[name] = data.reshape(tensor.shape)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return weights

output_dir = Path(__file__).parent
for variant in ("minicpmo", "minicpmv4_6", "minicpmv4_7"):
    omni = variant == "minicpmo"
    text = dict(text_base)
    if omni:
        text.update(model_type="qwen3_vl_text", num_hidden_layers=2,
                    rope_scaling=dict(type="default", mrope_section=[1,1,2]))
    else:
        text.update(hybrid)
    config = dict(model_type=variant, text_config=text, vision_config=dict(vision_config),
        query_num=4, audio_pool_step=2, insert_layer_id=0, downsample_mode="16x",
        merge_kernel_size=[2,2], merger_times=1, image_token_id=60, video_token_id=61,
        image_start_id=50, image_end_id=51, slice_start_id=52, slice_end_id=53, newline_id=54)
    if omni:
        config.update(audio_config=audio_config,init_audio=True,init_tts=False)
        cm = importlib.import_module("mlx_vlm.models.minicpmo.config")
        c = cm.ModelConfig.from_dict(config)
        language_cls = importlib.import_module("mlx_vlm.models.qwen3_vl.language").LanguageModel
    else:
        cm = importlib.import_module("mlx_vlm.models."+variant+".config")
        c = cm.ModelConfig.from_dict(config)
        language_cls = importlib.import_module("mlx_vlm.models."+variant+".language").LanguageModel
    model = nn.Module()
    model.language_model = language_cls(c.text_config,c)
    # Use each variant's unchanged SigLIP class (V selects tanh from hidden_act).
    if omni:
        model.vision_tower = vision_namespace["VisionModel"](c.vision_config)
        model.resampler = resampler_namespace["Resampler"](4,32,1,kv_dim=16)
        model.audio_tower = audio_namespace["AudioModel"](c.audio_config)
        model.audio_projection_layer = audio_namespace["AudioProjector"](8,32)
    else:
        vc = importlib.import_module("mlx_vlm.models.minicpmv4_6.vision")
        model.vision_tower = vc.VisionModel(c.vision_config)
        model.vit_merger = merger_namespace["VitMerger"](16,128,2,(2,2))
        model.merger = merger_namespace["Merger"](16,32,1,(2,2))
        if variant == "minicpmv4_7": model.vit_merger.act=nn.GELU(approx="tanh")
    weights = stamp(model)
    fixture={"weights."+name:value for name,value in weights.items()}
    patches=(mx.arange(16*12,dtype=mx.float32).reshape(16,12)-70)/101
    fixture["input.patches"] = patches
    # Reconstruct ordinary HWC pixels from the transport's row-major patches.
    pixels=patches.reshape(4,4,2,2,3).transpose(0,2,1,3,4).reshape(1,8,8,3)
    sizes=mx.array([[4,4]],dtype=mx.int32)
    hidden=model.vision_tower.embeddings(pixels,tgt_sizes=sizes)
    gh=gw=4
    for index,layer in enumerate(model.vision_tower.encoder.layers):
        hidden=layer(hidden)
        if not omni and index == 0:
            merged,gh,gw=model.vit_merger(hidden[0],gh,gw)
            hidden=merged[None]
    vision=model.vision_tower.post_layernorm(hidden)
    fixture["expected.vision"] = vision[0]
    if omni: projected=model.resampler(vision,mx.array([[gh,gw]],dtype=mx.int32))[0]
    else: projected=model.merger(vision[0],gh,gw)[0]
    fixture["expected.visual_projector"] = projected
    if not omni:
        unmerged=model.vision_tower.embeddings(pixels,tgt_sizes=sizes)
        for layer in model.vision_tower.encoder.layers:
            unmerged=layer(unmerged)
        projected4=model.merger(model.vision_tower.post_layernorm(unmerged)[0],4,4)[0]
        fixture["expected.visual_projector_4x"]=projected4
        ids4=[1,50]+[60]*int(projected4.shape[0])+[51,2]
        tokens4=mx.array([ids4],dtype=mx.int32)
        embedding4=model.language_model.model.embed_tokens(tokens4)
        embedding4[0,2:2+projected4.shape[0]]=projected4
        positions4=np.broadcast_to(np.arange(len(ids4)),(3,1,len(ids4))).copy()
        if variant == "minicpmv4_7":
            positions4,_=canvas_namespace["CanvasRoPE"](c).get_rope_index(np.array([ids4]),np.where(np.array([ids4])==60,1,0),np.array([[4,4]]),np.empty((0,2)),downsample_mode="4x")
        fixture["input.media_tokens_4x"]=tokens4
        fixture["expected.media_embeddings_4x"]=embedding4
        fixture["expected.canvas_4x"]=mx.array(positions4,dtype=mx.int32)
        cache4=[importlib.import_module("mlx_vlm.models.cache").KVCache() if (i+1)%2==0 else importlib.import_module("mlx_vlm.models.cache").ArraysCache(size=2) for i in range(text["num_hidden_layers"])]
        fixture["expected.media_prefill_4x"]=model.language_model(tokens4,inputs_embeds=embedding4,position_ids=mx.array(positions4),cache=cache4).logits
        fixture["expected.media_decode_4x"]=model.language_model(mx.array([[4]],dtype=mx.int32),position_ids=mx.full((3,1,1),int(positions4.max())+1,dtype=mx.int32),cache=cache4).logits
    embeddings=model.language_model.model.embed_tokens(mx.array([[1,2,3]],dtype=mx.int32))
    tokens=mx.array([[1,2,3]],dtype=mx.int32)
    fixture["input.tokens"]=tokens
    cache_module=importlib.import_module("mlx_vlm.models.cache")
    cache=[cache_module.KVCache() if omni or (i+1)%2==0 else cache_module.ArraysCache(size=2) for i in range(text["num_hidden_layers"])]
    fixture["expected.text_prefill"]=model.language_model(tokens,position_ids=mx.broadcast_to(mx.arange(3)[None,None,:],(3,1,3)),cache=cache).logits
    fixture["expected.text_decode"]=model.language_model(mx.array([[4]],dtype=mx.int32),position_ids=mx.full((3,1,1),3,dtype=mx.int32),cache=cache).logits
    media_ids=[1,50]+[60]*int(projected.shape[0])+[51,2]
    media_tokens=mx.array([media_ids],dtype=mx.int32)
    media_embeddings=model.language_model.model.embed_tokens(media_tokens)
    media_embeddings[0,2:2+projected.shape[0]]=projected
    fixture["input.media_tokens"]=media_tokens
    fixture["expected.media_embeddings"]=media_embeddings
    positions=np.broadcast_to(np.arange(len(media_ids)),(3,1,len(media_ids))).copy()
    if variant == "minicpmv4_7":
        modalities=np.where(np.asarray(media_tokens)==60,1,0)
        positions,_=canvas_namespace["CanvasRoPE"](c).get_rope_index(np.asarray(media_tokens),modalities,np.array([[4,4]]),np.empty((0,2)),downsample_mode="16x")
        fixture["expected.canvas"]=mx.array(positions,dtype=mx.int32)
    cache=[cache_module.KVCache() if omni or (i+1)%2==0 else cache_module.ArraysCache(size=2) for i in range(text["num_hidden_layers"])]
    fixture["expected.media_prefill"]=model.language_model(media_tokens,inputs_embeds=media_embeddings,position_ids=mx.array(positions),cache=cache).logits
    next_position=int(positions.max())+1
    fixture["expected.media_decode"]=model.language_model(mx.array([[4]],dtype=mx.int32),position_ids=mx.full((3,1,1),next_position,dtype=mx.int32),cache=cache).logits
    if omni:
        audio=mx.sin(mx.arange(12*8,dtype=mx.float32)*0.27).reshape(1,8,12)
        fixture["input.audio"]=audio.transpose(0,2,1)
        states=model.audio_tower(audio,feature_lengths=mx.array([12]))
        fixture["expected.audio"]=states
        projected_audio=model.audio_projection_layer(states)
        fixture["expected.audio_projector"]=mx.mean(projected_audio.reshape(1,3,2,32),axis=2)[0]
        from transformers import WhisperFeatureExtractor
        samples=np.sin(np.arange(1600,dtype=np.float32)*0.17)*0.1
        extractor=WhisperFeatureExtractor(feature_size=8,sampling_rate=16000,hop_length=160,n_fft=400)
        fixture["input.samples"]=mx.array(samples)
        fixture["expected.log_mel"]=mx.array(extractor._np_extract_fbank_features(samples[None],device="cpu")[0].T)
        resize_input=np.mod(np.arange(7*9*3).reshape(7,9,3)*13,256).astype(np.uint8)
        fixture["input.resize"]=mx.array(resize_input.astype(np.float32))
        fixture["expected.resize"]=mx.array(np.asarray(Image.fromarray(resize_input).resize((12,8),Image.Resampling.BICUBIC)).astype(np.float32))
    if variant == "minicpmv4_7":
        # Canvas fixture with a thumbnail, a 2x2 detail layout and adjacent video frame.
        ids=[7,50,60,51,52,60,53,52,60,53,54,52,60,53,52,60,53,50,61,51,8]
        grids=np.array([[4,4]]*5,dtype=np.int32)
        canvas_ids=np.array([ids],dtype=np.int32)
        modalities=np.where(canvas_ids==60,1,np.where(canvas_ids==61,2,0))
        canvas,deltas=canvas_namespace["CanvasRoPE"](c).get_rope_index(canvas_ids,modalities,grids,np.array([[4,4]]),downsample_mode="16x")
        fixture["input.canvas_tokens"]=mx.array(canvas_ids)
        fixture["expected.canvas_layout"]=mx.array(canvas)
        fixture["expected.canvas_deltas"]=mx.array(deltas)
    mx.eval(fixture)
    output=output_dir/(variant+"-reference.safetensors")
    mx.save_safetensors(str(output),fixture,metadata={"reference_sha":REFERENCE_SHA,"format":"mlx","purpose":"tiny deterministic MiniCPM numerical parity"})
    output.with_suffix(".json").write_text(json.dumps(config,indent=2)+"\n")
    print(output,flush=True)
