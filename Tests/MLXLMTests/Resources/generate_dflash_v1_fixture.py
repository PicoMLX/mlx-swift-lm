"""Generate dflash-v1-reference.safetensors from the z-lab Python reference.

Runs a tiny random DFlash v1 drafter (one sliding and one full-attention
layer) through three proposal rounds with the reference `dflash/model_mlx.py`
and saves every input and output the Swift test compares against.

Usage (from any directory other than the one holding the reference):

    python -I generate_dflash_v1_fixture.py \
        --reference /path/to/dflash/model_mlx.py \
        --output dflash-v1-reference.safetensors

The reference is pinned to z-lab/dflash commit
07ebd93db9f472af339b644bb70221ad8428328a. Needs Python 3.10+, mlx, mlx-lm, numpy and
safetensors.
"""

import argparse
import hashlib
import importlib.util
import json

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten
from safetensors.numpy import save_file

REFERENCE_COMMIT = "07ebd93db9f472af339b644bb70221ad8428328a"

# Mirrors MockTarget in DFlash2Tests.swift: 4 target layers, hidden 64, vocab 128.
CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "dflash_config": {
        "block_size": 4,
        "mask_token_id": 126,
        "target_layer_ids": [1, 3],
    },
    "head_dim": 16,
    "hidden_size": 64,
    "intermediate_size": 96,
    "layer_types": ["sliding_attention", "full_attention"],
    "max_position_embeddings": 1024,
    "model_type": "qwen3",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "num_target_layers": 4,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "sliding_window": 8,
    "tie_word_embeddings": True,
    "vocab_size": 128,
}

# Context rows appended per round and the anchor token of each block. Round 1
# pushes the sliding layer past its 7-row window; round 2 pushes it further.
ROUNDS = [(6, 7), (2, 42), (4, 99)]


def load_reference(path):
    spec = importlib.util.spec_from_file_location("dflash_model_mlx", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_config(ref):
    c = CONFIG
    d = c["dflash_config"]
    return ref.DFlashConfig(
        hidden_size=c["hidden_size"],
        num_hidden_layers=c["num_hidden_layers"],
        num_attention_heads=c["num_attention_heads"],
        num_key_value_heads=c["num_key_value_heads"],
        head_dim=c["head_dim"],
        intermediate_size=c["intermediate_size"],
        vocab_size=c["vocab_size"],
        rms_norm_eps=c["rms_norm_eps"],
        rope_theta=c["rope_parameters"]["rope_theta"],
        max_position_embeddings=c["max_position_embeddings"],
        block_size=d["block_size"],
        target_layer_ids=tuple(d["target_layer_ids"]),
        num_target_layers=c["num_target_layers"],
        mask_token_id=d["mask_token_id"],
        rope_scaling=c["rope_parameters"],
        layer_types=tuple(c["layer_types"]),
        sliding_window=c["sliding_window"],
    )


def randomize(draft):
    weights = []
    for name, value in tree_flatten(draft.parameters()):
        if name.endswith("norm.weight"):
            new = 1 + 0.1 * mx.random.normal(value.shape)
        else:
            new = 0.2 * mx.random.normal(value.shape)
        weights.append((name, new.astype(mx.float32)))
    draft.load_weights(weights)
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ref = load_reference(args.reference)
    reference_sha = hashlib.sha256(open(args.reference, "rb").read()).hexdigest()
    mx.random.seed(2026)

    config = build_config(ref)
    draft = ref.DFlashDraftModel(config)
    weights = randomize(draft)

    # A stand-in target: random tied embedding, no scale.
    embedding = nn.Embedding(config.vocab_size, config.hidden_size)
    embedding.weight = (0.5 * mx.random.normal(embedding.weight.shape)).astype(mx.float32)
    draft.embed_tokens = embedding
    draft.lm_head = embedding.as_linear
    draft.embed_scale = 1.0

    # Record the mask of every attention call; `None` is full visibility.
    masks = []
    original_sdpa = mx.fast.scaled_dot_product_attention

    def recording_sdpa(q, k, v, *, scale, mask=None):
        masks.append((mask, k.shape[2]))
        return original_sdpa(q, k, v, scale=scale, mask=mask)

    mx.fast.scaled_dot_product_attention = recording_sdpa

    tensors = {}
    for name, value in weights:
        tensors[f"weights.{name}"] = np.array(value)
    tensors["target.embed_tokens.weight"] = np.array(embedding.weight)

    cache = draft.make_cache()
    check_cache = draft.make_cache()
    taps = len(config.target_layer_ids) * config.hidden_size
    width = config.block_size
    for r, (rows, anchor) in enumerate(ROUNDS):
        target_hidden = mx.random.normal((1, rows, taps)).astype(mx.float32)
        block = mx.array([[anchor] + [config.mask_token_id] * (width - 1)], dtype=mx.int32)
        offsets = [c.offset for c in cache]
        assert len(set(offsets)) == 1, offsets

        masks.clear()
        h = draft.embed_tokens(block) * draft.embed_scale
        h_ctx = draft.hidden_norm(draft.fc(target_hidden))
        layer_outputs = []
        for layer, c in zip(draft.layers, cache):
            h = layer(h, h_ctx, draft.rope, c)
            layer_outputs.append(h)
        hidden = draft.norm(h[:, 1:])
        logits = draft.compute_logits(hidden)
        proposal = mx.argmax(logits, axis=-1)
        mx.eval(h_ctx, layer_outputs, logits, proposal)

        # The manual loop must agree with the reference's own entry point.
        check = draft(block, target_hidden, check_cache, logits_start=1)
        assert mx.allclose(check, logits, atol=1e-5, rtol=1e-5).item()
        assert len(masks) == 2 * len(draft.layers)

        p = f"round{r}."
        tensors[p + "block"] = np.array(block)
        tensors[p + "target_hidden"] = np.array(target_hidden)
        tensors[p + "context_offset"] = np.array([offsets[0]], dtype=np.int32)
        tensors[p + "fused_context"] = np.array(h_ctx)
        for i, out in enumerate(layer_outputs):
            tensors[p + f"layer{i}.output"] = np.array(out)
            mask, total = masks[i]
            if mask is None:
                mask = mx.ones((width, total), dtype=mx.bool_)
            tensors[p + f"layer{i}.mask"] = np.array(mask)
        tensors[p + "logits"] = np.array(logits)
        tensors[p + "proposal"] = np.array(proposal).astype(np.int32)

    metadata = {
        "config": json.dumps(CONFIG),
        "reference_commit": REFERENCE_COMMIT,
        "reference_sha256": reference_sha,
        "rounds": json.dumps(ROUNDS),
    }
    save_file(tensors, args.output, metadata=metadata)
    print(f"wrote {len(tensors)} tensors to {args.output}")


if __name__ == "__main__":
    main()
