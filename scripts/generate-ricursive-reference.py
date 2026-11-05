#!/usr/bin/env python3
"""Regenerate the tiny Ripple fixture with the official PyTorch implementation.

Requires torch==2.9.1, transformers==4.57.6, numpy==2.3.5 and safetensors==0.8.0.
Run from any directory; --source-dir may point to previously downloaded source.
Only small source/configuration files are downloaded, never the released model weights.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import urllib.request

import numpy as np
import safetensors
from safetensors.torch import save_file
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers


REVISION = "e27e171792f303d74c048184e8259ba663cfa0d5"
BASE_URL = f"https://huggingface.co/ricursiveAI/Ripple-65B-Preview/resolve/{REVISION}"
SOURCE_HASHES = {
    "config.json": "9a5654b256ead01855f0d74cf7a70fa65294267e6b7b754b5e5737457a45e14a",
    "configuration_ricursive.py": "002cb3915be9685eb962d617623bec14d45b5d6b5d46e689fac98a3b38cb6cf6",
    "modeling_ricursive.py": "f83865ef643e273a6c9aecf1575e85c95ad01fd995b1e43c3c7a892765006bb3",
}
OUTPUT = Path(__file__).resolve().parents[1] / "Tests/MLXLMTests/Resources/Ricursive"
CONFIG = {
    "model_type": "ricursive",
    "vocab_size": 16,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 5,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 128,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "sliding_window": 4,
    "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
    "use_gated_attention": True,
    "gate_full_precision": True,
    "use_valres": True,
    "ffn_clamp_threshold": 0.6,
    "embed_scale": 2.5,
    "tie_word_embeddings": False,
    "use_cache": False,
}
TOKENS = [1, 3, 5, 2, 9, 4, 8, 7, 6, 12, 10, 11, 14]
VARIANTS = {
    "reference": {},
    "ungated": {"use_gated_attention": False},
    "without_value_residual": {"use_valres": False},
    "unclipped": {"ffn_clamp_threshold": None},
    "tied_embeddings": {"tie_word_embeddings": True},
}


def load_reference(directory):
    package = types.ModuleType("ripple_reference")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    configuration = importlib.import_module("ripple_reference.configuration_ricursive")
    modeling = importlib.import_module("ripple_reference.modeling_ricursive")
    return configuration.RicursiveConfig, modeling.RicursiveForCausalLM


def deterministic_weights(model):
    weights = {}
    for name, parameter in model.state_dict().items():
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        generator = np.random.default_rng(seed)
        values = generator.integers(-64, 65, size=parameter.numel()).astype(np.float32)
        values = values / 256
        if "norm.weight" in name:
            values = 1 + values / 2
        elif name.endswith("value_res_lambda"):
            layer = int(name.split(".")[2])
            values[:] = (layer - 2) * 0.75
        weights[name] = torch.from_numpy(values.reshape(tuple(parameter.shape)))
    return weights


def generate(source_directory):
    for name, expected_hash in SOURCE_HASHES.items():
        actual_hash = hashlib.sha256((source_directory / name).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"{name} does not match pinned revision {REVISION}")
    config_type, model_type = load_reference(source_directory)
    torch.set_num_threads(1)
    model = model_type(config_type(**CONFIG)).eval()
    weights = deterministic_weights(model)
    expected = {}
    input_ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.inference_mode():
        for name, overrides in VARIANTS.items():
            candidate = model_type(config_type(**(CONFIG | overrides))).eval()
            candidate_weights = {
                key: value for key, value in weights.items() if key in candidate.state_dict()
            }
            if overrides.get("tie_word_embeddings"):
                candidate_weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
            candidate.load_state_dict(candidate_weights)
            expected[name] = candidate(input_ids, use_cache=False).logits.contiguous()
        model.load_state_dict(weights)
        model.to(torch.bfloat16)
        expected["bfloat16"] = model(input_ids, use_cache=False).logits.float().contiguous()
        with sdpa_kernel(SDPBackend.MATH):
            math_logits = model(input_ids, use_cache=False).logits.float()
        backend_difference = (math_logits - expected["bfloat16"]).abs()

    assert all(torch.isfinite(logits).all() for logits in expected.values())
    for name in VARIANTS.keys() - {"reference"}:
        assert torch.max(torch.abs(expected[name] - expected["reference"])).item() > 1e-3

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ripple-65b-config.json").write_bytes((source_directory / "config.json").read_bytes())
    (OUTPUT / "ricursive-config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    save_file(weights, OUTPUT / "ricursive-weights.safetensors")
    save_file(expected, OUTPUT / "ricursive-logits.safetensors")
    provenance = {
        "source_repository": "ricursiveAI/Ripple-65B-Preview",
        "source_revision": REVISION,
        "source_license": "Apache-2.0",
        "source_sha256": {
            name: hashlib.sha256((source_directory / name).read_bytes()).hexdigest()
            for name in SOURCE_HASHES
        },
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "safetensors": safetensors.__version__,
        "device": "cpu",
        "tokens": TOKENS,
        "variants": VARIANTS,
        "bfloat16_cpu_default_vs_math_sdpa": {
            "max_absolute_error": backend_difference.max().item(),
            "mean_absolute_error": backend_difference.mean().item(),
        },
        "notes": "Full uncached official forward; cached/chunked Swift outputs must match it.",
    }
    (OUTPUT / "ricursive-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Generated {len(weights)} tensors and {len(expected)} reference outputs in {OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    if args.source_dir:
        generate(args.source_dir)
        return
    with tempfile.TemporaryDirectory(prefix="ricursive-reference-") as temporary:
        directory = Path(temporary)
        for name in SOURCE_HASHES:
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", directory / name)
        generate(directory)


if __name__ == "__main__":
    main()
