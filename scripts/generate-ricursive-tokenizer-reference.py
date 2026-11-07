#!/usr/bin/env python3
"""Regenerate Ripple token/template fixtures with transformers==4.57.6.

Usage: python scripts/generate-ricursive-tokenizer-reference.py /path/to/checkpoint
The source directory needs tokenizer.json, tokenizer_config.json, and chat_template.jinja
from ricursiveAI/Ripple-65B-Preview at REVISION. No model weights are needed.
"""

import copy
import hashlib
import json
from pathlib import Path
import sys

import tokenizers
import transformers
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

REVISION = "e27e171792f303d74c048184e8259ba663cfa0d5"
SOURCE_HASHES = {
    "tokenizer.json": "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca",
    "tokenizer_config.json": "dbd4722605382159a16f2cf5779f4a8b29c51d8b397d2a05c099c90eeacfe771",
    "chat_template.jinja": "877c2dbf365243dda14738ca250b31398059b35ffd91d4ed0ec68be45ca6ebd3",
}
SOURCE = Path(sys.argv[1])
for name, expected_hash in SOURCE_HASHES.items():
    actual_hash = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"{name} does not match the pinned source revision")
OUTPUT = Path(__file__).resolve().parents[1] / "Tests/MLXLMTests/Resources/Ricursive"
OUTPUT.mkdir(parents=True, exist_ok=True)

raw_json = json.loads((SOURCE / "tokenizer.json").read_text())
raw = Tokenizer.from_file(str(SOURCE / "tokenizer.json"))
template = (SOURCE / "chat_template.jinja").read_text()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=str(SOURCE / "tokenizer.json"),
    bos_token="<|endoftext|>", eos_token="<|im_end|>", unk_token="<|endoftext|>",
    pad_token="<|pad|>", clean_up_tokenization_spaces=False,
)

probes = [
    "x = 1234567890 + 3.14159; y=-42",
    "I'M here, you'RE there, it's 4:05pm. DON'T",
    "a  b   c\n\n\n  d\t\te \r\n f",
    "    def f(self):\n        return {'k': [1, 2]}\n",
    "café naïve 日本語テキスト 한국어 🙂👍 Ελληνικά",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nok\n</think>\nHello<|im_end|>",
]

tools = [{
    "type": "function",
    "function": {
        "name": "execute_bash", "description": 'Run café code: "safe", \\path',
        "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}},
            "required": ["command"], "additionalProperties": False,
        },
    },
}]
cases = [
    {"name": "system_user", "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]},
    {"name": "code_whitespace_unicode", "messages": [
        {"role": "user", "content": "    def f(self):\n        return {'café': [1234567890, 2]}\n\n日本語 🙂\t"},
    ]},
    {"name": "reasoning_history", "messages": [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "reasoning_content": "Add the numbers.", "content": "4"},
        {"role": "user", "content": "Why?"},
    ]},
    {"name": "existing_thinking", "messages": [
        {"role": "assistant", "reasoning_content": "Do not duplicate.",
         "content": " \n<think>\nAlready there.\n</think>\n4"},
    ]},
    {"name": "clear_thinking", "context": {"clear_thinking": True}, "messages": [
        {"role": "assistant", "reasoning_content": "Do not include this.", "content": "4"},
        {"role": "user", "content": "Next."},
    ]},
    {"name": "tools_round_trip", "tools": tools, "messages": [
        {"role": "system", "content": "Preserve whitespace."},
        {"role": "user", "content": "Show the working directory."},
        {"role": "assistant", "reasoning_content": "Check both commands.", "content": "Running now.\n",
         "tool_calls": [
             {"id": "call_1", "type": "function", "function": {
                 "name": "execute_bash", "arguments": {"command": "printf 'café: 1,2\\n'", "limit": 1}}},
             {"name": "execute_bash", "arguments": {"command": "pwd", "enabled": True, "fallback": None}},
         ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "café: 1,2\n"},
        {"role": "tool", "tool_call_id": "call_2", "content": "/tmp/ripple"},
        {"role": "user", "content": "Continue."},
    ]},
    {"name": "tool_only_history", "messages": [
        {"role": "tool", "content": "first\n"},
        {"role": "tool", "content": "second"},
    ]},
    {"name": "empty_tool_content", "messages": [
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "execute_bash", "arguments": {"command": "pwd"}}},
        ]},
        {"role": "tool", "content": ""},
    ]},
    {"name": "later_system", "messages": [
        {"role": "system", "content": "First."},
        {"role": "user", "content": "Hi."},
        {"role": "system", "content": "Second."},
    ]},
]

# A Swift dictionary has no insertion order. Use the same sorted order for JSON
# objects given to the unmodified HF template; the template controls whitespace.
cases = json.loads(json.dumps(cases, ensure_ascii=False, sort_keys=True))
for case in cases:
    case["rendered"] = tokenizer.apply_chat_template(
        case["messages"], tools=case.get("tools"), chat_template=template,
        tokenize=False, add_generation_prompt=True, **case.get("context", {}),
    )
    case["ids"] = raw.encode(case["rendered"], add_special_tokens=False).ids

texts = probes + [case["rendered"] for case in cases]
pieces = {piece for text in texts for piece, _ in raw.pre_tokenizer.pre_tokenize_str(text)}
merges = [pair if isinstance(pair, list) else pair.split(" ") for pair in raw_json["model"]["merges"]]
kept_merges = [pair for pair in merges if any("".join(pair) in piece for piece in pieces)]
kept_tokens = {token for token in raw_json["model"]["vocab"] if len(token) == 1}
kept_tokens.update(token["content"] for token in raw_json["added_tokens"])
for left, right in kept_merges:
    kept_tokens.update((left, right, left + right))
reduced = copy.deepcopy(raw_json)
reduced["model"]["vocab"] = {token: index for token, index in raw_json["model"]["vocab"].items() if token in kept_tokens}
reduced["model"]["merges"] = kept_merges
reduced_raw = Tokenizer.from_str(json.dumps(reduced, ensure_ascii=False))
for text in texts:
    expected = raw.encode(text, add_special_tokens=False).ids
    assert reduced_raw.encode(text, add_special_tokens=False).ids == expected, text
    assert reduced_raw.decode(expected, skip_special_tokens=False) == raw.decode(expected, skip_special_tokens=False), text

reference = {
    "repository": "ricursiveAI/Ripple-65B-Preview", "revision": REVISION,
    "source_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                      for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")},
    "generator": {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
    "eos_token_ids": [100265, 100257],
    "probes": [{"text": text, "ids": raw.encode(text, add_special_tokens=False).ids} for text in probes],
    "cases": cases,
}
for name in ("tokenizer_config.json", "chat_template.jinja"):
    (OUTPUT / name).write_bytes((SOURCE / name).read_bytes())
(OUTPUT / "tokenizer.json").write_text(json.dumps(reduced, ensure_ascii=False, separators=(",", ":")) + "\n")
(OUTPUT / "tokenizer-reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n")
print(f"Wrote {len(kept_tokens)} vocabulary entries and {len(kept_merges)} merges to {OUTPUT}")
