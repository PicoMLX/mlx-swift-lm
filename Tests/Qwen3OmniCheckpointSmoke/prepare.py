"""Create bounded media and exact native chat-template/tokenizer IDs, offline."""
import json
from pathlib import Path
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

checkpoint, work = map(Path, sys.argv[1:])
work.mkdir(parents=True, exist_ok=True)
image = Image.new("RGB", (64, 64), "royalblue")
ImageDraw.Draw(image).rectangle((16, 16, 48, 48), fill="yellow")
image.save(work / "image.png")
samples = (np.sin(np.arange(16000, dtype=np.float32) * (2 * np.pi * 440 / 16000)) * 0.08
           + np.random.default_rng(314159).normal(0, 0.001, 16000)).astype(np.float32)
with wave.open(str(work / "audio.wav"), "wb") as output:
    output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
    output.writeframes((samples * 32767).astype("<i2").tobytes())
tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
cases = []
for label, kinds, prompt in [
    ("text", [], "Reply with the word OK."),
    ("image", ["image"], "Describe the picture briefly."),
    ("audio", ["audio"], "Describe the sound briefly."),
    ("video_audio", ["video"], "Describe the video and sound briefly."),
]:
    messages = [{"role": "user", "content": [{"type": kind} for kind in kinds]
                 + [{"type": "text", "text": prompt}]}]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=True, return_dict=False)
    if hasattr(tokens, "input_ids") or isinstance(tokens, dict):
        tokens = tokens["input_ids"]
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    tokens = list(tokens)
    cases.append(dict(label=label, prompt=prompt,
        signature=json.dumps(messages, sort_keys=True, separators=(",", ":")), tokens=tokens))
(work / "input.json").write_text(json.dumps(dict(image=str(work / "image.png"),
    audio=str(work / "audio.wav"), video=str(work / "video.mov"),
    cases=cases, specialTokens=tokenizer.get_added_vocab())))
