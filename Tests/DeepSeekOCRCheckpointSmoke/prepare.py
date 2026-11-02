"""Prepare native tokenizer IDs and a 64px source image for an opt-in checkpoint smoke.

No model weights or tokenizer implementation are copied into the Swift package.
Requires Pillow and Hugging Face tokenizers in the selected Python environment.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from tokenizers import Tokenizer

checkpoint, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
image = Image.new("RGB", (64, 64), "white")
ImageDraw.Draw(image).text((5, 20), "MLX OCR", fill="black")
image.save(output / "image.png")
prompts = ["<image>\nFree OCR.", "What is 2+2?"]
fixture = {
    "image": str((output / "image.png").resolve()),
    "prompts": prompts,
    "encodings": {p: tokenizer.encode(p, add_special_tokens=False).ids
                  for prompt in prompts for p in (prompt, prompt + " ")},
    "imageTokenId": tokenizer.token_to_id("<image>"),
    "specialTokens": {token.content: token_id for token_id, token
                      in tokenizer.get_added_tokens_decoder().items()},
}
(output / "input.json").write_text(json.dumps(fixture))
print(output / "input.json")
