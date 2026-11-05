#!/usr/bin/env python3
"""Generate independent RGB resize goldens using Pillow, without MLX or checkpoints."""

import json
from pathlib import Path
import random

from PIL import Image, __version__

rng = random.Random(314159)
cases = []
for name, height, width, target_height, target_width in [
    ("mild_downscale", 24, 24, 20, 20),
    ("upscale", 7, 9, 15, 18),
    ("asymmetric", 19, 8, 7, 21),
    ("singleton", 5, 7, 1, 3),
    ("identity", 8, 11, 8, 11),
]:
    pixels = bytes(rng.randrange(256) for _ in range(height * width * 3))
    image = Image.frombytes("RGB", (width, height), pixels)
    resized = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
    cases.append(dict(name=name, height=height, width=width,
                      targetHeight=target_height, targetWidth=target_width,
                      pixels=list(pixels), resized=list(resized.tobytes())))
output = Path(__file__).with_name("phi4mm-pixels.json")
output.write_text(json.dumps(dict(pillowVersion=__version__, resampling="RGB BILINEAR", cases=cases)) + "\n")
print(f"{output}: {len(cases)} cases, {output.stat().st_size} bytes")
