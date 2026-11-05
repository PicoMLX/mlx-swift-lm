#!/bin/sh
# Opt-in: the complete published checkpoint must already exist locally.
set -eu
runner="$1"
checkpoint="$2"
python="$3"
work="${MLX_SMOKE_WORK:-$(mktemp -d)}"
if [ -z "${MLX_SMOKE_WORK:-}" ]; then trap 'rm -rf "$work"' EXIT; fi
mkdir -p "$work"
"$python" "$(dirname "$0")/prepare.py" "$checkpoint" "$work"
LLVM_PROFILE_FILE="$work/smoke-%m.profraw" MLX_ENABLE_TF32=0 MLX_SMOKE_FLOAT32=1 "$runner" "$checkpoint" "$work/input.json" > "$work/report.json"
"$python" - "$checkpoint" "$work/report.json" <<'PY'
import json, sys
from pathlib import Path
from tokenizers import Tokenizer
report = json.loads(Path(sys.argv[2]).read_text())
tokenizer = Tokenizer.from_file(str(Path(sys.argv[1]) / "tokenizer.json"))
eos = tokenizer.token_to_id("<|im_end|>")
report["generatedText"] = [tokenizer.decode(tokens[:tokens.index(eos)] if eos in tokens else tokens,
                                           skip_special_tokens=True)
                           for tokens in report["generatedTokens"]]
assert report["labels"] == ["text", "image", "audio", "video_audio"], report
assert report["float32Compute"] and report["tf32Disabled"], report
assert report["batchRelativeDifference"] < 1e-4, report
assert all(len(tokens) == 8 for tokens in report["generatedTokens"]), report
print(json.dumps(report, indent=2))
PY
