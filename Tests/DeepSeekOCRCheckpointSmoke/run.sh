#!/bin/sh
# Run after building the opt-in DeepSeekOCRCheckpointSmoke executable.
set -eu
runner="$1"
checkpoint="$2"
python="$3"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"$python" "$(dirname "$0")/prepare.py" "$checkpoint" "$work"
MLX_ENABLE_TF32=0 "$runner" "$checkpoint" "$work/input.json" > "$work/report.json"
"$python" - "$checkpoint" "$work/report.json" <<'PY'
import json, sys
from pathlib import Path
from tokenizers import Tokenizer
report = json.loads(Path(sys.argv[2]).read_text())
tokenizer = Tokenizer.from_file(str(Path(sys.argv[1]) / "tokenizer.json"))
report["generatedText"] = [tokenizer.decode(t, skip_special_tokens=False)
                           for t in report["generatedTokens"]]
assert report["pixelShape"] == [1, 3, 1024, 1024], report
assert all(len(tokens) == 8 for tokens in report["generatedTokens"]), report
print(json.dumps(report, indent=2))
PY
