#!/bin/sh
set -eu
runner="$1"
checkpoint="$2"
python="$3"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"$python" "$(dirname "$0")/prepare.py" "$checkpoint" "$work"
MLX_ENABLE_TF32=1 MLX_SMOKE_FLOAT32=0 "$runner" "$checkpoint" "$work/input.json" > "$work/native.json"
MLX_ENABLE_TF32=0 MLX_SMOKE_FLOAT32=1 "$runner" "$checkpoint" "$work/input.json" > "$work/accurate.json"
"$python" - "$checkpoint" "$work/native.json" "$work/accurate.json" <<'PY'
import json,sys
from pathlib import Path
from tokenizers import Tokenizer
tokenizer=Tokenizer.from_file(str(Path(sys.argv[1])/'tokenizer.json'))
reports=[json.loads(Path(path).read_text()) for path in sys.argv[2:]]
for report in reports:
    report['generatedText']=[tokenizer.decode(t,skip_special_tokens=False) for t in report['generatedTokens']]
    assert all(len(t)==8 for t in report['generatedTokens']),report
accurate=reports[1]
assert accurate['float32Compute'] and accurate['batchMaxAbsDifference'] < 1e-3,accurate
assert accurate['batchRelativeDifference'] < 1e-4,accurate
print(json.dumps(reports,indent=2))
PY
