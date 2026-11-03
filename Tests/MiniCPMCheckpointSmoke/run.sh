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
from transformers import AutoTokenizer
reports=[json.loads(Path(path).read_text()) for path in sys.argv[2:]]
tokenizer=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True,trust_remote_code=False)
for report in reports:
    report['generatedText']=[tokenizer.decode(tokens,skip_special_tokens=False) for tokens in report['generatedTokens']]
    assert report['pixelShape']==[1,1024,588],report
    assert report['cropGrids']==[[1,32,32]],report
    assert all(len(tokens)==8 for tokens in report['generatedTokens']),report
accurate=reports[1]
assert accurate['float32Compute'] and accurate['batchMaxAbsDifference']<1e-3,accurate
assert accurate['batchRelativeDifference']<1e-4,accurate
print(json.dumps(reports,indent=2))
PY
