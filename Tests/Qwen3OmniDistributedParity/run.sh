#!/bin/sh
# Two actual MLX ring ranks on one Mac. For multi-host, use the same executable with MLX_HOSTFILE.
set -eu
runner="$1"
fixture="$2"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
ports="$(python3 -I -c 'import socket
s = [socket.socket() for _ in range(2)]
for x in s: x.bind(("127.0.0.1", 0))
print(" ".join(str(x.getsockname()[1]) for x in s))')"
set -- $ports
printf '[["127.0.0.1:%s"], ["127.0.0.1:%s"]]' "$1" "$2" > "$work/hosts.json"
MLX_RANK=0 MLX_HOSTFILE="$work/hosts.json" "$runner" "$fixture" > "$work/rank0.json" &
first=$!
MLX_RANK=1 MLX_HOSTFILE="$work/hosts.json" "$runner" "$fixture" > "$work/rank1.json" &
second=$!
wait "$first"
wait "$second"
python3 -I - "$work/rank0.json" "$work/rank1.json" <<'PY'
import json, sys
reports = [json.load(open(path)) for path in sys.argv[1:]]
for report in reports:
    assert report["size"] == 2, report
    for key, result in report.items():
        if key in ("rank", "size"): continue
        assert result["relativeDifference"] < 5e-3, (key, result)
        assert result["tokens"] == result["wholeTokens"], (key, result)
        assert result["qRows"] == 32, (key, result)
        assert result["expertRows"] == 32, (key, result)
        assert result["keyRows"] == 16, (key, result)
        assert result["kvHeads"] == [1, 1], (key, result)
        assert result["tokens"] == reports[0][key]["tokens"], (key, result)
print(json.dumps(reports, indent=2))
print("Qwen3 Omni float/4-bit solo/batched text/full-media parity passed on two actual ranks.")
PY
