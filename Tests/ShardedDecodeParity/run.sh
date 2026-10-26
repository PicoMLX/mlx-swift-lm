#!/bin/sh
# PicoMLX fork only. Runs two ranks of ShardedDecodeParity over MLX's ring backend on this
# Mac and checks that a sharded Qwen 3.5 (dense and mixture-of-experts, float and 4-bit)
# decodes like the whole model, with each rank holding half of every split weight.
#
# Usage: sh Tests/ShardedDecodeParity/run.sh <path to the built ShardedDecodeParity>
set -eu

runner="$1"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

ports="$(python3 -I -c 'import socket
s = [socket.socket() for _ in range(2)]
for x in s: x.bind(("127.0.0.1", 0))
print(" ".join(str(x.getsockname()[1]) for x in s))')"
set -- $ports
printf '[["127.0.0.1:%s"], ["127.0.0.1:%s"]]' "$1" "$2" > "$work/hosts.json"

MLX_RANK=0 MLX_HOSTFILE="$work/hosts.json" "$runner" > "$work/rank0.json" &
first=$!
MLX_RANK=1 MLX_HOSTFILE="$work/hosts.json" "$runner" > "$work/rank1.json" &
second=$!
wait "$first"
wait "$second"

python3 -I - "$work/rank0.json" "$work/rank1.json" <<'PY'
import json, sys

reports = [json.load(open(path)) for path in sys.argv[1:]]
expected_rows = {"dense": 64, "moe": 32}
failures = []
for report in reports:
    assert report["size"] == 2, report
    for key, result in report.items():
        if key in ("rank", "size"):
            continue
        label = f"rank {report['rank']}, {key}"
        # Float32 matmuls may run as TF32, so split sums can differ by about 1e-3.
        if result["relativeDifference"] > 5e-3:
            failures.append(f"{label}: sharded logits differ by {result['relativeDifference']}")
        if result["tokens"] != result["wholeTokens"]:
            failures.append(f"{label}: sharded tokens differ from the whole model's")
        if result["gateRows"] != expected_rows[key.split(",")[0]]:
            failures.append(f"{label}: holds {result['gateRows']} gate rows")
for key in reports[0]:
    if key not in ("rank", "size") and reports[0][key]["tokens"] != reports[1][key]["tokens"]:
        failures.append(f"{key}: ranks decoded different tokens")
print(json.dumps(reports, indent=1, sort_keys=True))
if failures:
    print("\n".join(failures))
    sys.exit(1)
print("Sharded decode matches the whole model on both ranks.")
PY
