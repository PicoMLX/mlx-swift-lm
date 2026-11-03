#!/bin/sh
# Two local ranks must hold distinct half-width projections and preserve mixed-mode logits.
set -eu
runner="$1"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
ports="$(python3 -I -c 'import socket
s=[socket.socket() for _ in range(2)]
for x in s: x.bind(("127.0.0.1",0))
print(" ".join(str(x.getsockname()[1]) for x in s))')"
set -- $ports
printf '[["127.0.0.1:%s"],["127.0.0.1:%s"]]' "$1" "$2" > "$work/hosts.json"
MLX_RANK=0 MLX_HOSTFILE="$work/hosts.json" "$runner" > "$work/0.json" &
first=$!
MLX_RANK=1 MLX_HOSTFILE="$work/hosts.json" "$runner" > "$work/1.json" &
second=$!
wait "$first"
wait "$second"
python3 -I - "$work/0.json" "$work/1.json" <<'PY'
import json,sys
reports=[json.load(open(p)) for p in sys.argv[1:]]
for r in reports:
    assert r['size']==2,r
    for label in [key for key in r if key not in ('rank','size')]:
        v=r[label]
        assert v['queryRows']*r['size']==v['wholeQueryRows'],(label,v)
        assert all(h==1 for h in v['kvHeads']),(label,v)
        assert v['relativeDifference']<5e-3,(label,v)
        assert v['tokens']==v['wholeTokens'],(label,v)
for label in [key for key in reports[0] if key not in ('rank','size')]:
    assert reports[0][label]['tokens']==reports[1][label]['tokens']
print(json.dumps(reports,indent=2))
print('MiniCPM dense and MoE distributed parity passed on both ranks.')
PY
