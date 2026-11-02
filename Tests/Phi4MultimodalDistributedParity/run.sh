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
    for label in ('float,text','4-bit,text','float,media','4-bit,media'):
        v=r[label]
        assert v['gateRows']==64,(label,v)
        assert v['keyRows']==16,(label,v)
        assert v['kvHeads']==[1,1],(label,v)
        assert v['relativeDifference']<5e-3,(label,v)
        assert v['tokens']==v['wholeTokens'],(label,v)
for label in ('float,text','4-bit,text','float,media','4-bit,media'):
    assert reports[0][label]['tokens']==reports[1][label]['tokens']
print(json.dumps(reports,indent=2))
print('Phi-4 mixed-mode distributed parity passed on both ranks.')
PY
