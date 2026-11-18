#!/usr/bin/env bash
# v4 work package 8: fair benchmark queue (tune -> final -> leaky -> sweep)
cd "$(dirname "$0")/.."
PY=/c/Users/hassan.waqas/AppData/Local/Programs/Python/Python312/python.exe
S="$PY -u scripts/41_v4_fair_benchmark.py"
run() { echo "=== $(date +%H:%M:%S) $*"; "$@" 2>&1 | grep -v -i "warn"; }
run $S --phase tune  --datasets anonnet
run $S --phase final --datasets anonnet
run $S --phase leaky --datasets anonnet --methods sage,gat,rgcn
run $S --phase tune  --datasets lodhalal --methods sage,gat,rgcn
run $S --phase tune  --datasets lodhalal --methods TransE,DistMult,ComplEx,RotatE
run $S --phase final --datasets lodhalal
run $S --phase sweep --datasets anonnet,lodhalal --methods sage,gat,rgcn
echo "=== queue2 finished $(date +%H:%M:%S)"
