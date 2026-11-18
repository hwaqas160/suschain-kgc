#!/usr/bin/env bash
# Runs the remaining v4 GPU experiments one after another.
# Usage (from suschain-kgc/): bash scripts/v4_queue.sh
set -u
PY=C:/Users/hassan.waqas/AppData/Local/Programs/Python/Python312/python.exe
mkdir -p logs
run() { echo "=== $(date '+%H:%M:%S') start $*"; "$PY" -u "$@"; echo "=== $(date '+%H:%M:%S') exit $? $*"; }
run scripts/36_v4_anonnet_substitution.py                       > logs/v4_36_anonnet_tau0.1.log 2>&1
run scripts/35_v4_analysis.py                                   > logs/v4_35_analysis.log 2>&1
run scripts/33_v4_nodeclf_no_cert.py                            > logs/v4_33_nodeclf.log 2>&1
run scripts/40_v4_coldstart_control.py                         > logs/v4_40_coldstart.log 2>&1
run scripts/37_v4_risk_sensitivity.py                           > logs/v4_37_risk_sensitivity.log 2>&1
for tau in 0.05 0.2 0.3; do
  run scripts/36_v4_anonnet_substitution.py --tau $tau --convs sage --lambdas 0,1 \
                                                                > logs/v4_36_anonnet_tau$tau.log 2>&1
done
echo "=== queue finished $(date '+%H:%M:%S')"
