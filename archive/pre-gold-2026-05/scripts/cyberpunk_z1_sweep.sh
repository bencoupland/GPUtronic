#!/bin/bash
# cyberpunk_z1_sweep.sh
# Back-to-back Z=1.0 gain sweep for Cyberpunk 2077 (4K Path Tracing)
# Usage: ./cyberpunk_z1_sweep.sh
#
# Prerequisites:
#   1. Launch Cyberpunk 2077 (4K PT, RT Overdrive, benchmark mode or free roam)
#   2. Note the PID or just let the governor attach via the persistent kernel
#   3. Run this script — it will do two 60-second captures with different gains

set -e

LAUNCHER=./gputronic_launcher_v25_z1
DURATION=60          # seconds per gain point
LOGDIR=results/cyberpunk_z1_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"

echo "═══════════════════════════════════════════════════════════════"
echo "GPUTronic Z=1.0 Cyberpunk Sweep"
echo "Target: Z=1.0 | Workload: Cyberpunk 2077 4K Path Tracing"
echo "Gains: (2.0, 0.25) then (4.5, 0.55)"
echo "═══════════════════════════════════════════════════════════════"
echo

# Gain set 1
echo "[SWEEP] Kp=2.0 Ki=0.25 — 60 s capture"
$LAUNCHER 2.0 0.25 ./gputronic_v24_minimal $DURATION 2>&1 | tee "$LOGDIR/kp2.0_ki0.25.log"
echo

sleep 5

# Gain set 2
echo "[SWEEP] Kp=4.5 Ki=0.55 — 60 s capture"
$LAUNCHER 4.5 0.55 ./gputronic_v24_minimal $DURATION 2>&1 | tee "$LOGDIR/kp4.5_ki0.55.log"
echo

echo "Sweep complete. Logs in $LOGDIR"
echo "Compare the two histograms for how close each gain set stays to Z=1.0 under real game load."