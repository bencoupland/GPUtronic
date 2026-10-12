#!/bin/bash
# Real Z=1.0 gain sweep using the integrated histogram governor
# Run this on the machine with Cyberpunk or the bursty workload

set -e

BIN=./gputronic_v25_z1
OUT=results/z1_sweep_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

echo "=== GPUTronic v25 Z=1.0 Cyberpunk Sweep ==="
echo "Gain 1: Kp=2.0 Ki=0.25 Target=1.0"
$BIN 2.0 0.25 1.0 > "$OUT/kp2.0_ki0.25.log" 2>&1 &
PID=$!
sleep 25
kill -INT $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
echo "Gain 1 complete."

echo
echo "Gain 2: Kp=4.5 Ki=0.55 Target=1.0"
$BIN 4.5 0.55 1.0 > "$OUT/kp4.5_ki0.55.log" 2>&1 &
PID=$!
sleep 25
kill -INT $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
echo "Gain 2 complete."

echo
echo "Sweep finished. Histograms in $OUT"