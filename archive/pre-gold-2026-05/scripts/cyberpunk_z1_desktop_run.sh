#!/bin/bash
# cyberpunk_z1_desktop_run.sh
# Single long gentle sweep for Cyberpunk 2077 Z=1.0 testing
#
# Usage: ./cyberpunk_z1_desktop_run.sh
# Run the 4K Path Tracing benchmark for ~5-6 minutes, then close the game.

set -e

RESULTS="results/cyberpunk_z1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS"

echo "═══════════════════════════════════════════════════════════════"
echo "GPUTronic v25 — Cyberpunk Z=1.0 Single Gentle Run"
echo "Results → $RESULTS"
echo "═══════════════════════════════════════════════════════════════"
echo

echo "[1/3] Launching Cyberpunk 2077..."
steam -applaunch 1091500 >/dev/null 2>&1 &
echo "    Steam launched. Start your benchmark when ready."
echo

sleep 20

# Very gentle single gain set
echo "[2/3] Starting governor — Kp=0.35 Ki=0.04 Target=1.0 (single 6 min run)"
./gputronic_v25_z1 0.35 0.04 1.0 > "$RESULTS/single_gentle.log" 2>&1 &
GOV_PID=$!
echo "    Governor running (PID $GOV_PID)"

# One long window
sleep 360   # 6 minutes

kill -INT $GOV_PID 2>/dev/null || true
wait $GOV_PID 2>/dev/null || true

echo
echo "[3/3] Run complete."
echo "    Close Cyberpunk when finished."
echo "    Histogram should be in: $RESULTS/single_gentle.log"