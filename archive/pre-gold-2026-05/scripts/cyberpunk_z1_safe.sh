#!/bin/bash
# cyberpunk_z1_safe.sh
# Cyberpunk 2077 + GPUTronic v26.1 SAFE governor (one-sided control)
#
# Usage: ./cyberpunk_z1_safe.sh [kp] [ki] [target] [duration_seconds]

set -e

KP=${1:-0.35}
KI=${2:-0.04}
TARGET=${3:-1.4}
DURATION=${4:-360}   # default 6 minutes

RESULTS="results/cyberpunk_z1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS"

GOVERNOR="./gputronic_q_axis_governor_v26_observe"

if [ ! -x "$GOVERNOR" ]; then
    echo "ERROR: $GOVERNOR not found or not executable"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo "GPUTronic v26.1 SAFE — Cyberpunk 2077 Z=1.0"
echo "Kp=$KP Ki=$KI Target=$TARGET Duration=${DURATION}s"
echo "Results → $RESULTS"
echo "═══════════════════════════════════════════════════════════════"
echo

echo "[1/3] Launching Cyberpunk 2077..."
steam -applaunch 1091500 >/dev/null 2>&1 &
echo "    Steam launched. Start your 4K Path Tracing benchmark when ready."
echo

sleep 25

echo "[2/3] Starting SAFE governor (one-sided control)..."
$GOVERNOR $KP $KI $TARGET > "$RESULTS/cyberpunk_safe.log" 2>&1 &
GOV_PID=$!
echo "    Governor running (PID $GOV_PID)"

# Run for the requested duration
sleep $DURATION

echo
echo "[3/3] Stopping governor..."
kill -INT $GOV_PID 2>/dev/null || true
sleep 2
kill -9 $GOV_PID 2>/dev/null || true
wait $GOV_PID 2>/dev/null || true

echo
echo "Run complete."
echo "Log: $RESULTS/cyberpunk_safe.log"
echo "Close Cyberpunk when finished."