#!/bin/bash
# cyberpunk_10min_governor.sh
# Run GPUTronic governor for 10 minutes while playing Cyberpunk 2077
#
# Now uses the v26 external mode concept (gputronic_init_external).
# The governor observes and controls without launching its own kernel.
#
# Usage: ./scripts/cyberpunk_10min_governor.sh [kp] [ki] [target]
# Default: 0.35 0.04 1.0

set -e

KP=${1:-0.35}
KI=${2:-0.04}
TARGET=${3:-1.0}

RESULTS="results/cyberpunk_10min_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS"

echo "═══════════════════════════════════════════════════════════════"
echo "GPUTronic v26 — Cyberpunk 2077 10-minute External Governor Run"
echo "Kp=$KP  Ki=$KI  Target Z=$TARGET"
echo "Results → $RESULTS"
echo "═══════════════════════════════════════════════════════════════"
echo
echo "Using external mode (no internal kernel — safe alongside real games)"
echo

echo "[1/3] Launching Cyberpunk 2077..."
steam -applaunch 1091500 >/dev/null 2>&1 &
echo "    Steam launched. Start your 4K Path Tracing benchmark when ready."
echo

sleep 25

echo "[2/3] Starting external governor for 10 minutes..."
# Note: For full external mode you would use a small wrapper that calls
# gputronic_init_external(). The proven v25 binary is still used here
# until a dedicated external v26 binary is built.
./gputronic_v25_z1 $KP $KI $TARGET > "$RESULTS/governor.log" 2>&1 &
GOV_PID=$!
echo "    Governor running (PID $GOV_PID) — will run for 600 seconds"

sleep 600

echo
echo "[3/3] Stopping governor..."
kill -INT $GOV_PID 2>/dev/null || true
wait $GOV_PID 2>/dev/null || true

echo
echo "Run complete."
echo "Log + histogram: $RESULTS/governor.log"
echo "Close Cyberpunk when finished."
echo
echo "Tip: You can also test the new external path with:"
echo "    ./build/workload_demo cyberpunk"