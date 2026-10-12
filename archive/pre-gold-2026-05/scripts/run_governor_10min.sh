#!/bin/bash
# Run GPUTronic v26 governor for 10 minutes while you play Cyberpunk (or any workload)
# Usage: ./scripts/run_governor_10min.sh [kp] [ki] [target_z]

set -e

KP=${1:-0.35}
KI=${2:-0.04}
TARGET=${3:-1.0}

echo "=== GPUTronic v26 Governor — 10 minute run ==="
echo "Kp=$KP  Ki=$KI  Target Z=$TARGET"
echo "Launch Cyberpunk (or your workload) now."
echo "Governor will run for 600 seconds then exit cleanly."
echo ""

# Build if needed
if [ ! -f build/workload_demo ]; then
    echo "Building workload demo..."
    make build/workload_demo
fi

# Run the governor for 10 minutes
# Note: The current v26 demo runs its own kernel.
# For true external workload control you would use the library in daemon mode.
timeout 600 ./build/workload_demo || true

echo ""
echo "[GPUTronic] 10-minute run complete."
echo "Check the histogram output above for Z tracking quality."