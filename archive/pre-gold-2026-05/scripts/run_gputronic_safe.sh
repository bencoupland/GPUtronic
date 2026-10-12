#!/bin/bash
# Safe wrapper for GPUTronic v26 (prevents hard lockups)
# Usage: ./run_gputronic_safe.sh [kp] [ki] [target_pm] [max_seconds]

set -e

KP=${1:-0.35}
KI=${2:-0.04}
TARGET=${3:-1.0}
MAX_SECONDS=${4:-300}   # 5 minutes default safety timeout

BINARY="./gputronic_q_axis_governor_v26_observe"

if [ ! -x "$BINARY" ]; then
    echo "Building v26 safe governor..."
    nvcc -O3 -arch=sm_120 -o "$BINARY" gputronic_q_axis_governor_v26_observe.cu -lcuda -lpthread
fi

echo "=== GPUTronic SAFE RUNNER ==="
echo "Kp=$KP Ki=$KI Target=$TARGET MaxRuntime=${MAX_SECONDS}s"
echo "Starting with hard timeout + kill safeguards..."
echo

# Run under timeout with aggressive cleanup on exit
timeout --signal=SIGTERM --kill-after=10s ${MAX_SECONDS}s \
    "$BINARY" $KP $KI $TARGET || EXIT_CODE=$?

echo
echo "=== Post-run cleanup ==="

# Force kill any remaining processes
pkill -f gputronic_q_axis_governor_v26_observe || true

# Try to recover the GPU
echo "Attempting GPU recovery (cudaDeviceReset)..."
cat > /tmp/gpu_reset.cu << 'EOF'
#include <cuda_runtime.h>
int main() {
    cudaDeviceReset();
    printf("GPU reset attempted.\n");
    return 0;
}
EOF
nvcc -o /tmp/gpu_reset /tmp/gpu_reset.cu && /tmp/gpu_reset || echo "Reset failed or not needed"

echo "Safe run finished (exit code: ${EXIT_CODE:-0})"
echo "If the machine is still responsive, we succeeded."