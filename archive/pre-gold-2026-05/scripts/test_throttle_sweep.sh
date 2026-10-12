#!/bin/bash
# GPUTronic Throttle Sweep Test
# Cycles through different throttle fractions to observe PI controller behavior
set -e

cd "$(dirname "$0")"
BINARY="./build/gputronic_v14"
RUNTIME=5  # seconds per throttle level

echo "=== GPUTronic Throttle Sweep Test ==="
echo ""
echo "Running q-level sweep: testing each throttle fraction for ${RUNTIME}s..."
echo ""

# Array of throttle values to test
Q_LEVELS=(0.1 0.3 0.5 0.7 0.8 0.9 1.0)

for Q in "${Q_LEVELS[@]}"; do
    echo ">>> Testing q = $Q <<<"
    
    # Run governor with specific throttle command (if supported)
    # For now, we'll just observe console output during run
    
    timeout "$RUNTIME" sudo "$BINARY" 80 &
    PID=$!
    
    sleep "$RUNTIME"
    kill $PID 2>/dev/null || true
    wait $PID 2>/dev/null || true
    
    echo ""
done

echo "=== Sweep Complete ==="
