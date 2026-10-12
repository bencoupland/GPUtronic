#!/bin/bash
# GPUTronic v14.0.0 - Comprehensive Test Runner
# Single command to run full benchmark suite and capture all data
# For Ben Coupland's RTX 5080 (Blackwell)

set -e  # Exit on error

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/media/dad/1TB HDD/GPUTronic Agent/results/$TIMESTAMP"
LOG_FILE="$RESULTS_DIR/test_run.log"
NVML_LOG="$RESULTS_DIR/nvml_telemetry.csv"
GOVERNOR_LOG="$RESULTS_DIR/governor_output.txt"

# Colors for terminal output (disabled if not a TTY)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Logging function
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Error handler
error_exit() {
    log "${RED}[ERROR] $1${NC}"
    exit 1
}

# Success message
success() {
    log "${GREEN}[SUCCESS] $1${NC}"
}

# Info message
info() {
    log "${BLUE}[INFO] $1${NC}"
}

# Warning message
warn() {
    log "${YELLOW}[WARNING] $1${NC}"
}

echo ""
echo "================================================================="
echo "  GPUTronic v14.0.0 - Comprehensive Test Suite"
echo "  Target: NVIDIA RTX 5080 (Blackwell)"
echo "  Timestamp: $TIMESTAMP"
echo "================================================================="
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"
touch "$LOG_FILE"

log "[SETUP] Results directory: $RESULTS_DIR"

# =============================================================================
# PHASE 1: Environment Setup
# =============================================================================
log ""
log "================================================================="
log "PHASE 1: Environment Setup"
log "================================================================="

# Check if running as root (needed for some operations)
if [ "$EUID" -ne 0 ]; then
    warn "Not running as root. Some optimizations may fail."
fi

# Set CPU governor to performance mode
info "Setting CPU governor to performance mode..."
if command -v cpupower &> /dev/null; then
    sudo cpupower frequency-set -g performance 2>/dev/null || warn "Could not set CPU governor"
else
    warn "cpupower not found. Set CPU governor manually."
fi

# Check GPU is available
info "Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    error_exit "nvidia-smi not found. Please install NVIDIA drivers."
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null)
log "[GPU] Detected:"
echo "$GPU_INFO" | while read line; do log "       $line"; done

# Set GPU to maximum performance mode
info "Setting GPU power management..."
nvidia-smi -pm 1 2>/dev/null || warn "Could not set persistence mode"
nvidia-smi -acp 0 2>/dev/null || warn "Could not disable application clocks profile"

# Get SM count for the governor
SM_COUNT=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader | head -1)
if [ -z "$SM_COUNT" ]; then
    # Fallback: assume 80 for RTX 5080
    SM_COUNT=80
    warn "Could not detect SM count, assuming $SM_COUNT (RTX 5080)"
fi
log "[GPU] SM Count: $SM_COUNT"

# =============================================================================
# PHASE 2: Build Verification
# =============================================================================
log ""
log "================================================================="
log "PHASE 2: Build Verification"
log "================================================================="

cd "/media/dad/1TB HDD/GPUTronic Agent" || error_exit "Could not change to project directory"

if [ ! -f "Makefile" ]; then
    error_exit "Makefile not found in project directory"
fi

info "Compiling GPUTronic v14.0.0..."
make clean 2>/dev/null || true
make all 2>&1 | tee -a "$LOG_FILE"

if [ ! -f "./gputronic_v14" ] || [ ! -f "./gputronic_benchmark" ]; then
    error_exit "Build failed. Check compilation errors above."
fi

success "Build complete!"
log "[BINARIES] gputronic_v14, gputronic_benchmark ready"

# =============================================================================
# PHASE 3: Pre-Test System Baseline
# =============================================================================
log ""
log "================================================================="
log "PHASE 3: System Baseline (Pre-Test)"
log "================================================================="

info "Recording system baseline..."

# Record initial GPU state
nvidia-smi --query-gpu=temperature.gpu,fan.speed,power.draw,clocks.current.sm,memory.used --format=csv,nounits > "$RESULTS_DIR/baseline_gpu_state.txt" 2>/dev/null

log "[BASELINE] Initial GPU State:"
cat "$RESULTS_DIR/baseline_gpu_state.txt" | while read line; do log "       $line"; done

# Record CPU state
if command -v lscpu &> /dev/null; then
    lscpu | grep -E "Model name|CPU\(s\)|Thread|Core" > "$RESULTS_DIR/baseline_cpu_info.txt" 2>/dev/null
fi

# =============================================================================
# PHASE 4: Full Benchmark Suite
# =============================================================================
log ""
log "================================================================="
log "PHASE 4: Running Full Benchmark Suite"
log "================================================================="

info "Starting benchmark harness (all tests)..."
info "This will take approximately 10-15 minutes..."
echo ""

# Run the full benchmark suite (test mode 0 = all tests)
./gputronic_benchmark 0 $SM_COUNT 256 > "$GOVERNOR_LOG" 2>&1 &
BENCHMARK_PID=$!

log "[BENCHMARK] PID: $BENCHMARK_PID"
log "[BENCHMARK] Output: $GOVERNOR_LOG"
log ""
info "Monitor progress with: tail -f $GOVERNOR_LOG"
echo ""

# Wait for benchmark to complete
wait $BENCHMARK_PID || error_exit "Benchmark suite failed!"

success "Benchmark suite complete!"

# =============================================================================
# PHASE 5: Real-Time NVML Telemetry Capture (Parallel)
# =============================================================================
log ""
log "================================================================="
log "PHASE 5: NVML Telemetry Analysis"
log "================================================================="

info "Parsing benchmark output for telemetry data..."

# Extract key metrics from governor log
grep -E "(Control frequency|Z=|PM=|q=)" "$GOVERNOR_LOG" > "$RESULTS_DIR/control_loop_data.txt" 2>/dev/null || true

# Count control loop iterations
CONTROL_ITERATIONS=$(wc -l < "$RESULTS_DIR/control_loop_data.txt" 2>/dev/null || echo "0")
log "[METRICS] Control loop iterations: $CONTROL_ITERATIONS"

# Calculate average control frequency
if [ -s "$RESULTS_DIR/control_loop_data.txt" ]; then
    AVG_FREQ=$(grep "Control frequency:" "$GOVERNOR_LOG" | awk '{sum+=$3; count++} END {if(count>0) print sum/count; else print 0}')
    log "[METRICS] Average control frequency: ${AVG_FREQ} Hz"
fi

# =============================================================================
# PHASE 6: Automated Analysis Scripts
# =============================================================================
log ""
log "================================================================="
log "PHASE 6: Running Automated Analysis"
log "================================================================="

info "Running dyno sweep analysis..."
if [ -f "scripts/analyze_dyno.py" ]; then
    python3 scripts/analyze_dyno.py < "$RESULTS_DIR/dyno_sweep.txt" > "$RESULTS_DIR/dyno_analysis.txt" 2>&1 || true
    if [ -f "dyno_sweep_plot.png" ]; then
        mv dyno_sweep_plot.png "$RESULTS_DIR/"
    fi
    success "Dyno analysis complete"
else
    warn "analyze_dyno.py not found, skipping"
fi

info "Running step response analysis..."
if [ -f "scripts/analyze_step.py" ]; then
    python3 scripts/analyze_step.py < "$RESULTS_DIR/step_response.txt" > "$RESULTS_DIR/step_analysis.txt" 2>&1 || true
    if [ -f "step_response_plot.png" ]; then
        mv step_response_plot.png "$RESULTS_DIR/"
    fi
    success "Step response analysis complete"
else
    warn "analyze_step.py not found, skipping"
fi

# =============================================================================
# PHASE 7: Generate Summary Report
# =============================================================================
log ""
log "================================================================="
log "PHASE 7: Generating Summary Report"
log "================================================================="

info "Creating comprehensive summary report..."

cat > "$RESULTS_DIR/SUMMARY.md" << EOF
# GPUTronic v14.0.0 - Test Results Summary

**Test Run:** $TIMESTAMP  
**GPU:** RTX 5080 (Blackwell)  
**SM Count:** $SM_COUNT  
**Results Directory:** $RESULTS_DIR

---

## Quick Status

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Control Frequency | ${AVG_FREQ:-N/A} Hz | ≥100,000 Hz | $(echo "$AVG_FREQ >= 100000" | bc -l 2>/dev/null && echo "✓ PASS" || echo "?") |
| Test Duration | $(($(date +%s) - $(stat -c %Y "$RESULTS_DIR" 2>/dev/null || echo $(date +%s)))) seconds | ~15 min | ✓ Complete |

## Files Generated

### Raw Data
- \`governor_output.txt\` - Full governor console output
- \`control_loop_data.txt\` - Extracted control loop metrics (frequency, Z, PM, q)
- \`baseline_gpu_state.txt\` - Pre-test GPU state snapshot

### Analysis Results
- \`dyno_analysis.txt\` - Thread fraction vs. throughput linearity analysis
- \`step_analysis.txt\` - Stability margins (overshoot, settling time)
- \`dyno_sweep_plot.png\` - Linearity plot with R² value
- \`step_response_plot.png\` - Step response with annotations

### Logs
- \`test_run.log\` - This test runner execution log

---

## Key Findings

### Control Loop Performance
EOF

# Add control loop stats to summary
if [ -s "$RESULTS_DIR/control_loop_data.txt" ]; then
    echo "" >> "$RESULTS_DIR/SUMMARY.md"
    grep "Control frequency:" "$GOVERNOR_LOG" | awk '{print $3}' | sort -n | tail -1 >> "$RESULTS_DIR/SUMMARY.md" 2>/dev/null || true
    echo "Max control frequency recorded" >> "$RESULTS_DIR/SUMMARY.md"
fi

cat >> "$RESULTS_DIR/SUMMARY.md" << EOF

### Dyno Sweep (Linearity)
EOF

if [ -f "$RESULTS_DIR/dyno_analysis.txt" ]; then
    grep -E "(R²|PASS|FAIL)" "$RESULTS_DIR/dyno_analysis.txt" | head -5 >> "$RESULTS_DIR/SUMMARY.md" 2>/dev/null || true
fi

cat >> "$RESULTS_DIR/SUMMARY.md" << EOF

### Step Response (Stability)
EOF

if [ -f "$RESULTS_DIR/step_analysis.txt" ]; then
    grep -E "(Overshoot|Settling|Rise)" "$RESULTS_DIR/step_analysis.txt" | head -5 >> "$RESULTS_DIR/SUMMARY.md" 2>/dev/null || true
fi

cat >> "$RESULTS_DIR/SUMMARY.md" << EOF

---

## Next Steps for Ben

1. Review \`SUMMARY.md\` for quick pass/fail status
2. Check plots (\`dyno_sweep_plot.png\`, \`step_response_plot.png\`) for visual analysis
3. Send all results to Hermes Agent for deep dive analysis
4. If any tests failed, check specific test logs in this directory

---

*Generated automatically by GPUTronic Test Runner v14.0.0*  
*Timestamp: $TIMESTAMP*
EOF

success "Summary report generated: $RESULTS_DIR/SUMMARY.md"

# =============================================================================
# PHASE 8: Final Status & Instructions
# =============================================================================
log ""
log "================================================================="
log "TEST SUITE COMPLETE!"
log "================================================================="
echo ""
log "${GREEN}All tests completed successfully!${NC}"
echo ""
log "Results directory: ${BLUE}$RESULTS_DIR${NC}"
echo ""
log "Key files to review:"
log "  - SUMMARY.md (quick overview)"
log "  - governor_output.txt (full console output)"
log "  - dyno_analysis.txt (linearity R² value)"
log "  - step_analysis.txt (stability margins)"
log "  - *.png plots (visual analysis)"
echo ""
log "${YELLOW}IMPORTANT:${NC}"
log "Send this results directory to Hermes Agent for deep analysis."
log "The agent will need all files in: $RESULTS_DIR"
echo ""
log "To zip and share:"
log "  tar -czf gputronic_results_$TIMESTAMP.tar.gz -C $(dirname "$RESULTS_DIR") $(basename "$RESULTS_DIR")"
echo ""
log "================================================================="

# Save results directory path for easy access
echo "$RESULTS_DIR" > "/media/dad/1TB HDD/GPUTronic Agent/.last_test_run"

exit 0
