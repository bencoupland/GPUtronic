#!/bin/bash
# ============================================================================
# GPUtronic Stage 13 — Blackwell PoC v4 Testing Script
# ============================================================================
# USAGE: ./test_gputronic_blackwell_v4.sh [mode]
# MODES: sweep, inference, gaming, gemm, daemon, all
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOV_BIN="${SCRIPT_DIR}/gputronic_stage13_poc_blackwell_v4"
LOG_FILE="${SCRIPT_DIR}/gputronic_v4_blackwell_log.csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v nvcc &> /dev/null; then
        log_error "nvcc not found. Please install CUDA Toolkit."
        exit 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        log_error "nvidia-smi failed. NVIDIA driver or GPU not available."
        exit 1
    fi
    
    if [ ! -f "$GOV_BIN" ]; then
        log_error "Governor v4 binary not found: $GOV_BIN"
        log_info "Compile with:"
        log_info "  nvcc -O3 -arch=sm_90 -std=c++17 gputronic_stage13_poc_blackwell_v4.cu -o gputronic_stage13_poc_blackwell_v4 -lcudart"
        exit 1
    fi
    
    if [ "$EUID" -ne 0 ]; then
        log_warn "Not running as root. Real-time scheduling may fail."
        log_info "Run with: sudo $0 [mode]"
    fi
    
    log_success "Prerequisites check passed"
}

run_throttle_sweep() {
    log_info "=========================================="
    log_info "THROTTLE SWEEP — Linearity Testing (v4)"
    log_info "=========================================="
    log_info "Measures IPS at each throttle fraction (Q=0.10 to Q=1.00)"
    log_info "Expected: Linear relationship (R² ≈ 1.0)"
    log_info ""
    
    rm -f "$LOG_FILE"
    
    log_info "Starting throttle sweep..."
    $GOV_BIN --mode=sweep
    
    log_success "Sweep complete!"
    log_info ""
    log_info "Analyzing results..."
    echo ""
    
    if [ -f "$LOG_FILE" ]; then
        echo "=== THROTTLE SWEEP RESULTS ==="
        echo ""
        column -t -s ',' "$LOG_FILE" | head -15
        echo "... (see full CSV for all data points)"
        echo ""
        
        log_info "Linearity Analysis:"
        awk -F',' 'NR>1 {
            q[NR-1]=$1; ips[NR-1]=$2; n++
            sum_q+=$1; sum_ips+=$2
        }
        END {
            mean_q=sum_q/n; mean_ips=sum_ips/n
            
            # Calculate R² (assuming proportional relationship)
            ss_tot=0; ss_res=0
            for(i=1;i<=n;i++) {
                ss_tot += (q[i]-mean_q)^2
                slope = ips[1]/q[1]  # Proportional model: IPS = slope * Q
                pred = slope * q[i]
                ss_res += (ips[i]-pred)^2
            }
            r2 = 1 - (ss_res/ss_tot)
            
            printf "  R² (linearity): %.4f\n", r2
            printf "  Points tested: %d\n", n
            
            if(r2 > 0.95) {
                print "  ✓ EXCELLENT linearity — governor responds predictably"
            } else if(r2 > 0.85) {
                print "  ✓ GOOD linearity — minor non-linearities detected"
            } else {
                print "  ⚠ MODERATE linearity — investigate non-linear regions"
            }
        }' "$LOG_FILE"
        
        echo ""
    fi
    
    log_info "CSV data saved to: $LOG_FILE"
}

run_inference_test() {
    log_info "=========================================="
    log_info "INFERENCE WORKLOAD TEST (v4)"
    log_info "=========================================="
    log_info "Simulates sparse neural network inference"
    log_info "Tests Kalman observer with noisy measurements"
    log_info ""
    
    rm -f "$LOG_FILE"
    
    log_info "Starting inference test (30 seconds)..."
    timeout 35 $GOV_BIN --mode=inference || true
    
    log_success "Test complete!"
    analyze_results "Inference"
}

run_gaming_test() {
    log_info "=========================================="
    log_info "GAMING WORKLOAD TEST (v4)"
    log_info "=========================================="
    log_info "Simulates shader-like computation with branching"
    log_info "Tests governor under irregular memory access patterns"
    log_info ""
    
    rm -f "$LOG_FILE"
    
    log_info "Starting gaming test (30 seconds)..."
    timeout 35 $GOV_BIN --mode=gaming || true
    
    log_success "Test complete!"
    analyze_results "Gaming"
}

run_gemm_test() {
    log_info "=========================================="
    log_info "CUSTOM GEMM WORKLOAD TEST (v4)"
    log_info "=========================================="
    log_info "Tests with 32x32 matrix multiply"
    log_info "Extension point for real inference integration"
    log_info ""
    
    rm -f "$LOG_FILE"
    
    log_info "Starting GEMM test (30 seconds)..."
    timeout 35 $GOV_BIN --mode=gemm || true
    
    log_success "Test complete!"
    analyze_results "GEMM"
}

run_daemon_mode() {
    log_info "=========================================="
    log_info "DAEMON MODE — Persistent Governor (v4)"
    log_info "=========================================="
    log_info "Starts governor in background for real workloads"
    log_info ""
    log_info "Usage:"
    log_info "  Terminal 1: $0 daemon"
    log_info "  Terminal 2: Run your workload (e.g., python train.py)"
    log_info "  Back to T1: Press 'q' to stop governor"
    log_info ""
    
    rm -f "$LOG_FILE"
    
    log_info "Starting governor in daemon mode..."
    $GOV_BIN --mode=daemon
}

analyze_results() {
    local test_name=$1
    
    if [ ! -f "$LOG_FILE" ]; then
        log_warn "No CSV data found — test may have failed"
        return
    fi
    
    echo ""
    log_info "${test_name} Test Analysis:"
    echo ""
    
    # Control loop timing analysis
    log_info "Control Loop Timing:"
    awk -F',' 'NR>1 && NF>=8 {
        cycle_time = $8
        sum += cycle_time
        count++
        if (cycle_time > max) max = cycle_time
        times[count] = cycle_time
    }
    END {
        if (count == 0) { print "  No timing data available"; exit }
        
        mean = sum / count
        
        # Sort for median/p95
        n = asort(times)
        median = times[int(n/2)]
        p95_idx = int(n * 0.95)
        if (p95_idx < 1) p95_idx = 1
        p95 = times[p95_idx]
        
        printf "  Median cycle time: %.2f µs (%.1f kHz)\n", median, 1000/median
        printf "  95th percentile:   %.2f µs\n", p95
        printf "  Max cycle time:    %.2f µs\n", max
        
        if (median < 10.0) {
            print "  ✓ PASS: Achieving >100kHz control frequency"
        } else {
            print "  ⚠ WARNING: Control frequency below target (100kHz)"
        }
    }' "$LOG_FILE"
    
    echo ""
    
    # Phase margin stability analysis
    log_info "Phase Margin Stability:"
    awk -F',' 'NR>1 && NF>=6 {
        pm = $6
        if (pm > 0 && pm < 10) {  # Filter outliers
            sum += pm
            count++
            if (pm < min || min == 0) min = pm
            if (pm > max) max = pm
            pms[count] = pm
        }
    }
    END {
        if (count == 0) { print "  No PM data available"; exit }
        
        mean = sum / count
        
        # Sort for median
        n = asort(pms)
        median = pms[int(n/2)]
        
        printf "  Mean PM:   %.3f\n", mean
        printf "  Median PM: %.3f\n", median
        printf "  Min PM:    %.3f\n", min
        printf "  Max PM:    %.3f\n", max
        
        # Check if PM stays within stable range [1.0, 1.5]
        in_range = 0
        for (i=1; i<=n; i++) {
            if (pms[i] >= 1.0 && pms[i] <= 1.5) in_range++
        }
        pct_in_range = 100.0 * in_range / n
        
        printf "  In stable range [1.0, 1.5]: %.1f%%\n", pct_in_range
        
        if (pct_in_range > 90) {
            print "  ✓ PASS: Phase margin stable"
        } else if (pct_in_range > 70) {
            print "  ⚠ MODERATE: Some PM excursions outside stable range"
        } else {
            print "  ⚠ WARNING: Phase margin unstable — tune PI gains or deadband"
        }
    }' "$LOG_FILE"
    
    echo ""
    
    # Throttle stability analysis
    log_info "Throttle (Q) Stability:"
    awk -F',' 'NR>1 && NF>=2 {
        q = $2
        if (q >= 0.1 && q <= 1.0) {
            sum += q
            sumsq += q*q
            count++
            if (q < min || min == 0) min = q
            if (q > max) max = q
        }
    }
    END {
        if (count == 0) { print "  No Q data available"; exit }
        
        mean = sum / count
        variance = (sumsq / count) - (mean * mean)
        stddev = sqrt(variance > 0 ? variance : 0)
        
        printf "  Mean Q:    %.3f\n", mean
        printf "  Std Dev:   %.3f\n", stddev
        printf "  Min Q:     %.3f\n", min
        printf "  Max Q:     %.3f\n", max
        
        if (stddev < 0.1) {
            print "  ✓ PASS: Throttle stable (low variance)"
        } else if (stddev < 0.2) {
            print "  ⚠ MODERATE: Some throttle variation"
        } else {
            print "  ⚠ WARNING: High throttle variance — may indicate instability"
        }
    }' "$LOG_FILE"
    
    echo ""
    log_info "Full CSV data saved to: $LOG_FILE"
}

run_all_tests() {
    log_info "=========================================="
    log_info "RUNNING ALL TESTS SEQUENTIALLY (v4)"
    log_info "=========================================="
    log_info "This will take ~5-10 minutes total"
    log_info ""
    
    echo -n "Press ENTER to continue, or Ctrl+C to cancel..."
    read
    
    run_throttle_sweep
    echo ""
    sleep 2
    
    run_inference_test
    echo ""
    sleep 2
    
    run_gaming_test
    echo ""
    sleep 2
    
    run_gemm_test
    
    log_success "=========================================="
    log_success "ALL TESTS COMPLETE!"
    log_success "=========================================="
}

# ============================================================================
# MAIN
# ============================================================================

check_prerequisites

MODE="${1:-daemon}"

case "$MODE" in
    sweep)
        run_throttle_sweep
        ;;
    inference)
        run_inference_test
        ;;
    gaming)
        run_gaming_test
        ;;
    gemm)
        run_gemm_test
        ;;
    daemon)
        run_daemon_mode
        ;;
    all)
        run_all_tests
        ;;
    *)
        echo "Usage: $0 [sweep|inference|gaming|gemm|daemon|all]"
        echo ""
        echo "Modes:"
        echo "  sweep     - Throttle sweep for linearity testing (recommended first)"
        echo "  inference - Inference workload test"
        echo "  gaming    - Gaming/shader workload test"
        echo "  gemm      - Custom GEMM workload test"
        echo "  daemon    - Persistent governor mode (default)"
        echo "  all       - Run all tests sequentially"
        exit 1
        ;;
esac
