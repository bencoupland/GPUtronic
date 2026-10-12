#!/bin/bash
# ============================================================================
# GPUtronic Stage 13 — Blackwell PoC Test Suite v2
# ============================================================================
# 
# USAGE:
#   ./test_gputronic_blackwell_v2.sh [mode]
#
# MODES:
#   sweep     - Throttle sweep for linearity testing (recommended first)
#   inference - Inference workload test
#   gaming    - Gaming/shader workload test  
#   daemon    - Persistent governor mode (default)
#   all       - Run all tests sequentially
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOV_BIN="${SCRIPT_DIR}/gputronic_stage13_poc_blackwell"
LOG_FILE="${SCRIPT_DIR}/gputronic_stage13_poc_blackwell.csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check CUDA installation
    if ! command -v nvcc &> /dev/null; then
        log_error "nvcc not found. Please install CUDA Toolkit."
        exit 1
    fi
    
    # Check GPU availability
    if ! nvidia-smi &> /dev/null; then
        log_error "nvidia-smi failed. NVIDIA driver or GPU not available."
        exit 1
    fi
    
    # Check binary exists
    if [ ! -f "$GOV_BIN" ]; then
        log_error "Governor binary not found: $GOV_BIN"
        log_info "Compile with: nvcc -O3 -arch=sm_90 -std=c++17 gputronic_stage13_poc_blackwell.cu -o gputronic_stage13_poc_blackwell -lcudart"
        exit 1
    fi
    
    # Check for root (required for SCHED_FIFO)
    if [ "$EUID" -ne 0 ]; then
        log_warn "Not running as root. Real-time scheduling may fail."
        log_info "Run with: sudo $0 [mode]"
    fi
    
    log_success "Prerequisites check passed"
}

run_throttle_sweep() {
    log_info "=========================================="
    log_info "THROTTLE SWEEP — Linearity Testing"
    log_info "=========================================="
    log_info "This test measures IPS at each throttle fraction (Q)"
    log_info "Expected: Linear relationship (R² ≈ 1.0)"
    log_info ""
    
    # Clear old logs
    rm -f "$LOG_FILE"
    
    # Run sweep
    log_info "Starting throttle sweep from Q=0.30 to Q=1.00..."
    $GOV_BIN --mode=sweep
    
    log_success "Sweep complete!"
    log_info ""
    log_info "Analyzing results..."
    echo ""
    
    # Display CSV data
    if [ -f "$LOG_FILE" ]; then
        echo "=== THROTTLE SWEEP RESULTS ==="
        echo ""
        cat "$LOG_FILE" | column -t -s ','
        echo ""
        
        # Calculate basic statistics
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
                slope = ips[1]/q[1]
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
    log_info "INFERENCE WORKLOAD TEST"
    log_info "=========================================="
    log_info "Simulates neural network inference (GEMM + ReLU)"
    log_info ""
    
    # Clear old logs
    rm -f "$LOG_FILE"
    
    log_info "Starting governor with inference workload..."
    log_info "Press [Q] to quit after 30 seconds..."
    echo ""
    
    timeout 30 $GOV_BIN --mode=inference || true
    
    log_success "Inference test complete!"
    log_info ""
    log_info "Results saved to: $LOG_FILE"
    log_info ""
    
    # Show summary
    if [ -f "$LOG_FILE" ]; then
        echo "=== INFERENCE WORKLOAD SUMMARY ==="
        echo ""
        tail -20 "$LOG_FILE" | column -t -s ','
        echo ""
        
        # Calculate average IPS and Q
        awk -F',' 'NR>1 {
            sum_ips+=$3; sum_q+=$2; n++
            if(NR==2) baseline=$3
        }
        END {
            avg_ips = sum_ips/n
            avg_q = sum_q/n
            printf "  Average IPS: %.0f\n", avg_ips
            printf "  Average Q:   %.3f\n", avg_q
            printf "  Baseline:    %.0f (at Q=1.0)\n", baseline
            printf "  Efficiency:  %.1f%%\n", (avg_ips/baseline)*100
        }' "$LOG_FILE"
        
        echo ""
    fi
}

run_gaming_test() {
    log_info "=========================================="
    log_info "GAMING WORKLOAD TEST"
    log_info "=========================================="
    log_info "Simulates shader computation with branching"
    log_info ""
    
    # Clear old logs
    rm -f "$LOG_FILE"
    
    log_info "Starting governor with gaming workload..."
    log_info "Press [Q] to quit after 30 seconds..."
    echo ""
    
    timeout 30 $GOV_BIN --mode=gaming || true
    
    log_success "Gaming test complete!"
    log_info ""
    log_info "Results saved to: $LOG_FILE"
    log_info ""
    
    # Show summary
    if [ -f "$LOG_FILE" ]; then
        echo "=== GAMING WORKLOAD SUMMARY ==="
        echo ""
        tail -20 "$LOG_FILE" | column -t -s ','
        echo ""
        
        # Calculate average IPS and Q
        awk -F',' 'NR>1 {
            sum_ips+=$3; sum_q+=$2; n++
            if(NR==2) baseline=$3
        }
        END {
            avg_ips = sum_ips/n
            avg_q = sum_q/n
            printf "  Average IPS: %.0f\n", avg_ips
            printf "  Average Q:   %.3f\n", avg_q
            printf "  Baseline:    %.0f (at Q=1.0)\n", baseline
            printf "  Efficiency:  %.1f%%\n", (avg_ips/baseline)*100
        }' "$LOG_FILE"
        
        echo ""
    fi
}

run_daemon_mode() {
    log_info "=========================================="
    log_info "DAEMON MODE — Persistent Governor"
    log_info "=========================================="
    log_info "Governor runs in background while you do real work"
    log_info ""
    
    # Clear old logs
    rm -f "$LOG_FILE"
    
    log_info "Starting governor daemon..."
    $GOV_BIN --mode=daemon &
    GOV_PID=$!
    
    log_success "Governor running (PID: $GOV_PID)"
    log_info ""
    log_info "Now do your real work in another terminal:"
    log_info "  - Run inference: python train_model.py"
    log_info "  - Play a game: steam / your_game.exe"
    log_info "  - Run custom kernel: ./your_cuda_app"
    log_info ""
    log_info "Press [ENTER] when done to stop governor..."
    read
    
    # Stop governor
    kill $GOV_PID 2>/dev/null || true
    wait $GOV_PID 2>/dev/null || true
    
    log_success "Governor stopped"
    log_info ""
    log_info "Results saved to: $LOG_FILE"
    
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "=== LAST 30 CONTROL CYCLES ==="
        tail -30 "$LOG_FILE" | column -t -s ','
    fi
}

run_all_tests() {
    log_info "=========================================="
    log_info "RUNNING ALL TESTS SEQUENTIALLY"
    log_info "=========================================="
    log_info ""
    
    check_prerequisites
    
    echo ""
    run_throttle_sweep
    echo ""
    sleep 2
    
    echo ""
    run_inference_test
    echo ""
    sleep 2
    
    echo ""
    run_gaming_test
    echo ""
    sleep 2
    
    log_info "=========================================="
    log_success "ALL TESTS COMPLETE"
    log_info "=========================================="
    log_info ""
    log_info "Review results:"
    log_info "  - Throttle sweep: $LOG_FILE (linearity R²)"
    log_info "  - Inference test: $LOG_FILE (IPS, Q efficiency)"
    log_info "  - Gaming test:    $LOG_FILE (IPS, Q efficiency)"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Verify R² > 0.95 for linearity"
    log_info "  2. Compare IPS across workload modes"
    log_info "  3. Tune Kalman parameters if needed (KALMAN_Q_PROC, KALMAN_R_MEAS)"
    log_info "  4. Test with real inference/games in daemon mode"
}

show_help() {
    echo "GPUtronic Stage 13 — Blackwell PoC Testing Script"
    echo ""
    echo "USAGE:"
    echo "  $0 [mode]"
    echo ""
    echo "MODES:"
    echo "  sweep     - Throttle sweep for linearity testing (recommended first)"
    echo "  inference - Inference workload test (GEMM + ReLU simulation)"
    echo "  gaming    - Gaming/shader workload test (branching computation)"
    echo "  daemon    - Persistent governor mode (run real work in background)"
    echo "  all       - Run all tests sequentially"
    echo "  help      - Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  sudo $0 sweep     # Linearity test (requires root for SCHED_FIFO)"
    echo "  sudo $0 inference # Inference workload benchmark"
    echo "  sudo $0 all       # Run complete test suite"
    echo ""
    echo "OUTPUT FILES:"
    echo "  gputronic_stage13_poc_blackwell.csv — Detailed control loop data"
    echo ""
}

# Main entry point
case "${1:-help}" in
    sweep)
        check_prerequisites
        run_throttle_sweep
        ;;
    inference)
        check_prerequisites
        run_inference_test
        ;;
    gaming)
        check_prerequisites
        run_gaming_test
        ;;
    daemon)
        check_prerequisites
        run_daemon_mode
        ;;
    all)
        run_all_tests
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown mode: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
