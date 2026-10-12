# GPUtronic Stage 13 — Blackwell PoC Test Plan

## Overview

This document outlines the complete testing procedure for validating the GPUtronic Stage 13 Proof-of-Concept on NVIDIA Blackwell hardware (RTX 5080).

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 5080 (Blackwell architecture, compute capability 9.0)
- **CPU**: Dual Xeon E5-2696v4 (or equivalent with dedicated core for control thread)
- **RAM**: ≥16GB system memory
- **VRAM**: ≥16GB (RTX 5080 spec)

### Software Requirements
- **OS**: Linux (kernel 6.17+ recommended) or Windows 10/11
- **CUDA Toolkit**: ≥12.0 (for sm_90 compilation)
- **NVIDIA Driver**: ≥550.00 (Blackwell support)
- **Root access**: Required for SCHED_FIFO real-time scheduling (Linux)

### Build Tools
```bash
# Verify CUDA installation
nvcc --version
# Expected: CUDA version 12.x, target architecture sm_90

# Verify GPU detection
nvidia-smi
# Expected: RTX 5080 listed with driver version ≥550.00

# Verify compilation
nvcc -O3 -arch=sm_90 -std=c++17 gputronic_stage13_poc_blackwell.cu \
     -o gputronic_stage13_poc_blackwell -lcudart
# Expected: Successful compilation (warnings OK)
```

---

## Test Suite

### Test 1: Throttle Sweep (Linearity Verification)

**Objective**: Verify that GPU throughput responds linearly to throttle fraction adjustments.

**Command**:
```bash
sudo ./test_gputronic_blackwell.sh sweep
# or directly:
sudo ./gputronic_stage13_poc_blackwell --mode=sweep
```

**Procedure**:
1. Governor sweeps Q from 0.30 to 1.00 in 10 steps (step size = 0.07)
2. At each step, measures IPS for 500ms
3. Calculates R² linearity metric
4. Logs data to `gputronic_stage13_poc_blackwell.csv`

**Expected Results**:
```
=== THROTTLE SWEEP RESULTS ===

Q         IPS          Z_filt  PM    z_rate
0.30      3750000      1.0     1.0   0.0
0.40      5000000      1.0     1.0   0.0
0.50      6250000      1.0     1.0   0.0
...
1.00      12500000     1.0     1.0   0.0

Linearity R²: 0.9876 (1.0 = perfectly linear)
✓ EXCELLENT linearity — governor responds predictably
```

**Pass Criteria**:
- **R² > 0.95**: PASS — Excellent linearity
- **R² > 0.85**: CONDITIONAL PASS — Minor non-linearities (investigate if <0.90)
- **R² < 0.85**: FAIL — Investigate thermal throttling, measurement noise, or kernel occupancy issues

**Failure Analysis**:
| R² Value | Likely Cause | Diagnostic Step |
|----------|--------------|-----------------|
| 0.70-0.85 | Thermal throttling at high Q | Run `nvidia-smi -l 1` during sweep, check for clock throttling |
| 0.60-0.70 | Measurement noise (too few samples) | Increase sweep duration from 500ms to 1000ms in source code |
| <0.60 | Kernel not fully utilizing SMs at low Q | Check occupancy with `nvprof --print-gpu-trace` |

**Duration**: ~8-10 minutes (10 steps × 500ms + overhead)

---

### Test 2: Inference Workload (GEMM + ReLU Simulation)

**Objective**: Verify governor stability and efficiency with neural network inference workloads.

**Command**:
```bash
sudo ./test_gputronic_blackwell.sh inference
# or directly:
timeout 30 sudo ./gputronic_stage13_poc_blackwell --mode=inference
```

**Procedure**:
1. Governor runs for 30 seconds with GEMM + ReLU workload
2. Starts at Q=0.75 (conservative), ramps up adaptively
3. Kalman observer estimates impedance, PI controller maintains phase margin
4. Logs detailed control loop data every ~50µs

**Expected Results**:
```
=== INFERENCE WORKLOAD SUMMARY ===

timestamp_us  Q       IPS        baseline_ips  Z_filt  PM    z_rate
1234567       0.752   12450000   13200000      1.089   1.158 -0.0023
1234617       0.754   12560000   13200000      1.086   1.161 -0.0021
...

Average IPS: 12,450,000
Average Q:   0.752
Baseline:    13,200,000 (at Q=1.0)
Efficiency:  94.3%
```

**Pass Criteria**:
- **Stability**: No oscillation in Q (should vary by <±0.05 after initial ramp-up)
- **Efficiency**: Achieves ≥85% of baseline IPS
- **Phase margin**: PM stays within [1.0, 1.3] after initial transient
- **No stalls**: work_delta > 0 in every control cycle (or <1% stall cycles)

**Failure Analysis**:
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Q oscillates ±0.2+ | PI gains too high or deadband too narrow | Reduce PM_KP to 0.40, widen deadband (Z_HIGH=1.60, Z_LOW=0.90) |
| Efficiency <80% | Kalman filter too aggressive (over-throttling) | Increase KALMAN_R_MEAS to 0.03, trust model more than measurements |
| Frequent stalls | Memory pressure or driver interference | Increase STALL_TIMEOUT_US to 15000, use daemon mode for production |

**Duration**: 30 seconds

---

### Test 3: Gaming Workload (Shader Simulation)

**Objective**: Verify governor handles branching-heavy workloads with irregular memory access patterns.

**Command**:
```bash
sudo ./test_gputronic_blackwell.sh gaming
# or directly:
timeout 30 sudo ./gputronic_stage13_poc_blackwell --mode=gaming
```

**Procedure**:
1. Governor runs for 30 seconds with shader-like workload (branching + texture lookups)
2. Tests Kalman filter's ability to smooth noisy impedance measurements
3. Verifies phase margin controller maintains stability despite Z-axis spikes

**Expected Results**:
```
=== GAMING WORKLOAD SUMMARY ===

timestamp_us  Q       IPS        baseline_ips  Z_filt  PM    z_rate
1234567       0.751   6250000    7100000       1.134   1.129 +0.0045
1234617       0.748   6180000    7100000       1.156   1.102 +0.0067
...

Average IPS: 6,200,000
Average Q:   0.749
Baseline:    7,100,000 (at Q=1.0)
Efficiency:  87.3%
```

**Pass Criteria**:
- **Stability**: No oscillation in Q (branching causes Z noise, but Kalman should filter it)
- **Noise rejection**: Compare `z_raw` vs `z_filt` — filtered values should be smoother
- **Efficiency**: Achieves ≥80% of baseline IPS (lower than inference due to branching overhead)

**Failure Analysis**:
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Z_filt tracks z_raw too closely | Kalman filter not smoothing enough | Increase KALMAN_R_MEAS to 0.05, trust model more |
| Q chases noise (jittery) | Deadband too narrow for noisy workload | Widen deadband (Z_HIGH=1.60, Z_LOW=0.85) |
| Efficiency <75% | Over-throttling due to false stall detection | Increase STALL_TIMEOUT_US to 20000 (20ms) |

**Duration**: 30 seconds

---

### Test 4: Daemon Mode (Real Workloads)

**Objective**: Verify governor operates correctly in background while user runs actual applications.

**Command**:
```bash
sudo ./test_gputronic_blackwell.sh daemon
```

**Procedure**:
1. Governor starts in background mode
2. User runs real workload in another terminal:
   - Inference: `python train_model.py --batch-size=32`
   - Gaming: Steam game or emulator
   - Custom kernel: `./your_cuda_app`
3. Governor logs control loop data throughout
4. User stops governor when done

**Expected Results**:
```
=== LAST 30 CONTROL CYCLES ===

timestamp_us  Q       IPS        baseline_ips  Z_filt  PM    z_rate
...
123456789     0.753   11200000   13200000      1.178   1.089 +0.0012
123461789     0.755   11350000   13200000      1.165   1.102 +0.0008
123466789     0.756   11420000   13200000      1.159   1.108 +0.0005

[OK] Engine cold. System safe.
[LOG] CSV data saved to: gputronic_stage13_poc_blackwell.csv
```

**Pass Criteria**:
- **Background operation**: Governor runs without interfering with user workflow
- **Adaptive control**: Q adjusts based on workload characteristics (inference vs gaming)
- **No crashes**: Governor survives 5+ minutes of continuous operation
- **Log integrity**: CSV file contains complete control loop data (no gaps or corruption)

**Duration**: User-defined (typically 5-30 minutes for real workloads)

---

## Performance Validation Matrix

| Test | Metric | Target | Acceptable | Critical Failure |
|------|--------|--------|------------|------------------|
| **Throttle Sweep** | R² linearity | >0.95 | >0.85 | <0.70 |
| **Inference** | Efficiency (IPS/baseline) | ≥92% | ≥85% | <75% |
| **Inference** | Q stability (std dev) | <0.03 | <0.05 | >0.10 |
| **Gaming** | Efficiency (IPS/baseline) | ≥88% | ≥80% | <70% |
| **Gaming** | Z noise rejection (z_filt vs z_raw) | Smooth | Moderate | Chases noise |
| **Daemon** | Uptime | >30 min | >10 min | <5 min |
| **All Tests** | Stall cycles | 0% | <1% | >5% |

---

## Control Loop Timing Validation

### Objective: Verify inner loop achieves ~100kHz frequency

**Method**: Analyze CSV timestamps for cycle-to-cycle delta

```bash
# Extract timestamp column and calculate deltas
awk -F',' 'NR>1 {print $1}' gputronic_stage13_poc_blackwell.csv | \
  awk '{if(NR>1) print $1-prev; prev=$1}' | \
  sort -n | \
  head -10

# Expected output: ~5-10µs deltas (5,000-10,000 nanoseconds)
```

**Pass Criteria**:
- **Median cycle time**: <10µs (100kHz+)
- **95th percentile**: <20µs (no long stalls in control loop)
- **Maximum**: <100µs (emergency stall timeout catches anything worse)

---

## Zero-Copy Latency Validation

### Objective: Verify throttle cable achieves <1µs latency

**Method**: Add timing instrumentation to source code (optional, for advanced testing)

```cpp
// In control loop, measure time from *h_q write to kernel visibility
auto t0 = std::chrono::high_resolution_clock::now();
*h_q = current_q;  // Write new throttle value

// Kernel reads this on next iteration — add instrumentation there
__global__ void g_scag_engine(float* d_q, ...) {
    auto t1 = clock64();  // GPU cycle counter
    float q = *d_q;       // Read throttle value
    auto t2 = clock64();
    
    // Log (t2 - t1) to shared memory for CPU readout
}

// Expected: <30 GPU cycles (~10ns at 3GHz), well under 1µs
```

**Pass Criteria**: Latency <1µs (verified by design — zero-copy mapped memory guarantees this)

---

## Kalman Observer Validation

### Objective: Verify impedance estimation and predictive capability

**Method**: Compare filtered vs. raw measurements, check dZ/dt predictions

```bash
# Plot z_raw vs z_filt from CSV (Python example)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('gputronic_stage13_poc_blackwell.csv')
plt.plot(df['timestamp_us'], df['z_raw'], label='Raw', alpha=0.5)
plt.plot(df['timestamp_us'], df['Z_filt'], label='Filtered', linewidth=2)
plt.legend()
plt.show()

# Expected: z_filt smoother than z_raw, tracks trends with ~1-2 cycle lag
```

**Pass Criteria**:
- **Smoothing**: z_filt variance < 0.5 × z_raw variance (for noisy workloads like gaming)
- **Predictive capability**: When dZ/dt > +0.1/sec, Q should start decreasing before Z reaches threshold
- **Phase margin accuracy**: PM = 1/Z_filt should correlate with actual stability margins

---

## Stress Testing

### Test: Extended Runtime (24-hour soak test)

**Objective**: Verify governor stability over extended operation

**Command**:
```bash
# Start governor in daemon mode, run for 24 hours
sudo ./gputronic_stage13_poc_blackwell --mode=daemon &
GOV_PID=$!

# Run continuous workload (e.g., inference loop)
while true; do
    python train_model.py --epochs=1
done &

# After 24 hours, stop and analyze
sleep 86400
kill $GOV_PID

# Check for crashes, memory leaks, or performance degradation
tail -100 gputronic_stage13_poc_blackwell.csv | \
  awk -F',' '{sum_q+=$2; sum_ips+=$3; n++} END {
    printf "Avg Q: %.3f\nAvg IPS: %.0f\n", sum_q/n, sum_ips/n
  }'
```

**Pass Criteria**:
- **No crashes**: Governor survives 24 hours without segfault or CUDA error
- **No memory leaks**: RSS stable (check with `ps -o rss= -p $GOV_PID`)
- **Consistent performance**: Avg Q and IPS within ±5% of initial values

---

## Regression Testing

### After any code change, run:

```bash
# Quick sanity check (2 minutes)
sudo ./gputronic_stage13_poc_blackwell --mode=sweep | grep "R²"

# Full test suite (~2 hours for 24-hour soak)
sudo ./test_gputronic_blackwell.sh all    # Basic tests: ~2 minutes
# Optional: Add 24-hour soak test for production releases
```

**Regression criteria**: All metrics must match or exceed previous version within ±5% tolerance.

---

## Success Criteria Summary

The GPUtronic Stage 13 PoC is considered **production-ready** when:

1. ✅ Throttle sweep R² > 0.95 (linearity verified)
2. ✅ Inference efficiency ≥85% with Q stability <±0.05
3. ✅ Gaming efficiency ≥80% with Kalman noise rejection working
4. ✅ Daemon mode survives 30+ minutes without crashes
5. ✅ Control loop achieves median cycle time <10µs (100kHz+)
6. ✅ Zero-copy throttle latency <1µs (verified by design)
7. ✅ No stall cycles >1% in any workload

---

## Author

Ben Coupland + Hermes Agent  
Date: 2026-05-11  
Version: 13.0-PoC-BLACKWELL
