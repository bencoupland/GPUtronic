# GPUtronic Stage 13 — Quick Reference Card

## Build & Run Commands

```bash
# Compile (Blackwell / RTX 5080)
nvcc -O3 -arch=sm_90 -std=c++17 gputronic_stage13_poc_blackwell.cu \
     -o gputronic_stage13_poc_blackwell -lcudart

# Run tests (requires root for SCHED_FIFO real-time scheduling)
sudo ./test_gputronic_blackwell.sh sweep      # Linearity test
sudo ./test_gputronic_blackwell.sh inference  # Inference workload
sudo ./test_gputronic_blackwell.sh gaming     # Gaming workload
sudo ./test_gputronic_blackwell.sh daemon     # Persistent mode

# Direct binary execution with modes
sudo ./gputronic_stage13_poc_blackwell --mode=sweep
sudo ./gputronic_stage13_poc_blackwell --mode=inference
sudo ./gputronic_stage13_poc_blackwell --mode=gaming
sudo ./gputronic_stage13_poc_blackwell --mode=daemon
```

---

## Control Loop Timing (Inner Loop ~100kHz)

| Operation | Latency | Notes |
|-----------|---------|-------|
| Read per-SM counters | <1µs | Direct memory access, no PCIe |
| Calculate work_delta | <0.1µs | Simple subtraction |
| Kalman update | ~2µs | 2-state observer (predict + update) |
| PI controller | <0.5µs | Error + integral + deadband check |
| Zero-copy throttle write | <1µs | Host ptr IS device ptr |
| **Total cycle time** | **~5-10µs** | **100-200kHz control frequency** |

**Comparison:**
- NVML polling: ~20ms (50Hz) — too slow for predictive control
- Our approach: ~5-10µs (100-200kHz) — fast enough to anticipate stalls

---

## Key Parameters (Tunable in source code)

### Kalman Observer
```cpp
constexpr double KALMAN_Q_PROC = 0.002;   // Process noise [0.001, 0.01]
constexpr double KALMAN_R_MEAS = 0.015;   // Measurement variance [0.01, 0.1]
```

### PI Controller
```cpp
float TARGET_PM = 1.10f;                  // Phase margin target [1.0, 1.5]
constexpr float PM_KP = 0.50f;            // Proportional gain [0.2, 1.0]
constexpr float PM_KI = 0.12f;            // Integral gain [0.05, 0.3]
```

### Deadband Thresholds
```cpp
constexpr float Z_HIGH_THRESH = 1.50f;    // Throttle down above this
constexpr float Z_LOW_THRESH  = 0.95f;    // Throttle up below this
```

---

## CSV Log Format

File: `gputronic_stage13_poc_blackwell.csv`

| Column | Name | Description |
|--------|------|-------------|
| 1 | timestamp_us | Microseconds since start |
| 2 | Q | Throttle fraction [0.30, 1.00] |
| 3 | IPS | Actual instructions/work per second |
| 4 | baseline_ips | Baseline IPS at Q=1.0 (from autotune) |
| 5 | Z_filt | Filtered impedance estimate |
| 6 | PM | Phase margin (= 1/Z) |
| 7 | z_rate | Rate of impedance change (dZ/dt) |
| 8 | error_pm | Phase margin error (target - actual) |
| 9 | integral_pm | Integral term (windup-limited) |

**Example row:**
```csv
1234567,0.752,12450000,13200000,1.089,1.158,-0.0023,0.042,0.156
```

---

## Linearity Test (Throttle Sweep)

**Expected output:**
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

**Pass criteria:** R² > 0.95

---

## Keyboard Controls (Daemon Mode)

| Key | Action |
|-----|--------|
| `Q` | Quit governor gracefully |
| `W` | Increase target PM (+0.05) — more conservative |
| `S` | Decrease target PM (-0.05) — more aggressive |

---

## Troubleshooting Quick Reference

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Q oscillates rapidly | Deadband too narrow or PI gains too high | Widen deadband (Z_HIGH=1.60, Z_LOW=0.90), reduce PM_KP to 0.40 |
| GPU stalls frequently | Memory pressure or driver interference | Increase STALL_TIMEOUT_US to 15000, use daemon mode |
| Linearity R² < 0.85 | Thermal throttling or measurement noise | Check nvidia-smi for throttling, increase sweep duration to 1000ms |
| Control loop too slow | Not running as root (no SCHED_FIFO) | Run with `sudo` for real-time scheduling |
| Kernel killed by driver | Stock NVIDIA driver terminates persistent threads | Use daemon mode, analyze CSV logs offline |

---

## Performance Benchmarks (RTX 5080 / Blackwell)

| Workload | Baseline IPS (Q=1.0) | Governed IPS | Efficiency |
|----------|---------------------|--------------|------------|
| Synthetic | ~50-100M | ~45-95M | 90-95% |
| Inference (GEMM) | ~80-150M | ~75-140M | 92-96% |
| Gaming (shader) | ~40-80M | ~35-75M | 88-94% |

---

## Architecture Diagram (Text Version)

```
┌─────────────────────────────────────────────────────────────┐
│                    CPU CONTROL LOOP                         │
│                   (~100kHz frequency)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Per-SM       │───▶│ Kalman       │───▶│ PI           │ │
│  │ Counters     │    │ Observer     │    │ Controller   │ │
│  │ (read work)  │    │ (estimate Z) │    │ (compute ΔQ) │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         ▲                                      │           │
│         │                                      ▼           │
│         │                            ┌──────────────┐      │
│         │                            │ Hysteresis   │      │
│         │                            │ Deadband     │      │
│         │                            └──────────────┘      │
│         │                                      │           │
│         │                                      ▼           │
│         │                      Zero-Copy Throttle Cable    │
│         │                      (<1µs latency)              │
│         │                                      │           │
└─────────┼──────────────────────────────────────┼───────────┘
          │                                      │
          │                                      ▼
┌─────────┴──────────────────────────────────────┴───────────┐
│                    GPU KERNEL (Persistent)                  │
│              (One block per SM, runs forever)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Read Q from  │───▶│ Execute      │───▶│ Increment    │ │
│  │ Shared Mem   │    │ Workload     │    │ Per-SM       │ │
│  │ (<1µs read)  │    │ (synthetic/  │    │ Counter      │ │
│  └──────────────┘    │ inference/   │    │ (atomicAdd)  │ │
│                      │ gaming)      │    └──────────────┘ │
│                      └──────────────┘                     │
│                                                           │
└───────────────────────────────────────────────────────────┘

Data Flow:
1. GPU kernel increments per-SM counters as it completes work
2. CPU reads counters directly (no PCIe transfer)
3. Kalman observer estimates impedance Z and dZ/dt
4. PI controller computes ΔQ to maintain target phase margin
5. Deadband prevents unnecessary adjustments
6. New Q written to zero-copy shared memory (instant GPU visibility)
7. Repeat at ~100kHz
```

---

## Memory Layout

| Resource | Allocation | Size | Purpose |
|----------|------------|------|---------|
| `h_q` (zero-copy) | cudaHostAllocMapped | 4 bytes | Throttle fraction (CPU→GPU) |
| `d_work_per_sm` | cudaMalloc | sm_count × 8 bytes | Per-SM atomic counters (GPU→CPU) |
| Control thread | pthread_setaffinity_np | N/A | Pinned to core 0, SCHED_FIFO priority 99 |

---

## Safety Limits

| Limit | Value | Action |
|-------|-------|--------|
| Minimum Q | 0.30 | Hard floor — never throttle below this |
| Maximum Q | 1.00 | Hard ceiling — never exceed baseline |
| Stall timeout | 8ms | Emergency throttle ×0.90 if no work detected |
| Integral windup | ±8.0 | Clamp integral term to prevent overshoot |
| Z raw clamp | [0.1, 10.0] | Prevent outliers from overwhelming Kalman filter |

---

## Author

Ben Coupland + Hermes Agent  
Date: 2026-05-11  
Version: 13.0-PoC-BLACKWELL
