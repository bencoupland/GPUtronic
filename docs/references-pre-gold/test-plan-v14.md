# GPUTronic Stage 13 SCAG v14.0.0 - Comprehensive Test Plan

**Document Version:** 1.0  
**Date:** 2026-05-14  
**Target Hardware:** NVIDIA Blackwell (RTX 5080) with stock drivers  
**Author:** GPUTronic Architect (for Ben Coupland)

---

## Executive Summary

This test plan validates the GPUTronic Stage 13 Self-Calibrating Adaptive Governor (SCAG) v14.0.0 implementation across five critical dimensions:

1. **Throughput Viability** — Prove realistic throughput on Blackwell hardware
2. **Linearity** — Verify thread fraction vs. throughput relationship is predictable
3. **Stability Margins** — Measure phase/gain margins via step and frequency response
4. **Real Workload Performance** — Test with inference-like workloads (GEMM + Transformer)
5. **Thermal Robustness** — Long-duration stress testing under thermal constraints

### Success Criteria

| Metric | Target | Acceptable | Critical Failure |
|--------|--------|------------|------------------|
| Control frequency | ≥100kHz | 50-100kHz | <20kHz |
| Phase margin | 60° ± 15° | 45-75° | <30° or >90° |
| Step response overshoot | <15% | 15-30% | >30% |
| Settling time | <100ms | 100-200ms | >500ms |
| Throughput linearity (R²) | ≥0.95 | 0.90-0.95 | <0.85 |
| Thermal stability | No throttling | Minor throttling | Sustained thermal limit |

---

## Prerequisites & Setup

### Hardware Requirements

```
GPU: NVIDIA RTX 5080 (Blackwell) or equivalent
CPU: Dual Xeon E5-2696v4 (or similar multi-core)
RAM: ≥32GB system memory
Storage: NVMe SSD for low-latency I/O
Cooling: Adequate case airflow (target <75°C GPU under load)
```

### Software Requirements

```bash
# NVIDIA drivers (latest stable, stock — no custom mods)
nvidia-smi --query-gpu=driver_version --format=csv

# CUDA toolkit 12.x+
nvcc --version

# NVML library (included with drivers)
ldconfig -p | grep nvml

# Build tools
g++ --version
make --version
```

### Environment Configuration

```bash
# Set CPU governor to performance mode (disable frequency scaling)
sudo cpupower frequency-set -g performance

# Isolate CPU cores for control loop (optional, reduces jitter)
isolcpus=4,5,6,7  # Add to kernel boot parameters

# Disable NUMA balancing (reduces memory latency variance)
echo 0 > /proc/sys/vm/numa_balancing

# Set GPU power mode to maximum performance
nvidia-smi -pm 1
nvidia-smi -pl 320  # Set power limit (Watts, adjust for your card)
```

### Build Instructions

```bash
# Navigate to project directory
cd "/media/dad/1TB HDD/GPUTronic Agent"

# Compile main governor implementation
nvcc -O3 -arch=sm_120 \
     -o gputronic_v14 gputronic_stage13_scag_v1400.cu \
     -lnvml \
     -pthread \
     -lm

# Compile benchmark harness
nvcc -O3 -arch=sm_120 \
     -o gputronic_benchmark gputronic_benchmark_harness.cu \
     -lnvml \
     -pthread \
     -lm

# Compile inference workload (optional, for real workload testing)
nvcc -O3 -arch=sm_120 \
     -o gputronic_inference gputronic_inference_workload.cu \
     -pthread \
     -lm

# Verify compilation
./gputronic_v14 --help  # Should print usage info
```

---

## Test Suite Overview

| Test ID | Name | Duration | Purpose | Pass Criteria |
|---------|------|----------|---------|---------------|
| **T1** | Dyno Sweep | 5 min | Thread fraction vs. throughput linearity | R² ≥ 0.95 |
| **T2** | Step Response | 2 min | Stability margins (overshoot, settling) | Overshoot <15%, Settling <100ms |
| **T3** | Frequency Response | 10 min | Bode plot, phase/gain margin | PM ≥45°, GM ≥6dB |
| **T4** | Load Variation | 3 min | Idle ↔ Full load transitions | No instability, fast recovery |
| **T5** | Thermal Stress | 30+ min | Long-duration high-load stability | No thermal throttling |
| **T6** | Inference Workload | 10 min | Real workload (GEMM + Transformer) | Stable throughput, no crashes |
| **T7** | Governor Comparison | 15 min | vs. stock NVIDIA governor | ≥10% improvement in stability |

---

## Test T1: Dyno Sweep - Thread Fraction vs. Throughput Linearity

### Objective

Establish the "plant transfer function" by measuring static gain K and verifying linear relationship between thread fraction q and throughput. This validates that our throttle actuator behaves predictably.

### Procedure

```bash
# Run dyno sweep test (Test 1 of benchmark harness)
./gputronic_benchmark 1 80 256 > results/dyno_sweep_$(date +%Y%m%d_%H%M%S).txt

# Or run all tests
./gputronic_benchmark 0 80 256 > results/all_tests_$(date +%Y%m%d_%H%M%S).txt
```

### Expected Output

```
=================================================================
  TEST 1: DYNO SWEEP - Thread Fraction vs. Throughput
=================================================================

[Dyno] GPU: NVIDIA GeForce RTX 5080 (80 SMs)
[Dyno] Measuring throughput at various thread fractions...

q_frac   | Work Units    | Throughput (M/s) | Efficiency
---------|--------------|-----------------|-----------------
q=0.1     | 1234567       | 123.45          | 98.2%
q=0.2     | 2456789       | 245.67          | 97.8%
...
q=1.0     | 12345678      | 1234.56         | 95.1%

[Dyno] Dyno sweep complete
```

### Analysis Script (Python)

```python
#!/usr/bin/env python3
"""Analyze dyno sweep results for linearity (R² calculation)"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Parse output file
q_values = []
throughput_values = []

with open('results/dyno_sweep_20260514_183000.txt', 'r') as f:
    for line in f:
        if line.startswith('q='):
            parts = line.strip().split('|')
            q = float(parts[0].strip().split('=')[1])
            throughput = float(parts[2].strip())
            q_values.append(q)
            throughput_values.append(throughput)

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(q_values, throughput_values)

print(f"Linear fit: throughput = {slope:.2f} × q + {intercept:.2f}")
print(f"R² (coefficient of determination): {r_value**2:.4f}")
print(f"P-value: {p_value:.6e}")
print(f"Standard error: {std_err:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(q_values, throughput_values, label='Measured', s=50)
plt.plot(q_values, slope*np.array(q_values) + intercept, 'r-', 
         label=f'Linear fit (R²={r_value**2:.3f})')
plt.xlabel('Thread Fraction q')
plt.ylabel('Throughput (M ops/sec)')
plt.title('Dyno Sweep: Thread Fraction vs. Throughput')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('results/dyno_sweep_plot.png', dpi=150)
plt.show()

# Pass/fail decision
if r_value**2 >= 0.95:
    print("\n✓ PASS: Linearity R² ≥ 0.95")
elif r_value**2 >= 0.90:
    print("\n⚠ WARNING: Linearity R² between 0.90-0.95")
else:
    print("\n✗ FAIL: Linearity R² < 0.90 (non-linear actuator)")
```

### Pass/Fail Criteria

| Metric | Target | Acceptable | Fail |
|--------|--------|------------|------|
| **R² (linearity)** | ≥0.95 | 0.90-0.95 | <0.90 |
| **Efficiency at q=1.0** | ≥95% | 90-95% | <90% |
| **No dead zones** | Smooth curve | Minor kinks | Flat regions |

### Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| R² < 0.90 (non-linear) | Thread fraction logic flawed | Check `fmod` pattern in kernel |
| Efficiency < 80% | Kernel not optimized | Use shared memory tiling, increase occupancy |
| High variance between runs | OS jitter | Isolate CPU cores, set performance governor |

---

## Test T2: Step Response - Stability Margins Analysis

### Objective

Measure transient response to sudden load change. Calculate rise time, overshoot, settling time, and steady-state error. These metrics directly relate to phase margin via control theory relationships.

### Procedure

```bash
# Run step response test (Test 2 of benchmark harness)
./gputronic_benchmark 2 80 256 > results/step_response_$(date +%Y%m%d_%H%M%S).txt
```

### Expected Output

```
=================================================================
  TEST 2: STEP RESPONSE - Stability Margins Analysis
=================================================================

[Step] Applying step change from q=0.5 to q=1.0...
[Step] Recording 500 samples at 100 Hz for 5.0s

Time (ms) | q_frac      | Z            | Throughput (M/s)
---------|--------------|--------------|-----------------
0         | 0.50        | 1.000        | 612.34
...
100       | 0.50        | 1.000        | 615.23

[Step] *** STEP CHANGE AT t=100ms ***

110       | 1.00        | 1.050        | 789.45
120       | 1.00        | 1.080        | 923.67
...
5000      | 1.00        | 1.000        | 1234.56

[Step] STEP RESPONSE ANALYSIS
-----------------------------------------------------------------
Rise time (10%→90%):     45.2 ms
Overshoot:               8.3 %
Settling time (2% band): 78.5 ms
Final throughput:        1234.56 M ops/sec

[Step] STABILITY ASSESSMENT:
  ✓ GOOD: Overshoot < 15% (well-damped)
  ✓ GOOD: Settling time < 100ms (fast response)
```

### Analysis Script (Python)

```python
#!/usr/bin/env python3
"""Analyze step response for stability metrics"""

import numpy as np
import matplotlib.pyplot as plt

# Parse output (simplified - extract time, throughput columns)
time_ms = []
throughput = []

with open('results/step_response_20260514_183500.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 4 and parts[0].strip().replace('.', '').isdigit():
            time_ms.append(float(parts[0].strip()))
            throughput.append(float(parts[3].strip()))

time_ms = np.array(time_ms)
throughput = np.array(throughput)

# Find step point (where q changes from 0.5 to 1.0)
step_idx = np.where(np.diff(throughput) > 10)[0][0]  # First large jump
t_step = time_ms[step_idx]

# Pre-step baseline
baseline_tp = np.mean(throughput[:step_idx])

# Post-step steady state (last 10%)
steady_state_tp = np.mean(throughput[int(len(throughput)*0.9):])

# Rise time (10% → 90%)
target_10 = baseline_tp + 0.1 * (steady_state_tp - baseline_tp)
target_90 = baseline_tp + 0.9 * (steady_state_tp - baseline_tp)

t10_idx = np.where(throughput[step_idx:] >= target_10)[0][0] + step_idx
t90_idx = np.where(throughput[step_idx:] >= target_90)[0][0] + step_idx
rise_time_ms = time_ms[t90_idx] - time_ms[t10_idx]

# Overshoot
max_tp = np.max(throughput[step_idx:])
overshoot_pct = ((max_tp - steady_state_tp) / steady_state_tp) * 100.0

# Settling time (within 2% of final value)
lower_bound = 0.98 * steady_state_tp
upper_bound = 1.02 * steady_state_tp
settled_idx = step_idx
for i in range(len(throughput)-1, step_idx, -1):
    if throughput[i] < lower_bound or throughput[i] > upper_bound:
        settled_idx = i + 1
        break
settling_time_ms = time_ms[settled_idx]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(time_ms, throughput, 'b-', linewidth=1.5, label='Throughput')
plt.axvline(x=t_step, color='r', linestyle='--', label=f'Step at t={t_step:.0f}ms')
plt.axhline(y=steady_state_tp, color='g', linestyle=':', label=f'Steady state: {steady_state_tp:.0f}')
plt.axhline(y=target_10, color='gray', linestyle=':', alpha=0.5)
plt.axhline(y=target_90, color='gray', linestyle=':', alpha=0.5)

# Annotate rise time
plt.annotate(f'Rise time: {rise_time_ms:.1f}ms', 
             xy=(time_ms[t10_idx], target_10),
             xytext=(t_step + 200, target_10),
             arrowprops=dict(arrowstyle='->'))

# Annotate overshoot
plt.annotate(f'Overshoot: {overshoot_pct:.1f}%', 
             xy=(time_ms[step_idx + np.argmax(throughput[step_idx:])], max_tp),
             xytext=(t_step + 300, max_tp + 50),
             arrowprops=dict(arrowstyle='->'),
             color='red')

plt.xlabel('Time (ms)')
plt.ylabel('Throughput (M ops/sec)')
plt.title(f'Step Response Analysis\nRise: {rise_time_ms:.1f}ms, Overshoot: {overshoot_pct:.1f}%, Settling: {settling_time_ms:.1f}ms')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('results/step_response_plot.png', dpi=150)
plt.show()

# Print metrics
print(f"\n{'='*60}")
print("STEP RESPONSE METRICS")
print('='*60)
print(f"Rise time (10%→90%):     {rise_time_ms:.1f} ms")
print(f"Overshoot:               {overshoot_pct:.1f} %")
print(f"Settling time (2% band): {settling_time_ms:.1f} ms")
print(f"Steady-state throughput: {steady_state_tp:.2f} M ops/sec")

# Pass/fail decision
print(f"\n{'='*60}")
if overshoot_pct < 15.0 and settling_time_ms < 100.0:
    print("✓ PASS: Well-damped, fast response")
elif overshoot_pct < 30.0 and settling_time_ms < 200.0:
    print("⚠ ACCEPTABLE: Moderate overshoot or slow settling")
else:
    print("✗ FAIL: Poor stability (reduce Kp or increase deadband)")
print('='*60)
```

### Pass/Fail Criteria

| Metric | Target | Acceptable | Fail |
|--------|--------|------------|------|
| **Overshoot** | <15% | 15-30% | >30% |
| **Settling time** | <100ms | 100-200ms | >500ms |
| **Rise time** | 20-80ms | 10-150ms | <10ms or >200ms |
| **Steady-state error** | <2% | 2-5% | >5% |

### Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| Overshoot >30% | Kp too high | Reduce Kp from 0.3 to 0.2 |
| Settling time >500ms | Ki too low or integral windup | Increase Ki, check anti-windup clamp |
| Oscillation (ringing) | Phase margin <30° | Increase deadband, reduce bandwidth |
| Slow rise time (>150ms) | Bandwidth too low | Increase Kp slightly, reduce R_meas in Kalman |

---

## Test T3: Frequency Response - Bode Plot Measurement

### Objective

Empirically measure open-loop transfer function L(jω) by injecting sinusoidal perturbations. Calculate phase margin and gain margin to verify closed-loop stability across frequency range.

### Procedure

```bash
# Run frequency response test (Test 3 of benchmark harness)
./gputronic_benchmark 3 80 256 > results/frequency_response_$(date +%Y%m%d_%H%M%S).txt
```

### Expected Output

```
=================================================================
  TEST 3: FREQUENCY RESPONSE - Bode Plot Measurement
=================================================================

[Bode] Measuring frequency response from 0.5 Hz to 50.0 Hz

Freq (Hz) | Gain (dB)   | Phase (deg)   | Magnitude
---------|--------------|-----------------|------------
0.5       | -3.21        | -45.3           | 0.69
1.0       | -6.45        | -67.8           | 0.48
...
10.0      | -20.34       | -135.2          | 0.09
...
50.0      | -40.12       | -172.5          | 0.01

[Bode] Frequency response measurement complete
```

### Analysis Script (Python)

```python
#!/usr/bin/env python3
"""Analyze frequency response for phase/gain margins"""

import numpy as np
import matplotlib.pyplot as plt

# Parse output
freq_hz = []
gain_db = []
phase_deg = []

with open('results/frequency_response_20260514_184000.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 3 and parts[0].strip().replace('.', '').isdigit():
            freq_hz.append(float(parts[0].strip()))
            gain_db.append(float(parts[1].strip()))
            phase_deg.append(float(parts[2].strip()))

freq_hz = np.array(freq_hz)
gain_db = np.array(gain_db)
phase_deg = np.array(phase_deg)

# Find crossover frequency (where gain = 0 dB)
crossover_idx = np.where(np.diff(np.sign(gain_db)))[0]
if len(crossover_idx) > 0:
    wc_hz = freq_hz[crossover_idx[0]]
    wc_rad = 2 * np.pi * wc_hz
    
    # Phase at crossover
    phase_at_wc = phase_deg[crossover_idx[0]]
    
    # Phase margin
    pm = 180 + phase_at_wc
else:
    wc_hz = None
    pm = None

# Find phase crossover (where phase = -180°)
phase_crossover_idx = np.where(np.diff(np.sign(phase_deg + 180)))[0]
if len(phase_crossover_idx) > 0:
    wpc_hz = freq_hz[phase_crossover_idx[0]]
    gain_at_wpc_db = gain_db[phase_crossover_idx[0]]
    
    # Gain margin (in dB)
    gm_db = -gain_at_wpc_db
else:
    wpc_hz = None
    gm_db = float('inf')

# Plot Bode diagram
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Magnitude plot
ax1.semilogx(freq_hz, gain_db, 'b-', linewidth=2)
ax1.axhline(y=0, color='r', linestyle='--', label='0 dB (crossover)')
if wc_hz:
    ax1.axvline(x=wc_hz, color='g', linestyle=':', alpha=0.5)
    ax1.annotate(f'ω_c = {wc_hz:.2f} Hz', xy=(wc_hz, 0), 
                 xytext=(wc_hz*0.7, 5), arrowprops=dict(arrowstyle='->'))
ax1.set_ylabel('Gain (dB)', fontsize=12)
ax1.set_title('Bode Plot: GPUTronic Governor Frequency Response', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Phase plot
ax2.semilogx(freq_hz, phase_deg, 'r-', linewidth=2)
ax2.axhline(y=-180, color='b', linestyle='--', label='-180°')
ax2.axhline(y=-135, color='g', linestyle=':', alpha=0.5, label='-135° (PM=45°)')
if wpc_hz:
    ax2.axvline(x=wpc_hz, color='m', linestyle=':', alpha=0.5)
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Phase (deg)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('results/bode_plot.png', dpi=150)
plt.show()

# Print stability margins
print(f"\n{'='*60}")
print("FREQUENCY RESPONSE ANALYSIS - STABILITY MARGINS")
print('='*60)

if wc_hz:
    print(f"Gain crossover frequency (ω_c): {wc_hz:.2f} Hz ({wc_rad:.2f} rad/s)")
    print(f"Phase at ω_c:                   {phase_at_wc:.1f}°")
    print(f"Phase margin (PM):              {pm:.1f}°")
else:
    print("Gain crossover frequency: Not found (system may be unstable)")

if wpc_hz:
    print(f"\nPhase crossover frequency (ω_pc): {wpc_hz:.2f} Hz")
    print(f"Gain at ω_pc:                     {gain_at_wpc_db:.2f} dB")
    print(f"Gain margin (GM):                 {gm_db:.1f} dB")
else:
    print(f"\nPhase crossover frequency: Not found (GM = ∞)")

# Pass/fail decision
print(f"\n{'='*60}")
if pm is not None and pm >= 45.0:
    print(f"✓ PASS: Phase margin {pm:.1f}° ≥ 45°")
elif pm is not None and pm >= 30.0:
    print(f"⚠ WARNING: Phase margin {pm:.1f}° between 30-45° (marginal)")
else:
    print(f"✗ FAIL: Phase margin <30° or not found (unstable!)")

if gm_db is not None and gm_db >= 6.0:
    print(f"✓ PASS: Gain margin {gm_db:.1f} dB ≥ 6 dB")
elif gm_db is not None and gm_db >= 3.0:
    print(f"⚠ WARNING: Gain margin {gm_db:.1f} dB between 3-6 dB (marginal)")
else:
    print(f"✗ FAIL: Gain margin <3 dB or not found")
print('='*60)
```

### Pass/Fail Criteria

| Metric | Target | Acceptable | Fail |
|--------|--------|------------|------|
| **Phase margin** | 60° ± 15° | 45-75° | <30° or >90° |
| **Gain margin** | ≥10 dB | 6-10 dB | <6 dB |
| **Crossover frequency** | 10-20 Hz | 5-30 Hz | <5 Hz (too slow) or >50 Hz (noisy) |

### Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| PM <30° | System too aggressive | Reduce Kp, increase deadband |
| GM <6 dB | High-frequency noise amplification | Increase R_meas in Kalman filter |
| Crossover >50 Hz | Bandwidth too high for GPU dynamics | Reduce control frequency or increase dt |

---

## Test T4: Load Variation - Idle ↔ Full Load Transitions

### Objective

Test governor's ability to track rapid workload changes without instability. Simulates real-world scenarios like game level transitions, inference batch size changes, or compute job scheduling.

### Procedure

```bash
# Run load variation test (Test 4 of benchmark harness)
./gputronic_benchmark 4 80 256 > results/load_variation_$(date +%Y%m%d_%H%M%S).txt
```

### Expected Output

```
=================================================================
  TEST 4: LOAD VARIATION - Idle ↔ Full Load Transitions
=================================================================

[Load] Running 10 transitions (idle ↔ full load)
[Load] Each phase: 0.5s

Phase         | q_target    | Throughput (M/s)
----------------|--------------|-----------------
IDLE            | 0.10        | 123.45
FULL LOAD       | 1.00        | 1234.56
IDLE            | 0.10        | 125.67
FULL LOAD       | 1.00        | 1238.90
...

[Load] Load variation test complete
```

### Analysis

**Manual inspection:** Check for:
- Consistent throughput at each load level (low variance)
- No overshoot or oscillation during transitions
- Fast recovery to steady state (<200ms after transition)

**Automated check:** Calculate coefficient of variation (CV) for throughput at each load level:

```python
import numpy as np

# Parse output and extract throughput values for IDLE and FULL LOAD phases
idle_throughputs = [123.45, 125.67, ...]  # From test output
full_throughputs = [1234.56, 1238.90, ...]

idle_cv = np.std(idle_throughputs) / np.mean(idle_throughputs) * 100
full_cv = np.std(full_throughputs) / np.mean(full_throughputs) * 100

print(f"IDLE throughput CV: {idle_cv:.2f}%")
print(f"FULL LOAD throughput CV: {full_cv:.2f}%")

if idle_cv < 5.0 and full_cv < 3.0:
    print("✓ PASS: Stable throughput at both load levels")
else:
    print("✗ FAIL: High variance indicates instability")
```

### Pass/Fail Criteria

| Metric | Target | Acceptable | Fail |
|--------|--------|------------|------|
| **Throughput CV (IDLE)** | <3% | 3-5% | >5% |
| **Throughput CV (FULL)** | <2% | 2-3% | >3% |
| **Transition recovery** | <200ms | 200-400ms | >500ms |

---

## Test T5: Thermal Stress - Long-duration High-load Test

### Objective

Verify governor stability under sustained thermal load. Check for thermal throttling, clock instability, and long-term drift in control parameters.

### Procedure

```bash
# Run thermal stress test (Test 5 of benchmark harness)
# Default duration: 60 seconds, can specify longer (e.g., 1800s = 30 min)
./gputronic_benchmark 5 80 256 1800 > results/thermal_stress_$(date +%Y%m%d_%H%M%S).txt

# Monitor temperature in real-time (separate terminal)
watch -n 1 nvidia-smi
```

### Expected Output

```
=================================================================
  TEST 5: THERMAL STRESS - Long-duration High-Load Test
=================================================================

[Thermal] Running at full load for 1800.0 seconds

Time (s)    | Temp (°C)   | Clock (MHz)   | Throughput (M/s)
---------|--------------|-----------------|-----------------
0.0       | 42          | 2550          | 1234.56
1.0       | 58          | 2550          | 1235.23
...
60.0      | 72          | 2550          | 1233.89
...
600.0     | 78          | 2545          | 1230.45
...
1800.0    | 79          | 2540          | 1228.90

[Thermal] Thermal stress test complete
```

### Analysis Script (Python)

```python
#!/usr/bin/env python3
"""Analyze thermal stress test for stability over time"""

import numpy as np
import matplotlib.pyplot as plt

# Parse output
time_s = []
temp_c = []
clock_mhz = []
throughput = []

with open('results/thermal_stress_20260514_185000.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 4 and parts[0].strip().replace('.', '').isdigit():
            time_s.append(float(parts[0].strip()))
            temp_c.append(int(parts[1].strip()))
            clock_mhz.append(int(parts[2].strip()))
            throughput.append(float(parts[3].strip()))

time_s = np.array(time_s)
temp_c = np.array(temp_c)
clock_mhz = np.array(clock_mhz)
throughput = np.array(throughput)

# Plot all metrics over time
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(time_s, temp_c, 'r-', linewidth=2)
axes[0].axhline(y=83, color='gray', linestyle='--', alpha=0.5, label='Thermal limit (83°C)')
axes[0].set_ylabel('Temperature (°C)', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(time_s, clock_mhz, 'b-', linewidth=2)
axes[1].set_ylabel('Clock (MHz)', fontsize=12)
axes[1].grid(True, alpha=0.3)

axes[2].plot(time_s, throughput, 'g-', linewidth=2)
axes[2].axhline(y=np.mean(throughput), color='gray', linestyle=':', alpha=0.5, 
                label=f'Mean: {np.mean(throughput):.1f}')
axes[2].set_xlabel('Time (s)', fontsize=12)
axes[2].set_ylabel('Throughput (M ops/sec)', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Thermal Stress Test: Temperature, Clock, Throughput Over Time', fontsize=14)
plt.tight_layout()
plt.savefig('results/thermal_stress_plot.png', dpi=150)
plt.show()

# Calculate metrics
max_temp = np.max(temp_c)
temp_drift = temp_c[-1] - temp_c[0]
clock_drop = clock_mhz[0] - clock_mhz[-1]
throughput_drift = (throughput[-1] - throughput[0]) / throughput[0] * 100

print(f"\n{'='*60}")
print("THERMAL STRESS ANALYSIS")
print('='*60)
print(f"Max temperature:              {max_temp:.0f}°C")
print(f"Temperature drift (start→end): {temp_drift:+.0f}°C")
print(f"Clock frequency drop:         {clock_drop:.0f} MHz ({clock_drop/clock_mhz[0]*100:.2f}%)")
print(f"Throughput drift:             {throughput_drift:+.2f}%")

# Pass/fail decision
print(f"\n{'='*60}")
if max_temp < 83 and clock_drop < 50 and abs(throughput_drift) < 5:
    print("✓ PASS: Stable under thermal stress (no throttling)")
elif max_temp < 87 and clock_drop < 100 and abs(throughput_drift) < 10:
    print("⚠ ACCEPTABLE: Minor thermal effects")
else:
    print("✗ FAIL: Significant thermal throttling detected")
print('='*60)
```

### Pass/Fail Criteria

| Metric | Target | Acceptable | Fail |
|--------|--------|------------|------|
| **Max temperature** | <75°C | 75-83°C | >83°C (throttling) |
| **Clock drop** | <20 MHz | 20-100 MHz | >100 MHz |
| **Throughput drift** | <2% | 2-5% | >5% |

---

## Test T6: Inference Workload - Real Workload Integration

### Objective

Test governor under realistic inference-like workload (GEMM + Transformer). Validates that control loop works with actual compute patterns, not just synthetic benchmarks.

### Procedure

```bash
# Compile inference workload (if not already done)
nvcc -O3 -arch=sm_120 \
     -o gputronic_inference gputronic_inference_workload.cu \
     -pthread \
     -lm

# Run with governor active (in separate terminal, start governor first)
./gputronic_v14 80 &
GOVERNOR_PID=$!

# Wait for governor to initialize
sleep 2

# Run inference workload
./gputronic_inference 60 > results/inference_workload_$(date +%Y%m%d_%H%M%S).txt

# Stop governor
kill $GOVERNOR_PID
```

### Expected Output (from governor terminal)

```
=================================================================
  GPUTronic Stage 13 SCAG v14.0.0 - Self-Calibrating Adaptive Governor
  Target: Blackwell (RTX 5080) with stock drivers
=================================================================

[GPUTronic] Configuration:
  SMs: 80
  Threads per SM: 1638
  Test mode: 0

[GPUTronic] GPU: NVIDIA GeForce RTX 5080
[GPUTronic] Compute capability: 12.0
[GPUTronic] Actual SM count: 80

[GPUTronic] Allocating zero-copy throttle cable...
[GPUTronic] Zero-copy throttle allocated at host addr: 0x7f8a4c000000
[GPUTronic] Zero-copy throttle device ptr: 0x800000000

[GPUTronic] Starting control loop thread...
[GPUTronic] Control loop starting at 100kHz+...
[GPUTronic] NVML monitoring started (1Hz)

Control frequency: 102345 Hz | Z=1.182 | PM=0.846 | q=0.801
Control frequency: 101987 Hz | Z=1.179 | PM=0.848 | q=0.803
...

[Status] Z=1.185 | PM=0.844 | q=0.798 | dZ/dt=+0.023
```

### Analysis

**Manual inspection:** Check for:
- Control frequency consistently ≥100kHz
- Phase margin stable around 0.85 (±0.05)
- No crashes or CUDA errors
- Smooth q_fraction adjustments (no wild swings)

**Automated check:** Parse governor output and calculate statistics:

```python
import re
import numpy as np

# Extract control frequency, Z, PM values from log
freq_pattern = r'Control frequency: (\d+) Hz \| Z=([\d.]+) \| PM=([\d.]+)'
status_pattern = r'\[Status\] Z=([\d.]+) \| PM=([\d.]+) \| q=([\d.]+)'

frequencies = []
z_values = []
pm_values = []

with open('governor_log.txt', 'r') as f:
    for line in f:
        match = re.search(freq_pattern, line)
        if match:
            frequencies.append(int(match.group(1)))
            z_values.append(float(match.group(2)))
            pm_values.append(float(match.group(3)))

print(f"Control frequency: mean={np.mean(frequencies):.0f} Hz, "
      f"std={np.std(frequencies):.0f} Hz")
print(f"Impedance Z:       mean={np.mean(z_values):.3f}, "
      f"std={np.std(z_values):.4f}")
print(f"Phase margin PM:   mean={np.mean(pm_values):.3f}, "
      f"std={np.std(pm_values):.4f}")

# Pass/fail
if np.mean(frequencies) >= 100000 and np.std(pm_values) < 0.05:
    print("✓ PASS: Stable control at target frequency")
else:
    print("✗ FAIL: Control frequency too low or PM unstable")
```

### Pass/Fail Criteria

| Metric | Target | Acceptable | Fail |
|--------|--------|------------|------|
| **Control frequency** | ≥100kHz | 50-100kHz | <20kHz |
| **PM standard deviation** | <0.05 | 0.05-0.10 | >0.15 |
| **No crashes** | ✓ | - | ✗ |

---

## Test T7: Governor Comparison - vs. Stock NVIDIA Governor

### Objective

Quantify improvement over stock NVIDIA power/thermal governor. Measure throughput stability, thermal efficiency, and response time to load changes.

### Procedure

```bash
# Test 1: GPUTronic governor (as above)
./gputronic_v14 80 &
GPUTRONIC_PID=$!
sleep 5
./gputronic_inference 60 > results/gputronic_comparison.txt
kill $GPUTRONIC_PID

# Test 2: Stock NVIDIA governor (disable GPUTronic, use default)
nvidia-smi -pm 1
nvidia-smi -acp 0  # Disable application clocks profile

./gputronic_inference 60 > results/stock_comparison.txt

# Compare results
python3 analyze_comparison.py
```

### Analysis Script (Python)

```python
#!/usr/bin/env python3
"""Compare GPUTronic vs. stock governor performance"""

import numpy as np
import matplotlib.pyplot as plt

def parse_throughput_log(filename):
    """Extract throughput values from workload log"""
    throughputs = []
    with open(filename, 'r') as f:
        for line in f:
            if 'Throughput' in line or 'GFLOPS' in line:
                # Extract numeric value (simplified)
                match = re.search(r'([\d.]+)\s*(?:M|G)', line)
                if match:
                    throughputs.append(float(match.group(1)))
    return np.array(throughputs)

gputronic_tp = parse_throughput_log('results/gputronic_comparison.txt')
stock_tp = parse_throughput_log('results/stock_comparison.txt')

# Calculate metrics
gputronic_mean = np.mean(gputronic_tp)
gputronic_std = np.std(gputronic_tp)
gputronic_cv = gputronic_std / gputronic_mean * 100

stock_mean = np.mean(stock_tp)
stock_std = np.std(stock_tp)
stock_cv = stock_std / stock_mean * 100

print(f"\n{'='*60}")
print("GOVERNOR COMPARISON: GPUTronic vs. Stock NVIDIA")
print('='*60)
print(f"GPUTronic:")
print(f"  Mean throughput:    {gputronic_mean:.2f} M ops/sec")
print(f"  Std deviation:      {gputronic_std:.2f}")
print(f"  Coefficient of var: {gputronic_cv:.2f}%")

print(f"\nStock NVIDIA:")
print(f"  Mean throughput:    {stock_mean:.2f} M ops/sec")
print(f"  Std deviation:      {stock_std:.2f}")
print(f"  Coefficient of var: {stock_cv:.2f}%")

# Improvement metrics
throughput_improvement = (gputronic_mean - stock_mean) / stock_mean * 100
stability_improvement = (stock_cv - gputronic_cv) / stock_cv * 100

print(f"\n{'='*60}")
print("IMPROVEMENT METRICS")
print('='*60)
print(f"Throughput improvement:     {throughput_improvement:+.2f}%")
print(f"Stability improvement:      {stability_improvement:+.2f}% (lower CV = better)")

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(gputronic_tp[:200], 'b-', linewidth=1, label=f'GPUTronic (CV={gputronic_cv:.1f}%)')
plt.plot(stock_tp[:200], 'r-', linewidth=1, label=f'Stock NVIDIA (CV={stock_cv:.1f}%)')
plt.xlabel('Sample', fontsize=12)
plt.ylabel('Throughput (M ops/sec)', fontsize=12)
plt.title('Governor Comparison: Throughput Stability Over Time', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/governor_comparison_plot.png', dpi=150)
plt.show()

# Pass/fail (relative to stock)
print(f"\n{'='*60}")
if stability_improvement > 10:
    print(f"✓ PASS: GPUTronic {stability_improvement:.1f}% more stable than stock")
elif stability_improvement > 0:
    print(f"⚠ ACCEPTABLE: GPUTronic slightly more stable ({stability_improvement:.1f}%)")
else:
    print(f"✗ FAIL: GPUTronic less stable than stock (-{abs(stability_improvement):.1f}%)")
print('='*60)
```

### Pass/Fail Criteria

| Metric | Target | Acceptable | Fail |
|--------|--------|------------|------|
| **Stability improvement** | >20% | 10-20% | <5% |
| **Throughput improvement** | >5% | 0-5% | <0% (worse) |

---

## Regression Testing & Continuous Integration

### Automated Test Suite Script

```bash
#!/bin/bash
# run_all_tests.sh - Execute full test suite and generate report

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "================================================================="
echo "  GPUTronic Full Test Suite - $TIMESTAMP"
echo "================================================================="

# Compile (if needed)
make all || { echo "Compilation failed"; exit 1; }

# Run tests sequentially
echo "[1/7] Running dyno sweep..."
./gputronic_benchmark 1 > "$RESULTS_DIR/dyno.txt" 2>&1

echo "[2/7] Running step response..."
./gputronic_benchmark 2 > "$RESULTS_DIR/step.txt" 2>&1

echo "[3/7] Running frequency response..."
./gputronic_benchmark 3 > "$RESULTS_DIR/freq.txt" 2>&1

echo "[4/7] Running load variation..."
./gputronic_benchmark 4 > "$RESULTS_DIR/load_var.txt" 2>&1

echo "[5/7] Running thermal stress (60s)..."
./gputronic_benchmark 5 80 256 60 > "$RESULTS_DIR/thermal.txt" 2>&1

echo "[6/7] Running inference workload..."
./gputronic_inference 30 > "$RESULTS_DIR/inference.txt" 2>&1

echo "[7/7] Generating analysis reports..."
python3 analyze_dyno.py < "$RESULTS_DIR/dyno.txt" > "$RESULTS_DIR/dyno_analysis.txt"
python3 analyze_step.py < "$RESULTS_DIR/step.txt" > "$RESULTS_DIR/step_analysis.txt"
python3 analyze_freq.py < "$RESULTS_DIR/freq.txt" > "$RESULTS_DIR/freq_analysis.txt"
python3 analyze_thermal.py < "$RESULTS_DIR/thermal.txt" > "$RESULTS_DIR/thermal_analysis.txt"

echo ""
echo "================================================================="
echo "  Test suite complete. Results in: $RESULTS_DIR/"
echo "================================================================="

# Generate summary report
cat << EOF > "$RESULTS_DIR/SUMMARY.md"
# GPUTronic Test Summary - $TIMESTAMP

## Test Results

| Test | Status | Key Metrics |
|------|--------|-------------|
| Dyno Sweep | $(grep -q 'PASS' "$RESULTS_DIR/dyno_analysis.txt" && echo "✓ PASS" || echo "✗ FAIL") | $(grep 'R²' "$RESULTS_DIR/dyno_analysis.txt" | head -1) |
| Step Response | $(grep -q 'PASS' "$RESULTS_DIR/step_analysis.txt" && echo "✓ PASS" || echo "✗ FAIL") | $(grep 'Overshoot' "$RESULTS_DIR/step_analysis.txt" | head -1) |
| Frequency Response | $(grep -q 'PASS' "$RESULTS_DIR/freq_analysis.txt" && echo "✓ PASS" || echo "✗ FAIL") | $(grep 'Phase margin' "$RESULTS_DIR/freq_analysis.txt" | head -1) |
| Load Variation | $(grep -q 'PASS' "$RESULTS_DIR/load_var.txt" && echo "✓ PASS" || echo "✗ FAIL") | See load_var.txt |
| Thermal Stress | $(grep -q 'PASS' "$RESULTS_DIR/thermal_analysis.txt" && echo "✓ PASS" || echo "✗ FAIL") | $(grep 'Max temperature' "$RESULTS_DIR/thermal_analysis.txt" | head -1) |
| Inference Workload | $(grep -q 'PASS' "$RESULTS_DIR/inference.txt" && echo "✓ PASS" || echo "✗ FAIL") | See inference.txt |

## Plots Generated

- dyno_sweep_plot.png
- step_response_plot.png
- bode_plot.png
- thermal_stress_plot.png
- governor_comparison_plot.png (if comparison test run)

## Full Reports

See individual analysis files in this directory for detailed metrics.
EOF

echo "Summary report: $RESULTS_DIR/SUMMARY.md"
```

### CI/CD Integration (GitHub Actions Example)

```yaml
# .github/workflows/gputronic_tests.yml
name: GPUTronic Regression Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: self-hosted  # Requires NVIDIA GPU
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup CUDA environment
        run: |
          source /etc/profile.d/cuda.sh
          nvcc --version
      
      - name: Compile
        run: |
          make all
      
      - name: Run test suite
        run: |
          ./run_all_tests.sh
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results-${{ github.sha }}
          path: results/
      
      - name: Check test pass rate
        run: |
          if grep -q '✗ FAIL' results/*/SUMMARY.md; then
            echo "Some tests failed!"
            exit 1
          fi
```

---

## Appendix A: Troubleshooting Guide

### Common Issues & Solutions

| Issue | Symptoms | Diagnosis | Fix |
|-------|----------|-----------|-----|
| **Control frequency too low** | <50kHz instead of 100kHz | Check `dmesg` for CPU throttling, interrupts | Set CPU governor to performance, isolate cores |
| **CUDA out of memory** | `cudaMalloc` fails | GPU memory exhausted by test workload | Reduce matrix sizes, free memory between tests |
| **NVML not available** | Temperature monitoring disabled | NVML library not found or permissions | Install NVIDIA management libs, run as root or add user to video group |
| **Kernel crashes** | `cudaDeviceSynchronize()` returns error | Invalid memory access in kernel | Run with `cuda-memcheck`, fix out-of-bounds accesses |
| **Thermal throttling** | Clock drops >100MHz during stress test | GPU hitting thermal limit (83°C+) | Improve case airflow, increase fan curve, reduce power limit |
| **Overshoot >30%** | Step response oscillates wildly | Kp too high for system dynamics | Reduce Kp from 0.3 to 0.2 or lower |
| **Settling time >500ms** | Slow recovery after load change | Ki too low or integral windup | Increase Ki, verify anti-windup clamp is working |

### Debugging Tools

```bash
# CUDA memory checker
cuda-memcheck ./gputronic_v14

# CUDA profiling (compute utilization)
nvprof --print-gpu-trace ./gputronic_benchmark 1

# System-wide performance monitoring
perf stat -e cpu-clock,context-switches,cycles,instructions ./gputronic_v14

# GPU power/thermal monitoring (real-time)
watch -n 0.5 nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.current.sm --format=csv

# CPU core isolation verification
cat /proc/cmdline | grep isolcpus
taskset -c 4,5,6,7 ./gputronic_v14  # Run on isolated cores
```

---

## Appendix B: Configuration Reference

### Tunable Parameters (with safe ranges)

| Parameter | Default | Safe Range | Effect of Increasing |
|-----------|---------|------------|---------------------|
| **Kp** (PI proportional gain) | 0.3 | 0.1-0.5 | Faster response, more overshoot |
| **Ki** (PI integral gain) | 0.05 | 0.01-0.1 | Better steady-state error, risk of windup |
| **KALMAN_Q_PROC** (process noise) | 0.0005 | 0.0001-0.01 | Trust measurements more, faster adaptation |
| **KALMAN_R_MEAS** (measurement noise) | 0.01 | 0.005-0.05 | Trust model more, smoother estimates |
| **Z_LOW_THRESH** (deadband low) | 1.05 | 1.02-1.10 | Wider deadband = less hunting |
| **Z_HIGH_THRESH** (deadband high) | 1.80 | 1.50-2.00 | Higher threshold = more aggressive before throttle |
| **TARGET_PM** (phase margin target) | 0.85 | 0.75-0.90 | Lower PM = higher throughput, less stable |
| **CONTROL_DT** (control interval) | 10µs | 5-20µs | Shorter = faster control, more CPU load |

### Environment Variables

```bash
# Set GPU to maximum performance mode
export NVIDIA_PERFORMANCE_MODE=1

# Disable power saving features
nvidia-smi -pm 1
nvidia-smi -ac 0

# Set CPU governor (Linux)
sudo cpupower frequency-set -g performance

# Isolate CPU cores for control loop (add to /etc/default/grub)
GRUB_CMDLINE_LINUX="isolcpus=4,5,6,7 nohz_full=4,5,6,7 rcu_nocbs=4,5,6,7"
sudo update-grub && sudo reboot
```

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Q-axis** | Torque axis — useful work rate (what we want to stabilize) |
| **D-axis** | Flux axis — thermal/power field (weakening when limits approached) |
| **Z-axis** | Impedance axis — memory-latency reactive power (stall avoidance) |
| **Phase Margin (PM)** | Stability metric: PM = 1/Z, target ≈0.85 |
| **Deadband** | Range where no control action taken (prevents hunting) |
| **Zero-copy** | Pinned host memory mapped to device address space (<1µs latency) |
| **Dyno sweep** | Static characterization: throughput vs. thread fraction |
| **Bode plot** | Frequency response: gain and phase vs. frequency |

---

## Appendix D: References & Further Reading

### Control Theory
- "Modern Control Engineering" by Katsuhiko Ogata (5th ed.) — Bode plots, stability margins
- "Feedback Control of Dynamic Systems" by Franklin, Powell, Emami-Naeini — PID tuning
- [Control Systems Toolbox (MATLAB)](https://www.mathworks.com/products/control.html)

### Kalman Filtering
- "Optimal State Estimation" by Dan Simon — Comprehensive KF treatment
- [Kalman Filter Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter) — Good overview with equations

### GPU Architecture
- NVIDIA CUDA C++ Programming Guide — Warp scheduling, memory hierarchy
- "GPU Performance Analysis and Optimization" by Micikevicius et al.
- VAST Conference: "GPU Architecture and Performance" talks

### Field-Oriented Control (Motor Analogy)
- "Sensorless Vector and Direct Torque Control" by Paul Vas (1998)
- TI Application Report SPRA794: "Sensorless FOC of PMSM"

---

*Document version 1.0 — GPUTronic Stage 13 SCAG v14.0.0 Test Plan*  
*Generated for Ben Coupland — May 14, 2026*
