# GPUTronic Kalman Observer Mathematics

## Overview

This document provides the complete mathematical foundation for the 3-state Kalman observer used in GPUTronic's Z-axis impedance control. The observer implements **sensorless rotor position estimation** analogous to Field-Oriented Control (FOC) of permanent-magnet synchronous motors (PMSM).

### Core Insight

Just as sensorless FOC estimates rotor position θ from stator currents alone, the GPUTronic Kalman observer estimates **instruction throughput phase** (impedance Z and its derivatives) from work completion counters. This enables predictive stall avoidance at 100kHz+ control rates with minimal latency.

---

## State Vector Definition

```cpp
// State vector: x = [Z, dZ/dt, d²Z/dt²]ᵀ
double x[3];  // In code: Kalman3State::x[]
```

| Index | Symbol | Physical Meaning | FOC Analogy | Why It Matters |
|-------|--------|------------------|-------------|----------------|
| 0 | **Z** | Impedance (how much slower than expected) | Rotor position θ | Current stall severity |
| 1 | **dZ/dt** | Rate of change (is impedance increasing?) | Angular velocity ω | Predictive: will we stall soon? |
| 2 | **d²Z/dt²** | Acceleration (is the rate changing?) | Angular acceleration α | Anticipates sudden load changes |

### Impedance Calculation

Raw impedance is computed from work completion counters:

```cpp
double expected_work = total_threads * cycles_per_cycle;
double actual_work = atomic_read(work_counter);
double z_raw = expected_work / actual_work;  // Z ≥ 1.0 (higher = more stalled)
```

- **Z = 1.0**: Perfect throughput, no stalls
- **Z = 1.2**: 20% slower than expected (mild impedance)
- **Z = 2.0**: 50% throughput (severe stall)

---

## State Transition Model: Constant-Acceleration Physics

We assume impedance evolves like a moving object with constant acceleration over the control interval dt:

### Continuous-Time Equations

```
Z(t+dt)     = Z(t) + dt × dZ/dt(t) + 0.5 × dt² × d²Z/dt²(t)
dZ/dt(t+dt) = dZ/dt(t) + dt × d²Z/dt²(t)
d²Z/dt²(t+dt) = d²Z/dt²(t)
```

### Discrete-Time Matrix Form: x_pred = F × x

The state transition matrix **F** encodes the physics model:

```
        | 1   dt    0.5×dt² |
    F = | 0   1     dt      |
        | 0   0     1       |
```

Applied as:

```cpp
double x0_p = x[0] + dt * x[1] + 0.5 * dt2 * x[2];  // Z prediction
double x1_p = x[1] + dt * x[2];                      // dZ/dt prediction
double x2_p = x[2];                                  // d²Z/dt² unchanged
```

**Physical interpretation:**  
"If impedance is currently 1.2, increasing at 0.3/sec, with acceleration 0.1/sec², then in dt=0.01s (10ms):"
- New Z = 1.2 + 0.01×0.3 + 0.5×0.0001×0.1 ≈ **1.203**
- We predict a small increase based on current rate and acceleration

---

## Covariance Matrix: Tracking Uncertainty

```cpp
double P[9];  // Flattened 3×3 matrix (row-major)
// P = | P00  P01  P02 |
//     | P10  P11  P12 |
//     | P20  P21  P22 |
```

### What Does P Represent?

**P is the error covariance matrix.** It quantifies uncertainty in our state estimates:

- **Diagonal elements (P₀₀, P₁₁, P₂₂)**: Variance of each state estimate
  - High value = we're very uncertain about this state
  - Low value = we trust our estimate
  
- **Off-diagonal elements**: Correlation between states
  - e.g., P₀₁ tells us if uncertainty in Z correlates with uncertainty in dZ/dt

### Initial Condition

```cpp
P[0] = P[4] = P[8] = 1.0;  // High initial uncertainty (we don't know the GPU yet)
// All off-diagonals = 0.0
```

**Why high initial uncertainty?**  
On startup, we have no prior knowledge of the GPU's impedance characteristics. The Kalman filter will quickly converge as measurements come in.

---

## Process Noise Matrix Q: Model Imperfection

```cpp
constexpr double KALMAN_Q_PROC = 0.0005;  // Applied to diagonal elements
```

### What Does Q Represent?

**Q models how much the real world deviates from our physics model.** Even with constant-acceleration, impedance can jump unexpectedly due to:

- Memory controller scheduling variations
- L2 cache miss bursts
- Thermal throttling events
- Driver interference (WDDM preemption)
- Warp scheduler stochasticity

### Q Matrix Structure

```
        | Q_proc  0       0      |
    Q = | 0       Q_proc  0      |
        | 0       0       Q_proc |
```

**Why diagonal only?**  
We assume process noise affects each state independently. Cross-correlations in process noise are negligible for this application.

### When to Tune Q_proc

| Value | Effect | When to Use |
|-------|--------|-------------|
| Low (0.001) | Trust model more, smoother estimates | Stable workloads (GEMM, inference) |
| High (0.01) | Trust measurements more, faster response | Noisy workloads (gaming with branching) |

**Rule of thumb:** If Z jumps around too much → decrease Q_proc. If governor reacts too slowly to changes → increase Q_proc.

---

## Measurement Noise R: Sensor Imperfection

```cpp
constexpr double KALMAN_R_MEAS = 0.01;  // Variance of our impedance measurement
```

### What Does R Represent?

**R models how noisy our Z measurements are.** Our raw impedance calculation is inherently noisy because:

- Work counters have quantization (integer counts)
- Timing has jitter (OS scheduling, interrupts)
- GPU execution is stochastic (warp scheduling non-determinism)

### When to Tune R_meas

| Value | Effect | When to Use |
|-------|--------|-------------|
| Low (0.005) | Trust measurements more, faster response | Clean counters, low jitter |
| High (0.05) | Trust model more, smoother estimates | Noisy workloads, high timing jitter |

**Rule of thumb:** If PM oscillates rapidly → increase R_meas. If governor lags behind actual changes → decrease R_meas.

---

## The PREDICT Step: Projecting Forward

This happens every control cycle (~5-10µs for 100kHz+ rate).

### Step 1: State Prediction (Already Covered)

```cpp
double x0_p = x[0] + dt * x[1] + 0.5 * dt2 * x[2];
double x1_p = x[1] + dt * x[2];
double x2_p = x[2];
```

### Step 2: Covariance Propagation (CRITICAL)

This is where proper Kalman filtering differs from naive implementations. The covariance must propagate through the physics model:

**Mathematical formula:**
```
P_pred = F × P × Fᵀ + Q
```

**Implementation (3×3 matrix multiplication):**

```cpp
// First compute F * P
double fp00 = P[0] + dt*P[3] + 0.5*dt2*P[6];
double fp01 = P[1] + dt*P[4] + 0.5*dt2*P[7];
double fp02 = P[2] + dt*P[5] + 0.5*dt2*P[8];
double fp10 = P[3] + dt*P[6];
double fp11 = P[4] + dt*P[7];
double fp12 = P[5] + dt*P[8];
double fp20 = P[6];
double fp21 = P[7];
double fp22 = P[8];

// Then compute (F * P) * Fᵀ + Q
double fpt00 = fp00 + dt*fp01 + 0.5*dt2*fp02 + KALMAN_Q_PROC;
double fpt01 = fp01 + dt*fp02;
double fpt02 = fp02;
double fpt10 = fp10 + dt*fp11 + 0.5*dt2*fp12;
double fpt11 = fp11 + dt*fp12 + KALMAN_Q_PROC;
double fpt12 = fp12;
double fpt20 = fp20 + dt*fp21 + 0.5*dt2*fp22;
double fpt21 = fp21 + dt*fp22;
double fpt22 = fp22 + KALMAN_Q_PROC;

// Update P matrix
P[0] = fpt00; P[1] = fpt01; P[2] = fpt02;
P[3] = fpt10; P[4] = fpt11; P[5] = fpt12;
P[6] = fpt20; P[7] = fpt21; P[8] = fpt22;
```

**Why this matters:**  
The covariance doesn't just grow uniformly (`P[i] += Q_proc`). It **propagates through the physics model**:

- Uncertainty in Z affects uncertainty in future Z (obvious)
- But uncertainty in dZ/dt ALSO affects future Z (through the dt×dZ/dt term)
- And uncertainty in d²Z/dt² affects both (through dt² terms)

This is why proper matrix math is essential — the states are **coupled** through the physics model. Without this, the Kalman gains will be incorrect, and the filter won't be optimal.

---

## The UPDATE Step: Correcting with Measurements

Now we get a new measurement `z_raw` and correct our prediction.

### Measurement Model: H × x = z

We only directly measure Z (not dZ/dt or d²Z/dt²), so the observation matrix **H** is:

```
    H = [1  0  0]
```

This means: `z_expected = H × x = x[0] = Z`

### Step 1: Innovation (Measurement Residual)

```cpp
double y = z_raw - x0_p;  // How wrong was our prediction?
```

**Physical interpretation:**  
We predicted Z=1.203, but measured Z=1.250. The innovation is **y = +0.047**. This tells us: "Impedance increased more than we expected."

### Step 2: Innovation Covariance S

```cpp
double S = P[0] + KALMAN_R_MEAS;  // Total uncertainty in the innovation
```

**Why?**  
The innovation has two sources of uncertainty:
1. Our prediction uncertainty (P₀₀)
2. Measurement noise (R_meas)

S combines both — it's the "expected variance" of the innovation if our model is correct.

### Step 3: Kalman Gain K (THE WEIGHTING FACTOR)

```cpp
double K0 = P[0] / S;  // How much to trust measurement vs. prediction for Z
double K1 = P[3] / S;  // How much to update dZ/dt based on Z innovation
double K2 = P[6] / S;  // How much to update d²Z/dt² based on Z innovation
```

**Physical interpretation:**  
- **K₀ ≈ 0.4**: Trust the measurement 40%, trust our prediction 60% for Z
- **K₁, K₂**: Even though we only measure Z (not dZ/dt directly), we use the correlation to update those states too!

**Why this is brilliant:**  
We're doing **sensorless estimation**. We only measure throughput (Z), but through the Kalman gain's off-diagonal elements, we can estimate dZ/dt and d²Z/dt² without direct sensors. This is exactly like sensorless FOC estimating rotor position from stator currents alone!

**Mathematical derivation:**
```
K = P_pred × Hᵀ × S⁻¹

Since H = [1 0 0]:
    Hᵀ = | 1 |
         | 0 |
         | 0 |

P_pred × Hᵀ extracts the first column of P_pred:
    | P₀₀  P₀₁  P₀₂ |   | 1 |   | P₀₀ |
    | P₁₀  P₁₁  P₁₂ | × | 0 | = | P₁₀ |
    | P₂₀  P₂₁  P₂₂ |   | 0 |   | P₂₀ |

So K = [P₀₀/S, P₁₀/S, P₂₀/S]ᵀ = [K0, K1, K2]ᵀ
```

### Step 4: State Correction

```cpp
x[0] = x0_p + K0 * y;  // Correct Z estimate
x[1] = x1_p + K1 * y;  // Correct dZ/dt estimate (even though we didn't measure it!)
x[2] = x2_p + K2 * y;  // Correct d²Z/dt² estimate
```

**Physical interpretation:**  
Our prediction was off by y=+0.047. We correct:
- Z by K₀×y (direct correction)
- dZ/dt by K₁×y (indirect — if Z increased unexpectedly, maybe the rate is higher than we thought!)
- d²Z/dt² by K₂×y (similarly for acceleration)

### Step 5: Covariance Update

```cpp
P[0] = (1 - K0) * P[0];  // Uncertainty decreases after measurement
// Other rows unchanged since H = [1, 0, 0] (we only measure Z)
```

**Why?**  
After incorporating a measurement, we're more certain about our estimates. The covariance shrinks by the factor (1 - K).

---

## Why Kalman Beats Simple Filtering

### Simple Moving Average (Naive Approach)

```cpp
// Naive exponential smoothing
z_filt = 0.9 * z_filt_prev + 0.1 * z_raw;
```

**Problems:**
| Issue | Impact |
|-------|--------|
| No predictive capability | Can't estimate dZ/dt → reactive only, not proactive |
| Fixed smoothing factor | Can't adapt to changing conditions (stable vs. noisy workloads) |
| Not optimal | Doesn't minimize mean-squared error (proven suboptimal) |
| Ignores physics | Treats impedance as random walk, not constant-acceleration system |

### Kalman Filter (GPUTronic Approach)

```cpp
kalman.update(dt, z_raw);  // Returns Z, dZ/dt, d²Z/dt²
```

**Advantages:**
| Advantage | Impact |
|-----------|--------|
| **Predictive** | Estimates dZ/dt and d²Z/dt² → can anticipate stalls before they happen |
| **Adaptive** | Kalman gains adjust automatically based on P and R (no manual gain scheduling) |
| **Optimal** | Minimizes mean-squared error (proven mathematically for linear-Gaussian systems) |
| **Physics-aware** | Uses constant-acceleration model, not just smoothing |

---

## FOC Analogy: Sensorless Rotor Position Estimation

### The Core Parallel

| Motor Control Domain | GPU Governor Domain | Why It's Analogous |
|----------------------|---------------------|-------------------|
| **Rotor position θ** | Impedance Z | Both are the "phase" we need to track for synchronous control |
| **Angular velocity ω = dθ/dt** | Rate dZ/dt | Both predict future state evolution |
| **Stator currents I_s, voltages V_s** | Throughput measurements (work counters) | Both are indirect observations of hidden state |
| **Back-EMF voltage E = kₑ × ω** | Work completion rate | Both reveal the "hidden" rotor/instruction phase |
| **Kalman observer** | Kalman observer | Same mathematical framework, different physical domain |

### In Sensorless FOC (Electric Motors)

You can't directly measure rotor position θ (no encoder). But you CAN:
1. Measure stator currents and voltages
2. Estimate back-EMF from those measurements
3. Use Kalman observer to combine physics model + measurements → estimate θ

**Physics model:** Electrical equations of PMSM
```
V_s = R_s × I_s + L_s × dI_s/dt + E_back_EMF
E_back_EMF = kₑ × ω × sin(θ)
```

### In GPU Governor (GPUTronic)

You can't directly measure "instruction phase" or future stalls. But you CAN:
1. Measure work completion counters (throughput)
2. Calculate impedance Z from expected vs. actual work
3. Use Kalman observer to combine physics model + measurements → estimate Z, dZ/dt, d²Z/dt²

**Physics model:** Constant-acceleration impedance evolution
```
Z(t+dt) = Z(t) + dt × dZ/dt(t) + 0.5×dt²×d²Z/dt²(t)
```

### The Deep Insight

Both systems have a **hidden state** (θ or Z) that affects observable behavior, and both use the same mathematical framework to estimate it optimally from indirect measurements. This is why FOC terminology maps so cleanly to GPU control:

- **"Rotor position"** = Instruction throughput phase (impedance Z)
- **"Back-EMF"** = Work completion rate (reveals the phase)
- **"Sensorless observer"** = Kalman filter estimating Z from counters alone
- **"Field-oriented"** = Decoupling Q-axis (useful work) from D-axis (thermal/power) and Z-axis (impedance)

---

## Predictive Stall Avoidance: The Real Power

This is where dZ/dt becomes critical for hyper-synchronous control.

### Traditional Reactive Control (Too Late!)

```cpp
// Old approach: wait until we're already stalled
if (z_raw > Z_HIGH_THRESH) {  // e.g., Z > 1.80
    // Already throttled down hard — we lost useful work!
    current_q -= aggressive_delta;
}
```

**Problem:** By the time Z crosses threshold, we've already experienced a stall. The damage is done.

### Predictive Proactive Control (GPUTronic v4+)

```cpp
double z_filt = kalman.getZ();       // Current impedance (filtered)
double z_rate = kalman.getZdot();    // Rate of change (PREDICTIVE!)
double pm = 1.0 / z_filt;            // Phase margin = 1/Z

// Predictive logic: anticipate future stalls
if (z_rate > 0.5 && z_filt < Z_HIGH_THRESH) {
    // Impedance is increasing rapidly, even though we're not at threshold yet
    // Proactive throttle down to avoid future stall!
    
    double time_to_threshold = (Z_HIGH_THRESH - z_filt) / z_rate;
    if (time_to_threshold < 0.5) {  // Will hit threshold in < 500ms?
        current_q -= 0.02;  // Small preemptive adjustment
    }
}

// Reactive logic (still needed for sudden jumps):
if (z_filt > Z_HIGH_THRESH) {
    current_q -= calc_delta_q(pm);  // Aggressive throttle down
}
```

### Example Scenario: Predicting a Stall Before It Happens

```
Time     Z_raw   Z_pred  Innovation  K0      Z_filt   dZ/dt    Governor Action
---------------------------------------------------------------------------------
t=0ms    1.25    1.20    +0.05       0.40    1.22     +0.3     Normal operation
t=10ms   1.28    1.23    +0.05       0.40    1.25     +0.4     Rate increasing!
t=20ms   1.32    1.27    +0.05       0.40    1.29     +0.5     Acceleration detected
t=30ms   1.38    1.32    +0.06       0.40    1.35     +0.7     Rapid increase! → PROACTIVE THROTTLE
t=40ms   1.42    1.40    +0.02       0.40    1.41     +0.6     Throttle working, rate slowing
```

**What's happening:**
- At t=30ms, dZ/dt jumped from +0.5 to +0.7 → governor sees this and throttles down preemptively by 0.02
- Without prediction: Z would hit 1.80 at ~t=60ms → stall!
- With prediction: We throttle early, Z stabilizes around 1.45 → no stall

**This is the "hyper-synchronous" control:** Anticipating back-EMF changes before they happen, just like FOC anticipates rotor position to maintain synchronous torque.

---

## Tuning Guidelines

### KALMAN_Q_PROC (Process Noise)

```cpp
constexpr double KALMAN_Q_PROC = 0.0005;  // v4 default
```

**What it controls:** How much we trust our physics model vs. measurements.

| Symptom | Adjustment | Reasoning |
|---------|------------|-----------|
| Z jumps around too much (noisy estimate) | Decrease Q_proc (try 0.001) | Trust model more, smooth out measurement noise |
| Governor reacts too slowly to changes | Increase Q_proc (try 0.01) | Trust measurements more, respond faster |
| PM oscillates around setpoint | Slightly decrease Q_proc | Reduce overshoot from over-reacting to noise |

### KALMAN_R_MEAS (Measurement Noise)

```cpp
constexpr double KALMAN_R_MEAS = 0.01;  // v4 default
```

**What it controls:** How much we trust our impedance measurements.

| Symptom | Adjustment | Reasoning |
|---------|------------|-----------|
| PM oscillates rapidly (high-frequency jitter) | Increase R_meas (try 0.05) | Trust model more, filter out measurement noise |
| Governor lags behind actual workload changes | Decrease R_meas (try 0.005) | Trust measurements more, respond faster |
| dZ/dt estimate is too noisy | Increase R_meas | Reduce sensitivity to measurement spikes |

### Practical Tuning Procedure

1. **Start with defaults:** Q_proc=0.0005, R_meas=0.01
2. **Run a stable workload** (e.g., GEMM matrix multiply)
3. **Observe Z_filt and dZ/dt:**
   - If Z_filt is too noisy → increase R_meas or decrease Q_proc
   - If dZ/dt doesn't respond to load changes → increase Q_proc
4. **Run a dynamic workload** (e.g., gaming with branching)
5. **Check predictive capability:** Does governor throttle before stalls occur?
6. **Iterate until:** Smooth Z_filt, responsive dZ/dt, no visible stalls

---

## Numerical Example: Step-by-Step Walkthrough

Let's trace through a complete Kalman update cycle with actual numbers.

### Initial State (Before Update)

```cpp
x[0] = 1.20;   // Z estimate
x[1] = 0.30;   // dZ/dt estimate (per second)
x[2] = 0.10;   // d²Z/dt² estimate (per second²)

// Covariance matrix P (simplified — only showing relevant elements)
P[0] = 0.50;   // Uncertainty in Z
P[3] = 0.20;   // Correlation between Z and dZ/dt
P[4] = 0.30;   // Uncertainty in dZ/dt
// ... (other elements omitted for clarity)

dt = 0.01;     // 10ms control interval
dt2 = 0.0001;  // dt²
```

### Step 1: State Prediction

```cpp
x0_p = 1.20 + 0.01*0.30 + 0.5*0.0001*0.10
     = 1.20 + 0.003 + 0.000005
     = 1.203005

x1_p = 0.30 + 0.01*0.10
     = 0.30 + 0.001
     = 0.301

x2_p = 0.10  // Unchanged
```

**Prediction:** "In 10ms, we expect Z ≈ 1.203, dZ/dt ≈ 0.301"

### Step 2: Covariance Propagation (Simplified)

```cpp
// F * P first column (we only need this for Kalman gain calculation)
fp00 = P[0] + dt*P[3] + 0.5*dt2*P[6]
     ≈ 0.50 + 0.01*0.20 + negligible
     = 0.502

// (F * P) * Fᵀ + Q
fpt00 = fp00 + dt*fp01 + 0.5*dt2*fp02 + Q_proc
      ≈ 0.502 + 0.01*(something) + negligible + 0.0005
      ≈ 0.503  // Uncertainty grew slightly
```

**Interpretation:** Uncertainty in Z increased from 0.50 to 0.503 due to process noise and prediction error accumulation.

### Step 3: Measurement Comes In

```cpp
z_raw = 1.25;  // Actual measurement from work counters
```

### Step 4: Innovation

```cpp
y = z_raw - x0_p
  = 1.25 - 1.203
  = +0.047

// Positive innovation means: "Impedance increased more than we predicted"
```

### Step 5: Innovation Covariance

```cpp
S = P[0] + R_meas
  = 0.503 + 0.01
  = 0.513

// Total uncertainty in the innovation
```

### Step 6: Kalman Gain

```cpp
K0 = P[0] / S = 0.503 / 0.513 ≈ 0.98
K1 = P[3] / S = 0.20 / 0.513 ≈ 0.39
K2 = P[6] / S = (assume 0.10) / 0.513 ≈ 0.19

// K0 ≈ 0.98 means: "Trust the measurement 98%, trust prediction only 2%"
// This makes sense because our uncertainty in Z (P[0]=0.503) is much higher than measurement noise (R=0.01)
```

### Step 7: State Correction

```cpp
x[0] = x0_p + K0 * y = 1.203 + 0.98*0.047 ≈ 1.249
x[1] = x1_p + K1 * y = 0.301 + 0.39*0.047 ≈ 0.319
x[2] = x2_p + K2 * y = 0.10 + 0.19*0.047 ≈ 0.109

// Updated estimates:
// Z increased from 1.20 to 1.249 (close to measurement of 1.25)
// dZ/dt increased from 0.30 to 0.319 (we learned the rate is higher than we thought!)
// d²Z/dt² increased from 0.10 to 0.109 (slight acceleration increase)
```

### Step 8: Covariance Update

```cpp
P[0] = (1 - K0) * P[0] = (1 - 0.98) * 0.503 ≈ 0.010

// Uncertainty in Z dropped dramatically from 0.50 to 0.01!
// We now have high confidence in our Z estimate after incorporating the measurement.
```

### Summary of This Cycle

| Quantity | Before | After | Change |
|----------|--------|-------|--------|
| Z | 1.200 | 1.249 | +0.049 (tracked measurement) |
| dZ/dt | 0.300 | 0.319 | +0.019 (learned rate is higher) |
| d²Z/dt² | 0.100 | 0.109 | +0.009 (slight acceleration increase) |
| P[0] (uncertainty in Z) | 0.503 | 0.010 | -98% (high confidence now!) |

**Key insight:** Even though we only measured Z, the Kalman filter updated dZ/dt and d²Z/dt² based on the innovation and their correlations with Z (P[3], P[6]). This is **sensorless estimation** in action!

---

## Implementation Notes

### Control Loop Timing

```cpp
// Target: 100kHz+ control rate (~10µs per cycle)
constexpr double CONTROL_DT = 0.00001;  // 10µs = 0.00001 seconds
constexpr double CONTROL_DT2 = CONTROL_DT * CONTROL_DT;

// In the main loop:
while (running) {
    double z_raw = measure_impedance();  // From work counters
    kalman.update(CONTROL_DT, z_raw);     // Predict + Update
    
    double z_filt = kalman.getZ();        // Use filtered Z for control
    double z_rate = kalman.getZdot();     // Use dZ/dt for prediction
    
    adjust_thread_fraction(z_filt, z_rate);  // Throttle control
    
    // Zero-copy mapped memory ensures <1µs read latency
}
```

### Numerical Stability

- **Use double precision** for Kalman calculations (float may lose precision over many cycles)
- **Clamp covariance matrix** to prevent negative values (numerical errors can cause this)
- **Reset P on startup** with high initial uncertainty (1.0 on diagonal)
- **Monitor innovation y**: If |y| is consistently large, your model or noise parameters are wrong

### Performance Considerations

The 3×3 matrix operations are lightweight:
- ~50 floating-point operations per update
- At 100kHz rate: ~5 MFLOPS (negligible compared to GPU workload)
- Can be implemented on CPU (pinned thread) or GPU (persistent kernel)

---

## References & Further Reading

### Kalman Filter Theory
- "Optimal Filtering" by Anderson and Moore (1979) — Classic text
- "Kalman Filtering: Theory and Practice" by Grewal and Andrews — Modern treatment
- [Wikibooks: Kalman Filter](https://en.wikibooks.org/wiki/Kalman_Filter) — Free online resource

### Sensorless FOC (Motor Control Analogy)
- "Sensorless Vector and Direct Torque Control" by Vas (1998)
- TI Application Report: "Sensorless Field Oriented Control of PMSM" (SPRA794)
- STMicroelectronics: "Sensorless FOC Algorithm" AN2653

### GPU Performance Modeling
- NVIDIA CUDA C++ Programming Guide — Warp scheduling, memory hierarchy
- "GPU Performance Analysis and Optimization" by Micikevicius et al.
- VAST Conference talks on GPU occupancy and stall analysis

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1 | 2026-05-XX | Initial 3-state Kalman observer implementation |
| v2 | 2026-05-XX | Added d²Z/dt² state for acceleration prediction |
| v3 | 2026-05-XX | Bug: Incorrect covariance propagation (P += Q only) |
| v4 | 2026-05-14 | **FIXED:** Proper F×P×Fᵀ + Q covariance propagation |

---

*Document generated for GPUTronic Stage 13 SCAG v13.0.2 — Self-Calibrating Adaptive Governor with Z-axis Kalman Observer*
