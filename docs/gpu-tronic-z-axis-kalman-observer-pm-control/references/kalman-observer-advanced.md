# GPUTronic Observer Extensions & Advanced Analysis

## Companion Document to Kalman Observer Mathematics

This document extends the core 3-state Kalman observer with:
1. **Signal flow diagrams** (visual representation)
2. **4+ state extensions** (thermal dynamics, memory pressure)
3. **Alternative observer comparisons** (Luenberger, EKF, UKF)
4. **Closed-loop stability analysis** (phase margin, gain margin, Bode plots)

---

## Part 1: Kalman Filter Signal Flow Diagrams

### Block Diagram: Complete Observer + Control Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GPUTronic Control Loop                            │
└─────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────┐
                                    │   Work       │
                                    │  Counters    │
                                    │ (Per-SM)     │
                                    └──────┬───────┘
                                           │
                                           ▼
                              ┌─────────────────────┐
                              │ z_raw = expected /  │
                              │ actual_work         │
                              │ (Noisy Measurement) │
                              └──────────┬──────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         │
        ┌─────────────────────┐                               │
        │   KALMAN OBSERVER   │                               │
        │                     │                               │
        │  ┌──────────────┐   │                               │
        │  │ PREDICT STEP │   │                               │
        │  │              │   │                               │
        │  │ x_pred = F×x │◄──┤ (Previous State)              │
        │  │              │   │                               │
        │  │ P_pred = F×P×Fᵀ+Q                              │   │
        │  └──────┬───────┘   │                               │
        │         │           │                               │
        │         ▼           │                               │
        │  ┌──────────────┐   │                               │
        │  │ UPDATE STEP  │   │                               │
        │  │              │   │                               │
        │  │ y = z_raw - x₀│  │ (Innovation)                  │
        │  │ S = P₀₀ + R  │   │                               │
        │  │ K = P×Hᵀ/S   │   │                               │
        │  │              │   │                               │
        │  │ x = x_pred + K×y│ │ (State Correction)           │
        │  │ P = (I-K×H)×P  │   │                               │
        │  └──────┬───────┘   │                               │
        │         │           │                               │
        └─────────┼───────────┘                               │
                  │                                           │
    ┌─────────────┴─────────────┐                             │
    ▼                           ▼                             │
┌─────────────┐         ┌─────────────┐                       │
│  z_filt     │         │  z_rate     │                       │
│  (Z)        │         │  (dZ/dt)    │◄──(Sensorless Est.)   │
│             │         │             │                       │
└──────┬──────┘         └──────┬──────┘                       │
       │                       │                              │
       ▼                       ▼                              │
┌─────────────────────────────────────────┐                   │
│        PHASE MARGIN CALCULATOR          │                   │
│                                         │                   │
│  pm = 1.0 / z_filt                      │◄──────────────────┘
│  target_pm = 0.85 (tunable)             │
│  error = target_pm - pm                 │
│                                         │
│  Z-axis deadband:                       │
│    if (z_filt < 1.05) → increase Q      │
│    if (z_filt > 1.80) → decrease Q      │
│    else → hold steady                   │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│              PI CONTROLLER (Q-Axis Torque)                  │
│                                                             │
│  integral += error * dt                                     │
│  clamp(integral, -0.5, +0.5)                               │
│                                                             │
│  delta_q = -(Kp * error + Ki * integral)                   │
│                                                             │
│  Kp = 0.3 (proportional gain)                              │
│  Ki = 0.05 (integral gain, anti-windup)                    │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│              THROTTLE ACTUATOR                              │
│                                                             │
│  current_q += delta_q                                       │
│  clamp(current_q, Q_MIN, Q_MAX)                            │
│                                                             │
│  // Zero-copy mapped memory update                         │
│  *throttle_ptr = current_q;  // <1µs latency               │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
        ┌───────────────┐
        │   GPU KERNEL  │
        │               │
        │  Thread       │
        │  Fraction     │◄──(Adjusts active SM occupancy)
        │  Control      │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │   Work        │
        │   Counters    │───(Back to start, cycle repeats)
        └───────────────┘

Control Loop Frequency: 100kHz+ (dt ≈ 10µs)
Total Latency: <20µs (measurement → actuation)
```

### State Transition Matrix Visualization

```
State Vector Evolution Over One Control Cycle:

Initial State (t):                    Predicted State (t+dt):

    ┌─────────┐                           ┌───────────────┐
    │  Z      │   ┌──────────────┐        │  Z_pred       │
    │  dZ/dt  │──►│   F MATRIX   │───────►│  dZ/dt_pred   │
    │ d²Z/dt² │   │              │        │ d²Z/dt²_pred  │
    └─────────┘   └──────────────┘        └───────────────┘
         │                                    │
         │         State Transition:          │
         │                                    │
         │  Z_pred     = Z + dt×dZ/dt + 0.5×dt²×d²Z/dt²    │
         │  dZ/dt_pred = dZ/dt + dt×d²Z/dt²                │
         │  d²Z/dt²_pred = d²Z/dt²                         │
         ▼                                    ▼

Measurement Update (when z_raw arrives):

    ┌───────────────┐     ┌──────────────┐      ┌─────────────┐
    │  Z_pred       │     │   INNOVATION │      │  Z_filt     │
    │  dZ/dt_pred   │────►│   y = z_raw -│─────►│  dZ/dt_filt │
    │ d²Z/dt²_pred  │     │   Z_pred     │      │ d²Z/dt²_filt│
    └───────────────┘     └──────────────┘      └─────────────┘
         │                       │                    │
         │                       ▼                    │
         │              ┌────────────────┐            │
         │              │ KALMAN GAIN K  │◄───────────┘
         │              │                │
         │              │ K₀ = P₀₀/S     │
         │              │ K₁ = P₁₀/S     │
         │              │ K₂ = P₂₀/S     │
         │              └───────┬────────┘
         │                      │
         ▼                      ▼

Final Corrected State:
    Z_filt     = Z_pred + K₀ × y
    dZ/dt_filt = dZ/dt_pred + K₁ × y  (sensorless!)
   d²Z/dt²_filt = d²Z/dt²_pred + K₂ × y  (sensorless!)
```

### Covariance Propagation Flow

```
Covariance Matrix Evolution:

Initial P(t):                    After Prediction P(t+dt):

┌─────────┐                      ┌──────────────────┐
│P₀₀ P₀₁ P₀₂│                   │P'₀₀ P'₀₁ P'₀₂  │
│P₁₀ P₁₁ P₁₂│   F×P×Fᵀ + Q      │P'₁₀ P'₁₁ P'₁₂  │
│P₂₀ P₂₁ P₂₂│ ────────────────► │P'₂₀ P'₂₁ P'₂₂  │
└─────────┘                      └──────────────────┘

Key Insight: Uncertainty doesn't just grow uniformly!

P'₀₀ (uncertainty in Z) depends on:
  - Original P₀₀ (Z uncertainty)
  - P₀₁ (correlation with dZ/dt) × dt
  - P₀₂ (correlation with d²Z/dt²) × dt²/2
  - Process noise Q_proc

This coupling is WHY proper matrix math matters!

After Measurement Update:

┌───────────────┐                ┌──────────────┐
│P'₀₀ P'₀₁ P'₀₂ │   (I-K×H)     │P''₀₀ P''₀₁  │
│P'₁₀ P'₁₁ P'₁₂ │ ────────────► │P''₁₀ P''₁₁  │
│P'₂₀ P'₂₁ P'₂₂ │               │P''₂₀ P''₂₁  │
└───────────────┘                └──────────────┘

Only first row shrinks significantly (we only measure Z):
    P''₀₀ = (1 - K₀) × P'₀₀  ← Dramatic reduction!
    Other rows change minimally
```

---

## Part 2: Extending to 4+ States

### Motivation for Higher-Order Models

The 3-state constant-acceleration model works well for impedance, but real GPU dynamics involve **multiple coupled physical processes**:

1. **Thermal dynamics** (D-axis): Temperature affects clock speeds, leakage current
2. **Memory pressure** (Z-axis extension): L2 cache misses, memory bandwidth saturation
3. **Power envelope** (new axis): Total board power vs. thermal design power (TDP)
4. **Frequency scaling lag**: GPU clocks don't change instantaneously

### Option A: 5-State Extended Model (Impedance + Thermal)

```cpp
// State vector: x = [Z, dZ/dt, d²Z/dt², T, dT/dt]ᵀ
double x[5];

// Physical meaning:
// x[0-2]: Impedance dynamics (same as 3-state)
// x[3]:   Temperature (normalized, e.g., T/100°C)
// x[4]:   Temperature rate of change
```

#### Extended State Transition Matrix (5×5)

```
        | 1   dt    0.5dt²  0      0     |
        | 0   1     dt      0      0     |
    F = | 0   0     1       0      0     |
        | 0   0     0       1      dt     |
        | 0   0     0       0      1      |

// Coupling terms (optional, more realistic):
// Temperature affects impedance: add -α×dt to F[0,3]
// Higher T → higher Z (thermal throttling)
F[0][3] = -alpha * dt;  // α ≈ 0.01 per °C (tune empirically)
```

#### Measurement Model for 5-State

We have **two measurements**:
1. Impedance Z (from work counters)
2. Temperature T (from NVML, slower ~1Hz)

```cpp
// Observation matrix H (2×5):
    | 1  0  0  0  0 |   // Measure Z directly
H = |               |
    | 0  0  0  1  0 |   // Measure T directly (slowly)

// Measurement noise R (2×2):
    | R_z     0    |   // Low noise for Z (fast counters)
R = |              |
    | 0      R_T   |   // High noise for T (slow, noisy NVML)
```

#### Implementation Challenges

| Challenge | Solution |
|-----------|----------|
| **Different measurement rates** | Run Kalman predict every dt=10µs, update Z every cycle, update T only when new NVML reading arrives |
| **NVML latency (~1ms)** | Use temperature prediction for fast control, NVML only for slow correction |
| **Increased computation** | 5×5 matrix ops ≈ 250 FLOPs vs. 3×3 ≈ 50 FLOPs (still negligible at 100kHz) |

### Option B: 6-State Dual-Z Model (Memory + Compute Impedance)

```cpp
// State vector: x = [Z_mem, dZ_mem/dt, Z_comp, dZ_comp/dt, pm, dpm/dt]ᵀ
double x[6];

// Physical meaning:
// x[0-1]: Memory impedance (L2 misses, bandwidth saturation)
// x[2-3]: Compute impedance (ALU stalls, warp divergence)
// x[4-5]: Phase margin dynamics (derived quantity for control)
```

#### Why Separate Memory vs. Compute?

Memory-bound and compute-bound workloads have **different stall characteristics**:

| Workload Type | Z Behavior | Control Strategy |
|---------------|------------|------------------|
| **Memory-bound** | High Z_mem, low Z_comp | Reduce memory pressure (lower occupancy) |
| **Compute-bound** | Low Z_mem, high Z_comp | Increase occupancy, hide latency |
| **Mixed** | Both elevated | Balance thread fraction + memory prefetch |

#### Measurement Model

```cpp
// We can't directly measure Z_mem vs. Z_comp separately!
// But we CAN estimate from:
// - L2 cache miss rate (from NVML or perf counters)
// - SM occupancy (from CUDA occupancy API)
// - Memory throughput (from NVML)

// Observation matrix H (3×6):
    | 1  0  0  0  0  0 |   // Total Z = Z_mem + Z_comp (approx)
H = | 0  0  1  0  0  0 |   // L2 miss rate correlates with Z_mem
    | 0  0  0  0  1  0 |   // Phase margin from work counters

// This is a "weakly observable" system — need careful tuning!
```

### Option C: 7-State Full Thermal-Power-Impedance Model

```cpp
// State vector: x = [Z, dZ/dt, T, dT/dt, P, dP/dt, f]ᵀ
double x[7];

// Physical meaning:
// x[0-1]: Impedance (Z) and rate
// x[2-3]: Temperature (T) and rate
// x[4-5]: Power (P) and rate
// x[6]:   GPU clock frequency (f, from NVML)
```

#### Coupled Dynamics Matrix (7×7)

```
        | 1    dt    0     0      0     0     -β×dt  |
        | 0    1     0     0      0     0      0     |
        | 0    0     1     dt     γ×dt  0      0     |  // T rises with P
    F = | 0    0     0     1      0     0      0     |
        | 0    0     δ×dt  0      1     dt     0     |  // P depends on T (leakage)
        | 0    0     0     0      0     1      0     |
        | 0    0     0     0      0     0      1     |

// Coupling coefficients (tune empirically):
// β: Frequency scaling affects impedance (higher f → lower Z)
// γ: Power dissipation raises temperature
// δ: Temperature increases leakage power
```

#### Practical Considerations

| Factor | Recommendation |
|--------|----------------|
| **NVML update rate** | ~1Hz for T/P, too slow for fast control. Use as "slow correction" only |
| **Computation cost** | 7×7 matrix ≈ 500 FLOPs per cycle (still <1µs on CPU) |
| **Observability** | Need multiple sensors: work counters + NVML T/P + clock frequency |
| **Numerical stability** | Use double precision, clamp covariance to prevent divergence |

### When to Use Higher-Order Models

| Scenario | Recommended Model | Why |
|----------|-------------------|-----|
| **Stable workload** (GEMM, inference) | 3-state (current) | Impedance dominates, thermal is slow |
| **Dynamic workload** (gaming, mixed) | 5-state (impedance + thermal) | Thermal throttling becomes relevant |
| **Memory-intensive** (large GEMM, graph NN) | 6-state (dual-Z) | Memory vs. compute decoupling helps |
| **Power-constrained** (laptop, edge) | 7-state (full model) | Power envelope management critical |

### Implementation Pattern for Multi-Rate Measurements

```cpp
// Main control loop: 100kHz (dt = 10µs)
while (running) {
    // Fast measurement: work counters (every cycle)
    double z_raw = measure_impedance();
    
    // Kalman predict step (always)
    kalman.predict(dt);
    
    // Kalman update with Z (every cycle)
    kalman.update_z(z_raw);
    
    // Slow measurement: NVML temperature/power (only when available)
    if (nvml_new_reading()) {
        double T_raw = nvml_get_temperature();
        double P_raw = nvml_get_power();
        kalman.update_slow(T_raw, P_raw);  // Different H matrix for slow sensors
    }
    
    // Control output
    double z_filt = kalman.getZ();
    double T_est = kalman.getTemperature();  // Predicted temperature (fast!)
    
    adjust_throttle(z_filt, T_est);
}

// Key insight: Kalman filter provides "virtual sensors" for fast estimates
// of slow-measured quantities (like temperature) through state coupling!
```

---

## Part 3: Alternative Observer Comparisons

### Luenberger Observer

#### What Is It?

A **deterministic observer** that estimates states using a model and correction term:

```
x̂ₖ₊₁ = F × x̂ₖ + L × (yₖ - H × x̂ₖ)

Where:
- x̂ is the state estimate
- y is the measurement
- L is the observer gain matrix (designed, not computed)
```

#### Comparison with Kalman Filter

| Aspect | Luenberger | Kalman Filter |
|--------|------------|---------------|
| **Gain design** | Manual (pole placement) | Optimal (computed from Q, R) |
| **Noise handling** | Assumes deterministic | Explicitly models noise (Q, R) |
| **Optimality** | Suboptimal (no cost function) | Minimum mean-squared error |
| **Tuning complexity** | High (need control theory expertise) | Medium (tune Q, R intuitively) |
| **Computational cost** | Lower (no covariance updates) | Higher (P matrix propagation) |
| **Adaptivity** | Fixed gain | Adaptive gains (K changes each cycle) |

#### When to Use Luenberger

**Pros:**
- Simpler implementation (no 3×3 matrix math for P)
- Faster execution (~10µs vs. ~15µs per cycle)
- Deterministic behavior (no stochastic elements)

**Cons:**
- No optimality guarantee
- Poor noise rejection without careful tuning
- Gain L must be manually designed (eigenvalue placement)

#### Example: 3-State Luenberger for GPUTronic

```cpp
// Observer gain L (designed via pole placement):
double L[9] = {
    0.8,  0.0,  0.0,   // Z correction from innovation
    0.5,  0.3,  0.0,   // dZ/dt correction
    0.2,  0.1,  0.1    // d²Z/dt² correction
};

// Observer update (simplified):
void luenberger_update(double z_raw, double dt) {
    // Predict
    double x_pred[3] = {
        x[0] + dt*x[1] + 0.5*dt2*x[2],
        x[1] + dt*x[2],
        x[2]
    };
    
    // Innovation
    double y = z_raw - x_pred[0];
    
    // Correct (L × innovation)
    x[0] = x_pred[0] + L[0]*y;
    x[1] = x_pred[1] + L[3]*y;  // Note: row-major indexing
    x[2] = x_pred[2] + L[6]*y;
}
```

**Verdict for GPUTronic:** Kalman filter is superior because:
1. Impedance measurements are noisy (work counters have jitter)
2. Adaptive gains help with varying workload characteristics
3. Optimality matters when control bandwidth is limited (~100kHz)

---

### Extended Kalman Filter (EKF)

#### What Is It?

The EKF handles **non-linear systems** by linearizing around the current estimate:

```
x̂ₖ₊₁ = f(x̂ₖ) + Kₖ × (yₖ - h(x̂ₖ))

Where:
- f(·) is the non-linear state transition function
- h(·) is the non-linear measurement function
- Kₖ uses Jacobians: F = ∂f/∂x, H = ∂h/∂x (evaluated at x̂ₖ)
```

#### When Would GPUTronic Need EKF?

Our current model is **linear**:
- State transition: Z(t+dt) = Z(t) + dt×dZ/dt + ... (linear in states)
- Measurement: z_raw = Z + noise (linear)

**EKF would be needed if:**
1. Impedance dynamics are non-linear (e.g., Z ∝ 1/√threads for some workloads)
2. Temperature coupling is non-linear (Arrhenius equation for leakage)
3. Frequency scaling has non-linear effects on throughput

#### Example: Non-Linear Impedance Model

```cpp
// Hypothetical non-linear model:
// Z depends on thread fraction q in a non-linear way:
// Z = Z₀ + α × (1 - q)^β  where β ≈ 2.5 (empirical)

double nonlinear_impedance(double q, double base_Z) {
    return base_Z + alpha * pow(1.0 - q, 2.5);
}

// EKF would need Jacobian:
// H = ∂z/∂x evaluated at current estimate
// For this model: H = [1, 0, 0] still (we measure Z directly)
// But F matrix changes if dynamics are non-linear!
```

#### EKF Implementation Complexity

| Factor | Linear KF | Extended KF |
|--------|-----------|-------------|
| **State prediction** | x_pred = F × x | x_pred = f(x) (non-linear function) |
| **Covariance propagation** | P = F×P×Fᵀ + Q | P = Fⱼₐc×P×Fⱼₐcᵀ + Q (Jacobian!) |
| **Tuning** | Q, R matrices | Q, R + Jacobian accuracy |
| **Numerical stability** | Good (well-conditioned) | Can diverge if linearization poor |
| **Computation** | ~50 FLOPs (3×3) | ~200+ FLOPs (Jacobian + 3×3) |

#### Verdict for GPUTronic

**Stick with linear Kalman filter unless:**
- You observe systematic prediction errors that correlate non-linearly with states
- Temperature coupling becomes dominant (Arrhenius leakage model)
- Frequency scaling effects are strongly non-linear

**Test for non-linearity:**
```cpp
// Run a sweep of thread fractions and measure Z:
for (double q = 0.1; q <= 1.0; q += 0.1) {
    double z = measure_impedance_at_fraction(q);
    printf("q=%.1f, Z=%.3f\n", q, z);
}

// If Z vs. q is well-fit by a line → linear KF is fine
// If Z vs. q needs quadratic/cubic fit → consider EKF
```

---

### Unscented Kalman Filter (UKF)

#### What Is It?

The UKF avoids Jacobians entirely by using **sigma points**:

1. Generate 2n+1 sigma points around current estimate (n = state dimension)
2. Propagate each point through non-linear function f(·)
3. Compute new mean and covariance from transformed points

**Advantages over EKF:**
- No Jacobian calculation (easier to implement)
- More accurate for strongly non-linear systems
- Numerically more stable

**Disadvantages:**
- Higher computational cost (~10× more FLOPs)
- Still requires tuning process/measurement noise

#### UKF Computational Cost for 3-State

```
Sigma points: 2×3 + 1 = 7 points
Each point propagated through f(·): ~50 FLOPs
Total per cycle: 7 × 50 = 350 FLOPs (vs. 50 FLOPs for linear KF)

At 100kHz rate: 35 MFLOPS vs. 5 MFLOPS
```

#### Verdict for GPUTronic

**Not recommended unless:**
- You have strong non-linearities that EKF can't handle well
- You have CPU headroom (pinned thread has spare cycles)
- You've benchmarked and proven linear KF is insufficient

For now, **linear Kalman filter is the right choice**: simple, fast, and adequate for the physics model.

---

### Particle Filter (Sequential Monte Carlo)

#### What Is It?

A **non-parametric** approach using weighted particles:

1. Maintain N particles (state hypotheses) with weights
2. Propagate each particle through process model + noise
3. Update weights based on measurement likelihood
4. Resample to avoid weight degeneracy

**Advantages:**
- Handles any non-linearity and non-Gaussian noise
- Can represent multi-modal distributions (e.g., "GPU is either stalled or not")

**Disadvantages:**
- Very high computational cost (N particles, N ≥ 100 typically)
- Weight degeneracy requires resampling (complex)
- Hard to tune (number of particles, proposal distribution)

#### Verdict for GPUTronic

**Absolutely not.** Particle filters are overkill:
- Our noise is approximately Gaussian (central limit theorem on many warp schedulers)
- Our dynamics are linear (constant-acceleration model)
- 100kHz control rate leaves no room for N=100 particle propagation

---

### Summary: Observer Selection Matrix

| Observer | Complexity | Non-linear? | Optimal? | GPUTronic Fit |
|----------|------------|-------------|----------|---------------|
| **Linear Kalman** | Low (50 FLOPs) | No | Yes (linear-Gaussian) | ✅ **BEST** |
| Luenberger | Lowest (~20 FLOPs) | No | No | ⚠️ Possible, but suboptimal |
| Extended KF | Medium (200+ FLOPs) | Yes | Approximate | ❌ Only if non-linearities proven |
| Unscented KF | High (350+ FLOPs) | Yes | Approximate | ❌ Overkill for current model |
| Particle Filter | Very High (N×100s FLOPs) | Yes | Monte Carlo approx | ❌ Way too expensive |

---

## Part 4: Closed-Loop Stability Analysis

### Control System Block Diagram

```
                    ┌─────────────────────────────────────┐
                    │                                     │
        r(t) ─────►│ (+)    e(t)      ┌──────────┐     │     ┌──────────┐     y(t)
   (target_pm)     │◄───────◄────────│  PI      │────►│────►│  GPU     │────►│ (Z, pm)
                   │        │        │ Controller│     │     │ Plant    │     │
                   │        │        └──────────┘     │     │          │     │
                   │        │                         │     └────┬─────┘     │
                   │        │                         │          │           │
                   │        │              ┌──────────┴┐        │           │
                   │        └────────────►│ Kalman    │◄────────┘           │
                   │                      │ Observer  │                     │
                   │                      └───────────┘                     │
                   └────────────────────────────────────────────────────────┘

Closed-loop transfer function: T(s) = G_c(s) × G_p(s) / (1 + G_c(s) × G_p(s))

Where:
- G_c(s): Controller transfer function (PI controller)
- G_p(s): Plant transfer function (GPU dynamics + observer)
```

### PI Controller Transfer Function

```cpp
// Discrete-time PI controller:
delta_q[k] = -(Kp * error[k] + Ki * integral[k])
integral[k] = integral[k-1] + error[k] * dt

// Z-transform (discrete):
C(z) = -(Kp + Ki × T / (z - 1))

// Continuous approximation (for stability analysis):
G_c(s) = -(Kp + Ki/s) = -(Kp×s + Ki) / s

// Poles: s = 0 (integrator, marginally stable)
// Zeros: s = -Ki/Kp
```

### Plant Transfer Function (GPU Dynamics)

The GPU plant is complex, but we can approximate it as a **first-order system with delay**:

```
G_p(s) = K / (τ×s + 1) × e^(-s×T_delay)

Where:
- K: Static gain (how much Z changes per unit q change) ≈ -0.5 to -2.0 (negative!)
- τ: Time constant (how fast GPU responds to thread fraction change) ≈ 1-10ms
- T_delay: Pure delay (kernel launch latency + measurement lag) ≈ 50-200µs

// First-order Padé approximation for delay:
e^(-s×T) ≈ (1 - s×T/2) / (1 + s×T/2)
```

### Open-Loop Transfer Function

```
L(s) = G_c(s) × G_p(s)
     = [-(Kp×s + Ki)/s] × [K/(τ×s + 1)] × [(1 - s×T/2)/(1 + s×T/2)]

     = -K × (Kp×s + Ki) × (1 - s×T/2) / [s × (τ×s + 1) × (1 + s×T/2)]
```

### Bode Plot Analysis

#### Key Frequencies

| Frequency | Significance |
|-----------|--------------|
| **ω_c** (crossover) | Where \|L(jω)\| = 1 (0 dB). Determines bandwidth. |
| **ω_pc** (phase crossover) | Where ∠L(jω) = -180°. Determines stability margin. |
| **ω_n** (natural frequency) | Closed-loop resonant frequency. |

#### Phase Margin Calculation

```cpp
// Phase margin: PM = 180° + ∠L(jω_c)
// Where ω_c is the gain crossover frequency (\|L(jω_c)\| = 1)

// For our system:
// - PI controller adds -90° (integrator) at low freq, approaches 0° at high freq
// - First-order plant adds up to -90° phase lag
// - Delay adds -ω×T radians of phase lag

// Total phase at crossover:
∠L(jω_c) ≈ -90° (PI) - arctan(ω_c×τ) (plant) - ω_c×T_delay (delay in radians)

// Example calculation:
// Kp = 0.3, Ki = 0.05, K = -1.0, τ = 5ms, T_delay = 100µs
// ω_c ≈ 100 rad/s (crossover at ~16 Hz)

∠L(j100) ≈ -90° - arctan(100×0.005) - 100×0.0001×(180/π)
         ≈ -90° - 2.9° - 0.6°
         ≈ -93.5°

PM = 180° + (-93.5°) = 86.5°  // Very stable!
```

#### Gain Margin Calculation

```cpp
// Gain margin: GM = 1 / \|L(jω_pc)\|
// Where ω_pc is the phase crossover frequency (∠L(jω_pc) = -180°)

// For our system, phase never reaches -180° because:
// - Maximum phase lag from PI + plant ≈ -180° (at very high freq)
// - But delay adds only a few degrees at crossover

// So GM is effectively infinite (system is unconditionally stable for this model)

// In practice, unmodeled dynamics (higher-order effects) will limit GM:
// Typical GM > 6 dB is considered safe.
```

### Stability Margins: Target Values

| Metric | Minimum | Recommended | Ideal |
|--------|---------|-------------|-------|
| **Phase Margin** | 30° | 45-60° | 60-80° |
| **Gain Margin** | 3 dB | 6-10 dB | >10 dB |
| **Crossover Frequency** | — | 1/10 of control rate | Maximize (within limits) |

### Tuning PI Gains for Stability

```cpp
// Current GPUTronic defaults:
constexpr double Kp = 0.3;   // Proportional gain
constexpr double Ki = 0.05;  // Integral gain

// Effect of increasing Kp:
// - Increases crossover frequency (faster response)
// - Decreases phase margin (less stable, more overshoot)
// - May cause oscillation if too high

// Effect of increasing Ki:
// - Adds low-frequency gain (better steady-state error)
// - Decreases phase margin (integrator adds -90°)
// - Can cause windup and instability if too high

// Tuning procedure:
// 1. Start with Ki = 0, increase Kp until ~15° overshoot
// 2. Add Ki slowly until steady-state error is acceptable
// 3. Verify phase margin > 45° via Bode plot or step response
```

### Step Response Analysis

#### Ideal Step Response (Well-Tuned)

```
      y(t)
       │
   1.0 ├───────────────┐
       │               │ Stable, no oscillation
       │          ┌────┘
       │     ┌────┘
       │  ┌──┘
   0.5 ├─┘
       │
       └──────────────────────────► t
         0    10ms   20ms   30ms

Rise time: ~10-20ms (acceptable for thermal/impedance control)
Overshoot: <15% (good damping)
Settling time: ~30ms (within 2% of final value)
```

#### Poorly Tuned: High Kp (Underdamped)

```
      y(t)
       │
   1.0 ├─────────┐
       │        ╱╲
       │     ┌─╱  ╲──┐
       │    ╱╱    ╲╲ │
       │  ─╱╱      ╲╲╲────
   0.5 ├─╱╱         ╲╲╲
       │╱             ╲╲
       └───────────────────► t
         0    10ms   20ms

Overshoot: >30% (too aggressive)
Oscillation: Visible ringing
Risk: May cause thermal runaway if coupled with D-axis
```

#### Poorly Tuned: High Ki (Windup)

```
      y(t)
       │
   1.0 ├───────────────┐
       │              ╱│
       │             ╱ │
       │            ╱  │
       │           ╱   │
       │          ╱    │
   0.5 ├────────╱      │
       │       ╱       │
       └──────╱────────┼─────────► t
         0   20ms     40ms      60ms

Slow rise (Ki too small) OR
Huge overshoot + long settling (Ki too large, windup)
```

### Nyquist Stability Criterion

For the closed-loop system to be stable:

**The Nyquist plot of L(jω) must not encircle the point (-1, 0).**

```
Imaginary Axis
    │
    │        ╭─────╮
    │      ╱         ╲
    │     │           │  ← Nyquist contour (L(jω))
    │     │           │
    │      ╲         ╱
    │        ╰───╯
    │            ● (-1, 0)  ← Critical point
────┼───────────────────────► Real Axis
    │
```

For our PI + first-order plant system:
- Nyquist plot stays in right half-plane (doesn't encircle -1)
- System is **unconditionally stable** for this simplified model
- In practice, unmodeled high-frequency dynamics may cause issues

### Routh-Hurwitz Stability Criterion

For the characteristic equation: `1 + L(s) = 0`

```
Characteristic polynomial (approximate, ignoring delay):

s³ + (1/τ + Kp×K/τ)s² + (Ki×K/τ)s + ... = 0

Routh array:
s³ │  1               Ki×K/τ
s² │  1/τ+Kp×K/τ      0
s¹ │  (1/τ+Kp×K/τ)×(Ki×K/τ) / (1/τ+Kp×K/τ) = Ki×K/τ
s⁰ │  Ki×K/τ

For stability: All first column elements > 0

Conditions:
1. 1/τ + Kp×K/τ > 0  →  Kp > -1/K (always true since K < 0, Kp > 0)
2. Ki×K/τ > 0        →  Ki < 0 (since K < 0)!

Wait, this suggests Ki must be negative! But we use positive Ki...

Resolution: The sign convention in our controller is delta_q = -(Kp*e + Ki*integral),
so the effective Ki in the transfer function is -Ki. With this correction:

Condition becomes: -Ki×K/τ > 0  →  Ki > 0 (since K < 0) ✓

So positive Ki is correct for our sign convention!
```

### Robustness Analysis: Sensitivity to Plant Variations

How does the system behave if GPU dynamics change?

| Variation | Effect on Stability | Mitigation |
|-----------|---------------------|------------|
| **K changes ±50%** (workload-dependent gain) | Phase margin varies by ~10° | Adaptive Kp based on measured Z rate |
| **τ changes 2×** (different GPU architecture) | Crossover frequency shifts | Auto-tune based on step response |
| **T_delay increases** (OS jitter, driver latency) | Phase margin decreases | Increase deadband, reduce bandwidth |

### Adaptive Gain Scheduling (Advanced)

```cpp
// Gain scheduling: adjust Kp, Ki based on operating point

double adaptive_kp(double z_filt, double z_rate) {
    // Base gain
    double kp = 0.3;
    
    // Reduce gain at high impedance (near stall)
    if (z_filt > 1.5) {
        kp *= 0.7;  // More conservative near instability
    }
    
    // Increase gain for fast-changing workloads
    if (fabs(z_rate) > 0.5) {
        kp *= 1.2;  // Faster response to dynamics
    }
    
    return clamp(kp, 0.1, 0.5);
}

double adaptive_ki(double z_filt) {
    // Base gain
    double ki = 0.05;
    
    // Reduce integral action near stall (prevent windup)
    if (z_filt > 1.6) {
        ki *= 0.5;
    }
    
    return clamp(ki, 0.01, 0.1);
}
```

### Summary: Stability Recommendations

| Aspect | Recommendation | Rationale |
|--------|----------------|-----------|
| **Phase Margin** | Target 60° ± 15° | Good trade-off between speed and stability |
| **PI Gains** | Kp=0.3, Ki=0.05 (defaults) | Verified stable for typical GPU dynamics |
| **Deadband** | Z ∈ [1.05, 1.80] | Prevents hunting around setpoint |
| **Anti-windup** | Clamp integral to ±0.5 | Prevents integral windup during large errors |
| **Adaptive gains** | Optional (advanced) | Improves robustness across workloads |

---

## Part 5: Practical Stability Testing

### Dyno Sweep Test Plan

```cpp
// Test procedure to verify stability margins empirically:

void stability_test_plan() {
    // 1. Step response test
    set_target_pm(0.85);
    apply_step_change_in_load();  // e.g., switch from idle to 100% compute
    
    measure:
    - Rise time (10% → 90%)
    - Overshoot (%)
    - Settling time (within 2%)
    - Steady-state error
    
    // 2. Frequency response test
    for (double freq = 0.1; freq <= 50.0; freq += 0.5) {
        apply_sinusoidal_load_modulation(freq);  // Load varies as sin(2πft)
        measure_amplitude_ratio();
        measure_phase_shift();
        
        printf("f=%.1fHz, gain=%.2f, phase=%.1f°\n", 
               freq, amp_ratio, phase_deg);
    }
    
    // 3. Disturbance rejection test
    apply_step_disturbance();  // e.g., sudden thermal event
    measure_recovery_time();
    
    // 4. Robustness test
    for (double temp = 40; temp <= 85; temp += 10) {
        set_ambient_temperature(temp);
        run_stability_test();
        printf("T=%d°C, PM=%.1f°, GM=%.1fdB\n", temp, pm, gm);
    }
}
```

### Bode Plot Measurement (Empirical)

```cpp
// Inject sinusoidal perturbation in throttle and measure response:

double bode_test_frequency = 5.0;  // Hz
double bode_amplitude = 0.02;      // ±2% thread fraction modulation

void bode_test_loop() {
    double t = 0;
    while (bode_test_running) {
        // Inject sinusoidal perturbation
        double perturbation = bode_amplitude * sin(2 * M_PI * bode_test_frequency * t);
        current_q = base_q + perturbation;
        
        // Measure response
        double z_measured = measure_impedance();
        
        // Store for analysis (compute amplitude ratio and phase shift)
        bode_data.push({t, z_measured});
        
        t += dt;
    }
    
    // Post-process: compute FFT, extract gain/phase at test frequency
    bode_analysis(bode_data, bode_test_frequency);
}
```

---

## References & Further Reading

### Control Theory
- "Modern Control Engineering" by Ogata (5th ed.) — Bode plots, Nyquist, Routh-Hurwitz
- "Feedback Control of Dynamic Systems" by Franklin, Powell, Emami-Naeini — PID tuning, stability margins
- [Control Systems Tools (MATLAB)](https://www.mathworks.com/products/control.html) — Bode plot generation

### Kalman Filtering
- "Optimal State Estimation" by Simon — Comprehensive treatment of KF variants
- [Kalman Filter Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter) — Good overview with equations

### GPU Performance Modeling
- NVIDIA "Occupancy Calculator" spreadsheet — Thread fraction vs. occupancy
- "Understanding GPU Performance" by AMD (applicable to NVIDIA too)
- VAST Conference: "GPU Architecture and Performance" talks

---

*Document generated for GPUTronic Stage 13 SCAG v13.0.2 — Advanced Observer Extensions & Stability Analysis*
