# GPUTronic Gold 1.0.0 — Function Frame

| Field | Value |
|-------|--------|
| Document | GT-FF-GOLD-1.0 |
| Title | Closed-loop impedance control of a cooperative GPU plant |
| Status | **Released** — Gold 1.0.0 (`1.0.0-gold`) |
| Hardware of record | NVIDIA GeForce RTX 5080 (GB203, 84 SM), CUDA 12.8, driver 595.x |
| Software of record | `include/gputronic.h`, `src/gputronic_gold.cu`, `src/gputronic_transfer.cu` |
| Gate | `make gold && ./build/gputronic_gold check` · `make transfer && ./build/gputronic_transfer check` |
| Author | Ben Coupland |
| Date | 2026-08-15 |
| Classification | Foundational. Do not treat later forks as this document unless they keep the contracts below. |

This is the function frame for Gold: what the system is, which signals exist, how each function computes, which constants are calibration parameters, where the C/CUDA lives, and what was measured. It is written to be read with the source open.

The claim is narrow. Instruction-flow throughput is a feedback variable. If the tachometer is honest and the actuator has authority, closed-loop control of that rate is the same job we already do on every other plant we take seriously. Gold is the interface that makes the job reproducible after earlier forks lost the tachometer.

---

## 0. How to read this document

1. Section 1 is the claim and the scope. If a later experiment leaves that scope, it is not Gold.
2. Section 2 is the signal list. Every later function only names signals from that list.
3. Sections F-01 … F-14 are the functions. Each has purpose, I/O, equations, calibration notes, and a code map.
4. Section 3 is the measured plant on this 5080. Numbers here are evidence, not calibration targets for other GPUs.
5. Section 4 is the guided walk through `gputronic_gold.cu`.
6. Section 5 is what Gold is not.

Function IDs (`F-04`, …) appear in comments in the source.

---

## 1. Purpose, scope, and out of scope

### 1.1 Purpose

Treat a GPU execution path as a controllable plant.

- Observe useful work rate (Q) from an aggregate pulse tachometer.
- Form an impedance proxy \(Z\) so free-run sits at 1.
- Estimate \([Z,\dot Z]\) with a 2-state Kalman filter.
- Actuate with `__nanosleep` over a mapped zero-copy cable.
- Leave thermal and power (D-axis) to the stock NVIDIA driver.

The control problem is ordinary: setpoint, observer, plant gain, actuator authority. The unusual part was getting an honest plant interface on a massively parallel device.

### 1.2 In scope

| Mode | Contract |
|------|----------|
| Self-test | Gold launches `gold_persistent_kernel`. Pulses and sleep live in one binary. |
| Cooperative | A foreign kernel publishes pulses into `total_work_pulses` and honours `throttle_sleep_ns`. Transfer plant is the existence proof. |

Both modes share the same host control thread, the same Z definition, the same PI.

### 1.3 Out of scope

- Uninstrumented third-party kernels (games, closed shaders, stock llama.cpp). That is a *contending* plant: different physics. Do not call it FOC.
- Thermal / NVML in the fast loop. Stock NVSMC owns D-axis. Polling at ~25 ms cannot sit in a 10 µs path.
- Claiming 100 kHz *effective* control. The path is 10 µs; under load the thread lands ~15–20 kHz. Rate for Z is formed over ~8 ms.

### 1.4 Lineage (why Gold exists)

| Era | What it proved | What it lost |
|-----|----------------|--------------|
| v0.1 (`archive/v0.1-github/main.cu`) | GPU as a controllable plant on GTX 1080; 10 Hz loop | Timescale too slow; “RPM” metaphor hid the plant |
| v14–v26 ladder | Persistent kernels, zero-copy, PI, Kalman, Cyberpunk coupling | Tachometer contract: last-SM overwrite; uncalibrated Z stuck at 0.25 / 4.0 |
| Gold 1.0.0 | Honest Q, sustained \(Z\), actuator that can move useful work on two plants | — |

Cyberpunk v1.13 (~Z 1.76, sleep ~9.5 µs, ~7350 PT frames) remains historical high-water for *game* coupling. It is not a Gold result. Different sign, different plant, different night.

---

## 2. Signals (the cable)

The only shared object between host and device is `struct GPUTronicControl`, allocated with `cudaHostAllocMapped`. On this 5080, host pointer equals device pointer.

Direction is from the control thread’s point of view.

### 2.1 Device → host (sensors)

| Signal | Type | Writer | Meaning |
|--------|------|--------|---------|
| `total_work_pulses` | `volatile ull` | Device `atomicAdd` (or host `gputronic_pulse`) | Aggregate Q tachometer. **Never** last-writer assign. |
| `sm_counters[sm]` | `ull` device mem | Warp leader | Diagnostic only. Not used for Z. |

### 2.2 Host → device (actuators / flags)

| Signal | Type | Writer | Meaning |
|--------|------|--------|---------|
| `throttle_sleep_ns` | `volatile int` | Control thread | Commanded `__nanosleep` duration |
| `max_sleep_ns` | `int` | Create | Clamp |
| `base_sleep_ns` | `int` | Create | Free-run / one-sided hold |
| `control_flags` | `volatile uint` | Host | bit0 STOP, bit1 PAUSE, bit2 RESET |
| `target_z` | `float` | Host | Echo for device/telemetry |

### 2.3 Host-only telemetry (snapshots)

`z_raw`, `z_hat`, `dzdt_hat`, `rate_ema`, `pm_error`, `integral`, `proactive_corrections`, `control_hz_ema`, `num_sm`.

Device may read `target_z`. It must not need the snapshots to run.

### 2.4 Internal (control thread, not on the cable)

| Symbol | Meaning |
|--------|---------|
| `rate_ref` | Sustained free-run pulse rate (1/s) |
| `rate_ema` | 8 ms window rate, then 0.70/0.30 EMA |
| `calibrated` | 0 until F-06 completes |
| `open_loop_sleep` | ≥0 forces that sleep; −1 = closed loop |
| `P[2][2]`, `z_hat`, `dzdt_hat` | Kalman state |
| `integral` | PI integrator, clamped |

---

## F-01 Zero-copy cable

**Purpose.** Put the control block in mapped pinned memory so the device re-reads `throttle_sleep_ns` without a 20–50 µs memcpy.

**Equation.** After `cudaHostAlloc(..., cudaHostAllocMapped)` and `cudaHostGetDevicePointer`:

\[
\texttt{d\_ctrl} = \mathrm{map}(\texttt{h\_ctrl})
\]

On this 5080, \(\texttt{h\_ctrl} = \texttt{d\_ctrl}\) as pointer values. That equality is sufficient, not necessary; a non-equal mapped pointer is still zero-copy. Gold records `zero_copy_ok = (d_ctrl != NULL)`.

**Code.** `gputronic_create` in `src/gputronic_gold.cu`. After writing `throttle_sleep_ns`, host issues `__sync_synchronize()`.

**Failure.** If mapping fails, create aborts. There is no PCIe fallback. That is deliberate: a slow cable is a different plant.

---

## F-02 Persistent self-test plant

**Purpose.** A known plant inside Gold so the dyno and closed-loop gate do not depend on a foreign binary.

**Topology.**

- Grid = `num_sm` (one block per SM).
- Block = `threads_per_block` (default 64 = 2 warps).
- Infinite loop until STOP.

**Inner work (per pulse).**

```
GOLD_BATCH (16) × GOLD_WORK_ITERS (8) × 8 fmaf
```

per thread, then one pulse from each warp leader (`lane == 0`). That is 2 pulses per SM per loop on the default launch.

The fmaf chain exists so the compiler cannot DCE the loop. It is not a FLOP claim. Counted FLOPs live on the transfer plant (F-14).

**Period on this 5080.** Mapped `atomicAdd` plus the persistent loop set \(T \approx 300\,\mu\mathrm{s}\), not the fmaf count. Batching 16× left pulse rate in the same band as a single tile would have: the atomic is the limiter. Sleep is applied *after* the batch, once per loop.

**Flags.**

| Flag | Device action |
|------|----------------|
| STOP | `break` |
| PAUSE | `__nanosleep(5000)` and continue (no pulse) |
| RESET | SM counter cleared by tid 0; no aggregate reset |

**Code.** `gold_work`, `gold_persistent_kernel`, `gold_launch_kernel`.

---

## F-03 Cooperative plant contract

**Purpose.** Let a foreign kernel be the plant without forking Gold.

**Must.**

1. `launch_selftest_kernel = 0`.
2. After `gputronic_create`, map `gputronic_get_control()` with `cudaHostGetDevicePointer`.
3. Launch the foreign persistent kernel *before or immediately after* `gputronic_start`, so F-06 sees pulses during the sustain window.
4. Increment `total_work_pulses` with `atomicAdd` (device) or `gputronic_pulse` (host, single-writer).
5. Read `throttle_sleep_ns` every chunk and `__nanosleep` it, clamped to `max_sleep_ns`.
6. Honour STOP.

**Must not.**

- Assign `total_work_pulses = sm_counters[sm]` (last-SM overwrite; late v26).
- Fence every pulse (`__threadfence_system` every 32 is the Gold pattern).
- Assume Gold’s `sleep_scale` before F-06 / F-09 have run.

**Code.** Contract text: this document. Example: `src/gputronic_transfer.cu` (`launch_transfer_plant`).

---

## F-04 Tachometer (Q-axis)

**Purpose.** One monotonically increasing integer whose delta is useful work.

**Regel.** Warp or block leaders only:

```
atomicAdd(&ctrl->total_work_pulses, 1)
```

Optional per-SM `atomicAdd(&sm_counters[sm], 1)` is diagnostic. Z never reads it.

**Host path.** `gputronic_pulse` does a non-atomic `+=`. Single-writer assumed. Do not mix host `+=` with device `atomicAdd` on the same run.

**Forbidden pattern (historical).**

```
total_work_pulses = sm_counters[sm];   /* last SM wins */
```

Symptom: Z pinned at floor or ceiling; PI never engages; “capacity” numbers from science fiction.

**Code.** Device: `gold_persistent_kernel` / `transfer_persistent_kernel`. Host: `gputronic_pulse`.

---

## F-05 Rate formation

**Purpose.** Turn pulse deltas into a rate the observer can use.

The control thread *wakes* on a 10 µs path (`control_dt_us`, with a 0.85× early-out). That is not the rate sample time.

**Algorithm.**

1. Each accepted tick: \(\Delta p\), \(\Delta t\) (clamped to \([10^{-7}, 10^{-2}]\) s).
2. Accumulate \(\sum\Delta p\), \(\sum\Delta t\).
3. When \(\sum\Delta t \ge 8\,\mathrm{ms}\):

\[
r_{\mathrm{win}} = \frac{\sum\Delta p}{\sum\Delta t},\qquad
r_{\mathrm{ema}} \leftarrow 0.70\,r_{\mathrm{ema}} + 0.30\,r_{\mathrm{win}}
\]

4. First valid sample: \(r_{\mathrm{ema}} \leftarrow r_{\mathrm{win}}\) (or the raw \(\Delta p/\Delta t\) if EMA is still empty).

**Why 8 ms.** A 10 µs EMA with “hold on empty tick” stayed at free-run (~500 kpulse/s) while a wall window already showed 314 kpulse/s at 500 µs sleep. The plant period is ~300 µs. You cannot observe a 300 µs plant with a 10 µs hold-EMA.

**Code.** `control_thread`, fields `rate_win_dt`, `rate_win_pulses`, `rate_ema`.

---

## F-06 Calibration (`rate_ref`)

**Purpose.** Define free-run so \(Z \approx 1\) when the actuator is at `base_sleep`.

**Sequence (closed-loop start).**

| Phase | Time | Actuator | Action |
|-------|------|----------|--------|
| Warmup | 0 – 0.7 s | `base_sleep_ns` | Discard. Let occupancy settle. |
| Sustain | 0.7 – 1.8 s | `base_sleep_ns` | Mark pulses and time at 0.7 s. |
| Close | t = 1.8 s | — | \(r_{\mathrm{ref}} = \Delta p / \Delta t\) over the 1.1 s window. |

Validity: \(\Delta t > 0.2\,\mathrm{s}\) and \(\Delta p > 1000\). Else fall back to `rate_ema` if it is already \(> 1000\).

**Not this.** Peak-hold of `rate_ema` during cal. Peak overshoot parked \(Z = r_{\mathrm{peak}}/r_{\mathrm{sustain}} \approx 1.4\text{–}1.7\) at free-run. One-sided PI then saw \(z \ge z^\star\) and never added sleep. That was Gold’s first closed-loop lie.

During cal, `z_raw` is published as 1.0 (nominal). The PI is held at base sleep.

**Code.** `control_thread` cal block; then `gold_apply_auto_scale` (F-09).

---

## F-07 Impedance \(Z\)

**Purpose.** A dimensionless plant output that is 1 at free-run and rises when rate falls.

\[
Z_{\mathrm{raw}} = \frac{r_{\mathrm{ref}}}{r_{\mathrm{ema}} + 1}
\]

Clamped to \([z_{\mathrm{floor}}, z_{\mathrm{ceiling}}] = [0.3, 10]\).

**Plant gain (this actuator).**

\[
\tau\uparrow \;\Rightarrow\; r\downarrow \;\Rightarrow\; Z\uparrow
\]

Positive. The PI has **no extra minus sign**.

**Reading.** \(Z = 1.5\) means the plant is delivering \(2/3\) of calibrated free-run rate. It is an impedance *proxy*, not ohms, not phase margin in degrees. The FOC names (Q, Z, D) are axes, not a claim that the GPU is a PMSM.

**Code.** `control_thread` after `calibrated == 1`.

---

## F-08 Kalman observer

**Purpose.** Filter \(Z_{\mathrm{raw}}\) and estimate \(\dot Z\) on a constant-velocity model.

**State.** \(x = [Z, \dot Z]^\top\).

**Predict.**

\[
\hat Z^- = \hat Z + \dot Z\,\Delta t,\qquad
\dot Z^- = \dot Z
\]

**Covariance predict** (explicit expansion of \(P^- = FPF^\top + Q\), \(F = \begin{bmatrix}1 & \Delta t\\ 0 & 1\end{bmatrix}\)):

\[
\begin{aligned}
p_{00}^- &= P_{00} + 2\Delta t\,P_{01} + \Delta t^2 P_{11} + Q \\
p_{01}^- &= P_{01} + \Delta t\,P_{11} \\
p_{10}^- &= P_{10} + \Delta t\,P_{11} \\
p_{11}^- &= P_{11} + Q
\end{aligned}
\]

Gold uses a scalar process \(Q = 8\cdot 10^{-4}\) added on the diagonal terms as written, and scalar measurement \(R = 1.8\cdot 10^{-2}\).

**Update.**

\[
y = Z_{\mathrm{raw}} - \hat Z^-,\quad
S = p_{00}^- + R,\quad
K_0 = p_{00}^-/S,\quad
K_1 = p_{10}^-/S
\]

\[
\hat Z \leftarrow \hat Z^- + K_0 y,\qquad
\dot Z \leftarrow \dot Z^- + K_1 y
\]

\(P\) is then reduced by the usual rank-one update coded in `kalman_update`.

**Note.** \(\dot Z\) is estimated. Gold 1.0’s PI does **not** yet feed \(\dot Z\) as a D-term. The estimate is on the cable for science and for a later Gold that might use it. Do not invent a derivative kick that is not in the source.

**Code.** `kalman_update`. Disabled if `enable_kalman = 0`; then \(z = Z_{\mathrm{raw}}\), \(\dot Z = 0\).

---

## F-09 Automatic `sleep_scale`

**Purpose.** Map a typical tracking error onto the actuator’s authority band.

On this 5080 the plant is flat to ~200 µs sleep and cuts hard near 500 µs. Stock `GPUTRONIC_GOLD_SLEEP_SCALE = 1.2\cdot 10^5` at \(e = 0.1\), \(K_p = 0.55\) commands

\[
u\cdot\mathrm{scale} \approx 0.055 \times 1.2\cdot 10^5 \approx 7\,\mu\mathrm{s}
\]

which is inside the dead zone. The loop “tracks” Z only if F-06 has already lied.

**Schedule** (after F-06, if `auto_sleep_scale = 1`):

\[
N_{\mathrm{src}} =
\begin{cases}
N_{\mathrm{SM}}\cdot (N_{\mathrm{thr}}/32) & \text{self-test (one pulse per warp)} \\
N_{\mathrm{SM}} & \text{cooperative (assume one pulse per SM/block)}
\end{cases}
\]

\[
T = \frac{N_{\mathrm{src}}}{r_{\mathrm{ref}}}\quad[\mathrm{s}],\qquad
\mathrm{scale} = \frac{1.25\,T_{\mathrm{ns}}}{K_p\cdot e_{\mathrm{des}}}
\]

with \(e_{\mathrm{des}} = 0.30\), then clamp scale to \([2\cdot 10^5, 8\cdot 10^6]\).

Measured on this 5080: self-test \(T \approx 340\,\mu\mathrm{s}\), scale \(\approx 2.5\text{–}2.9\cdot 10^6\); transfer \(T \approx 310\,\mu\mathrm{s}\), scale \(\approx 2.3\text{–}2.4\cdot 10^6\). Both put \(K_p e_{\mathrm{des}}\cdot\mathrm{scale}\) near 400 µs — the start of the cliff.

**Code.** `gold_apply_auto_scale`. The create-time LAW print still shows the *default* 120000; the live value is the `[CAL] auto sleep_scale=...` line.

---

## F-10 Tracking PI (one-sided)

**Purpose.** Drive \(Z\) toward \(z^\star\) by adding sleep, never by undersleeping below base.

**Engage.**

\[
\mathrm{engage} =
\begin{cases}
0 & \text{if } \texttt{one\_sided}\land z \ge z^\star \\
1 & \text{otherwise, once calibrated and not open-loop}
\end{cases}
\]

**Law (engage = 1).**

\[
e = z^\star - z,\qquad
I \leftarrow \mathrm{clip}(I + e\,\Delta t,\; \pm I_{\max})
\]

\[
u = K_p e + K_i I,\qquad
u \leftarrow 0 \text{ if } |e| < e_{\mathrm{db}}
\]

\[
\tau = \mathrm{clip}\big(\tau_{\mathrm{base}} + u\cdot\mathrm{scale},\; [0, \tau_{\max}]\big)
\]

**Disengage.** \(I \leftarrow 0\), \(\tau \leftarrow \tau_{\mathrm{base}}\) (max Q).

**Open-loop.** `open_loop_sleep >= 0` overrides \(\tau\) and zeros \(I\). Used by dyno and step.

**Sign.** No extra negation. Confirm on any new plant: sleep up must raise Z. If a future plant inverts (e.g. sleep up somehow raises rate), re-derive; do not flip this sign “because Cyberpunk did.”

**Deadband.** \(e_{\mathrm{db}} = 0.08\). Zeros \(u\) only, not \(I\). Integrator still accumulates inside the band. That is a known Gold 1.0 quirk, not a mistake to “fix” in-place without a new tag.

**Code.** `control_thread` PI block. Runtime: `gputronic_set_target`, `gputronic_set_gains` (Kp/Ki only; scale is not a runtime setter).

---

## F-11 Actuator

**Purpose.** Convert \(\tau\) into duty on the persistent loop.

Device, every chunk, after work and pulse:

```
sleep = clamp(ctrl->throttle_sleep_ns, 0, ctrl->max_sleep_ns)
if (sleep > 0) __nanosleep((unsigned)sleep)
```

**Authority on this 5080 (open-loop dyno, wall pulse Δ).**

| \(\tau\) | Gold self-test | Transfer GFLOP/s |
|---------:|---------------:|-----------------:|
| 0 | ~499 kpulse/s, \(Z\approx 1.00\) | 9.03 |
| 200 µs | ~469 kpulse/s, almost flat | 8.82 |
| 500 µs | ~314 kpulse/s, \(Z\approx 1.58\) | 5.15 |

Default \(\tau_{\max} = 500\,\mu\mathrm{s}\) so closed-loop *can* reach the cliff. Dyno raises it to 2 ms to sweep.

`__nanosleep` granularity and scheduler jitter are absorbed by F-05 / F-08. Do not chase sub-microsecond command precision on this plant.

---

## F-12 Host scheduler and lifecycle

```
gputronic_config_gold
gputronic_create          /* map cable, query SM count */
[cooperative: launch foreign kernel on d_ctrl]
gputronic_start           /* optional self-test launch + control pthread */
… run / dyno / set_target …
gputronic_stop            /* STOP flag, join, device sync, CSV close */
gputronic_destroy
```

Create installs SIGINT/SIGTERM → `g_signal_stop`. That is process-global. Two Gold instances in one process will share the signal flag.

CSV header:

```
time_s,z_raw,z_hat,dzdt,rate_ema,sleep_ns,error,integral,ctrl_hz,rate_ref
```

---

## F-13 Verification (dyno, step, closed-loop, transfer)

### Dyno (open-loop)

Sleep vector: \(0, 1, 5, 10, 20, 50, 100, 200, 500\,\mu\mathrm{s}\).

Rate is \((\Delta p)/(\Delta t_{\mathrm{wall}})\) after 1.5 s settle — **not** `rate_ema` alone.

Pass: free-run \(> 1000\) and 500 µs rate \(< 0.75\times\) free-run, **and** (\(R^2 \ge 0.85\) or mostly non-increasing with 15 % tolerance).

### Closed-loop gate (Gold)

After cal: useful-work window during the first second (still base sleep) versus windows from t ≥ 4 s. Pass requires tach alive, Z in \([0.3,10]\), rate sane, **and** closed window \(< 0.90\times\) free-run with the loop engaged.

Release numbers (this 5080): 445 → 324 kpulse/s (−27 %), \(Z_{\mathrm{mean}}\approx 1.37\), \(\tau\approx 415\,\mu\mathrm{s}\).

### Transfer gate

Same Gold controller. Second plant. Pass requires tile rate **and** GFLOP/s to drop together in dyno, and GFLOP/s to drop in closed-loop when Z is commanded to 1.5.

Release numbers: 8.77 → 6.55 GFLOP/s (−25 %), \(Z_{\mathrm{mean}}\approx 1.34\).

---

## F-14 Transfer plant (application of F-03)

**Purpose.** Prove F-03 on counted FLOPs, not on Gold’s synthetic fmaf.

- 8×8×8 GEMM in shared memory, 32 tiles per pulse = 32768 FLOP/pulse.
- One pulse per block (`tid == 0`), so F-09 uses \(N_{\mathrm{src}} = N_{\mathrm{SM}}\).
- `launch_selftest_kernel = 0`. No transfer-side `sleep_scale` override in 1.0.

If tile rate and GFLOP/s ever disagree in sign beyond noise, the tachometer is no longer measuring that work.

**Code.** `src/gputronic_transfer.cu`.

---

## 3. Calibration parameters (Gold 1.0.0)

| Name | Symbol | Default | Where | Why |
|------|--------|---------|-------|-----|
| `GPUTRONIC_DEFAULT_NUM_SM` | \(N_{\mathrm{SM}}\) | 84 (or query) | header | GB203 enablement |
| `THREADS_PER_BLOCK` | \(N_{\mathrm{thr}}\) | 64 | header | 2 warps; conservative occupancy |
| `CONTROL_DT_US` | — | 10 | header | Path period, not observer period |
| `MAX_SLEEP_NS` | \(\tau_{\max}\) | 500000 | header | Must reach the cliff |
| `BASE_SLEEP_NS` | \(\tau_{\mathrm{base}}\) | 5 | header | Near-zero free-run sleep |
| `TARGET_Z` | \(z^\star\) | 1.5 | header | 2/3 of free-run rate |
| `KP` | \(K_p\) | 0.55 | header | Cyberpunk-era starting point; works once scale is honest |
| `KI` | \(K_i\) | 0.08 | header | Same |
| `SLEEP_SCALE` | — | 120000 | header | **Fallback only.** F-09 overwrites |
| `INTEGRAL_CLAMP` | \(I_{\max}\) | 1.0 | header | Anti-windup |
| `DEADBAND` | \(e_{\mathrm{db}}\) | 0.08 | header | Quiets \(u\), not \(I\) |
| `Z_CEILING` / `Z_FLOOR` | — | 10 / 0.3 | header | Sanitise |
| `auto_sleep_scale` | — | 1 | config | F-09 on |
| `one_sided` | — | 1 | config | F-10 |
| `enable_kalman` | — | 1 | config | F-08 |
| Kalman \(Q\) | — | 0.0008 | source | Not a header calibration constant |
| Kalman \(R\) | — | 0.018 | source | Not a header calibration constant |
| Rate window | — | 8 ms | source | F-05 |
| EMA | — | 0.70 / 0.30 | source | F-05 |
| Cal warmup / total | — | 0.7 s / 1.8 s | source | F-06 |
| F-09 \(e_{\mathrm{des}}\) | — | 0.30 | source | Typical error for scale |
| F-09 period factor | — | 1.25 | source | Slightly more than one loop |
| `GOLD_BATCH` × `ITERS` | — | 16 × 8 | source | F-02 work per pulse |
| Transfer tile / batch | — | 8 / 32 | transfer | F-14 |

Changing a source-only calibration constant is a Gold tag, not a silent edit.

---

## 4. Guided walk through the source

Read in this order. Do not start at `main`.

1. **`include/gputronic.h`** — the cable (`GPUTronicControl`) and the calibration constants. If a signal is not here, the device cannot see it.
2. **`gold_work` / `gold_persistent_kernel`** — F-02, F-04, F-11. Confirm pulse is `atomicAdd`, sleep is after work, fence is every 32.
3. **`gputronic_create`** — F-01. Confirm mapped alloc and the LAW banner.
4. **`gputronic_start`** — launches F-02 only if configured; always starts F-12’s thread.
5. **`control_thread`** — F-05 → F-06 → F-07 → F-08 → F-10 → write cable. This is the control loop.
6. **`gold_apply_auto_scale`** — F-09, called once from F-06.
7. **`kalman_update`** — F-08. Check \(P^-=FPF^\top+Q\), not `P[i]+=Q`.
8. **`mode_dyno` / `mode_closedloop_check`** — F-13. Dyno rate is wall \(\Delta p/\Delta t\).
9. **`src/gputronic_transfer.cu`** — F-03 + F-14. Note `launch_selftest_kernel = 0` and no scale override.

`main` is a CLI multiplexer (`run`, `dyno`, `step`, `check`). It is not the function.

Handle fields that look redundant (`h_sm`, `proactive_corrections`) are telemetry or reserved. They are not in the Z equation.

---

## 5. Failure modes and anti-patterns

| Symptom | Likely cause | Function |
|---------|--------------|----------|
| \(Z\) stuck 0.25 or 4.0 | Uncalibrated legacy scale, or last-SM overwrite | F-04, F-06 |
| \(Z_{\mathrm{free}} \ge 1.5\), sleep stays at base | Peak-EMA `rate_ref` | F-06 |
| Sleep ~8 µs, useful work unchanged | `sleep_scale` still 1.2e5; F-09 off | F-09, F-11 |
| Sleep at \(\tau_{\max}\), \(Z_{\mathrm{ema}}\approx 1\), window rate already down | 10 µs hold-EMA | F-05 |
| Dyno FAIL, EMA flat, pulse Δ already down | Dyno used EMA instead of wall Δ | F-13 |
| Transfer GFLOP/s and tile rate disagree | Pulse no longer equals a tile | F-03, F-14 |
| “FOC of Cyberpunk” with no pulses | Observe-only or contending; out of scope | §1.3 |

---

## 6. Why this should look ordinary

Closed-loop regulation of a measured work rate is not a new idea. It is standard wherever the plant is worth controlling. Gold’s only unusual claim is that a GPU execution path is such a plant: you can observe useful work, form an impedance from it, and actuate without fighting the stock thermal path.

If that holds on real cooperative workloads, instruction-throughput control belongs in the same category as any other feedback loop we no longer argue about. Gold 1.0 is the interface and the evidence on two plants. Universality is still ahead.

---

## 7. Document control

| Version | Date | Note |
|---------|------|------|
| 1.0.0 | 2026-08-15 | First function frame. Matches Gold 1.0.0 as released on `bencoupland/GPUtronic` (`a50f840`), including transfer lessons folded into unpublished 1.0. |

If the source and this frame disagree, the **source plus the gate reports** win, and this document must be patched. Do not patch Gold to match a prettier equation.

Related files: `GOLD.md` (short), `STATUS.md` (living), `TRANSFER.md` (F-14 only), `archive/v0.1-github/` (ancestor tree).
