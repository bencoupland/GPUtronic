# GPUTronic 1.0 Gold — Function Frame

| Field | Value |
|-------|--------|
| Document | GT-FF-GOLD-1.0 |
| Title | Closed-loop impedance control of a cooperative GPU plant |
| Status | **Released** — GPUTronic 1.0 Gold (`1.0.0-gold`) |
| Hardware of record | NVIDIA GeForce RTX 5080 (GB203, 84 SM), CUDA 12.8, driver 595.x |
| Software of record | `include/gputronic.h`, `src/gputronic_gold.cu`, `src/gputronic_transfer.cu`, `src/gputronic_episode.cu` |
| Gate | `make gold && ./build/gputronic_gold check` · `make transfer && ./build/gputronic_transfer check` |
| Author | Ben Coupland |
| Date | 2026-08-15, control section updated 2026-09-23 |
| Classification | Foundational. Do not treat later forks as this document unless they keep the contracts below. |

This is the function frame for Gold, updated 2026-09-23 so it matches the code. The control output is how many SMs run (`active_sms`). An earlier version used sleep duration. That source is in `archive/gold-1.0.0-nanosleep/`.

This is the function frame for Gold: what the system is, which signals exist, how each function computes, which constants are calibration parameters, where the C/CUDA lives, and what was measured. It is written to be read with the source open.

The claim is narrow. The rate of finished work can be a feedback signal. Gold is the version where the counter is one total, full speed is measured over a real window, and the control is how many SMs run. Earlier versions lost the counter.

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
- Change the rate by choosing how many SMs run the work. The CPU and GPU share one mapped control block.
- Leave thermal and power (D-axis) to the stock NVIDIA driver.

The control problem is ordinary: setpoint, observer, plant gain, actuator authority. The unusual part was getting an honest plant interface on a massively parallel device.

### 1.2 In scope

| Mode | What it does |
|------|----------------|
| Self-test | Gold launches `gold_persistent_kernel`. Pulses and the SM count live in one binary. |
| Cooperative | Another kernel adds to `total_work_pulses` and runs work only on the first `active_sms` SMs. Transfer is the resident-kernel test. Episode is the test where the kernel returns. |

Both modes share the same host control thread, the same Z definition, the same PI.

### 1.3 Out of scope

- Kernels we did not write (games, closed libraries, stock llama.cpp). They do not add pulses and do not read `active_sms`. Do not describe that as this controller.
- Thermal / NVML in the fast loop. Stock NVSMC owns D-axis. Polling at ~25 ms cannot sit in a 10 µs path.
- Claiming 100 kHz *effective* control. The path is 10 µs; under load the thread lands ~15–20 kHz. Rate for Z is formed over ~8 ms.

### 1.4 Lineage (why Gold exists)

| Era | What it proved | What it lost |
|-----|----------------|--------------|
| v0.1 (`archive/v0.1-github/main.cu`) | GPU as a controllable plant on GTX 1080; 10 Hz loop | Timescale too slow; “RPM” metaphor hid the plant |
| v14–v26 ladder | Persistent kernels, shared memory, PI, Kalman, game coupling | The last SM overwrote the pulse counter; Z stuck near 0.25 or 4.0 |
| Gold 1.0.0, first posted | Pulse counter, sustained \(Z\), sleep duration as the control | Sleep barely worked below ~200 µs and jumped near 500 µs |
| Gold 1.0.0, 2026-09-23 | Same counter and \(Z\). Control is how many SMs run | — |

Cyberpunk v1.13 (~Z 1.76, sleep ~9.5 µs, ~7350 PT frames) remains historical high-water for *game* coupling. It is not a Gold result. Different sign, different plant, different night.

---

## 2. Signals (the shared control block)

The only shared object between host and device is `struct GPUTronicControl`, allocated with `cudaHostAllocMapped`. On this 5080, the host pointer and the device pointer are the same address.

Direction is from the control thread’s point of view.

### 2.1 Device → host (sensors)

| Signal | Type | Writer | Meaning |
|--------|------|--------|---------|
| `total_work_pulses` | `volatile ull` | Device `atomicAdd` (or host `gputronic_pulse`) | Aggregate Q tachometer. **Never** last-writer assign. |
| `sm_counters[sm]` | `ull` device mem | Warp leader | Diagnostic only. Not used for Z. |

### 2.2 Host → device (actuators / flags)

| Signal | Type | Writer | Meaning |
|--------|------|--------|---------|
| `active_sms` | `volatile int` | Control thread | How many SMs run the work. 0 or ≥ `num_sm` means all of them. The kernel also refuses a value below 4. |
| `throttle_sleep_ns` | `volatile int` | Control thread | Kept in the struct. The current law writes `base_sleep_ns` and does not use this as the control. |
| `max_sleep_ns` | `int` | Create | Old sleep limit. Not read by the SM-count law. |
| `base_sleep_ns` | `int` | Create | Value written into `throttle_sleep_ns` so the old field stays small. |
| `control_flags` | `volatile uint` | Host | bit0 STOP, bit1 PAUSE, bit2 RESET |
| `target_z` | `float` | Host | Echo for device/telemetry |

### 2.3 Host-only telemetry (snapshots)

`z_raw`, `z_hat`, `dzdt_hat`, `rate_ema`, `pm_error`, `integral`, `proactive_corrections`, `control_hz_ema`, `num_sm`.

Device may read `target_z`. It must not need the snapshots to run.

### 2.4 Internal (control thread, not in the shared struct)

| Symbol | Meaning |
|--------|---------|
| `rate_ref` | Sustained free-run pulse rate (1/s) |
| `rate_ema` | 8 ms window rate, then 0.70/0.30 EMA |
| `calibrated` | 0 until F-06 completes |
| `open_loop_sms` | >0 forces that many SMs; 0 means all SMs; −1 means closed loop |
| `held_sms` | Last SM count, kept when the error is inside the deadband |
| `sms_per_u` | How many SMs to turn off per unit of \(u = K_p e + K_i I\) |
| `probing` | 1 during the half-SM measurement after free-run |
| `P[2][2]`, `z_hat`, `dzdt_hat` | Kalman state |
| `integral` | PI integrator, clamped |

---

## F-01 Shared control block

**Purpose.** Put the control block in mapped pinned memory so the device re-reads `active_sms` without a 20–50 µs copy.

**Equation.** After `cudaHostAlloc(..., cudaHostAllocMapped)` and `cudaHostGetDevicePointer`:

\[
\texttt{d\_ctrl} = \mathrm{map}(\texttt{h\_ctrl})
\]

On this 5080, \(\texttt{h\_ctrl} = \texttt{d\_ctrl}\) as pointer values. That equality is sufficient, not necessary; a non-equal mapped pointer is still zero-copy. Gold records `zero_copy_ok = (d_ctrl != NULL)`.

**Code.** `gputronic_create` in `src/gputronic_gold.cu`. After writing `active_sms`, the host calls `__sync_synchronize()`.

**Failure.** If mapping fails, create aborts. There is no copy fallback. A slow link between CPU and GPU would be a different system.

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

**Who runs.** Each pass reads `active_sms`. If the block index is at or above that count, the block waits 5 µs and does not pulse. Otherwise it does the work and one pulse per warp leader. A value below 4 is treated as 4. 0 or a value above `num_sm` means every SM.

**Period on this 5080.** Mapped `atomicAdd` plus the loop set \(T \approx 300\,\mu\mathrm{s}\) when every SM runs, not the fmaf count. The atomic is the limiter on this kernel. Turning SMs off removes their atomics as well as their work, so the rate falls with the count.

**Flags.**

| Flag | Device action |
|------|----------------|
| STOP | `break` |
| PAUSE | `__nanosleep(5000)` and continue (no pulse) |
| RESET | SM counter cleared by tid 0; no aggregate reset |

**Code.** `gold_work`, `gold_persistent_kernel`, `gold_launch_kernel`.

---

## F-03 Using Gold from another kernel

**Purpose.** Let another kernel be the plant without copying Gold’s control thread.

**Must.**

1. `launch_selftest_kernel = 0`.
2. After `gputronic_create`, map `gputronic_get_control()` with `cudaHostGetDevicePointer`.
3. Launch that kernel before or immediately after `gputronic_start`, so F-06 sees pulses during the free-run window.
4. Increment `total_work_pulses` with `atomicAdd` (device) or `gputronic_pulse` (host, one writer).
5. Read `active_sms` every pass. Blocks at or above that index skip the work and do not pulse.
6. Stop when `control_flags` says STOP.

**Must not.**

- Assign `total_work_pulses = sm_counters[sm]` (the last SM overwriting the sum; late v26).
- Fence every pulse (`__threadfence_system` every 32 is the Gold pattern).
- Assume the half-SM probe (F-09) has finished before the first pulses exist.

**Code.** `src/gputronic_transfer.cu` (kernel stays resident). `src/gputronic_episode.cu` (kernel returns). `src/gputronic_show.cu` (demo).

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

**Why 8 ms.** A 10 µs average that held its value on an empty tick stayed at free-run (~500 kpulse/s) while an 8 ms count of pulses already showed the slowdown. The loop period is ~300 µs. A window shorter than one loop cannot see the plant. The code uses 8 ms. About 1–2 ms is enough when a frame is the deadline and the window still holds a few hundred pulses. Under about 1 ms, on the kernels measured here, the noise is larger than the 0.08 deadband.

**Code.** `control_thread`, fields `rate_win_dt`, `rate_win_pulses`, `rate_ema`.

---

## F-06 Calibration (`rate_ref`)

**Purpose.** Define full speed so \(Z \approx 1\) when every SM is running.

**Sequence (closed-loop start).**

| Phase | Time | SM count | Action |
|-------|------|----------|--------|
| Warmup | 0 – 0.7 s | all | Discard. Let the kernel settle. |
| Sustain | 0.7 – 1.8 s | all | Mark pulses and time at 0.7 s. |
| Close | t = 1.8 s | all | \(r_{\mathrm{ref}} = \Delta p / \Delta t\) over the 1.1 s window. |
| Probe | next 0.5 s | half, at least 4 | Measure the rate. F-09 turns that into `sms_per_u`. |

Validity: \(\Delta t > 0.2\,\mathrm{s}\) and \(\Delta p > 1000\). Else fall back to `rate_ema` if it is already \(> 1000\), and skip the probe (F-09 uses a fallback drop of \(36/84\) of the SMs).

**Not this.** Taking the peak of `rate_ema` during the free-run. That parked \(Z\) around 1.4–1.7 while the GPU was already at full speed. The one-sided PI then saw \(z \ge z^\star\) and never turned any SMs off. The log said tracking. Nothing moved.

During calibration and the probe, `z_raw` is published as 1.0. The PI does not run yet.

**Code.** `control_thread` cal block; then `gold_apply_width_scale` (F-09).

The host can ask for this same window again with `gputronic_recalibrate`. The clock starts at the request: 0.7 s warmup, then the 1.1 s sustain, every SM on. `probe = 0` stores the new `rate_ref` and keeps `sms_per_u`. Any other value runs the half-SM probe and replaces `sms_per_u`. Kp and Ki stay at 0.55 and 0.08. Calling this is how a running loop takes the free-run of the load that is actually pulsing. A human does not type a new SM count.

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

**Note.** \(\dot Z\) is estimated. The PI does not use it yet. It is written into the control block so it can be logged. Do not add a derivative term that is not in the source.

**Code.** `kalman_update`. Disabled if `enable_kalman = 0`; then \(z = Z_{\mathrm{raw}}\), \(\dot Z = 0\).

---

## F-09 Half-SM probe (`sms_per_u`)

**Purpose.** Turn one measurement into a first guess of how many SMs to turn off for a typical error. Kp and Ki stay at 0.55 and 0.08.

**Takes.** `rate_ref` from F-06, and the pulse rate over the next 0.5 s with `active_sms = N_{\mathrm{SM}}/2` (at least 4).

**Returns.** `sms_per_u`, used by F-10.

**Steps.**

\[
Z_{1/2} = \frac{r_{\mathrm{ref}}}{r_{1/2}}
\]

If \(Z_{1/2} > 1.05\):

\[
g = \frac{Z_{1/2} - 1}{N_{\mathrm{SM}}/2},\qquad
\Delta n_{1.5} = \mathrm{round}(0.5 / g)
\]

Else use the fallback \(\Delta n_{1.5} = N_{\mathrm{SM}}\cdot 36/84\).

Clamp \(\Delta n_{1.5}\) to \([1,\, N_{\mathrm{SM}}-4]\). Then

\[
\mathrm{sms\_per\_u} = \frac{\Delta n_{1.5}}{K_p \cdot 0.30}
\]

so an error of 0.30, with the integrator at 0, asks to turn off about the number of SMs the probe thinks will reach Z = 1.5. The gain is not linear on a real kernel (SMs that stay on often speed up), so this is a first guess. F-10’s integrator walks from there.

The old `sleep_scale` (map an error of 0.30 onto about one loop period of sleep) is not used. Stock \(1.2\cdot 10^5\) commanded about 8 µs, and sleep on this 5080 does little until a few hundred microseconds, then jumps. That source is archived.

**Code.** `gold_apply_width_scale`. Log line: `[CAL] width Z_half=... drop_for_1.5=... sms_per_u=...`.

---

## F-10 Tracking PI (one-sided)

**Purpose.** Drive \(Z\) toward \(z^\star\) by turning SMs off. Never by running fewer than 4, and never by holding a throttle once Z is already at the target.

**Inputs.** \(z\) from F-08 (or F-07 if the Kalman is off), \(z^\star\), \(K_p\), \(K_i\), `sms_per_u` from F-09, `open_loop_sms`.

**Output.** Integer \(n\), written to `active_sms`. Also stored as `held_sms`.

**Cases, in order.**

1. Probe running: \(n = N_{\mathrm{SM}}/2\) (at least 4).
2. `open_loop_sms > 0`: \(n\) is that value. Integrator cleared.
3. `open_loop_sms == 0`: \(n = N_{\mathrm{SM}}\). Integrator cleared. This is the free-run command.
4. Not calibrated yet: \(n = N_{\mathrm{SM}}\).
5. `one_sided` and \(z \ge z^\star\): \(n = N_{\mathrm{SM}}\), integrator cleared. Full speed.
6. Otherwise:

\[
e = z^\star - z
\]

If \(|e| < 0.08\), keep the previous \(n\). Do not clear it. Clearing it would turn every SM back on and the rate would jump.

If \(|e| \ge 0.08\):

\[
I \leftarrow \mathrm{clip}(I + e\,\Delta t,\; \pm 1),\qquad
u = \max(0,\; K_p e + K_i I)
\]

\[
n = N_{\mathrm{SM}} - \mathrm{round}(u \cdot \mathrm{sms\_per\_u})
\]

Then clamp \(n\) to \([4,\, N_{\mathrm{SM}}]\).

**Sign.** No extra negation. Fewer SMs must raise Z. If a future kernel does the opposite, fix that kernel. Do not flip this sign.

**Code.** `control_thread`. Runtime: `gputronic_set_target`, `gputronic_set_gains` (Kp and Ki only), `gputronic_set_open_loop_sms`.

---

## F-11 Actuator (SM count)

**Purpose.** Turn \(n\) from F-10 into who does the work.

Device, every pass, before the work:

```
active = ctrl->active_sms
if (active <= 0 || active > num_sm) active = num_sm
if (active < 4) active = 4
if (blockIdx.x >= active) { wait 5 µs; continue; }  /* no pulse */
```

The lowest `active` blocks run the tiles and pulse. Waiting 5 µs on the idle blocks is only so they do not spin on the control block. It is not the control.

**What sleep duration did, and why it is not the control.** On this 5080 `__nanosleep` did little below about 200 µs, cut the rate near 500 µs (Gold open-loop Z about 1.58, transfer about 9.03 to 5.15 GFLOP/s), and longer sleeps did not cut further. Sleeping some SMs while others kept working did nothing until the last SM joined. The rate only moved when every SM slept. That is one step, not a range. The old numbers are in `archive/gold-1.0.0-nanosleep/` and in the 2026-08-15 gate reports.

**What the count does (open loop, this 5080, 2026-09-23).** Rate falls as \(n\) falls. Gold dyno R² about 0.98 from 84 SMs down to 4. Transfer R² about 0.99. The SMs that stay on often speed up, so Z is not exactly \(84/n\). The count that lands near Z = 1.5 depends on the kernel: about 48 on the transfer tiles, about 21 on the short episode, about 54–58 on the heavier show kernel. Closed loop on all three averaged about 1.36–1.38, with the count moving a little, and returned to every SM when the target went back to full speed.

---

## F-12 Host scheduler and lifecycle

```
gputronic_config_gold
gputronic_create          /* map the control block, query SM count */
[cooperative: launch foreign kernel on d_ctrl]
gputronic_start           /* optional self-test launch + control pthread */
… run / dyno / set_target …
gputronic_stop            /* STOP flag, join, device sync, CSV close */
gputronic_destroy
```

Create installs SIGINT/SIGTERM → `g_signal_stop`. That is process-global. Two Gold instances in one process will share the signal flag.

CSV header:

```
time_s,z_raw,z_hat,dzdt,rate_ema,active_sms,error,integral,ctrl_hz,rate_ref
```

---

## F-13 Verification (dyno, step, closed-loop, transfer)

### Dyno (open-loop)

SM counts from `num_sm` down through quarters and thirds to 4.

Rate is \((\Delta p)/(\Delta t_{\mathrm{wall}})\) — **not** `rate_ema` alone.

Pass: full-width rate \(> 1000\) and the narrowest count \(< 0.75\times\) that rate, **and** (\(R^2 \ge 0.85\) of rate vs count, or the rate mostly falls as the count falls, 15 % tolerance).

### Closed-loop gate (Gold)

After calibration and the half-SM probe: compare an early full-speed window with windows from t ≥ 4 s. Pass requires the counter alive, Z in \([0.3,10]\), rate sane, the later window \(< 0.90\times\) the early rate, the loop actually using fewer than all SMs, and \(|Z_{\mathrm{mean}} - 1.5| < 0.25\).

2026-09-23 on this 5080: about 582k → 426k pulse/s, \(Z_{\mathrm{mean}} \approx 1.38\). The older sleep-duration run was 445k → 324k, \(Z_{\mathrm{mean}} \approx 1.37\).

### Transfer gate

Same Gold controller. Second kernel. Pass requires tile rate and GFLOP/s to move together in the dyno, and GFLOP/s to drop in closed loop when Z is asked for 1.5.

2026-09-23: closed loop 7.25 → 5.39 GFLOP/s, \(Z_{\mathrm{mean}} \approx 1.36\). The older sleep-duration run was 8.77 → 6.55 GFLOP/s, \(Z_{\mathrm{mean}} \approx 1.34\).

---

## F-14 Transfer plant (application of F-03)

**Purpose.** Prove F-03 on counted FLOPs, not on Gold’s synthetic fmaf.

- 8×8×8 GEMM in shared memory, 32 tiles per pulse = 32768 FLOP/pulse.
- One pulse per block (`tid == 0`).
- `launch_selftest_kernel = 0`. Reads `active_sms` from the shared struct. No separate gain.

If tile rate and GFLOP/s ever disagree in sign beyond noise, the counter is no longer measuring that work.

**Code.** `src/gputronic_transfer.cu`.

---

## F-15 Episode plant

**Purpose.** Same question when the kernel returns, instead of looping forever.

- Same 8×8×8 tiles. Default 32 tiles per pulse, 4 pulses, then return. The host launches again after `cudaDeviceSynchronize`.
- Blocks at or above `active_sms` return without pulsing.
- The 8 ms window still forms Z across launches.

2026-09-23 closed loop: 20.3 → 15.0 GFLOP/s, \(Z_{\mathrm{mean}} \approx 1.36\). Open loop, the short default shape reached about Z = 1.5 near 21 SMs, because the SMs left on sped up. That is a different curve from transfer. F-09 has to be run on this kernel. The transfer number does not carry over.

**Code.** `src/gputronic_episode.cu`.

---

## 3. Calibration parameters (Gold 1.0.0)

| Name | Symbol | Default | Where | Why |
|------|--------|---------|-------|-----|
| `GPUTRONIC_DEFAULT_NUM_SM` | \(N_{\mathrm{SM}}\) | 84 (or query) | header | GB203 enablement |
| `THREADS_PER_BLOCK` | \(N_{\mathrm{thr}}\) | 64 | header | 2 warps; conservative occupancy |
| `CONTROL_DT_US` | — | 10 | header | Path period, not observer period |
| `MAX_SLEEP_NS` | — | 500000 | header | Unused by the SM-count law. Left so old fields stay valid. |
| `BASE_SLEEP_NS` | — | 5 | header | Written into `throttle_sleep_ns`. Not the control. |
| `TARGET_Z` | \(z^\star\) | 1.5 | header | About 2/3 of the free-run rate |
| `KP` | \(K_p\) | 0.55 | header | Leave this. F-09 absorbs the plant gain. |
| `KI` | \(K_i\) | 0.08 | header | Same |
| `SLEEP_SCALE` | — | 120000 | header | Old fallback. F-09 no longer writes it. |
| SM floor | — | 4 | source | F-10 and F-11. The kernel is never empty. |
| `INTEGRAL_CLAMP` | \(I_{\max}\) | 1.0 | header | Anti-windup |
| `DEADBAND` | \(e_{\mathrm{db}}\) | 0.08 | header | Inside this, below the target, hold \(n\) |
| `Z_CEILING` / `Z_FLOOR` | — | 10 / 0.3 | header | Drop nonsense samples |
| `one_sided` | — | 1 | config | F-10 |
| `enable_kalman` | — | 1 | config | F-08 |
| Kalman \(Q\) | — | 0.0008 | source | Not a header calibration constant |
| Kalman \(R\) | — | 0.018 | source | Not a header calibration constant |
| Rate window | — | 8 ms | source | F-05 |
| EMA | — | 0.70 / 0.30 | source | F-05 |
| Cal warmup / total | — | 0.7 s / 1.8 s | source | F-06 |
| F-09 design error | — | 0.30 | source | Error used to turn the probe into `sms_per_u` |
| F-09 probe | — | 0.5 s at half the SMs | source | After the 1.8 s free-run |
| `GOLD_BATCH` × `ITERS` | — | 16 × 8 | source | F-02 work per pulse |
| Transfer tile / batch | — | 8 / 32 | transfer | F-14 |

Changing a source-only calibration constant is a Gold tag, not a silent edit.

---

## 4. Guided walk through the source

Read in this order. Do not start at `main`.

1. **`include/gputronic.h`** — `GPUTronicControl` and the constants. If a signal is not here, the device cannot see it.
2. **`gold_work` / `gold_persistent_kernel`** — F-02, F-04, F-11. Pulse is `atomicAdd`. Blocks at or above `active_sms` skip the work. Fence is every 32.
3. **`gputronic_create`** — F-01. Mapped allocation.
4. **`gputronic_start`** — launches F-02 only if configured; always starts the control thread.
5. **`control_thread`** — F-05, F-06, the half-SM probe, F-07, F-08, F-10, then write `active_sms`.
6. **`gold_apply_width_scale`** — F-09, once the probe window ends.
7. **`kalman_update`** — F-08. Check \(P^-=FPF^\top+Q\), not `P[i]+=Q`.
8. **`mode_dyno` / `mode_closedloop_check`** — F-13. Dyno rate is wall \(\Delta p/\Delta t\).
9. **`src/gputronic_transfer.cu`** and **`src/gputronic_episode.cu`** — F-14 and F-15. Both set `launch_selftest_kernel = 0` and read `active_sms`.

`main` is a CLI multiplexer (`run`, `dyno`, `step`, `check`). It is not the function.

Handle fields that look redundant (`h_sm`, `proactive_corrections`) are telemetry or reserved. They are not in the Z equation.

---

## 5. Failure modes and anti-patterns

| Symptom | Likely cause | Function |
|---------|--------------|----------|
| \(Z\) stuck 0.25 or 4.0 | Uncalibrated legacy scale, or last-SM overwrite | F-04, F-06 |
| \(Z_{\mathrm{free}} \ge 1.5\), every SM stays on | Peak of the rate used as `rate_ref` | F-06 |
| Asked for 1.5, Z stays near 1, all SMs on | Probe skipped or `sms_per_u` tiny; or free-run Z already above the target | F-06, F-09 |
| Count jumps between 4 and 84 every sample | Window shorter than the pulse count can support | F-05 |
| Z moves on one kernel and not another at the same \(n\) | Different kernel. Run F-09 on that kernel. Do not copy \(n\). | F-09, F-11 |
| Dyno FAIL, EMA flat, pulse Δ already down | Dyno used EMA instead of wall Δ | F-13 |
| Transfer GFLOP/s and tile rate disagree | Pulse no longer equals a tile | F-03, F-14 |
| A game or a library slows down with no pulses of its own | Out of scope. Those kernels do not read `active_sms`. | §1.3 |

---

## 6. Why this should look ordinary

A feedback loop on a measured work rate is ordinary. The unusual part was getting that measurement on a GPU without the last writer destroying the counter, and without calling a temperature poll the same thing. Gold 1.0 is that measurement, a free-run zero, and a control output that is how many SMs run. It has been checked on three kernels on one 5080. It does not yet cover a program that never reads the control block.

---

## 7. Document control

| Version | Date | Note |
|---------|------|------|
| 1.0.0 | 2026-08-15 | First function frame. Sleep duration was the control. |
| 1.0.0-sms | 2026-09-23 | Control changed to `active_sms`. F-09, F-10, F-11, F-13 rewritten. F-15 added. Sleep version archived. |
| 1.0.0-sms | 2026-10-04 | Host can request F-06 again (`gputronic_recalibrate`). Gains and `active_sms` unchanged. |

If the source and this frame disagree, the source and the check reports win, and this document should be updated. Do not change Kp to make a sentence in this file true.

Related files: `GOLD.md` (short), `STATUS.md` (what is running), `TRANSFER.md`, `EPISODE.md`, `archive/gold-1.0.0-nanosleep/` (sleep version), `archive/v0.1-github/` (older tree).
