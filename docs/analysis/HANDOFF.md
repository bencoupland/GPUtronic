# GPUTronic: project status and next steps

This page summarises where the `controller-fixes` branch (PR #1) stands. **Nothing on this
branch has been run on a GPU yet.** Results marked "model" come from simulations built on the
RTX 5080 measurements that are already in `results/`.

## 1. Project overview

GPUTronic holds a GPU at a requested rate of useful work.

- **Sensor:** the kernel counts finished work units ("pulses") in a counter that the host and
  the GPU share through mapped (zero-copy) memory.
- **Measure:** `Z = reference rate / measured rate`. The reference rate is measured at full speed
  during calibration, so Z is about 1.0 at full speed. The default target is **Z = 1.5**, which
  means roughly two-thirds of full speed.
- **Actuator:** the number of SMs allowed to do work (`active_sms`). Blocks above that count skip
  the work. At least 4 SMs stay on.
- **No driver changes:** it uses the stock NVIDIA driver and the standard CUDA runtime. Clocks,
  temperature and power stay with the driver.

## 2. What works in simulation (model only)

The controller on this branch:

- **Two-sided PI:** it turns SMs off when Z is below target and back on when Z is above it. The
  integral is clamped and does not grow while the SM count is pinned at 4 or at full width.
- **Deadband 0.015:** inside it, the SM count is held.
- **Four-point calibration table:** after the full-speed reference (0.7–1.8 s), it measures the
  rate at 3/4, 1/2, 1/4 of the SMs and at 4 SMs, 300 ms each. Calibration takes about 3 s, about
  **0.7 s longer** than before (the old single half-width probe took 0.5 s).

Model results on the RTX 5080 dyno curves (mean Z, target 1.50):

| Kernel | Before PR #1 | After PR #1 |
|---|---|---|
| Gold | 1.36 | 1.49 |
| Episode | 1.42 | 1.50 |
| Transfer | 1.32 | 1.46 |

GTX 1080 model (20 SMs, 5080 curve shape, spin-wait idle assumed to cost a few percent):

- With runtime calibration the mean Z is 1.48, but the controller flips between 13 and 14 SMs
  almost every 8 ms window, because one SM moves Z by about 0.1 there.
- With a reference taken from a table that is 5–10% off, it locks on a single count and Z sits
  0.04–0.07 away from target.
- A 20-SM card needs either alternating between two counts over 8–16 windows or a wider deadband
  (about 0.06).

## 3. What does not work yet, or is unverified

- **Partly GPU-verified.** See "RTX 5080 hardware results, 2026-10-12" below; T4 (GTX 1080) is not run.
- **The RTX 5080 loop is limited by PCIe, not clocks.** One kernel pass takes about 288 µs, while the
  arithmetic in it takes about 1.6 µs. The time goes to atomic updates on mapped host memory.
- **Transfer settles about 0.04 below target.** Its rate curve bends between 63 and 42 SMs, and the
  straight-line interpolation in the table picks slightly too many SMs.
- **The Kalman derivative is off (Kd = 0).** As coded, dZ/dt jumps at each 8 ms window edge instead of
  following a smooth slope, and in the model it never helped. See `kalman-analysis.md`.
- **The tensor-core sensor costs 30–39% of the CUDA-core rate** and adds no information on Gold (measured).
- **sm_61 (GTX 10 series):** flip the Makefile `ARCH` from sm_120 to sm_61. See `gtx1080-bringup-plan.md`.
- **The Makefile defaults to sm_120** and CUDA 12.8 paths; override with
  `make NVCC=... ARCH=-arch=sm_XX`.

## 4. Architecture

- **One actuator:** the SM count, `active_sms`.
- **Two sensors:**
  - CUDA-core counter `total_work_pulses`: two warps per block run a chain of FMAs and pulse.
  - Tensor-core counter `tensor_work_pulses`: when enabled (Gold default, sm_70 and newer), one extra
    warp per block runs FP16 matrix steps (`wmma`) and pulses. Both stop on SMs above `active_sms`.
- **Each sensor has its own** 8 ms rate window, reference rate and Z.
- **The controller follows the higher Z.** Both Z values rise as SMs are turned off, so following the
  higher one reaches the target with the fewest SMs turned off, and neither kind of work is slowed
  more than requested.
- **Switching:** it only changes sensor when the other Z is higher by 0.02. On a switch it recomputes
  the integral so the SM count does not jump (bumpless transfer).
- **Fallback:** if the tensor counter produces no pulses during calibration, the controller uses the
  CUDA-core sensor only. It is also off for the Transfer and Episode kernels.

## 5. Timing

- **Control tick:** about 10 µs (`control_dt_us`).
- **Kernel pass:** about 288 µs on the RTX 5080 (84 SMs), so a change to `active_sms` shows up in the
  pulse count within roughly one pass.
- **Rate window:** 8 ms, then an average weighted 0.7 old / 0.3 new. This window, not the tick,
  sets how fast the controller sees a change.
- **Feed-forward:** the author has described this as a "99% dial-back". That figure is not in the
  code. What the code does: the calibration table gives a starting SM count (`n_ff`) at which Z is
  expected to equal the target. The controller jumps straight to that count after calibration and
  then works around it: `n = n_ff - (Kp*e + Ki*I - Kd*dZ/dt) * sms_per_u`. In the model that start
  point already lands within about 0.01–0.04 of the target, and the PI only trims the rest.

## 5a. Findings from the models (not measured)

- **The RTX 5080 bottleneck is PCIe atomics, not clocks.** Every pulse is an `atomicAdd` to mapped host
  memory across PCIe. Gold's ~288 µs pass is set by that path, not by SM clock or arithmetic rate, so
  spec-sheet tables mislead. Example: a tensor curve built from the published FP16 tensor rate is linear
  in SMs and ~1,000x faster than the kernel can pulse. The atomic-bound tensor curve instead tracks
  Gold's shape at a fixed fraction of the CUDA-core rate (model said 1/3; hardware measured exactly 1/2),
  so the two Z values are identical and the sensors are redundant with this kernel.
- **Transfer's curve bends between 63 and 42 SMs**, so a straight line from one probe point is wrong
  there; the four-point table covers it.
- **Two-sensor rule.** The default rule follows the **higher** Z (the max selector). It has also been
  called "min-of-two-Z": both mean the same thing, because the higher Z is the sensor with the smaller
  measured/reference rate ratio, i.e. the more-throttled sensor. The rule is switchable
  (`GPUTRONIC_SENSOR_RULE=max|min`). Head-to-head model (Gold, Transfer, Episode; atomic-bound and
  spec-ideal tensor curves): max kept the other sensor at or below 1.5 in every case; min let it reach
  1.83 (Gold), 1.59 (Transfer) and 3.65 (Episode) with the spec-ideal curve and settled further below
  target with the atomic-bound curve. Max stays the default; the full table is in the source comment
  above the selector. Neither rule gets back within 0.02 after a sustained −20% load step.
- **8 ms sensing window vs 10 µs control tick.** The loop runs every 10 µs but only sees new data every
  8 ms. Bursts shorter than 8 ms are invisible; reaction to a real change takes 8–16 ms.
- **Fixed pulse-batch size only holds for kernels we own.** Gold, Transfer and Episode pulse at a known
  cadence. An unmodified workload cannot be counted this way.

## 5b. RTX 5080 hardware results, 2026-10-12

Source: `docs/analysis/5080-2026-10-12.md` on branch `origin/test-results/2026-10-12_5080`, issue #2.
These replace the model-only claims above where they differ.

| Test | Model | Measured |
|---|---|---|
| Gold closed loop, mean Z (last 30 s) | 1.49 | **1.487** |
| Gold SM count | 46–47, ±1 | 44–52, mean 48.2 |
| Tensor-warp cost to CUDA-core rate | up to ~1/3 | **39% at 84 SMs**, 30–39% across widths |
| Tensor rate / CUDA rate | ~1/3 | **exactly 1/2** at every width, so tensor Z = CUDA Z |
| Transfer mean Z (after wait fix) | 1.46 | 1.456 |
| Episode mean Z (after wait fix) | 1.50 | 1.468 |

- **Recommendation: tensor sensor off by default.** It costs up to 39% of throughput and its Z is the same
  number as the CUDA-core Z. Not yet changed in code (`GPUTRONIC_TENSOR_SENSOR=0` turns it off).
- **Integral clamp bug seen on hardware.** One Episode run parked on the clamp (1.0), and the reserve-4 run
  hit it. Clamp × Ki × `sms_per_u` is only about 4 SMs of authority, so a low starting count cannot be
  walked far. The clamp (`GPUTRONIC_GOLD_INTEGRAL_CLAMP`, hard-coded 1.0 in `include/gputronic.h`) should
  scale with SM count, e.g. chosen so `Ki * clamp * sms_per_u` covers a meaningful fraction of the SMs.
- **T3 needs the 4.5 s wait fix.** `mode_closedloop` in Transfer and Episode slept 2.2 s, then sampled the
  free-run rate while the four-point table (now ~3 s) was still probing, so the gates failed with Z in band.
  Commit `ff4e97c` on `test-results/2026-10-12_5080` waits 4.5 s. It is not yet on `controller-fixes`;
  without it the T3 gates fail on this branch.
- **No real load-step test.** There is no load generator in the code, so the load-step and burst
  predictions remain model-only.

## 6. Origin

The project was inspired by Bosch Motronic 7 engine management on the Audi B5 S4.

## 7. Next steps

1. Port the 4.5 s free-run wait (`ff4e97c`) to `controller-fixes`.
2. Fix the integral clamp: scale it to SM count instead of the hard-coded 1.0.
3. Add a load generator so load steps and bursts can be tested on hardware.
4. Make the tensor sensor off by default.
5. Code to-do: the kernels still contain sleep calls in their idle/pause loops (Gold, Transfer, Episode, Show) that a later code pass removes, required before an sm_61 build compiles.

## 8. Git state

All work is on branch `controller-fixes` (PR #1). Nothing has been merged to `main`.

| Commit | Summary |
|---|---|
| `efe3913` | Controller fixes: two-sided PI, deadband 0.015, four-point table, real step test, counter reset, zero-copy check, sleep API removed |
| `05bfc23` | Kalman analysis, GTX 1080 bring-up plan, modelled pulse-rate table |
| `7cfba97` | Plain-language comments and one section-banner style in source |
| `68aba6c` | Stop tracking `.grok/` |
| `46b05a4` | Rewrite function reference and contributor notes |
| `4fdb297` | Tensor-core sensor: second counter and Z, controller follows the higher Z |
| `bb3a90f` | This status page |
| `edb6dac` | Switchable sensor rule (max default) |
| `0a4d495` | Test runbook, test suite, model findings |
| (next) | RTX 5080 results and revised roadmap |
