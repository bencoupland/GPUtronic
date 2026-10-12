# GPUTronic: project status and next steps

Measured on the RTX 5080 on 2026-10-12. The suite score is
[`5080-2026-10-12.md`](5080-2026-10-12.md). What to do with it is
[`5080-2026-10-12-dispatch.md`](5080-2026-10-12-dispatch.md). Driver 595.99.02,
kernel 7.0.0-38-generic, CUDA 12.8, on `test-results/2026-10-12_5080`.

Short version of that run: the two-sided law with the four-point table holds Gold near
Z 1.49 and Transfer near Z 1.46. Episode holds near Z 1.45–1.47. The old one-sided law
on this same driver, the same afternoon, was still Z 1.34. Kp and Ki were not changed.
The tensor warp cuts the CUDA-core rate by about 39% at full width and then reports half
that rate, so the two sensors agree. The 0.015 deadband does not hold the SM count still.
The transfer and episode gates failed on the first pass because their free-run sample
started at 2.2 s, inside the new 3 s table. Waiting until 4.5 s makes both gates pass.
The model text below was written before this run.

This page summarises where the `controller-fixes` branch (PR #1) stands. The paragraphs
under "What works in simulation" were written before any GPU run. Results marked "model"
come from simulations built on the RTX 5080 measurements that were already in `results/`.

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

- **Nothing is GPU-verified.** All PR #1 results are compiled-only plus models.
- **The RTX 5080 loop is limited by PCIe, not clocks.** One kernel pass takes about 288 µs, while the
  arithmetic in it takes about 1.6 µs. The time goes to atomic updates on mapped host memory.
- **Transfer settles about 0.04 below target.** Its rate curve bends between 63 and 42 SMs, and the
  straight-line interpolation in the table picks slightly too many SMs.
- **The Kalman derivative is off (Kd = 0).** As coded, dZ/dt jumps at each 8 ms window edge instead of
  following a smooth slope, and in the model it never helped. See `kalman-analysis.md`.
- **The tensor-core sensor is untested** on hardware.
- **sm_61 (GTX 10 series) does not build:** the kernels call `__nanosleep`, which needs sm_70.
  See `gtx1080-bringup-plan.md`.
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
  Gold's shape at about 1/3 of the CUDA-core rate. With that curve both sensors settle near Z 1.49, so
  with this kernel the two sensors are largely redundant.
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

## 6. Origin

The project was inspired by Bosch Motronic 7 engine management on the Audi B5 S4.

## 7. Next steps

1. Run the test suite (`docs/analysis/TEST_SUITE.md`) on the RTX 5080 following
   `docs/analysis/GROK_BUILD_INSTRUCTIONS.md`: tensor-sensor overhead first, then Gold, Transfer, Episode.
   Record mean Z, SM-count range and the calibration table.
2. Run on a GTX 1080 after adding a `clock64` spin fallback for `__nanosleep` on sm_61.
3. Measure the tensor warp's overhead: compare the CUDA-core rate with the tensor sensor on and off.
4. Add a tensor sensor to the Transfer and Episode kernels.

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
| (next) | Test runbook, test suite, model findings, switchable sensor rule |
