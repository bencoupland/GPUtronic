# GPUTronic

Closed-loop control of instruction-flow throughput on a GPU.

Reference: **GPUTronic 1.0 Gold**

![GPUTRONIC](docs/assets/gputronic-mark.jpg)

A GPU driver will slow the chip if it gets hot. It will not tell you whether the work you asked for is the work that finished. You launch a kernel, wait, and take whatever rate comes out.

Other machines we control have a measured rate, an estimate of what we cannot see directly, and a command that moves the rate. A GPU can have those too. What was missing was a counter of finished work, a way to say “this is full speed,” and a command that actually changes that speed.

GPUTronic is that interface, checked on three test kernels. The control in this tree is how many SMs do the work (`active_sms`). An earlier version used sleep duration instead. That source is in `archive/gold-1.0.0-nanosleep/`. The function frame matches this tree.

The function-by-function writeup is [`docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md`](docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md). This README is the short tour. The same function frame is copied at the end of this file.

## The claim

The rate of finished work can be a feedback signal.

Count that work. Compare it with the speed you measured when nothing was holding the GPU back. Change how many SMs are allowed to run, and leave clocks and temperature to the driver. Then a normal PI controller can hold a requested rate. This does not control a program you did not build. It does not beat an unthrottled run. Full speed is every SM on.

## Three axes, one loop

The names are borrowed from motor control because the split is the same: measure the thing you care about, estimate what you cannot read, and do not fight the limit that already owns temperature.

| Name | What it is | Who handles it |
|------|------------|----------------|
| **Q** | Finished work per second | GPUTronic. One running total of pulses. |
| **Z** | \(Z = r_{\mathrm{ref}} / r\) | GPUTronic. Full speed is about 1. |
| **D** | Temperature and power | The NVIDIA driver. Not in this loop. |

Q is not “GPU utilization.” Utilization can read 100% while useful work is stalled. Q is a count of finished work units, added with `atomicAdd` into one total, `total_work_pulses`. Do not let the last SM overwrite that total with its own count. That bug made Z stick at a floor or a ceiling.

Z is not ohms and not a phase margin. \(Z = 1.5\) means the GPU is doing about two-thirds of the speed it had when nothing was holding it back. Turning SMs off lowers that speed and raises Z. There is no extra minus sign in the controller. Full speed is every SM running. The point of the loop is to land near the speed you asked for when conditions are not ideal.

Temperature and power stay with the NVIDIA driver. A 25 ms temperature poll does not belong in the 10 µs control thread. If the driver slows the chip, the work rate falls, Z rises, and this loop eases off. That is the intended split.

A 2-state Kalman filter estimates Z and how fast Z is changing. The controller does not use the rate-of-change yet. It is logged.

The control value is `active_sms` in a shared control block. That many SMs run the tiles and increment the counter. The others skip the work. At least 4 stay on, so the kernel is never empty. On this RTX 5080 the CPU and the GPU see the same memory. The kernel reads the count every pass. There is no 20–50 µs copy. If the mapping fails, startup aborts. A slow link between the two would be a different system. Sleep duration was the first attempt. On this 5080 it does little below about 200 µs, jumps near 500 µs, and then longer sleeps do not slow the chip further. Sleeping some SMs while others kept working did nothing. That is why the count replaced it.

## Why 1.0 is tagged Gold

The repo starts with a GTX 1080 prototype in January 2026, then a run of experiments through May 2026, then this 1.0 tree. Those older trees are still in `archive/`. What failed in the later experiments was the counter: it was not one total, Z stuck near 0.25 or 4.0, or the free-run measurement was a peak instead of a sustained window, so the loop thought it was already at the target and never did anything.

1.0 is the version you can build and check again tomorrow.

Three mistakes that look like a working controller until you measure the work:

1. **Peak-hold calibration.** Taking the peak of a rate EMA during free-run inflates \(r_{\mathrm{ref}}\). Free-run Z then sits at 1.4–1.7. A one-sided PI, which only adds sleep when \(Z\) is *below* target, never engages. The log says “tracking.” The actuator is at base sleep. Nothing moved.

2. **A 10 µs rate EMA.** The persistent loop on this 5080 is about 300 µs, set by mapped atomics, not by the math inside the tile. Forming rate on a 10 µs tick and holding on empty samples leaves the EMA at free-run while a wall-clock pulse window already shows the cut. The observer is blind to the plant it is supposed to regulate. GPUTronic forms rate over ~8 ms, then a slow EMA.

3. **A gain that commands about 8 µs of sleep.** On this 5080, `__nanosleep` does little below about 200 µs and then jumps near 500 µs. The old default gain, at a normal error, asked for about 8 µs. The GPU did not slow down. Changing Kp does not fix a control input the hardware ignores.

The controller works in both directions around a starting SM count measured at calibration: below target it turns SMs off, above target it turns them back on. If Z is within 0.08 of the target, the SM count stays put and the integral does not move. Temperature stays with the driver. The 8 µs scale in the list above is a reason the sleep-duration control was dropped. Adjusting Kp does not fix a control that cannot move the speed smoothly.

On this 5080, 2026-10-12, with the tensor sensor off and that 0.08 band, Gold, transfer, and episode `check` all passed. Gold's free-run was about 540k–585k pulses/s. A 30 s closed-loop run held mean Z at 1.47, and about 85% of the last 20 s was 47 or 48 SMs. Kp is 0.55 and Ki is 0.08. The sleep tables under this heading are the old actuator. They are not this controller.

## What it does on this machine

Hardware of record: RTX 5080 (GB203, 84 SMs), CUDA 12.8.

The sleep tables below are from the older sleep-duration control, kept because they show the jump. Current numbers are in `STATUS.md`. On 2026-09-23 all three checks passed with the SM count as the control. Closed-loop average Z was about 1.36–1.38. Turning the request back to full speed returned the rate to the unthrottled run.

**Self-test plant** — GPUTronic’s own persistent kernel. Dyno, open-loop sleep sweep, rate from wall \(\Delta p / \Delta t\):

| sleep | pulse/s | Z |
|------:|--------:|--:|
| 0 | ~499k | ≈ 1.00 |
| 200 µs | ~469k | almost flat |
| 500 µs | ~314k | ≈ 1.58 |

Closed-loop, target \(Z = 1.50\): 445k → 324k pulse/s (−27%). Mean Z ≈ 1.37. Sleep ≈ 415 µs, in the authority band.

**Transfer plant** — a second, cooperative kernel. Real 8×8×8 FP32 GEMM tiles, 32 tiles per pulse, 32768 FLOP/pulse. Same GPUTronic controller. No gain override.

Open-loop: 9.03 → 5.15 GFLOP/s at 500 µs, locked to tile rate. Free-run Z ≈ 1.02.

Closed-loop: 8.77 → 6.55 GFLOP/s (−25%). Mean Z ≈ 1.34.

That second number is the one that matters. Pulse rate and counted FLOPs moved together, on a kernel that is not the self-test’s dummy math. If those two ever move in opposite directions, the counter is no longer measuring that work.

Both gates are one command:

```bash
make gold && ./build/gputronic_gold check
# === GOLD GATE: dyno=PASS closedloop=PASS ===

make transfer && ./build/gputronic_transfer check
# === TRANSFER GATE: dyno=PASS closedloop=PASS ===

make episode && ./build/gputronic_episode check
# === EPISODE GATE: dyno=PASS closedloop=PASS ===
```

CSV lands under `results/` every run. Dyno rate is always a wall window, never the lagging EMA alone.

## The shared control block

The CPU and the GPU share one struct, `GPUTronicControl`, allocated with `cudaHostAllocMapped`.

The GPU writes one thing that matters: `total_work_pulses`, with `atomicAdd`, from the thread that finishes a unit of work. Fence every 32 pulses, not every pulse. A fence on every pulse swamps the copy.

The CPU writes `active_sms`, then `__sync_synchronize()`. That many SMs do the work. The others skip it.

The rest of the struct is readings or flags (STOP / PAUSE / RESET). If a value is not in that struct, the GPU cannot see it.

A kernel we compile ourselves does this:

1. `launch_selftest_kernel = 0`
2. `gputronic_create`, then `cudaHostGetDevicePointer` on `gputronic_get_control()`
3. Launch the kernel before or with `gputronic_start`, so the free-run measurement sees pulses
4. `atomicAdd` finished work into `total_work_pulses`
5. Only the first `active_sms` SMs run that work
6. Stop when asked

Transfer is the test where the kernel stays resident. Episode (`src/gputronic_episode.cu`) is the test where the kernel returns and the 8 ms window still sees Z. A program we do not modify does not do any of this. That is a separate problem.

## Timescales, because they are the whole story

| Clock | Period | Role |
|-------|--------|------|
| GPU inner work | nanoseconds | fmaf / tile |
| Persistent loop | ~300 µs here | mapped atomic, then the SM check |
| Control *path* | 10 µs | thread wake, not the observer |
| Rate window | ~8 ms | what Z is actually formed from |
| Calibration | 0.7 s warmup + 1.1 s sustain | \(r_{\mathrm{ref}}\) |
| NVML / thermal | ~25 ms | not in this loop |

The control thread can wake every 10 µs. That does not mean the rate estimate is that fast. The rate is a count of pulses over about 8 ms. The 10 µs wake is only so the CPU is not late writing the SM count.

## What this is not

It does not control a game or a library you did not build, just because a process is running. Those kernels do not add to the pulse count and do not read `active_sms`.

It does not replace the NVIDIA driver. Clocks, power, and temperature stay there.

The transfer kernel’s few GFLOP/s are not a speed claim. The tiles are there so the FLOPs can be counted. They moved when Z was commanded. cuBLAS is not the comparison.

It is not general yet. Three kernels on one 5080. The next step is the same loop on those kernels when the workload changes, without fitting a new curve by hand.

## Two sensors

The Gold kernel can count two kinds of work: CUDA-core FMAs (the original counter) and tensor-core
matrix steps (one extra warp per block, sm_70 and newer). Each has its own reference rate and its own Z.
The SM count is still the only control. The controller follows whichever Z is higher when both are on.
The tensor warp is off unless `GPUTRONIC_TENSOR_SENSOR=1`. On this 5080 it cut the CUDA-core rate by
about 39% at full width, and the two Z values were the same number, so the default leaves it off.

## Test kernels

- **Gold**: GPUTronic's own self-test kernel and the reference controller (`src/gputronic_gold.cu`).
- **Transfer**: a persistent kernel doing small GEMM tiles that reports its own work (`src/gputronic_transfer.cu`).
- **Episode**: a GEMM kernel launched once per work episode, closer to a normal application (`src/gputronic_episode.cu`).

Calibration takes about 3 s from start: a 1.8 s full-speed reference, then four 300 ms steps at fewer SMs. That is about 0.7 s longer than before PR #1 (the old half-width probe took 0.5 s).

## Quick start (RTX 5080 / CUDA 12.8)

```bash
make gold
./build/gputronic_gold check
./build/gputronic_gold run 30
make transfer && ./build/gputronic_transfer check
make episode && ./build/gputronic_episode check
make demo && ./build/gputronic_demo
```

```
include/gputronic.h                      public C API
src/gputronic_gold.cu                    core + CLI
src/gputronic_transfer.cu                Transfer kernel (persistent)
src/gputronic_episode.cu                 Episode kernel (launch-complete)
docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md
GOLD.md  STATUS.md  TRANSFER.md  EPISODE.md
archive/v0.1-gtx1080-2026-01/            first public PoC (GTX 1080)
```

Read the function frame with the source open, starting at the control block, not at `main`. Function IDs F-01…F-15 in that file match comments in the code.

## Next

A feedback loop on a measured rate is ordinary. The work here was getting the counter, the full-speed measurement, and the SM count to agree with each other. On 12 October 2026 those three checks passed on an RTX 5080, and a per-SM tally of finished work matched the pulse counter. The SMs did not each do the same amount.

Next, when it is asked for: a counter that keeps going after a kernel returns, in code we compile ourselves. A GTX 1080 run is still open. The source compiles for that card. It has not been executed there.

## License

MIT

## Function frame

The rest of this file is the function reference. The standalone copy is [`docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md`](docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md). When one changes, change the other.

# GPUTronic 1.0 Gold: Function Reference

This document lists every part of the Gold controller, what it does, and where it lives in the code.
It matches `include/gputronic.h` and `src/gputronic_gold.cu` in this tree.
The law below was run on an RTX 5080 on 2026-10-12 (driver 595.99.02, CUDA 12.8). The 2026-09-23
numbers further down are the earlier one-sided controller, kept so the change is visible.
The same text is copied at the end of `README.md`.

Function IDs (F-01 … F-16) appear in brackets in source comments so you can jump between this file
and the code.

## 1. What it is

GPUTronic holds a GPU at a requested rate of useful work. It counts finished work units, compares
the rate with the rate measured at full speed, and changes how many SMs are allowed to do the work.
Clocks, temperature and power stay with the NVIDIA driver. It uses only the stock driver and the
standard CUDA runtime.

- **Q**: finished work units per second (the pulse counter).
- **Z**: `rate_ref / rate`. About 1.0 at full speed; 1.5 means about two-thirds of full speed.
- **D**: temperature and power, left to the driver.

In scope: the self-test kernel in `src/gputronic_gold.cu`, and other kernels that add pulses and
honour `active_sms` (Transfer, Episode). Out of scope: programs that do not count their own work.

## 2. The shared control block (`struct GPUTronicControl`)

Allocated with `cudaHostAlloc(..., cudaHostAllocMapped)`; the device pointer comes from
`cudaHostGetDevicePointer`. Host and device read and write the same memory, with no copies.

| Field | Type | Written by | Read by | Meaning |
|---|---|---|---|---|
| `control_flags` | volatile uint | host | kernel | bit0 stop, bit1 pause, bit2 reset, bit3 load |
| `active_sms` | volatile int | control thread | kernel | number of working SMs (floor 4) |
| `total_work_pulses` | volatile u64 | kernel (`atomicAdd` only) | control thread | the pulse counter |
| `throttle_sleep_ns`, `max_sleep_ns`, `base_sleep_ns` | int | create / Transfer / Episode | Transfer and Episode kernels | legacy sleep fields, not used by Gold's controller |
| `tensor_work_pulses` | volatile u64 | tensor warp (`atomicAdd` only) | control thread | tensor-core pulse counter, appended at the end of the struct |
| `z_cuda`, `z_tensor`, `rate_tensor_ema`, `ctrl_sensor` | float / int | control thread | host tools | per-sensor telemetry, appended at the end |
| `target_z`, `z_raw`, `z_hat`, `dzdt_hat`, `rate_ema`, `pm_error`, `integral`, `proactive_corrections`, `control_hz_ema`, `num_sm` | various | control thread | host tools | telemetry snapshots |

## 3. Functions

### Shared control block and zero-copy check [F-01]
`gputronic_create` picks device 0, reads the SM count, maps the control block, and calls
`gold_verify_zero_copy`. That check uses `cudaPointerGetAttributes` (memory type must be host,
host and device pointers must match the mapping) and a round trip: the host writes a value, the small
kernel `gold_zc_probe_kernel` reads it and writes back its complement, and the host checks it.
If any part fails, create returns NULL. There is no copy-based fallback.

### Persistent self-test kernel [F-02]
`gold_persistent_kernel`: one block per SM, 64 threads (two warps), launched once by
`gputronic_start`. Each pass it reads `control_flags` and `active_sms`. Blocks numbered at or above
`active_sms` skip the work and read those two words again. There is no device `__nanosleep` and no
`clock64` spin. The others run 16 × 8 calls of `gold_work` (8 dependent FMAs each), then pulse.
`work_scale_milli` is read only while bit3 (load) is set. A normal pass does not touch it.
An optional per-SM work tally (T11) is off unless `GPUTRONIC_SM_ACCUM=1` or
`GPUTRONIC_SM_ACCUM_SM` is set. Off, the kernel only tests a null pointer. On, each
active block adds its own CUDA-core ops in shared memory and, about once a millisecond,
posts one 40-byte slot `{ops_cuda, ops_tensor, t_ns, phys_sm, seq}` to a mapped host
array. `phys_sm` is the hardware SM id, read once at block start. The slot stays keyed
by the block index. That tally is a measurement. It is not an input to the PI.
`GPUTRONIC_IDLE_QUIET=1` is a test switch, off by default: idle blocks then wait 1 ms
between samples of the mapped control words.

### Using Gold from another kernel [F-03]
Another kernel can add to `total_work_pulses` with `atomicAdd` and run only `active_sms` blocks.
Set `launch_selftest_kernel = 0` and use `gputronic_get_control` to get the block.
`gputronic_pulse` adds from the host with an atomic add; only mix it with device atomics on platforms
that report `hostNativeAtomicSupported`.

### Pulse counter [F-04]
Each warp leader does `atomicAdd(&sm_counters[sm], 1)` (per-SM diagnostics in device memory) and
`atomicAdd(&ctrl->total_work_pulses, 1)`, and calls `__threadfence_system()` every 32 pulses.
Never assign the total from one SM; that bug made Z stick at a floor or ceiling in earlier versions.

`gputronic_reset_counters` clears it safely: the control thread raises the reset flag (blocks stop
pulsing and zero their per-SM counters), waits 2 ms for in-flight work, zeroes the total from the host,
clears the flag, and restarts its rate window. The 2 ms margin is an assumption, not a measurement.

### Rate [F-05]
The control thread ticks about every 10 µs. Pulses are summed until 8 ms have passed; that window
rate feeds an average weighted 0.7 old / 0.3 new (`rate_ema`). A 10 µs rate is never used directly,
because one kernel pass takes about 300 µs on the RTX 5080.

### Reference rate [F-06]
After start, 0.7 s of warm-up, then the pulse rate over 0.7–1.8 s becomes `rate_ref`. It is a
sustained average, not a peak.

### Z [F-07]
`Z = rate_ref / (rate_ema + 1)`, clamped to 0.3–10. Fewer SMs means a lower rate and a higher Z.

### Kalman filter [F-08]
`kalman_update`: two states (Z and dZ/dt), constant-velocity model, Q = 0.0008 per tick on both
diagonal terms, R = 0.018, run every tick. Because the measurement only changes every 8 ms, dZ/dt
behaves like spikes at window edges. See `docs/analysis/kalman-analysis.md`.

### Calibration table [F-09]
After `rate_ref`, the control thread runs the SM count at 3/4, 1/2, 1/4 of full width and at 4 SMs,
300 ms each (`probe_ms`), measuring the rate over the last two-thirds of each step.
`gold_build_table` converts the points to Z, finds the two points around the target, and
interpolates the starting SM count `n_ff`. `sms_per_u` is one over the local slope dZ/dn, so a
controller output of 1 means "one unit of Z". If the target is outside the table, it pins to the
nearest end. Calibration now takes about 3.0 s in total: 1.8 s reference plus 1.2 s of table steps, replacing the old 0.5 s half-width probe, so about 0.7 s longer than before PR #1.

### Controller [F-10]
Two-sided PI with an optional D term, around `n_ff`:

```
e = target_z - Z
u = Kp*e + Ki*I - Kd*dZ/dt
n = round(n_ff - u * sms_per_u),  clamped to 4..num_sm
```

- Inside the deadband (|e| < 0.08) the SM count is held and the integral is frozen.
- The integral is clamped to ±`integral_clamp` and does not grow further while `n` is pinned at 4
  or at full width (anti-windup).
- Below target it turns SMs off; above target it turns them back on gradually. There is no reset to
  full width at the target.
- Open-loop: `gputronic_set_open_loop_sms(h, n)` (n > 0 fixed count, 0 all SMs, < 0 closed loop).

### Actuator [F-11]
The control thread writes `active_sms`, then a memory barrier. The kernel reads it every pass.

### Two sensors, one actuator [F-16]
When `enable_tensor_sensor` is on (Gold default is off; `GPUTRONIC_TENSOR_SENSOR=1` turns it on;
forced off for Transfer/Episode, for `launch_selftest_kernel = 0`, and below sm_70), each block gets one extra warp. That warp runs 64
chained 16×16×16 FP16→FP32 `wmma` steps per pass and adds to `tensor_work_pulses`, with the same
fence every 32 pulses. The CUDA-core warps and `total_work_pulses` are unchanged. Blocks at or above
`active_sms` skip both kinds of work, so one actuator gates both sensors. The wmma code is compiled
only for sm_70 and newer.

The host keeps an 8 ms window and 0.7/0.3 average for each counter, its own reference rate
(`rate_ref`, `rate_ref_t`), and its own Z (`z_cuda`, `z_tensor`). The calibration table records both
rates at every probe count and fits a starting SM count and `sms_per_u` for each sensor.

**Control law: max selector with hysteresis.** Both Z values rise as SMs are turned off, so the PI runs
on the larger one. The target is reached by turning off the fewest SMs, and neither kind of work is
slowed more than requested. The controller switches sensor only when the other Z is higher by
`sensor_hysteresis` (0.02). On a switch it loads that sensor's starting count and `sms_per_u`, then
recomputes the integral so the commanded SM count does not jump (bumpless transfer), and moves the
Kalman estimate to the new value. A blend was rejected because it lets the more sensitive sensor go
past the target; in the model a 50/50 blend put the tensor Z at 1.54 when the CUDA Z was 1.42.
If the tensor counter produces no pulses during the reference window, the controller continues on
the CUDA-core sensor only.

Getters: `gputronic_get_z_cuda`, `gputronic_get_z_tensor`, `gputronic_get_rate_tensor`,
`gputronic_get_ctrl_sensor`. CSV columns added: `z_cuda, z_tensor, rate_tensor_ema, rate_ref_tensor,
ctrl_sensor`.

### Lifecycle [F-12]
`gputronic_config_gold` (defaults) → `gputronic_create` → `gputronic_start` (CSV, kernel, control
thread) → `gputronic_stop` (stop flag, join, device sync, stats) → `gputronic_destroy`.

### Checks [F-13]
`gputronic_gold dyno` sweeps fixed SM counts and fits rate against SM count. `step` runs all SMs, then
half, then all. `check` runs the dyno and a 20 s closed-loop run.

### Transfer and Episode kernels [F-14, F-15]
Cooperative GEMM tiles (`src/gputronic_transfer.cu`) and launch-per-episode GEMM
(`src/gputronic_episode.cu`) on the same interface. Their older sleep experiments set
`throttle_sleep_ns` themselves through a local `legacy_set_sleep` helper.

## 4. Parameters

| Name | Value | Where |
|---|---|---|
| Target Z | 1.5 | `GPUTRONIC_GOLD_TARGET_Z` |
| Kp, Ki, Kd | 0.55, 0.08, 0 | `GPUTRONIC_GOLD_KP/KI/KD`, `gputronic_set_gains`, `gputronic_set_kd` |
| Deadband | 0.08 | `GPUTRONIC_GOLD_DEADBAND`, `gputronic_set_deadband`, env `GPUTRONIC_DEADBAND` |
| Integral clamp | scaled after calibration: 0.04 × num_sm / (Ki × sms_per_u) | `integral_clamp` (> 0 = fixed override), `GPUTRONIC_GOLD_INTEGRAL_SMS_FRAC` |
| Z limits | 0.3 – 10 | `GPUTRONIC_GOLD_Z_FLOOR/CEILING` |
| Calibration | warm-up 0.7 s, reference 0.7–1.8 s, then 4 × 300 ms table | `control_thread`, `GPUTRONIC_GOLD_PROBE_MS` |
| Control tick | 10 µs | `GPUTRONIC_DEFAULT_CONTROL_DT_US` |
| Rate window | 8 ms, average 0.7 / 0.3 | `control_thread` |
| Threads per block | 64 | `GPUTRONIC_DEFAULT_THREADS_PER_BLOCK` |
| SM floor | 4 | kernel and control thread |
| Tensor sensor | off unless `GPUTRONIC_TENSOR_SENSOR=1`. One extra warp, 64 wmma steps per pulse | `enable_tensor_sensor`, `GOLD_TENSOR_STEPS` |
| Sensor rule | follow the higher Z (max) | `sensor_rule` (0 = max, 1 = min), env `GPUTRONIC_SENSOR_RULE` |
| Reserve | 0 extra SMs | `reserve_sms`, env `GPUTRONIC_RESERVE_SMS` |
| Tensor on/off | off | env `GPUTRONIC_TENSOR_SENSOR=0/1` |
| Sensor hysteresis | 0.02 | `sensor_hysteresis` (config field) |
| Per-SM tally | off | `GPUTRONIC_SM_ACCUM=1` or `GPUTRONIC_SM_ACCUM_SM=<id>` |

## 5. API changes in PR #1

Removed: `gputronic_set_open_loop_sleep`, `gputronic_get_sleep_ns`, config fields `sleep_scale`,
`auto_sleep_scale`, `one_sided`, and `gold_apply_width_scale` (replaced by `gold_build_table`).
Added: `gputronic_set_kd`, `gputronic_set_deadband`, `gputronic_reset_counters`,
`gputronic_get_active_sms`, config fields `kd` and `probe_ms`. The control block layout is unchanged.

Two-sensor change: new fields appended at the end of the control block; new config
fields `enable_tensor_sensor`, `sensor_hysteresis`; four getters listed under F-16.
`gold_launch_kernel` takes the tensor-sensor flag, the optional T9 work-counter
pointer, and the optional T11 accumulator pointer, SM id, and bin length. A normal
start passes nulls for both optional counters.

## 6. Reading order

1. `include/gputronic.h`: control block and API.
2. `gold_persistent_kernel` and `gold_work`.
3. `gputronic_create`, `gold_verify_zero_copy`.
4. `control_thread`: reset handling, rate window, calibration, controller.
5. `gold_build_table`, `kalman_update`.
6. `mode_dyno`, `mode_step`, `mode_closedloop_check`.

## 7. Known limits

- Whole-SM steps: about 0.015–0.033 Z per SM near target on 84 SMs, about 0.1 on a 20-SM card.
- Linear interpolation in the table leaves Transfer about 0.04 under target in the model.
- On 2026-10-12, tensor sensor off and deadband 0.08, the three checks passed on this 5080.
  Gold free-run was 543k–585k pulses/s across the evening. On the build that also carries
  the optional per-SM tally, left off, Gold `check` was free-run 541k pulses/s and closed
  363k, gate mean Z 1.463. Transfer was 8.48 GFLOP/s down to 6.03, mean Z 1.450. Episode
  was 22.38 GFLOP/s down to 14.94, mean Z 1.468. A 30 s run the same evening, on the
  previous build of this same law, ended at mean Z 1.466 over the last 20 s, with 85% of
  samples on 47 or 48 SMs. Open loop at a fixed 48 SMs had the same Z standard deviation
  (0.05) and the same pulse rate. The integral clamp in use is about 3.4 SMs
  (`0.04 × num_sm / (Ki × sms_per_u)`). It was not on the rail.
- Turning the tensor warp on cut the full-width CUDA-core rate by about 39% on this card, and
  the tensor rate was half the CUDA rate at every width. That is why the default is off.
- Idle blocks now poll the mapped control words with no pause. Full-width rate on this evening
  stayed in the same band as the previous binary. A half-width open-loop hold matched the
  closed-loop rate at that count.
- Open loop at 46 SMs, the optional per-SM tally on, the sum of the tallies matched the
  pulse counter at 100 ms: ratio 1.000, drift 0.01%, correlation 0.995. The working SMs
  were not equal. The coefficient of variation of mean ops per bin was 29%. SM 0, scaled
  by 46, sat at 0.75 of the pulse counter and still moved with it (correlation 0.977 at
  100 ms). The Z swing at a fixed count is a swing in how many passes finish.

## Test hooks (T6–T9)

- `GPUTronicControl.work_scale_milli` (appended last; host writes; 0/1000 = no load). The kernel
  reads it only while `GPUTRONIC_FLAG_LOAD` is set. At S > 1000 a pulsing warp emits one pulse
  per S/1000 batches (`gold_load_pulse`).
- Kernel argument `work_count` (device memory, `num_sm*32` slots, NULL by default): per-warp count of
  completed batches, enabled by `GPUTRONIC_WORK_COUNTER=1` (set by `ratio` mode).
- Environment: `GPUTRONIC_INTEGRAL_CLAMP` (> 0 fixes the clamp; unset = scaled).
- CLI modes: `loadstep`, `bursts`, `loadfile`, `ratio`, `fullwidth`, `noise` (see `gputronic_gold help` and `TEST_SUITE.md`).
- T11 per-SM tally, off by default. `GPUTRONIC_SM_ACCUM=1` reports every SM.
  `GPUTRONIC_SM_ACCUM_SM=<id>` reports one. The host writes
  `results/gputronic_gold_smaccum.csv` during `noise` and the other trace modes.
  `scripts/analyze_smaccum.py` compares that tally with `bin_pulses × 65,536`.
  The design note is `docs/analysis/per-sm-accumulator.md`.
