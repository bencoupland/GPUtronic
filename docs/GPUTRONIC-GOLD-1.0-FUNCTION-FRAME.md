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
| `active_sms` | volatile int | control thread | kernel | number of working SMs (floor 2). 0 or at least `num_sm` means every launched block |
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
`gold_persistent_kernel`: one block per launched SM, 64 threads (two warps), launched once by
`gputronic_start`. The launch count is the device SM count minus two. Those two SMs get no block,
so the desktop still has a place to run. CUDA chooses which physical SMs they are. Thread 0 of
each launched block writes `%smid` once into a mapped page, and `gputronic_log_smids` prints that
map and the two ids that were left out. Each pass the kernel reads `control_flags` and `active_sms`.
Blocks numbered at or above `active_sms` skip the work and read those two words again. There is no
device `__nanosleep` and no `clock64` spin. The others run 16 × 8 calls of `gold_work` (8 dependent
FMAs each), then pulse. The command floor is two working SMs.
`work_scale_milli` is read only while bit3 (load) is set, unless the control-word cache is on. A normal pass does not touch it. With the cache on, the block leader reads it in that refresh.
An optional per-SM work tally (T11) is off unless `GPUTRONIC_SM_ACCUM=1` or
`GPUTRONIC_SM_ACCUM_SM` is set. Off, the kernel only tests a null pointer. On, each
active block adds its own CUDA-core ops in shared memory and, about once a millisecond,
posts one 40-byte slot `{ops_cuda, ops_tensor, t_ns, phys_sm, seq}` to a mapped host
array. `phys_sm` is the hardware SM id, read once at block start. The slot stays keyed
by the block index. That tally is a measurement. It is not an input to the PI.
`GPUTRONIC_IDLE_QUIET=1` is a test switch, off by default: idle blocks then wait 1 ms
between samples of the mapped control words.
`GPUTRONIC_ACTIVE_PASSES` and `GPUTRONIC_ACTIVE_SMS` are test switches, off by
default. After calibration the host sets bit 4 (`GPUTRONIC_FLAG_SELECT`) and the
block-index gate is replaced by that list. Unset, the gate is still `active_sms`.
`GPUTRONIC_PULSE_LOCAL=1` is a test switch, off by default. Each warp keeps its
pulse count in a register, and the block leader adds the batch to the mapped
pulse counter about once a millisecond. Reads of the mapped control words stay
on every pass. Unset, every pulsing warp still adds to the mapped counter.
`GPUTRONIC_CTRL_CACHE=1` is a separate test switch, off by default. The block
leader reads the mapped control words, including `work_scale_milli`, once per
`GPUTRONIC_CTRL_CACHE_US` (default 1000, floor 50) and the other threads use
that copy. The pulse batch stays at one millisecond. Unset, every pass reads
the command words, and `work_scale_milli` is still read only while the load
flag is set. Tensor-on blocks do not take this cache.
`GPUTRONIC_ICLAMP_SMS` is a test switch, off by default. Set, it is the integral
clamp in SMs for that run (`Ki * clamp * sms_per_u`). Unset, the clamp is the
scaled law, or the fixed config value when one is set.
`GPUTRONIC_IDLE_EXIT=1` is a test switch, off by default. An idle block returns
instead of polling. The host launches a new block when the count later includes
that SM. Each refill is its own stream, so an earlier refill does not block the next. `GPUTRONIC_RELAUNCH=1` is the other test switch. The host retires the
grid and launches the commanded count, and it waits `GPUTRONIC_RELAUNCH_HOLD_US`
(default 5000, floor 500, cap 1000000) so a one-tick flicker does not relaunch. Setting both
turns both off. `GPUTRONIC_PEAK_REPS` (max 256), `GPUTRONIC_PEAK_CHAINS` (max 16), and
`GPUTRONIC_THREADS` (warp-aligned, 32 to 1024) are test switches, off by default. They repeat the FMA batch,
add independent FMA chains, or raise the thread count. Setting either peak
switch selects a second kernel, including when the counts are 1, and that
kernel stores the nest so the compiler keeps it. `GPUTRONIC_PEAK_ACC` (4, 8, 16, or 32)
selects a different second kernel: that many independent FMAs per thread, repeated
`GPUTRONIC_PEAK_ROUNDS` times (default 32, cap 4096). One pulse is one finished block.
When both are set, the independent kernel is the one that runs. Unset, the launch is the
other kernel: the 16 × 8 `gold_work` loop at 64 threads.

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
After `rate_ref`, the control thread runs the SM count at 3/4, 1/2, 1/4 of full width and at the
command floor (2 SMs), 300 ms each (`probe_ms`), measuring the rate over the last two-thirds of each step.
`gold_build_table` converts the points to Z, finds the two points around the target, and
interpolates the starting SM count `n_ff`. `sms_per_u` is one over the local slope dZ/dn, so a
controller output of 1 means "one unit of Z". If the target is outside the table, it pins to the
nearest end. Calibration now takes about 3.0 s in total: 1.8 s reference plus 1.2 s of table steps, replacing the old 0.5 s half-width probe, so about 0.7 s longer than before PR #1.

### Controller [F-10]
Two-sided PI with an optional D term, around `n_ff`:

```
e = target_z - Z
u = Kp*e + Ki*I - Kd*dZ/dt
n = round(n_ff - u * sms_per_u),  clamped to 2..num_sm
```

- Inside the deadband (|e| < 0.08) the SM count is held and the integral is frozen.
- The integral is clamped to ±`integral_clamp` and does not grow further while `n` is pinned at 2
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
| Integral clamp | scaled after calibration: 0.04 × num_sm / (Ki × sms_per_u) | `integral_clamp` (> 0 = fixed override), `GPUTRONIC_GOLD_INTEGRAL_SMS_FRAC`. Test only: `GPUTRONIC_ICLAMP_SMS` sets the same budget in SMs for one run |
| Z limits | 0.3 – 10 | `GPUTRONIC_GOLD_Z_FLOOR/CEILING` |
| Calibration | warm-up 0.7 s, reference 0.7–1.8 s, then 4 × 300 ms table | `control_thread`, `GPUTRONIC_GOLD_PROBE_MS` |
| Control tick | 10 µs | `GPUTRONIC_DEFAULT_CONTROL_DT_US` |
| Rate window | 8 ms, average 0.7 / 0.3 | `control_thread` |
| Threads per block | 64 | `GPUTRONIC_DEFAULT_THREADS_PER_BLOCK` |
| SM floor | 2 | kernel and control thread. A quiet-idle sweep may still start at 4 |
| OS reserve | 2 | launch count is the device SM count minus 2. `GPUTRONIC_OS_RESERVE` overrides the spare for one run. No test block on those SMs |
| Fast trace | on | `GPUTRONIC_NO_CSV=1` skips the 10 µs file and the binned trace for one run |
| Tensor sensor | off unless `GPUTRONIC_TENSOR_SENSOR=1`. One extra warp, 64 wmma steps per pulse | `enable_tensor_sensor`, `GOLD_TENSOR_STEPS` |
| Sensor rule | follow the higher Z (max) | `sensor_rule` (0 = max, 1 = min), env `GPUTRONIC_SENSOR_RULE` |
| Reserve | 0 extra SMs | `reserve_sms`, env `GPUTRONIC_RESERVE_SMS` |
| Tensor on/off | off | env `GPUTRONIC_TENSOR_SENSOR=0/1` |
| Sensor hysteresis | 0.02 | `sensor_hysteresis` (config field) |
| Per-SM tally | off | `GPUTRONIC_SM_ACCUM=1` or `GPUTRONIC_SM_ACCUM_SM=<id>` |
| Working-set filter | off | `GPUTRONIC_ACTIVE_PASSES` or `GPUTRONIC_ACTIVE_SMS`, then bit 4 |
| Local pulse count | off | `GPUTRONIC_PULSE_LOCAL=1`. One mapped atomic per block per 1 ms |
| Cached control words | off | `GPUTRONIC_CTRL_CACHE=1`. Period `GPUTRONIC_CTRL_CACHE_US`, default 1 ms. The refresh includes `work_scale_milli` |
| Clamp in SMs | off | `GPUTRONIC_ICLAMP_SMS`. Unset keeps the scaled law |
| Idle block returns | off | `GPUTRONIC_IDLE_EXIT=1`. Host launches replacements |
| Relaunch the grid | off | `GPUTRONIC_RELAUNCH=1`. Hold `GPUTRONIC_RELAUNCH_HOLD_US`, default 5 ms, floor 0.5 ms, cap 1 s |
| Peak work | off | `GPUTRONIC_PEAK_REPS` (max 256), `GPUTRONIC_PEAK_CHAINS` (max 16), `GPUTRONIC_THREADS` (32 to 1024), `GPUTRONIC_PEAK_ACC` (4, 8, 16, or 32) with `GPUTRONIC_PEAK_ROUNDS` (default 32, cap 4096) |

## 5. API changes in PR #1

Removed: `gputronic_set_open_loop_sleep`, `gputronic_get_sleep_ns`, config fields `sleep_scale`,
`auto_sleep_scale`, `one_sided`, and `gold_apply_width_scale` (replaced by `gold_build_table`).
Added: `gputronic_set_kd`, `gputronic_set_deadband`, `gputronic_reset_counters`,
`gputronic_get_active_sms`, config fields `kd` and `probe_ms`. The control block layout is unchanged.

Two-sensor change: new fields appended at the end of the control block; new config
fields `enable_tensor_sensor`, `sensor_hysteresis`; four getters listed under F-16.
`gold_launch_kernel` takes the tensor-sensor flag, the optional T9 work-counter
pointer, the optional T11 accumulator pointer, SM id, and bin length, the idle-quiet
flag, the working-set masks, the test flags for a local pulse count and cached
control words, and that cache period. A normal start passes nulls for both
optional counters and leaves the test flags off. The eighth-pass switches add
the idle-exit flag, the slot generation pointer, the launch length and base,
the peak repeat and chain counts, the independent-chain count and its round
count, whether that peak kernel is used, and the stream. A normal start passes
zeros and the default stream.

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

- `GPUTronicControl.work_scale_milli` (appended last; host writes; 0/1000 = no load). With the
  cache switch off, the kernel reads it only while `GPUTRONIC_FLAG_LOAD` is set. With
  `GPUTRONIC_CTRL_CACHE=1`, the block leader reads it in the same refresh as the other control
  words and the warps use that copy. At S > 1000 a pulsing warp emits one pulse per S/1000
  batches (`gold_load_pulse`, or `gold_load_from_scale` when the cache is on).
- Kernel argument `work_count` (device memory, `num_sm*32` slots, NULL by default): per-warp count of
  completed batches, enabled by `GPUTRONIC_WORK_COUNTER=1` (set by `ratio` mode).
- Environment: `GPUTRONIC_INTEGRAL_CLAMP` (> 0 fixes the clamp; unset = scaled).
- CLI modes: `loadstep`, `bursts`, `loadfile`, `ratio`, `fullwidth`, `noise` (see `gputronic_gold help` and `TEST_SUITE.md`).
- T11 per-SM tally, off by default. `GPUTRONIC_SM_ACCUM=1` reports every SM.
  `GPUTRONIC_SM_ACCUM_SM=<id>` reports one. The host writes
  `results/gputronic_gold_smaccum.csv` during `noise` and the other trace modes.
  `scripts/analyze_smaccum.py` compares that tally with `bin_pulses × 65,536`.
  The design note is `docs/analysis/per-sm-accumulator.md`.
