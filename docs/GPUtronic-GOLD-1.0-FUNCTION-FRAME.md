# GPUtronic 1.0 Gold: Function Reference

This document is the contract for each part of the Gold controller. A function page names
what the piece is for, when it runs, what it reads, what it writes, and the calculation.
It matches `include/gputronic.h` and `src/gputronic_gold.cu` in this tree.
The law below was run on an RTX 5080 on 2026-10-12 (driver 595.99.02, CUDA 12.8).
The same text is copied at the end of `README.md`.

Function IDs (F-01 … F-16) appear in brackets in source comments so you can jump between this file
and the code. Measured runs live in `docs/analysis/`. They are not repeated here.

## 1. What it is

GPUtronic holds a GPU at a requested rate of useful work. It counts finished work units, compares
the rate with the rate measured at full speed, and changes how many SMs are allowed to do the work.
Clocks, temperature and power stay with the NVIDIA driver. It uses only the stock driver and the
standard CUDA runtime.

- **Q**: finished work units per second (the pulse counter).
- **Z**: `rate_ref / rate`. About 1.0 at full speed; 1.5 means about two-thirds of full speed.
- **D**: temperature and power, left to the driver.

In scope: the self-test kernel in `src/gputronic_gold.cu`, and other kernels that add pulses and
honour `active_sms` (Transfer, Episode). Out of scope: programs that do not count their own work.

Two different reserves appear below. The OS reserve is how many SMs get no block at launch
(default 2). `reserve_sms` is an extra number of SMs added to the starting command after
calibration (default 0). They are not the same field.

## 2. Signals (`struct GPUtronicControl`)

Allocated with `cudaHostAlloc(..., cudaHostAllocMapped)`. The device pointer comes from
`cudaHostGetDevicePointer`. Host and device use that mapping. There is no copy fallback.

If a value is not in this struct, the kernel cannot see it. The kernel reads the command fields.
The telemetry fields are host snapshots. The kernel does not read them.

| Field | Type | Written by | Read by | Meaning |
|---|---|---|---|---|
| `control_flags` | volatile uint | host | kernel | bit0 stop, bit1 pause, bit2 reset, bit3 load, bit4 select, bit5 retire |
| `active_sms` | volatile int | control thread | kernel | number of working SMs (floor 2). 0 or at least `num_sm` means every launched block |
| `total_work_pulses` | volatile u64 | kernel (`atomicAdd` only) | control thread | the pulse counter |
| `throttle_sleep_ns`, `max_sleep_ns`, `base_sleep_ns` | int | create / Transfer / Episode | Transfer and Episode kernels | legacy sleep fields. Gold's controller does not use them |
| `tensor_work_pulses` | volatile u64 | tensor warp (`atomicAdd` only) | control thread | tensor-core pulse counter, appended at the end of the struct |
| `z_cuda`, `z_tensor`, `rate_tensor_ema`, `ctrl_sensor` | float / int | control thread | host tools | per-sensor telemetry, appended at the end |
| `target_z`, `z_raw`, `z_hat`, `dzdt_hat`, `rate_ema`, `pm_error`, `integral`, `proactive_corrections`, `control_hz_ema`, `num_sm` | various | control thread | host tools | telemetry snapshots |
| `work_scale_milli` | volatile uint | host | kernel, when the load flag or the cache is on | test load. 0 or 1000 means no load. Appended last |

`num_sm` in this struct is the launch count, after the OS reserve.

## 3. Functions

Each page is one piece of software. Test switches are not separate functions. They are off
unless a run sets them, and they are listed in section 4.

### Shared control block and zero-copy check [F-01]

**Purpose.** Give the host and the kernel one control block, and refuse to start if that block is not mapped.

**Runs when.** `gputronic_create`.

**Reads.** `GPUtronicConfig`, or the Gold defaults when the pointer is null. Device 0's SM count.

**Writes.** A `GPUtronicHandle`. The mapped `GPUtronicControl`. The launch count, which is the device SM count minus the OS reserve when the caller did not ask for a smaller grid.

**Calculation.** Allocate the block with `cudaHostAllocMapped` and take the device pointer. `gold_verify_zero_copy` requires host memory, matching host and device pointers, and a round trip: the host writes a value, `gold_zc_probe_kernel` writes back its complement, and the host checks it. Any failure frees the block and returns NULL.

**Code.** `gputronic_create`, `gold_verify_zero_copy`. Returns the handle, or NULL.

### Persistent self-test kernel [F-02]

**Purpose.** A known plant so the checks do not depend on another program.

**Runs when.** `gputronic_start`, if `launch_selftest_kernel` is set. One launch. The grid stays resident.

**Reads.** `control_flags`, `active_sms`, and `num_sm`. `work_scale_milli` only while the load flag is set, unless the control-word cache switch is on.

**Writes.** `total_work_pulses`, by `atomicAdd` from each warp leader. One `%smid` per block, once, from thread 0. `gputronic_log_smids` prints that map and the device SM ids that got no block.

**Calculation.** The launch count is the device SM count minus two, unless the caller set a smaller grid or `GPUTRONIC_OS_RESERVE`. CUDA chooses which physical SMs are left out. Each block is 64 threads (two warps) unless a test sets another count. Each pass, a block whose index is at or above `active_sms` skips the work and reads the command again. There is no device `__nanosleep` and no `clock64` spin. The others run 16 × 8 calls of `gold_work` (8 dependent FMAs each), then one pulse per warp leader. A command below 2 is treated as 2. A command of 0 or at least `num_sm` means every launched block.

STOP leaves the loop. PAUSE skips the pulse. RESET clears that block's per-SM counter and is how F-04 zeroes the total. RETIRE leaves the loop so the relaunch switch can start a new grid. Unset, the gate is `active_sms` and the grid is not relaunched.

**Code.** `gold_persistent_kernel`, `gold_work`, `gold_launch_kernel`.

### Using Gold from another kernel [F-03]

**Purpose.** Let a kernel we compile be the plant, using Gold's control thread.

**Runs when.** The caller sets `launch_selftest_kernel = 0` before `gputronic_create`.

**Reads.** The mapped block from `gputronic_get_control`, and its device pointer.

**Writes.** `total_work_pulses`, by `atomicAdd` in the kernel. The kernel runs the counted work on the first `active_sms` blocks only.

**Calculation.** Launch that kernel before or with `gputronic_start`, so the full-speed window sees pulses. `gputronic_pulse` adds from the host. Mix that host add with device atomics only when the device reports `hostNativeAtomicSupported`.

**Code.** `gputronic_get_control`, `gputronic_pulse`. Transfer is the resident form. Episode is the form that returns.

### Pulse counter [F-04]

**Purpose.** One total of finished work.

**Runs when.** A warp leader finishes a unit of work. Also when `gputronic_reset_counters` is called.

**Reads.** The block's own finished count.

**Writes.** `total_work_pulses`, and a per-SM diagnostic counter in device memory. Both by `atomicAdd`. Never by a plain store of one SM's own count.

**Calculation.** `__threadfence_system()` every 32 pulses, not every pulse. Reset, when a kernel is running: the control thread raises RESET, waits 2 ms, zeroes the total from the host, clears the flag, and restarts the rate window. The caller waits up to 50 ms. Return 0 when the clear finished, −1 when it did not. `rate_ref` and the calibration table are kept. The 2 ms wait is a margin for one in-flight pass, not a measured bound.

**Code.** The pulse site in `gold_persistent_kernel`. `gputronic_reset_counters`.

### Rate [F-05]

**Purpose.** Turn the pulse total into the rate the controller uses.

**Runs when.** Every control tick, about 10 µs. The rate itself updates when the window has been accumulated. The window is 8 ms unless a run sets `GPUTRONIC_RATE_WIN_US`.

**Reads.** `total_work_pulses`. The time since the previous tick.

**Writes.** `rate_ema`.

**Calculation.** Sum the pulse deltas and the tick times. When the summed time reaches the window (8 ms when `GPUTRONIC_RATE_WIN_US` is unset):

```
r_win = (pulses in the window) / (time in the window)
rate_ema = 0.70 * rate_ema + 0.30 * r_win
```

The first sample copies `r_win` straight into `rate_ema`. A rate over one 10 µs tick is not used. On this 5080 one kernel pass is about 300 µs, so a tick is shorter than the work.

**Code.** `control_thread`. Fields `rate_win_dt`, `rate_win_pulses`, `rate_ema`.

### Reference rate [F-06]

**Purpose.** Define full speed, so Z is about 1 when every launched block is working.

**Runs when.** Once, at the start of the control thread, before the PI runs.

**Reads.** `total_work_pulses` over the sustain window. The tensor counter too, when that sensor is on.

**Writes.** `rate_ref`. `rate_ref_t` when the tensor sensor stays on.

**Calculation.**

| Phase | Time | SM count | Action |
|---|---|---|---|
| Warm-up | 0 – 0.7 s | full launch | Discard. |
| Sustain | 0.7 – 1.8 s | full launch | Mark pulses at 0.7 s. |
| Close | t = 1.8 s | full launch | Accept the window or fall back. |

The window is accepted when its duration is longer than 0.2 s and it contains more than 1000 pulses. Then `rate_ref` is pulses divided by time. Otherwise `rate_ref` is `rate_ema`. The tensor reference uses the same window. If that window holds fewer than 100 tensor pulses, the tensor sensor is turned off and the run continues on the CUDA-core counter. During this phase the published Z is 1 and the PI does not run. The reference is not measured again if the clock changes later.

**Code.** The calibration block in `control_thread`.

### Z [F-07]

**Purpose.** A dimensionless output that is about 1 at the reference rate and rises when the rate falls.

**Runs when.** Every control tick after calibration, once `rate_ema` is a real sample.

**Reads.** `rate_ref`, `rate_ema`. The tensor pair when that sensor is on.

**Writes.** `z_raw`, then clamped. Per-sensor `z_cuda` and `z_tensor`.

**Calculation.**

```
z_raw = rate_ref / (rate_ema + 1)
```

Clamp to 0.3–10. The same form is used for the tensor counter. Before calibration, and while a table step is running, the published value is 1. If no window has arrived since a reset, the previous estimate is held.

**Code.** `control_thread`. `gputronic_get_z_raw` returns `z_raw`. `gputronic_get_z` returns the filtered value from F-08.

### Kalman filter [F-08]

**Purpose.** Estimate Z and how fast Z is changing.

**Runs when.** Every control tick after calibration, if `enable_kalman` is on. The Gold default is on.

**Reads.** `z_raw` from F-07. The tick duration.

**Writes.** `z_hat`, `dzdt_hat`. F-10 reads `z_hat` when this filter is on, and `z_raw` when it is off.

**Calculation.** Two states, Z and dZ/dt, constant-velocity prediction. Process noise Q = 0.0008 on both diagonal terms. Measurement noise R = 0.018. The measurement only changes every 8 ms, so dZ/dt spikes at the window edges. Kd is 0, so the controller does not use dZ/dt. The value is logged. See `docs/analysis/kalman-analysis.md`.

**Code.** `kalman_update`.

### Calibration table [F-09]

**Purpose.** Find the starting SM count at the requested Z, and the local slope that turns a controller output into SMs.

**Runs when.** Immediately after F-06. Four steps, then once at the end.

**Reads.** The pulse rate at each step. `target_z`. `probe_ms` (default 300). `reserve_sms` (default 0).

**Writes.** `n_ff` and `sms_per_u` for the CUDA-core sensor, and the same pair for the tensor sensor when it is on. Then the integral clamp, by F-10's rule.

**Calculation.** The steps are 3/4, 1/2, and 1/4 of the launch, then the command floor (2). Each step lasts `probe_ms`. The first third is settle. The rate is pulses over the last two-thirds, kept when that interval is longer than 0.02 s. Full width is the reference point, Z = 1. Each other point is `rate_ref / rate`. `gold_build_table` finds the two points around `target_z` and interpolates `n_ff`. `sms_per_u` is one over the slope dZ/dn on that segment, so a controller output of 1 means one unit of Z. If the slope is flatter than 0.001 per SM, it is treated as 0.001. `sms_per_u` is capped at the launch count. If the target is outside the table, `n_ff` pins to the nearest end. `reserve_sms` is then added to `n_ff` and capped at the launch count. That addition is not the OS reserve.

**Code.** `control_thread` probe block, `gold_fit_table`, `gold_build_table`.

### Controller [F-10]

**Purpose.** Hold Z near the request by choosing the SM count.

**Runs when.** Every control tick.

**Reads.** Z from F-08 when the filter is on, otherwise `z_raw` from F-07. `target_z`. Kp, Ki, Kd. `n_ff` and `sms_per_u` from F-09. The deadband. The integral clamp. `open_loop_sms`. The command floor and the launch count.

**Writes.** `active_sms`. `held_sms`. `integral`. The telemetry snapshots in the control block.

**Calculation.** In this order:

1. Not calibrated, or a table step is running: the count is the calibration command. The integral is 0.
2. `open_loop_sms > 0`: the count is that value. The integral is 0.
3. `open_loop_sms == 0`: the count is the full launch. The integral is 0.
4. Otherwise the PI runs:

```
e = target_z - Z
u = Kp*e + Ki*I - Kd*dZ/dt
n = round(n_ff - u * sms_per_u)
```

While `|e|` is below the deadband (0.08), the count stays where it is and the integral does not change. Outside the deadband the integral steps by `e * dt`, then stops at `±integral_clamp`. If that trial count is below the floor while `e` is positive, or above the full launch while `e` is negative, the integral is left unchanged for that tick. After the update, `n` is clamped to the floor through the launch count.

The clamp in force is, in this order: `GPUTRONIC_ICLAMP_SMS` if that test switch is set; else `integral_clamp` when the config value is greater than 0; else the scaled law

```
Ki * clamp * sms_per_u = 0.04 * num_sm
```

so the largest integral correction is 4% of the launch count, in SMs. On an 82-block launch that is 3.28 SMs. If Ki or `sms_per_u` is not positive, the clamp falls back to 1.

`gputronic_set_open_loop_sms(h, n)` sets case 2, 3, or 4: `n > 0` holds that count, `0` holds the full launch, and a negative value returns to the PI.

**Code.** `control_thread`. `gold_set_iclamp`. `gputronic_set_target`, `gputronic_set_gains`, `gputronic_set_kd`, `gputronic_set_deadband`, `gputronic_set_open_loop_sms`.

### Actuator [F-11]

**Purpose.** Make the count from F-10 the set of blocks that do the work.

**Runs when.** Every control tick, after F-10. The kernel reads the result every pass.

**Reads.** The count just computed.

**Writes.** `active_sms`, then a memory barrier.

**Calculation.** The default grid stays resident. Blocks below `active_sms` do the work and pulse. The others skip both. `gold_service_width` does not launch a new grid unless a run set idle-exit or relaunch. Those two switches are off. They are in section 4.

**Code.** The store in `control_thread`. The gate in `gold_persistent_kernel`. `gold_service_width`.

### Two sensors, one actuator [F-16]

**Purpose.** A second pulse counter for tensor-core work, on the same SM count.

**Runs when.** `enable_tensor_sensor` is on. The Gold default is off. `GPUTRONIC_TENSOR_SENSOR=1` turns it on for one run. It is forced off when `launch_selftest_kernel` is 0, and on a device below sm_70.

**Reads.** Its own pulse total, its own reference, and `sensor_rule`.

**Writes.** `tensor_work_pulses`. `z_tensor`, `rate_tensor_ema`, `ctrl_sensor`. On a switch, that sensor's `n_ff`, `sms_per_u`, and a recomputed integral.

**Calculation.** Each block gains one warp. That warp runs 64 chained 16×16×16 FP16-to-FP32 `wmma` steps and adds to `tensor_work_pulses`, with the same fence every 32 pulses. The CUDA-core warps are unchanged. The block-index gate skips both kinds of work, so there is still one actuator. Each counter has the F-05 window, the F-06 reference, the F-07 Z, and an F-09 table.

The default rule follows the higher Z. The PI then turns off the fewer SMs, and neither sensor is taken further past the request than the other. The controller changes sensor only when the other Z leads by `sensor_hysteresis` (0.02). On a change it loads that sensor's `n_ff` and `sms_per_u`, recomputes the integral so the commanded count does not jump, copies Z into the filter, and sets dZ/dt to 0. `GPUTRONIC_SENSOR_RULE=min` follows the lower Z instead. That comparison was made on recorded curves, not as a closed-loop run on the device. The default remains max.

**Code.** The tensor warp in `gold_persistent_kernel`. The selector in `control_thread`. Getters `gputronic_get_z_cuda`, `gputronic_get_z_tensor`, `gputronic_get_rate_tensor`, `gputronic_get_ctrl_sensor`.

### Lifecycle [F-12]

**Purpose.** Start and stop the controller in a fixed order.

**Runs when.** The caller drives it.

**Reads.** A `GPUtronicConfig` for create. Nothing further for stop.

**Writes.** The running flag, the CSV, the kernel, and the control thread. Stop sets the stop flag.

**Calculation.**

```
gputronic_config_gold
gputronic_create
gputronic_start
gputronic_stop
gputronic_destroy
```

`gputronic_create` returns the handle, or NULL. `gputronic_start` returns 0, or −1 if the handle is missing, already started, or the launch fails.

**Code.** Those five calls.

### Checks [F-13]

**Purpose.** Show that the counter works and that the rate moves with the SM count.

**Runs when.** The `gputronic_gold` commands `dyno`, `step`, and `check`.

**Reads.** The plant, through F-04 and F-05. `check` also reads Z from the closed loop.

**Writes.** A pass or a fail on stdout. CSV under `results/`.

**Calculation.** `dyno` holds fixed SM counts and fits rate against the count. `step` runs the full launch, then half, then the full launch. `check` runs `dyno` and then a 20 s closed loop. The rate in these checks is a wall-clock window of pulses.

**Code.** `mode_dyno`, `mode_step`, `mode_closedloop_check`.

### Transfer and Episode [F-14, F-15]

**Purpose.** Two other plants on the same interface.

**Runs when.** `gputronic_transfer` and `gputronic_episode`.

**Reads.** `active_sms` and the pulse total, through F-03.

**Writes.** Pulses for their own tiles. Their older sleep path can still store `throttle_sleep_ns` through a local helper. Gold's PI does not write that field.

**Calculation.** Transfer keeps a cooperative GEMM resident (`src/gputronic_transfer.cu`). Episode launches a GEMM once per piece of work and returns (`src/gputronic_episode.cu`). Both use the launch count, the floor, and the SM-id page from F-01 and F-02. The tensor sensor is forced off for both.

**Code.** Those two files. They do not replace F-10.

## 4. Parameters

Values in force when a run sets nothing. A test switch left unset is off, and the plant is the F-02 kernel above.

| Name | Value | Where |
|---|---|---|
| Target Z | 1.0 | `GPUTRONIC_GOLD_TARGET_Z`. Gold, transfer, and episode `check` command 1.5 |
| Kp, Ki, Kd | 0.55, 0.08, 0 | `GPUTRONIC_GOLD_KP/KI/KD`. Setters `gputronic_set_gains`, `gputronic_set_kd` |
| Deadband | 0.08 | `GPUTRONIC_GOLD_DEADBAND`, `gputronic_set_deadband`, env `GPUTRONIC_DEADBAND` |
| Integral clamp | scaled: `Ki * clamp * sms_per_u = 0.04 * num_sm` | `integral_clamp` (> 0 = fixed override), `GPUTRONIC_GOLD_INTEGRAL_SMS_FRAC`. Test only: `GPUTRONIC_ICLAMP_SMS` sets the budget in SMs for one run |
| Z limits | 0.3 – 10 | `GPUTRONIC_GOLD_Z_FLOOR/CEILING` |
| Calibration | warm-up 0.7 s, reference 0.7–1.8 s, then 4 × 300 ms | `control_thread`, `GPUTRONIC_GOLD_PROBE_MS` |
| Control tick | 10 µs | `GPUTRONIC_DEFAULT_CONTROL_DT_US` |
| Rate window | 8 ms, then 0.7 / 0.3 | `control_thread`. Test only: `GPUTRONIC_RATE_WIN_US` in microseconds. Unset keeps 8000. Floor 500, cap 100000 |
| Threads per block | 64 | `GPUTRONIC_DEFAULT_THREADS_PER_BLOCK` |
| SM floor | 2 | kernel and control thread. A quiet-idle sweep may still start at 4 |
| OS reserve | 2 | launch count is the device SM count minus 2. `GPUTRONIC_OS_RESERVE` overrides it for one run. Those SMs get no block |
| Starting-count reserve | 0 | `reserve_sms`, env `GPUTRONIC_RESERVE_SMS`. Added to `n_ff` after the table. Not the OS reserve |
| Fast trace | on | `GPUTRONIC_NO_CSV=1` skips the 10 µs file and the binned trace for one run |
| Tensor sensor | off | `GPUTRONIC_TENSOR_SENSOR=1` adds one warp and 64 wmma steps per pulse |
| Sensor rule | higher Z | `sensor_rule` (0 = max, 1 = min), env `GPUTRONIC_SENSOR_RULE` |
| Sensor hysteresis | 0.02 | `sensor_hysteresis` |
| Per-SM tally | off | `GPUTRONIC_SM_ACCUM=1` or `GPUTRONIC_SM_ACCUM_SM=<id>`. Not an input to the PI |
| Working-set filter | off | `GPUTRONIC_ACTIVE_PASSES` or `GPUTRONIC_ACTIVE_SMS`, then bit 4. Both at once selects neither |
| Local pulse count | off | `GPUTRONIC_PULSE_LOCAL=1`. One mapped add per block per 1 ms |
| Cached control words | off | `GPUTRONIC_CTRL_CACHE=1`. Period `GPUTRONIC_CTRL_CACHE_US`, default 1 ms, floor 50 µs. The refresh includes `work_scale_milli` |
| Quiet idle | off | `GPUTRONIC_IDLE_QUIET=1`. Idle blocks wait 1 ms between reads |
| Idle block returns | off | `GPUTRONIC_IDLE_EXIT=1`. The host launches a replacement on its own stream |
| Idle blocks still count | off | `GPUTRONIC_IDLE_WORK=1`. Blocks past `active_sms` run the same body and `atomicAdd` into `total_work_pulses`. Unset, those blocks do not. Setting this with idle-exit, relaunch, or quiet idle turns those three off |
| Relaunch the grid | off | `GPUTRONIC_RELAUNCH=1`. Hold `GPUTRONIC_RELAUNCH_HOLD_US`, default 5 ms, floor 0.5 ms, cap 1 s. Setting this and idle-exit together turns both off |
| Dependent peak body | off | `GPUTRONIC_PEAK_REPS` (max 256), `GPUTRONIC_PEAK_CHAINS` (max 16), `GPUTRONIC_THREADS` (32 to 1024). Selects a second kernel |
| Independent peak body | off | `GPUTRONIC_PEAK_ACC` (4, 8, 16, or 32) and `GPUTRONIC_PEAK_ROUNDS` (default 32, cap 4096). When both peak bodies are set, this one runs |
| Real-work load | off | `GPUTRONIC_WORK_LOAD=1`, and only with the independent body. `work_scale_milli` / 1000 multiplies the round count. Every batch still pulses. Without `GPUTRONIC_PEAK_ACC` the switch stays off |
| Open-loop schedule | off | Test only: `GPUTRONIC_OPEN_SCHED` is a loadfile-format path (`t_ms scale`, optional third column `active_sms`). Unset keeps scale 1.0 and the launch count. A count at or above the launch means every block. Floor 2. Closed-loop loadfile ignores the third column |
| Guest kernel | off | `GPUTRONIC_GUEST_BLOCKS` sets the block count. `GPUTRONIC_GUEST_AT_MS` (default 8000) is when `run` launches it. Its own mapped pulse word. It does not read `active_sms`. Each block records the SM id, the GPU time it entered, and the GPU time of its first published pulse. `GPUTRONIC_GUEST_STREAM_FIRST=1` creates the guest stream before Gold's streams. `GPUTRONIC_GUEST_PRIORITY=1` asks for the highest stream priority. `GPUTRONIC_GOLD_NONBLOCK=1` launches Gold's kernel on a non-blocking stream. All three stay off when unset. `guest` runs that kernel with the Gold kernel not launched |

## 5. Calls added and removed

Removed with the sleep-duration command: `gputronic_set_open_loop_sleep`, `gputronic_get_sleep_ns`, and the config fields `sleep_scale`, `auto_sleep_scale`, and `one_sided`.

Added for the SM-count command: `gputronic_set_kd`, `gputronic_set_deadband`, `gputronic_reset_counters`, `gputronic_get_active_sms`, and the config fields `kd` and `probe_ms`.

Added for the second sensor, appended so earlier offsets stay put: `tensor_work_pulses` and the four telemetry fields, config fields `enable_tensor_sensor` and `sensor_hysteresis`, and the four getters named under F-16.

The test switches in section 4 are arguments of the Gold launch. A normal start passes zeros, null counters, and the default stream. They are not part of F-03.

## 6. Reading order

1. `include/gputronic.h`: the signal list and the calls.
2. `gold_persistent_kernel` and `gold_work`.
3. `gputronic_create` and `gold_verify_zero_copy`.
4. `control_thread`: reset, rate window, reference, table, controller.
5. `gold_build_table` and `kalman_update`.
6. `mode_dyno`, `mode_step`, `mode_closedloop_check`.

## 7. Limits of the interface

- The command moves in steps of one SM. Near the request, that step was about 0.015–0.033 Z on an 84-SM card, and about 0.1 Z on a 20-SM card.
- The table interpolates a straight line between measured points. In the model that left the transfer plant about 0.04 under the request. That figure is the model, not a later hardware run.
- Z is clamped to 0.3–10. A very small rate prints the ceiling, so the printed Z stops being the true ratio.
- The integral correction cannot exceed `0.04 * num_sm` SMs unless a run sets a fixed clamp. The integral also stops growing when the count is already on the floor or on the full launch.
- The reference rate is one window at the start. A later clock change moves Z against that window. The controller can only answer with the SM count.
- The tensor sensor is off unless a run turns it on. The run that fixed that default is in the analysis notes.

## Test hooks (T6–T9)

These are inputs a check can set. A normal run leaves them alone.

- `work_scale_milli`, appended last. The host writes it. 0 or 1000 means no load. With the cache switch off, the kernel reads it only while `GPUTRONIC_FLAG_LOAD` is set. With `GPUTRONIC_CTRL_CACHE=1`, the block leader reads it in the same refresh as the other command words. At S > 1000 a pulsing warp emits one pulse per S/1000 batches (`gold_load_pulse`, or `gold_load_from_scale` when the cache is on). On the independent-FMA body this scale skips pulses and still runs the arithmetic, so it is not a reduction of that work. `GPUTRONIC_WORK_LOAD=1` is the other use of the same word: the round count changes with the scale, and every batch pulses. It stays off unless the run sets it.
- `work_count`, a kernel argument, `num_sm * 32` device slots, NULL unless `GPUTRONIC_WORK_COUNTER=1` (`ratio` mode). Per-warp count of finished batches.
- `GPUTRONIC_INTEGRAL_CLAMP` greater than 0 fixes the clamp for the process. Unset keeps the scaled law.
- CLI modes `loadstep`, `bursts`, `loadfile`, `ratio`, `fullwidth`, and `noise`. See `gputronic_gold help` and `TEST_SUITE.md`.
- The per-SM tally, off by default. `GPUTRONIC_SM_ACCUM=1` reports every SM. `GPUTRONIC_SM_ACCUM_SM=<id>` reports one. The host writes `results/gputronic_gold_smaccum.csv` during `noise` and the other trace modes. `scripts/analyze_smaccum.py` compares that tally with `bin_pulses × 65,536`. The note is `docs/analysis/per-sm-accumulator.md`. The tally is not an input to F-10.
