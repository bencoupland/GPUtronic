# Fifteenth pass: sweep the workload from idle to the measured peak and map where rate_ref sits

This follows `docs/directives/fourteenth-pass-2026-10-18.md` and its results on `test-results/2026-10-18_fourteenth-pass` (`74601ce`, report `docs/analysis/fourteenth-pass-2026-10-18.md`). Start from that results branch at `74601ce`. The source is `b85393e`: `add4304` plus the test-only `GPUTRONIC_RATE_WIN_US`. Post everything to a new branch, `test-results/2026-10-18_fifteenth-pass`, with logs under `logs/2026-10-18_fifteenth-pass/`. If you run on a later date, use that date in the branch, the log directory and the report name. Don't force-push, and don't merge anything.

## Start here

Do item 1's zero and floor points first, then items 2 and 3, then item 1's ceiling hold, then item 4, then item 5. Then the success criteria in item 6 and the report and checks in item 7. Follow the window policy below.

Ben's ask for this pass is a sweep, not more steps. Steps only sample the space. Sweep the workload across its whole range, from no work at all up to the highest counted throughput the RTX 5080 sustains. Measure that ceiling. Don't assume a number for it. At every point map:

1. `rate_ref`, and the per-SM rate;
2. the achieved counted ops/s (TFLOPS from the counted FMAs only);
3. the strength and period of the 136 ms wave, and the `z_raw` SD at a fixed count.

Most of it is open loop (`fullwidth` and `noise`), to isolate the plant. A small number of closed-loop confirmations run at any point that reaches the 49 million group, against a same-session baseline in the 47 million group.

**You may argue.** If you disagree with a directive, say so in the report with your reasoning and evidence, before or instead of carrying it out. A reasoned objection counts as a result. Don't quietly work around a directive. The source reading below is Grok Bot's. If the code says otherwise, the code wins, and say so.

## How GPUtronic is meant to work (carried forward)

- **Z = 1.0 is the normal default.** The loop stays out of the way and runs at the fastest rate, the plant's own free-run rate. `GPUTRONIC_GOLD_TARGET_Z` is `1.0f`. Keep it.
- **Z above 1 is an explicit margin mode.** You command it when margin matters more than speed. It carries a throughput penalty. In this pass 1.5 is that mode, set with `GPUTRONIC_TARGET_Z` or in the check gates, never as a default.
- **The clamp is a guardrail.** It should almost never trip. If it trips on real workloads, the fix is upstream.
- **The stock driver owns thermal and power. GPUtronic owns scheduling.** Keep logging clocks and flags beside Z, read only.
- **Closed loop is the differentiator.** Stay standalone and loosely coupled, and add no knob a result doesn't need.

## Where things stand after the fourteenth pass

- **The low rate is in the plant, not the loop.** Eight `fullwidth 15` launches, 82 SMs, PI idle (`sm_moves=0`, `integral_updates=0`), gave `rate_ref` 46,979,676 to 47,258,972 and a per-SM hold rate of 579,545 to 581,155. The thirteenth quiet group was 49.1–49.6 million, about 601k per SM. That group did not appear in the fourteenth session. Every launch was low from the first warm-up window.
- **Every SM is slow together.** `GPUTRONIC_SM_ACCUM=1`: per-SM op rate max/min 1.008 and 1.006. `GPUTRONIC_WORK_COUNTER=1`: pulse-to-batch ratio medians 1.00018 and 0.99992. The low rate is the work rate, not lost pulses and not a subset of SMs.
- **The 136 ms line is in Z with the count fixed.** `fw-01` `z_raw` autocorrelation at 82 SMs: 0.438 at 128 ms, 0.560 at 136 ms, 0.429 at 144 ms. With `GPUTRONIC_PULSE_LOCAL` unset the rate falls to about 1.3–1.4 million, flat from 20 to 82 SMs, and the 136 ms lag stays (0.480 at 136 ms on `fw-localoff-a`). The SM moves follow the wave rather than create it.
- **A shorter rate window didn't stop the hunt.** On the same 47 million plant: `sm_moves` 2,294 at 8 ms, 5,916 at 4 ms and 16,566 at 2 ms. Z is seen sooner, the count moves more, and the count still autocorrelates at 136 ms.
- **Other clusters.** Same settings gave 43,830,116 (`i1-b`), 43,762,896 (cache 250 µs), 44,398,800 (cache 100 µs and window 2 ms), and 19,779,312 (cache 100 µs). The accumulator launches were 46,355,140 and 46,201,280. Name a cluster by its measured `rate_ref`, not by "low".
- **Thirteenth-pass history.** Run b went from slow to fast at 254.3 s (14:43:33 PDT), and every later run that session was fast. `nvidia-smi` showed `Disp.A On` at the start and `Off` at the end, and when that changed isn't logged. In the fourteenth session `display_active` was `Enabled` on all 1,762 samples, and every launch was slow. That doesn't make the display the cause. It's one difference that hasn't been tested.
- **Untested so far:** whether any workload setting moves a launch from the 47 million plant to the 49 million plant, and whether the 136 ms wave changes with the work per pass, the threads per block or the SM count. Every launch since the twelfth pass has used `PEAK_ACC=32`, `PEAK_ROUNDS=512`, `THREADS=512`. That is one point in the space.

## What the source says (read it yourself and correct this if it's wrong)

On `b85393e`, the workload-side switches and their ranges:

| Switch or constant | Unset | Range in the source | What it moves |
| --- | --- | --- | --- |
| `GPUTRONIC_PEAK_ACC` | off (default nest) | 4, 8, 16 or 32. Anything else prints `[PEAK] acc … is not 4, 8, 16, or 32` and stays off | Independent FMA chains per thread. Selects the `<1, Acc>` kernel |
| `GPUTRONIC_PEAK_ROUNDS` | 32 | floor 1, cap 4096 | Rounds of `Acc` FMAs per pass. Only with `PEAK_ACC` |
| `GPUTRONIC_THREADS` | 64 (`GPUTRONIC_DEFAULT_THREADS_PER_BLOCK`) | 32 to 1024, rounded down to a multiple of 32 | Warps per block. One block per launched SM, so this is the per-SM occupancy |
| `GPUTRONIC_WORK_LOAD=1` | off | needs `PEAK_ACC`. Rounds = `(PEAK_ROUNDS × scale_milli + 500) / 1000`, floor 1, cap 4096 | The schedule's scale changes the rounds. Every pass pulses |
| `loadfile` scale | 1.0 | `gold_load_read` keeps only `scale > 0`. `scale_milli` 0 means 1000 in the kernel. At `PEAK_ROUNDS=512`, scale 0.001 gives 1 round and scale 8.0 gives 4096 | Load schedule. Applied in `loadfile`, `loadstep` and `bursts` only, which are all closed loop |
| `fullwidth [sec]` / `noise [sms] [sec]` | 120 s / 46 SMs, 30 s | `noise` count floor 2 (`GPUTRONIC_SM_FLOOR`). A count at or above the launch means every block | Open-loop SM count. These modes pass no schedule (`np = 0`), so the scale stays 1.0 |
| `GPUTRONIC_STEP_AT_MS` / `GPUTRONIC_STEP_SMS`, and `STEP2_*` | off | two count changes per launch, floor 2 | Open-loop count steps during `noise` |
| `GPUTRONIC_PULSE_LOCAL=1` | off | on or off. The publish period is the compiled `GOLD_PULSE_LOCAL_NS`, 1 ms, with no switch | One mapped add per block per 1 ms |
| `GPUTRONIC_CTRL_CACHE=1`, `GPUTRONIC_CTRL_CACHE_US` | off; 1000 when on | floor 50, cap 100000 | How late a block sees a new count or scale |
| `GPUTRONIC_TRACE_BIN_US` | 10 ms (1 ms with the accumulator) | floor 100 | Trace bin only |
| `GPUTRONIC_RATE_WIN_US` | 8000 (the `0.008` literal) | floor 500, cap 100000 | Rate window. Test only. Leave it unset in this pass |
| `GPUTRONIC_PEAK_REPS`, `GPUTRONIC_PEAK_CHAINS` | 1, 1 | 256, 16 | Dependent nest. Not used when `PEAK_ACC` is set |
| `GPUTRONIC_TENSOR_SENSOR=1` | off | on or off | Adds a wmma warp. That block skips the cache and the local pulse |
| `GPUTRONIC_IDLE_WORK`, `IDLE_QUIET`, `IDLE_EXIT`, `RELAUNCH` | off | on or off | Idle-block behaviour. Leave them unset |
| `GPUTRONIC_OS_RESERVE` | 2 | changes the launch count | Don't use it. The 82-block launch is locked |
| `GOLD_BATCH` | 16, compiled | no switch | The default nest, the dependent nest, and the work counter's batch unit (`ratio = pulses × 16 / batches`). It doesn't appear in the `PEAK_ACC` body |
| `GOLD_WORK_ITERS` | 8, compiled | no switch | The same two nests. Not in the `PEAK_ACC` body |

What that means for the sweep:

- **The work per pulse on the full-pipe path is `PEAK_ACC × rounds` FMAs per thread.** `GOLD_BATCH` isn't in it. Sweeping `GOLD_BATCH` would not change the full-pipe workload, so no batch switch is needed for that. The "batch" dimensions that exist are the round count (work per pass) and the 1 ms publish period (pulses per mapped add).
- **Counted ops.** On the `PEAK_ACC` body with `GPUTRONIC_WORK_LOAD=1`, one pulse is one warp pass, and `ops_add = Acc × rounds × 2 × 32`. Grok Bot's read: counted ops/s = pulse rate × `PEAK_ACC` × rounds × 64, with an FMA as 2 FP32 ops, and TFLOPS is that over 1e12. The extra adds that sum the accumulators aren't counted. Check the formula against `GPUTRONIC_SM_ACCUM=1` at a few points. If they disagree, the tally wins, and say so.
- **What the existing knobs cover.** On the counted path the lowest work per pulse is `PEAK_ACC=4`, rounds 1, `THREADS=32`: 4 FMAs per thread per pass, on 2 SMs at the floor with `noise 2`. The highest is `PEAK_ACC=32`, rounds 4096, `THREADS=1024`, on all 82 SMs. Rounds can't reach 0. `gold_scaled_rounds`, the `indep_rounds` setup and `gold_work_indep` all floor at 1. No existing mode runs the kernel with zero counted work. The unset kernel (64 threads, default nest) isn't a counted point either: the comment on the launch says ptxas removes that nest. Treat it as a pulse-path reference only.
- **Zero work is the GPU without GPUtronic.** The floor of 2 SMs keeps at least two blocks working, and idle blocks spin on the control words. The only true zero is no process: `nvidia-smi` alone.
- **Open loop has no schedule.** `fullwidth` and `noise` call `gold_load_run` with no points, so the scale stays 1.0 for the whole launch. With the existing switches an open-loop sweep is one launch per point, each with its own calibration. A ramp exists only in closed loop, through `loadfile`.
- **Occupancy per SM is threads per block.** The launch is one persistent block per launched SM (82). `THREADS` tops out at 1024, so more than 32 warps per SM would take more than one block per SM. That changes the 82-block launch, so it's out of scope this pass. Read `maxThreadsPerMultiProcessor` and the register count of each kernel variant from the device and the build, and report them.
- **`rate_ref` is measured at the launch's own settings, at scale 1.0.** Warm-up 0.7 s, then the sustained window to 1.8 s at 82 SMs, then the 300 ms probes at 61, 41, 20 and 2. A launch at `PEAK_ROUNDS=2048` does more work per pulse, so its `rate_ref` is lower on the same plant. In the fourteenth pass the loaded per-SM rate sat near 512/614 of the return. Only compare `rate_ref` across launches at the same `PEAK_ACC`, rounds and threads. Compare everything else in counted ops/s and ops/s per SM.
- **The per-tick file.** `control_thread` writes one row per control tick to `cfg.csv_path`: `results/gputronic_gold_fullwidth.csv` for `fullwidth` and `noise`, and `results/gputronic_gold_load.csv` for `loadfile`. Each launch overwrites it, so copy it out before the next launch. `GPUTRONIC_NO_CSV=1` turns off both that file and the 1 ms trace, so don't use it.

### Switches you may add

Each is optional, test only and off by default. Unset, it stores the existing literal and prints nothing, so with every switch unset the binary behaves exactly like `b85393e`. Name each one in the report, with what it does, its range, and confirmation that it's off by default, the way `GPUTRONIC_RATE_WIN_US` was reported. Update the function frame and the README copy together if a parameter table changes.

1. **An open-loop schedule.** Let `fullwidth` and `noise` read a schedule in the `loadfile` format (`t_ms scale`), so one launch can step or ramp the work with the PI idle. You may give it an optional third column for the open-loop SM count. It's the one addition that turns items 2–4 from one launch per point into real sweeps. If you don't add it, run items 2 and 3 as one launch per point, and run item 4 closed loop through `loadfile` only. Say which you did.
2. **A zero-round pass.** Allow rounds 0 on the `PEAK_ACC` body, for one point that keeps the kernel, the pulse and the control path, does no FMA rounds, and counts 0 ops. Unset keeps the floor of 1. Only if you think the zero-work end needs more than the no-process baseline.
3. **The pulse publish period.** `GOLD_PULSE_LOCAL_NS` is a 1 ms literal. A switch that stores `1000000` when unset would let the pulses per mapped add be swept as a batch dimension, for example 250 µs to 4 ms. Optional. Keep it inside the 8 ms rate window.

Don't add a block-count switch, a `GOLD_BATCH` switch, or a tick switch.

## Window policy

- **30 s windows, not a minute or more.** The closed-loop holds in item 5 are 30 s. Report the whole 30 s hold and its last 10 s.
- **Sweep points are 10–15 s.** In the fourteenth pass Z was half-way to its new value about three windows (about 24 ms) after a count change. Drop the first 2 s of each point and analyse the rest. Spend the saved time on more points and on repeats. More points beat settling precision in this pass.
- Avoid 5–20 min runs. No single launch longer than about 5 minutes unless a result needs it. If you lengthen one, say why. Split a long sweep into launches that each start and end on the base point.
- **Base point.** `PEAK_ACC=32`, rounds 512, `THREADS=512`, 82 SMs, scale 1.0, the full-pipe switches, `GPUTRONIC_RATE_WIN_US` unset. Bracket every sweep with it: at the start and end of every launch when the schedule switch is in, or as an interleaved launch at least every eight points when it isn't. A plant change caused by a sweep point should then show as a step in the base per-SM rate.
- **Check the group on every launch.** Read the `[CAL] rate_ref` line. At base settings, the fourteenth fullwidth band was 46.98–47.26 million (about 580k per SM), and the thirteenth quiet band was 49.1–49.6 million (about 601k per SM). Report the base per-SM rate in 10 s chunks, so a mid-launch change like run b's isn't read as the effect of a setting.
- Keep the eleventh-pass definitions for the closed-loop holds: the band is `|z_hat - target| <= 0.08`, "enters and stays" is the time after the last out-of-band bin, and a limit cycle is exactly two SM counts in the window, each at least five times, with at least six changes. Say which window each share uses.
- **The wave, measured the same way every time.** Use the fourteenth method (`numbers.py`): the 1 ms series downsampled to 4 ms, the autocorrelation coefficients at 128, 136 and 144 ms, the strongest lag from 50 to 400 ms, and the 7.25–7.45 Hz share of the 0.5–60 Hz power. Report the coefficients, not just the lag. If the strongest lag moves off 136 ms, give the share at the new period as well. `z_raw` SD at a fixed count: in open loop, the SD over the point's analysis window and the median SD of its 500 ms stretches.

## Standing rules

- Kp 0.55, Ki 0.08, Kd 0, deadband 0.08, 82-block launch, floor 2. Physical SMs 82 and 83 stay unlaunched. The gains are locked. None of these change.
- The compiled target stays at 1.0, and the check gates keep commanding 1.5.
- Ben hasn't picked a compiled clamp width. The compiled clamp stays at 3.28 SMs on an 82-block launch. `GPUTRONIC_ICLAMP_SMS=15` and `GPUTRONIC_TARGET_Z=1.5` are for the closed-loop hunting tests only, not defaults.
- `GPUTRONIC_RATE_WIN_US` stays available, test only, off by default. Unset is 8 ms. Leave it unset in this pass unless a result needs it, and say why if you set it.
- `libsmctrl` stays out.
- **The controller's view has to be complete.** Every batch that runs is counted. No uncounted work.
- **Accepted limit:** idle SMs spin at the GPU clock on the stock driver.
- **Out of scope:** custom drivers, per-SM sleep, per-SM clock or power gating, thermal or power logic in the controller, and anything below the CUDA runtime and driver API.
- Any new switch is test only and off by default. A logging-only addition is gated the same way and defaults off. With every switch unset, the binary behaves exactly like `b85393e`. Report every switch or log you add, with its name, what it does and confirmation that it's off by default.
- One GPU job at a time. Stop any run at 83 °C; the log up to the stop is the result.
- No number that isn't measured in this pass or already in the repo, quoted from its file. An early stop or a ruled-out idea is a result.
- The hold region is a deadband, not hysteresis. The Kalman filter is logged only. No motor-control terms for the loop. Source note: the PI error is computed from the Kalman `z_hat` (`enable_kalman = 1`), and only `dZ/dt` goes unused, because Kd is 0. Don't change that this pass. If you read it differently, say so.
- **The desktop is Ben's call.** Log display state once a second, read only: `nvidia-smi --query-gpu=timestamp,display_active,display_mode,pstate,utilization.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active --format=csv -lms 1000`. Take the `nvidia-smi` process list at the start and end of every launch. Record any change you see. Don't change display blanking, the compositor or open applications.
- Leave every earlier results branch, the notes branch, `archive/` and all earlier directives as they are.

## Logging on every launch

- Full-pipe switches unless a point changes one: `GPUTRONIC_PEAK_ACC=32`, `GPUTRONIC_PEAK_ROUNDS=512`, `GPUTRONIC_THREADS=512`, `GPUTRONIC_PULSE_LOCAL=1`, `GPUTRONIC_CTRL_CACHE=1`, `GPUTRONIC_TRACE_BIN_US=1000`, `GPUTRONIC_WORK_LOAD=1`. Idle-exit, relaunch, quiet idle, idle work, guest, tensor sensor and `GPUTRONIC_OS_RESERVE` unset.
- **SM count and Z every control cycle.** Keep the per-tick file of every launch. Commit it the fourteenth way: rows with `time_s` ≤ 3 s, plus every later row whose printed `z_raw`, `rate_ema` or `active_sms` differs from the previous row, gzipped as `*-tick-cut.csv.gz`, with a `cut-manifest.txt` of raw rows, kept rows and gzip bytes. Delete the raw file only after the gzip reads back.
- The 1 ms trace for every launch. If the total for the session is too large to commit whole, gzip the sweep traces and say so.
- Clocks and flags (`.flags`, `.dmon`), display state at 1 s, and the process list at the start and end.
- One table row per sweep point in a committed CSV (`sweep-points.csv` or similar): launch, point start and end on the trace clock, `PEAK_ACC`, rounds (the formula integer), threads, SM count, scale, the launch `rate_ref`, the median pulse rate, the per-SM pulse rate, the counted ops/s and TFLOPS, the ops/s per SM, `z_raw` SD, the three autocorrelation coefficients, the strongest lag, the narrow-band share, the median SM clock, the median power, the highest die temperature, the reason words seen, and `display_active`. Also give the pass time per warp, `(active SMs × warps per block) / pulse rate`, which is derived from counted values.

## 1. The two ends of the range

**Zero.** With no GPUtronic process running, log 30 s of the display-state query and the flags. That's the zero-work point: clocks, power, P-state and utilization. There's no `rate_ref` for it, so don't print one.

**The floor of the counted path.** `fullwidth 15` and `noise 2 15` at `PEAK_ACC=4`, `PEAK_ROUNDS=1`, `THREADS=32`, with the other full-pipe switches. Report the same row as every sweep point. If you added the zero-round switch, run it here too, at 82 SMs and at 2.

**The ceiling.** Don't run a separate ceiling search first. The ceiling is the highest counted ops/s that items 2 and 3 measure, sustained over the point's analysis window. When the sweeps are done, take the setting with the highest counted ops/s and run it for one 30 s open-loop hold at 82 SMs. Then run it once with `GPUTRONIC_SM_ACCUM=1` and once with `GPUTRONIC_WORK_COUNTER=1`. Both switches add traffic, so say they were on. Report the ceiling in counted TFLOPS, the setting, the SM clock, the power and the reason words over that hold. If the driver lowers the clock at that point, that's the driver's job and it's a result.

## 2. Work-per-pass sweep, open loop at 82 SMs

The question is whether the work per pass moves the plant off the 47 million floor, and whether the 136 ms wave tracks the pass time or stays put.

`PEAK_ACC=32`, `THREADS=512`, 82 SMs, PI idle. Sweep the round count over the whole range from 1 to 4096. At least these points (formula integers): 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 410, 422, 435, 448, 461, 474, 486, 499, 512, 525, 538, 550, 563, 576, 589, 602, 614, 640, 768, 896, 1024, 1280, 1536, 2048, 2560, 3072, 4096. The run from 410 to 614 is the 0.8–1.2 range in steps of about 0.025 at base 512.

- **With the schedule switch:** `PEAK_ROUNDS=512` and a schedule of scales whose formula rounds land on those points, 10 s each, the base point at the start and end of each launch and at least every eight points. Split it so no launch passes about 5 minutes. Run the whole sweep twice, once ascending and once descending, so order and elapsed time aren't read as the effect of the work.
- **Without it:** one `fullwidth 15` launch per point with `PEAK_ROUNDS` set to the point, and a base launch at least every eight points. Run each point twice, in a different order the second time.

Report the per-point table and plots of the per-SM pulse rate, the counted ops/s per SM, the 136 ms coefficient and the strongest lag against rounds. Report the base per-SM rate at each bracket. Say plainly:

- whether any round count gives counted ops/s per SM above the base point's by at least the ratio of the two groups, about 601k over 580k;
- whether the base point stepped up after any sweep point;
- whether the strongest lag stays at 136 ms from 1 round to 4096, or moves with the pass time.

## 3. Occupancy sweep, open loop

The same question, on the other workload axes. Base point otherwise. 10–15 s per point, base bracket as in item 2.

1. **Threads per block.** `THREADS` from 32 to 1024 in steps of 32 (32 points), rounds 512, 82 SMs. Threads are fixed per launch, so this is one `fullwidth 15` launch per point, with a base launch every eight. Report each launch's `[CAL] rate_ref` (comparable only at 512 threads), the counted ops/s, the ops/s per SM and the wave.
2. **Independent chains.** `PEAK_ACC` 4, 8, 16 and 32, each at rounds 512, and each at equal work per pass (rounds 4096, 2048, 1024 and 512), threads 512. Twice each.
3. **SM count.** `noise N 15` at N = 2, 4, 6, 8, then every 4 from 12 to 80, then 57 and 82, at the base work. With the schedule switch's count column, you may run the whole staircase inside a few launches, ascending and descending, 10 s per count. Report the per-SM rate and the wave against N. The fourteenth pass's table rates at 61, 41, 20 and 2 SMs are probes inside calibration. These are holds.
4. **A coarse grid.** Threads 64, 128, 256, 512 and 1024 against rounds 32, 128, 512 and 2048, at 82 SMs, once each. It shows whether two axes together reach something one axis alone doesn't.

Report the highest counted ops/s seen in items 2 and 3 and its setting. That's the ceiling input for item 1. Name every point that reaches the 49 million group: a base bracket near 601k per SM, or ops/s per SM above the base point's by the ratio in item 2.

## 4. Slow ramp and fine stairs

The question is whether the plant has a memory: does the per-SM rate on the way up match the way down, and can a load path move the plant between groups?

- **Ramp.** Scale 0.8 to 1.2 to 0.8, linear, in `scale_milli` steps of 1 (rounds 410 to 614 and back), from a base point. Two speeds: 30 s per leg and 60 s per leg. A 60 s ramp is about 140 s with a 10 s base point at each end.
- **Stairs.** 0.8 to 1.2 in steps of 0.025 and back down, 10 s per level. Split it into an up launch and a down launch, each starting and ending on the base point, so neither runs past about 3.5 minutes.
- **Open loop** if you added the schedule switch, at 82 SMs. Then one closed-loop ramp at 60 s per leg through `loadfile`, at the item 5 closed-loop settings, as the comparison. **Without the switch,** run both ramps and the stairs closed loop through `loadfile`, and say that the PI was running.

Report the per-SM rate, the counted ops/s, `z_raw` SD and the 136 ms coefficient in 0.025-wide scale bins, up and down. In closed loop also give `sm_moves` and the in-band share per bin. Say whether up and down match and whether the base point after the ramp is in the same group as before it.

## 5. Closed-loop confirmations

Only at points that reached the 49 million group in items 2–4. If none did, say so, and run one confirmation pair at the setting with the highest base-bracket per-SM rate instead, labelled as such.

Settings: `GPUTRONIC_ICLAMP_SMS=15`, `GPUTRONIC_TARGET_Z=1.5`, the point's workload switches, `loadfile`, a 15 s pre-roll at 1.0, 30 s at 1.2, 30 s at 1.0, 75 s in all. Run them as ABAB pairs, back to back with the same schedule at the base point, in the same session, and the base launch must land in the 47 million group. Per launch report the `[CAL]` group, the per-SM rate in 10 s chunks, `sm_moves`, moves in the whole 30 s hold and its last 10 s, the in-band shares, the limit-cycle check, the 136 ms coefficient and the strongest lag on the count and on Z, and the move-to-response delay from the per-tick file (first Z change and half the change, in ms and in windows, as in the fourteenth pass).

Then give a plain answer: **at a high-group workload point, does the loop stop hunting?** Yes, no, or partly. If a launch changes group mid-hold, say so and don't count that hold.

## 6. What success looks like

These are criteria, not predictions.

- **Full range.** The sweep covers zero (no process), the floor of the counted path, and the measured ceiling, with the ceiling given in counted TFLOPS and its setting.
- **Map.** Every sweep point has a committed row with the per-SM rate, counted ops/s, `z_raw` SD and the 136 ms figures, and the launch `rate_ref` and group.
- **Floor answered.** A plain yes or no: does any swept workload setting move the base point, or the per-SM ops rate, from the 47 million plant to the 49 million plant? If yes, which setting, how many times out of how many tries, and whether it stayed.
- **Wave answered.** Does the 136 ms period track the pass time, the threads, the chains or the SM count, or stay at 136 ms across the range? Plain answer, with the points that show it.
- **Hysteresis answered.** Ramp and stairs up against down, with a yes or no on a group change caused by the load path.
- **Confirmations.** Item 5 run at every high-group point with a same-session low-group baseline, or a plain statement that no point reached the high group.
- **Guardrail.** Clamp 15 doesn't trip on any closed-loop hold, or any trip is named.
- **Driver orthogonality.** Clocks, flags and display state are logged on every launch, and the loop didn't fight the driver.
- **No regressions.** With every switch unset, the `[CTRL]` line shows target 1.00, Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, launch 82, no `[RATE]` line and no line from any new switch. Every new switch and log is off by default, and the three check gates pass at commanded 1.5.

## 7. Report and checks

Report at `docs/analysis/fifteenth-pass-2026-10-18.md`, in the style of the fourteenth-pass report. Include:

- the range the existing knobs covered, and any switch you added to reach further;
- the window used for each run, and the `[CAL]` group of each launch;
- item 1's zero, floor and ceiling rows;
- items 2 and 3 as tables and plots, with the base brackets;
- item 4's up and down bins;
- item 5's table and the yes or no on hunting;
- the clock, flag and display-state findings, including any display change and its timestamp in PDT;
- any switch or log added, with confirmation it's off by default;
- any directive you argued with, and why;
- the item 6 criteria, with a plain yes or no for each.

Checks:

- `make gold`, `make transfer` and `make episode` build and pass `CHECK` with every switch unset. Record them beside the fourteenth pass's 1.415, 1.437 and 1.499.
- Every number in the report matches a committed log in `logs/2026-10-18_fifteenth-pass/`.
- `main` and every earlier results branch are unchanged (`git ls-remote origin`).
