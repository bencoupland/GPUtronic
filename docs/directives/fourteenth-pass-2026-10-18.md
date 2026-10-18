# Fourteenth pass: watch the hunt cycle by cycle, raise the control rate, and find where the low rate_ref comes from

This follows `docs/directives/thirteenth-pass-2026-10-18.md` and its results on `test-results/2026-10-18_thirteenth-pass` (`0255d1d`, report `docs/analysis/thirteenth-pass-2026-10-18.md`). Start from that results branch at `0255d1d`. The binary is still `add4304`. Post everything to a new branch, `test-results/2026-10-18_fourteenth-pass`, with logs under `logs/2026-10-18_fourteenth-pass/`. Don't force-push, and don't merge anything.

## Start here

Do the three probes in order: item 1, item 2, item 3. Then the success criteria in item 4 and the report and checks in item 5. Follow the window policy below.

Ben asked for these three probes on a hunting run:

1. Log SM count and Z every control cycle, so the phase between a correction and its response is visible.
2. Raise the control rate on a low-`rate_ref` run and see whether the hunting stops.
3. Find out whether the lower `rate_ref` comes from the loop adding latency or from the workload.

**You may argue.** If you disagree with a directive, say so in the report with your reasoning and evidence, before or instead of carrying it out. A reasoned objection counts as a result. Don't quietly work around a directive. The source reading below is Grok Bot's. If the code says otherwise, the code wins, and say so.

## How GPUtronic is meant to work (carried forward)

- **Z = 1.0 is the normal default.** The loop stays out of the way and runs at the fastest rate, the plant's own free-run rate. `GPUTRONIC_GOLD_TARGET_Z` is `1.0f` on `add4304`. Keep it.
- **Z above 1 is an explicit margin mode.** You command it when margin matters more than speed. It carries a throughput penalty. In this pass 1.5 is that mode, set with `GPUTRONIC_TARGET_Z` or in the check gates, never as a default.
- **The clamp is a guardrail.** It should almost never trip. If it trips on real workloads, the fix is upstream.
- **The stock driver owns thermal and power. GPUtronic owns scheduling.** Keep logging clocks and flags beside Z, read only.
- **Closed loop is the differentiator.** Stay standalone and loosely coupled, and add no knob a result doesn't need.

## Where things stand after the thirteenth pass

- **Two kinds of run at the same settings.** Three clamp-15 runs at Z = 1.5. Runs a and b hunted: `rate_ref` 47,114,836 and 47,127,304, `sm_moves` 6560 and 6472, short exits at some hold ends. Run c was quiet: `rate_ref` 49,314,120, `sm_moves` 2388, returns 99.1%, 99.6% and 99.0% of the last 30 s in band. The 9 SM run matched c (`rate_ref` 49,311,496, `sm_moves` 2642). Clocks, memory clock and reason words were the same in both groups. You called the tail miss run-to-run variance and left the `rate_ref` split unexplained.
- **The compiled clamp curve is in.** 1.15 holds without a trip over a 60 s hold, and 1.175 trips inside 15 s. No new width was proposed. That stays Ben's call.
- **Grok Bot's read of your committed traces** (derived from the `-1ms.csv` files, not new measurements; check them):
  - The low group is slower for the whole run, not just at calibration. That is ~576k pulses/s per SM on scale-1.0 returns against ~601k. The calibration table is low by the same ~4–6% at 61, 41, 20 and 2 SMs. `z1-unset`, with zero SM moves, is in the low group (`rate_ref=47040876`).
  - The low group carries a 136–137 ms line in `z_raw`, about 34–44% of the 0.5–60 Hz power. It shows on `z1-unset` at a fixed 82 SMs too. With the count held ≥ 500 ms, `z_raw` SD is ~0.04 in the low group and ~0.011 in the high group, against the 0.08 deadband. In the loaded holds of a and b the SM count follows the same 136–137 ms period.
  - Correction-to-response lag is the same in both groups. Z moves in the next 8 ms window and is half-way at ~23 ms after a count increase, in a, b, c and c9.
  - Run b left the low state at 254.3 s (14:43:33 PDT): per-SM rate went from ~578k to ~602k, `z_raw` SD per 250 ms fell to ~0.005, and the count settled. Every later run in the session was in the high group (18 sweeps, `rate_ref` 49.13–49.60 million). The 1 Hz flags show nothing at that second. `nvidia-smi-start.txt` shows `Disp.A On`, and `nvidia-smi-end.txt` shows `Off`. When it changed isn't logged.
- **Green contexts and guest-first launch weren't run.** They're still Ben's call. Don't run them this pass.

## What the source says (read it yourself and correct this if it's wrong)

- **Per-cycle logging already exists.** `gputronic_start` opens `results/gputronic_gold_load.csv` unless `GPUTRONIC_NO_CSV=1`. `control_thread` writes one row per control tick (`time_s,z_raw,z_hat,dzdt,rate_ema,active_sms,error,integral,ctrl_hz,rate_ref,…`), from thread start, so calibration is included. This is the "10 µs file" the thirteenth pass deleted. `GPUTRONIC_NO_CSV=1` also turns off the 1 ms trace in `gold_load_run`, so don't use it to save time.
- **The tick is not the correction rate.** `control_dt_us` is `GPUTRONIC_DEFAULT_CONTROL_DT_US` (10 µs) in `include/gputronic.h`, with no environment switch on the `loadfile` path. The loop prints 100 kHz and measures about 15,500–17,500 ticks per second. Z only changes when the rate window closes: `rate_win_dt >= 0.008` (8 ms, compiled), then `rate_ema = 0.70 × rate_ema + 0.30 × inst_rate`. In the thirteenth traces, 98–99% of SM moves land within ~1 ms of a new `z_raw`. Pulses arrive in 1 ms block batches (`GOLD_PULSE_LOCAL_NS`, with `GPUTRONIC_PULSE_LOCAL=1`). Blocks see a new count up to `GPUTRONIC_CTRL_CACHE_US` late (default 1000, floor 50, with `GPUTRONIC_CTRL_CACHE=1`).
- **`rate_ref` is measured open loop, before the PI runs.** In `control_thread`: warm-up 0.7 s, then `rate_ref = Δpulses / Δt` over the window to 1.8 s ("sustained 1.10s window"), at `n_cmd = N` (82), with the integral held at 0 and `z_raw` held at 1.0. Then the 300 ms table probes at 61, 41, 20 and 2 SMs. It isn't recomputed in the run. During calibration the loop is present only as the control thread itself, the 1 ms pulse batches and the 1 ms control cache. The PI correction is not there yet.
- **`fullwidth [sec]` and `noise [sms] [sec]`** calibrate the same way, then run open loop (all SMs, or a fixed count) with no controller. They write the same 1 ms trace.
- **`GPUTRONIC_WORK_COUNTER=1`** (T9) prints `[WORK]` pulses against completed batches once a second in `loadfile`. **`GPUTRONIC_SM_ACCUM=1`** (T11) writes per-SM ops per bin to `results/gputronic_gold_smaccum.csv`. Both are test only, off by default, and add memory traffic.

## Window policy

- **Hunting holds (items 1 and 2) are clamp-style tests.** The 45 s holds you argued for in the thirteenth pass are accepted. Report the last 30 s and the last 10 s. A 20 s pre-roll and one load-and-return pair (45 s at 1.2, 45 s at 1.0, 110 s) is enough per setting. Use the full three-pair schedule (`steps-item1.txt`, 290 s) only where a result needs it, and say why.
- **Step-sweep and calibration launches (item 3):** where the behaviour shows within about 10 s, use holds of about 10–15 s. Spend the saved time on repeats.
- Avoid 5–20 min runs. No single run longer than about 5 minutes unless a result needs it. If you lengthen one, say why.
- Keep the eleventh-pass definitions: the band is `|z_hat - target| <= 0.08`, "enters and stays" is the time after the last out-of-band bin, and a limit cycle is exactly two SM counts in the window, each at least five times, with at least six changes. Say which window each share uses.
- **Check the group on every launch.** Read the `[CAL] rate_ref` line. In the thirteenth pass the low group was 47.0–47.2 million and the high group 49.1–49.6 million. Report the per-SM rate for each hold, so a mid-run change like run b's at 254.3 s isn't read as the effect of a setting.

## Standing rules

- Kp 0.55, Ki 0.08, Kd 0, deadband 0.08, 82-block launch, floor 2. Physical SMs 82 and 83 stay unlaunched. The gains are locked. None of these change.
- The compiled target stays at 1.0, and the check gates keep commanding 1.5.
- Ben hasn't picked a compiled clamp width. Don't change the compiled clamp or adopt 9 or 15 as a default. Use `GPUTRONIC_ICLAMP_SMS` to test. Clamp 15 is the hunting configuration here.
- `libsmctrl` stays out.
- **The controller's view has to be complete.** Every batch that runs is counted.
- **Accepted limit:** idle SMs spin at the GPU clock on the stock driver.
- **Out of scope:** custom drivers, per-SM sleep, per-SM clock or power gating, thermal or power logic in the controller, and anything below the CUDA runtime and driver API.
- Any new switch is test only and off by default. A logging-only addition is gated the same way and defaults off. With every switch unset, the binary behaves exactly like `add4304`. Report every switch or log you add, with its name, what it does and confirmation that it's off by default.
- One GPU job at a time. Stop any run at 83 °C; the log up to the stop is the result.
- No number that isn't measured in this pass or already in the repo, quoted from its file. An early stop or a ruled-out idea is a result.
- The hold region is a deadband, not hysteresis. The Kalman filter is logged only. No motor-control terms for the loop. Source note: on `add4304` the PI error is computed from the Kalman `z_hat` (`enable_kalman = 1`), and only `dZ/dt` goes unused, because Kd is 0. Don't change that this pass. If you read it differently, say so.
- Leave every earlier results branch, the notes branch, `archive/` and all earlier directives as they are.

## 1. Log SM count and Z every control cycle on a hunting run

The question is the phase between a correction and its response, seen cycle by cycle.

Run clamp 15, `GPUTRONIC_TARGET_Z=1.5`, `GPUTRONIC_WORK_LOAD=1`, the full-pipe switches (`GPUTRONIC_PEAK_ACC=32`, `GPUTRONIC_PEAK_ROUNDS=512`, `GPUTRONIC_THREADS=512`, `GPUTRONIC_PULSE_LOCAL=1`, `GPUTRONIC_CTRL_CACHE=1`, `GPUTRONIC_TRACE_BIN_US=1000`), idle-exit and guest unset, and clocks and flags logged. Use the hunting-hold window.

- **Keep the per-tick file.** Don't delete `results/gputronic_gold_load.csv` after the hold. Commit it compressed beside the 1 ms trace. If it is too large to commit, commit the rows where `z_raw` changes or `active_sms` changes, plus the whole first 3 s (calibration), and say what was cut. No new logging is needed. If you find a field you need is missing, you may add a logging-only change, gated and off by default, and report it.
- **Get a low-group run.** If the launch lands in the high group, keep it as the quiet comparison and launch again. If no launch lands low within a reasonable number of tries, stop and report the `rate_ref` of every try. That is a result, and item 3 then matters more.
- Beside the flags, log the display state once a second, read only, e.g. `nvidia-smi --query-gpu=timestamp,display_active,display_mode,pstate,utilization.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active --format=csv -lms 1000`. Take the `nvidia-smi` process list at the start and end of every launch.

Report, from the per-tick file:

- a cycle-by-cycle excerpt of 1–2 s in a hunting stretch (tick time, `z_raw`, `z_hat`, error, integral, `active_sms`), with the window closes marked;
- the delay from each SM move to the first Z change and to half its change, in ms and in windows;
- the period of the count oscillation and of Z at fixed counts, including whether the 136–137 ms line is there;
- moves per hold, moves that reverse within one window, and the in-band shares;
- the same numbers for one high-group run if you have one.

## 2. Raise the control rate on a low-rate_ref run

The question is whether the hunting stops when the loop corrects faster.

From the source, faster PI ticks probably won't change anything, because Z only refreshes when the 8 ms window closes. Test what can actually be made faster, on a low-group launch, back to back with an unchanged baseline in the same group:

- **Baseline:** the item 1 settings.
- **Faster actuation, existing switch:** `GPUTRONIC_CTRL_CACHE_US=250`, then `100`.
- **Faster Z refresh:** the window has no switch. You may add a test-only switch for the rate-window length, off by default (unset keeps 8 ms and behaves exactly like `add4304`), name it in the report, and run something like 4 ms and 2 ms. With `GPUTRONIC_PULSE_LOCAL=1`, a shorter window holds fewer 1 ms batches per block, so the Z noise may rise. Report the `z_raw` SD at fixed counts beside the result.
- **Faster ticks:** optional. Only if you think it matters, and not by `GPUTRONIC_NO_CSV=1`, which drops the 1 ms trace. A test-only tick switch, off by default, is allowed.

Per setting, report the `[CAL]` group, the per-SM rate, `sm_moves`, moves in the last 30 s and last 10 s, the in-band shares, the limit-cycle check, and the 136–137 ms share of Z. Then give a plain answer: **did raising the control rate kill the hunting?** Yes, no, or partly, and at which setting. If the run changes group mid-hold, say so and don't count that hold.

## 3. Loop latency or workload: where the low rate_ref comes from

`rate_ref` is measured open loop, before the PI runs (see the source section). These tests are meant to separate the two causes.

1. **Repeat calibration with the loop never closed.** At least eight `fullwidth 15` launches with the full-pipe switches and `GPUTRONIC_WORK_LOAD=1`, interleaved with short closed-loop launches (item 1 settings, one pair). For each launch report `rate_ref`, the four table rates, the open-loop per-SM rate over the run and the 136–137 ms share. If the 47/49 split appears in `fullwidth` launches, where the PI never runs, the loop isn't causing it. If the split appears only when the loop closes, it is the loop. Use `noise 57 15` as well if a fixed count near `n_ff` is easier to compare.
2. **Calibration window detail.** For every launch, from the per-tick file's first ~3 s, give `rate_ema` through the warm-up (0–0.7 s), the sustained window (0.7–1.8 s) and each table probe. Say whether the low group is low from the first window or drops during calibration. If you need the window's raw pulse counts and those aren't in the file, you may add a logging-only print, gated and off by default, and report it.
3. **What is slow in the low state.** On one low-group and one high-group launch, if you get both, run `GPUTRONIC_WORK_COUNTER=1`. Fewer completed batches means slower work. The same batches with fewer pulses counted means the counting path. Separately, run `GPUTRONIC_SM_ACCUM=1`: is every SM slower, or a subset? Say that these switches add traffic and were on.
4. **The pulse path.** In `fullwidth`, compare `GPUTRONIC_PULSE_LOCAL=1` with it unset (one mapped atomic per pass). Report whether the 136–137 ms line and the per-SM gap stay.
5. **The environment.** Use the display-state and process logs from item 1 on every launch. If a change of group lines up with a display or compositor change, report the timestamps. Changing the desktop setup (display blanking, closing applications) is Ben's call. Don't do it unless he sets it up.

Then say which cause the `rate_ref` gap points to:

- **the loop adding latency**;
- **the workload or plant state**, naming the part if the counters show it;
- or **not separated by these logs**.

## 4. What success looks like

These are criteria, not predictions.

- **Cycle by cycle.** Item 1 commits the per-tick file, or a stated cut of it, for at least one low-group hunting run. It reports the move-to-response delay and the oscillation period from it, or says plainly that no launch landed low.
- **Control rate answered.** Item 2 says whether raising the control rate killed the hunting, at each setting tried, on low-group launches with a same-group baseline.
- **Cause named.** Item 3 says whether the `rate_ref` gap points to loop latency or to the workload, with the `fullwidth` repeats that show it, or plainly says the logs don't separate them.
- **Guardrail.** Clamp 15 doesn't trip on any hold, or any trip is named.
- **Driver orthogonality.** Clocks, flags and display state are logged on every run, and the loop didn't fight the driver.
- **No regressions.** With every switch unset, the `[CTRL]` line shows target 1.00, Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, launch 82. Every new switch and log is off by default, and the three check gates pass at commanded 1.5.

## 5. Report and checks

Report at `docs/analysis/fourteenth-pass-2026-10-18.md`, in the style of the thirteenth-pass report. Include:

- the window used for each run, and the `[CAL]` group of each launch;
- item 1's cycle-by-cycle excerpt, delays and periods, or a summary of them if the excerpt doesn't fit;
- item 2's table per setting, and the yes or no on hunting;
- item 3's launch table and the verdict on the cause;
- the clock, flag and display-state findings;
- any switch or log added, with confirmation it's off by default;
- any directive you argued with, and why;
- the item 4 criteria, with a plain yes or no for each.

Checks:

- `make gold`, `make transfer` and `make episode` build and pass `CHECK` with every switch unset. Record them beside the thirteenth pass's 1.443, 1.407 and 1.457.
- Every number in the report matches a committed log in `logs/2026-10-18_fourteenth-pass/`.
- `main` and every earlier results branch are unchanged (`git ls-remote origin`).
