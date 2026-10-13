# Tenth pass: a step-response test that drives Z outside the 0.08 band

This pass is one GPU job, a step-response run on the RTX 5080, plus a short report. Its purpose is to push Z clearly outside the band, so the PI corrections can be measured. Start from `main` at `b611a2b`. Post the results to a new branch, `test-results/2026-10-13_tenth-pass`, with the logs under `logs/2026-10-13_tenth-pass/`. Do not merge anything, and do not force-push.

## Start here

Run the pass in this order:

1. Build from `main` at `b611a2b` with every switch unset, and run the checks that need no switches (see Checks).
2. Then item 1, the load-step run. This is the long hardware job, with the logging in item 2 turned on for this run only.
3. Then item 1's optional target-Z variant, only if the small test switch it needs has been added and builds cleanly. If it hasn't, skip it and say so in the report.
4. Then write the report (items 3 and 4).

## Where things stand

- The ninth pass ran the full-pipe workload with the controller engaged (`logs/2026-10-13_ninth-pass/closed.txt` on `test-results/2026-10-13_ninth-pass`). The switches were `GPUTRONIC_PEAK_ACC=32 GPUTRONIC_PEAK_ROUNDS=512 GPUTRONIC_THREADS=512 GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_NO_CSV=1`, and the command was `./build/gputronic_gold run 25`. Target Z 1.50, Kp 0.55, Ki 0.08, Kd 0, deadband 0.08, an 82-SM launch, floor 2, loop about 17 kHz.
- Its 22 one-second samples from t=4 to 25 s had mean 1.502, sd 0.036, min 1.439 and max 1.561. All 22 were inside the band. The STATS line gave pi_ticks 373822, deadband_held 366683 (98.1%), integral_updates 7139, integral_on_rail 0, sm_moves 74, mean_Z 1.4964, active_sms 47..64. Calibration gave n_ff 55.9, dZ/dn 0.0323, and an integral clamp of 1.324 (3.28 SMs at most).
- From t=13 to 17 s, Z drifted up to 1.561 at a constant 54 SMs, with no visible correction. That is expected: 1.561 is still inside the deadband, so the count is held.
- The ninth-pass README claim "all 22 inside the band" is true, but it says less than it sounds. The band is the same width as the deadband, 0.08. Inside it the PI loop holds the count and freezes the integral. So the ninth pass only exercised sub-second corrections (74 SM moves, 7,139 integral updates), and never a sustained error outside the band. A step test is what shows the corrective behaviour.
- The fast CSV was off in the ninth pass (`GPUTRONIC_NO_CSV=1`), so the PI terms were not logged.
- An earlier full-pipe run (`docs/analysis/full-pipe-2026-10-13.md`) gave a 1-s mean of 1.497, sd 0.027, 45 sm_moves and 3479 integral updates.
- The README intro quotes 1.497 (the earlier full-pipe run) while its Results section quotes 1.502 (the ninth pass). That mismatch is a separate ninth-pass cleanup. It is not this pass's job, unless it is a one-line fix done on its own commit.
- A load step has been measured before, but on the cached plant, not the full pipe. In the eighth pass (`docs/analysis/eighth-pass-2026-10-13.md`, section 1) a 1.2 load from 20 to 45 s left `z_hat` held near 1.63 with the scaled clamp, and wider test clamps of 6, 9 and 12 SMs did not bring it back into the band either. After release Z settled near 1.41. That is the history this pass tests against on the full-pipe plant. Don't assume the full pipe behaves the same, and don't assume it behaves differently.

## Standing rules

- Do not modify the ninth-pass results, its logs, or its branch, `test-results/2026-10-13_ninth-pass`.
- Kp 0.55, Ki 0.08, Kd 0, the 0.08 deadband, the 82-SM ceiling, the floor of 2 and the scaled integral clamp all stay as they are. The main run sets none of `GPUTRONIC_DEADBAND`, `GPUTRONIC_INTEGRAL_CLAMP` or `GPUTRONIC_ICLAMP_SMS`.
- Any new switch is test only, and off by default. Unset, the binary must behave exactly as it does on `main`.
- Do not add any number that isn't measured in this run or already written in the repo. Quote each one from its source file. The step sizes below are test settings, worked out from the calibration slope. They are not results.
- Describe the Kalman filter as the code does it, no more. In `src/gputronic_gold.cu`, around lines 1285 to 1297, when the filter is on (the ninth-pass log prints `kalman=1`) the controller's error is the target minus `z_hat`, and when it is off `z_hat` is set equal to `z_raw`. Quote what the code does, and don't claim anything beyond it.
- The hold region is a deadband: inside `|error| < 0.08` the count is held and the integral frozen. Don't call it hysteresis, and don't describe the loop in Park/Clarke or any other motor-control terms.
- Keep the plain-English style the README and these directives already use.
- Leave `archive/`, `logs/` and `results/` on `main` alone.

## 1. Generate the step

**Mechanism.** Use the synthetic load that already exists. It needs no new code. `./build/gputronic_gold loadfile <file> [sec]` runs the closed loop at the Gold target, waits for calibration, and then follows a schedule file. The file has one `t_ms scale` pair per line, with `t_ms` counted from the end of calibration (`gold_load_read` and `gold_load_run` in `src/gputronic_gold.cu`, around lines 2299 to 2495; usage line around 2576). While the scale is above 1.0 the host sets `GPUTRONIC_FLAG_LOAD`, and each warp pulses only once per `scale` batches (`gold_load_pulse` and `gold_load_from_scale`, around lines 212 to 238). The pulse rate falls to 1/scale at the same SM count, and since Z = rate_ref / rate, Z rises by the factor `scale`. The same pulse gate runs in the full-pipe kernel, and with `GPUTRONIC_CTRL_CACHE=1` the block leader picks up `work_scale_milli` once per 1000 µs. Scales below 1.0 are not supported (`docs/analysis/TEST_SUITE.md`, T6). So the down step is taking the load off after the loop has settled under it.

This is a synthetic load: it thins the pulse count, it isn't a second program competing for the SMs. Say so in the report. A real competing process is a later pass.

**Magnitude.** Use scale 1.2. That is the same scale as T6 and the eighth pass, so the results can be compared. At a fixed count, it takes Z from 1.50 to about 1.80, an up step of about 0.30. At the ninth-pass slope of 0.0323 per SM, that's roughly 9 SMs of correction needed. Taking the load off at a fixed count takes Z down by a sixth, from 1.50 to about 1.25, a down step of about 0.25 or roughly 8 SMs. Both are well outside the 0.08 band. These are settings worked out from the slope, not predictions of what the loop will do.

**Timing.** Settle for at least 60 s, then step up, hold for 60 s, step down, and hold for 60 s. Do that three times, so there are at least three steps in each direction. Use this schedule file, committed as `logs/2026-10-13_tenth-pass/steps.txt`:

```
# t_ms scale. t=0 is the end of calibration.
60000  1.2
120000 1.0
180000 1.2
240000 1.0
300000 1.2
360000 1.0
```

Run it for 420 s, so the last hold is a full 60 s:

```
GPUTRONIC_PEAK_ACC=32 GPUTRONIC_PEAK_ROUNDS=512 GPUTRONIC_THREADS=512 \
GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_TRACE_BIN_US=1000 \
timeout 480 ./build/gputronic_gold loadfile logs/2026-10-13_tenth-pass/steps.txt 420
```

`GPUTRONIC_NO_CSV` must be unset. It turns off both the per-tick CSV and the binned trace in this mode (around lines 1653 and 2402).

**Thermal steady state.** The run is about seven minutes, so the card should warm up during it. Record the temperature and power the whole time (item 2). If the temperature is still climbing in the last 60 s, say so, and treat the later steps as taken on a card that was still warming. Don't call the run thermally steady unless the log shows it.

**Target-Z step (optional, needs a switch).** There is no mid-run target change today. `gputronic_set_target` exists (around line 1938), but only `run [sec] [Z]` sets the target, and only at the start. If you do this variant, add one test switch, off by default: `GPUTRONIC_TARGET_STEP_AT_MS` and `GPUTRONIC_TARGET_STEP_Z`, read in `gold_load_run` next to the existing `GPUTRONIC_STEP_AT_MS` block. It calls `gputronic_set_target` once at that time, prints a `[TSTEP]` line, and changes nothing else. Note in the report that `n_ff` and the slope are still the ones calibrated at 1.50. A step to 1.80 at 60 s and back to 1.50 at 120 s, in a separate 180 s run with no load schedule (`loadfile /dev/null 180`), is enough. If the switch isn't added, skip this variant and say so.

Do not use `GPUTRONIC_STEP_AT_MS` / `GPUTRONIC_STEP_SMS` for this pass. They set an open-loop SM count, which takes the controller out of the loop.

## 2. What to log

For this run only, turn the fast logs on:

- **Per-tick CSV**, `results/gputronic_gold_load.csv`, about 10 µs per tick: `time_s, z_raw, z_hat, dzdt, rate_ema, active_sms, error, integral, ctrl_hz, ...`. This already exists. Copy it to `logs/2026-10-13_tenth-pass/ticks.csv`. If it's too big to commit, commit a decimated copy (for example every 10th row, with all rows kept for 2 s after each step) and say how it was decimated.
- **Binned trace**, `results/gputronic_gold_loadfile_10ms.csv` with 1 ms bins: `wall_s, t_ms, scale, z_raw, z_hat, active_sms, integral, clamp, ..., p_term, ..., pi_ticks, deadband_held, integral_updates, rail_ticks, sm_moves, ...`. The `scale` column is the step marker. Copy it to `logs/2026-10-13_tenth-pass/trace_1ms.csv`.
- The 1-s TEL status lines, the `[LOAD]` lines, the `[CAL]` lines and both `[STATS]` lines. Keep all of stdout as `logs/2026-10-13_tenth-pass/steps.txt.out` (or `run.txt`).
- `nvidia-smi dmon -s pucm -d 1` for the whole run, as `logs/2026-10-13_tenth-pass/steps.dmon`, plus plain `nvidia-smi` at the start and end.

Some terms are not logged as their own column today: the I term (Ki × integral), the controller output u, and whether each tick was inside the deadband. `p_term` in the trace is the last value computed outside the deadband, so it goes stale while the count is held. If these can be worked out from the logged columns (error = 1.5 − z_hat with the filter on; I = 0.08 × integral; u = p_term + I outside the deadband), do that in the analysis script and say so. If they can't be worked out cleanly, add them to the binned trace behind one test switch, off by default, for example `GPUTRONIC_TRACE_PI=1` adding `error, i_term, u, in_deadband` columns. Don't change what's logged when it's unset.

## 3. What to measure and report

These are reporting criteria. They are not targets with expected values. For each of the six steps, report:

- the time from the step until Z (`z_hat`, 1 ms bins) enters the 0.08 band and stays there, and the time until it is within ±0.04, or "never" for either;
- the peak overshoot or undershoot past the band, and its sign;
- the number of SM moves during the transient, and the SM count before the step and at the end of the hold;
- whether Z comes back and stays in the band for the rest of the hold, and the share of 1 ms bins in band over the last 30 s of each hold;
- the share of bins with the integral on the clamp (`rail_ticks`), per hold;
- any oscillation or limit cycle, meaning a repeating SM move pattern or a Z swing that doesn't die out. Give its period and size if there is one.

Also report the whole-run STATS lines, and the temperature, power and clock from dmon at the start and end and in each hold.

Ben sets the pass/fail thresholds after he sees these first numbers. A minimal pass is: Z comes back into the band after every step and stays there for the rest of each hold, with no limit cycle and no integral saturation. If any step doesn't meet that, say which one and how, plainly. Given the eighth-pass history, that is a possible outcome, and it is a result, not a failed run.

## 4. Where the results go

- Branch: `test-results/2026-10-13_tenth-pass`, from `main` at `b611a2b`.
- Logs: `logs/2026-10-13_tenth-pass/`, with the schedule file, stdout, the tick CSV (or its decimated copy), the 1 ms trace, dmon, and plain `nvidia-smi` at start and end.
- Report: `docs/analysis/tenth-pass-2026-10-13.md`.
- If it's feasible, a plot of `z_hat` and `active_sms` against time, with the band shaded and a vertical line at each step, committed as `docs/analysis/tenth-pass-2026-10-13-steps.png`, along with the script that drew it. A zoomed panel for the first 5 s after one up step and one down step is useful.

## Checks

- Build the three standard targets with every switch unset, and confirm they still compile and the `CHECK` gates pass.
- If you added a switch, confirm the unset run prints the same `[CTRL]` line and log columns as `main`.
- Confirm the `[CTRL]` line in the run shows Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, and the launch is 82 blocks.
- Confirm the `scale` column in the trace changes at the six scheduled times.
- Check that every number in the report matches a committed log in `logs/2026-10-13_tenth-pass/`.
- Confirm `test-results/2026-10-13_ninth-pass` and `main` are unchanged (`git ls-remote origin`).

## The report

Put it at `docs/analysis/tenth-pass-2026-10-13.md`. It should be short: the exact command and switches, the schedule, a table with one row per step for the item 3 measurements, the whole-run STATS lines, temperatures and power, the plot if it was made, whether the target-Z variant was run, any switch added (its name, and confirmation it's off by default), and the result of the checks.
