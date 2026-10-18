# Tenth pass: hold Z while a second kernel uses the SMs the loop turned off, then a load-step test

This directive replaces the step-response-only draft of 13 October. It takes the question from Grok Build's note on `notes/tenth-pass-suggestions-2026-10-17` (`docs/analysis/tenth-pass-suggestions-2026-10-17.md`) and keeps the step-response run from the earlier draft as the second job. Start from GitHub `main` at `b611a2b`. Post everything to a new branch, `test-results/2026-10-17_tenth-pass`, with logs under `logs/2026-10-17_tenth-pass/`. Do not merge anything, and do not force-push.

## Start here

Run the pass in this order. Each step can end the pass, and an early stop is a result, not a failed run.

1. Build from `main` at `b611a2b` with CUDA 12.8 and `-arch=sm_120`, every test switch unset. Run `make gold`, `make transfer` and `make episode`, and record the three `CHECK` mean-Z lines. These are the gates this pass has to keep passing.
2. Add the guest-kernel test switch (item 2). Rebuild, and confirm the unset binary prints the same `[CTRL]` line and log columns as `main`.
3. Item 3, phases A, B and C, in that order. This is the main hardware job.
4. Item 4, the load-step run, only if phase A stayed in band. If phase A left the band, skip item 4 and say so.
5. The report (item 5).

## Where things stand

- The ninth pass (`logs/2026-10-13_ninth-pass/closed.txt` on `e818644`) held mean Z 1.502 over the 22 one-second samples from 4 s to 25 s, min 1.439, max 1.561, all 22 inside the 0.08 band, mostly 54 SMs, about 181–182 W at 59–60 °C. The full-pipe loop the night before gave 1.497. Two undisturbed runs agree, so a third one is not part of this pass.
- Both of those runs had no second kernel, and every count change sat inside the deadband. The ninth-pass README names the next step as holding Z under a real competing GPU load. That is the main question here.
- On the compiled default, a block past `active_sms` stays resident and spins on the control words. It still owns the SM, so nothing else can use it. The eighth pass measured `GPUTRONIC_IDLE_EXIT=1` (an idle block returns, and the host launches a replacement when the count rises) only on the empty kernel. Exit on the 32×512 kernel has never been measured.
- The lab folder at `/media/dad/1TB HDD/GPUtronic` is on `b8b66e7`, 18 commits behind `main`. Update it, or build in a fresh clone of `main`, before running anything.

## Standing rules

- Kp 0.55, Ki 0.08, Kd 0, deadband 0.08, scaled integral clamp, 82-block launch, floor 2. Physical SMs 82 and 83 stay unlaunched for the desktop. Do not set `GPUTRONIC_DEADBAND`, `GPUTRONIC_INTEGRAL_CLAMP`, `GPUTRONIC_ICLAMP_SMS`, `GPUTRONIC_RELAUNCH` or `GPUTRONIC_IDLE_QUIET`.
- Any new switch is test only and off by default. Unset, the binary behaves exactly as `main`. Do not promote `GPUTRONIC_IDLE_EXIT` or any other switch to the default in this pass.
- One GPU job at a time. Stop any run at 83 °C; the log up to the stop is the result. The stock driver still owns clocks, power and temperature.
- No number that isn't measured in this pass or already written in the repo. Quote each one from its source file. The guest block count and the step scale below are test settings, not predictions.
- The hold region is a deadband, not hysteresis. Describe the Kalman filter only as the code does it. No motor-control terms for the loop.
- Do not touch `test-results/2026-10-13_ninth-pass`, its logs, or `archive/`. Leave the notes branch as it is.
- No game or second-process co-run. A second process timeslices with Gold instead of using the freed SMs, so that is a different experiment for later.

## 1. Common switches

Every Gold run in this pass uses the ninth-pass full-pipe switches:

```
GPUTRONIC_PEAK_ACC=32 GPUTRONIC_PEAK_ROUNDS=512 GPUTRONIC_THREADS=512 \
GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1
```

Record `nvidia-smi dmon -s pucm -d 1` for every run, plus plain `nvidia-smi` at the start and end of the pass.

## 2. The guest kernel (new test switch)

Add one switch, for example `GPUTRONIC_GUEST_BLOCKS=<n>` with `GPUTRONIC_GUEST_AT_MS=<t>`, read in the Gold process. Unset, nothing launches.

- After the given time, launch a second kernel on its own non-blocking stream, in the same process (`concurrentKernels` must be 1; print it).
- Fixed grid of `n` blocks. It does not read `active_sms` and does not add to `total_work_pulses`. It counts its own finished blocks in its own mapped word with `atomicAdd`, so it cannot move Z.
- Same body as Gold: the 32-chain, 512-round independent FMA, so one guest pulse is the same work as one Gold pulse.
- The one-second line prints the guest rate next to Z and the SM count. Leave the existing `[SPAWN]` print on.
- A guest-only mode (Gold not resident) for phase B, for example `gputronic_gold guest <sec>`.

## 3. The main run: three phases

**Phase A, idle-exit, no guest.** `run 25` with the common switches plus `GPUTRONIC_IDLE_EXIT=1` and `GPUTRONIC_NO_CSV=1`. Report the same window as the ninth pass: mean Z, min, max, how many of the 22 samples from 4 s to 25 s are inside the band, the SM counts, power, temperature, and the `[SPAWN]` lines. If Z leaves the band, stop the pass here: exit changed the plant, and adding a guest on top would mix two causes. Write the log and the report.

**Phase B, guest alone.** Ten seconds, Gold not resident. Use 28 blocks, or the idle count phase A actually settled at (82 minus the settled count), and write down which. Record the guest pulse rate, power and temperature. That rate is the solo figure for phase C.

**Phase C, both.** Gold as in phase A, guest at the phase B block count starting 8 s after launch, so calibration and the first settle are Gold alone. Hold 40 s, or until 83 °C. Report, for the one-second samples after the guest starts: mean Z, min, max, the share inside the band, the SM counts, the guest rate as a fraction of phase B, the `[SPAWN]` lines, power, temperature, and the power and thermal violation flags.

How to read it, without changing anything:

- Z in band and the guest near its solo rate: the loop held the requested rate while the freed SMs did other work. That is the result the landing page asked for.
- Z in band and the guest near zero: the blocks didn't give the SMs up, or the guest couldn't start. Also a finished result.
- Z out of band and the count climbing: the guest took time, power or an SM Gold still needed. Record where the count stops, including whether it sits on the 3.28-SM clamp. Leave the gains and clamp alone.

## 4. Load-step run (only if phase A stayed in band)

This is the step-response test from the 13 October draft. It drives Z clearly outside the band so the PI corrections themselves can be measured, which none of the runs so far have done.

Use the existing `loadfile` mode and this schedule, committed as `logs/2026-10-17_tenth-pass/steps.txt` (t=0 is the end of calibration):

```
60000  1.2
120000 1.0
180000 1.2
240000 1.0
300000 1.2
360000 1.0
```

```
<common switches> GPUTRONIC_TRACE_BIN_US=1000 \
timeout 480 ./build/gputronic_gold loadfile logs/2026-10-17_tenth-pass/steps.txt 420
```

`GPUTRONIC_NO_CSV` must be unset here. Idle-exit stays unset, so this run is on the default plant. Scale 1.2 matches T6 and the eighth pass.

Say in the report that this is a synthetic load. On the 32×512 kernel, `work_scale_milli` skips pulses while the FMAs keep running, so the counter reports a slower job while the SMs do the same work. It tests how the loop answers a drop in counted rate, not a real competitor. The eighth pass saw the same 1.2 load park `z_hat` near 1.63 on the cached plant. The scaled clamp adds about 3.28 SMs, roughly 6% of the settled rate, so a 20% loss may well park Z high here too. That would be a result, not a failed run.

For each of the six steps, report: the time until `z_hat` (1 ms bins) enters the band and stays, or "never"; the peak past the band and its sign; the SM moves during the transient; the count before the step and at the end of the hold; the share of bins in band over the last 30 s of each hold; the share on the integral clamp; and any limit cycle. Keep the tick CSV (decimated if too big, saying how), the 1 ms trace, stdout and dmon in the logs folder. A plot of `z_hat` and `active_sms` against time with the band shaded is welcome, with the script that drew it. Ben sets pass/fail thresholds after he sees these numbers.

## 5. Report and checks

Report at `docs/analysis/tenth-pass-2026-10-17.md`, short: the exact commands and switches, the three `CHECK` lines, a table per phase, the step table if item 4 ran, temperatures and power, the switch added (name, and confirmation it's off by default), and what was skipped and why.

Checks:

- The three standard targets build and pass `CHECK` with every switch unset.
- The unset binary's `[CTRL]` line shows Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, launch 82, the same as `main`.
- Every number in the report matches a committed log in `logs/2026-10-17_tenth-pass/`.
- `main` and `test-results/2026-10-13_ninth-pass` are unchanged (`git ls-remote origin`).

## Left for Ben, not this pass

- Whether to merge the ninth-pass results branch (`e818644`) onto `main`.
- Which idle path becomes the compiled default, after phase A's log.
- A disturbance that skips the FMA block itself, more FMA rounds, a tensor-core or memory-heavy body, the GTX 1080, and a game co-run.
