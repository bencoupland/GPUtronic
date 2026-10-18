# Twelfth pass: a load that really changes the work, not the count

This follows `docs/directives/eleventh-pass-2026-10-18.md` and its results on `test-results/2026-10-18_eleventh-pass` (`1647954`, report `docs/analysis/eleventh-pass-2026-10-18.md`). Start from that results branch at `1647954`. Post everything to a new branch, `test-results/2026-10-18_twelfth-pass`, with logs under `logs/2026-10-18_twelfth-pass/`. Do not force-push, and don't merge anything.

## Start here

The full budget is available. Do item 1 first: it's the main test of this pass. Then items 2 and 3, the success criteria in item 4, and the report and checks in item 5. The secondary notes in item 6 are optional and come last.

**You may argue.** As in the eleventh pass, if you disagree with a directive, say so in the report with your reasoning and evidence, before or instead of carrying it out. A reasoned objection counts as a result. Don't quietly work around a directive.

## How GPUtronic is meant to work (carried forward)

- **Z = 1 is the home state.** By default the system targets Z = 1 and runs as fast as free run. The eleventh pass showed it: on all 82 SMs, 22 of 22 seconds in band, and the integral never moved.
- **Commanded Z is the opt-in deterministic mode.** You command a slower rate in exchange for bounded latency, for hard real-time work. It isn't the floor. If throughput needs limiting, commanded Z does it, not the clamp.
- **The clamp is a guardrail.** It should almost never trip. If it trips on real workloads, the fix is upstream.
- **The stock driver owns thermal and power. GPUtronic owns scheduling.** The eleventh pass found no throttle flags and clocks inside their boost range. Keep logging the clock beside Z, read only. Add no thermal or power logic.
- **Closed loop is the differentiator.** Keep the design standalone and loosely coupled, and add no knob a result doesn't need.

## Why this pass

Every load step so far uses pulse-skipping. At `work_scale_milli` = S > 1000, a warp still runs every batch but publishes a pulse only once per S/1000 batches (`gold_load_pulse` in `src/gputronic_gold.cu`). The FMAs keep running. Only the count changes. The eleventh-pass report traced the 3.28-SM rail trip to exactly this, and listed a disturbance that changes the real work as still open. The code comment explains why a simple FMA multiplier wasn't used: one pass is about 288 µs of mapped atomics and about 1.6 µs of arithmetic, so 20% more arithmetic barely moves the rate. Any real-work load has to change something that actually dominates the pass time, and that has to be shown, not assumed.

## Standing rules

- Kp 0.55, Ki 0.08, Kd 0, deadband 0.08, 82-block launch, floor 2. Physical SMs 82 and 83 stay unlaunched. None of these change.
- **Ben hasn't decided yet** whether Z = 1 becomes the compiled default (`GPUTRONIC_GOLD_TARGET_Z`), which clamp width is compiled in, whether `libsmctrl` is allowed, or whether to accept the "one grid, one total" ruling. Don't change any of them. Command the target with `GPUTRONIC_TARGET_Z` and the clamp with `GPUTRONIC_ICLAMP_SMS`.
- `libsmctrl` stays out.
- **The controller's view has to be complete.** Every batch that runs is counted. A real-work load must not reintroduce uncounted work.
- **Accepted limit:** idle SMs spin at the GPU clock on the stock driver.
- **Out of scope:** custom drivers, per-SM sleep, per-SM clock or power gating, thermal or power logic in the controller, and anything below the CUDA runtime and driver API.
- Any new switch is test only and off by default. With every switch unset, the binary behaves exactly like `1647954`.
- One GPU job at a time. Stop any run at 83 °C; the log up to the stop is the result.
- No number that isn't measured in this pass or already in the repo, quoted from its file. An early stop or a ruled-out idea is a result.
- The hold region is a deadband, not hysteresis. The Kalman filter is logged only. No motor-control terms for the loop.
- Leave every earlier results branch, the notes branch, `archive/` and all earlier directives as they are.

## 1. The main test: a load that changes the real work

**Build the load.** Add a test-only real-work load, off by default, driven by the same schedule format as `loadfile` (lines of `t_ms scale`). Under it, every batch that runs publishes its pulse. What changes is the real work per batch:

- **Heavier (scale above 1.0).** Each batch does more real work, sized so the pulse rate drops by roughly the scale. Since the pass is atomic-dominated, extra FMAs alone probably won't be enough. Pick work that really costs time: more batch iterations, real memory traffic, or more of the operation that dominates the pass. Explain the choice.
- **Lighter (scale below 1.0).** Each batch does less real work, for example skipping part of the arithmetic or memory work. Then the rate rises and Z falls below target, which tests the direction where the loop has to give SMs back.

**Prove the work really changed.** Using the T11 op counters (`s_ops`: CUDA ops and tensor ops) or an equivalent independent measure, show that ops per pulse changed by about the scale. The pulse count must not have been thinned. If you can't make a real-work load that moves the rate by at least the 1.2 step's share, say so, show what you tried, and stop item 1 there. That's a result.

**Runs.** Same full-pipe switches as the eleventh pass, `GPUTRONIC_TRACE_BIN_US=1000`, idle-exit and guest unset, the clock and flags logged as in eleventh-pass item 3.

1. **Commanded Z = 1.5, heavier load.** The continuation's `steps.txt` schedule, 420 s, with `GPUTRONIC_ICLAMP_SMS=15`, the candidate default (its returns were 90–92% in band in the eleventh pass).
2. The same schedule and load at `GPUTRONIC_ICLAMP_SMS=9`, and once at the compiled clamp, for comparison with the pulse-skip tables.
3. **Commanded Z = 1.5, lighter load.** A schedule that steps to a lighter scale and back, at the clamp 15 setting.
4. **Z = 1, heavier load** (`GPUTRONIC_TARGET_Z=1`), to repeat the eleventh pass's Z = 1 load step with real work. At 82 SMs the loop can't add SMs. Report what Z does and whether the integral stays put.

Report each run with the eleventh-pass step table, and put the pulse-skip run at the same clamp beside it.

## 2. Compare real work against pulse-skip

For each matching pair, say whether the loop behaves the same or differently, and why, in terms of the calibration (`n_ff`, `sms_per_u`, `dZ/dn`), the integral and the count. In particular: does the rail trip at 3.28 SMs still happen under real work? Is 15 SMs still the setting with the best returns? Does real work change the down-step undershoot seen at 9 and 12?

## 3. The guardrail under real work

Report the clamp trip rate for every hold in item 1. If the clamp trips under real work at 15 SMs, look upstream for the cause, as in eleventh-pass item 2, rather than proposing a wider clamp.

## 4. What success looks like

These are criteria, not predictions. Use the eleventh-pass definitions: the band is `|z_hat - target| <= 0.08`, "enters and stays" is the time after the last out-of-band bin, and the last 30 s is anchored on the hold's last sample.

- **The load is real.** Ops per pulse change with the scale, and the pulse count wasn't thinned.
- **Settling time.** For every step, report the time from the step to "enters and stays". Success is that every hold at clamp 15 enters and stays in band before it ends. A hold that ends outside the band is a failure, reported as one.
- **Overshoot.** For every step, report the highest and lowest `z_hat` against the target, and compare them with the pulse-skip run at the same clamp. Don't set a threshold for Ben.
- **Returns to and stays in band.** For every return to scale 1.0, report the last-30-s in-band share, the count changes in the last 30 s, and whether it meets the limit-cycle definition. Success is that every return enters and stays in band, with the share reported beside the pulse-skip 90–92%.
- **Guardrail.** The trip rate is reported for every hold, and any trip at 15 SMs has a named cause or is plainly marked open.
- **Driver orthogonality.** The clock and flags are logged on every run, and you say whether the loop fought the driver.
- **No regressions.** With every switch unset, the `[CTRL]` line shows Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, launch 82. Every new switch is off by default, and all three checks pass.

## 5. Report and checks

Report at `docs/analysis/twelfth-pass-2026-10-18.md`, in the style of the eleventh-pass report. Include:

- how the real-work load works and the evidence the work changed;
- each run's step table beside its pulse-skip twin;
- settling time, overshoot and return results for every step;
- trip rates;
- the clock and flag findings;
- any switch added, with confirmation it's off by default;
- any directive you argued with, and why;
- the item 4 criteria, with a plain yes or no for each.

Checks:

- `make gold`, `make transfer` and `make episode` build and pass `CHECK` with every switch unset. Record them beside the eleventh pass's 1.430, 1.457 and 1.500.
- Every number in the report matches a committed log in `logs/2026-10-18_twelfth-pass/`.
- `main` and every earlier results branch are unchanged (`git ls-remote origin`).

## 6. Secondary notes (optional, after everything above)

The eleventh pass argued against the second-kernel guest angles and never ran green contexts or launching the guest first. Ben hasn't ruled on that yet. If time remains after item 5, you may run either one as a short trial with a stock CUDA feature, to replace reasoning with a log. Do it under a test switch, with the guest counting its own work, and report whether the guest runs while Gold is resident and whether Z holds. That data doesn't decide the "one grid, one total" question either way, and nothing about it goes into the controller. If you'd rather not, say why in the report.
