# Thirteenth pass: is the 15 SM tail miss real, and where does the compiled clamp start to trip

This follows `docs/directives/twelfth-pass-2026-10-18.md` and its results on `test-results/2026-10-18_twelfth-pass` (`add4304`, report `docs/analysis/twelfth-pass-2026-10-18.md`). Start from that results branch at `add4304`. Post everything to a new branch, `test-results/2026-10-18_thirteenth-pass`, with logs under `logs/2026-10-18_thirteenth-pass/`. Do not force-push, and don't merge anything.

## Start here

Do item 1, then item 2, then the success criteria in item 3 and the report and checks in item 4. Follow the window policy below. It replaces the 420 s schedules.

**You may argue.** If you disagree with a directive, say so in the report with your reasoning and evidence, before or instead of carrying it out. A reasoned objection counts as a result. Don't quietly work around a directive.

## How GPUtronic is meant to work (carried forward)

- **Z = 1.0 is the normal default.** The loop stays out of the way and runs at the fastest rate, the plant's own free-run rate. Ben approved making it the compiled default, and you agreed. `GPUTRONIC_GOLD_TARGET_Z` is `1.0f` on `add4304`. Keep it.
- **Z above 1 is an explicit margin mode.** You command it when margin matters more than speed, such as bounded latency for real-time vision, and it carries a throughput penalty. In this pass 1.5 is that mode. It's always set explicitly with `GPUTRONIC_TARGET_Z` or in the check gates, never as a default.
- **The clamp is a guardrail.** It should almost never trip. If it trips on real workloads, the fix is upstream.
- **The stock driver owns thermal and power. GPUtronic owns scheduling.** Keep logging clocks and flags beside Z, read only.
- **Closed loop is the differentiator.** Stay standalone and loosely coupled, and add no knob a result doesn't need.

## Where things stand after the twelfth pass

- **The real-work load works.** `GPUTRONIC_WORK_LOAD=1` changes the independent-FMA rounds per batch (614 at 1.2, 410 at 0.8, from base 512). Every batch still pulses: pulses per batch stayed about 1.000, against 0.833 under pulse-skip. Ops per pulse went from a median of 1,044,578 to 1,257,472. The open-loop rate moved by 1.21. Carry this switch forward unchanged: test only, off by default, every batch counts, and the real work per batch is what changes.
- **Settling and returns failed at 15 SMs.** On the heavy schedule at commanded 1.5 and `GPUTRONIC_ICLAMP_SMS=15`, the returns were 87.1%, 81.4% and 84.7% in band over the last 30 s, against 90.1–91.6% for the pulse-skip twin. The 360 s return ended outside the band for its last 23 bins. The count kept moving (179–318 changes in the last 30 s). The clamp never tripped.
- **9 SMs beat 15 on real work.** All six holds entered and stayed in band. The clamp never tripped, and the returns were 86.5%, 87.6% and 86.5%, in line with the 9 SM pulse-skip shares. No undershoot went deeper than the eleventh pass's 1.28799.
- **The compiled 3.28 SM clamp still trips under real work.** Each loaded hold spent its last 30 s on the rail, and was 18.0%, 17.3% and 18.8% in band. Your reading: the cause is the 1.2 step's size against that rail, not pulse-skipping.
- **Light load (0.8, clamp 15).** The loop gave SMs back, to 43–46. All six holds entered and stayed. Loaded shares were 72–73% and returns 81–83%, with no trips.
- **Z = 1.0 on the heavy load.** The count stayed at 82, the integral never moved, and the loaded holds sat above the band because no SMs were left to add. Returns were 95.2–96.5% in band, with none of the eleventh pass's undershoot.
- **Checks pass with the new default.** The gates command 1.5: Gold 1.449, Transfer 1.441, Episode 1.493. Gains unchanged. Clocks stayed in boost, every reason word was 0, and the peak die temperature was 68 °C.
- **Green contexts and guest-first launch weren't run.** They're still Ben's call. Don't run them this pass.

## Window policy

The 420 s schedules were longer than the questions needed. Trade some settling precision for more steps tested.

- **Clamp tests (item 1):** each hold at least 30 s. The "last 30 s" metrics then cover the whole hold, so also report the last 10 s for comparison.
- **Step sweeps (item 2):** where the behaviour shows within about 10 s, use holds of about 15 s, report the last 10 s, and spend the saved time on more step sizes and repeats. If a point clearly hasn't settled in that window, lengthen that point only, and say so.
- No single run longer than about 5 minutes unless a result needs it. If you lengthen one, say why.
- Keep the eleventh-pass definitions: the band is `|z_hat - target| <= 0.08`, "enters and stays" is the time after the last out-of-band bin, and a limit cycle is exactly two SM counts in the window, each at least five times, with at least six changes. Say which window each share uses.

## Standing rules

- Kp 0.55, Ki 0.08, Kd 0, deadband 0.08, 82-block launch, floor 2. Physical SMs 82 and 83 stay unlaunched. None of these change.
- The compiled target stays at 1.0, and the check gates keep commanding 1.5.
- Ben hasn't picked a compiled clamp width. Don't change the compiled clamp or adopt 9 or 15 as a default. Use `GPUTRONIC_ICLAMP_SMS` to test.
- `libsmctrl` stays out.
- **The controller's view has to be complete.** Every batch that runs is counted.
- **Accepted limit:** idle SMs spin at the GPU clock on the stock driver.
- **Out of scope:** custom drivers, per-SM sleep, per-SM clock or power gating, thermal or power logic in the controller, and anything below the CUDA runtime and driver API.
- Any new switch is test only and off by default. With every switch unset, the binary behaves exactly like `add4304`.
- One GPU job at a time. Stop any run at 83 °C; the log up to the stop is the result.
- No number that isn't measured in this pass or already in the repo, quoted from its file. An early stop or a ruled-out idea is a result.
- The hold region is a deadband, not hysteresis. The Kalman filter is logged only. No motor-control terms for the loop.
- Leave every earlier results branch, the notes branch, `archive/` and all earlier directives as they are.

## 1. The 15 SM tail miss: artifact or regression?

The question is whether the twelfth pass's 15 SM return miss is repeatable, and whether the Z = 1.0 default changed anything. Note that the twelfth-pass 15 SM holds ran before the macro moved, at a compiled target of 1.5.

Run on the current binary (compiled target 1.0), with `GPUTRONIC_WORK_LOAD=1`, the full-pipe switches, `GPUTRONIC_TRACE_BIN_US=1000`, `GPUTRONIC_ICLAMP_SMS=15`, idle-exit and guest unset, and clocks and flags logged. Use the same heavy shape as the twelfth pass (1.2 and 1.0 alternating, three pairs), with holds at the 30 s floor or a little longer.

1. **Z = 1.5, then Z = 1.0, back to back.** The same heavy schedule, once with `GPUTRONIC_TARGET_Z=1.5` and once with `GPUTRONIC_TARGET_Z` unset (the 1.0 default).
2. **Repeat the Z = 1.5 run** at least twice more, so you have three Z = 1.5 runs to compare. If time allows, one Z = 1.5 run at 9 SMs on the same schedule, as the control.

Report the step table for each run. Then say which of these the logs support:

- **Run-to-run variance:** the miss comes and goes across repeats.
- **A regression from the default change:** Z = 1.5 behaves differently now than at `add4304` before the macro moved.
- **A real 15 SM behaviour:** the extra room lets the count hunt on returns, and the miss repeats.

If the logs don't separate them, say so.

## 2. Compiled clamp versus step size: a curve, not single points

You traced the compiled-clamp trip to the step's size against the 3.28 SM rail. Measure that as a curve.

- Compiled clamp (no `GPUTRONIC_ICLAMP_SMS`), commanded Z = 1.5, `GPUTRONIC_WORK_LOAD=1`, the same switches as item 1.
- Sweep the heavy step size: the current 1.2, half of it (1.1), and a quarter of it (1.05). Each step size gets its own load and return pair, using the step-sweep windows, with at least two repeats per size.
- If the curve has room, add points between those sizes, such as 1.15 and 1.075, to find where the integral first reaches the rail.

For each step size report:

- trip rate;
- whether the loaded hold enters and stays;
- the last-10 s in-band share;
- highest and lowest `z_hat`;
- the count at the end against `n_ff`;
- the correction the integral would need against the 3.28 SM rail.

Plot the trip rate and in-band share against step size. Name the largest step the compiled clamp holds without tripping. Don't propose a new compiled width. The curve is what Ben decides from.

## 3. What success looks like

These are criteria, not predictions.

- **The tail miss is explained.** Item 1 says whether the 15 SM return miss is variance, a regression or a real 15 SM behaviour, with the logs that show it, or plainly says the logs don't separate them.
- **The default is clean.** At Z = 1.0 unset, the count sits at 82 and the integral doesn't move, as in the twelfth pass. The commanded Z = 1.5 runs behave the same as at `add4304`, or the difference is reported.
- **The curve is in.** Item 2 gives trip rate and in-band share at at least three step sizes, with repeats, and names the largest step the compiled clamp holds.
- **Guardrail.** Clamp 15 and clamp 9 don't trip on any hold, or any trip is named.
- **Driver orthogonality.** Clocks and flags are logged on every run, and the loop didn't fight the driver.
- **No regressions.** With every switch unset, the `[CTRL]` line shows target 1.00, Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, launch 82. Every new switch is off by default, and the three check gates pass at commanded 1.5.

## 4. Report and checks

Report at `docs/analysis/thirteenth-pass-2026-10-18.md`, in the style of the twelfth-pass report. Include:

- the window used for each run;
- item 1's step tables and verdict;
- item 2's table and plot;
- trip rates;
- the clock and flag findings;
- any switch added, with confirmation it's off by default;
- any directive you argued with, and why;
- the item 3 criteria, with a plain yes or no for each.

Checks:

- `make gold`, `make transfer` and `make episode` build and pass `CHECK` with every switch unset. Record them beside the twelfth pass's 1.449, 1.441 and 1.493.
- Every number in the report matches a committed log in `logs/2026-10-18_thirteenth-pass/`.
- `main` and every earlier results branch are unchanged (`git ls-remote origin`).
