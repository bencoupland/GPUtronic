# Directive for Grok Build: controller implications of the PCIe stall

Written 12 October 2026. Read `docs/analysis/fifth-pass-review-2026-10-13.md` first. Start from `test-results/2026-10-13_fifth-pass` (`1a8fdd3`).

## Ground rules

- Put results on `test-results/2026-10-13_sixth-pass`. Never push to or merge into `main`.
- **Do not turn on `GPUTRONIC_PULSE_LOCAL` or `GPUTRONIC_CTRL_CACHE` by default.** Set them only per run, in that run's environment. The default build and the three checks stay exactly as they are, with both switches off.
- Do not change Kp 0.55, Ki 0.08, the deadband 0.08, or the scaled clamp. If something misbehaves with the switches on, report it. Do not tune around it.
- Record GPU temperature, power draw, and SM clock (for example `nvidia-smi dmon -s pucv`) on every hold from here on. The cached plant is about 470 times more work.
- Stop and write a blocker note after three failures, or if the GPU reaches its thermal or power limit and throttles in a way that makes a run meaningless.

## Experiment 1: re-baseline the cached plant (open loop, test-only)

With both switches on, sweep the working SM count: 4, 12, 24, 36, 42, 48, 56, 64, 72, 84. For each, run open loop for 30 s and report:
- pulses/s, GFLOP/s, and Z against the 84-SM free-run, next to the equal-share value `84/n`
- per-GPC means and CV
- temperature, power, and clock

Repeat 84 and 48 with the switches off, the same evening, for comparison. Question: is the cached plant compute-bound, with Z close to `84/n`, and where does it land 1.5?

## Experiment 2: closed loop on the cached plant (test-only)

Run the Gold closed loop with both switches on, at target Z 1.5, with stock gains, deadband, and clamp, for 60 s. Then run the T6 load step once. Report:
- Whether it holds 1.5, the settled SM count, Z std, and whether it hunts.
- What `gputronic_recalibrate` and the half-SM probe produce (`sms_per_u`), against the switches-off values.
- Whether the 1 ms lumpiness of the batched pulse word shows up as noise in `rate_ema`.

Do not change gains if it fails. Report what failed.

## Experiment 3: command latency

With the words cached at 1 ms, a new `active_sms` reaches each block up to 1 ms late. Measure the step response, the time from a command change to the new tally level, with the cache period at 0.25, 1, and 4 ms. If the switch does not take a period, add one as a test-only variable.

## Experiment 4: what the pulse stream should be (measurement only, no redesign)

Gather the numbers needed for a redesign decision. Do not build the redesign.
- Compare the per-SM tally (summed, read every 1 ms and every 20 ms) as a rate sensor against the batched pulse word: noise (std of rate at a fixed SM count) and lag.
- Measure the PCIe traffic at 84 SMs with switches off versus on (`nvidia-smi dmon -s t`).
- State, from these numbers, whether further throughput gains need the pulse stream redesigned rather than more GPU-side tuning.

## Report back

- A results note in `docs/analysis/`, with one table per experiment.
- One paragraph at the top covering: what the cached plant's Z against SM count looks like, whether the stock loop holds 1.5 on it, and what the pulse stream should become. Ben and the caller will make the design call from it.
- Confirm the three checks still pass with both switches off.
