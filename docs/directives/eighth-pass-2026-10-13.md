# Eighth pass: clamp reach, real idling, and sensor range

Start from the seventh-pass code and results. The directive was commit `c441785` (with `6a7171f` and the catch-up note `778e4b4`). The code you ran is on `test-results/2026-10-13_seventh-pass` (`863898a`), and the spare-SM work sits on top of it on `test-results/2026-10-13_spare-sm` (`2d32ba8`). Build from `2d32ba8` so the reserve override and `GPUTRONIC_NO_CSV` are in. Post the results to a new branch, `test-results/2026-10-13_eighth-pass`, with the logs under `logs/2026-10-13_eighth-pass/`. Do not merge anything, and do not force-push.

## Where things stand

- Part two stopped the T6 collapse. Under the 1.2 load, though, Z settled near 1.64, with the integral on the 3.28-SM rail 89% of the time. The switches-off run did the same at 1.66. Gains were not touched.
- Quiet idle did not cut power. Power stayed at 84–89 W from 4 SMs to 82, because the wait spins on the GPU timer and keeps the SM busy.
- The spare-SM default of 2 is confirmed (`test-results/2026-10-13_spare-sm`).
- The GTX 1080 run was not started.
- Nothing from part two has been merged to controller-fixes itself.

## Standing rules

- Testing runs on the host against the real pinned, mapped control page, `struct GPUTronicControl`. There is no emulation layer.
- Two SMs stay reserved for the operating system. Launch 82 blocks, with a floor of 2 active SMs. Log the smids that were left out.
- Do not measure 84 against 82 or 80 SMs. With desktop load in the background, that difference is not useful data.
- Run plain `nvidia-smi` at the start and the end, and record the display state and the processes on the GPU.
- Set the switches only per run, never as defaults. Kp 0.55, Ki 0.08 and the 0.08 deadband stay unchanged. Item 1 is the only place the clamp changes, and only behind a test switch.
- Record temperature, power and clock on every hold. Stop if the card throttles or after three failures, and push what you have.
- Run the three standard checks with every switch unset, and confirm they pass.

## 1. Widen the integral clamp (clamp only)

The 3.28-SM clamp can't reach far enough to absorb a 20% load. Add a test-only switch, for example `GPUTRONIC_ICLAMP_SMS`, that sets the clamp in SMs for one run. Unset must keep today's scaled law exactly. Do not change Kp, Ki or the deadband.

Rerun the seventh-pass T6 test on the cached plant: Z 1.5, load 1.2 from 20 s, 60 s, tally off. Use clamp settings of 3.28 (today), 6, 9 and 12 SMs. For each one, report:
- Z mean and standard deviation, and the SM count before the load and while it's held.
- How long Z takes to get back inside the 0.08 band after the step, if it does.
- How much of the time the integral is on the rail.
- Any overshoot, or slow wind-down when the load is removed. Add a step back to scale 1.0 at 45 s for this.

Then pick the smallest clamp that holds Z in the band under the 1.2 load. Run that one once with both switches off as well.

## 2. Real idling: end blocks or relaunch instead of spinning

Spinning on the GPU timer keeps the SM issued, so it doesn't save power. Try two ways of actually freeing idle SMs, each behind its own test switch:
- (a) An idle block exits instead of waiting. When the count goes up, the host launches fresh blocks to cover the new SMs.
- (b) Relaunch the grid at the commanded count whenever the count changes, with a minimum hold time so it doesn't thrash.

For each approach, give power, temperature, clock and pulse rate at 4, 12, 24, 36, 48, 56, 72 and 82 SMs, open loop, 30 s holds, beside the spinning baseline. Also report:
- Command latency for a step from 82 down to 42 and back.
- Whether CUDA puts relaunched blocks on the same SMs, from the smid log.
- Whether the reserved SMs 82 and 83 stay empty.
- Whether the desktop stays smooth during count changes. Reuse the present probe from the spare-SM pass.

If neither approach lowers power, say so plainly, and say what you think holds the power floor up.

## 3. Sensor range at maximum throughput

At 82 SMs with both switches on, the cached plant read about 354 million pulses/s. The sixth pass read about 17 TFLOP/s on the full chip. Find out whether the work counter can see the card's theoretical peak, or whether this workload just can't generate it.
- Work out the card's theoretical FP32 peak from its published SM count, cores per SM and the clock you measure. Show the arithmetic.
- Run a max-throughput hold at 82 SMs. Raise work per pass and occupancy (more threads per block, or more independent FMA chains) behind a test switch until the rate stops rising. Report achieved FLOP/s against the peak.
- Check whether `total_work_pulses` or the per-SM tally can wrap or saturate at that rate over a 60 s hold and over an hour. Give the counter widths and the time to overflow.
- Say plainly which limit you hit: the sensor, the workload, or the clock or power budget.

## Open questions for the caller and Ben

1. Should the integral clamp be widened for real, and to what?
2. How should an SM actually be idled: ending blocks, relaunching, or something else?

## The report

Put it at `docs/analysis/eighth-pass-2026-10-13.md`. Open with one paragraph the caller and Ben can make both calls from. Then give one section per item, with tables.
