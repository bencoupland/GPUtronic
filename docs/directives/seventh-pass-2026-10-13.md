# Seventh pass: cache the test-load word, rerun T6, and check quiet idle on the cached plant

This is one run with three pieces. Start from the sixth-pass code on `test-results/2026-10-13_sixth-pass` (commit `eaa9cb1`), because that is where the cache switch lives. Post the results to a new branch, `test-results/2026-10-13_seventh-pass`. Do not merge to main, and do not force-push.

## Why

The sixth pass showed that the cached plant tracks the SM count and that the stock loop holds Z near 1.5. Two things were left open.

- The T6 load step collapsed. The load flag makes every warp read `work_scale_milli` on every pass, and that word was not in the cache. The pulse rate fell from about 266 million/s to about 1.3 million/s, Z went to the ceiling of 10, and the controller opened all 84 SMs.
- Power sat at 83–86 W at every SM count, even at 4 SMs, because idle blocks still spin on the cached words. Quiet idle was off during that sweep.

## Reserved SMs for the operating system

The 5080 also runs the desktop. Keep two SMs free at all times for the operating system's display and desktop work, so the screen keeps refreshing and the machine stays responsive. Do not schedule test work onto them.

- The test ceiling is 82 SMs, not 84. Wherever this directive or the sixth pass says 84, use 82. That covers the full-chip holds, the reference rate Z is measured against, and the closed loop's upper limit. Launch only 82 persistent blocks, so the last two SMs carry no resident test block at all, not even an idle one spinning on the control words. CUDA picks which physical SMs those are. Record the smid of each launched block so we know which two were left out.
- The floor is two active SMs. Do not command fewer than two, in open loop or closed loop. In the quiet-idle sweep the lowest count is still 4.
- Do not try to measure the difference between 84, 82 and 80 SMs. With the desktop load in the background, that difference is unlikely to give useful data. Reserving two SMs is how we handle it.
- Run plain `nvidia-smi` once at the start and once at the end. Record whether the display is attached and which processes are listed against the GPU.

## How the testing works

The tests run on the host against the real control-word page. That page is GPUtronic's own pinned and mapped block, `struct GPUTronicControl`, read and written across PCIe. There is no GPU emulation layer, and nothing is simulated.

## Piece 1: put the test-load word in the cache

When `GPUTRONIC_CTRL_CACHE=1`, the block leader should read `work_scale_milli` together with the other control words, once per `GPUTRONIC_CTRL_CACHE_US`, and warps should use that cached copy. With the switch off, the behaviour must not change at all. Keep the switch off by default. Say in the report exactly which lines changed.

## Piece 2: rerun T6 on the cached plant

Gold closed loop, both switches on (`GPUTRONIC_PULSE_LOCAL=1`, `GPUTRONIC_CTRL_CACHE=1`), tally off, target Z 1.5, stock gains, 60 s. Load scale 1.2 from 20 s to the end, the same as the sixth pass. Report these:

- Z mean, standard deviation, min and max, and SM count mean and range, before the load (10–19 s) and while the load is held (25–55 s).
- How long Z took to return inside the 0.08 deadband after the step.
- The pulse rate before and after the step, and how it compares with the roughly 83% you would expect from a scale of 1.2.
- Whether the count settles higher, as it should to make up for the load.

Also rerun T6 once with both switches off, as the baseline.

## Piece 3: quiet idle on the cached plant

Open loop, 30 s holds, both switches on, tally off, with quiet idle on (`GPUTRONIC_IDLE_QUIET=1`, or whichever name the binary uses). Use these counts: 4, 12, 24, 36, 48, 56, 72, 82. Then repeat 4, 36 and 82 with quiet idle off, so the two can be compared directly.

For every hold, record board power in watts, temperature, the SM clock, the pulse rate, and Z. The question is plain: does power now follow the SM count? Give the watts at 4 SMs and at 82 SMs for quiet on and quiet off. Also check that quiet idle does not slow the command latency. Rerun the sixth-pass Experiment 3 step at the 1 ms cache period with quiet on, 82 down to 42 and back.

## Rules (same as the sixth pass)

- Set the switches only per run, never as defaults.
- Do not touch the gains, deadband or clamp: Kp 0.55, Ki 0.08, deadband 0.08, scaled clamp 3.36 SMs.
- Record temperature, power and clock on every hold.
- Stop if the card throttles (a thermal or power violation, or the clock leaving its step), or after three failures. Write down what happened and push what you have.
- Run the three standard checks with the switches unset, and confirm they still pass.
- Write in plain English.

## The report

Put it at `docs/analysis/seventh-pass-2026-10-13.md` on the results branch. Open it with one paragraph that the caller and Ben can make the design call from. That paragraph should answer three questions: does Z hold under load on the cached plant, does power follow the SM count with quiet idle, and is the cached design ready to become the real controller. Then give one section for each piece, with tables.
