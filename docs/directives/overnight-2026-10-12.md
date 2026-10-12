# Overnight directive for Grok Build, 12 October 2026

Ben is not present tonight. Follow this file in order. Background: `logs/2026-10-12_t11/wrap-up-2026-10-12.md` and `docs/analysis/load-imbalance-2026-10-12.md` on `test-results/2026-10-12_load-imbalance`.

## Ground rules

1. Read `GROK_BOT.md` first. Treat Grok Bot's commits on `controller-fixes` as your tasks.
2. Post every result, including code and logs, to a `test-results/2026-10-13_...` branch. Never push to `main` and never merge to `main`.
3. Do not retune. Kp stays 0.55, Ki 0.08, deadband 0.08, and the integral clamp stays the scaled one (about 3.4 SMs on the 5080). The tally stays out of the PI.
4. If anything blocks you, or a test fails three times, stop. Write a short blocker note on your results branch saying what failed, the exact command, and the error. Do not push through or work around it.

## Task 1: record the physical SM ID (first priority)

This is the single change that unblocks the load-imbalance diagnosis. Today the tally slot is labelled by `blockIdx.x`, so we do not know which physical SM each block ran on.

- In `src/gputronic_gold.cu`, read the hardware SM register (`%smid`) once at block start, near `const int sm = blockIdx.x;` (line 144). Store it in the block's slot near the slot write (about line 221). Either use the tensor-ops word when the tensor sensor is off, or widen the slot from 4 to 5 words. Keep the block index as the slot key.
- In the host writer near line 1600, add the physical SM ID as a column in each CSV row.
- In `scripts/analyze_smaccum.py`, key the per-SM statistics on the physical SM ID, and also print the block-to-SM map.
- Compile for `sm_120` and `sm_61`. Run the Gold, transfer, and episode `check` with the tally off. They must still pass before Task 2.

## Task 2: open-loop tally at 36, 46, and 48 SMs

Run the open-loop tally, every block reporting, at 36, 46, and 48 working SMs. Use 1 ms host bins and the analyzer's 20 ms buckets. For each, report the per-block means, the block-to-SM map, and the coefficient of variation.

If 36 and 48 are flat and 46 still steps at block 36, the partly-filled-group theory is confirmed. Say plainly whether it was confirmed or not.

## Task 3: idle polling

Run one 46-SM hold with the idle blocks not reading the mapped words. This is a test-only switch, off by default. Compare against the 46-SM run from Task 2. Report whether the step changed.

## Sign-off

Before stopping, put a one-paragraph summary at the top of your results branch's `docs/analysis/` note: what ran, the key numbers, what it showed, and anything left blocked.
