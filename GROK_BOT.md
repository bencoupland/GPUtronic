# GROK_BOT.md: standing instructions for Grok Build

Read this file at the start of every session on GPUtronic.

## How the two agents work together
- Grok Bot pushes commits to `controller-fixes`. Treat them as your task list. Read each new commit's message and diff, and run what it asks.
- Post what you ran and what you found to a results branch named `test-results/YYYY-MM-DD_description`. Include logs, CSVs, and a short summary in `docs/analysis/`.
- Your results branches are your answers. Grok Bot checks them.
- If something blocks you, such as missing hardware, a failing build, or an unclear instruction, say so in the commit message or in a short note on the results branch. Don't guess around it.
- Never merge to `main` without an explicit go-ahead from Grok Bot or the owner (bencoupland).

## Current state (update when it changes)
- On 2026-10-12 Ben Coupland told the 5080 seat to commit this law to `main`. That push is the tree. Check `git log` before assuming the tip below is still current.
- T11 is done. Open loop at 46 SMs, 1 ms host bins. At 100 ms the sum of the per-SM tallies matched the pulse counter: ratio 1.000, drift 0.01%, correlation 0.995. The Z spread at a fixed count is finished passes. Working SMs were uneven (coefficient of variation 29%). SM 0's share was 0.75 and it still tracked (correlation 0.977). Note: `docs/analysis/per-sm-accumulator.md`. Logs: `logs/2026-10-12_t11/`.
- The same build, tally off, passed Gold, transfer, and episode `check`. Gold gate mean Z 1.463 (free-run 541k, closed 363k). Transfer 8.48 to 6.03 GFLOP/s, mean Z 1.450. Episode 22.38 to 14.94 GFLOP/s, mean Z 1.468. Tensor sensor off. Deadband 0.08. Kp 0.55. Ki 0.08.
- Do not retune Kp or Ki. Do not start the counter that keeps going after a kernel returns until Ben asks. A GTX 1080 run and a game co-run are still open. `sm_61` compiled. It was not executed on a 1080.
- `archive/lab-2026-10-05` is the October 5 working tree (follow, share, beside). It is not this controller.
