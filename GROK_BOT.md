# GROK_BOT.md: standing instructions for Grok Build

Read this file at the start of every session on GPUtronic.

## How the two agents work together
- Grok Bot pushes commits to `controller-fixes`. Treat them as your task list. Read each new commit's message and diff, and run what it asks.
- Post what you ran and what you found to a results branch named `test-results/YYYY-MM-DD_description`. Include logs, CSVs, and a short summary in `docs/analysis/`.
- Your results branches are your answers. Grok Bot checks them.
- If something blocks you, such as missing hardware, a failing build, or an unclear instruction, say so in the commit message or in a short note on the results branch. Don't guess around it.
- Never merge to `main` without an explicit go-ahead from Grok Bot or the owner (bencoupland).

## Current state (update when it changes)
- `controller-fixes` tip: `7400273` (Grok Build's re-check: tensor sensor now off by default). Check `git log` for anything newer.
- Results of `7400273` (`docs/analysis/ready-2026-10-12.md`):
  - Free-run ceiling 543k-585k pulses/s with the tensor sensor off, against 533k for the older binary. Idle polling after the nanosleep removal did not measurably cost rate. Keep idle polling as is.
  - Gold, Transfer and Episode `check` all PASS (Gold mean Z 1.445-1.469).
  - Deadband 0.08: the SM count changed on 0.10% of samples and sat on 47/48 SMs 85% of the time.
  - Open loop at a fixed 48 SMs (10 ms bins): Z 1.404, std 0.051, rate 401k. That is the same spread as closed loop (std 0.052), so the swing is in the plant or the sensing path, not the controller.
- Active on the branch:
  - 1 ms Z logging (`GPUTRONIC_TRACE_BIN_US=1000`).
  - The deadband defaults to 0.08. `GPUTRONIC_DEADBAND=0.015` gives the old band for T6 comparisons.
  - Integral-vs-SM-count counters, in `[STATS]` and the trace CSV.
  - The 4.5 s free-run wait port.
  - `__nanosleep` removed from all kernels.
  - Tensor sensor off by default.
- Next run:
  1. T10 at 46 SMs with 1 ms bins (not yet done; the last noise run used 48 SMs and 10 ms bins).
  2. T11 per-SM ops accumulator, pushed after this file: it decides whether the ~0.05 std open-loop spread is in the work itself or in the pulse/PCIe sensing path. Commands in `docs/analysis/TEST_SUITE.md` (T11).
- Full commands are in `docs/analysis/TEST_SUITE.md` and `docs/analysis/GROK_BUILD_INSTRUCTIONS.md`.
