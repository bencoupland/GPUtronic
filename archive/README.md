# Archive index

Older GPUTronic work stays here. Do not delete it to tidy the repo. Current code is the repo root (`make gold`).

| Path | What it is |
|------|------------|
| `v0.1-gtx1080-2026-01/` | First public tree, committed 11 January 2026. GTX 1080, 10 Hz loop, `main.cu`. |
| `pre-gold-2026-05/` | Experiments from early 2026 through May 2026. Source ladder v14–v26, logs, scripts. Many of the long writeups in `docs/` are only on disk, not all of them are in git. |
| `gold-1.0.0-nanosleep/` | 1.0 as first posted, when the control was sleep duration. Kept when the control became an SM count on 2026-09-23. |

There is no commit in this repo from 2025. History in git starts on 11 January 2026.

Inside `pre-gold-2026-05/`:

## Quick map

| Path | What |
|------|------|
| `sources/` | Experimental `.cu`/`.cpp` (v14–v26, PoC, launchers) |
| `binaries/` | Prebuilt governors / harnesses |
| `logs/` | Run logs (compressed `.xz` if large) |
| `docs/` | Old READMEs, test plans, stage reports, "From Grok" |
| `scripts/` | Cyberpunk launch/sweep/capture scripts |
| `results/` | Dated cyberpunk_* campaigns + early analysis |
| `README.md` | Archive policy |

## Active tree (do not put experiments back in root)

```
.
├── AGENTS.md STATUS.md GOLD.md README.md Makefile
├── include/gputronic.h
├── src/gputronic_gold.cu
├── examples/gold_demo.c
├── scripts/analyze_*.py
├── docs/          # essay, tuning PDF, Kalman notes
├── results/       # Gold CSVs/reports only
├── build/         # gputronic_gold, demo, lib
└── archive/
```

## Restore an old experiment

```bash
cp archive/pre-gold-2026-05/sources/gputronic_q_axis_governor_v26_safe_fixed.cu /tmp/
# build ad-hoc — not via make gold
```
