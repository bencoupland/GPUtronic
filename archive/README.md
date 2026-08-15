# Archive index — pre-Gold material

All historical GPUTronic work before **Gold 1.0.0** lives under:

```
archive/pre-gold-2026-05/
```

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
