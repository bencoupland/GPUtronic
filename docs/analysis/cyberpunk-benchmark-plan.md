# Plan: Cyberpunk 2077 co-run benchmark (proposed, not run)

Read `NEXT.md` section C7 before using the table below. Condition B is not
`gputronic_gold run <sec> 1.0`. On this branch that request sat at a mean of 83 SMs,
and only 57% of the samples were at 84. Full width is an open-loop count of every SM
(`gputronic_set_open_loop_sms(h, 0)`). There is no `run` argument that does that.
Condition D is Gold alone at the same closed-loop target. It is not the free-run
ceiling. The ceiling is that run's calibration `rate_ref`. This machine's measured
game was Space Marine 2 at 4K, Ultra, DLSS Quality, frame generation on. Do not
change those settings to match the preset below. A co-run is a demo, and it is later
than the changes in section A of `NEXT.md`.

## What this can and cannot show
GPUTronic controls **only its own kernel's SM count** (`active_sms`). It does not throttle, schedule or
see the game. The experiment is therefore a co-run: Gold runs beside the game on the same RTX 5080, and
we measure whether holding Gold at Z = 1.5 hurts the game's frame times less than an uncontrolled,
full-width Gold, while Gold still delivers useful throughput.

## Conditions (each 5 runs, first discarded)
| ID | Game | GPUTronic |
|---|---|---|
| A | benchmark | none (baseline) |
| B | benchmark | open-loop, every SM (`gputronic_set_open_loop_sms(h, 0)`). `run` at Z 1.0 is not this |
| C | benchmark | `gputronic_gold run <sec> 1.5` (closed loop) |
| D | none | `gputronic_gold run <sec> 1.5` (Gold alone). Ceiling is that run's `rate_ref`, not this row |
Tensor sensor off (`GPUTRONIC_TENSOR_SENSOR=0`, per 2026-10-12 results). Start GPUTronic first and let
calibration finish (~3–4.5 s) before launching the game; run it long enough to cover the game run.

## Fixed game settings
One preset (e.g. High), fixed resolution (e.g. 2560x1440), ray tracing on or off (state which), DLSS
fixed (state mode or off), **frame generation off, vsync off**, frame cap off. Pin the settings by
saving `UserSettings.json` once and restoring it before each run. Same Proton version for all runs.

## Launching on Linux (Steam + Proton; box: kernel 7.0, driver 595.99)
| Fact | Status | Source |
|---|---|---|
| Steam launch options `%command% --launcher-skip --intro-skip` skip the launcher/intro | verified (community) | ProtonDB https://www.protondb.com/app/1091500 ; Steam forum https://steamcommunity.com/app/1091500/discussions/7/603020130763412267/ |
| `-benchmark` argument starts the built-in benchmark and exits | verified (community, not CDPR docs) | https://www.dolpa.me/running-the-cyberpunk-2077-benchmark-from-the-ubuntu-terminal-with-proton-my-journey/ ; https://github.com/dolpa/dolpa-gaming-on-linux/tree/main/games/cyberpunk%202077 |
| Results written to `<compatdata>/1091500/pfx/drive_c/users/steamuser/Documents/CD Projekt Red/Cyberpunk 2077/benchmarkResults/benchmark_<timestamp>/summary.json` | verified (community) | same dolpa sources |
| `PROTON_ENABLE_NVAPI=1` needed for DLSS; `VKD3D_CONFIG=dxr11` for ray tracing | verified (community) | ProtonDB page above |
| `-resolution` / `-upscale` arguments | **unverified** (one blog only); set these through `UserSettings.json` instead | — |
| Any official CD Projekt Red documentation of `-benchmark` | **unverified / not found** | — |
| Manual fallback: main menu → Settings → Run Benchmark | **unverified** in this pass | — |

Launch options example:
`PROTON_ENABLE_NVAPI=1 MANGOHUD=1 MANGOHUD_CONFIG=log_interval=0,output_folder=$LOG/mangohud %command% --launcher-skip --intro-skip -benchmark`
(Check MangoHud option names against the installed MangoHud version before use.)

## Logs per run
- MangoHud frame-time log (per frame, with its own timestamps).
- Game `summary.json`.
- GPUTronic CSV (`results/gputronic_gold_run.csv`: `time_s`, Z, SM count, rate) and stdout.
- `nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,pcie.link.gen.current --format=csv -lms 100` and `nvidia-smi dmon -s t`.

## Time alignment
GPUTronic `time_s` is relative to its own start. Before each run, record `date +%s.%N` immediately
before starting GPUTronic and before launching the game; convert every log to wall-clock seconds
(CLOCK_REALTIME). nvidia-smi timestamps are wall-clock. MangoHud log start time is taken from its
file header or file mtime (check which). Accept ±10 ms alignment error; GPUTronic's rate window is 8 ms.

## Metrics
Per run: average FPS, 1% low, 0.1% low, frame-time variance and 99th-percentile frame time (from
MangoHud), plus GPUTronic mean Z, SM count, delivered rate as % of condition D, and mean clocks/power.

## What would show closed-loop benefit (proposed thresholds, to tune after the first pass)
- C's 1% low within **5%** of A, while C delivers at least **50%** of D's free-run rate.
- B's 1% low is measurably worse than C's (e.g. >10% below A).
- C's frame-time variance no worse than 1.2× A.
If C and B look the same, the controller is not helping the game and the result says so.
