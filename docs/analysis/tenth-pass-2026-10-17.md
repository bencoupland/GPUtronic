# Tenth pass, 17 October 2026

Measured on the RTX 5080 in this machine. Driver 595.99.02. CUDA build 12.8, `nvcc` 12.8, `-arch=sm_120`. Plain `nvidia-smi` at the start of the pass was 44 °C and 36 W (`logs/2026-10-17_tenth-pass/nvidia-smi-start.txt`). At the end it was 48 °C and 32 W (`nvidia-smi-end.txt`). No run was stopped for temperature. The 83 °C stop was armed and did not fire.

The tree is `main` at `b611a2b` plus the guest switch below. Gains stayed at Kp 0.55, Ki 0.08, Kd 0, deadband 0.08. The scaled integral clamp stayed. The launch stayed at 82 blocks. The command floor stayed at 2. `GPUTRONIC_DEADBAND`, `GPUTRONIC_INTEGRAL_CLAMP`, `GPUTRONIC_ICLAMP_SMS`, `GPUTRONIC_RELAUNCH`, and `GPUTRONIC_IDLE_QUIET` were left unset.

Every Gold run used:

```
GPUTRONIC_PEAK_ACC=32 GPUTRONIC_PEAK_ROUNDS=512 GPUTRONIC_THREADS=512 \
GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1
```

`nvidia-smi dmon -s pucm -d 1` was recorded for each run. Phases B and C and the load step also sampled `clocks_event_reasons.active` once a second, because `dmon -s pucm` on this driver has no violation column.

## The switch

`GPUTRONIC_GUEST_BLOCKS` sets the block count of a second kernel. `GPUTRONIC_GUEST_AT_MS` (default 8000) is when `run` launches it. Unset, nothing launches. The guest has its own mapped pulse word, its own stream, and its own keep buffer. It does not read `active_sms` and it does not add to `total_work_pulses`. One guest pulse is one warp finishing one pass of the same 32-chain, 512-round independent FMA block. A block whose physical SM id is `device_sms-2` or `device_sms-1` (82 and 83 on this card) returns without pulsing. `guest` runs that kernel with the Gold kernel not launched.

The same row was added to `docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md` and to the copy of that text in `README.md`. From the shared heading onward the two copies are byte-identical.

A short `run 4` with every `GPUTRONIC_` variable unset was compared with the binary built from `main` before this edit (`logs/2026-10-17_tenth-pass/ctrl-baseline.txt` and `ctrl-unset.txt`). Both print:

```
[GPU] NVIDIA GeForce RTX 5080 | SMs=84 (launching 82, 2 reserved for the OS) | thr/block=64
[LAW] target_z=1.50 Kp=0.550 Ki=0.0800 Kd=0.0000 deadband=0.080 width-actuator two-sided
[CTRL] Gold | target=1.50 Kp=0.550 Ki=0.0800 Kd=0.0000 deadband=0.080 kalman=1 two-sided | cal 1.8s + 4-point table
```

The `[TEL]` columns match. The unset run prints no `[GUEST]` line. The thread count on that pair is 64 because `GPUTRONIC_THREADS` was unset. The phase runs below set it to 512.

## Checks

Built with `make gold`, `make transfer`, and `make episode`. Switches unset.

The binary from `main`, before the guest edit:

| Target | CHECK mean Z | Gate |
| --- | --- | --- |
| Gold | 1.492 | `dyno=PASS closedloop=PASS` |
| Transfer | 1.435 | `dyno=PASS closedloop=PASS` |
| Episode | 1.457 | `dyno=PASS closedloop=PASS` |

Logs: `check-gold.txt`, `check-transfer.txt`, `check-episode.txt`.

The binary that ran the phases, switches still unset:

| Target | CHECK mean Z | Gate |
| --- | --- | --- |
| Gold | 1.436 | `dyno=PASS closedloop=PASS` |
| Transfer | 1.457 | `dyno=PASS closedloop=PASS` |
| Episode | 1.474 | `dyno=PASS closedloop=PASS` |

Logs: `check2-gold.txt`, `check2-transfer.txt`, `check2-episode.txt`.

These are the `[CHECK] z_mean` lines. The `[STATS]` mean on a closed-loop run is a different window and is not the gate.

## Phase A

Idle-exit, no guest. Command:

```
GPUTRONIC_PEAK_ACC=32 GPUTRONIC_PEAK_ROUNDS=512 GPUTRONIC_THREADS=512 \
GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 \
GPUTRONIC_IDLE_EXIT=1 GPUTRONIC_NO_CSV=1 \
./build/gputronic_gold run 25
```

Log: `phase-a.txt`, `phase-a.dmon`. The launch line is `[GPU] ... launching 82 ... thr/block=512`. The law line is Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080. The clamp line is `integral clamp=1.233 (scaled, max integral correction 3.28 SMs of 82)`. Physical SMs 82 and 83 are the ones left out at launch.

The window is the 22 one-second samples from 4 s to 25 s, the same window as the ninth pass. Inside the band means `|Z-1.5| <= 0.08`. The test uses `Z_raw`. `Z_hat` is listed beside it.

| | Z_raw | Z_hat |
| --- | --- | --- |
| mean | 1.4825 | 1.4825 |
| min | 1.435 | 1.435 |
| max | 1.553 | 1.553 |
| inside the band | 22 of 22 | 22 of 22 |

All 22 samples have `n=54`. The settled count is 54, so the guest block count for the later phases is 82 − 54 = 28.

`[IDLE] spawns=17 relaunches=0`. All 17 `[SPAWN]` lines print `ok`. The full lines are in `phase-a.txt`. `dmon` shows board power from 38 W to 212 W and die temperature from 50 °C to 64 °C across 26 samples. The 38 W sample is the idle line before the kernel is up. The other 25 samples are 143–212 W and 54–64 °C.

Phase A stayed inside the band, so phases B and C and the load step ran.

## Phase B

Guest alone, 28 blocks, 10 s. Gold's persistent kernel was not started (`[GUEST] gold resident=0`). `gputronic_create` still prints `launching 82`, which is the configured Gold grid, and that grid was not launched. Command:

```
GPUTRONIC_PEAK_ACC=32 GPUTRONIC_PEAK_ROUNDS=512 GPUTRONIC_THREADS=512 \
GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 \
./build/gputronic_gold guest 10 28
```

Log: `phase-b.txt`, `phase-b.dmon`, `phase-b.flags`.

`concurrentKernels=1`. The guest line is `blocks=28 acc=32 rounds=512 threads=512 skip_sm=82,83`. All ten one-second samples report `live=28`. The ten rates run from 9,025,575 to 9,057,553 pulses/s. Their mean is 9,040,655 pulses/s. That mean is the solo figure for phase C. The stop line is `pulses=90424848 live=28`.

`dmon` peaks at 83 W. After the idle sample and one climb sample at 68 W, the readings are 82–83 W at 45–46 °C. Every `clocks_event_reasons.active` sample is `0x0000000000000000`.

## Phase C

Gold as in phase A, guest of 28 blocks launched 8 s after the run starts. Command:

```
GPUTRONIC_PEAK_ACC=32 GPUTRONIC_PEAK_ROUNDS=512 GPUTRONIC_THREADS=512 \
GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 \
GPUTRONIC_IDLE_EXIT=1 GPUTRONIC_NO_CSV=1 \
GPUTRONIC_GUEST_BLOCKS=28 GPUTRONIC_GUEST_AT_MS=8000 \
./build/gputronic_gold run 40
```

Log: `phase-c.txt`, `phase-c.dmon`, `phase-c.flags`.

`concurrentKernels=1`. The launch print is `[GUEST] launched at 8001 ms`, on the `t=8` line. That line is `Z_raw=1.515`, `n=54`, `guest=0`. The hold after the guest was launched is the 32 samples from `t=9` through `t=40`.

| | Z_raw | Z_hat |
| --- | --- | --- |
| mean | 1.4963 | 1.4967 |
| min | 1.443 | 1.443 |
| max | 1.573 | 1.573 |
| inside the band | 32 of 32 | 32 of 32 |

`t=9` is `n=54`. The other 31 samples are `n=57`. The guest field on all 32 samples is 0. That is 0 relative to the phase B mean of 9,040,655 pulses/s. The stop line, after Gold was told to stop, is `pulses=448 live=28`. 448 is 28 blocks times 16 warps, which is one count from each warp on one pass.

`[IDLE] spawns=16 relaunches=0`. All 16 `[SPAWN]` lines print `ok`. The whole-run stats line, which includes calibration, says `integral_on_rail=0` and `mean_Z=1.5036`. That mean is a different window from the table above.

`dmon` spans 36–209 W and 43–57 °C over 42 samples. The 209 W sample is the climb. Through the overlap the one-second power samples sit at 161–163 W (`phase-c.flags`). Every `clocks_event_reasons.active` sample is 0.

Reading the three phases together: with idle-exit and no guest, Z stayed inside the band at 54 SMs. The same 28 blocks, run on their own, held about 9.04 million pulses/s at about 83 W. With both kernels in the same process, Z stayed inside the band and the guest counter stayed at 0 until Gold was stopped. The count moved from 54 to 57 and stayed there. The run's integral-on-rail counter is 0, so that step of 3 SMs is not the clamp sitting on its rail for the run.

## Load step

Phase A stayed in band, so the load step ran. Idle-exit was left unset. `GPUTRONIC_NO_CSV` was left unset. This is a synthetic load. On this kernel `work_scale_milli` skips pulses and the FMAs keep running, so the counter reports a slower job while the SMs do the same work. It is a drop in the counted rate. The schedule is `logs/2026-10-17_tenth-pass/steps.txt`, with t = 0 at the end of calibration:

```
60000  1.2
120000 1.0
180000 1.2
240000 1.0
300000 1.2
360000 1.0
```

Command:

```
GPUTRONIC_PEAK_ACC=32 GPUTRONIC_PEAK_ROUNDS=512 GPUTRONIC_THREADS=512 \
GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 \
GPUTRONIC_TRACE_BIN_US=1000 \
timeout --signal=INT --kill-after=30s 480 \
  ./build/gputronic_gold loadfile logs/2026-10-17_tenth-pass/steps.txt 420
```

The clamp line on this run is `integral clamp=1.402 (scaled, max integral correction 3.28 SMs of 82)`. Log: `load.txt`, `load.dmon`, `load.flags`. The run returned 0 at 51 °C.

The 1 ms trace is the loadfile sampler with `GPUTRONIC_TRACE_BIN_US=1000`. Its header includes `wall_s,t_ms,scale,z_raw,z_hat,active_sms,integral,clamp`. `t_ms` starts at the end of calibration. The file the program wrote is `results/gputronic_gold_loadfile_10ms.csv` (the name is older than the 1 ms bin). The committed copy is `loadfile-1ms.csv`, 52,189,194 bytes, 419,971 data rows, from t = 0 to t = 419,999 ms.

The control thread also wrote `results/gputronic_gold_load.csv` at about 20,000 rows/s. That file was 760,220,277 bytes. One row was kept for each new millisecond of its `time_s` column, which starts when the control thread starts and so includes calibration. The kept file is `load-tick-1ms.csv`, 422,698 data rows, 46,411,104 bytes. The full 10 µs file was then deleted. The step table below is computed from `loadfile-1ms.csv`, not from the decimated tick file.

`dmon` has 420 samples, power 38–212 W (mean 188 W, the 38 W sample included), die 43–63 °C. Every `clocks_event_reasons.active` sample is 0.

A tick is on the integral clamp when `|integral| >= 0.999 * clamp`. That is the test in the source that increments `st_rail`. The clamp share below is `rail_ticks / pi_ticks` in the 1 ms trace, which is that counter summed over the bin. The snapshot of `integral` and `clamp` on the same rows agrees: every last-30 s row on a 1.2 hold is on that test, and none of the last-30 s rows on a return to 1.0 is.

The band test for this section is on `z_hat`. "Enters and stays" means the first 1 ms bin where `|z_hat-1.5| <= 0.08` and every later bin in that hold is also inside. On all six holds that bin falls in the last few seconds. The tail after it is the only stretch that actually stays inside. On the three 1.2 holds that tail is 26 ms, 19 ms, and 3 ms. The loop does not settle inside the band and remain there for the rest of a hold.

A limit cycle, for this table, means the last 30 s uses exactly two SM counts, each seen at least five times, with the count changing at least six times. None of the holds meets that. They visit more than two counts.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end of hold | SM range in the hold | Last 30 s inside the band | Last 30 s on the clamp | SM changes in the last 30 s | Counts in the last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | 1.2 | 10.1 ms | 59.973 s | 26 ms | 1.79186 | 1.54186 | 55 | 61 | 55–65 | 16,509 / 30,001 (55.0%) | 500,206 / 500,206 (100%) | 937 | 61, 62, 63, 64 |
| 120 s | 1.0 | 21.2 ms | 58.634 s | 1.365 s | 1.75856 | 1.35543 | 61 | 55 | 55–61 | 28,843 / 30,000 (96.1%) | 0 / 504,459 (0%) | 122 | 55, 56, 58, 59, 60, 61 |
| 180 s | 1.2 | 24.1 ms | 59.980 s | 19 ms | 1.88327 | 1.46427 | 55 | 61 | 55–66 | 17,180 / 30,000 (57.3%) | 481,643 / 481,643 (100%) | 862 | 61, 62, 63, 64, 66 |
| 240 s | 1.0 | 21.2 ms | 56.814 s | 3.185 s | 1.74508 | 1.35735 | 61 | 55 | 55–62 | 28,896 / 30,001 (96.3%) | 0 / 500,245 (0%) | 111 | 55, 56, 58, 59, 61 |
| 300 s | 1.2 | 19.1 ms | 59.996 s | 3 ms | 1.76039 | 1.47646 | 55 | 61 | 55–64 | 16,973 / 29,993 (56.6%) | 482,491 / 482,491 (100%) | 897 | 61, 62, 63, 64 |
| 360 s | 1.0 | 19.2 ms | 59.555 s | 0.444 s | 1.71847 | 1.35820 | 61 | 55 | 55–61 | 29,256 / 30,001 (97.5%) | 0 / 492,238 (0%) | 33 | 55, 56, 58, 59, 60 |

On each 1.2 hold the miss is on the high side of the band. On each return to 1.0, `z_hat` also goes below the band (about 1.36) while the larger miss is still high. Over the whole hold, the share of 1 ms bins inside the band is 32.9%, 77.1%, 49.5%, 77.1%, 49.1%, and 78.0%, in step order. The whole-run stats line says `integral_on_rail=2328122` out of `pi_ticks=6870756`, and `active_sms=49..66`. That line mixes calibration with both scales.

The count at the end of each 1.2 hold is 61. During the last 30 s of those holds the integral is on the clamp on every control tick, and the count still moves among 61 and a few counts above it. The printed clamp authority on this run is 3.28 SMs.

`load-zhat.svg` is `z_hat` and `active_sms` against time, with the band shaded and the six steps marked. `plot_load.py` drew it from `loadfile-1ms.csv`, keeping one point about every 100 ms. Matplotlib is not installed here, so the script writes the SVG directly.

## What this pass leaves alone

Merging the ninth-pass results, choosing which idle path is the compiled default, a disturbance that skips the FMA block, more rounds, a tensor or memory body, the GTX 1080, and a game beside the loop are all still open. The gains and the clamp were not changed.
