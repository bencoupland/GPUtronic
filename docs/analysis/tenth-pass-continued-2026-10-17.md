# Tenth pass, continued, 17 October 2026

This continues the tenth pass. The parent is `test-results/2026-10-17_tenth-pass` at `4224435`. The guest switch was already in that tree. Logs are `logs/2026-10-17_tenth-pass-continued/`. Kp 0.55, Ki 0.08, deadband 0.08, the 82-block launch, and the floor of 2 were not changed. The compiled integral clamp was not changed.

Item 2 was pushed before these load steps. The source was not changed for them.

## 1. Ninth-pass results on main

`e818644` was not an ancestor of `origin/main` (`b611a2b`). Pull request 6 merged `test-results/2026-10-13_ninth-pass` into `main` with a merge commit. The merge was clean. `origin/main` is `41a4598`, parents `b611a2b` and `e818644`. That commit lands the README landing page, the Results section, the renamed `docs/GPUtronic-GOLD-1.0-FUNCTION-FRAME.md`, and `logs/2026-10-13_ninth-pass/`.

`test-results/2026-10-17_tenth-pass` is still `4224435`. The ninth-pass branch is still `e818644`.

## 2. Where the guest's work went

The guest kernel now records, per block, the physical SM id, the GPU time the block entered, and the GPU time of its first published pulse. The host prints those at the stop, next to `[GUEST] launched at` and `[GUEST] gold stop at`. It also prints, once a second, how many of those words it can already see. Unset, no guest is launched.

Phase C was repeated from the tenth pass: `run 40`, 28 guest blocks at 8000 ms, idle-exit on, `GPUTRONIC_NO_CSV=1`, and the full-pipe switches (`GPUTRONIC_PEAK_ACC=32`, `GPUTRONIC_PEAK_ROUNDS=512`, `GPUTRONIC_THREADS=512`, `GPUTRONIC_PULSE_LOCAL=1`, `GPUTRONIC_CTRL_CACHE=1`). The window below is the 32 printed samples from `t=9` through `t=40`. Inside the band means `|Z-1.5| <= 0.08`.

The timestamps are the same shape on every run. All 28 blocks enter, and the host first sees the entry word and the first pulse on the stop sample. On the GPU timer the first pulse is 43 µs to 47 µs after entry, and `last_ns` equals `first_ns`. The stop line is `pulses=448 live=28` (28 blocks × 16 warps, one pass). During the hold the one-second guest rate is 0, which is 0 of the phase B solo rate 9,040,655 pulses/s. The blocks waited until Gold was told to stop, then each warp finished one pass. They did not enter early, and the mapped word is not hiding an earlier pulse: once a block runs, the host sees it on that sample.

The SM ids are the low ones (0–27, or 1–28 on two of the runs). Those ids are also in the Gold resident list printed at launch. The blocks arrived on them only as Gold stopped, so this is not a hold spent sharing an SM.

| Run | What was set | Z_raw mean, min, max, in band | Z_hat mean, min, max, in band | SM counts | Guest rate | Stop |
| --- | --- | --- | --- | --- | --- | --- |
| `h1-connections` | `CUDA_DEVICE_MAX_CONNECTIONS=32` | 1.4947, 1.422, 1.597, 30 of 32 | 1.4937, 1.422, 1.597, 30 of 32 | 54 on 12, 57 on 20 | 0 | pulses=448, stop 40005 ms, seen 40006 ms |
| `h2-stream-first` | `GPUTRONIC_GUEST_STREAM_FIRST=1` | 1.4968, 1.448, 1.575, 32 of 32 | 1.4973, 1.447, 1.575, 32 of 32 | 54 on 16, 57 on 16 | 0 | pulses=448, stop 40006 ms, seen 40007 ms |
| `h2-priority` | `GPUTRONIC_GUEST_PRIORITY=1` | 1.5035, 1.420, 1.562, 32 of 32 | 1.5046, 1.420, 1.562, 32 of 32 | 54 on 6, 57 on 26 | 0 | pulses=448, stop 40005 ms, seen 40006 ms |
| `h3-nonblock` | `GPUTRONIC_GOLD_NONBLOCK=1` | 1.5117, 1.429, 1.616, 30 of 32 | 1.5103, 1.429, 1.616, 30 of 32 | 54 on 3, 55 on 5, 57 on 9, 58 on 15 | 0 | pulses=448, stop 40006 ms, seen 40006 ms |

The two samples outside the band on `h1-connections` are `t=13` (1.597) and `t=27` (1.581). On `h3-nonblock` they are `t=14` (1.616) and `t=16` (1.584 on Z_raw, 1.583 on Z_hat).

`h2-priority` printed `stream priority=-5 (highest; range -5..0)`. `h2-stream-first` printed `stream created before the Gold launch`. `h3-nonblock` printed `persistent kernel on a non-blocking stream`. None of the four changed the guest rate.

The 54 to 57 step is a refill, not the guest. In `h2-stream-first.txt`, between `t=8` (`n=54`) and `t=9` (`n=57`), the log has `[SPAWN] base=54 n=3`. That refill had not published an SM id 2 ms later (`ready=0`). The last spawn in that file is between `t=8` and `t=9` (`count=15`). From `t=10` through `t=40` there is no `[SPAWN]` line, and the guest stays at 0. A pending refill is not what held the guest for the rest of the run.

`dmon -s pucm -d 1` and `clocks_event_reasons.active` (every sample `0x0000000000000000`):

| Run | dmon samples | Power | Die |
| --- | --- | --- | --- |
| `h1-connections` | 42 | 61–211 W | 52–66 °C |
| `h2-stream-first` | 41 | 37–205 W | 43–57 °C |
| `h2-priority` | 41 | 60–210 W | 48–61 °C |
| `h3-nonblock` | 41 | 40–206 W | 43–57 °C |

No run was stopped for temperature. Plain `nvidia-smi` before the clean checks was 43 °C and 38 W (`nvidia-smi-start.txt`). After the checks it was 50 °C and 42 W (`nvidia-smi-end.txt`).

The listed tries are ruled out. A shared hardware queue at the default connection count is ruled out by `CUDA_DEVICE_MAX_CONNECTIONS=32`. Stream order and the highest stream priority are ruled out by the two `h2` runs. The refill spawns are ruled out by the quiet stretch in `h2-stream-first`. A hidden mapped counter is ruled out because the entry stamp and the first pulse are one pass apart and the host sees both at the stop. Launching Gold itself on a non-blocking stream, which was not one of the listed tries, did not change it either.

The controller's view is still incomplete while a second kernel is resident. The guest's 448 pulses show up only as Gold stops.

`GPUTRONIC_GUEST_STREAM_FIRST`, `GPUTRONIC_GUEST_PRIORITY`, and `GPUTRONIC_GOLD_NONBLOCK` are test switches. Unset, the guest stream is still created after Gold's launch, at the default priority, and Gold's kernel stays on the default stream.

## 3. The integral under the 1.2 load

Idle-exit was left unset. The guest was left unset. `GPUTRONIC_NO_CSV` was left unset. The plant is the same full-pipe set as the tenth-pass load step (`GPUTRONIC_PEAK_ACC=32`, `GPUTRONIC_PEAK_ROUNDS=512`, `GPUTRONIC_THREADS=512`, `GPUTRONIC_PULSE_LOCAL=1`, `GPUTRONIC_CTRL_CACHE=1`). `GPUTRONIC_ICLAMP_SMS` was set to 6, then 9, then 12, one run each. These are test settings. The compiled clamp is the one the binary already prints when that variable is unset. A wider clamp was not compiled in. 15 was not run: at 9 and at 12 the integral does not reach the rail, so the condition for going wider is not met.

This is a synthetic load. On this kernel `work_scale_milli` skips pulses and the FMAs keep running, so the counter reports a slower job while the SMs do the same work. The schedule is `logs/2026-10-17_tenth-pass-continued/steps.txt`, copied from the tenth pass, with t = 0 at the end of calibration:

```
60000  1.2
120000 1.0
180000 1.2
240000 1.0
300000 1.2
360000 1.0
```

Command, with `GPUTRONIC_ICLAMP_SMS` set to 6, 9, or 12:

```
GPUTRONIC_PEAK_ACC=32 GPUTRONIC_PEAK_ROUNDS=512 GPUTRONIC_THREADS=512 \
GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 \
GPUTRONIC_TRACE_BIN_US=1000 GPUTRONIC_ICLAMP_SMS=6 \
timeout --signal=INT --kill-after=30s 480 \
  ./build/gputronic_gold loadfile logs/2026-10-17_tenth-pass-continued/steps.txt 420
```

Each run returned 0. No run was stopped for temperature. The clamp lines are:

- 6 SMs: `integral clamp=2.053 (iclamp_sms, max integral correction 6.00 SMs of 82)`. Log `clamp-6.txt`.
- 9 SMs: `integral clamp=3.439 (iclamp_sms, max integral correction 9.00 SMs of 82)`. Log `clamp-9.txt`.
- 12 SMs: `integral clamp=4.729 (iclamp_sms, max integral correction 12.00 SMs of 82)`. Log `clamp-12.txt`.

The `[CTRL]` line on each of those logs is `target=1.50 Kp=0.550 Ki=0.0800 Kd=0.0000 deadband=0.080`. The launch line is 82 blocks, 512 threads.

The 1 ms trace is the loadfile sampler with `GPUTRONIC_TRACE_BIN_US=1000`. The step table uses that file. The program writes it as `results/gputronic_gold_loadfile_10ms.csv`. The committed copies are:

- `clamp-6-loadfile-1ms.csv`, 52,234,760 bytes, 419,964 data rows, t = 0.00 to t = 419,999.17 ms
- `clamp-9-loadfile-1ms.csv`, 52,089,716 bytes, 419,962 data rows, t = 0.00 to t = 419,999.13 ms
- `clamp-12-loadfile-1ms.csv`, 52,160,100 bytes, 419,983 data rows, t = 0.00 to t = 419,999.19 ms

The control thread also wrote `results/gputronic_gold_load.csv` at the 10 µs tick. That file was deleted after each run and is not committed.

A tick is on the integral clamp when `|integral| >= 0.999 * clamp`, the test that increments `st_rail`. The clamp share is `rail_ticks / pi_ticks` summed over the 1 ms bins. The band test is on `z_hat`. "Enters and stays" means the first 1 ms bin where `|z_hat - 1.5| <= 0.08` and every later bin in that hold is also inside. "Ends outside" means the hold's last bin is outside that band, so there is no such tail. A limit cycle means the last 30 s uses exactly two SM counts, each seen at least five times, with the count changing at least six times. None of the eighteen holds meets that.

The same definitions on the tenth-pass file reproduce that pass's table. On that run the printed clamp was 3.28 SMs, the last 30 s of each 1.2 hold was on the clamp for every control tick, 55.0–57.3% of those bins were inside the band, and the count ended at 61. The returns to 1.0 were 96.1–97.5% inside over the last 30 s, with none of those ticks on the clamp, and the lowest `z_hat` on each return was about 1.36.

### 6 SMs

`dmon -s pucm -d 1` has 421 samples, power 42–208 W (mean 184.5 W, the 42 W sample included), die 47–63 °C. Every `clocks_event_reasons.active` sample is 0. The whole-run counter is `integral_on_rail=1404459` out of `pi_ticks=6572896`, and `active_sms=42..63`. The sum of `rail_ticks` in the 1 ms file is 1,404,459, the same count. That line mixes calibration with both scales. The last 30 s rows below are the hold figures.

On the first two 1.2 holds the last 30 s is on the clamp for every control tick, and the in-band share is 35.3% and 90.6%. On the third, the integral reaches the clamp at t = 353,442.02 ms (53.442 s into the hold, `z_hat` 1.64058). Every rail tick in that last 30 s falls from that bin to the end of the hold: 102,450 of 102,461 control ticks. That is the 22% in the table. The last 30 s of each return to 1.0 has no ticks on the clamp. The lowest `z_hat` on those returns is 1.31809, 1.32526, and 1.32384. The staying tail on each return is 65 ms, 99 ms, and 43 ms.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end of hold | SM range in the hold | Last 30 s inside the band | Last 30 s on the clamp | SM changes in the last 30 s | Counts in the last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | 1.2 | 0.1 ms | 59.959 s | 40 ms | 1.80896 | 1.46467 | 54 | 60 | 54–63 | 10,582 / 29,993 (35.3%) | 467,849 / 467,849 (100%) | 1,826 | 60, 61, 62, 63 |
| 120 s | 1.0 | 15.1 ms | 59.934 s | 65 ms | 1.76442 | 1.31809 | 60 | 53 | 52–60 | 27,488 / 30,001 (91.6%) | 0 / 470,483 (0%) | 184 | 52, 53, 56, 57, 58, 60 |
| 180 s | 1.2 | 15.1 ms | 59.903 s | 96 ms | 1.77002 | 1.44905 | 53 | 60 | 53–62 | 27,191 / 30,001 (90.6%) | 471,705 / 471,705 (100%) | 102 | 60, 61, 62 |
| 240 s | 1.0 | 16.1 ms | 59.900 s | 99 ms | 1.62436 | 1.32526 | 60 | 51 | 51–60 | 27,996 / 30,001 (93.3%) | 0 / 469,524 (0%) | 174 | 51, 52, 55, 56 |
| 300 s | 1.2 | 21.2 ms | 59.874 s | 0.125 s | 1.86045 | 1.44910 | 51 | 60 | 51–62 | 26,409 / 30,001 (88.0%) | 102,450 / 469,834 (22%) | 270 | 59, 60, 61, 62 |
| 360 s | 1.0 | 27.1 ms | 59.956 s | 43 ms | 1.62935 | 1.32384 | 60 | 51 | 51–60 | 27,302 / 30,000 (91.0%) | 0 / 468,952 (0%) | 77 | 51, 52, 54, 55 |

Over the whole hold, the share of 1 ms bins inside the band is 27.5%, 80.3%, 78.6%, 71.4%, 74.1%, and 70.2%, in step order. The count at the end of each 1.2 hold is 60.

### 9 SMs

`dmon` has 421 samples, power 59–216 W (mean 189.9 W), die 51–64 °C. Every `clocks_event_reasons.active` sample is 0. The whole-run counter is `integral_on_rail=0` out of `pi_ticks=6822467`, and `active_sms=48..67`. The sum of `rail_ticks` in the 1 ms file is 0.

The integral is off the rail on every hold. The last 30 s of the 1.2 holds is 83.0%, 82.0%, and 82.7% inside the band, and the count ends at 65, 65, and 66. Four holds end outside the band, all on the high side, so they have no staying tail:

- 60 s, last 10.0 ms, last `z_hat` 1.58623, 65 SMs
- 180 s, last 11.1 ms, last `z_hat` 1.61323, 65 SMs
- 240 s, last 6.1 ms, last `z_hat` 1.61486, 59 SMs
- 300 s, last 9.0 ms, last `z_hat` 1.64271, 66 SMs

The two returns that do end inside stay there for 90 ms and 18 ms. The lowest `z_hat` on the three returns is 1.31365, 1.31242, and 1.31281. The highest `z_hat` on the first two 1.2 holds is 1.81009 at 20.258 s into the hold and 1.81334 at 19.510 s into the hold.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end of hold | SM range in the hold | Last 30 s inside the band | Last 30 s on the clamp | SM changes in the last 30 s | Counts in the last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | 1.2 | 4.1 ms | ends outside | — | 1.81009 | 1.45409 | 55 | 65 | 55–67 | 24,912 / 29,997 (83.0%) | 0 / 482,865 (0%) | 361 | 63, 64, 65, 66 |
| 120 s | 1.0 | 0.2 ms | 59.909 s | 90 ms | 1.65328 | 1.31365 | 65 | 56 | 55–65 | 26,228 / 30,001 (87.4%) | 0 / 487,987 (0%) | 66 | 55, 56, 59, 60 |
| 180 s | 1.2 | 13.2 ms | ends outside | — | 1.81334 | 1.45578 | 56 | 65 | 56–67 | 24,611 / 29,998 (82.0%) | 0 / 471,991 (0%) | 384 | 63, 64, 65, 66 |
| 240 s | 1.0 | 0.0 ms | ends outside | — | 1.61732 | 1.31242 | 65 | 59 | 55–65 | 25,869 / 29,995 (86.2%) | 0 / 504,390 (0%) | 99 | 55, 56, 59, 60 |
| 300 s | 1.2 | 0.3 ms | ends outside | — | 1.77352 | 1.45284 | 59 | 66 | 58–67 | 24,799 / 30,000 (82.7%) | 0 / 485,647 (0%) | 332 | 63, 64, 65, 66, 67 |
| 360 s | 1.0 | 0.2 ms | 59.981 s | 18 ms | 1.65957 | 1.31281 | 66 | 58 | 55–66 | 25,494 / 29,997 (85.0%) | 0 / 518,764 (0%) | 154 | 55, 56, 58, 59, 60 |

Over the whole hold, the share of 1 ms bins inside the band is 64.9%, 64.3%, 66.8%, 63.5%, 66.7%, and 62.9%, in step order.

### 12 SMs

`dmon` has 421 samples, power 59–218 W (mean 189.5 W), die 51–64 °C. Every `clocks_event_reasons.active` sample is 0. The whole-run counter is `integral_on_rail=0` out of `pi_ticks=6545569`, and `active_sms=47..70`. The sum of `rail_ticks` in the 1 ms file is 0.

The integral is off the rail on every hold. The last 30 s of the 1.2 holds is 81.0%, 82.2%, and 81.8% inside the band, and the count ends at 65. The highest `z_hat` in the run is 1.97583, 23.715 s into the 180 s hold, with the count ranging up to 70 on that hold. The returns to 1.0 end at 55 SMs. Their lowest `z_hat` is 1.30843, 1.31385, and 1.31253, and their last-30 s in-band shares are 84.4%, 86.0%, and 85.5%. The staying tails on those returns are 0.764 s, 0.157 s, and 33 ms.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end of hold | SM range in the hold | Last 30 s inside the band | Last 30 s on the clamp | SM changes in the last 30 s | Counts in the last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | 1.2 | 26.1 ms | 59.876 s | 0.123 s | 1.90096 | 1.44719 | 55 | 65 | 55–66 | 24,308 / 30,001 (81.0%) | 0 / 464,628 (0%) | 433 | 63, 64, 65, 66 |
| 120 s | 1.0 | 21.1 ms | 59.235 s | 0.764 s | 1.65323 | 1.30843 | 65 | 55 | 55–65 | 25,307 / 30,001 (84.4%) | 0 / 465,785 (0%) | 149 | 55, 56, 58, 59, 60 |
| 180 s | 1.2 | 23.2 ms | 59.826 s | 0.173 s | 1.97583 | 1.44415 | 55 | 65 | 55–70 | 24,672 / 29,999 (82.2%) | 0 / 466,402 (0%) | 398 | 63, 64, 65, 66, 67 |
| 240 s | 1.0 | 11.1 ms | 59.842 s | 0.157 s | 1.61268 | 1.31385 | 65 | 55 | 55–65 | 25,807 / 30,000 (86.0%) | 0 / 474,811 (0%) | 140 | 55, 56, 58, 59, 60 |
| 300 s | 1.2 | 10.2 ms | 59.970 s | 29 ms | 1.78579 | 1.45302 | 55 | 65 | 55–66 | 24,533 / 29,996 (81.8%) | 0 / 467,598 (0%) | 396 | 63, 64, 65, 66 |
| 360 s | 1.0 | 11.1 ms | 59.966 s | 33 ms | 1.61327 | 1.31253 | 65 | 55 | 55–65 | 25,654 / 30,000 (85.5%) | 0 / 469,455 (0%) | 143 | 55, 56, 58, 59 |

Over the whole hold, the share of 1 ms bins inside the band is 62.9%, 62.7%, 66.3%, 62.5%, 66.1%, and 62.5%, in step order.

Read against the tenth-pass load on this same plant: at 6 SMs the loaded holds still finish on the clamp, and the first 1.2 hold's last 30 s is inside the band less often than at 3.28 SMs (35.3% against 55.0%). At 9 and at 12 the integral stays off the clamp, the 1.2 last-30 s share rises to about 81–83%, and the count ends near 65 rather than 61. The returns to 1.0 get worse as the clamp widens. The last-30 s in-band share falls from 96–98% at the compiled clamp, to 91–93% at 6 SMs, 85–87% at 9, and 84–86% at 12. The undershoot goes from about 1.36 to about 1.31. The plots are `clamp-6-zhat.svg`, `clamp-9-zhat.svg`, and `clamp-12-zhat.svg`. `plot_clamp.py` drew them from the 1 ms files, one point about every 100 ms.

Plain `nvidia-smi` at the end of the three runs was 51 °C and 59.77 W (`nvidia-smi-clamp-end.txt`).

## Checks

Switches unset, on this source. The `[CHECK] z_mean` lines are:

- Gold `z_mean=1.468` (`check-gold.txt`). `=== GOLD GATE: dyno=PASS closedloop=PASS ===`
- Transfer `z_mean=1.497` (`check-transfer.txt`). `=== TRANSFER GATE: dyno=PASS closedloop=PASS ===`
- Episode `z_mean=1.443` (`check-episode.txt`). `=== EPISODE GATE: dyno=PASS closedloop=PASS ===`

The unset Gold lines are `[GPU] ... launching 82 ... thr/block=64` and `[CTRL] Gold | target=1.50 Kp=0.550 Ki=0.0800 Kd=0.0000 deadband=0.080`.
