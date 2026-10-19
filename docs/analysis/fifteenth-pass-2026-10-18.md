# Fifteenth pass, 2026-10-18

Branch `test-results/2026-10-18_fifteenth-pass`, started from `74601ce`. The switch commit is `b8089c9`. Logs in `logs/2026-10-18_fifteenth-pass/`. The card is an RTX 5080, driver 595.99.02, built with CUDA 12.8, `sm_120`. `nvidia-smi` reports driver CUDA 13.2. The build is 12.8. The sweep ran from 21:39:58 to 23:01:45 PDT. No run was stopped for temperature. The hottest die sample is 68 °C, on `grid-t1024`. Every `clocks_event_reasons.active` sample in the display and flag files is 0.

The directive is on `origin/controller-fixes` at `fb3405f` (`docs/directives/fifteenth-pass-2026-10-18.md`). That branch is not merged. The two switches that were not added, and the way item 5 names the 49 million group, are argued at the end.

The fourteenth pass left the full-pipe plant at 46.98–47.26 million and did not land in the thirteenth quiet group, 49.1–49.6 million. This session has both. The step is one launch, and it stayed for the rest of the session. A round count, a thread count, and a chain count did not produce it.

## What this pass changed

One test switch, `GPUTRONIC_OPEN_SCHED`. It is a path. `fullwidth` and `noise` read that file as a load schedule, `t_ms scale`, with an optional third column for the open-loop SM count. Unset, both modes pass no points, the scale stays 1.0, the launch count is unchanged, and the run prints nothing. A set file with no usable points exits before the kernel starts. A count below 2 is raised to the floor. A count at or above the launch means every block. Closed-loop `loadfile` ignores the third column. The help line is in `help.txt`.

`unset-home.txt` has no `[SCHED]` line, no `[RATE]` line, and no `[TARGET]` line. Its `[CTRL]` line is target 1.00, Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, kalman 1, two-sided. The launch leaves out SMs 82 and 83. Its `rate_ref` is 585,557.2. That is the stock plant, with the full-pipe switches off, and it is not a point in the sweep.

The smoke launch is not a sweep point. `[SCHED]` reports 3 points. On the 1 ms trace the three regions are 82 SMs at scale 1.0, 40 SMs at scale 1.2, and 82 SMs at scale 1.0.

Kp, Ki, Kd, the deadband, the 82-block launch, floor 2, and the compiled clamp were not changed. `GPUTRONIC_ICLAMP_SMS=15` and `GPUTRONIC_TARGET_Z=1.5` are the closed-loop runs only. `GPUTRONIC_RATE_WIN_US` stayed unset. The function frame and the copy at the end of `README.md` name the schedule switch in the same words.

Two optional switches were not added. Rounds already floor at 1, and the zero-work end is the idle GPU, which was measured. The 1 ms pulse publish was already on every point. The lag measured below does not follow that period.

## How the sweep was run

Full-pipe switches on every sweep launch unless the point changes one: `GPUTRONIC_PEAK_ACC=32`, `GPUTRONIC_PEAK_ROUNDS=512`, `GPUTRONIC_THREADS=512`, `GPUTRONIC_PULSE_LOCAL=1`, `GPUTRONIC_CTRL_CACHE=1`, `GPUTRONIC_TRACE_BIN_US=1000`, `GPUTRONIC_WORK_LOAD=1`. Idle-exit, relaunch, quiet idle, idle work, guest, the tensor sensor, and `GPUTRONIC_OS_RESERVE` were unset.

Open-loop sweeps are one launch with `GPUTRONIC_OPEN_SCHED` set, bracketed by the base point: acc 32, rounds 512, 512 threads, 82 SMs, scale 1.0. Threads are fixed for a launch, so the thread sweep is one `fullwidth` per count, with a base launch every eight points. The coarse grid opens and closes on rounds 512 at that launch's thread count. Those brackets are not the 512-thread base. `base-grid-0` and `base-grid-1` are.

Sweep points are 10 s. The first 2 s of each point are dropped. Ramp bins are 0.025 of scale and are not trimmed again inside the bin. The opening base already covers the first 2 s of the launch. Stairs are 10 s at each 0.025 step. Closed-loop holds are the whole window. Item 5 is 15 s at scale 1.0, 30 s at 1.2, 30 s at 1.0.

Round counts follow `(PEAK_ROUNDS × scale_milli + 500) / 1000`, floor 1, cap 4096. A ramp bin's printed round count is that formula at the median scale inside the bin, so a bin labelled 0.800 can print 416 rather than 410. The stairs are the exact formula integers.

Counted ops/s = pulse rate × `PEAK_ACC` × rounds × 64. TFLOPS is that over 1e12. `rate_ref` is compared only across launches with the same acc, rounds, and threads. Everything else is compared in counted ops/s and in ops/s per SM.

The band is `|z_hat − target| ≤ 0.08` after rounding the absolute error to five decimal places. "Enters and stays" is the first in-band sample after the last out-of-band sample. A limit cycle is exactly two SM counts, each at least five times, with at least six changes. A hit where one count has almost every sample is recorded and is not called a sustained oscillation.

The wave uses the fourteenth method, in `numbers.py`: the 1 ms series downsampled to 4 ms, autocorrelation at 128, 136, and 144 ms, the strongest lag from 50 to 400 ms, and the 7.25–7.45 Hz share of the 0.5–60 Hz power. If the strongest lag is more than 4 ms off 136, the share at that period is reported too. Open-loop `z_raw` SD is the analysis window and the median of the 500 ms stretches. Closed-loop wave figures quoted as fixed-count use stretches of at least 500 ms. The narrow-band share on these logs is about 0 to 2 percent. The autocorrelation is the period.

The per-tick file is `*-tick-cut.csv.gz`. The cut keeps `time_s` ≤ 3 s and every later row whose `z_raw`, `rate_ema`, or `active_sms` changes. `cut-manifest.txt` has the raw, kept, and gzip sizes. The raw per-tick files were deleted after the gzip read back. The 1 ms traces are gzipped in this commit because the session was 447 MB uncompressed. `GPUTRONIC_NO_CSV` stayed unset.

Move-to-response uses the per-tick cut. The trace clock is zeroed at the end of calibration. The tick anchor is the last window close before 4 s with the count at 2. On these four launches that anchor is 2.996–2.999 s. A move counts only when the new count is still in force at the next window close. The first Z change is the next close whose `z_raw` differs by at least 5e-6. Half the change is halfway from that `z_raw` toward the extreme in the expected direction over the next 80 ms.

`display_active` was read once a second for the whole session. The process list was taken at the start and end of every launch.

## 1. Zero, the floor, and the ceiling

### Zero

No GPUtronic process. `zero.display` has 30 samples, 21:39:58.752 to 21:40:27.821 PDT. `display_active` is Enabled. P-state is P3. GPU utilization is 2 to 10 percent, median 8. The SM clock is 637 to 1537 MHz, median 1020. Memory clock is 7001 MHz on every sample. Power is 32.11 to 42.20 W, median 37.30. There is no `rate_ref`.

### Floor

`PEAK_ACC=4`, rounds 1, 32 threads. Both launches are open loop, `sm_moves=0`, `integral_on_rail=0`.

| Launch | SMs | rate_ref | Per-SM pulse rate | Counted TFLOPS | z_raw SD | 500 ms SD | ac at 136 ms | Lag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| floor-fw | 82 | 149,950,800 | 1,845,053 | 0.0387 | 0.03561 | 0.02772 | 0.413 | 136 ms |
| floor-n2 | 2 | 161,769,776 | 1,894,472 | 0.0010 | 0.00000 | 0.00000 | none | none |

The pulse rate is high because each pulse is 4 FMAs per thread. The counted rate is the bottom of the sweep. At 82 SMs the 136 ms coefficient is already present. At 2 SMs `z_raw` does not move at the printed precision, so there is no lag to report. `floor-fw` drew a median 86.5 W at 2895 MHz. `floor-n2` drew 81.2 W. Display was Enabled. The reason word is 0.

### Ceiling

The highest counted rate in the round, thread, chain, SM, and grid sweeps is `grid-t256` at 2048 rounds, 256 threads, acc 32, 82 SMs: 53.7420 TFLOPS over the analysis window, per-SM pulse rate 156,257. The next grid point, 1024 threads at 2048 rounds, is 53.4064 TFLOPS. The 30 s open-loop hold at the winning setting, `ceil-hold`, measures 53.8927 TFLOPS, per-SM pulse rate 156,696, median SM clock 2835 MHz, median power 223.6 W, die up to 64 °C, reason word 0, display Disabled. The count stays at 82. `integral_updates=0`. `z_raw` SD is 0.01761, and the median 500 ms SD is 0.01272. Autocorrelation at 136 ms is 0.047. The strongest lag is 48 ms, and the share at that period is 0.0019.

`ceil-accum` is the same hold with `GPUTRONIC_SM_ACCUM=1`. The pulse formula on that window is 53.3875 TFLOPS, which is 6.511×10^11 ops/s per SM. The per-SM counter, over 27.999 s after the first 2 s, has median 6.486757×10^11 ops/s. Across 82 SMs that is 53.19 TFLOPS. The two readings differ by 0.4 percent. The counter's per-SM rates span 1.000684, max over min. The raw end-of-window totals span 1.038, because each SM's counter includes whatever it had already added before the window. The windowed rate is the comparison, and the SMs agree. `ceil-work` sets `GPUTRONIC_WORK_COUNTER=1`. The 29 printed ratios of `pulses × 16 / batches` run from 0.99977 to 1.00042. Both extra switches add traffic. The formula and the counter agree, so the sustained ceiling is reported as 53.19 TFLOPS at acc 32, 2048 rounds, 256 threads, 82 SMs.

Pass time at that hold is 51 µs per warp. That is not 48 ms and not 136 ms.

## 2. Rounds

Acc 32, 512 threads, 82 SMs, open loop, ascending and descending. Every listed round count is a row in `sweep-points.csv`. The plots are `plot-ps-rounds.svg`, `plot-ops-rounds.svg`, `plot-ac136-rounds.svg`, and `plot-lag-rounds.svg`.

These launches sit before the step in section 3. Their `rate_ref` values run from 46,793,012 to 47,423,460, except `r2-asc-1` at 43,895,688. `r2-asc-0` opens at 581,033 pulses/s per SM and closes at 554,110. The next launch is the 43.9 million one. The launch after that is back in the 46.8–47.2 million range. That dip is a separate cluster from the 49 million group, and it did not stay.

On a clean pair, `r2-asc-4` (`rate_ref` 47,113,024) and `r2-desc-0` (`rate_ref` 47,016,408):

| Rounds | Asc per-SM | Desc per-SM | Asc ops/s per SM | Asc ac136 | Asc lag | Desc ac136 | Desc lag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512, open | 579,103 | 580,989 | 0.607×10^12 | 0.461 | 272 ms | 0.572 | 136 ms |
| 1024 | 289,676 | 289,681 | 0.607×10^12 | 0.581 | 136 ms | 0.542 | 136 ms |
| 2048 | 143,742 | 143,811 | 0.603×10^12 | 0.538 | 136 ms | 0.558 | 136 ms |
| 4096 | 72,053 | 72,053 | 0.604×10^12 | 0.561 | 136 ms | 0.611 | 136 ms |
| 512, close | 580,000 | 580,160 | 0.608×10^12 | 0.576 | 136 ms | 0.548 | 136 ms |

The short end, from `r2-asc-0` and `r2-desc-4`, has no measurable wave. `z_raw` SD prints 0.00000.

| Rounds | Asc per-SM | Desc per-SM | Ops/s per SM |
| --- | --- | --- | --- |
| 1 | 20,369,066 | 20,375,873 | 0.042×10^12 |
| 64 | 3,812,911 | 3,798,343 | 0.500×10^12 |
| 96 | 2,710,941 | 2,694,677 | 0.530×10^12 |

Where the coefficient exists, 84 item-2 points, ac136 runs from −0.004 to 0.662, median 0.551. The lag is 136 ms, or 272 ms while ac136 stays near 0.5. One point at 128 rounds has lag 396 ms and ac136 −0.004, with SD 0.00020. From 192 rounds through 4096 the 136 ms coefficient is the one that repeats.

Counted ops/s per SM rises from 0.042×10^12 at 1 round and is flat near 0.60×10^12 from a few hundred rounds through 4096. No non-512 point in this sweep reaches the base ops/s per SM times 601/580. The base brackets of these launches stay near 577,000–582,000 per SM, apart from the one dip above. None of them steps up to 598,000.

Pass time at 512 rounds is 28 µs per warp. At 4096 rounds it is about 220 µs. At 1 round it is under 1 µs. The lag stays at 136 ms across that range.

## 3. Occupancy

### Threads

One launch per thread count, rounds 512, before the step. `rate_ref` is comparable only at 512 threads. `th-0512` is 47,148,248. The neighbouring base launches are 46.83 to 47.17 million. Several other thread counts land in the 43–46 million range. Their ops/s is lower with that plant, so the thread curve is not one plant.

At the launches whose `rate_ref` is near 47 million, counted throughput is already on the plateau by 256 threads: `th-0256` is 49.15 TFLOPS, `th-0512` is 49.87, `th-1024` is 50.21. The short blocks are below it: 32 threads 9.54 TFLOPS, 64 threads 19.03, 128 threads 38.17. ac136 on these rows is 0.45 to 0.64, and the lag is 136 ms, or 272 ms on a few rows with the 136 ms coefficient still about 0.5.

The coarse grid is after the step, so it is the 49 million plant at every thread count that uses rounds 512. `rate_ref` still cannot be compared across thread counts. Counted throughput at 512 rounds: 64 threads 20.55 TFLOPS, 128 threads 40.81, 256 threads 52.12, 512 threads 51.88, 1024 threads 52.36. The plateau starts at 256 threads. The same grid at 2048 rounds is where the ceiling came from. On this plant ac136 is about −0.1 to 0.2 and the strongest lag is 48 ms.

The device allows 1536 threads per SM and 1024 per block. The acc-32 kernel uses 63 registers, so a 1024-thread block fits in the 65536-register SM. The acc-16 kernel uses 56, acc 8 and acc 4 use 48, the dependent peak uses 72, and the unset kernel uses 40. Shared memory is 2240 bytes. One block per SM was the launch. A second block was not added.

### Chains

Acc 4, 8, 16, and 32, each at 512 rounds and at equal FMAs per thread (rounds 4096, 2048, 1024, and 512). Twice, the second time reversed. This is before the step. ac136 is 0.46 to 0.63 and the lag is 136 ms on every one of these rows.

Equal FMAs per thread are not equal throughput. Acc 4 at 4096 rounds is 17.3 TFLOPS. Acc 8 at 2048 is 30.8. Acc 16 at 1024 is 44.9 to 45.9. Acc 32 at 512 is 49.7 to 49.9. More independent chains raise the counted rate. They do not move the 512-round, 512-thread base. Those brackets are 46.75 to 47.20 million.

### SM count

The staircase is open loop at the base work, ascending and descending, 10 s at each count. Per-SM pulse rate on the way down through `sms-desc-2`, before the closing base, is 584,225 to 588,088 from 2 SMs to 24 SMs. On the way up it is 576,675 to 587,784 from 2 SMs to 80 SMs, including 57. The rate per SM does not climb with N. Total ops/s does, because the pulse rate is N times that.

At 2, 4, and 6 SMs, `z_raw` SD prints 0 and there is no lag. From 8 SMs upward, ac136 is about 0.4 to 0.57 and the lag is 136 ms, or 272 ms with ac136 still near 0.5.

`sms-desc-2` is the step. It calibrates at 46,699,180. The opening base, after the 2 s drop, is 578,669 per SM, ac136 0.558, lag 136 ms. The interior counts stay near 585,000 per SM. The closing base is 602,128 per SM. That is 49.37 million if multiplied back by 82. ac136 on that closing base is still 0.513, and the strongest lag has moved to 48 ms. The launch's own `rate_ref` is the opening plant. The hold at the end is the other one.

The sample that records `display_active` changing from Enabled to Disabled is 22:32:00.846 PDT, inside that closing base. Power on the session log steps from 128 W to 218 W at 22:31:57, one sample before the display change, which is the return from a small SM count to 82. The reason word stays 0 on both sides. The display change is logged. It is not a cause this pass can assign. The thirteenth session saw Disp.A go off and the fast plant stay. This session has the timestamp, and the same refusal to promote it.

Every later base bracket is the high group. Fifteen base rows are at or above 598,000 per SM. The opening and closing bases of the ramps and the stairs are 603,079 to 603,393. `base-grid-0`, the next launch, calibrates at 49,303,052. `base-grid-1` calibrates at 49,301,496. The high reading stayed through 23:01.

## 4. Ramps and stairs

These launches calibrate at 49.28 to 49.43 million. Open-loop SM changes inside the ramp bins are 0.

The stairs, exact formula rounds, up against down. The largest gap is 563 rounds: 545,068 per SM on the way up, 547,656 on the way down, 0.47 percent. The smallest is 0.01 percent. The opening and closing bases are 603,079 to 603,393 per SM, ac136 about −0.09 to 0.15, strongest lag 48 ms.

The 60 s open-loop ramp, matched 0.025 bins, stays inside 1.3 percent. The worst bin is scale 1.150, 517,726 against 511,200. The 30 s ramp is the same shape. Counted ops sit near 52 TFLOPS across the ramp, because the pulse rate falls as the rounds rise. The base after each ramp is the same group as the base before it.

The closed-loop 60 s ramp, clamp 15, target 1.5, calibrates at 49,297,172. `integral_on_rail=0` of 2,198,558 PI ticks. `sm_moves=2226` for the launch. The opening 10 s is 97.4 percent in band and ends at 55 SMs. The closing 10 s is 87.9 percent in band and ends at 52 SMs, with the integral at 2.08. Both brackets are about 603,500–604,000 per SM.

The count does not retrace. On the way up the median count goes from 51 SMs at scale 0.800 to 61 at 1.200, and the in-band share is under 1 percent at the light end, above 90 percent from about 0.875 to 1.000, and 0 from 1.075 through 1.200. On the way down the count stays near 58–61 through the heavy half and is still 49 at scale 0.800. The in-band share on the way down is high from 1.025 to 1.150 and low again at the light end. The per-SM pulse rate at a matched scale still agrees: 749,818 against 741,894 at the 0.800 bin, and 507,840 against 507,422 at 1.175. The plant rate has no memory of the direction. The count path does.

## 5. Closed loop on the high group

The base workload itself reached the high group, so the four confirmations are that workload: acc 32, rounds 512, 512 threads, clamp 15, target 1.5. There is no closed-loop confirmation whose calibration is in the 46.98–47.26 million band. By item 5 every base-settings launch was already high, and a high calibration was not retried. The same-session 47 million baselines are the open-loop brackets before `sms-desc-2` closed.

| Launch | rate_ref | STATS sm_moves | integral_on_rail |
| --- | --- | --- | --- |
| conf-a1 | 49,561,284 | 839 | 0 |
| conf-b1 | 49,307,692 | 944 | 0 |
| conf-a2 | 49,308,268 | 795 | 0 |
| conf-b2 | 49,061,892 | 913 | 0 |

`conf-b2` is 49.06 million, just under the 49.1 edge. Per SM at calibration it is 598,316. The other three are inside 49.1–49.6.

Per-SM pulse rate in 10 s chunks. The load is scale 1.2, so its pulse rate sits near 512/614 of the return when the plant is unchanged.

| Launch | Pre-roll | Load | Return |
| --- | --- | --- | --- |
| conf-a1 | 606,913 606,555 | 506,518 503,651 502,445 | 598,630 605,672 603,481 |
| conf-b1 | 591,625 594,151 | 490,893 491,518 493,902 | 594,363 593,174 592,809 |
| conf-a2 | 603,859 603,733 | 497,802 503,253 502,660 | 604,561 603,242 603,544 |
| conf-b2 | 601,021 580,772 | 501,789 500,475 504,679 | 603,593 603,298 603,123 |

`conf-b2`'s pre-roll steps from 601,021 to 580,772. That hold is not counted. Its load and return stay flat, and the load is 512/614 of the 603k return, so those two holds stay on the high plant. `conf-b1`'s return sits near 593k, flat, under the 598k line and above the 581k line. It is reported as its own number. It does not step inside the hold.

In band, whole window and last 10 s. The tick-file move count is the whole hold. The last-10 s change count is the 1 ms trace.

| Hold | In band | Last 10 s | SM range | End | Tick moves | Last 10 s changes | Limit, whole | Limit, last 10 s | Enters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a1 pre | 97.9% | 97.9% | 48–62 | 55 | 42 | 18 | no | yes | stays |
| a1 load | 39.6% | 90.0% | 55–66 | 63 | 506 | 35 | no | no | stays, tail short |
| a1 return | 60.9% | 98.5% | 57–66 | 57 | 291 | 18 | no | no | stays |
| b1 pre | 98.3% | 99.0% | 47–82 | 57 | 77 | 13 | no | yes | stays |
| b1 load | 36.3% | 88.0% | 57–66 | 64 | 621 | 90 | no | no | stays, tail short |
| b1 return | 67.3% | 98.8% | 58–64 | 58 | 246 | 8 | no | yes | stays |
| a2 pre | 97.5% | 97.8% | 47–63 | 58 | 67 | 31 | no | no | stays |
| a2 load | 47.6% | 94.4% | 58–67 | 65 | 467 | 37 | no | no | ends outside |
| a2 return | 68.3% | 97.5% | 57–64 | 57 | 261 | 18 | no | no | stays |
| b2 load | 41.5% | 77.7% | 54–67 | 63 | 547 | 129 | no | no | stays, tail short |
| b2 return | 68.4% | 98.9% | 57–63 | 57 | 293 | 2 | no | no | stays |

The three last-10 s limit-cycle hits are one count with a few samples of a second count. `conf-b1`'s return last 10 s is 9,858 samples at 58 and 142 at 61. That meets the letter. It is not a sustained two-count oscillation. `conf-a2`'s load ends at `z_hat` 1.68977, outside the band.

On the loads, the strongest lag of Z and of the count is 72 ms. ac136 is still 0.59 to 0.73 because the correlation is broad: ac128 and ac144 sit next to it. The narrow-band share is under 0.001. Fixed-count stretches on the load are about 1 s long, ac136 from about 0 to 0.3, lag 72 to 80 ms. On the returns the count's strongest lag is 48 ms, and fixed-count ac136 is near 0 or negative.

First Z change and half the change, loads, from the per-tick file. The window is 8.03 ms.

| Load | First change | Half the change |
| --- | --- | --- |
| conf-a1 | 7.90 ms, 0.98 windows, n=420 | 23.86 ms, 2.97 windows, n=354 |
| conf-b1 | 7.92 ms, 0.99 windows, n=520 | 24.14 ms, 3.00 windows, n=467 |
| conf-a2 | 7.89 ms, 0.98 windows, n=366 | 24.02 ms, 2.99 windows, n=314 |
| conf-b2 | 7.86 ms, 0.98 windows, n=446 | 31.87 ms, 3.97 windows, n=390 |

The first change matches the fourteenth pass. Half the change on these loads is still about three windows. Two returns reach half the change on the next window, at 8.07 ms and 15.87 ms. The load is the hunting comparison.

**At a high-group workload point, does the loop stop hunting?** No. Each 30 s load is 36 to 48 percent in band, with 467 to 621 tick moves, and the count uses about ten values. The last 10 s is 78 to 94 percent in band and still moving. The thirteenth quiet run entered and stayed. These do not.

## Clocks

Flag files and the per-launch display files together contribute 9,329 reason samples. Every one is `0x0000000000000000`. Die temperature on the flag files is 40 to 68 °C.

Display samples with GPU utilization at or above 90 percent: 4,160. Memory clock is 14801 MHz on all of them. SM clock is 2317 to 2895 MHz, median 2835. Power on those samples is 47.19 to 231.11 W.

`display-all.csv` has 4,898 samples and one change. Enabled from 21:39:58 until 22:32:00.846 PDT, then Disabled through the end of the session. The loop did not fight the driver. The reason word stayed 0 while the count hunted and while the plant stepped.

## Checks

Switches unset. Commanded target 1.5 on the gate, after a `[CTRL]` line that still shows target 1.00.

| Gate | This pass z_mean | Fourteenth z_mean | Result |
| --- | --- | --- | --- |
| gold | 1.498 | 1.415 | PASS |
| transfer | 1.495 | 1.437 | PASS |
| episode | 1.476 | 1.499 | PASS |

Gold `[CHECK]`: `pulses=9442014 z=1.477 z_mean=1.498 rate=396894.4 n=54`. Transfer: `z=1.485 z_mean=1.495`. Episode: `z=1.447 z_mean=1.476`, JSON `z_mean=1.4764`. Each file records `dyno=PASS closedloop=PASS`. The gains were not changed.

## Success criteria

- Full range. Yes. Zero is the idle GPU. The counted floor is 0.0387 TFLOPS at 82 SMs and 0.0010 TFLOPS at 2 SMs. The ceiling is 53.19 TFLOPS at acc 32, 2048 rounds, 256 threads, 82 SMs, from the per-SM counter on the 30 s hold.
- Map. Yes. `sweep-points.csv` has 403 rows. Each row has the launch `rate_ref`, the per-SM pulse rate, counted ops/s, `z_raw` SD, and the 136 ms figures. The 1 ms traces are gzipped. The plots are the four SVGs.
- Floor answered. Yes, the base point moved, and no workload knob is the setting that moved it. Before the close of `sms-desc-2`, base-settings calibrations were 46.69 to 47.42 million, plus one launch at 43.90 million. The closing base of `sms-desc-2` is 602,128 per SM. Fifteen later base rows are at or above 598,000, and the later calibrations are 49.06 to 49.56 million. It stayed for the rest of the session. No round count, thread count, or chain count raised ops/s per SM by 601/580 while the plant was still the 47 million one. The move is the pulse rate at the base work.
- Wave answered. The period stays at 136 ms across rounds, threads, chains, and SM count on the 47 million plant, wherever `z_raw` actually moves. It does not follow pass time. On the 49 million plant the same base work has its strongest lag at 48 ms, and ac136 falls to about 0. The closed-loop loads on that plant have their strongest lag at 72 ms. Short passes and counts of 2 to 6 SMs leave `z_raw` flat, so those points have no lag.
- Hysteresis answered. No group change from the load path. Stairs match within 0.47 percent. The open-loop ramp matches within 1.3 percent. Opening and closing bases stay near 603,000 per SM. The closed-loop count does not retrace. The per-SM rate does.
- Confirmations. Yes, at the base workload, because that is the point that reached the high group. Four launches, ABAB. The same-session 47 million baselines are the earlier open-loop brackets. The hunting answer is no.
- Guardrail. Yes. `integral_on_rail=0` on every `[STATS]` line in the logs. Clamp 15 did not trip.
- Driver orthogonality. Yes. Clocks, flags, and display state are logged on every launch. Every reason word is 0. The loop did not fight the driver.
- No regressions. Yes. With every switch unset, `[CTRL]` shows target 1.00, Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, and the launch leaves out 82 and 83. `GPUTRONIC_OPEN_SCHED` and `GPUTRONIC_RATE_WIN_US` are off by default. The three check gates pass at commanded 1.5.

## What was argued

A zero-round switch was not added. The source floors rounds at 1 in `gold_scaled_rounds`, in the independent-round setup, and in `gold_work_indep`. A round count of 0 would keep the kernel and the pulse and add no FMAs. The directive's own zero is the GPU with no GPUtronic process. That was measured: P3, about 37 W, memory at 7001 MHz. The counted floor at 1 round is 0.039 TFLOPS. The gap between those two is idle blocks spinning, which this pass was told to leave alone.

A pulse-period switch was not added. `GOLD_PULSE_LOCAL_NS` is 1 ms, and every sweep point already published at that period. On the 47 million plant the lag is 136 ms from 192 rounds to 4096 rounds, and at the floor. Pass time over that span is under a millisecond. The publish period is a candidate for a later pass. It is not what this pass's lag is following. On the 49 million plant the lag moves to 48 ms with the publish period still 1 ms, which makes the same point a second time.

Item 5 says to confirm a point that reached the 49 million group, against a same-session baseline in the 47 million group. The 49 million group here is a base-like point, acc 32, rounds 512, 512 threads, 82 SMs, whose per-SM pulse rate reaches about 598,000. A higher ops/s from more FMAs per pulse is a different question, and section 2 answers it: the ops rate is flat, and the short passes are slower. Sending the closed loop to 4096 rounds would have measured the heaviest arithmetic, not the plant that the thirteenth pass called quiet. The base point itself entered the high group, so the four confirmations are that workload. They calibrated at 49.56, 49.31, 49.31, and 49.06 million. None was retried. A retry would have been an attempt to push a high calibration back into the 47 million band, and the directive says not to do that to a launch that has already landed in either group. The 47 million baselines in this session are open loop. They are named above.

The ramp file holds scale 0.800 only at the start of the up leg, holds 1.200 for one step, and walks down through 0.801. The closing base takes the endpoint, so 0.800 is not a second held point on the way down.

The grid launches whose thread count is not 512 bracket themselves on rounds 512 at that thread count. `base-grid-0` and `base-grid-1` are the 512-thread bases around that grid.

Gains, the compiled clamp, the launch, and the floor were left where they are. Clamp 15 was not adopted as a default. `libsmctrl` was not used. The desktop was not changed. Display went from Enabled to Disabled at 22:32:00.846 PDT while `sms-desc-2` was on its closing base. That is recorded. It is not offered as the cause of the step.
