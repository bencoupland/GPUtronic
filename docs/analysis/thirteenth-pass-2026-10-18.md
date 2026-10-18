# Thirteenth pass, 2026-10-18

Branch `test-results/2026-10-18_thirteenth-pass`, started from `add4304`. Logs in `logs/2026-10-18_thirteenth-pass/`. The card is an RTX 5080, driver 595.99.02, built with CUDA 12.8, `sm_120`. `nvidia-smi` reports driver CUDA 13.2. The build is 12.8. No run was stopped for temperature. The hottest die sample is 67 °C, on the first clamp-15 hold. Every `clocks_event_reasons.active` sample in the hold flag files is 0.

The directive is on `origin/controller-fixes` at `4184d12` (`docs/directives/thirteenth-pass-2026-10-18.md`). That branch is not merged. The hold length and the extra step sizes are argued at the end. The controller source is unchanged.

## What this pass changed

Nothing in the source. The binary is `add4304`. `GPUTRONIC_GOLD_TARGET_Z` is `1.0f`. The gold, transfer, and episode check gates command 1.5. Kp is 0.55 and Ki is 0.08. On an 82-block launch the compiled clamp prints 3.28 SMs. No switch was added.

## How the holds were run

Full-pipe switches on every hold: `GPUTRONIC_PEAK_ACC=32`, `GPUTRONIC_PEAK_ROUNDS=512`, `GPUTRONIC_THREADS=512`, `GPUTRONIC_PULSE_LOCAL=1`, `GPUTRONIC_CTRL_CACHE=1`, `GPUTRONIC_TRACE_BIN_US=1000`, `GPUTRONIC_WORK_LOAD=1`. Idle-exit, guest, the work counter, the SM accumulator, relaunch, the tensor sensor, and `GPUTRONIC_INTEGRAL_CLAMP` were unset. Clocks and flags were read with `nvidia-smi`. The 10 µs file was deleted after each hold. It is not committed.

Item 1 sets `GPUTRONIC_ICLAMP_SMS`. Item 2 leaves that unset, so the clamp is the compiled 3.28 SMs. The target-1.5 holds set `GPUTRONIC_TARGET_Z=1.5`. The Z = 1 hold sets only `GPUTRONIC_ICLAMP_SMS=15`, so the compiled target stays 1.0. That run's clamp prints 15.00 SMs and the integral stays at 0, so the wider budget never binds. The twelfth Z = 1 hold used the compiled clamp. The difference is only that unused budget.

Item 1 schedule `steps-item1.txt`: 20 s pre-roll, then 45 s at 1.2 and 45 s at 1.0, three pairs, duration 290 s. The points are 20000, 65000, 110000, 155000, 200000, and 245000 ms. The last 30 s of a 45 s hold starts 15 s after the step. A 30 s hold would have made "last 30 s" the whole hold, including the step, which is a different window from the twelfth pass (last 30 s of a 60 s hold, starting 30 s after the step). Six holds fit under five minutes. The last 10 s is reported beside the last 30 s. The step sweeps use a 15 s loaded hold and a 15 s return, duration 45 s (`15000` at the scale, `30000` at 1.0). The lengthened 1.15 hold is 60 s loaded and 15 s at 1.0, duration 90 s (`steps-1150-long.txt`).

The band is `|z_hat − target| ≤ 0.08`, applied to `z_hat` rounded to five decimal places, so a printed 1.58000 is inside. "Enters and stays" is the first in-band bin after the last out-of-band bin. The tail is that entry to the hold's last sample, in milliseconds. A limit cycle is exactly two SM counts in the named window, each at least five times, with at least six changes. The last N seconds are anchored on the hold's last sample.

On the curve, the trip rate is that loaded hold's `rail_ticks / pi_ticks`. The whole-run `[STATS]` rate is given as well. Rail ticks on a return, while the integral unwinds, are not a second trip. The twelfth compiled-clamp returns had 674, 667, and 1,096 such ticks. The same unwind shows up here.

"Needed" is `(median count − n_ff) + (median z_hat − target) / (dZ/dn)` from the last 5 s of that loaded hold. The stored correction is `abs(integral) × 0.08 × sms_per_u`. Inside the deadband the integer count is held and the integrator is frozen, so a large need with a small stored integral means the deadband is holding the count.

Physical SMs 82 and 83 are the left-out pair on every hold. The launch line is 82 blocks.

The round counts below are the device formula `(512 × scale_milli + 500) / 1000`, with `scale_milli` the nearest integer of `scale × 1000`. They are formula integers. The twelfth pass measured 614 rounds at scale 1.2 on this plant. This pass did not repeat the op tally. Scale 1.05 is 538 rounds, 1.06 is 543, 1.075 is 550, 1.10 is 563, 1.15 is 589, 1.175 is 602, and 1.20 is 614.

## 1. Clamp 15, target 1.5, first run

`GPUTRONIC_ICLAMP_SMS=15`, `GPUTRONIC_TARGET_Z=1.5`. Log `c15-z15-a.txt`. Trace `c15-z15-a-1ms.csv`, 35,963,634 bytes, 289,982 data rows, t = 0.00 to 289,999.20 ms. The plot is `c15-z15-a-zhat.svg`.

Calibration: `rate_ref=47114836`, `n_ff=56.5`, `dZ/dn=0.0317`, `sms_per_u=31.5`, integral clamp 5.947, printed as 15.00 SMs of 82. `[STATS]`: `integral_on_rail=0` of `pi_ticks=4613106`, `sm_moves=6560`, `active_sms=48..67`, `mean_Z=1.5006`. The six holds' `rail_ticks` sum to 0.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end | SM range | Last 30 s inside | Last 30 s on the clamp | SM changes, last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 s | 1.2 | 19.0 ms | 44.995 s | 4 ms | 1.80703 | 1.47744 | 55 | 64 | 55–67 | 21,482 / 29,995 (71.6%) | 0 / 469,509 (0%) | 639 |
| 65 s | 1.0 | 19.1 ms | 44.870 s | 129 ms | 1.62992 | 1.31641 | 64 | 56 | 55–64 | 20,413 / 29,994 (68.1%) | 0 / 473,454 (0%) | 297 |
| 110 s | 1.2 | 31.1 ms | ends outside | — | 1.79649 | 1.45115 | 56 | 65 | 56–66 | 22,116 / 30,000 (73.7%) | 0 / 474,398 (0%) | 584 |
| 155 s | 1.0 | 0.2 ms | ends outside | — | 1.65534 | 1.31576 | 65 | 59 | 56–65 | 21,621 / 30,001 (72.1%) | 0 / 487,318 (0%) | 243 |
| 200 s | 1.2 | 0.2 ms | 44.987 s | 12 ms | 1.81266 | 1.47602 | 59 | 64 | 59–67 | 22,032 / 30,000 (73.4%) | 0 / 485,282 (0%) | 620 |
| 245 s | 1.0 | 19.0 ms | 44.878 s | 121 ms | 1.76061 | 1.31640 | 64 | 56 | 55–66 | 20,600 / 29,995 (68.7%) | 0 / 482,974 (0%) | 276 |

The loaded holds end at 64, 65, and 64. The returns end at 56, 59, and 56. The 110 s loaded hold leaves the band for its last 51 bins, about 50 ms, at `z_hat` 1.63099 and 65 SMs. The 155 s return leaves for its last 7 bins, about 6 ms, at `z_hat` 1.60194 and 59 SMs. The other four enter and stay. Whole-hold in-band shares: 54.4%, 53.3%, 57.7%, 56.1%, 58.9%, 53.7%. Last 10 s in band: 78.0%, 76.3%, 78.3%, 84.9%, 78.3%, 76.1%.

The 155 s return's last 10 s meets the letter of a limit cycle: 7 changes, and the only counts are 56 (9,873 bins) and 59 (127 bins). The last 30 s of that return is six counts and 243 changes. It is a short visit to 59 at the end of the hold, while 56 holds almost all of the window.

Loaded integrals end at −2.55398, −2.57149, and −2.60846. That is 6.44, 6.48, and 6.57 SMs, against a 15 SM budget. Over the last 3 s the integral is still moving by about −0.060 to −0.065. The twelfth clamp-15 loaded holds, which were 60 s, had finished near 8.0 to 9.4 SMs. These 45 s shares are a moving integral.

`dmon` has 294 samples, power 41–219 W (mean 190.9 W), die 48–67 °C. While the SM is busy the SM clock is 2835–2872 MHz and the memory clock is 14801 MHz. 296 flag samples, every reason word 0, die 50–67 °C, power 59.86–219.82 W.

## 2. Clamp 15, target 1.5, second run

Same switches. Log `c15-z15-b.txt`. Trace `c15-z15-b-1ms.csv`, 35,959,531 bytes, 289,956 data rows, t = 0.00 to 289,999.18 ms. The plot is `c15-z15-b-zhat.svg`.

Calibration: `rate_ref=47127304`, `n_ff=56.2`, `dZ/dn=0.0301`, `sms_per_u=33.2`, integral clamp 5.645, printed as 15.00 SMs of 82. `[STATS]`: `integral_on_rail=0` of `pi_ticks=5084509`, `sm_moves=6472`, `active_sms=47..67`, `mean_Z=1.4949`. Every hold's `rail_ticks` sum is 0.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end | SM range | Last 30 s inside | Last 30 s on the clamp | SM changes, last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 s | 1.2 | 39.1 ms | 44.984 s | 15 ms | 1.77824 | 1.46698 | 58 | 64 | 58–66 | 21,843 / 29,998 (72.8%) | 0 / 464,815 (0%) | 615 |
| 65 s | 1.0 | 16.2 ms | ends outside | — | 1.62358 | 1.32001 | 64 | 56 | 56–64 | 22,611 / 30,001 (75.4%) | 0 / 466,389 (0%) | 214 |
| 110 s | 1.2 | 0.0 ms | ends outside | — | 1.76063 | 1.41572 | 56 | 65 | 56–67 | 22,615 / 29,987 (75.4%) | 0 / 512,630 (0%) | 624 |
| 155 s | 1.0 | 0.2 ms | 44.558 s | 441 ms | 1.62360 | 1.31836 | 65 | 55 | 55–65 | 21,283 / 29,997 (71.0%) | 0 / 569,259 (0%) | 287 |
| 200 s | 1.2 | 21.2 ms | 44.936 s | 63 ms | 1.83095 | 1.47538 | 55 | 64 | 55–67 | 22,398 / 29,998 (74.7%) | 0 / 559,216 (0%) | 570 |
| 245 s | 1.0 | 16.1 ms | 36.354 s | 8645 ms | 1.83455 | 1.32339 | 64 | 54 | 54–66 | 20,172 / 29,991 (67.3%) | 0 / 629,758 (0%) | 187 |

The loaded holds end at 64, 65, and 64. The returns end at 56, 55, and 54. The 65 s return leaves the band for its last 4 bins, about 3.2 ms, at `z_hat` 1.41572 and 56 SMs. The 110 s loaded hold leaves for its last 13 bins, about 12 ms, at `z_hat` 1.59256 and 65 SMs. The 245 s return does enter and stay. Its tail is 8.6 s, and the last 10 s is 9,953 / 10,000 (99.5%) at 54 SMs with 4 changes. The last 30 s of that return is 67.3% because the earlier part of the window is still hunting. Last 10 s in band, in step order: 80.8%, 84.6%, 80.6%, 74.9%, 80.9%, 99.5%.

Loaded integrals end at −2.48242, −2.50589, and −2.44457. That is 6.59, 6.66, and 6.49 SMs. Over the last 3 s they are still moving by about −0.066. The 245 s return's integral has crossed through zero and stopped at +0.10298.

`dmon` has 294 samples, power 32–220 W (mean 191.1 W), die 48–64 °C. Busy SM clock 2835–2872 MHz, busy memory clock 14801 MHz. 296 flag samples, every reason word 0, die 49–64 °C. The samples under 2700 MHz are the start and the stop.

## 3. Clamp 15, target 1.5, third run

Same switches. Log `c15-z15-c.txt`. Trace `c15-z15-c-1ms.csv`, 35,993,787 bytes, 289,962 data rows, t = 0.00 to 289,999.03 ms. The plot is `c15-z15-c-zhat.svg`.

Calibration: `rate_ref=49314120`, `n_ff=56.1`, `dZ/dn=0.0327`, `sms_per_u=30.6`, integral clamp 6.129, printed as 15.00 SMs of 82. `[STATS]`: `integral_on_rail=0` of `pi_ticks=4508221`, `sm_moves=2388`, `active_sms=48..68`, `mean_Z=1.4988`. Every hold's `rail_ticks` sum is 0.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end | SM range | Last 30 s inside | Last 30 s on the clamp | SM changes, last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 s | 1.2 | 27.2 ms | 44.706 s | 293 ms | 1.84677 | 1.42981 | 57 | 63 | 57–68 | 24,609 / 29,993 (82.0%) | 0 / 467,251 (0%) | 219 |
| 65 s | 1.0 | 19.3 ms | 41.355 s | 3644 ms | 1.92487 | 1.36139 | 63 | 57 | 57–66 | 29,718 / 29,994 (99.1%) | 0 / 466,462 (0%) | 29 |
| 110 s | 1.2 | 26.1 ms | 44.569 s | 430 ms | 1.78455 | 1.43059 | 57 | 63 | 57–66 | 27,903 / 30,001 (93.0%) | 0 / 466,045 (0%) | 107 |
| 155 s | 1.0 | 19.1 ms | 38.679 s | 6320 ms | 1.62816 | 1.36076 | 63 | 57 | 57–63 | 29,881 / 30,000 (99.6%) | 0 / 467,569 (0%) | 10 |
| 200 s | 1.2 | 26.1 ms | 44.719 s | 280 ms | 1.78381 | 1.42991 | 57 | 63 | 57–66 | 27,302 / 30,001 (91.0%) | 0 / 465,745 (0%) | 148 |
| 245 s | 1.0 | 41.0 ms | 42.583 s | 2416 ms | 1.65743 | 1.35416 | 63 | 57 | 57–63 | 29,698 / 29,992 (99.0%) | 0 / 466,063 (0%) | 24 |

All six holds enter and stay. The loaded holds end at 63. The returns end at 57. Loaded last-30 s shares are 82.0%, 93.0%, and 91.0%. Return shares are 99.1%, 99.6%, and 99.0%. Last 10 s in band: 88.9%, 98.7%, 93.5%, 99.5%, 90.5%, 98.3%. No last-30 s window is a limit cycle.

Loaded integrals end at −2.35965, −2.35752, and −2.42648. That is 5.78, 5.77, and 5.94 SMs of a 15 SM budget. The returns park near −1.14, about 2.8 SMs, with the integrator essentially still and the count at 57. That stored correction is inside the band. It is not a rail.

`dmon` has 293 samples, power 22–210 W (mean 192.6 W), die 48–64 °C. After the first busy sample the SM clock is in the boost steps, and the busy memory clock is 14801 MHz. 296 flag samples, every reason word 0, die 48–64 °C.

## 4. Clamp 9, target 1.5

`GPUTRONIC_ICLAMP_SMS=9`, `GPUTRONIC_TARGET_Z=1.5`. Log `c9-z15.txt`. Trace `c9-z15-1ms.csv`, 35,994,949 bytes, 289,972 data rows, t = 0.00 to 289,999.12 ms. The plot is `c9-z15-zhat.svg`.

Calibration: `rate_ref=49311496`, `n_ff=56.1`, `dZ/dn=0.0331`, `sms_per_u=30.2`, integral clamp 3.726, printed as 9.00 SMs of 82. `[STATS]`: `integral_on_rail=0` of `pi_ticks=4515919`, `sm_moves=2642`, `active_sms=48..67`, `mean_Z=1.4989`. Every hold's `rail_ticks` sum is 0.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end | SM range | Last 30 s inside | Last 30 s on the clamp | SM changes, last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 s | 1.2 | 25.1 ms | 44.420 s | 579 ms | 1.81827 | 1.43616 | 57 | 63 | 57–67 | 24,712 / 29,997 (82.4%) | 0 / 467,325 (0%) | 186 |
| 65 s | 1.0 | 23.1 ms | 43.130 s | 1869 ms | 1.69593 | 1.35996 | 63 | 57 | 57–63 | 29,564 / 30,000 (98.5%) | 0 / 467,261 (0%) | 45 |
| 110 s | 1.2 | 23.1 ms | 44.875 s | 124 ms | 1.83723 | 1.43554 | 57 | 63 | 57–66 | 26,836 / 29,996 (89.5%) | 0 / 466,294 (0%) | 162 |
| 155 s | 1.0 | 20.0 ms | 43.313 s | 1686 ms | 1.68450 | 1.35668 | 63 | 57 | 57–63 | 29,888 / 30,001 (99.6%) | 0 / 465,349 (0%) | 12 |
| 200 s | 1.2 | 24.1 ms | ends outside | — | 1.86511 | 1.43251 | 57 | 63 | 57–66 | 27,352 / 29,996 (91.2%) | 0 / 466,520 (0%) | 128 |
| 245 s | 1.0 | 0.2 ms | 44.967 s | 32 ms | 1.64548 | 1.35739 | 63 | 57 | 57–64 | 28,759 / 29,997 (95.9%) | 0 / 468,288 (0%) | 18 |

The 200 s loaded hold leaves the band for its last 2 bins, about 1.1 ms, at `z_hat` 1.59027 and 63 SMs. The other five enter and stay. Loaded holds end at 63. Returns end at 57. Last 10 s in band: 93.6%, 98.8%, 90.4%, 98.9%, 92.2%, 88.0%. Loaded integrals end at −2.36254, −2.46941, and −2.42674. That is 5.71, 5.97, and 5.86 SMs of a 9 SM budget. The returns park near 2.5 to 2.7 SMs of stored correction, inside the band, with the count at 57.

`dmon` has 294 samples, power 28–223 W (mean 192.8 W), die 48–64 °C. Busy memory clock 14801 MHz. 296 flag samples, every reason word 0, die 48–64 °C.

This run and the third clamp-15 run are the same shape: `rate_ref` 49.3 million, about 2,400 to 2,600 count moves, returns near 99% of the last 30 s in band, loaded counts ending at 63. The first two clamp-15 runs are the other shape: `rate_ref` 47.1 million, about 6,500 count moves, and a short exit at the end of some holds. Nine SMs did not beat fifteen. The quiet fifteen-SM run is as well behaved as the nine. Neither width tripped.

## 5. Target 1, clamp budget 15, integral still

No `GPUTRONIC_TARGET_Z`. The log has no `[TARGET]` line. `[CTRL]` prints target 1.00. `GPUTRONIC_ICLAMP_SMS=15` is set and does not bind. Log `z1-unset.txt`. Trace `z1-unset-1ms.csv`, 35,705,524 bytes, 289,964 data rows, t = 0.00 to 289,999.24 ms. The plot is `z1-unset-zhat.svg`.

Calibration: `rate_ref=47040876`, `n_ff=82.0`, `dZ/dn=0.0173`, `sms_per_u=57.7`, integral clamp 3.249, printed as 15.00 SMs of 82. `[STATS]`: `integral_on_rail=0` of `pi_ticks=4545804`, `integral_updates=0`, `sm_moves=0`, `active_sms=82..82`, `mean_Z=1.1021`. The integral is 0.00000 at the end of every hold.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end | SM range | Last 30 s inside | Last 30 s on the clamp | SM changes, last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 s | 1.2 | 15.0 ms | ends outside | — | 1.41161 | 1.03932 | 82 | 82 | 82–82 | 0 / 30,001 (0%) | 0 / 470,609 (0%) | 0 |
| 65 s | 1.0 | 0.0 ms | 44.101 s | 898 ms | 1.20807 | 0.95838 | 82 | 82 | 82–82 | 29,425 / 29,996 (98.1%) | 0 / 468,198 (0%) | 0 |
| 110 s | 1.2 | 33.2 ms | ends outside | — | 1.36437 | 0.98639 | 82 | 82 | 82–82 | 0 / 29,994 (0%) | 0 / 475,176 (0%) | 0 |
| 155 s | 1.0 | 0.1 ms | 44.177 s | 822 ms | 1.18881 | 0.95895 | 82 | 82 | 82–82 | 29,489 / 29,995 (98.3%) | 0 / 471,040 (0%) | 0 |
| 200 s | 1.2 | 13.2 ms | ends outside | — | 1.39170 | 1.03834 | 82 | 82 | 82–82 | 0 / 29,991 (0%) | 0 / 471,040 (0%) | 0 |
| 245 s | 1.0 | 0.2 ms | 43.133 s | 1866 ms | 1.16527 | 0.95983 | 82 | 82 | 82–82 | 29,433 / 30,000 (98.1%) | 0 / 475,373 (0%) | 0 |

The loaded holds end outside the band at 1.22291, 1.18855, and 1.17045, on 82 SMs. The returns enter and stay. Their last 30 s is 98.1%, 98.3%, and 98.1% in band. The last 10 s is 98.5%, 98.4%, and 98.3%. Whole-hold return shares are 98.3%, 98.3%, and 98.2%. The twelfth Z = 1 real-work returns were 95.9%, 96.5%, and 95.2%, also at 82 SMs with a still integral. The default is clean. The count is the launch, and there is no SM left to add.

`dmon` has 294 samples, power 33–221 W (mean 216.5 W), die 48–66 °C. The higher power is the count staying at 82. Busy SM clock 2835–2887 MHz, busy memory clock 14801 MHz. 296 flag samples, every reason word 0.

## 6. What the 15 SM tail is

Run-to-run variance.

The end nick is the same class of event as the twelfth clamp-15 return, which left the band for its last 23 bins, about 22 ms, at `z_hat` 1.40105 and 57 SMs. Here the exits that fail "enters and stays" are 50 ms, 6 ms, 3.2 ms, 12 ms, and, on the 9 SM run, 1.1 ms. They show up on two of the three clamp-15 runs at target 1.5, and not on the third. The third clamp-15 run and the 9 SM run both enter and stay on five or six of the six holds, and their returns sit near 99% of the last 30 s.

The compiled target moving to 1.0 does not produce this. Every target-1.5 log prints `[CTRL]` target 1.50 and an `n_ff` near 56. The unset run prints target 1.00, `n_ff=82`, `integral_updates=0`, and `sm_moves=0`. The Z = 1.5 loop is the spare-SM loop. The Z = 1 loop stands aside.

Fifteen SMs of room does not, by itself, make the return hunt. The two hunting runs and the two quiet runs used the same clocks. Busy SM clock is the boost band, busy memory clock is 14801 MHz, and every reason word is 0. What separates them in the log is `rate_ref`: 47,114,836 and 47,127,304 on the hunting runs, 49,314,120 and 49,311,496 on the quiet runs. The twelfth clamp-15 hold was in the lower group (`rate_ref=46720372`, `sm_moves=9903` over 420 s) and it hunted. This pass does not identify why the reference lands in one group or the other. The 1 Hz clock samples do not.

The last-30 s percentages are not the twelfth metric. There the window starts 30 s after the step. Here it starts 15 s after. On the hunting runs the loaded integral is still moving at 45 s, about 0.02 per second, and it has stored 6.4 to 6.6 SMs against the twelfth's 8.0 to 9.4 SMs at 60 s. The return shares on those runs, 67% to 75%, sit under the twelfth's 81% to 87%. The quiet run's returns sit near 99%. No last-30 s window in this pass is a limit cycle. The rail is 0 on all five holds.

## 7. Compiled clamp against step size

No `GPUTRONIC_ICLAMP_SMS`. `GPUTRONIC_TARGET_Z=1.5`. Each size is its own load and return, two repeats, duration 45 s, except the two lengthened 1.15 holds. Every run prints the scaled clamp as 3.28 SMs of 82, launches 82, leaves out 82 and 83, and has reason word 0. `rate_ref` on these runs is 49.1 to 49.6 million, the quiet group from item 1. Die on the flag files is 40–64 °C. The 40 °C sample is the start of `sweep-1175-a`, which began from a cool card. Busy memory clock is 14801 MHz. A single end sample on `sweep-1200-b` reads 15001 MHz while the SM clock is already falling. That is the stop, not a mid-hold change.

The 15 s traces are 5,536,198 to 5,553,928 bytes and 44,980 to 45,000 data rows, t = 0.00 to about 44,999 ms. The two 90 s traces are `sweep-1150-long-a-1ms.csv` (11,129,740 bytes, 89,980 rows, t to 89,999.12 ms) and `sweep-1150-long-b-1ms.csv` (11,120,232 bytes, 90,000 rows, t to 89,999.06 ms).

The trip rate and the last-10 s in-band share for the 15 s holds are `curve.svg`, drawn from `curve.txt`. Repeat a is blue, repeat b is orange. The 60 s holds are not on that series. The loaded-hold plots are `sweep-1175-a-zhat.svg`, `sweep-1200-a-zhat.svg`, and `sweep-1150-long-a-zhat.svg`.

Needed SMs are from the last 5 s. The in-band share on the curve is the last 10 s.

| Scale | Run | Loaded trip rate | Enters and stays | Tail | Last 10 s in band | Lowest z_hat | Highest z_hat | End count | n_ff | Stored, SMs | Needed, SMs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.05 | a | 0 / 234,249 (0%) | 14.384 s | 615 ms | 9,876 / 10,001 (98.8%) | 1.46424 | 1.69989 | 58 | 56.3 | 0.10 | 1.14 |
| 1.05 | b | 0 / 280,027 (0%) | 14.892 s | 107 ms | 9,696 / 10,000 (97.0%) | 1.43347 | 1.68324 | 58 | 56.1 | 0.14 | 1.37 |
| 1.06 | a | 0 / 235,838 (0%) | 14.304 s | 695 ms | 9,763 / 9,995 (97.7%) | 1.48149 | 1.66073 | 58 | 56.2 | 0.13 | 1.68 |
| 1.06 | b | 0 / 233,383 (0%) | 14.632 s | 367 ms | 9,803 / 10,001 (98.0%) | 1.48137 | 1.66795 | 58 | 56.2 | 0.18 | 1.63 |
| 1.075 | a | 0 / 238,477 (0%) | 14.055 s | 944 ms | 9,446 / 10,001 (94.5%) | 1.48327 | 1.72133 | 58 | 56.5 | 0.27 | 2.14 |
| 1.075 | b | 0 / 233,947 (0%) | 14.445 s | 554 ms | 9,711 / 10,000 (97.1%) | 1.43031 | 1.80558 | 58 | 56.0 | 0.21 | 2.60 |
| 1.10 | a | 0 / 235,973 (0%) | 14.859 s | 140 ms | 9,232 / 10,000 (92.3%) | 1.42663 | 1.73746 | 58 | 55.7 | 0.41 | 3.87 |
| 1.10 | b | 0 / 231,651 (0%) | 14.317 s | 682 ms | 9,279 / 10,001 (92.8%) | 1.48749 | 1.81472 | 58 | 56.3 | 0.33 | 3.07 |
| 1.15 | a | 0 / 235,781 (0%) | 14.734 s | 265 ms | 5,978 / 10,001 (59.8%) | 1.42765 | 1.76698 | 60 | 55.9 | 2.41 | 6.19 |
| 1.15 | b | 0 / 232,449 (0%) | 14.083 s | 916 ms | 8,352 / 10,000 (83.5%) | 1.48261 | 1.74530 | 60 | 56.5 | 1.77 | 5.60 |
| 1.175 | a | 56,610 / 232,703 (24.3%) | 14.999 s | 0 ms | 2,080 / 10,001 (20.8%) | 1.48948 | 1.78360 | 61 | 56.0 | 3.28 | 7.58 |
| 1.175 | b | 56,756 / 232,633 (24.4%) | ends outside | — | 970 / 10,000 (9.7%) | 1.50797 | 1.86304 | 61 | 55.7 | 3.28 | 7.12 |
| 1.20 | a | 77,438 / 232,441 (33.3%) | ends outside | — | 0 / 9,996 (0.0%) | 1.48434 | 1.81794 | 61 | 56.3 | 3.28 | 7.81 |
| 1.20 | b | 88,582 / 233,127 (38.0%) | ends outside | — | 0 / 10,001 (0.0%) | 1.44307 | 1.95454 | 61 | 56.0 | 3.28 | 8.42 |

From 1.05 through 1.10 the rail is not touched. The count ends at 58, about two above `n_ff`, and the integral stores 0.10 to 0.41 SMs. Each of those steps does leave the band: the peaks run from 1.66 to 1.81. They then sit inside it. The deadband holds the count. At 1.10 the last-5 s need is 3.87 and 3.07 SMs, which meets or passes the 3.28 SM rail, while the integral has stored 0.41 and 0.33 SMs. The need is the error that would remain if the median of that window had to be removed by the integral. Once `z_hat` is inside the band, the deadband does not spend it.

A few last-10 s windows meet the letter of a limit cycle, and none of them is a two-level oscillation. At 1.05 b the last 10 s is 58 on 9,926 bins and 59 on 74, with 8 changes. At 1.06 a it is 58 on 9,958 and 59 on 37. At 1.06 b it is 58 on 9,955 and 59 on 46. The last 5 s of 1.075 b is 58 on 4,946 and 59 on 54. The second count is a handful of visits.

At 1.15 in the 15 s window the rail is still untouched, and the hold has not finished winding. The integral ends at 2.41 and 1.77 SMs of the 3.28, still moving by about −0.024 over the last 3 s. The last 10 s is 59.8% and 83.5% in band. The last 5 s is 92.3% and 93.5%. The count ends at 60. That is why this size was given a 60 s loaded hold.

At 1.175 both repeats pin the integral at the printed rail, 3.28 SMs, and `i_move` over the last 3 s is 0. The first bin whose own rail fraction exceeds one half is at +11,351 ms and +11,355 ms. At +5 s the integral is near −0.67 and that bin's rail ticks are 0. At +10 s it is near −1.17 to −1.21, still off the rail. Run a's last bin is `z_hat` 1.57845, a one-bin touch of the inside of the band, which is why the table prints a 0 ms tail. The last 10 s of that hold is 20.8% in band, and 56,610 of its 155,007 control ticks are on the rail. Run b ends outside for its last 553 bins, about 552 ms, at `z_hat` 1.59117 and 61 SMs. Whole-run `[STATS]` rail ticks are 57,305 of 699,448 (8.2%) and 57,470 of 696,972 (8.2%). The returns contribute 695 and 714 of those, which is the unwind, and their last 10 s is 98.5% and 98.4% in band at 57 SMs.

At 1.20 the same pin happens sooner. The first bin past half its ticks on the rail is at +10,015 ms and +9,308 ms. The last 5 s is 99.8% and 100% on the rail, and the last 10 s is 0% in band. End `z_hat` is 1.61471 and 1.59490, at 61 SMs, on the rail. Whole-run `[STATS]` is 78,111 of 697,669 (11.2%) and 89,213 of 701,624 (12.7%). The returns contribute 673 and 631 rail ticks, the same unwind as the twelfth compiled-clamp returns, and their last 10 s is 99.2% and 99.1% in band at 57 SMs. The integral on those returns has come back to 2.57 and 2.89 SMs.

The 60 s loaded holds at 1.15, logs `sweep-1150-long-a.txt` and `sweep-1150-long-b.txt`:

| Run | rate_ref | n_ff | Loaded trip rate | Enters and stays | Tail | Last 10 s in band | Last 30 s in band | End count | End integral | Stored, SMs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | 49,132,868 | 55.9 | 0 / 929,154 (0%) | 59.327 s | 672 ms | 9,454 / 10,001 (94.5%) | 28,152 / 29,996 (93.9%) | 60 | −1.32163 | 3.21 |
| b | 49,298,036 | 56.5 | 0 / 938,544 (0%) | 59.602 s | 397 ms | 9,395 / 10,000 (94.0%) | 28,134 / 30,001 (93.8%) | 61 | −1.10057 | 2.90 |

Both enter and stay. Neither hits the rail. `integral_on_rail` in `[STATS]` is 0. On run a the integral is −0.570 at +5 s, −0.961 at +15 s, −1.178 at +40 s, and −1.322 at the end. The printed clamp is 1.350, so the end is 0.07 SM short of 3.28, and the last 3 s still moves by −0.023. On run b it is −0.509 at +5 s, −0.683 at +15 s, −0.965 at +40 s, and −1.101 at the end, against a printed clamp of 1.246. That is 0.38 SM short of 3.28, still moving by −0.046 over the last 3 s. The +15 s values sit within 0.006 of the two 15 s runs, which ended at −0.956 and −0.686. The longer window is the same wind, continued. It gets close to the rail and the count stays in band.

The largest step this clamp holds without tripping is 1.15, on the 60 s hold. In the 15 s window, 1.15 is also the largest step with zero loaded rail ticks, and that window had not settled. The first step in the set that trips is 1.175, inside the 15 s hold, with the integral pinned at 3.28 SMs from about +11 s. Scale 1.20 trips as well. No new clamp width is proposed.

The open-loop estimate `56 × (rounds / 512 − 1)` crosses 3.28 SMs near scale 1.06. The measured integral at 1.06 is 0.13 to 0.18 SMs. The count moves about two SMs while Z is outside the band, Z comes back, and the deadband freezes the rest. The estimate assumed the whole rate deficit had to be cancelled while Z stayed at the open-loop value. It is not on the plot.

## Clocks

Read only. No thermal or power term was added. On every hold the reason word is 0. After the start sample, and apart from a single first busy sample on some runs while the clock is still climbing, the SM clock stays in the boost steps this card has used on earlier full-pipe holds, 2835–2887 MHz. The busy memory clock is 14801 MHz on the hunting runs and on the quiet runs. The 47.1 million against 49.3 million split in `rate_ref` is real, and these logs do not explain it.

The Z = 1 hold draws more power (mean 216.5 W, peak die 66 °C) because the count stays at 82. The target-1.5 holds sit near 183–193 W. The hottest die sample of the pass is 67 °C, on `c15-z15-a`. The 83 °C test stop was not reached.

## Item 3

- **Tail miss explained.** Yes. It is run-to-run variance, section 6. The logs separate that from a change caused by the compiled target, and from a miss that 15 SMs produce on every return.
- **Default clean.** Yes. With `GPUTRONIC_TARGET_Z` unset, the count stays at 82 and the integral stays at 0. The loaded holds sit above the band. The returns enter and stay.
- **Z = 1.5 beside `add4304`.** Reported, not identical. The hunting pair resembles the twelfth clamp-15 hold: lower `rate_ref`, the count still moving, and a short exit at the end of a hold. The quiet clamp-15 run and the clamp-9 run settle, with return shares near 99% of a last-30 s window that starts earlier than the twelfth's. The window difference is in section 6. The target line is 1.50 and `n_ff` is near 56, so the macro at 1.0 is not what moved them.
- **Curve.** Yes. Scales 1.05, 1.06, 1.075, 1.10, 1.15, 1.175, and 1.20, each twice, plus 1.15 twice at 60 s. The largest step held without a trip is 1.15. The first trip in the set is 1.175.
- **Guardrail.** Yes. Clamp 15 and clamp 9 have zero rail ticks on every hold. The compiled clamp trips at 1.175 and at 1.20. Those trips are named above.
- **Driver orthogonality.** Yes. Clocks and flags are on every run. The reason word is 0. The loop did not fight the driver.
- **No regressions.** Yes. The unset `[CTRL]` line is below. No switch was added. The three checks pass at commanded 1.5.

## Checks

Unset, switches off, on this binary. Log `unset-home.txt`. `[CTRL] Gold | target=1.00 Kp=0.550 Ki=0.0800 Kd=0.0000 deadband=0.080`, launching 82 blocks, leaving out SMs 82 and 83, 64 threads. There is no `[TARGET]` line. The holds above use 512 threads because the full-pipe switches are set. The check gates are the stock kernel.

The closed-loop half of each check commands 1.5. The dyno half boots at 1.00. Both halves pass.

| Check | This pass | Twelfth, commanded 1.5 |
| --- | --- | --- |
| Gold | 1.443 | 1.449 |
| Transfer | 1.407 | 1.441 |
| Episode | 1.457 | 1.493 |

Gold: free-run 555,616.8 pulse/s, closed 384,437.7, `useful_drop=YES` (`err=0.057`), dyno PASS, closedloop PASS. Transfer: free-run 244,862.5 pulse/s (8.0237 GFLOP/s), closed 173,712.4 (5.6922), PASS. Episode: free-run 666,033.5 pulse/s (21.8246 GFLOP/s), closed 456,262.3 (14.9508), `window=OK`, PASS. The closed-loop `[CTRL]` line is target 1.50, Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, launch 82. Transfer is the soft one beside 1.441. It passed. Kp and Ki were not changed.

## What was argued

**Forty-five second holds.** The directive's floor is 30 s, and it asks for the last 30 s. On a 30 s hold that window would include the step. The twelfth figure is the last 30 s of a 60 s hold. These holds are 45 s, so the last 30 s starts 15 s after the step, and the last 10 s is reported too. On the hunting runs the integral is still moving at 45 s. The twelfth 60 s holds already show where that wind finishes. Another five-minute run was not added.

**The extra sizes.** The directive asks for 1.2, half of that step (1.1), and a quarter (1.05), and for more points if the curve has room to show where the integral first reaches the rail. The open-loop estimate crossed 3.28 SMs near 1.06, so 1.06, 1.075, and 1.15 were in the first suite. The measurement put the first trip higher than that estimate. Scale 1.15 had not settled at 15 s, so it was repeated at 60 s of loaded time, and 1.175 was added on the 15 s window to see whether the trip starts just above 1.15. It does.

**Not a third length at 1.15.** At 60 s both repeats have entered and stayed, with zero rail ticks. The integral is 0.07 SM and 0.38 SM short of the rail and is still creeping. A longer hold would ask whether that remainder eventually pins. The boundary the curve was meant to find is already in the logs: 1.15 holds over 60 s, and 1.175 trips inside 15 s.

**The Z = 1 clamp budget.** That run kept `GPUTRONIC_ICLAMP_SMS=15` so the only change from the target-1.5 runs was the target. The budget never binds. The twelfth Z = 1 hold used the compiled clamp and the same still integral.

**No width, and no retune.** The compiled clamp stays 3.28 SMs. Nine and fifteen stay test budgets. Kp and Ki stay 0.55 and 0.08. The estimate that the rail would be crossed near 1.06 is not drawn as a point.
