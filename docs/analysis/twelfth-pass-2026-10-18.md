# Twelfth pass, 2026-10-18

Branch `test-results/2026-10-18_twelfth-pass`, started from `1647954`. Logs in `logs/2026-10-18_twelfth-pass/`. The card is an RTX 5080, driver 595.99.02, built with CUDA 12.8, `sm_120`. No run was stopped for temperature. The hottest die sample is 68 °C, on the Z = 1 hold. Every `clocks_event_reasons.active` sample in the five hold flag files is 0.

The directive is on `origin/controller-fixes` at `ad3c3cb` (`docs/directives/twelfth-pass-2026-10-18.md`). That branch is not merged. Two of its instructions are argued below. The measurement that carries them is the real-work load, and the Z = 1 hold on that load.

The five holds were run while `GPUTRONIC_GOLD_TARGET_Z` was still `1.5f`. The Z = 1 hold set `GPUTRONIC_TARGET_Z=1`. After that hold stayed at the ceiling, the macro moved to `1.0f` and the three check gates were pointed at 1.5. The earlier unset checks are `check-gold.txt`, `check-transfer.txt`, and `check-episode.txt`. The checks after the move are `check-commanded-gold.txt`, `check-commanded-transfer.txt`, and `check-commanded-episode.txt`.

## What this pass changed

`GPUTRONIC_WORK_LOAD=1` is a test switch, off by default, and only with the independent-FMA body (`GPUTRONIC_PEAK_ACC` set). The host sets `GPUTRONIC_FLAG_LOAD` when the schedule scale is not 1.0, and clears it at 1.0. On device, the round count is the nearest integer of `base * scale_milli / 1000`, clamped to 1..4096. At base 512 that is 614 rounds at scale 1.2 and 410 rounds at scale 0.8. Every batch still pulses. Unset, the same scale still pulse-skips. The op tally uses the rounds actually run. The row is in the function frame and in the README copy of it. From the shared title onward the two copies match.

The full-pipe plant is the one the eleventh pass measured. Extra FMAs on the stock atomic pass would not have moved the rate. On this plant they do. That is why the load is a longer independent-FMA body and not a second kernel, a sleep, or a thinner pulse.

## 1. The work really changed

Open loop, 56 SMs, 8 s, full-pipe switches, `GPUTRONIC_WORK_LOAD` unset, so the round count is the launch constant.

`PEAK_ROUNDS=512`, log `proof-noise-512.txt`. `rate_ref=45733304`. The t = 4..7 rates are 32,062,646, 31,575,418, 31,540,874, and 31,605,582 pulses/s.

`PEAK_ROUNDS=614`, log `proof-noise-614.txt`. `rate_ref=37769180`. The t = 4..7 rates are 26,028,200, 26,079,350, 26,035,024, and 25,979,920 pulses/s.

The reference ratio is 45,733,304 / 37,769,180 = 1.211. The ratio of those two four-second means is 1.218. The round ratio is 614 / 512 = 1.199. The plant moved by at least the 1.2 step's share. Item 1 continues.

The load-file proof is `proof-steps.txt` (`8000 1.2`), 16 s, full-pipe, `PEAK_ROUNDS=512`, `GPUTRONIC_WORK_COUNTER=1`. The ratio is `pulses * 16 / work_batches`.

Pulse-skip, `WORK_LOAD` unset, log `proof-skip.txt`. Through t = 8 s the ratio is 0.99926 to 1.00075. From t = 9 s it is 0.83303 to 0.83391. Pulses were thinned. One pulse per 1.2 batches is 1/1.2 = 0.833.

`GPUTRONIC_WORK_LOAD=1` and `GPUTRONIC_SM_ACCUM_SM=0`, log `proof-work.txt`. The ratio stays 0.99959 to 1.00038 on both sides of the step, including t = 9..15 s. Pulses were not thinned.

Ops per pulse are estimated from `proof-work-smaccum.csv` as `d_ops_cuda * active_sms / bin_pulses`. The file is SM 0 only, and `bin_pulses` is the chip. The expected warp figure is `32 * rounds * 2 * 32`. From 4 s to 8 s (scale 1.0) the median is 1,044,578, against 1,048,576. From 10 s to 15 s (scale 1.2) the median is 1,257,472, which is `32 * 614 * 2 * 32`. The mean sits above the median because some bins catch a transition. The median is the figure.

## How the holds were run

Same full-pipe switches as the eleventh pass: `GPUTRONIC_PEAK_ACC=32`, `GPUTRONIC_PEAK_ROUNDS=512`, `GPUTRONIC_THREADS=512`, `GPUTRONIC_PULSE_LOCAL=1`, `GPUTRONIC_CTRL_CACHE=1`, `GPUTRONIC_TRACE_BIN_US=1000`, `GPUTRONIC_WORK_LOAD=1`. Idle-exit, guest, and the work counter were unset. Clocks and flags were read with `nvidia-smi`. The 10 µs file was deleted after each hold. It is not committed.

Heavier schedule `steps-heavy.txt`: 60 s at 1.2, 120 s at 1.0, and the same pair twice more, through 420 s. Lighter schedule `steps-light.txt` uses 0.8 in place of 1.2. The band is `|z_hat - target| <= 0.08`. "Enters and stays" is the first in-band bin after the last out-of-band bin. The last 30 s is anchored on the hold's last sample. A limit cycle is exactly two SM counts in that window, each at least five times, with at least six changes. No hold in this pass is a limit cycle.

Physical SMs 82 and 83 are the left-out pair on every hold. The launch line is 82 blocks.

The pulse-skip twin for the heavier target-1.5 tables is the eleventh pass's clamp-15 run (`clamp15.txt` on `1647954`): `n_ff=56.0`, `sms_per_u=31.3`, loaded last-30 s shares 91.4%, 85.3%, 84.4% ending at 67, 64, and 65 SMs, return shares 90.1%, 91.5%, 91.6% ending at 56, lowest `z_hat` 1.28799, `integral_on_rail=0`. Its 300 s hold ended outside on the last 2.9 ms. The 9 SM pulse-skip return shares cited there are 87.4%, 86.2%, and 85.0%, also with a zero trip rate.

## 2. Heavier load, clamp 15

`GPUTRONIC_ICLAMP_SMS=15`, target 1.50. Log `heavy-c15.txt`. Trace `heavy-c15-1ms.csv`, 52,087,953 bytes, 419,964 data rows, t = 0.00 to 419,999.09 ms. The plot is `heavy-c15-zhat.svg`.

Calibration: `rate_ref=46720372`, `n_ff=55.7`, `dZ/dn=0.0298`, `sms_per_u=33.6`, integral clamp 5.584, printed as 15.00 SMs of 82. `[STATS]`: `integral_on_rail=0` of `pi_ticks=6726736`, `sm_moves=9903`, `active_sms=46..69`, `mean_Z=1.5018`. Every hold's `rail_ticks` sum is 0.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end | SM range | Last 30 s inside | Last 30 s on the clamp | SM changes, last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | 1.2 | 22.3 ms | 59.962 s | 37 ms | 1.87833 | 1.44494 | 54 | 65 | 54–67 | 22,554 / 30,001 (75.2%) | 0 / 466,920 (0%) | 538 |
| 120 s | 1.0 | 26.1 ms | 59.791 s | 208 ms | 1.64119 | 1.30737 | 65 | 57 | 57–65 | 26,111 / 29,991 (87.1%) | 0 / 468,439 (0%) | 179 |
| 180 s | 1.2 | 8.2 ms | 59.945 s | 54 ms | 1.84216 | 1.41929 | 57 | 67 | 57–69 | 24,982 / 30,001 (83.3%) | 0 / 553,693 (0%) | 424 |
| 240 s | 1.0 | 17.1 ms | 59.656 s | 343 ms | 1.63516 | 1.29593 | 67 | 56 | 56–67 | 24,419 / 30,000 (81.4%) | 0 / 480,031 (0%) | 318 |
| 300 s | 1.2 | 17.1 ms | 59.994 s | 5 ms | 1.90650 | 1.41116 | 56 | 67 | 56–69 | 26,022 / 29,996 (86.8%) | 0 / 473,507 (0%) | 314 |
| 360 s | 1.0 | 11.1 ms | ends outside | — | 1.60355 | 1.29889 | 67 | 57 | 56–67 | 25,405 / 29,995 (84.7%) | 0 / 468,877 (0%) | 207 |

The three loaded holds enter and stay, and end at 65, 67, and 67. The returns end at 57, 56, and 57. The 360 s return had been inside and leaves the band for its last 23 bins, about 22 ms, at `z_hat` 1.40105 and 57 SMs. That hold does not enter and stay. Whole-hold in-band shares: 57.2%, 71.3%, 66.7%, 61.7%, 63.6%, 63.6%.

At the end of the loaded holds the integral is −2.96606, −3.44534, and −3.51309. With Ki 0.08 and `sms_per_u` 33.6 those are 7.97, 9.26, and 9.44 SMs of correction. The counts are 9.3, 11.3, and 11.3 above `n_ff` 55.7. The proportional term carries the rest. Fifteen SMs of room is more than the settled integral uses. The rail does not trip.

Against the pulse-skip twin the calibration is the same shape (`n_ff` 55.7 against 56.0, `sms_per_u` 33.6 against 31.3). The loaded last-30 s shares are 75.2%, 83.3%, and 86.8%, beside 91.4%, 85.3%, and 84.4%. The return shares are 87.1%, 81.4%, and 84.7%, beside 90.1%, 91.5%, and 91.6%. The count keeps moving on the returns (179, 318, and 207 changes), where the pulse-skip returns settled to 4 changes. The deepest return is 1.29593, beside 1.28799. Real work did not deepen that undershoot. It did leave the last return outside the band, and it did not reproduce the 90–92% return share.

`dmon` has 422 samples, power 34–217 W (mean 190.8 W), die 48–65 °C, SM clock 997–2887 MHz. The 997 MHz sample is the start. 425 flag samples, every reason word 0, power 34.11–217.41 W.

## 3. Heavier load, clamp 9

`GPUTRONIC_ICLAMP_SMS=9`, same schedule. Log `heavy-c9.txt`. Trace `heavy-c9-1ms.csv`, 51,972,575 bytes, 419,968 data rows, t = 0.00 to 419,999.15 ms. The plot is `heavy-c9-zhat.svg`.

Calibration: `rate_ref=44425312`, `n_ff=55.5`, `dZ/dn=0.0318`, `sms_per_u=31.5`, integral clamp 3.573, printed as 9.00 SMs of 82. `[STATS]`: `integral_on_rail=0` of `pi_ticks=6757556`, `sm_moves=8049`, `active_sms=47..64`, `mean_Z=1.4976`. Every hold's `rail_ticks` sum is 0. The reference is a little under the clamp-15 run (44.4 million against 46.7). `n_ff` and `sms_per_u` still sit with the other holds.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end | SM range | Last 30 s inside | Last 30 s on the clamp | SM changes, last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | 1.2 | 20.1 ms | 59.796 s | 203 ms | 1.79692 | 1.43670 | 53 | 62 | 53–63 | 25,435 / 30,000 (84.8%) | 0 / 467,815 (0%) | 346 |
| 120 s | 1.0 | 13.1 ms | 59.726 s | 273 ms | 1.75129 | 1.30710 | 62 | 52 | 52–62 | 25,932 / 29,996 (86.5%) | 0 / 474,403 (0%) | 201 |
| 180 s | 1.2 | 15.2 ms | 59.956 s | 43 ms | 1.76129 | 1.43715 | 52 | 61 | 52–62 | 24,659 / 29,997 (82.2%) | 0 / 506,764 (0%) | 348 |
| 240 s | 1.0 | 12.0 ms | 59.925 s | 74 ms | 1.62572 | 1.31117 | 61 | 52 | 52–61 | 26,276 / 29,993 (87.6%) | 0 / 527,918 (0%) | 293 |
| 300 s | 1.2 | 14.1 ms | 59.888 s | 111 ms | 1.83314 | 1.45073 | 52 | 61 | 52–64 | 24,667 / 30,000 (82.2%) | 0 / 472,754 (0%) | 344 |
| 360 s | 1.0 | 25.0 ms | 59.812 s | 187 ms | 1.66580 | 1.30575 | 61 | 52 | 52–61 | 25,947 / 30,000 (86.5%) | 0 / 475,363 (0%) | 222 |

All six holds enter and stay. The loaded holds end at 62, 61, and 61. The returns end at 52. Loaded last-30 s shares are 84.8%, 82.2%, and 82.2%. Return shares are 86.5%, 87.6%, and 86.5%, beside the pulse-skip 9 SM shares 87.4%, 86.2%, and 85.0%. The return lows are 1.30710, 1.31117, and 1.30575. The eleventh pass's deepest pulse-skip undershoot, across 6, 9, 12, and 15, was 1.28799 on the 15 SM run. This 9 SM real-work run does not go deeper than that. Whole-hold in-band shares: 71.3%, 62.9%, 67.8%, 63.1%, 67.6%, 63.7%.

The loaded integrals end at −1.82336, −1.72496, and −1.72300. That is 4.59, 4.35, and 4.34 SMs of correction, against a 9 SM budget. The counts are 6.5, 5.5, and 5.5 above `n_ff` 55.5.

`dmon` has 421 samples, power 45–213 W (mean 187.7 W), die 50–64 °C, SM clock 1440–2887 MHz. The 1440 MHz sample is the start. 425 flag samples, every reason word 0, power 45.33–213.69 W.

On this real-work schedule the 9 SM returns all stay in band and the 15 SM run's last return does not. The last-30 s shares are close, and both sit under the pulse-skip 90–92% at clamp 15. No width is picked. The compiled clamp is not widened.

## 4. Heavier load, compiled clamp

No `GPUTRONIC_ICLAMP_SMS`. Log `heavy-compiled.txt`. Trace `heavy-compiled-1ms.csv`, 52,225,217 bytes, 419,948 data rows, t = 0.00 to 419,999.15 ms. The plot is `heavy-compiled-zhat.svg`.

Calibration: `rate_ref=47235612`, `n_ff=55.8`, `dZ/dn=0.0331`, `sms_per_u=30.2`, integral clamp 1.356, printed as 3.28 SMs of 82. `[STATS]`: `integral_on_rail=2490061` of `pi_ticks=6563794`, `sm_moves=17241`, `active_sms=47..65`, `mean_Z=1.5365`. The six holds' `rail_ticks` sum to 2,490,061, the same count, so calibration added none.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end | SM range | Last 30 s inside | Last 30 s on the clamp | SM changes, last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | 1.2 | 0.0 ms | 59.999 s | 0 ms | 1.83956 | 1.41972 | 55 | 60 | 55–65 | 5,404 / 29,999 (18.0%) | 473,374 / 473,374 (100%) | 2078 |
| 120 s | 1.0 | 30.1 ms | ends outside | — | 1.66952 | 1.38466 | 60 | 55 | 55–60 | 24,884 / 29,994 (83.0%) | 0 / 468,847 (0%) | 497 |
| 180 s | 1.2 | 0.2 ms | ends outside | — | 1.88362 | 1.41500 | 55 | 62 | 55–65 | 5,202 / 30,000 (17.3%) | 468,727 / 468,727 (100%) | 2145 |
| 240 s | 1.0 | 0.0 ms | 59.903 s | 96 ms | 1.67283 | 1.36901 | 62 | 55 | 55–62 | 25,523 / 30,001 (85.1%) | 0 / 466,923 (0%) | 394 |
| 300 s | 1.2 | 0.2 ms | 59.999 s | 0 ms | 1.82920 | 1.56128 | 55 | 60 | 58–64 | 5,629 / 29,995 (18.8%) | 471,676 / 471,676 (100%) | 2116 |
| 360 s | 1.0 | 23.2 ms | 59.963 s | 36 ms | 1.80077 | 1.37370 | 60 | 55 | 55–62 | 25,440 / 30,000 (84.8%) | 0 / 467,591 (0%) | 477 |

The rail trip still happens under real work. Each loaded hold ends with the integral at −1.35618, which is the clamp. The correction that represents is 3.28 SMs. The counts end at 60, 62, and 60, which is 4.2, 6.2, and 4.2 above `n_ff` 55.8. The proportional term carries those few steps, and then the count stops. The last 30 s of each loaded hold is entirely on the clamp, and only 18.0%, 17.3%, and 18.8% of those bins are inside the band. The 60 s hold's last bin is `z_hat` 1.57815, and the 300 s hold's last bin is 1.57747. Both are a one-bin touch of the band edge at 1.58, which is why the table prints a 0 ms tail. The 180 s hold ends outside at 1.67283 and 62 SMs. They do not stay in band.

The returns come off the rail. Their last 30 s is 0% on the clamp. A few hundred `rail_ticks` in each return hold (674, 667, and 1,096) are the unwind at the start of the step, not a second trip. The return shares are 83.0%, 85.1%, and 84.8%, ending at 55 SMs. The 120 s return leaves the band for its last 6 bins at `z_hat` 1.41923. Whole-hold in-band shares: 16.4%, 76.1%, 14.8%, 75.0%, 17.1%, 75.8%.

This is the same mechanism as the pulse-skip load against this rail. A scale of 1.2 lengthens each batch, the counted rate falls, and Z rises. Restoring about a fifth of the rate from a feedforward count near 56 is on the order of 11 SMs if the rate tracks the count. The compiled rail is 3.28 SMs. The 9 and 15 SM runs had room and did not trip. The cause is the size of this step against that rail. No wider clamp is proposed.

`dmon` has 422 samples, power 52–219 W (mean 189.4 W), die 50–65 °C, SM clock 1912–2887 MHz. 425 flag samples, every reason word 0, power 56.64–219.57 W, clock 2835–2887 MHz.

## 5. Lighter load, clamp 15

`GPUTRONIC_ICLAMP_SMS=15`, schedule `steps-light.txt`, scale 0.8. Log `light-c15.txt`. Trace `light-c15-1ms.csv`, 51,752,279 bytes, 419,970 data rows, t = 0.00 to 419,999.19 ms. The plot is `light-c15-zhat.svg`. This is the disturbance the eleventh pass left open: the work itself gets smaller, the pulse rate rises, and Z falls, so the loop has to give SMs back. The T11 tally was not repeated on these holds. The round formula is the same one the 1.2 proof measured.

Calibration: `rate_ref=46864636`, `n_ff=56.6`, `dZ/dn=0.0310`, `sms_per_u=32.3`, integral clamp 5.813, printed as 15.00 SMs of 82. `[STATS]`: `integral_on_rail=0` of `pi_ticks=6948933`, `sm_moves=11715`, `active_sms=43..62`, `mean_Z=1.4908`. Every hold's `rail_ticks` sum is 0.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end | SM range | Last 30 s inside | Last 30 s on the clamp | SM changes, last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | 0.8 | 22.1 ms | 59.939 s | 60 ms | 1.66182 | 1.22577 | 55 | 44 | 43–55 | 21,742 / 29,991 (72.5%) | 0 / 497,730 (0%) | 497 |
| 120 s | 1.0 | 20.2 ms | 59.988 s | 11 ms | 1.84371 | 1.44950 | 44 | 54 | 44–56 | 24,843 / 30,001 (82.8%) | 0 / 478,105 (0%) | 511 |
| 180 s | 0.8 | 12.1 ms | 59.991 s | 8 ms | 1.84467 | 1.27609 | 54 | 46 | 43–54 | 22,029 / 30,000 (73.4%) | 0 / 513,352 (0%) | 560 |
| 240 s | 1.0 | 31.2 ms | 59.914 s | 85 ms | 1.85764 | 1.44792 | 46 | 54 | 46–56 | 24,361 / 30,000 (81.2%) | 0 / 502,697 (0%) | 500 |
| 300 s | 0.8 | 10.3 ms | 59.968 s | 31 ms | 1.69790 | 1.27342 | 54 | 43 | 43–54 | 21,640 / 29,996 (72.1%) | 0 / 481,134 (0%) | 509 |
| 360 s | 1.0 | 16.0 ms | 59.960 s | 39 ms | 1.83496 | 1.44651 | 43 | 54 | 43–57 | 24,453 / 30,000 (81.5%) | 0 / 512,578 (0%) | 500 |

All six holds enter and stay. The tails are short, and the count is still moving. The 0.8 steps end at 44, 46, and 43 SMs. Their last-30 s shares are 72.5%, 73.4%, and 72.1%. The deepest dip is 1.22577. The returns end at 54, with shares 82.8%, 81.2%, and 81.5%. Whole-hold in-band shares: 46.6%, 63.6%, 53.7%, 62.6%, 54.3%, 62.5%.

The 0.8 integrals end at 4.48750, 4.54034, and 4.53347, against a clamp of 5.813. That is 11.60, 11.73, and 11.71 SMs of correction, inside the 15 SM budget. The counts are 12.6, 10.6, and 13.6 below `n_ff` 56.6. The rail does not trip. The compiled clamp was not repeated on this schedule. The directive asked for the lighter load at clamp 15.

`dmon` has 421 samples, power 33–218 W (mean 181.8 W), die 49–65 °C, SM clock 1170–2880 MHz. The 1170 MHz sample is the start. 425 flag samples, every reason word 0, power 33.79–218.09 W.

## 6. Z = 1 on the heavier load

`GPUTRONIC_TARGET_Z=1`, compiled clamp, schedule `steps-heavy.txt`. Log `z1-heavy.txt`. The log line is `[TARGET] GPUTRONIC_TARGET_Z=1.000 (unset keeps 1.5)`, because this run was before the macro moved. Trace `z1-heavy-1ms.csv`, 51,772,201 bytes, 419,952 data rows, t = 0.00 to 419,999.21 ms. The plot is `z1-heavy-zhat.svg`. The band here is `|z_hat - 1.0| <= 0.08`.

Calibration: `rate_ref=46942300`, `n_ff=82.0`, `dZ/dn=0.0163`, `sms_per_u=61.4`, integral clamp 0.668, printed as 3.28 SMs of 82. `[STATS]`: `integral_updates=0`, `integral_on_rail=0` of `pi_ticks=7078477`, `sm_moves=0`, `active_sms=82..82`, `mean_Z=1.0958`. The integral is 0.00000 at the end of every hold. The count never leaves 82.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end | SM range | Last 30 s inside | Last 30 s on the clamp | SM changes, last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | 1.2 | 14.1 ms | ends outside | — | 1.37284 | 1.02435 | 82 | 82 | 82–82 | 0 / 29,999 (0.0%) | 0 / 507,745 (0%) | 0 |
| 120 s | 1.0 | 0.1 ms | ends outside | — | 1.26318 | 0.95607 | 82 | 82 | 82–82 | 28,751 / 29,990 (95.9%) | 0 / 587,474 (0%) | 0 |
| 180 s | 1.2 | 0.1 ms | ends outside | — | 1.37139 | 1.08047 | 82 | 82 | 82–82 | 0 / 29,999 (0.0%) | 0 / 485,715 (0%) | 0 |
| 240 s | 1.0 | 0.2 ms | 59.314 s | 685 ms | 1.34485 | 0.95485 | 82 | 82 | 82–82 | 28,943 / 30,000 (96.5%) | 0 / 487,682 (0%) | 0 |
| 300 s | 1.2 | 14.2 ms | ends outside | — | 1.41010 | 1.04120 | 82 | 82 | 82–82 | 0 / 30,001 (0.0%) | 0 / 467,151 (0%) | 0 |
| 360 s | 1.0 | 0.0 ms | 59.913 s | 86 ms | 1.30880 | 0.95621 | 82 | 82 | 82–82 | 28,555 / 30,000 (95.2%) | 0 / 473,971 (0%) | 0 |

On a 1.2 step the count is already 82, so the loop cannot add SMs. Z sits above 1.08 for the rest of each loaded hold. Those holds end at 1.26318, 1.28587, and 1.19740. The last 30 s of each is 0% inside the band. Whole-hold in-band shares on the loads are 0.0%. That is the 1.2 share of the rate, seen as Z, with the actuator already at the ceiling.

The returns to scale 1.0 stay at 82 as well. Their last-30 s shares are 95.9%, 96.5%, and 95.2%. The 240 s and 360 s returns enter and stay, ending at 1.05570 and 0.98735. The 120 s return ends at 1.08047, which is 0.00047 above the band edge, so that hold is "ends outside" with 95.9% of its last 30 s inside. The integral does not step, and the count does not step. Whole-hold return shares: 95.9%, 94.3%, 95.3%.

The eleventh-pass pulse-skip Z = 1 load is the twin. There the integral drifted by a few thousandths, the returns undershot to about 0.915, and the count stepped down to 79, with one return meeting the limit-cycle definition. Real work does not do that. The rounds change and the pulse is kept, so when the scale returns to 1.0 the plant is back at the calibration work, Z is near 1, and the error stays inside the deadband. The integral stays at 0.

`dmon` has 421 samples, power 55–222 W (mean 218.6 W), die 51–68 °C, SM clock 2835–2887 MHz. 416 of the 421 clocks are 2835 MHz. 425 flag samples, every reason word 0, power 55.46–221.89 W.

## The compiled target

The directive left `GPUTRONIC_GOLD_TARGET_Z` at 1.5 until Ben decided. He then asked for Z = 1 as the home: full launch, the loop out of the stock driver's way, and a commanded Z above 1 only when a slower rate is the point. This hold is that measurement. The command stayed at 82, the integral never moved, and the 1.2 steps sat above the band because there was no SM left to add.

Idling SMs when Z rises was not adopted. The law adds SMs when Z is above target. At Z = 1 the count is already the launch. Measured plants get slower with fewer SMs, so taking SMs away would deepen the stall the rise is reporting.

`GPUTRONIC_GOLD_TARGET_Z` is now `1.0f` in `include/gputronic.h`. `run 3` with every switch unset, log `unset-home.txt`, prints `[CTRL] Gold | target=1.00 Kp=0.550 Ki=0.0800 Kd=0.0000 deadband=0.080` and launches 82 blocks, leaving out SMs 82 and 83. There is no `[TARGET]` line. The gold, transfer, and episode check gates set `cfg.target_z = 1.5f` after `gputronic_config_gold`, because those gates pass only when the closed rate falls below 90% of free run. A home of 1.0 would measure free run and fail that test. The function-frame row says so, and the README copy matches.

## Clocks

Read only. No thermal or power term was added. On every hold the reason word is 0 for all 425 flag samples. After the start sample the SM clock stays inside the boost steps this card has used on earlier full-pipe holds, 2835–2887 MHz. The Z = 1 hold draws more power (mean 218.6 W, peak die 68 °C) because the count stays at 82. The target-1.5 holds sit near 180–190 W and peak at 64–65 °C. The count steps line up with Z. No decoupling is proposed. The 83 °C test stop was not reached.

## Switches

| Switch | Unset | This pass |
| --- | --- | --- |
| `GPUTRONIC_WORK_LOAD` | off | `1` on the proof-work run and the five holds |
| `GPUTRONIC_ICLAMP_SMS` | compiled 3.28 SMs | 15, 9, or unset, as named on each hold |
| `GPUTRONIC_TARGET_Z` | compiled target | `1` only for `z1-heavy.txt` |
| `GPUTRONIC_WORK_COUNTER` | off | on for the 16 s proof only |
| `GPUTRONIC_SM_ACCUM_SM` | off | `0` for the ops-per-pulse proof only |
| Idle-exit, guest | off | unset |

The commanded check logs have no `[WORK] real-work load` line. The switch is off when the environment does not set it, and it stays off when `GPUTRONIC_PEAK_ACC` is unset.

## Item 6

Not run. The eleventh pass already logged that a second kernel's one-second rate stays 0 while Gold is resident, and closed that account as one grid and one total. Green contexts and launching the guest first are still Ben's call. A short trial would not go into the controller, and it would not change the result of this pass. The result is the real-work load and the home target.

## Item 4

- **The load is real.** Yes. Ops per pulse move with the round count, the pulse-to-batch ratio stays about 1.000, and the open-loop rate moves by the 1.2 share.
- **Settling time.** No. On the heavier clamp-15 schedule the three loaded holds enter and stay, and the 360 s return ends outside. On the lighter clamp-15 schedule every hold enters and stays.
- **Overshoot.** Reported above, for every step, beside the pulse-skip twin where this pass has one. No threshold is set.
- **Returns.** No. The heavier clamp-15 returns are 87.1%, 81.4%, and 84.7% in the last 30 s, beside 90.1%, 91.5%, and 91.6%, and the last of them ends outside. The lighter returns enter and stay at 82.8%, 81.2%, and 81.5%. The 9 SM real-work returns enter and stay at 86.5%, 87.6%, and 86.5%.
- **Guardrail.** Yes. The trip rate is in each section. Clamp 15 and clamp 9 are 0 on every hold. The compiled-clamp trip is the 1.2 step against a 3.28 SM rail. Nothing at 15 SMs tripped, so nothing there is left open.
- **Driver orthogonality.** Yes. Clocks and flags are logged on every hold. The reason word is 0. The loop did not fight the driver.
- **No regressions.** Yes, with the target argument in the next section. Every new switch is off by default. The `[CTRL]` line keeps Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, and launch 82. All three checks pass.

## Checks

Unset, on the binary whose compiled target was still 1.5, before the holds:

| Check | This binary | Eleventh pass |
| --- | --- | --- |
| Gold | 1.432 | 1.430 |
| Transfer | 1.448 | 1.457 |
| Episode | 1.471 | 1.500 |

Gold: free-run 564,075.8 pulse/s, closed 392,527.9, `useful_drop=YES` (`err=0.068`), dyno PASS, closedloop PASS. Transfer: free-run 250,073.7 pulse/s (8.1944 GFLOP/s), closed 170,970.3 (5.6024), PASS. Episode: free-run 627,002.4 pulse/s (20.5456 GFLOP/s), closed 428,845.1 (14.0524), `window=OK`, PASS. Each `[CTRL]` line is target 1.50, Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080.

After the macro moved to 1.0, the dyno half of each check boots at target 1.00 and the closed-loop half commands 1.50. Both halves pass. The closed-loop means:

| Check | Commanded 1.5 | Eleventh pass |
| --- | --- | --- |
| Gold | 1.449 | 1.430 |
| Transfer | 1.441 | 1.457 |
| Episode | 1.493 | 1.500 |

Gold: free-run 547,334.3 pulse/s, closed 376,369.6, `err=0.051`, PASS. Transfer: free-run 264,723.3 pulse/s (8.6745 GFLOP/s), closed 189,715.7 (6.2166), PASS. Episode: free-run 712,799.5 pulse/s (23.3570 GFLOP/s), closed 477,272.2 (15.6393), `window=OK`, PASS. The closed-loop `[CTRL]` line is target 1.50 with the same gains, deadband, and launch of 82.

## What was argued

**The compiled target.** The directive said to leave `GPUTRONIC_GOLD_TARGET_Z` at 1.5 because Ben had not decided. He then asked for Z = 1 as the home, and to proceed. The Z = 1 real-work hold is the log that decision was waiting on: 82 SMs, integral still, Z above the band on the heavier steps. The macro is now 1.0. The check gates command 1.5 so they still require a closed-rate cut. Kp and Ki were not changed. The clamp was not widened, and 9 was not adopted as a new default.

**Item 6.** Not run, for the reason above.
