# Fourteenth pass, 2026-10-18

Branch `test-results/2026-10-18_fourteenth-pass`, started from `0255d1d`. Logs in `logs/2026-10-18_fourteenth-pass/`. The card is an RTX 5080, driver 595.99.02, built with CUDA 12.8, `sm_120`. `nvidia-smi` reports driver CUDA 13.2. The build is 12.8. The suite ran from 17:10:04 to 17:39:38 PDT. No run was stopped for temperature. The hottest die sample is 66 °C. Every `clocks_event_reasons.active` sample in the flag files is 0.

The directive is on `origin/controller-fixes` at `4b94e4f` (`docs/directives/fourteenth-pass-2026-10-18.md`). That branch is not merged. The hold length, the retry bound, and the runs that were skipped are argued at the end.

The thirteenth pass left two plants at the same settings. The low group was `rate_ref` 47.0–47.2 million and hunted. The high group was 49.1–49.6 million and was quiet. This session did not land in the high group. The clusters below are the measured `rate_ref` values. A label that calls everything under 48 million "low" would mix 47.1 million with 43.8 million, 19.8 million, and 1.4 million. Those are different plants.

## What this pass changed

One test switch, `GPUTRONIC_RATE_WIN_US`. It sets the rate window in microseconds for that run. Unset keeps 8000 (8 ms): `gputronic_start` stores the literal `0.008`, which is the value the old compare used, and it prints nothing. A set value is floored at 500 and capped at 100000, and the run prints `[RATE] window N us (unset keeps 8000)`. The 0.70 / 0.30 mix is unchanged. The help line is in `help.txt`.

`unset-home.txt` has no `[RATE]` line and no `[TARGET]` line. Its `[CTRL]` line is target 1.00, Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, kalman 1, two-sided. The launch line is 82 blocks on 84 SMs, and the left-out pair is 82 and 83.

Kp, Ki, Kd, the deadband, the 82-block launch, floor 2, and the compiled clamp were not changed. `GPUTRONIC_ICLAMP_SMS=15` is the hunting configuration for the closed-loop runs. It was not adopted as a default. The Kalman error path is unchanged: the PI error is `z_hat`, and Kd is 0 so `dZ/dt` is unused.

The function frame and the copy at the end of `README.md` name the switch in the same words. From `# GPUtronic 1.0 Gold: Function Reference` the two copies match.

## How the holds were run

Full-pipe switches on every hunting and fullwidth launch: `GPUTRONIC_PEAK_ACC=32`, `GPUTRONIC_PEAK_ROUNDS=512`, `GPUTRONIC_THREADS=512`, `GPUTRONIC_PULSE_LOCAL=1`, `GPUTRONIC_CTRL_CACHE=1`, `GPUTRONIC_TRACE_BIN_US=1000`, `GPUTRONIC_WORK_LOAD=1`. Idle-exit and guest were unset. The two pulse-path launches leave `GPUTRONIC_PULSE_LOCAL` unset. The work-counter launches set `GPUTRONIC_WORK_COUNTER=1`. The accumulator launches set `GPUTRONIC_SM_ACCUM=1`. Both of those add traffic.

Closed-loop launches set `GPUTRONIC_ICLAMP_SMS=15` and `GPUTRONIC_TARGET_Z=1.5`. Fullwidth launches do not. Fullwidth calibrates, then holds all 82 SMs with `sm_moves=0` and `integral_updates=0`.

The schedule is `steps-item1.txt`: `20000 1.2`, `65000 1.0`, duration 110 s. That is a 20 s pre-roll at scale 1.0, 45 s at 1.2, and 45 s at 1.0. The 1 ms trace clock starts at the end of calibration. On that clock the load is t = 20–65 s and the return is t = 65–110 s. The per-tick clock starts when the control thread starts, so its first ~3 s is calibration and the same holds sit about 3 s later. The last 30 s and the last 10 s are anchored on the hold's last sample.

The band is `|z_hat − target| ≤ 0.08`, applied to `z_hat` rounded to five decimal places, so a printed 1.58000 is inside. "Enters and stays" is the first in-band bin after the last out-of-band bin. The tail is that entry to the hold's last sample. A limit cycle is exactly two SM counts in the named window, each at least five times, with at least six changes. No hold in this pass meets that test on its last 30 s.

An SM move in the delay figures counts only after t = 3 s on the per-tick clock, and only when the new count is still in force at the next window close. The `z_raw` on the move's own row is the window that just closed. It is written on the same tick as the new count, so it is not the response. The first Z change is the next close whose `z_raw` differs from that value by at least 5e-6. Half the change is halfway from that `z_raw` toward the extreme in the expected direction over the next 80 ms. More SMs should take Z down. "In windows" uses the median gap between `rate_ema` changes on that run.

The 136 ms figures use the 1 ms series downsampled to 4 ms. A Goertzel scan runs from 0.5 to 60 Hz at 0.25 Hz, then the best peak between 2 and 20 Hz is refined by ±0.4 Hz at 0.02 Hz. The narrow-band share is the power in 7.25–7.45 Hz over that 0.5–60 Hz sum. The autocorrelation lag is the strongest lag from 50 to 400 ms on the same 4 ms grid. 136.0 ms is lag 34. Fixed-count Z stats use only the longest eight stretches of at least 500 ms at one count. Slow wander often wins the Goertzel power and parks the peak at 200–500 ms. The autocorrelation is the period quoted below. The method is `numbers.py`.

Per-SM rate is `rate_ema / active_sms` on the 1 ms trace, median over the hold, ignoring a zero rate. Ten-second chunks of that ratio are how a mid-hold plant change would show up.

Clocks and flags were read with `nvidia-smi`. Display state was read once a second: `timestamp,display_active,display_mode,pstate,utilization.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active`. `display_mode` returns `[Requested functionality has been deprecated]`. `display_active` is the field that moves. The process list was taken at the start and end of every launch.

The per-tick file is committed as `*-tick-cut.csv.gz`. A 110 s closed-loop file is 180–300 MB raw, and GitHub rejects a file at 100 MB. The cut keeps every row with `time_s` ≤ 3 s, and every later row whose printed `z_raw`, `rate_ema`, or `active_sms` differs from the previous row. The `rate_ema` rows are the window closes. A window whose `z_raw` prints the same five decimals would otherwise disappear, and the delay is measured on those closes. `cut-manifest.txt` lists raw rows, kept rows, and gzip bytes. The raw per-tick files were deleted after each gzip read back. The raw `*-smaccum.csv` files are 102 MB each and were not committed. `smaccum-summary.csv` is the per-SM rate. The 1 ms traces are committed whole. `GPUTRONIC_NO_CSV` stayed unset, so the trace and the per-tick file were both written.

Round counts follow the device formula `(512 × scale_milli + 500) / 1000`. Scale 1.2 is 614 rounds and scale 1.0 is 512. This pass did not repeat the op tally. The load's per-SM pulse rate sits near `512/614` of the return on the runs whose per-SM rate is flat, which is the same ratio.

## 1. The hunting run, cycle by cycle

Two item-1 launches, same settings. The first is the thirteenth low group. The second is slower. Neither is the 49 million group.

### i1-a, rate_ref 47,111,548

Log `i1-a.txt`. Trace `i1-a-1ms.csv`, 13,553,651 bytes, 109,987 data rows, t = 0.000 to 109.999 s. Per-tick cut `i1-a-tick-cut.csv.gz` keeps 69,494 of the raw 1,765,532 rows, 906,846 bytes gzipped. The raw file ended at 113.002 s.

Calibration: `rate_ref=47111548`, `n_ff=56.4`, `dZ/dn=0.0317`, `sms_per_u=31.5`, integral clamp 5.945, printed as 15.00 SMs of 82. Table rates: 61 SMs 34,811,983.9, 41 SMs 23,704,827.7, 20 SMs 11,775,326.0, 2 SMs 1,155,188.2. `[STATS]`: `integral_on_rail=0` of `pi_ticks=1711687`, `sm_moves=2469`, `p_only_moves=14`, `active_sms=48..67`, `mean_Z=1.5078`, `std_Z=0.0854`.

The first window is already in this band. Warm-up closes (0–0.7 s): 87 windows, median `rate_ema` 47,647,536, first 48,351,032, last 48,097,036. Sustained window (0.7–1.8 s): 137 windows, median 47,766,616, first 47,574,896, last 45,974,452. The reference is the pulse delta over that sustained second, not the last EMA sample.

| Hold | In band | Last 30 s | Last 10 s | SM range | End | Changes | Last 30 s changes | Last 10 s changes | Enters | Tail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Load 20–65 s | 21,613 / 44,990 (48.0%) | 20,131 / 29,990 (67.1%) | 5,822 / 9,995 (58.2%) | 55–67 | 65 | 1,574 | 717 | 331 | 44.996 s | 3 ms |
| Return 65–110 s | 29,177 / 44,997 (64.8%) | 25,929 / 30,000 (86.4%) | 7,985 / 10,000 (79.8%) | 57–65 | 58 | 758 | 204 | 113 | 44.979 s | 20 ms |

The load's last 30 s uses five counts, 63 through 67, and 717 changes. The return's last 30 s uses 57, 58, 59, 61, and 62, and 204 changes. Neither is a two-count limit cycle. The load meets "enters and stays" by 3 ms. The last 10 s is still 58.2% in band with 331 changes, so the hold has not sat down. End of load: `z_hat` 1.53928, integral −2.99919. End of return: `z_hat` 1.48594, integral −1.09471.

Per-SM rate, load, median 473,632. Ten-second chunks: 480,370, 471,427, 480,034, 465,672, 470,166. Return median 555,696. Chunks: 575,110, 560,318, 546,944, 555,636, 543,456. The return drifts from about 575k to about 543k inside the hold. It does not step to the thirteenth high group's ~602k. Load over return is 473,632 / 555,696 = 0.852. On the flat runs below, the same ratio is about 0.83, which is 512/614.

The per-tick file has 14,061 window closes, median gap 8.03 ms. Moves after calibration: pre-roll 48, load 1,635 (last 30 s 736, last 10 s 340), return 786 (last 30 s 212, last 10 s 116). Those three regions sum to 2,469, the `[STATS]` `sm_moves`. The other four `active_sms` changes in the file are the calibration probes. Inside a window the count reversed on 239 of the 14,061 closes: the count left a value and came back before the next `rate_ema` change.

Control-loop samples in the per-tick file run from about 98,000 iterations per second near the start down to about 16,500 later. The nominal period is 10 µs. The measured loop is slower while it writes the CSV. No tick-period switch was added.

### One and a half seconds of the hunt

The densest 1.6 s of SM moves in the load is tick time 27.196–28.796 s: 126 moves, 199 window closes. The file is `i1-a-excerpt.csv`. These are the closes, which is the grain Z moves on. Between closes the per-tick rows repeat the same `z_raw`. The opening of that stretch:

| time_s | z_raw | z_hat | error | integral | active_sms |
| --- | --- | --- | --- | --- | --- |
| 27.199574 | 1.59939 | 1.61128 | −0.11128 | −0.58953 | 60 |
| 27.207600 | 1.65362 | 1.60967 | −0.10967 | −0.59034 | 60 |
| 27.215638 | 1.65145 | 1.65321 | −0.15321 | −0.59156 | 61 |
| 27.223642 | 1.72925 | 1.66622 | −0.16622 | −0.59277 | 61 |
| 27.231708 | 1.65741 | 1.71564 | −0.21564 | −0.59460 | 62 |
| 27.239774 | 1.63629 | 1.65341 | −0.15341 | −0.59589 | 61 |
| 27.247800 | 1.60573 | 1.63049 | −0.13049 | −0.59699 | 60 |
| 27.255841 | 1.59945 | 1.60453 | −0.10453 | −0.59784 | 60 |
| 27.263864 | 1.58865 | 1.59739 | −0.09739 | −0.59864 | 60 |
| 27.271865 | 1.61112 | 1.59290 | −0.09290 | −0.59935 | 59 |
| 27.279889 | 1.65259 | 1.61899 | −0.11899 | −0.60024 | 60 |
| 27.287909 | 1.62869 | 1.64806 | −0.14806 | −0.60145 | 60 |

The count steps 60, 61, 62, 61, 60, 59, 60 inside a few windows. `z_raw` swings from about 1.59 to 1.73 over the same stretch. The integral crawls by about −0.001 per window. The integer count is doing the moving.

### Delay, from the per-tick file

2,092 moves were still in force at the next close. All 2,092 show a later `z_raw` change. The median wait is 7.86 ms, 0.98 windows. 2,091 of 2,092 land inside 1.5 windows. Half the change is available on 1,923 moves, median 23.93 ms, 2.98 windows. 169 moves had no half-change inside 80 ms.

Z follows the count on the next window, and it is halfway there about three windows later. That is the same shape the thirteenth pass reported for both of its groups.

### Period

On the load, the longest fixed-count stretches (6 of them, each ≥ 500 ms) have median `z_raw` SD 0.0376. Their Goertzel peak sits near 135.8 ms and the 7.25–7.45 Hz share is 5.5% of the 0.5–60 Hz sum. Their autocorrelation lag is 264 ms, not 136 ms. The count series over the whole load autocorrelates at 136.0 ms. Its Goertzel peak is 111.1 ms and the narrow-band share of the count series is 0.8%.

On the return, fixed-count `z_raw` SD is 0.0357, the peak is 181.7 ms, the narrow-band share is 0.6%, and the autocorrelation lag is 152 ms. The count series over that return autocorrelates at 48.0 ms, with a Goertzel peak at 200 ms and a narrow-band share of 0.0%. This return's count does not prefer 136 ms.

The whole-load `z_raw` series, count moving, has autocorrelation 0.548 at 128 ms, 0.586 at 136 ms, 0.545 at 144 ms, and 0.595 at 272 ms. 136 ms is a local peak. The series is also slow: the coefficient at 48 ms is already 0.489. The clean 136 ms line in this session is on the fixed-count runs in section 3, and on the count series of the other closed-loop loads.

### i1-b, same settings, rate_ref 43,830,116

Log `i1-b.txt`. Trace `i1-b-1ms.csv`, 13,517,151 bytes, 109,995 rows. This is the comparison launch. It is not the high group.

`n_ff=55.6`, `dZ/dn=0.0307`, `sms_per_u=32.6`, clamp printed 15.00 SMs. `[STATS]`: `sm_moves=2078`, `integral_on_rail=0`, `mean_Z=1.4916`, `std_Z=0.0778`, `active_sms=47..69`. Table: 32,865,073.1, 22,501,306.2, 10,784,121.6, 1,081,534.7. The shape matches the 47 million table, about 7% lower.

The first warm-up window is 47,142,312. The last warm-up window is 44,108,328. The sustained median is 43,746,224. The drop is inside the 0.7 s warm-up, before the table probes, with the cache still at 1000 µs and the PI not commanding yet.

Load last 30 s: 23,269 / 29,997 (77.6%) in band, 434 SM changes, counts 58–63, SM range over the hold 52–63, ends outside the band at `z_hat` 1.60870. Return last 30 s: 23,124 / 30,001 (77.1%), 231 changes, enters with a 52 ms tail. Per-SM median 484,029 on the load and 582,012 on the return. The return chunks are flat (583,112 down to 581,231). Fixed-count autocorrelation is 136.0 ms on both holds. Narrow-band share 11.1% on the load and 21.2% on the return. `z_raw` SD at fixed counts 0.0406 and 0.0421.

Delay: median 7.86 ms to the first Z change (0.98 windows), 24.07 ms to half of it (3.00 windows). 1,738 of 1,739 first changes fall inside 1.5 windows. Reversals inside a window: 208 of 14,060 closes.

There is no high-group run to put beside these.

## 2. Did a faster control rate stop the hunting?

No.

The same-group comparison is three launches in the 47 million band. The baseline is the item-1 settings (`i2-base`). Then the window is 4 ms (`i2-w4`) and 2 ms (`i2-w2`), with the cache left at 1000 µs. The cache trials and the combined trial landed on a different plant. They are in the table so the move counts are on the record, and they are not the comparison.

| Run | rate_ref | What changed | sm_moves | Load last 30 s in band | Load changes, last 30 s / 10 s | Return last 30 s in band | Return changes, last 30 s / 10 s | Fixed-count z_raw SD | Count autocorr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| i2-base | 46,960,432 | window 8 ms, cache 1000 µs | 2,294 | 20,799 / 30,000 (69.3%) | 600 / 137 | 20,405 / 29,999 (68.0%) | 247 / 4 | 0.0424 / 0.0424 | 136 / 136 ms |
| i2-w4 | 47,422,684 | window 4000 µs | 5,916 | 21,885 / 30,000 (73.0%) | 1,427 / 396 | 18,306 / 29,993 (61.0%) | 1,106 / 224 | no 500 ms stretch | 136 / 136 ms |
| i2-w2 | 47,023,244 | window 2000 µs | 16,566 | 23,769 / 29,995 (79.2%) | 2,829 / 810 | 17,456 / 29,995 (58.2%) | 3,048 / 787 | no 500 ms stretch | 136 / 136 ms |
| i2-c250 | 43,762,896 | cache 250 µs | 2,320 | 22,691 / 30,001 (75.6%) | 553 / 98 | 22,852 / 29,991 (76.2%) | 259 / 32 | 0.0448 / 0.0457 | 136 / 48 ms |
| i2-c100 | 19,779,312 | cache 100 µs | 6,455 | 12,590 / 30,001 (42.0%) | 1,686 / 583 | 20,350 / 29,994 (67.8%) | 917 / 311 | no 500 ms stretch | 48 / 48 ms |
| i2-both | 44,398,800 | cache 100 µs and window 2000 µs | 15,931 | 20,453 / 30,001 (68.2%) | 3,051 / 865 | 9,735 / 30,001 (32.4%) | 4,211 / 1,437 | no 500 ms stretch | 136 / 136 ms |

`i2-both` is an extra. The directive asked for the cache and the window separately. `integral_on_rail=0` on every one of these runs. No last-30 s window is a two-count limit cycle.

### The 47 million trio

`i2-base` matches `i1-a`. `[STATS]` `mean_Z=1.4984`, `std_Z=0.0881`, `active_sms=48..66`, clamp printed 15.00 SMs. Per-SM median 483,625 on the load (chunks 484,019 to 481,892) and 581,359 on the return (chunks 582,215 to 581,476). Both holds are flat. Load over return is 0.832, against 512/614 = 0.834. Fixed-count narrow-band share is 6.9% on the load and 21.5% on the return, and the return's Goertzel peak is 136.8 ms. The return's whole-hold `z_raw` autocorrelation is 0.616 at 128 ms, 0.695 at 136 ms, 0.633 at 144 ms, and 0.675 at 272 ms. 272 ms is two periods of the same wave. Delay: 7.88 ms (0.98 windows) to the first Z change, 24.06 ms (3.00 windows) to half of it. Reversals: 233 of 14,065 windows. Median window 8.03 ms. Warm-up median `rate_ema` 47,407,932, sustained median 47,378,978. Low from the first window.

`i2-w4` prints `[RATE] window 4000 us (unset keeps 8000)`. `rate_ref=47422684`, still the same cluster. `std_Z=0.0936`, `active_sms=48..68`, `sm_moves=5916`. Median window 4.03 ms. The first Z change arrives at 3.84 ms (0.95 of the new window) and half the change at 15.75 ms (3.91 windows). The loop sees Z sooner. The count moves more: load last 30 s has 1,427 changes across eight counts (61–68), against 600 changes on the baseline. The return's last 30 s in-band share falls from 68.0% to 61.0%. The count never holds for 500 ms, so there is no fixed-count `z_raw` SD to put beside the baseline's 0.0424. The count series still autocorrelates at 136 ms on both holds. Warm-up and sustained medians are both about 48.2 million. Reversals: 482 of 28,014 windows.

`i2-w2` prints `[RATE] window 2000 us`. `rate_ref=47023244`, `n_ff=56.1`, `std_Z=0.1012`, `active_sms=47..75`, `sm_moves=16566`. Median window 2.03 ms. First Z change 1.80 ms (0.89 windows), half the change 11.17 ms (5.50 windows). Load last 30 s: 2,829 changes across 13 counts (60–72), in-band share 79.2%. Return last 30 s: 3,048 changes, in-band share 58.2%. Again no 500 ms hold at one count. Per-SM rate stays in the cluster: load median 489,516, return 588,977, return chunks 589,483 to 589,247. Count autocorrelation 136 ms on both holds. Reversals: 1,335 of 55,556 windows. Warm-up and sustained medians are about 48.2 million.

A shorter window follows the same 136 ms wave and spends more moves on it. It does not remove it.

### The launches that left the 47 million band

`i2-c250`, cache 250 µs, window still 8 ms. `rate_ref=43762896`. The first warm-up window is already 42,562,568. Move counts sit next to `i1-b`, which reached 43,830,116 at a 1000 µs cache. The 250 µs period is not what put the run in this cluster, and it did not stop the hunt: 2,320 moves, load last 30 s still 553 changes, fixed-count `z_raw` SD 0.0448. Delay 7.87 ms and 24.01 ms, the 8 ms-window figures. Per-SM load median 476,843, return 556,026.

`i2-c100`, cache 100 µs. `rate_ref=19779312`. The first warm-up window is 39,912,416 and the last warm-up window is 19,851,494. The sustained median is 20,178,424. The plant fell during the 0.7 s warm-up, before the table, while the 100 µs cache was already on. The table is not a scaled copy of the 47 million table: Z is 1.785 at 61 SMs, 3.446 at 41, 9.824 at 20, and 112.282 at 2. `[STATS]` `sm_moves=6455`, `std_Z=0.1760`, `active_sms=55..77`, rail 0. Per-SM rate on the load is 173,559, about a third of the 47 million load. The count autocorrelation is 48 ms. The count reached 77. The 15 SM clamp did not trip. This is not a same-group test of the cache.

`i2-both`, the extra, cache 100 µs and window 2 ms. `rate_ref=44398800`, `sm_moves=15931`, `std_Z=0.1304`, `active_sms=53..75`, rail 0. The load ends outside the band. The return's last 30 s is 32.4% in band with 4,211 changes. Per-SM load median 431,637. More hunting, on the slower cluster.

## 3. Where the low rate comes from

The low rate is the plant's work rate. It is present with the PI open, and it is present in the first calibration window. The 49 million band did not appear, so this session does not show what produces that band.

### Eight fullwidth launches, PI open

Order: `fw-01`, `cl-01`, `fw-02`, `fw-03`, `cl-02`, `fw-04`, `fw-05`, `cl-03`, `fw-06`, `fw-07`, `fw-08`. Each fullwidth hold is 15 s at 82 SMs after the same calibration. `sm_moves=0` and `integral_updates=0` on all eight. The count does not move, so a 136 ms line here is in Z, not in the actuator.

| Launch | rate_ref | 61 SMs | 41 SMs | 20 SMs | 2 SMs | Per-SM, hold | z_raw SD | 7.25–7.45 Hz | ac lag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fw-01 | 47,258,972 | 35,655,277.8 | 23,902,677.4 | 11,823,465.8 | 1,147,588.3 | 581,155 | 0.0295 | 0.1% | 136 ms |
| fw-02 | 47,252,276 | 35,077,104.1 | 24,168,947.6 | 11,623,316.6 | 1,154,078.3 | 580,575 | 0.0304 | 0.0% | 136 ms |
| fw-03 | 47,153,056 | 35,051,683.0 | 23,809,796.7 | 11,799,642.2 | 1,175,325.0 | 580,618 | 0.0301 | 0.2% | 136 ms |
| fw-04 | 46,979,676 | 35,557,091.3 | 23,440,594.4 | 11,551,390.2 | 1,160,763.7 | 580,444 | 0.0299 | 0.6% | 136 ms |
| fw-05 | 46,995,668 | 34,604,152.8 | 23,654,135.3 | 11,790,947.8 | 1,175,045.9 | 579,845 | 0.0315 | 1.4% | 136 ms |
| fw-06 | 47,168,244 | 35,571,840.1 | 23,795,394.7 | 11,653,236.8 | 1,174,031.4 | 579,545 | 0.0315 | 1.1% | 136 ms |
| fw-07 | 47,173,700 | 35,421,823.5 | 24,211,626.3 | 11,763,328.5 | 1,161,960.6 | 580,070 | 0.0300 | 1.1% | 136 ms |
| fw-08 | 47,035,232 | 32,799,289.3 | 24,006,429.8 | 11,794,062.1 | 1,162,094.0 | 580,357 | 0.0311 | 0.2% | 136 ms |

The eight references span 46,979,676 to 47,258,972. That is the thirteenth low group. The 61 SM probe on `fw-08` is lower than the others. The 15 s hold at 82 SMs is not: its per-SM median is 580,357, and the two 10 s chunks are 580,125 and 580,985. `fw-06` dips from 580,856 to 576,701 across its two chunks. The other holds are flat. None of them steps to ~602k.

`fw-01` autocorrelation of `z_raw`, count fixed at 82: 0.438 at 128 ms, 0.560 at 136 ms, 0.429 at 144 ms, 0.544 at 272 ms. The lag is real. The narrow-band share is 0.1% because most of the 0.5–60 Hz power is elsewhere. The Goertzel peak on this hold is 333 ms. Quoting the narrow band as "a third of the power" would not describe these files.

`fw-01` calibration from the per-tick file, PI not updating: warm-up median 47,554,476 (first 46,355,200), sustained median 47,837,856. Low from the first window.

The three interleaved closed loops are the same plant. `cl-01` 47,169,252 and 2,269 moves, `cl-02` 47,239,100 and 2,419 moves, `cl-03` 47,122,940 and 2,217 moves. All three have `integral_on_rail=0`. Return per-SM medians are 582,761, 583,006, and 580,986, chunks flat. Load per-SM medians are 485,478, 486,042, and 482,122. Count autocorrelation is 136 ms on both holds of each run. `cl-01` delay matches `i1-a`: 7.86 ms and 23.94 ms. Its warm-up median is 47,723,004 and its sustained median is 47,573,566.

`noise 57 15` was not run. The fullwidth holds already keep the count fixed, at 82, with the PI idle.

### Work counter and per-SM accumulator

The high group never appeared, so each switch ran twice on this plant.

`wc-01`, `rate_ref=47159552`. `wc-02`, `rate_ref=47028428`. Both print `[T9] work counter on` and `[PULSE]`. `sm_moves=0`, count 82. The printed ratio is `pulses × 16 / batches`, and 16 is `GOLD_BATCH`. A ratio of 1 means the pulse counter matches completed batches.

`wc-01.txt` prints 14 lines. The first three are `pulses=47177504 batches=755447360 ratio=0.99920`, then 1.00023, then 1.00044. Across the 14 lines the ratio runs from 0.99920 to 1.00044, median 1.00018. `wc-02.txt` runs from 0.99936 to 1.00065, median 0.99992. The low rate is the work rate. Pulses are not being dropped on the way to the counter.

`acc-01`, `rate_ref=46355140`. `acc-02`, `rate_ref=46201280`. About 1.5–2% under the 47.1 million cluster. The table still scales (61 / 41 / 20 / 2). `[T11]` was on. The raw accumulator files were 102 MB and are not committed. `smaccum-summary.csv` has every launched SM: 82 physical SMs, bins of 1 ms. Per-SM CUDA op rate, `acc-01`: min 5.906e11, median 5.923e11, max 5.953e11, max/min 1.008. `acc-02`: min 5.910e11, median 5.924e11, max 5.948e11, max/min 1.006. The slowest and the fastest SM are inside 0.8%. The low rate is not a subset of SMs. Both holds still autocorrelate at 136 ms (`z_raw` SD 0.0308 and 0.0277, narrow-band share 2.1% and 0.0%).

### Pulse path

`fw-localoff-a` and `fw-localoff-b` are fullwidth with `GPUTRONIC_PULSE_LOCAL` unset. There is no `[PULSE]` line. The cache stays at 1000 µs. One mapped atomic per pass.

| Launch | rate_ref | 61 | 41 | 20 | 2 | Per-SM |
| --- | --- | --- | --- | --- | --- | --- |
| fw-localoff-a | 1,409,173.5 | 1,405,335.7 | 1,394,363.0 | 1,390,239.6 | 821,298.0 | 15,918 |
| fw-localoff-b | 1,328,181.6 | 1,319,601.6 | 1,319,567.9 | 1,321,606.4 | 813,881.9 | 14,537 |

The per-SM gap does not stay. From 20 SMs to 82 SMs the rate is flat, near 1.3–1.4 million, and the first warm-up window is already there (`fw-localoff-a` warm-up median 1,418,229). Two SMs fall to about 0.81–0.82 million. `sm_moves=0`.

The 136 ms lag stays. `fw-localoff-a` autocorrelation is 0.397 at 128 ms, 0.480 at 136 ms, 0.391 at 144 ms, and 0.368 at 272 ms. Narrow-band share 2.0% and 0.1%. `z_raw` SD 0.0263 and 0.0355.

### Display and processes

`display_active` is `Enabled` on all 1,762 display samples. Changes: 0. `nvidia-smi-start.txt` and `nvidia-smi-end.txt` both show `Disp.A On`. The process lists are the desktop: Xorg and Cinnamon on both ends, Firefox on the start list. The compute-apps section is empty on every launch. No group change lines up with a display change, because the display did not change and the 49 million group did not appear. The desktop was left as it was.

### Verdict

The 47 million rate is in the plant before the PI runs. Fullwidth repeats, the first calibration window, a work-counter ratio of 1, and a flat per-SM accumulator all say the same thing. The 136 ms autocorrelation is in `z_raw` with the count fixed at 82, so the SM moves are following that wave rather than creating it. Turning the local pulse batch off removes the scaling with SM count and leaves the rate near 1.4 million pulses/s, about 1/33 of the 47 million plant. The 136 ms lag is still there.

The thirteenth high group, 49.1–49.6 million, is absent from every launch in this session. These logs do not show what produces it.

## Clocks

Flag files: 1,595 samples, die 44–66 °C, every reason word `0x0000000000000000`.

Display samples with GPU utilization at or above 90%: 1,658. Memory clock 14801 MHz on all of them. SM clock 2377–2895 MHz. 1,062 of them are 2835 MHz. Four are below 2700 MHz. Power on those busy samples is 56–221 W.

The loop did not fight the driver. The reason word stayed 0 while the count hunted, while the window was shortened, and while the cache was shortened. No thermal or memory-clock mechanism is claimed for the clusters. The memory clock did not move, and the die stayed at or below 66 °C.

## Checks

Switches unset. Commanded target 1.5 on the gate, after a `[CTRL]` line that still shows target 1.00.

| Gate | This pass z_mean | Thirteenth z_mean | Result |
| --- | --- | --- | --- |
| gold | 1.415 | 1.443 | PASS |
| transfer | 1.437 | 1.407 | PASS |
| episode | 1.499 | 1.457 | PASS |

Gold `[CHECK]`: `pulses=9278388 z=1.422 z_mean=1.415 rate=387370.0 n=50`. Transfer: `z=1.482 z_mean=1.437`. Episode: `z=1.506 z_mean=1.499`, JSON `z_mean=1.4995`. Each file also records `dyno=PASS closedloop=PASS`.

Transfer was the soft gate on the thirteenth pass. It is higher here. The gains were not changed.

`unset-home.txt` is a 3 s run with every switch unset, so its `rate_ref` is the stock plant (595,265.6), not the full-pipe plant. Its job in this pass is the `[CTRL]` line and the launch line.

## Success criteria

- Cycle by cycle. Yes. `i1-a` is a low-group hunting run, `rate_ref` 47,111,548. Its per-tick cut is committed. The median wait from an SM move to the first Z change is 7.86 ms (0.98 windows), and to half the change is 23.93 ms (2.98 windows). The load's count series autocorrelates at 136 ms. The return's count series autocorrelates at 48 ms. No launch landed in the 49 million group.
- Control rate. Yes, and the answer is no. On the same 47 million plant, a 4 ms window and a 2 ms window moved the count more and saw Z sooner. Neither stopped the hunting. The 250 µs cache and the 100 µs cache are a different plant and are not that comparison.
- Cause. Yes for the low rate: the fullwidth repeats put it in the plant, ahead of the PI. The logs do not separate the cause of the missing 49 million band, because that band did not appear.
- Guardrail. Yes. `integral_on_rail=0` on every `[STATS]` line in the logs (32 lines). The count reached 75 on `i2-w2` and 77 on `i2-c100`. Clamp 15 did not trip.
- Driver. Yes. Clocks, flags, and display state are logged on every run. Every reason word is 0. `display_active` stayed Enabled. The loop did not fight the driver.
- No regressions. Yes. With every switch unset, `[CTRL]` shows target 1.00, Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, and the launch is 82. `GPUTRONIC_RATE_WIN_US` is off by default: unset prints no `[RATE]` line. The three check gates pass at commanded 1.5.

## What was argued

The hunting holds are one pair, 110 s, rather than the 290 s three-pair schedule. The directive accepts the short pair unless a result needs the long one. The hunt is already in the first pair: thousands of moves, the same delay, the same 136 ms count lag. A second and third pair would not have produced the 49 million group. Eight later fullwidth launches and three later closed loops stayed in the 47 million band.

Item 1 stopped after two launches. The bound used was four, and stop once a low-group run and one later comparison exist. The first launch was 47,111,548. The second, identical settings, was 43,830,116. That is a real second cluster, and it is not the quiet group. Item 3 then spent the repeats on fullwidth, which is the split test.

`noise 57 15` was not run. Fullwidth already holds a fixed count with the integral frozen. The eight references sit in a 0.6% band. A ninth fixed count would not have changed the split result.

The cache and the window were also run together, once, as `i2-both`. That was extra. It landed at 44.4 million and hunted harder. It is not used as the same-group answer.

The rate window had no switch. The compare was a hardcoded `0.008`. The new switch stores that same literal when it is unset, and `unset-home.txt` has no `[RATE]` line. The frame and the README copy were updated together because the parameter table changed.

Faster PI ticks were left alone. Z changes when the window closes, and the measured loop was already 16,000–100,000 iterations per second with the CSV on. The window and the cache are the rates item 2 could actually raise.

The raw per-tick files and the raw accumulator files are not in the commit. The cut keeps calibration and the window closes, which is what the delay and the group questions use. `rate_ema` changes are kept along with `z_raw` and `active_sms` changes so a window that reprints the same `z_raw` is still in the file.

The desktop was not changed. Display stayed Enabled for the whole session, and the 47 million rate was there the whole time. That does not identify the thirteenth session's later `Disp.A Off`. It does say this session's low rate happened while the display was on.

Gains, the compiled clamp, the launch, and the floor were left where they are. Clamp 15 was not adopted as a default. `libsmctrl` was not used. The law was not changed to idle SMs when Z rises.
