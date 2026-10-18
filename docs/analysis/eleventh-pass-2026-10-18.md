# Eleventh pass, 2026-10-18

Branch `test-results/2026-10-18_eleventh-pass`, started from `740efea`. Logs in `logs/2026-10-18_eleventh-pass/`. The card is an RTX 5080, driver 595.99.02, CUDA 12.8, `sm_120`. No run was stopped for temperature. The hottest die sample is 66 °C. Every `clocks_event_reasons.active` sample in the five run flag files is 0.

The directive is on `origin/controller-fixes` at `24c55dd` (`docs/directives/eleventh-pass-2026-10-18.md`). That branch is not merged. Two of its instructions are refused below, with the reason. The measurement that replaces them is the idle-work run.

## What this pass refused

**A second kernel is not the scheme.** Section 4 of `docs/analysis/tenth-pass-continued-2026-10-17.md` says the loop reads one total, `total_work_pulses`, and a launch beside the grid is outside that account. Ben asked for that conclusion and it is on `740efea`. Item 4 of this directive, and the standing rule that a guest must be schedulable at any load, ask for another kernel that runs while Gold is resident and is counted as it runs. That reopens the split the last report closed. The continuation already logged the scheduling tries: a higher connection limit, the guest stream created first, a higher stream priority, and Gold on a non-blocking stream. In each case the guest's one-second rate was 0 until Gold was told to stop. Those logs stand. They were not repeated.

The angles, and why each stays unrun:

1. Two totals inside one grid, Gold governed and the guest only seen, is the split again. The run that was done puts the idle blocks' pulses into `total_work_pulses`, the same total the loop already reads. The result is in the idle-work section.
2. A per-kernel counter printed beside Z is the guest probe. The continuation showed Z in band while that probe's rate was 0.
3. Green contexts are a stock way to put a second grid on a hardware fence. A fence does not put that grid's pulses into the one total. Unrun. If a later pass wants a stock partition, it is a different question from this one.
4. `libsmctrl` is evaluation only. The case is below, and it argues against allowing it.
5. Launching the guest kernel so it occupies SMs before Gold starts was not run. Creating the guest stream first was already measured, and the guest still waited until Gold stopped. Those two are different, and the stream-first log does not answer a guest that is already resident. The idle-work run is the one-grid test, so a second grid was not added to get this answer.
6. A second grid did not run, so there is no second calibration to compare.

**The compiled clamp stayed on every closed-loop run except the 15 SM test.** The directive says that if the clamp stays a switch, use 9 SMs for every closed-loop run this pass. Ben has not picked 9, 12, or the compiled value. Putting 9 on the Z = 1 holds would mix a clamp change into a ceiling measurement. The Z = 1 holds, the Z = 1.5 hold, and the idle-work run use the scaled clamp, which still prints 3.28 SMs. `GPUTRONIC_ICLAMP_SMS=15` is only the load step that finishes the 6 / 9 / 12 series.

Kp 0.55, Ki 0.08, Kd 0, deadband 0.08, the 82-block launch, the floor of 2, and physical SMs 82 and 83 are unchanged. No thermal or power logic was added.

## 1. Where the target is set

Unset, the target is 1.50. `GPUTRONIC_GOLD_TARGET_Z` is `1.5f` in `include/gputronic.h`. `gputronic_config_gold` copies it into `cfg->target_z`. The three check modes use that config and do not pass a Z argument. `run [sec] [Z] [Kp] [Ki]` sets `cfg.target_z` from the argument after the config is filled, so `run 25 1` commands 1 without an environment variable.

`loadfile` and `loadstep` have no Z argument. They read `GPUTRONIC_TARGET_Z` when it is set. Unset, they keep 1.5. `run` does not read that variable. The Z = 1 load step is the only run that set it. The log line is `[TARGET] GPUTRONIC_TARGET_Z=1.000 (unset keeps 1.5)`.

The unset checks on this binary, switches off, launch 82, 64 threads:

| Check | This pass | Continuation (`9efadb8`) | Gate |
| --- | --- | --- | --- |
| Gold | `z_mean=1.430` | 1.468 | dyno=PASS closedloop=PASS, err=0.070 |
| Transfer | `z_mean=1.457` | 1.497 | dyno=PASS closedloop=PASS |
| Episode | `z_mean=1.500` | 1.443 | dyno=PASS closedloop=PASS |

Each `[CTRL]` line is `target=1.50 Kp=0.550 Ki=0.0800 Kd=0.0000 deadband=0.080`. Each launch line is 82 blocks. No `[IDLE] work` line and no `[TARGET]` line. Logs `check-gold.txt`, `check-transfer.txt`, `check-episode.txt`.

**Proposal, not applied.** The home state can be Z = 1. The smallest change is `GPUTRONIC_GOLD_TARGET_Z` itself. The checks follow that macro, so they would then measure free run. A commanded 1.5 would be `run` with an argument, or an explicit set in the check modes if the gates should stay where they are. That waits for Ben. The switch above is how this pass commanded 1 on a load file without moving the default.

## Z = 1 on the full-pipe plant

Both holds use the full-pipe switches (`GPUTRONIC_PEAK_ACC=32`, `GPUTRONIC_PEAK_ROUNDS=512`, `GPUTRONIC_THREADS=512`, `GPUTRONIC_PULSE_LOCAL=1`, `GPUTRONIC_CTRL_CACHE=1`), `GPUTRONIC_NO_CSV=1`, idle-exit unset, guest unset, compiled clamp. The hold figures are the one-second `[TEL]` lines from t = 4 s through the last second. The band is `|z_hat - target| <= 0.08`.

`run 25 1`, log `z1-hold.txt`. Calibration: `rate_ref=46333680`, `n_ff=82.0`, `dZ/dn=0.0183`, `sms_per_u=54.7`, integral clamp 0.750, printed as 3.28 SMs of 82. The target sits on the full-launch end of the table (Z = 1 at 82 SMs, Z = 1.384 at 61), so the feedforward count is 82.

From t = 4 to t = 25, 22 samples: `z_hat` 0.948 to 1.066, mean 0.997, 22/22 inside the band, every sample at 82 SMs. `[STATS]` for the whole process, calibration included: `integral_updates=0`, `integral_on_rail=0`, `sm_moves=0`, `active_sms=82..82`, `mean_Z=0.9920`. The integral does not move. Z = 1 on an empty load is the full launch, inside the deadband.

`dmon -s pucm -d 1` has 27 samples, power 38–220 W (mean 198.3 W, the 38 W sample included), die 48–65 °C, SM clock 1057–2887 MHz. 23 of the 27 clocks are 2835 MHz. The flag file has 25 samples, all with `clocks_event_reasons.active` of 0, power 38.26–219.95 W.

## Z = 1.5 on the same plant

`run 25 1.5`, log `z15-hold.txt`. Same switches, compiled clamp. Calibration: `rate_ref=47426280`, `n_ff=55.5`, `dZ/dn=0.0327`, `sms_per_u=30.6`, integral clamp 1.341, printed as 3.28 SMs. This is the same shape as the earlier full-pipe holds.

From t = 4 to t = 25, 22 samples: `z_hat` 1.406 to 1.588, mean 1.495. The count is 54, and 57 on three seconds (t = 19, 22, and 25). Two seconds print outside the band: t = 21, `z_hat` 1.406 at 54 SMs, and t = 22, `z_hat` 1.588 at 57 SMs. t = 19 prints 1.580, which is 0.080 from 1.50. `[STATS]`: `integral_on_rail=0` of `pi_ticks=373993`, `sm_moves=196`, `integral_updates=26815`, `mean_Z=1.4936`, `active_sms=47..62` (that range includes calibration; the t = 4..25 lines are only 54 and 57).

`dmon` has 27 samples, power 50–218 W (mean 174.8 W), die 50–64 °C, SM clock 2835–2895 MHz. 25 flag samples, every reason word 0, power 50.40–218.40 W.

## Z = 1 on the 1.2 load step

Same full-pipe switches, `GPUTRONIC_TRACE_BIN_US=1000`, `GPUTRONIC_TARGET_Z=1`, compiled clamp, idle-exit and guest unset. Schedule `steps.txt` (the continuation's six steps), `loadfile ... 420`. Log `z1-load.txt`. The 1 ms trace is `z1-load-loadfile-1ms.csv`, 51,591,803 bytes, 419,922 data rows, t = 0.00 to 419,999.25 ms. The 10 µs file was 1,170,136,528 bytes and was deleted. It is not committed.

Calibration: `rate_ref=44987008`, `n_ff=82.0`, `dZ/dn=0.0160`, `sms_per_u=62.4`, integral clamp 0.657, printed as 3.28 SMs. `[STATS]`: `integral_on_rail=0` of `pi_ticks=10698010`, `integral_updates=2234`, `sm_moves=41`, `active_sms=79..82`, `mean_Z=1.0652`. The whole-run mean mixes both scales. The table is the hold figure.

The band is `|z_hat - 1.0| <= 0.08`. "Enters and stays" is the suffix after the last out-of-band bin. "Ends outside" means the hold's last bin is outside. These are the continuation's definitions. A limit cycle means the last 30 s uses exactly two SM counts, each at least five times, with at least six changes. The last 30 s is anchored on the hold's last sample.

On a 1.2 step the count is already 82, so the loop cannot add SMs. The integral stays where it is for the whole of each loaded hold. It is 0.00376 through the first load, 0.00754 through the second, and 0.01072 through the third, against a clamp of 0.657. None of those ticks are on the rail. Z sits above 1.08 for the rest of the hold. The three loaded holds end at 1.31623, 1.13320, and 1.13835, all at 82 SMs.

On each return to scale 1.0, Z undershoots to about 0.915 and the count steps down to 79. The integral steps up by about 0.003 on each return and ends the run at 0.01319. That is the direction that removes SMs (Z below target). It is not a wind toward the ceiling. The last 30 s of each return is 99.7–99.9% inside the band. The 120 s return meets the limit-cycle definition: the last 30 s is 24,059 bins at 79 and 5,942 bins at 82, with 8 changes. The 240 s and 360 s returns change count 4 times, so they do not. The loaded holds do not change count in the last 30 s.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end | SM range | Last 30 s inside | Last 30 s on the clamp | SM changes, last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | 1.2 | 34.1 ms | ends outside | — | 1.36079 | 1.00130 | 82 | 82 | 82–82 | 0 / 29,996 (0.0%) | 0 / 787,576 (0%) | 0 |
| 120 s | 1.0 | 0.1 ms | 58.510 s | 1.489 s | 1.31623 | 0.91488 | 82 | 79 | 79–82 | 29,922 / 30,001 (99.7%) | 0 / 770,618 (0%) | 8 |
| 180 s | 1.2 | 27.2 ms | ends outside | — | 1.31259 | 0.96303 | 79 | 82 | 79–82 | 0 / 30,000 (0.0%) | 0 / 712,366 (0%) | 0 |
| 240 s | 1.0 | 0.2 ms | 35.578 s | 24.421 s | 1.13320 | 0.91740 | 82 | 79 | 79–82 | 29,960 / 29,989 (99.9%) | 0 / 783,605 (0%) | 4 |
| 300 s | 1.2 | 29.0 ms | ends outside | — | 1.41185 | 0.97200 | 79 | 82 | 79–82 | 0 / 29,988 (0.0%) | 0 / 760,014 (0%) | 0 |
| 360 s | 1.0 | 0.2 ms | 45.309 s | 14.690 s | 1.13798 | 0.91701 | 82 | 79 | 79–82 | 29,964 / 30,001 (99.9%) | 0 / 764,992 (0%) | 4 |

The 60 s hold's last bin is outside, so the high of 1.36079 at +54.6 s has no staying tail. The 180 s and 300 s holds are the same shape. Whole-hold in-band shares, in step order: 0.1%, 98.7%, 0.0%, 99.9%, 0.0%, 99.9%. The plot is `z1-load-zhat.svg`.

`dmon` has 421 samples, power 51–220 W (mean 215.0 W), die 49–66 °C, SM clock 2835–2895 MHz, 416 of 421 at 2835. 410 flag samples, every reason word 0.

**Answer to the ceiling question.** On the empty hold, Z = 1 sits at free run and the integral does not wind. On the 1.2 step, Z cannot be brought back to 1, because the command is already at 82, and anti-windup leaves the integral frozen. The clamp never trips. The thing that runs out is the SM count, which is what "commanded Z limits throughput, the clamp does not" looks like when the command is already full speed.

## 2. The clamp at 15 SMs

The trip counter is already there. `st_rail` increments when `|integral| >= 0.999 * clamp`. `[STATS]` prints `integral_on_rail` and `pi_ticks`. The 1 ms file has `rail_ticks` and `pi_ticks` per bin. The per-hold fraction below is that ratio. Nothing was added to the control path, and no second counter was added. The whole-run `[STATS]` line mixes calibration with both scales, so the table is the hold figure.

`GPUTRONIC_ICLAMP_SMS=15`, target left at 1.50, same full-pipe switches, same `steps.txt`, same 420 s, idle-exit and guest unset. Log `clamp15.txt`. Trace `clamp15-loadfile-1ms.csv`, 52,174,449 bytes, 419,967 data rows, t = 0.00 to 419,999.03 ms. The 10 µs file was 734,481,151 bytes and was deleted.

Calibration matches the 9 and 12 SM runs, not the 6 SM outlier. `rate_ref=47347064`, `n_ff=56.0`, `dZ/dn=0.0319`, `sms_per_u=31.3`, integral clamp 5.985, printed as 15.00 SMs of 82. `[STATS]`: `integral_on_rail=0` of `pi_ticks=6641678`, `sm_moves=7577`, `active_sms=47..70`, `mean_Z=1.4961`. The sum of `rail_ticks` in the 1 ms file is 0. The sum of `pi_ticks` in that file is 6,641,373, against 6,641,678 on the `[STATS]` line.

The band is `|z_hat - 1.5| <= 0.08`. No hold is a limit cycle. The integral is off the rail on every hold, including the last 30 s.

| Step | Scale | First bin outside | Enters and stays | Tail that stays | Highest z_hat | Lowest z_hat | SM before | SM at end | SM range | Last 30 s inside | Last 30 s on the clamp | SM changes, last 30 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 s | 1.2 | 11.2 ms | 59.659 s | 340 ms | 1.84494 | 1.41546 | 58 | 67 | 58–70 | 27,430 / 30,000 (91.4%) | 0 / 470,882 (0%) | 263 |
| 120 s | 1.0 | 5.2 ms | 59.917 s | 82 ms | 1.58776 | 1.28799 | 67 | 56 | 56–67 | 27,028 / 29,997 (90.1%) | 0 / 480,870 (0%) | 59 |
| 180 s | 1.2 | 22.1 ms | 59.964 s | 35 ms | 1.76357 | 1.42659 | 56 | 64 | 56–65 | 25,596 / 30,001 (85.3%) | 0 / 472,832 (0%) | 311 |
| 240 s | 1.0 | 13.1 ms | 59.956 s | 43 ms | 1.59133 | 1.31894 | 64 | 56 | 56–64 | 27,454 / 29,998 (91.5%) | 0 / 473,572 (0%) | 4 |
| 300 s | 1.2 | 28.2 ms | ends outside | — | 1.78465 | 1.46718 | 56 | 65 | 56–68 | 25,334 / 30,001 (84.4%) | 0 / 470,892 (0%) | 322 |
| 360 s | 1.0 | 0.2 ms | 59.705 s | 294 ms | 1.62745 | 1.31496 | 65 | 56 | 56–65 | 27,492 / 30,001 (91.6%) | 0 / 472,054 (0%) | 4 |

The 300 s hold ends outside on the last 2.9 ms (`z_hat` 1.62744 at 65 SMs). It had entered. The loaded holds end at 67, 64, and 65. The returns end at 56. Whole-hold in-band shares: 57.8%, 58.5%, 69.4%, 67.1%, 68.3%, 66.9%. The plot is `clamp15-zhat.svg`.

`dmon` has 421 samples, power 56–218 W (mean 190.2 W), die 51–66 °C, SM clock 2835–2887 MHz. 410 flag samples, every reason word 0, power 56.31–218.98 W.

### Trip rate across the widths

| Setting | Where | `integral_on_rail` / `pi_ticks` | Loaded last 30 s on the clamp | Return last 30 s inside the band |
| --- | --- | --- | --- | --- |
| 3.28 SMs, compiled | tenth-pass load, cited in the continuation | on the rail for the loaded windows | 100% of each loaded last 30 s | 96.1–97.5% |
| 6 SMs | continuation `clamp-6.txt` | 1,404,459 / 6,572,896 | 100%, 100%, 22% | 91.6%, 93.3%, 91.0% |
| 9 SMs | continuation `clamp-9.txt` | 0 / 6,822,467 | 0%, 0%, 0% | 87.4%, 86.2%, 85.0% |
| 12 SMs | continuation `clamp-12.txt` | 0 / 6,545,569 | 0%, 0%, 0% | 84.4%, 86.0%, 85.5% |
| 15 SMs | this pass `clamp15.txt` | 0 / 6,641,678 | 0%, 0%, 0% | 90.1%, 91.5%, 91.6% |

The 9, 12, and 15 lines are zero on every hold, not only the last 30 s. The smallest setting in this series with a zero trip rate on every hold is 9 SMs. The 6 SM run's calibration does not match the others (`rate_ref` 43.4 million against about 47 million), so its trip rate is a poor ranking point. That was already true in the continuation. 15 does not move the trip rate off zero. It is not a new operating point: the settled count is still the mid-60s on a load and the mid-50s on a return.

The return in-band share fell from the compiled clamp through 12 SMs. At 15 it is 90–92%, which is above the 12 SM share. The deepest undershoot of the four widths is on this run, 1.28799, on the first return. The later two returns change count 4 times and end at 56. This pass does not treat "wider is worse on the way down" as a rule past 12. The numbers are in the table. No width is picked.

### Why the integral wanted that room

On this run the feedforward count is 56.0 and `sms_per_u` is 31.3, in the same band as the 9 and 12 SM calibrations. The slope is not the odd part. A scale of 1.2 skips counted pulses, so the counted rate falls and Z rises while the FMAs keep running. Restoring about a fifth of the counted rate, starting from 56 SMs, is on the order of 11 SMs if the rate tracks the count.

At the end of the three loaded holds the integral is −3.63843, −2.72248, and −2.74571. The correction those values represent is `Ki * |integral| * sms_per_u` with Ki 0.08: 9.11, 6.82, and 6.88 SMs. The counts at those moments are 67, 64, and 65, which is 8 to 11 SMs above 56. The compiled rail is 3.28 SMs, so the integral sits on it and the count stops nearer 61, which is the tenth-pass load. The trip is the size of this step against that rail.

15 SMs of room is more than the settled integral uses. The extra shows up in the hunt: the first loaded hold reaches `z_hat` 1.84494 and a count of 70, then ends at 67 with 263 count changes in the last 30 s. The same pattern was already visible from 9 to 12. The cause of the rail trip at 3.28 SMs is this plant step. What is still open is a disturbance that actually removes work, rather than skipping the pulse and leaving the FMAs running. That was not run.

## 3. The clock beside Z

Read only, `nvidia-smi dmon -s pucm -d 1` and a one-second flag sample of temperature, power, SM clock, and `clocks_event_reasons.active`. No thermal or power term was added to the controller.

On the Z = 1 hold the count stays at 82 while the SM clock sits at 2835 MHz (23 of 27 dmon samples; the 1057 MHz sample is the start). On the Z = 1.5 hold the clock stays inside 2835–2895 MHz while the count moves between 54 and 57. The three seconds at 57 are t = 19 (`z_hat` 1.580), t = 22 (`z_hat` 1.588), and t = 25 (`z_hat` 1.467). Those are the high-Z seconds and one second back inside. The reason word is 0 on every flag sample of both holds, and on the Z = 1 load, the 15 SM load, and the idle-work run.

There is no thermal or power flag for the loop to fight. The clock moves inside the boost steps this card has used on earlier full-pipe holds. The count steps on the Z = 1.5 hold line up with Z, which is the error term. No decoupling is proposed. The 83 °C test stop was not reached.

## 4. Idle blocks in the one total

`GPUTRONIC_IDLE_WORK=1` is a test switch, off by default. A block with `sm >= active_sms` then runs the same body and the same `atomicAdd` into `total_work_pulses`. Unset, that block still takes the old path: return if idle-exit is on, wait if quiet-idle is on, otherwise `continue`. Setting idle-work together with idle-exit, relaunch, or quiet idle turns those three off for the run. The row is in the function frame and in the README copy of it. From the shared title onward the two copies match.

The run is `run 40` at target 1.50, full-pipe switches, `GPUTRONIC_NO_CSV=1`, compiled clamp. Log `idle-work.txt`. The launch line is still 82 blocks. SMs 82 and 83 stay unlaunched.

The calibration table goes flat. Every probe returns about the full rate:

| Commanded count | Rate, pulses/s | Z |
| --- | --- | --- |
| 82 (`rate_ref`) | 47,239,092 | 1 |
| 61 | 47,378,589 | 0.997 |
| 41 | 47,899,406 | 0.986 |
| 20 | 47,541,532 | 0.994 |
| 2 | 47,825,238 | 0.988 |

`n_ff=2.0`, `dZ/dn=0.0010`, `sms_per_u=82.0`, and the log says the target is outside the table. Z at every probed count is about 1, so a target of 1.50 has no segment to sit on and the fit pins the start at the floor. The scaled clamp still prints 3.28 SMs (the clamp value is 0.500 because `sms_per_u` is capped at the launch count).

From t = 4 to t = 40, 37 samples: `z_hat` 0.964 to 1.038, mean 0.991, 0/37 inside the band around 1.50, every sample at 2 SMs. `[STATS]`: `deadband_held=0`, `integral_updates=0`, `integral_on_rail=0`, `sm_moves=0`, `mean_Z=1.0008`, `active_sms=2..2`. The error is about +0.5, so the deadband does not hold the count. The count is already at the floor, anti-windup does not integrate further into that floor, and the rate does not fall. The same binary, on `z15-hold.txt` a few minutes earlier with the switch unset, held mean `z_hat` 1.495 on 54 and 57 SMs. The flat table is the switch.

`dmon` has 41 samples, power 59–219 W (mean 213.4 W), die 51–66 °C, SM clock 2835–2895 MHz, 40 of 41 at 2835. 39 flag samples, every reason word 0. Power at a commanded count of 2 matches the full-launch holds, which is the same fact as the flat rate.

**What this says.** A complete count and a working SM command are the same constraint. The blocks the command turns off have to be blocks that stop adding to `total_work_pulses`. Once they add to that total, the rate no longer moves with `active_sms`, the calibration has no slope, and the loop parks at the floor with Z at free run. The account is complete and the actuator is gone. That is the result of the experiment.

The other way, a second total the loop only watches, leaves Z describing the governed blocks while the rest of the card is invisible to the command. The continuation already measured that, with the guest rate at 0 and Z inside the band. Either the idle blocks are outside the total, and the command works on what it counts, or they are inside the total, and the command cannot cut the rate. One grid can do the first. It cannot do both.

## libsmctrl

UNC's RTAS 2023 and 2024 papers, as the directive summarises them, say a field in the kernel descriptor that they call the SM mask actually disables TPCs, and that `libsmctrl` sets it by hooking undocumented CUDA internals. Their stated limits are 64 TPCs and no coverage for kernels launched from the GPU. Those are their findings on their hardware. This pass did not run the library, and nothing here confirms the field, the TPC width, or the 64 limit on this card. This card has 84 SMs. How that maps onto their TPC cap was not measured.

The case against allowing it:

The project stays above the CUDA runtime and the driver API. `libsmctrl` does not. It depends on an undocumented descriptor field. That is the same class of dependency as a custom driver: it breaks when the field moves, and this tree has already kept that class out. A documented driver call would be a different conversation. This one is not documented.

A TPC fence does not answer the account. Work running on the other TPCs still has to land in `total_work_pulses` or Z describes only the fenced kernel. The idle-work run is what happens when that work does land in the total and the command cannot turn it off. A mask around a second kernel would be angle 3 with an undocumented hook: a second grid, a second actuator, and the loop still reading one total unless the counter is changed too.

It also does not give something the stock API's green contexts would not already be for, if a later pass decides a hardware fence is the question. Both are partitions. The library's extra reach is the undocumented field and, on their report, coverage limits this pass did not test. That is a poor trade for a tree whose actuator is a count of blocks inside one launch.

Ben decides. The recommendation is to leave it out.

## Switches

| Switch | Default | This pass |
| --- | --- | --- |
| `GPUTRONIC_IDLE_WORK` | off | on only for `idle-work.txt` |
| `GPUTRONIC_TARGET_Z` | unset, loadfile keeps 1.5 | set to 1 only for `z1-load.txt` |
| `GPUTRONIC_ICLAMP_SMS` | unset, scaled law, 3.28 SMs | 15 only for `clamp15.txt` |
| Idle-exit, guest, relaunch, quiet idle | off | unset on every run |

`make gold`, `make lib`, `make demo`, `make transfer`, and `make episode` built with `/usr/local/cuda-12.8/bin/nvcc -arch=sm_120`. The unused-function warnings under `-DGPUTRONIC_NO_MAIN` are the ones earlier passes already had.

## Item 5

- **Home state.** Yes. The unset target is 1.50, and the place it is set is named above. Z = 1 is a proposal, not the default. The empty Z = 1 hold runs at 82 SMs, 22/22 inside the band, with `integral_updates=0`.
- **Guardrail.** Yes. The 15 SM run is in, the trip rate is 0 there and the 6 / 9 / 12 rates are cited, and the rail trip at 3.28 SMs is the 1.2 pulse-skip against a feedforward count near 56. A disturbance that removes the FMAs is still open. No width is picked.
- **Driver orthogonality.** Yes. The clock sat in the boost range, every reason word was 0, and the loop was not answering a throttle flag. No decoupling is proposed.
- **The guest is counted or understood.** Yes, as a written ruling-out plus one measurement. No new guest log was taken. The idle-work run puts the idle blocks into the one total, and Z then stays near 1 while the command sits at 2. The second-kernel angles stay unrun for the reason in the first section. The `libsmctrl` case is written, and it argues against.
- **No regressions.** Yes. Unset `[CTRL]` is Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, launch 82. Both new switches are off by default. All three checks pass. The check means moved relative to the continuation (1.430, 1.457, 1.500 beside 1.468, 1.497, 1.443) and the gates still pass.
