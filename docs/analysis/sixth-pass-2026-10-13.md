# The cached plant tracks the SM count

With both test switches on, Z against the 84-SM free-run sits within about 4% of an equal share, `84/n`. The 30 s sweep puts a Z of 1.5 near 54 SMs: 1.70 at 48 SMs and 1.45 at 56. Per-SM rate stays near 210 million ops per host bin from 4 SMs through 72, and the full chip dips only a few percent. The stock loop on that plant held a mean Z of 1.476 over the settled 45 s, on about 57 SMs, with the count moving inside 55–62. The pulse stream is now a sensor. Batching it, and caching the command words, already took the mapped reads off the hot path, and this kernel's rate follows the SM count. What remains is a 1 ms comb on the pulse word: a 1 ms bin has a rate coefficient of variation of 36%, the summed tally at 1 ms is 30% and does not lead it, and by 20 ms the two agree. `rate_ema`, the 8 ms window the PI uses, moves about 3% while the SM count is held. A redesign of the pulse stream would be a cleaner meter. It would also be the place to remove the leftover bus traffic. It is not a way to make this kernel faster. Ben and the caller make that call.

The T6 load step is the exception. The load flag makes every warp read `work_scale_milli` on every pass. That word is not in the cache. The pulse rate fell from about 266 million/s to about 1.3 million/s in the first second, Z went to the ceiling of 10, and the controller opened all 84 SMs. Gains were not changed.

All of the numbers below were measured on this RTX 5080 on 12 October 2026, driver 595.99.02, CUDA 12.8. Nothing here is modelled. Kp stayed 0.55, Ki stayed 0.08, the deadband stayed 0.08, and the scaled integral clamp stayed 3.36 SMs. The tally stayed out of the PI. The version string stayed 1.0.0-gold. `GPUTRONIC_PULSE_LOCAL` and `GPUTRONIC_CTRL_CACHE` stay off unless a run sets them. The cache period is `GPUTRONIC_CTRL_CACHE_US`, used only while the cache switch is on, default 1000 µs, floor 50 µs. The pulse batch stays at 1 ms.

Logs: `logs/2026-10-13_sixth-pass/`. The three checks, with both switches unset, passed on this binary.

```
[CHECK] pulses=8660728 z=1.560 z_mean=1.523 rate=345474.8 n=53
[CHECK] free-run 538742.5 pulse/s | closed 360773.1 pulse/s
[CHECK] pulses=5395146 z=1.480 z_mean=1.467 target=1.50 ema=178886.4 sleep=5
[CHECK] free-run 253515.9 pulse/s (8.3072 GFLOP/s) | closed 179934.6 pulse/s (5.8961 GFLOP/s)
[CHECK] pulses=13955040 z=1.432 z_mean=1.458 target=1.50 ema=464862.0 sleep=5
[CHECK] free-run 664445.4 pulse/s (21.7725 GFLOP/s) | closed 458076.7 pulse/s (15.0103 GFLOP/s)
```

Gold, transfer, and episode each printed `GATE: dyno=PASS closedloop=PASS`. The check log has no local-pulse line and no cached-control line. Busy samples through the holds stayed at 2887–2910 MHz. No thermal violation and no power violation were recorded. Temperature of the busy samples ran from 56 °C at the start of the evening to 42 °C at the full-chip cached hold.

## Experiment 1

Open loop, 30 s, tally on, quiet off. Both switches on, except the last two rows. The gate is the block index, so 4, 12, 24, 36, 42, 48, 56, 64, 72, and 84 cut through GPC boundaries. A count that is not a multiple of 12 leaves a partial last GPC, and that GPC's mean is only those SMs. Ops per host bin are the per-GPC means. Z is this row's pulse rate divided into the 84-SM rate of the same plant. `84/n` is the equal share. The 100 ms tally-to-pulse ratio was 1.0000 on every row except the cached 84-SM hold, which was 1.0001.

The sweep rows carry the per-SM tally, which posts one mapped slot per block per millisecond. That costs rate. The cached 84-SM hold below is 264 million pulses/s. The closed loop in Experiment 2, tally off, calibrated at 394 million pulses/s. Compare Z inside one plant.

| n | Switches | GPC 0 | GPC 1 | GPC 2 | GPC 3 | GPC 4 | GPC 5 | GPC 6 | CV | Pulses/s | GFLOP/s | Z | 84/n | °C | W | MHz |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | on | 205.2M | | | | | | | | 12.52M | 820 | 21.05 | 21.00 | 55 | 86 | 2887 |
| 12 | on | 208.3M | | | | | | | | 38.12M | 2498 | 6.91 | 7.00 | 56 | 86 | 2887 |
| 24 | on | 210.1M | 210.6M | | | | | | 0.1% | 76.97M | 5044 | 3.42 | 3.50 | 53 | 85 | 2887 |
| 36 | on | 211.8M | 212.4M | 212.5M | | | | | 0.1% | 116.5M | 7634 | 2.26 | 2.33 | 49 | 85 | 2895 |
| 42 | on | 212.5M | 213.3M | 213.1M | 212.6M | | | | 0.2% | 136.3M | 8934 | 1.93 | 2.00 | 46 | 84 | 2895 |
| 48 | on | 211.0M | 211.9M | 211.8M | 211.6M | | | | 0.2% | 154.8M | 10144 | 1.70 | 1.75 | 45 | 84 | 2895 |
| 56 | on | 211.7M | 212.8M | 213.6M | 213.0M | 212.9M | | | 0.3% | 181.6M | 11904 | 1.45 | 1.50 | 44 | 84 | 2895 |
| 64 | on | 212.0M | 213.2M | 214.4M | 214.6M | 214.0M | 213.1M | | 0.4% | 208.4M | 13658 | 1.26 | 1.31 | 43 | 84 | 2910 |
| 72 | on | 211.8M | 212.1M | 213.9M | 214.7M | 213.4M | 211.8M | | 0.5% | 233.7M | 15318 | 1.13 | 1.17 | 42 | 83 | 2910 |
| 84 | on | 204.4M | 204.9M | 207.2M | 208.5M | 207.4M | 204.9M | 203.5M | 0.8% | 263.5M | 17270 | 1.00 | 1.00 | 42 | 84 | 2910 |
| 84 | off | 398k | 401k | 402k | 761k | 401k | 399k | 399k | 28% | 578k | 37.9 | 1.00 | 1.00 | 39 | 65 | 2910 |
| 48 | off | 398k | 399k | 401k | 759k | | | | 32% | 358k | 23.5 | 1.62 | 1.75 | 40 | 64 | 2910 |

CV is the coefficient of variation of the GPC means that have work. One GPC has no CV. On the cached 84-SM hold the largest and smallest SMs were 1.044× apart, and no neighbouring blocks differed by 1.2×. With both switches off, GPC 3 is 1.91× GPC 0 (761k against 398k), and the step in block order is 1.90× between blocks 47 and 48. That is the same shape as the fifth-pass 84-SM hold.

Power on the cached rows is 83–86 W at every count, including 4 SMs. Idle blocks still spin on the cached words. The quiet switch was off, so turning SMs off did not hand the power budget back. Clocks stayed on the same step the whole sweep. The 17.3 TFLOP/s point was 42 °C and 84 W.

Interpolating the pulse rate to the 84-SM rate divided by 1.5 puts Z = 1.5 at about 54 SMs on this tally-on plant.

## Experiment 2

Gold closed loop, both switches on, tally off, target Z 1.5, stock gains, 60 s. Then one T6 load step, scale 1.2 from 20 s to the end. This tree has no `gputronic_recalibrate`. The half-SM probe is the four-point table printed at startup.

| Run | Tally | rate_ref | n_ff | sms_per_u | Clamp, SMs |
|---|---|---|---|---|---|
| Closed loop, switches on | off | 394M | 58.2 | 33.8 | 3.36 |
| T6, switches on | off | 391M | 57.9 | 29.7 | 3.36 |
| Open loop 84, switches on | on | 263M | 57.4 | 26.2 | 3.36 |
| Open loop 84, switches off | on | 583k | 56.1 | 29.7 | 3.36 |

`sms_per_u` stays in the high 20s to the low 30s on both plants. The tally-on 84-SM rate is 456× the switches-off rate. The closed-loop calibration, tally off, is 675× that switches-off `rate_ref`. The shape of Z against SM count did not follow those factors. The switches-on probes already differ from each other by more than either differs from the switches-off probe. That is the table moving between runs.

Settled window of the closed loop, 10 s to 55 s, 45,000 samples at 1 ms:

| | Value |
|---|---|
| Z mean | 1.476 |
| Z std | 0.055 |
| Z min, max | 1.367, 1.694 |
| Samples inside the 0.08 deadband | 37,760 / 45,000 (84%) |
| SM count mean | 57.2 |
| SM count std | 1.6 |
| SM count min, max | 55, 62 |
| Times the count changed | 869 |

The count keeps moving, inside a seven-SM band, and Z stays on the target. It does not sit on one count, and it does not run away. `rate_ema` over that window has a coefficient of variation of 3.6%. Inside the 493 stretches where the count held for at least 40 ms, the median coefficient of variation of `rate_ema` is 3.1%. The 1 ms comb on the pulse word, measured in Experiment 4, is a 36% coefficient of variation. That comb is not what `rate_ema` shows. The PI's window is 8 ms.

The load step, same plant, same gains:

| Window | Z | SMs | Pulses in the 8 ms window | Scale |
|---|---|---|---|---|
| 10–19 s, before the load | 1.473 | 56.7 | 2.13 million | 1.00 |
| 25–55 s, load held | 10.000 | 84 | 10.3 thousand | 1.20 |

At 20 s, the second the scale becomes 1.2, Z is already 9.7 and the count is 84. From 21 s on, Z is the ceiling of 10 and the window holds about 10 thousand pulses, which is 1.3 million pulses/s. A scale of 1.2 would have left about 83% of the pulse rate if the extra read were free. The read is on every pass. The controller did what the law says when Z is far above the target: it opened every SM. The rate stayed collapsed for the rest of the minute. No gain was changed.

## Experiment 3

One open-loop run at 84 SMs, both switches on, tally bin 100 µs. The host set `active_sms` to 42 at 1.0 s and back to 84 at 2.0 s. Latency is the host-sample time from that write to the new tally level on blocks 42 and above. The host samples landed 0.11 ms apart, so that is the resolution. Each figure is one step. The delay can be as long as the cache period. These three draws landed inside it.

| Cache period | Down to 42 | Back to 84 | Host column saw the write |
|---|---|---|---|
| 0.25 ms | 0.37 ms | 0.31 ms | 0.14 ms, then 0.10 ms |
| 1 ms | 1.21 ms | 1.33 ms | 0.12 ms, then 0.12 ms |
| 4 ms | 2.37 ms | 3.46 ms | 0.12 ms, then 0.11 ms |

The host column, which is the control thread's copy of `active_sms`, updates inside one sample. The tally waits out the cache period. The pulse batch stayed at 1 ms on all three runs.

## Experiment 4

Noise is the coefficient of variation of the rate at a fixed SM count. The tally rate is the sum of `d_ops_cuda` across SMs, divided by 65,536. The pulse rate is `bin_pulses` in the same host bin. Lag is the shift, in host bins, of the best correlation. It was 0 on every row. PCIe receive and transmit are the busy samples of `nvidia-smi dmon -s pucvt` during that hold.

| Hold | 1 ms pulse | 1 ms tally | 1 ms corr | 20 ms pulse | 20 ms tally | 20 ms corr | 20 ms ratio | Rx | Tx |
|---|---|---|---|---|---|---|---|---|---|
| 84, switches on | 36.2% | 29.8% | 0.57 | 8.3% | 8.5% | 0.95 | 1.0006 | 41 MB/s | 23 MB/s |
| 48, switches on | 30.7% | 25.0% | 0.60 | 5.3% | 5.5% | 0.93 | 1.0003 | 23 MB/s | 15 MB/s |
| 84, switches off | 20.4% | 26.6% | 0.48 | 4.6% | 5.0% | 0.91 | 1.0001 | 104 MB/s | 69 MB/s |
| 48, switches off | 21.1% | 26.4% | 0.54 | 5.1% | 5.5% | 0.93 | 1.0001 | 106 MB/s | 62 MB/s |

On the cached 84-SM hold the analyzer's 1 ms ratio mean is 3.14. A 1 ms bin sometimes catches two batches and sometimes catches none. At 20 ms the ratio is 1.0006 and the correlation is 0.95. The summed tally is the same sensor, read on the same clock. It does not lead the pulse word, and at 1 ms it is not quieter. The switches-off 1 ms ratio is 1.06, because that plant pulses every pass and a 1 ms bin holds hundreds of pulses. Its coefficient of variation is ordinary counting noise.

PCIe traffic at 84 SMs dropped from 104 MB/s receive and 69 MB/s transmit with the switches off to 41 and 23 with them on, while the pulse rate rose from 578 thousand/s to 264 million/s. The bus is the leftover of the 1 ms batches and the tally slots. Another GPU-side change to how often this kernel touches those words will not produce another large rate gain. The rate is the SM count.
