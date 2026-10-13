# Eighth pass, 12 October 2026

The scaled integral clamp stays. On the cached plant it prints 3.28 SMs of the 82-block launch, and widening the test clamp to 6, 9, or 12 SMs did not hold Z inside the 0.08 band under a 1.2 load, so nothing in the default was changed. Ending an idle block, and retiring the whole grid, both drop board power when only a few SMs are working: at 4 SMs the spinning plant drew 84 W, a relaunch drew 66 W, and an exit that actually returned drew 71 W. At 82 SMs all three sit near 85 W. Whether the real actuator should end the idle block or retire the grid is still open, and so is whether any wider clamp is worth a load longer than the 25 s step measured here.

All of the numbers below were measured on this RTX 5080 on 12 October 2026, driver 595.99.02, CUDA 12.8, `sm_120`, unless a line says modelled. Kp stayed 0.55, Ki stayed 0.08, and the deadband stayed 0.08. The launch is 82 persistent blocks. Physical SMs 82 and 83 got no test block on every full launch, including the relaunch back to 82. The command floor is 2. The version string stayed 1.0.0-gold. `GPUTRONIC_PULSE_LOCAL` and `GPUTRONIC_CTRL_CACHE` stay off unless a run sets them. The new switches stay off the same way.

Two binaries. The first suite is the kernel whose FMA nest the compiler deleted, and whose second idle-exit refill shared one stream and did not start. Those logs are in `logs/2026-10-13_eighth-pass/before-fix/`. Spin, relaunch, the clamp steps, and the first checks are from that binary and stayed in `logs/2026-10-13_eighth-pass/`. The follow-up binary gives each refill its own stream, and it compiles the peak nest as a second kernel so the unset loop stays the one the compiler already ships. Exit power, the exit step, the present window, the live peak search, and the second checks are that binary.

Plain `nvidia-smi` at the start and at the end of the first suite showed the display attached on DP-0 at 3840×2160 and two graphics processes, Xorg and Cinnamon. No compute process was listed. The card began at 44 °C and 38 W and ended the first suite at 42 °C and 36 W. The follow-up suite ended at 43 °C and about 32 W.

## Checks

Switches unset. Quote the CHECK line.

| Binary | Gate | Mean Z | Free-run | Closed |
|---|---|---|---|---|
| First | Gold | 1.415 | 543391 /s | 382707 /s |
| First | Transfer | 1.477 | 259374 /s (8.50 GFLOP/s) | 181418 /s (5.94 GFLOP/s) |
| First | Episode | 1.461 | 666590 /s (21.84 GFLOP/s) | 456440 /s (14.96 GFLOP/s) |
| Follow-up | Gold | 1.484 | 603316 /s | 409925 /s |
| Follow-up | Transfer | 1.448 | 261601 /s (8.57 GFLOP/s) | 181793 /s (5.96 GFLOP/s) |
| Follow-up | Episode | 1.484 | 700932 /s (22.97 GFLOP/s) | 473665 /s (15.52 GFLOP/s) |

All six gates passed.

## 1. Integral clamp

Cached plant, both switches on, tally off, 1 ms bins. Load scale 1.2 from 20 s to 45 s, 60 s total. Z in the table is `z_hat`. The pulse rate is the mean of the 8 ms window count, divided by 0.008. "Before" is 10–19 s, "held" is 25–44 s, "release" is 50–58 s. In band means `|z_hat - 1.5| < 0.08`. On the rail means the integral is within 0.1% of the clamp. The smallest of 6, 9, and 12 that holds the band would have been the candidate. The scaled law already failing means the candidate is none even if a wider clamp had held. None held.

| Run | Clamp print | Before Z / SMs / rate | Held Z / SMs / rate | In band | On rail | Held/before | Left the band | Release Z / SMs |
|---|---|---|---|---|---|---|---|---|
| Scaled | 1.384, 3.28 SMs of 82 | 1.479 / 55.7 / 232e6 | 1.629 / 62.1 (60–65) / 210e6 | 0.06 | 0.79 | 0.91 | 25 ms | 1.437 / 57.4 |
| 6 SMs | 2.534, 6.00 SMs | 1.481 / 55.5 / 231e6 | 1.612 / 62.5 (60–66) / 212e6 | 0.28 | 0.15 | 0.92 | 18 ms | 1.407 / 58.2 |
| 9 SMs | 3.458, 9.00 SMs | 1.487 / 54.7 / 230e6 | 1.601 / 62.4 (60–66) / 213e6 | 0.35 | 0.00 | 0.93 | 12 ms | 1.408 / 57.9 |
| 12 SMs | 4.938, 12.00 SMs | 1.480 / 55.7 / 232e6 | 1.608 / 62.9 (60–66) / 213e6 | 0.29 | 0.00 | 0.92 | 10 ms | 1.407 / 58.7 |

`logs/2026-10-13_eighth-pass/clamp-pick.txt` ends `PICK none`. The switches-off repeat was not run.

The integral grows at about the size of the error per second. At an error near 0.1, a 9 SM clamp (integral 3.46) takes longer than the 25 s the load is on, which is why 9 and 12 never reached the rail. Widening took the integral off the rail and moved held Z from 1.629 to about 1.60. The count rose by about 7 SMs and stopped near 63. A 1.2 thin was not absorbed. After the load came off, Z settled near 1.41 and the count came back to about 58, not a wind in the wrong direction. Kp and Ki were not changed. The default clamp stays the scaled law until a longer load says otherwise.

## 2. Idling an SM

Open loop, cached plant, 30 s holds. Power, temperature, SM busy, and clock are the `nvidia-smi dmon` mean over schedule 5–28 s. The pulse rate is the mean of the telemetry samples in the middle of the hold. Spin and relaunch are the first binary. Exit is the follow-up binary, after each refill got its own stream. On that binary the second wave's smid line says `ready` equal to the block count. On the first binary the second wave said `ready=0` and those high counts are not in this table.

| SMs | Spin W / million pulses/s | Exit W / million pulses/s | Relaunch W / million pulses/s |
|---|---|---|---|
| 4 | 84.0 / 16.7 | 71.0 / 20.4 | 66.0 / 18.0 |
| 12 | 84.0 / 50.1 | 72.6 / 60.0 | 68.7 / 53.5 |
| 24 | 84.0 / 96.8 | 74.9 / 118 | 71.0 / 104 |
| 36 | 84.6 / 151 | 78.1 / 175 | 73.2 / 156 |
| 48 | 84.3 / 201 | 79.1 / 235 | 75.7 / 206 |
| 56 | 84.3 / 233 | 79.8 / 276 | 78.0 / 240 |
| 72 | 84.0 / 299 | 83.5 / 347 | 82.4 / 303 |
| 82 | 84.3 / 340 | 85.1 / 389 | 84.5 / 340 |

SM busy stayed at 100% on every row. Spin clocks were 2895 MHz. Exit clocks were 2887–2895 MHz. Relaunch clocks were 2910 MHz. Temperatures on these holds stayed in the mid-40s to the mid-50s. The desktop idle before the first suite was about 38 W.

Ending the block, or launching a new grid, is what cuts the 84 W floor. At 4 SMs the cut is about 13 W for exit and about 18 W for relaunch. At 82 SMs the three plants meet, near 85 W, so the remaining power is the clocks, the working blocks, and the display. The follow-up exit rates sit above the first binary's spin rates. Both unset kernels had the FMA nest deleted by the compiler. The comparison that matters for the actuator is the power column.

A step 82 → 42 → 82, same cached plant, tally on:

| | Down to 42 | Back to 82 |
|---|---|---|
| Exit, tally | last ops on SM ≥ 42 at 1.14 ms | first ops on SM ≥ 42 at 0.71 ms |
| Exit, spawn | the blocks returned | `base=42 n=40`, ready=40, 309 µs |
| Relaunch, sync | 832 µs, launch 106 µs | 609 µs, launch 122 µs |
| Relaunch, tally | 0.62 ms | 0.70 ms |

The exit up-step is measured. On the first binary that spawn reported `ready=0` and the tally never saw SM ≥ 42 resume. The follow-up full 82-block exit map left out only physical SMs 82 and 83, `live_count=82`. A 4-SM exit ended at `live_count=4`, with 82 and 83 still in the left-out list. Relaunch of 82 after a smaller grid reported `same=82 moved=0` and left out 82 and 83. A 4-block relaunch landed on physical SMs 0–3.

The desktop probe is a 960×540 GTK window, title "GPUTronic present probe". Exit: 719 frames, median 16.65 ms, and the windows that cover the two steps had no interval over 20 ms. One later interval reached 33.0 ms. Relaunch: median 16.66 ms, p99 16.96 ms, max 18.0 ms, no interval over 20 ms.

## 3. Sensor range at high throughput

The first binary's peak search did not issue the extra FMAs. At 64 threads the pulse rate stayed near 330–360 million/s from 1 chain through 32 repeats. Raising threads to 1024, still with the nest deleted, raised the rate to 3.99 billion/s. A 60 s hold of that 1024-thread grid sat near 147 W and 53 °C at 2880 MHz. The hour of that same empty grid finished: last telemetry at 3599 s, 4.01 billion pulses/s, `n=82`, pulses 14244357339648, monotonic across the hour. `dmon` over that hour averaged 149 W and 58 °C, max 59 °C and 154 W. That hour is a counter and a thermal result of the empty kernel. It is not a FLOP result. The script's product `rate * 65536 * reps * chains` on that binary is not achieved work.

The follow-up binary keeps the unset kernel free of FFMA instructions (cuobjdump, 0 FFMA, 616 ops) and puts the nest in a second kernel (192 FFMA in the image). A short cached smoke with the switches unset was about 360 million pulses/s. Setting the peak switch, even at 1 repeat and 1 chain, dropped that to about 9.5 million pulses/s on the smoke and 10.2 million/s on the search. The result is stored, because an empty asm was not enough to keep the FMAs through ptxas.

Search holds are 8 s and are labelled as such. Cached plant, tally off. The rate is the last telemetry line.

| Repeats | Chains | Threads | Pulses/s |
|---|---|---|---|
| 1 | 1 | 64 | 10.23 million |
| 1 | 4 | 64 | 2.68 million |
| 1 | 8 | 64 | 1.40 million |
| 1 | 16 | 64 | 0.705 million |
| 2 | 16 | 64 | 0.352 million |
| 1 | 16 | 128 | 1.41 million |
| 1 | 16 | 256 | 2.26 million |
| 1 | 16 | 512 | 2.35 million |

A second repeat did not raise the script product, so the search stopped there. Threads stopped at 512 because 512 did not beat 256 by the script's 5% rule. The 60 s confirmation of 1 repeat, 16 chains, 512 threads: middle telemetry 2.43 million pulses/s. Over schedule 10–55 s, power averaged 238 W (232–241) and temperature averaged 66 °C (64–68). The clock held at 2835 MHz. The hardware thermal slowdown and the hardware power brake were not active. The pulse counter moved from 7765760 to 147904672 and stayed monotonic.

Modelled, not measured: 84 SMs × 128 CUDA cores × 2 FLOP/FMA × 2835 MHz is 6.10×10^13 FLOP/s, and 82 SMs is 5.95×10^13. The script product at the 60 s rate is 2.54×10^12, and that product assumes 1024 FMAs per chain per pulse. The kernel image has 192 FFMA instructions, so the product overstates the issued work. Either reading is a small fraction of the 82-SM peak. Board power was 238 W with the clock held and the die at 68 °C, so the limit this search hit is the nest, not the sensor and not a clock or power cap. The counters are `unsigned long long`. They wrap. They do not saturate. At 2.43 million pulses/s a 64-bit pulse counter wraps in about 2.4×10^5 years. At the brochure per-SM peak, 6.70×10^11 FLOP/s, a 64-bit ops counter wraps in about 0.9 years. Neither a 60 s hold nor an hour can wrap either counter at a rate this card can issue.

The live hour was not run. The 60 s hold was the thermal result asked for, and it stayed under 70 °C. A one-minute fragment of the same grid was started by the copy of the script that had already been read, and it was stopped. `peak-hour.txt` in the main log directory is that fragment. The finished hour remains the empty-kernel file under `before-fix/`.

## Left open

The default clamp, the gains, the deadband, and the idle behaviour are unchanged. The test switches stay off unless a run sets them. Two questions are still Ben's:

1. Should the integral clamp be widened for real, and to what? This 25 s load did not justify a wider default.
2. How should an SM actually be idled: by ending the block, by retiring the grid, or by something else? Both of the measured mechanisms cut power at low counts, and both put the command into effect in well under 2 ms. The desktop stayed on the 16.7 ms refresh through the steps.
