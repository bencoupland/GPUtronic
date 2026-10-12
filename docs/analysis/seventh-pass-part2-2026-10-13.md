# Eighty-two blocks, and the same two answers

The reserved-SM rules apply to every hold in this note. The launch is 82 persistent blocks on this 84-SM card. Physical SMs 82 and 83 got no test block, on every run, including both passes of each check. The command floor is 2. The quiet-idle sweep still started at 4. Plain `nvidia-smi` at the start and at the end showed the display attached (`Disp.A` On) and three graphics processes: Xorg, Cinnamon, and Firefox. No compute process was listed at either snapshot.

On the cached plant, both switches on, Z was 1.487 on about 55 SMs before the load. After the scale went to 1.2 the count rose to about 61 and Z settled at 1.639, outside the 0.08 band, and it stayed there. The integral sat on the clamp. The pulse rate settled at 91% of the pre-step rate, against about 83% if the count had not moved. The same shape showed up with both switches off, at a few hundred thousand pulses per second. Quiet idle did not make power follow the SM count. The wait spins on the GPU timer, the SM busy reading stayed at 100%, and board power stayed near 84 W from 4 SMs to 82. A command step at the 1 ms cache, quiet on, showed in the tally in 0.55 ms going down to 42 SMs and 0.42 ms coming back. The cached plant is still a real rate loop, and a test load can move it. It is still not ready to become the default controller. Both switches stay off.

The holds in `docs/analysis/seventh-pass-2026-10-13.md`, commit `53e3e67`, all ran before this note was read. They launched up to 84 blocks.

All of the numbers below were measured on this RTX 5080 on 12 October 2026, driver 595.99.02, CUDA 12.8. Nothing here is modelled. Kp stayed 0.55, Ki stayed 0.08, and the deadband stayed 0.08. The clamp formula is the same scaled law. With `num_sm` equal to the 82-block launch it prints as 3.28 SMs. The tally stayed out of the PI. The version string stayed 1.0.0-gold. `GPUTRONIC_PULSE_LOCAL` and `GPUTRONIC_CTRL_CACHE` stay off unless a run sets them. The cache period on these runs was 1 ms. The pulse batch stays at 1 ms. No thermal violation and no power violation were recorded. Clocks on the settled windows stayed on the 2887–2895 MHz step, with a 2910 MHz sample only in the wider dmon files around startup. Temperature on the holds ran from 41 °C to 52 °C.

Logs: `logs/2026-10-13_seventh-pass-part2/`. The three checks, with both switches unset, passed on this binary.

## What changed in the source

`include/gputronic.h` defines `GPUTRONIC_OS_RESERVE_SMS` as 2 and `GPUTRONIC_SM_FLOOR` as 2. `gputronic_create` launches `device_sms - 2` blocks when the caller asked for the whole device. The control block layout is unchanged. The smid page is a separate mapped allocation, one `unsigned int` per launched block. Thread 0 writes `%smid` once, before the work loop, and `gputronic_log_smids` prints the map. The two device SMs with no block never write, so the host names them by absence.

The lines in `src/gputronic_gold.cu`:

- The smid store is at lines 233–236, inside the thread-0 setup at lines 221–237. It runs before the loop, including on a block that will sit idle.
- The device floor is line 287. A command below 2 is raised to 2.
- The host floor is lines 970, 981, 991, and 996. The calibration probe's last point is the floor, line 741.
- `gputronic_log_smids` is at line 1083. The reserve is applied at lines 1151–1153. The launch print, including the floor, is at lines 1226–1236. The self-test launch calls the logger at line 1397.

Transfer and episode use the same launch count, the same floor, and the same logger. Episode writes the smid page on the first launch of a process and passes a null pointer after that, so the later launches do not add a mapped write. The function frame, the README control-block row, and `STATUS.md` say floor 2 and a launch that leaves 2 SMs empty.

Gains, the deadband, and the clamp formula were not edited.

## The block map

Every process printed the same pair left out: physical SMs 82 and 83, with 82 of 82 blocks recorded. CUDA placed the blocks. The full `block:smid` line is in each log. One Gold launch began `0:0 1:1 2:14 3:15` and ended `80:68 81:69`. The walk is pairs 14 apart, and it stops before SMs 82 and 83.

## Piece 2

Gold closed loop, target Z 1.5, stock gains, tally off, 60 s. Load scale 1.2 from 20 s to the end. One run with both switches off, then one with both on. The ceiling on both runs is the 82-block launch, and `rate_ref` is that launch's free-run. Z in the table is `z_hat`. The pulse rate is the mean of the 8 ms window count, divided by 0.008. The standard deviation is across the 1 ms samples in the window.

| Run | Window | Z mean | Z std | Z min | Z max | SMs mean | SMs range | Pulses/s | Versus before |
|---|---|---|---|---|---|---|---|---|---|
| Switches off | 10–19 s | 1.446 | 0.031 | 1.370 | 1.550 | 52.0 | 51–52 | 399 thousand | 1.000 |
| Switches off | 25–55 s | 1.658 | 0.034 | 1.549 | 1.848 | 60.2 | 58–63 | 348 thousand | 0.872 |
| Switches on | 10–19 s | 1.487 | 0.042 | 1.399 | 1.618 | 55.4 | 54–58 | 238.5 million | 1.000 |
| Switches on | 25–55 s | 1.639 | 0.038 | 1.527 | 1.934 | 61.3 | 60–66 | 216.3 million | 0.907 |

Startup on the cached run: `rate_ref` 353 million pulses/s, `n_ff` 55.9, `sms_per_u` 29.6, integral clamp 1.387, which is 3.28 SMs of 82. The switches-off run: `rate_ref` 574 thousand pulses/s, `n_ff` 54.5, `sms_per_u` 29.5, clamp 1.390, again 3.28 SMs.

The count did settle higher on both runs. Cached: 55.4 SMs to 61.3. Switches off: 52.0 to 60.2. On the cached hold, 89% of the samples had the integral within 0.02 of the rail. On the switches-off hold, 84% did. The held count on the cached run was mostly 61, then 62.

Z left the 0.08 band about 8 ms after the step on the cached run, and about 16 ms after it with the switches off. It did not settle back. From 25 s to 55 s the cached run had 2.6% of samples inside the band, and the longest touch was 96 samples, about 100 ms. The switches-off hold had 0.5% inside, and the longest touch was 64 samples, about 67 ms. A scale of 1.2 would leave about 83% of the pulse rate if the SM count stayed put. The cached run settled at 91% because the count had risen. The first second of the load was already 210 million pulses/s, 88% of the pre-step rate, with the count already near 59. The sixth-pass collapse did not repeat.

The clamp is why Z sits near 1.64 rather than 1.50. The integral can move the count by 3.28 SMs. The proportional term moves it a little further, out to about 61 on the cached run and about 60 with the switches off. That is not enough to restore the pre-step rate. Gains were not changed.

## Piece 3

Open loop, 30 s, both switches on, tally off. Quiet idle on for 4, 12, 24, 36, 48, 56, 72, and 82 SMs, then quiet idle off for 4, 36, and 82. Z is `rate_ref` divided by the settled pulse rate, so the 4-SM rows are not clipped. `z_hat` on those two rows is the ceiling of 10. On the other rows `z_hat` agrees with this Z within about 0.02. Power, temperature, and clock are the `nvidia-smi dmon` samples from 3 s to 28 s of the hold, after calibration. The SM busy reading on every one of those windows was 100%.

| n | Quiet | Pulses/s | Z | °C | W | MHz |
|---|---|---|---|---|---|---|
| 4 | on | 18.6 million | 19.07 | 50–52 | 89.4 | 2887–2895 |
| 12 | on | 55.7 million | 6.35 | 49–50 | 88 | 2895 |
| 24 | on | 111.3 million | 3.17 | 48 | 87 | 2895 |
| 36 | on | 165.6 million | 2.13 | 47 | 86.8 | 2895 |
| 48 | on | 217.3 million | 1.63 | 46 | 86 | 2895 |
| 56 | on | 248.5 million | 1.42 | 45–46 | 86 | 2895 |
| 72 | on | 315.0 million | 1.12 | 45 | 85 | 2895 |
| 82 | on | 354.5 million | 0.99 | 44–45 | 84 | 2895 |
| 4 | off | 17.4 million | 20.23 | 44 | 84.2 | 2895 |
| 36 | off | 156.2 million | 2.23 | 44 | 84 | 2895 |
| 82 | off | 353.5 million | 1.00 | 44 | 84 | 2895 |

Watts at the two ends: quiet on, 89.4 W at 4 SMs and 84 W at 82 SMs. Quiet off, 84.2 W at 4 SMs and 84 W at 82 SMs. The 4-SM quiet hold was the first of the sweep, and the die was still at 50–52 °C after the two T6 minutes. The three quiet-off holds, taken after the card had cooled to 44 °C, sit on 84 W at 4, 36, and 82 SMs. Power does not follow the SM count. The quiet path waits on the GPU timer in a loop. That keeps the SM issued. It does not hand the SM back, and it does not hand the power back.

The command step was one draw, at the 1 ms cache, with quiet on. Open loop at 82 SMs, tally bin 100 µs. The host set `active_sms` to 42 at 1.0 s and back to 82 at 2.0 s. The host samples landed 0.11 ms apart.

| | Down to 42 | Back to 82 |
|---|---|---|
| Host column saw the write | 0.13 ms | 0.11 ms |
| Tally reached the new level | 0.55 ms | 0.42 ms |

## Checks

Both switches unset. Tensor sensor off. Each gate printed `dyno=PASS closedloop=PASS`. Each of the six launches (dyno and closed loop, three plants) recorded SMs 82 and 83 left out.

```
[CHECK] pulses=9678447 z=1.411 z_mean=1.440 rate=415232.2 n=54
[CHECK] free-run 589836.3 pulse/s | closed 406750.9 pulse/s
[CHECK] pulses=5041642 z=1.449 z_mean=1.427 target=1.50 ema=164872.2 sleep=5
[CHECK] free-run 242801.5 pulse/s (7.9561 GFLOP/s) | closed 167983.2 pulse/s (5.5045 GFLOP/s)
[CHECK] pulses=14119484 z=1.510 z_mean=1.469 target=1.50 ema=448539.2 sleep=5
[CHECK] free-run 675400.5 pulse/s (22.1315 GFLOP/s) | closed 462900.3 pulse/s (15.1683 GFLOP/s)
```

Gold, then transfer, then episode. None of the three logs printed the local-pulse line, the cached-control line, or the quiet-idle line. Gold's closed-loop mean Z was 1.440. The gate passed. Gains were not changed.
