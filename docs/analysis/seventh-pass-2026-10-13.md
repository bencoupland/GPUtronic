# The load no longer collapses, and power still does not follow the count

These holds ran before the reserved-SM note was read. That note is commits `6a7171f` and `778e4b4`. Every hold below launched up to 84 blocks. The holds taken after the note, on an 82-block launch, are in `docs/analysis/seventh-pass-part2-2026-10-13.md`.

Caching `work_scale_milli` with the other control words stopped the T6 collapse. On the cached plant, both switches on, Z was 1.487 on about 56 SMs before the load. After the scale went to 1.2 the count rose to about 63 and Z settled at 1.629, outside the 0.08 band, and it stayed there. The integral sat on the clamp. The pulse rate settled at 91% of the pre-step rate, against about 83% if the count had not moved. The same shape, at a few hundred thousand pulses per second, showed up with both switches off. Quiet idle did not make power follow the SM count. The wait spins on the GPU timer, the SM busy reading stayed at 100%, and board power stayed near 84 W from 4 SMs to 84. A command step at the 1 ms cache, quiet on, showed in the tally in 1.24 ms going down to 42 SMs and 0.61 ms coming back. The cached plant is a real rate loop, and a test load can now move it. It is not ready to become the default controller. A 20% load still parks Z about a tenth above the target, and turning SMs off does not hand the power back. Both switches stay off.

All of the numbers below were measured on this RTX 5080 on 12 October 2026, driver 595.99.02, CUDA 12.8. Nothing here is modelled. Kp stayed 0.55, Ki stayed 0.08, the deadband stayed 0.08, and the scaled integral clamp stayed 3.36 SMs. The tally stayed out of the PI. The version string stayed 1.0.0-gold. `GPUTRONIC_PULSE_LOCAL` and `GPUTRONIC_CTRL_CACHE` stay off unless a run sets them. The cache period on these runs was 1 ms. The pulse batch stays at 1 ms. No thermal violation and no power violation were recorded. Clocks stayed on the 2887–2910 MHz step. Temperature on the holds ran from 41 °C to 53 °C.

Logs: `logs/2026-10-13_seventh-pass/`. The three checks, with both switches unset, passed on this binary.

## Piece 1

`GPUTRONIC_CTRL_CACHE` is still off unless a run sets it. When it is on, the block leader reads `work_scale_milli` in the same refresh as `control_flags` and `active_sms`, once per `GPUTRONIC_CTRL_CACHE_US`, and the CUDA warps use that shared copy. When it is off, `gold_load_pulse` is unchanged: the mapped word is read only while the load flag is set, and only by the warp lane that decides the pulse.

The lines in `src/gputronic_gold.cu`:

- The cache comment at lines 110–115 now includes `work_scale_milli`.
- `gold_load_pulse` at lines 150–155 is the same read and the same credit math. The comment above it says this is the switch-off path.
- `gold_load_from_scale` at lines 158–166 is new. It takes the shared scale and does not read the mapped word.
- Shared `s_scale` is declared at line 217 and set to 1000 at line 225, before the first refresh.
- The refresh at lines 254–257 reads `control_flags`, then `active_sms`, then `work_scale_milli`.
- The CUDA pulse at lines 356–360 calls `gold_load_from_scale` when the cache is on and `gold_load_pulse` when it is off. Only lane 0 takes that call.
- The tensor warp at line 303 still calls `gold_load_pulse`. Tensor-on blocks still skip the cache, because a barrier there is not uniform. These runs left the tensor sensor off, so that path was not taken.
- The host line at 1298 prints that the cached read includes `work_scale_milli`. The checks do not print that line.

The README and the function frame say the same thing. The gains, the deadband, and the clamp were not edited.

## Piece 2

Gold closed loop, target Z 1.5, stock gains, tally off, 60 s. Load scale 1.2 from 20 s to the end. One run with both switches off, then one with both on. Z in the table is `z_hat`. The pulse rate is the mean of the 8 ms window count, divided by 0.008. The standard deviation is across the 1 ms samples in the window.

| Run | Window | Z mean | Z std | Z min | Z max | SMs mean | SMs range | Pulses/s | Versus before |
|---|---|---|---|---|---|---|---|---|---|
| Switches off | 10–19 s | 1.471 | 0.049 | 1.342 | 1.625 | 57.6 | 56–61 | 401 thousand | 1.000 |
| Switches off | 25–55 s | 1.724 | 0.037 | 1.593 | 1.925 | 66.4 | 64–70 | 342 thousand | 0.853 |
| Switches on | 10–19 s | 1.487 | 0.039 | 1.393 | 1.625 | 56.3 | 55–60 | 241.1 million | 1.000 |
| Switches on | 25–55 s | 1.629 | 0.035 | 1.548 | 1.765 | 63.0 | 61–65 | 219.7 million | 0.911 |

Startup on the cached run: `rate_ref` 356 million pulses/s, `n_ff` 57.4, `sms_per_u` 32.5, integral clamp 1.290, which is still 3.36 SMs. The switches-off run: `rate_ref` 587 thousand pulses/s, `n_ff` 59.4, `sms_per_u` 30.0, clamp 1.401, again 3.36 SMs.

The count did settle higher on both runs. Cached: 56.3 SMs to 63.0. Switches off: 57.6 to 66.4. On the cached hold, 89% of the samples had the integral within 0.02 of the rail. On the switches-off hold, 95% did.

Z left the 0.08 band about 10 ms after the step on the cached run, and about 8 ms after it with the switches off. It did not settle back. From 25 s to 55 s the cached run had 2.8% of samples inside the band, and the longest touch was about 80 ms. The switches-off hold had none. A scale of 1.2 would leave about 83% of the pulse rate if the SM count stayed put. The cached run settled at 91% because the count had risen. The first half-second was already 211 million pulses/s, 88% of the last pre-step second, with the count already near 61. The extra read is no longer the thing that drops the rate. The sixth-pass collapse, Z at the ceiling of 10 and about 1.3 million pulses/s, did not repeat.

The clamp is why Z sits near 1.63 rather than 1.50. The integral can move the count by 3.36 SMs. The proportional term moves it a little further, out to about 63 on the cached run and about 66 with the switches off. That is not enough to restore the pre-step rate. Gains were not changed.

## Piece 3

Open loop, 30 s, both switches on, tally off. Quiet idle on for 4, 12, 24, 36, 48, 56, 72, and 84 SMs, then quiet idle off for 4, 36, and 84. Z is `rate_ref` divided by the settled pulse rate, so the 4-SM rows are not clipped. `z_hat` on those two rows is the ceiling of 10. On the other rows `z_hat` agrees with this Z within 0.02. Power, temperature, and clock are the `nvidia-smi dmon` samples from 3 s to 28 s of the hold, after calibration. The SM busy reading on every one of those windows was 100%.

| n | Quiet | Pulses/s | Z | °C | W | MHz |
|---|---|---|---|---|---|---|
| 4 | on | 18.4 million | 19.47 | 51–53 | 89.5 | 2887–2895 |
| 12 | on | 55.3 million | 6.44 | 49–51 | 88 | 2895 |
| 24 | on | 110.4 million | 3.25 | 48–49 | 87 | 2895 |
| 36 | on | 164.3 million | 2.19 | 47 | 87 | 2895 |
| 48 | on | 215.6 million | 1.67 | 46–47 | 86 | 2895 |
| 56 | on | 248.4 million | 1.44 | 45–46 | 86 | 2895 |
| 72 | on | 312.8 million | 1.14 | 45 | 85 | 2895 |
| 84 | on | 360.6 million | 0.98 | 44–45 | 84 | 2895 |
| 4 | off | 17.1 million | 21.13 | 44 | 83.4 | 2895 |
| 36 | off | 155.6 million | 2.31 | 44 | 84 | 2895 |
| 84 | off | 360.8 million | 0.99 | 43–44 | 84 | 2895 |

Watts at the two ends: quiet on, 89.5 W at 4 SMs and 84 W at 84 SMs. Quiet off, 83.4 W at 4 SMs and 84 W at 84 SMs. The 4-SM quiet hold was the first of the sweep, and the die was still at 51–53 °C after the two T6 minutes. The three quiet-off holds, taken after the card had cooled to the mid-40s, sit on 83–84 W at 4, 36, and 84 SMs. Power does not follow the SM count. The quiet path waits on the GPU timer in a loop. That keeps the SM issued. It does not hand the SM back, and it does not hand the power back.

The command step was the sixth-pass experiment, one draw, at the 1 ms cache, with quiet on. Open loop at 84 SMs, tally bin 100 µs. The host set `active_sms` to 42 at 1.0 s and back to 84 at 2.0 s. The host samples landed 0.11 ms apart.

| | Down to 42 | Back to 84 |
|---|---|---|
| Host column saw the write | 0.12 ms | 0.11 ms |
| Tally reached the new level | 1.24 ms | 0.61 ms |

The sixth pass, quiet off, same cache period, was 1.21 ms down and 1.33 ms up. This draw did not get slower. Quiet idle did not add a visible wait on top of the cache period.

## Checks

Both switches unset. Tensor sensor off. Each gate printed `dyno=PASS closedloop=PASS`.

```
[CHECK] pulses=9242932 z=1.454 z_mean=1.449 rate=383189.9 n=52
[CHECK] free-run 554983.0 pulse/s | closed 386666.9 pulse/s
[CHECK] pulses=5418999 z=1.458 z_mean=1.460 target=1.50 ema=180473.4 sleep=5
[CHECK] free-run 257120.7 pulse/s (8.4253 GFLOP/s) | closed 179955.1 pulse/s (5.8968 GFLOP/s)
[CHECK] pulses=14594708 z=1.506 z_mean=1.499 target=1.50 ema=472131.2 sleep=5
[CHECK] free-run 712298.9 pulse/s (23.3406 GFLOP/s) | closed 474907.3 pulse/s (15.5618 GFLOP/s)
```

Gold, then transfer, then episode. None of the three logs printed the local-pulse line, the cached-control line, or the quiet-idle line.
