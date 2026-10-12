# GPUTronic Transfer Plant

Second plant on the Gold 1.0.0 interface.

```bash
make transfer
./build/gputronic_transfer check
# === TRANSFER GATE: dyno=PASS closedloop=PASS ===
```

Uses stock Gold (`launch_selftest_kernel = 0`). No separate gain. The kernel reads `active_sms`.

## Plant

- Persistent kernel, 84×64
- Real FP32 GEMM microtiles: 8×8×8 × 32 = 32768 FLOP per pulse
- `atomicAdd` on Gold `total_work_pulses`
- SMs at or above `active_sms` skip the tiles

## Current numbers (RTX 5080, 2026-09-23)

Closed loop: 7.25 → 5.39 GFLOP/s, Z mean ≈ 1.36. Open-loop rate tracks the SM count (R² ≈ 0.99).

## Older sleep-duration numbers (RTX 5080)

Open-loop: 9.03 → 5.15 GFLOP/s at 500 µs, in step with the tile rate. Free-run Z≈1.02.

Closed loop: 8.77 → 6.55 GFLOP/s (−25%), Z mean ≈ 1.34.

That is the transfer test: measure and control Z on something that is not the Gold synthetic kernel, and move a counted FLOP metric in the predicted direction.

## Per-SM sleep (2026-09-23)

`./build/gputronic_transfer mask`. Gold untouched. Stock 500 µs clamp. One block per SM, no grid barrier. The lowest k blocks take `__nanosleep` after their tiles. The others run the same tiles and do not wait.

200 µs does nothing at every k, including all 84. The dead zone stays a dead zone.

500 µs, wall Z against a 261 kpulse/s free-run:

| k asleep | Z | GFLOP/s |
|---------:|--:|--------:|
| 0–70 | 0.96–1.00 | ~8.6 |
| 82 | 1.04 | 8.08 |
| 83 | 1.18 | 7.12 |
| 84 | 1.64 | 5.11 |

One SM left issuing holds the chip near free-run. The stall arrives when the last SM joins. `Z = 84/(84−k)`, the rate if those SMs had produced nothing, is not this curve. The masked SMs kept pulsing. On this 5080 `__nanosleep` does not idle an SM while another SM is still in the tile.

The duration axis remains one chip-wide relay. A count of sleeping SMs is not a second gain.

## Working-SM count (2026-09-23)

`./build/gputronic_transfer work`. The lowest `n` SMs run the tiles and pulse. The rest skip the work, do not pulse, and backoff 5 µs. Floor is 4, so the kernel never goes empty. Workers are not duration-throttled in the sweep. Gold untouched.

The rate moves with `n`, and it is monotonic. The working SMs speed up as neighbours drop out, so Z sits below `84/n` through the middle of the range. On this card the point that lands on the setpoint is 48 SMs, not 56.

| n working | Z | Z if equal share | GFLOP/s |
|----------:|--:|-----------------:|--------:|
| 84 | 1.01 | 1.00 | 8.41 |
| 64 | 1.21 | 1.31 | 7.04 |
| 56 | 1.34 | 1.50 | 6.37 |
| 48 | 1.49 | 1.75 | 5.73 |
| 42 | 1.80 | 2.00 | 4.72 |
| 28 | 3.02 | 3.00 | 2.82 |
| 4 | 21.0 | 21.0 | 0.41 |

Around the setpoint one SM is about 0.03 of Z. `n=4` still runs. Ẑ clips at the Gold ceiling of 10 down there. Wall Z does not.

Stock PI, same clamp. The command now selects `n`: base keeps all 84, full clamp selects 42 (the measured point past 1.5). It held a steady Z of 1.35, command near 360 µs, not a bang between base and the clamp. Target back to 1.0 released to 8.26 GFLOP/s against a free-run of 8.51, sleep at 5 ns. The open-loop knob can sit on 1.49. The stock law, with its integral already at the clamp of this scale, sits one step short of that, on about 54 SMs.

## Workload change, same request (2026-10-04)

`./build/gputronic_transfer follow`. One control thread. Target Z = 1.5 the whole time. Kp 0.55, Ki 0.08. The run starts on this persistent kernel, switches to the episode shape (4 pulses, 32 of the same tiles, the kernel returns), then switches back. `gputronic_recalibrate` measures free-run again on whichever kernel is pulsing. The half-SM probe either runs again or the previous `sms_per_u` is kept. Nobody typed an SM count.

Score is the last 6 s of each 10 s phase. Kernel `7.0.0-38-generic`, driver 595.91.07.

| Phase | Z | SMs | Free-run | sms_per_u | In band |
|---|---:|---:|---:|---:|---|
| Transfer, probe at boot | 1.30 | 60 | 217k pulses/s | 139 | yes |
| Episode shape, that scale kept | 1.21 | 51 | 827k | 139 | no |
| Episode shape, probe again | 1.37 | 39 | 825k | 455 | yes |
| Transfer, episode scale kept | 1.41 | 53 | 211k | 455 | yes |
| Transfer, probe again | 1.33 | 60 | 213k | 139 | yes |

In band means |Z − 1.50| < 0.25, the wall rate under 90% of that phase's free-run, and the count below 84. With the probe left on, the count moved by about 21 SMs and both loads stayed in the band.

The transfer scale cannot reach the short-launch neighborhood. At the integrator clamp it turns off about 30 SMs, the count sticks near 51, and Z stays at 1.21. The probe on that shape saw Z 1.28 at half the SMs and asked for a drop of 75. After that probe the loop held Z 1.37 around 39 SMs.

The large scale, brought back onto this persistent kernel, stayed in the band. Z averaged 1.41 and the count breathed between 49 and 60. Reaching the target turns every SM back on, so the count did not stick at the floor.

The 827k free-run is this process's short-launch ceiling. The episode binary the same evening measured about 612k pulses/s on the same shape with its own host loop. Use 827k only inside this run. Report: `results/gputronic_follow_report.txt`.

## Leaving room for a second kernel (2026-10-05)

`./build/gputronic_transfer share`. The metered kernel is the episode shape. The host launches `active_sms` blocks of it, not a full grid whose extra blocks return or sleep. A second kernel runs the same tiles on the blocks that were not launched, on another stream, and adds to its own counter. Kp 0.55, Ki 0.08 on the closed-loop phase. Kernel `7.0.0-38-generic`, driver 595.91.07. The device reports concurrent kernels are supported.

Alone, 84, 42, and 21 metered blocks all landed near 800k pulses/s. Fewer blocks fill the chip by themselves. The guest kernel alone was 1.54M pulses/s. Same tiles, less time on the control block. Compare each kernel with itself.

| Metered blocks | Metered | Guest | Guest / guest solo |
|---:|---:|---:|---:|
| 63 | 759k | 253k | 0.16 |
| 42 | 558k | 558k | 0.36 |
| 21 | 325k | 975k | 0.63 |
| 4 | 71k | 1.41M | 0.92 |

At 42, the metered kernel kept 0.70 of the rate it had at 42 blocks with the chip to itself. The other kernel is real work, and it costs the metered kernel rate once it is present.

The resident transfer grid, command 42, idle blocks sleeping 5 µs: metered 114k pulses/s, guest 156k (0.10 of guest solo). The same guest on a split of 42 did 558k. The pulses to compare are the guest's. The metered kernels are different shapes, so 114k and 558k are not a pure occupancy comparison.

Closed loop after a free-run of 842k pulses/s, guest off until the probe finished. Half the blocks alone were Z 1.05, so the probe used the fallback drop of 36. With the guest on, the score window was Z 1.35, 49 SMs (46–52), metered 621k pulses/s, guest 441k. Report: `results/gputronic_share_report.txt`.

## Files

- `src/gputronic_transfer.cu`
- `results/gputronic_transfer_*_report.txt`
