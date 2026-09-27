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

## Files

- `src/gputronic_transfer.cu`
- `results/gputronic_transfer_*_report.txt`
