# GPUTronic Status – Gold 1.0.0

## Current Working State

**GPUTronic Gold 1.0.0**, changed 2026-09-23. Full speed is every SM running. The aim is to get close to the rate you asked for when the workload is not ideal. The sleep-duration version is in `archive/gold-1.0.0-nanosleep/`.

| Item | Value |
|------|--------|
| Binary | `build/gputronic_gold` |
| Header | `include/gputronic.h` |
| Source | `src/gputronic_gold.cu` |
| Transfer | `src/gputronic_transfer.cu` |
| Episode | `src/gputronic_episode.cu` |
| Show | `make show` → `results/gputronic_show.html` |
| Sweep | `./build/gputronic_show sweep` → `results/gputronic_sweep.svg` |
| Library | `make lib` → `build/libgputronic_gold.a` |
| Nanosleep archive | `archive/gold-1.0.0-nanosleep/` |
| Control period | 10 µs path; rate formed over about 1–8 ms (8 ms is the steady end) |
| Z model | Sustained-window `Z = rate_ref / rate` (free-run ≈ 1.0) |
| Target Z | 1.5 |
| Gains | Kp=0.55, Ki=0.08, Kd=0. Do not retune them to chase the tenth. The four-point table sets `n_ff` and `sms_per_u`. |
| Policy | Two-sided PI around `n_ff` + 2-state Kalman. Deadband 0.08 holds the count and freezes the integral. Clamp is about 3.4 SMs on this 84-SM card. |
| Actuator | `active_sms`. Idle SMs skip the work and do not pulse. Floor 2. Launch leaves 2 SMs with no block. |
| Tachometer | `atomicAdd` on aggregate `total_work_pulses`. Optional per-SM tally, off unless `GPUTRONIC_SM_ACCUM` is set. |
| Tensor sensor | Off unless `GPUTRONIC_TENSOR_SENSOR=1`. On this 5080 it cut the CUDA rate about 39% and its Z matched the CUDA Z. |

## Verified on RTX 5080 (2026-09-23, after the width remake)

```
make gold && ./build/gputronic_gold check
→ GOLD GATE: dyno=PASS closedloop=PASS

make transfer && ./build/gputronic_transfer check
→ TRANSFER GATE: dyno=PASS closedloop=PASS

make episode && ./build/gputronic_episode check
→ EPISODE GATE: dyno=PASS closedloop=PASS
```

- Zero-copy: host_ctrl == device_ctrl (mapped)
- Free-run Z ≈ 1.0 after the sustained calibration
- 2026-10-12, this law, tensor sensor off, deadband 0.08, same 5080: Gold free-run 541k pulses/s, closed 363k, gate mean Z 1.463. Transfer 8.48 to 6.03 GFLOP/s, mean Z 1.450. Episode 22.38 to 14.94 GFLOP/s, mean Z 1.468. All three gates passed.
- Open loop at 46 SMs: Z standard deviation 0.051, every bin at 46. The per-SM tally summed to the pulse counter (100 ms ratio 1.000, correlation 0.995). Working SMs varied (29% coefficient of variation).
- The 2026-09-23 one-sided law, for comparison, held mean Z about 1.36–1.38. Calibration is about 3 s: 1.8 s reference plus four 300 ms table steps.
- The count that holds 1.5 is not one number. It comes from the table on the kernel that is running.

## How to run

```bash
make gold
./build/gputronic_gold check
./build/gputronic_gold dyno
./build/gputronic_gold run 30
make transfer && ./build/gputronic_transfer check
make demo
```

## What 1.0 learned from the transfer plant

1. Peak-EMA `rate_ref` inflates Z and can swallow target 1.5
2. 10 µs rate EMA cannot observe a ~300 µs plant
3. (History, sleep control) The old default `sleep_scale=1.2e5` commanded ~8 µs — inside the nanosleep dead zone
4. Mapped atomics dominate the loop; batch work between pulses
5. With sleep as the control, the limit had to reach about 500 µs or the command did nothing

## Proof demo (episode, Gold frozen)

`./build/gputronic_episode proof` — 2026-09-21, this 5080, stock 10 µs path.

32768 tiles/pulse. Without the counter (reference kernel): 1367 GFLOP/s. With the counter on, Z_hat 1.02: 1247 GFLOP/s, so counting costs about 9%. Host wait of half the launch holds Z 1.50 with FLOPs on the same slope. Stock PI then releases to 1240 GFLOP/s. Closed-loop wall Z is 1.34, banging on the clamp, same character as Gold. Detail in `EPISODE.md`.

## Next

1. One shared counter that keeps going after a kernel returns, in code we compile ourselves. Not started.
2. Only then a hardware counter through the stock driver, on a program we do not modify.
3. A GTX 1080 run. The source compiles for sm_61. It has not been executed there.
4. Keep the three checks passing. Do not retune Kp or Ki. Do not clone llama.cpp in place of 1.

## Legacy note

Pre-Gold Cyberpunk v1.13 equilibrium remains historical high-water for *game* coupling. Pre-Gold tree: `archive/pre-gold-2026-05/`.

---
*Last updated: 2026-10-12 — two-sided law, deadband 0.08, tensor sensor off, three gates pass, per-SM tally matches the pulse counter*
