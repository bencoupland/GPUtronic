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
| Gains | Kp=0.55, Ki=0.08. Half-population probe sets `sms_per_u`. Do not retune Kp. |
| Policy | One-sided PI + 2-state Kalman. Deadband holds the count. At target, full width. |
| Actuator | `active_sms`. Idle SMs skip the work and do not pulse. Floor 4. |
| Tachometer | `atomicAdd` on aggregate `total_work_pulses` |

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
- Width dyno: rate falls as SMs drop, R² 0.98 (Gold) and 0.99 (transfer), 84 down to 4
- Closed-loop average Z about 1.36–1.38 on all three kernels, about a tenth under 1.50. The SM count moves a little around that. Setting the target back to full speed brings every SM back on.
- The count that holds 1.5 is not one number. Transfer ~48. Short episode ~21. Show kernel ~54–58. Tiles from 1024 to 65536 on the show kernel barely moved it.

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
3. Stock `sleep_scale=1.2e5` commands ~8 µs — inside the nanosleep dead zone
4. Mapped atomics dominate the loop; batch work between pulses
5. With sleep as the control, the limit had to reach about 500 µs or the command did nothing

## Proof demo (episode, Gold frozen)

`./build/gputronic_episode proof` — 2026-09-21, this 5080, stock 10 µs path.

32768 tiles/pulse. Twin 1367 GFLOP/s. Tachometer on, Z_hat 1.02: 1247 GFLOP/s (tax 0.91). Host wait of half the launch holds Z 1.50 with FLOPs on the same slope. Stock PI then releases to 1240 GFLOP/s. Closed-loop wall Z is 1.34, banging on the clamp, same character as Gold. Detail in `EPISODE.md`.

## Next

1. Same binary, half-SM probe left on, no hand-adjusted gain. Hold a band under 1.5 on the self-test, transfer, and episode kernels while the workload changes and the request stays the same.
2. One shared counter that keeps going after a kernel returns, in code we compile ourselves.
3. Only then try a hardware counter through the stock driver, on a program we do not modify. That will show whether we still need a driver change for the count, for holding blocks back, or for both.
4. Keep the three checks passing. Do not clone llama.cpp in place of 1–3.

## Legacy note

Pre-Gold Cyberpunk v1.13 equilibrium remains historical high-water for *game* coupling. Pre-Gold tree: `archive/pre-gold-2026-05/`.

---
*Last updated: 2026-09-23 — width actuator, three gates PASS on RTX 5080*
