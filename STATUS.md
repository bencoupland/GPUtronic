# GPUTronic Status – Gold 1.0.0

## Tonight, 2026-10-12

The source in this folder is still the October 5 tree (`main` at `c5a4c84`). The law that passed Gold, transfer, and episode `check` on this 5080 is GitHub branch `controller-fixes` at `7400273`. It is not merged. Deadband 0.08, tensor sensor off by default, Kp 0.55, Ki 0.08, actuator `active_sms`. The write-up is `docs/analysis/ready-2026-10-12.md` on that branch. The sections below describe the tree checked out here.

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

## Verified on RTX 5080 (2026-10-04, kernel 7.0.0-38, driver 595.91.07)

```
make gold && ./build/gputronic_gold check
→ GOLD GATE: dyno=PASS closedloop=PASS

make transfer && ./build/gputronic_transfer check
→ TRANSFER GATE: dyno=PASS closedloop=PASS

make episode && ./build/gputronic_episode check
→ EPISODE GATE: dyno=PASS closedloop=PASS
```

| Check | Free-run | Closed loop | Z mean |
|---|---|---|---|
| Gold | 533k pulses/s | 386k pulses/s | 1.34 |
| Transfer | 7.00 GFLOP/s | 5.46 GFLOP/s | 1.33 |
| Episode | 20.05 GFLOP/s | 14.91 GFLOP/s | 1.35 |

Open-loop fit of rate against SM count: Gold R² 0.98, transfer R² 0.99, episode R² 0.83. The episode curve bends (SMs that stay on speed up). The gate still passed. Open loop, 21 SMs was Z 1.55 on that kernel.

`./build/gputronic_transfer follow` keeps one control thread and holds the request at 1.5 while the work changes from the persistent transfer kernel to the episode shape and back. `gputronic_recalibrate` takes a new free-run on the load that is pulsing. With the half-SM probe run again at the change, both loads stayed in the band under 1.5 (Z 1.30–1.37) and the count moved by about 21 SMs (about 60 on transfer, about 39 on the short launches). Keeping the previous `sms_per_u` and skipping the probe left the short launches at Z 1.21. Detail in `TRANSFER.md`. Kp and Ki were not changed.

On 2026-10-05, `./build/gputronic_transfer share` launched only the metered blocks and gave the same tiles to a second kernel on the rest. At a fixed split of 42 the second kernel ran at 0.36 of its solo rate, and the metered kernel kept 0.70 of its own 42-block rate. Closed loop, same gains, held Z 1.35 at about 49 SMs while the second kernel did 441k pulses/s. A resident grid whose idle blocks sleep instead of exiting left that second kernel at about 0.10 of its solo rate. Detail in `TRANSFER.md`.

Same night, with Space Marine 2 on the 5080 at 4K60, short guest launches finished in 0.2–1.4 ms. A one-second run beside the game did about 540k pulses/s, about a third of that kernel's solo rate, and 21 blocks already filled that slice. The game's SM sample stayed near 50%. The game does not feed our counter. Detail in `results/gputronic_beside_sm2.txt`.

The 2026-09-23 and 2026-09-27 checks were the same band, on kernel 7.0.0-31 for the later of those two.

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
./build/gputronic_transfer follow
./build/gputronic_transfer share
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

1. One shared counter that keeps going after a kernel returns, in code we compile ourselves. The follow run already counted across a returning launch inside one process with its own pump. The share run used a second word so the other kernel could not move Z. The open job is one word as the normal meter, without a special mode.
2. Only then try a hardware counter through the stock driver, on a program we do not modify. That will show whether we still need a driver change for the count, for holding blocks back, or for both.
3. Keep the three checks passing. The self-test kernel was not in the follow trace. Its closed loop is still the gold check. Do not clone llama.cpp in place of 1–2.

## Legacy note

Pre-Gold Cyberpunk v1.13 equilibrium remains historical high-water for *game* coupling. Pre-Gold tree: `archive/pre-gold-2026-05/`.

---
*Last updated: 2026-10-05 — share run: launch width leaves a real rate for a second kernel*
