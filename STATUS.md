# GPUTronic Status – Gold 1.0.0

## Current Working State

**GPUTronic Gold 1.0.0** is the unpublished-until-now reference. Transfer lessons are in this tag.

| Item | Value |
|------|--------|
| Binary | `build/gputronic_gold` |
| Header | `include/gputronic.h` |
| Source | `src/gputronic_gold.cu` |
| Transfer | `src/gputronic_transfer.cu` |
| Library | `make lib` → `build/libgputronic_gold.a` |
| Control period | 10 µs path; rate formed over ~8 ms |
| Z model | Sustained-window `Z = rate_ref / rate` (free-run ≈ 1.0) |
| Target Z | 1.5 |
| Gains | Kp=0.55, Ki=0.08, `sleep_scale` auto from loop period |
| Policy | One-sided tracking PI + 2-state Kalman |
| Actuator | `__nanosleep` via zero-copy `throttle_sleep_ns` |
| Tachometer | `atomicAdd` on aggregate `total_work_pulses` |

## Verified on RTX 5080 (this machine)

```
make gold && ./build/gputronic_gold check
→ GOLD GATE: dyno=PASS closedloop=PASS

make transfer && ./build/gputronic_transfer check
→ TRANSFER GATE: dyno=PASS closedloop=PASS
```

- Zero-copy: host_ctrl == device_ctrl (mapped)
- Free-run Z ≈ 1.0 after sustained cal (not peak-EMA parked at 1.4+)
- Dyno: 500 µs sleep cuts pulse rate ≥25% and (on transfer) GFLOP/s with it
- Closed-loop: useful work drops when Z is commanded to 1.5

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
5. Default `max_sleep_ns` must reach the cliff (~500 µs)

## Next

1. llama.cpp / custom CUDA cooperative hooks
2. Contending mode, named as such
3. Keep both gates green

## Legacy note

Pre-Gold Cyberpunk v1.13 equilibrium remains historical high-water for *game* coupling. Pre-Gold tree: `archive/pre-gold-2026-05/`.

---
*Last updated: Gold 1.0.0 release candidate — both gates PASS on RTX 5080*
