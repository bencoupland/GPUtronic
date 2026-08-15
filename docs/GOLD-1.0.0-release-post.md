# GPUTronic Gold 1.0.0

Most GPU software still treats the device like a firehose with a thermal fuse. Measure heat. Hope. Throttle clocks.

GPUTronic starts from a different premise: a modern GPU is a dynamical system. It has a useful-work rate, an impedance, and a response to load. If you can observe the right state and act on a short enough cable, the rest is ordinary control engineering.

Gold 1.0.0 is the reference that makes that claim reproducible. It had not been posted yet; this is the first public 1.0, with the transfer-plant corrections already in.

## What it is

On an RTX 5080 (Blackwell, CUDA 12.8):

- Mapped zero-copy control block (`host_ctrl == device_ctrl`)
- Aggregate `atomicAdd` tachometer on `total_work_pulses`
- `Z = rate_ref / rate` after a *sustained* free-run window (not a peak hold)
- Rate formed over ~8 ms, because a 10 µs EMA cannot see a ~300 µs plant
- `sleep_scale` taken from the measured loop period so the PI commands land in the nanosleep authority band (~200–500 µs), not the 8 µs dead zone
- One-sided tracking PI + 2-state Kalman
- One-command gates

```text
make gold && ./build/gputronic_gold check
→ GOLD GATE: dyno=PASS  closedloop=PASS

make transfer && ./build/gputronic_transfer check
→ TRANSFER GATE: dyno=PASS  closedloop=PASS
```

## Numbers from this machine

Gold self-test plant: free-run 445 kpulse/s → 324 kpulse/s at target Z = 1.50 (−27%). Z_mean ≈ 1.37, sleep ≈ 415 µs.

Transfer plant — cooperative 8×8×8 GEMM microtiles, 32768 FLOP per pulse, stock Gold controller, no gain override: 8.77 → 6.55 GFLOP/s (−25%). Free-run Z ≈ 0.98. That is the second-plant test: useful work moves in the direction the model predicts.

Dyno on both plants is flat to ~200 µs and cuts ~30–45% at 500 µs. Mapped atomics set the loop period. That is a fact about this cable, not a tuning footnote.

## Scope

In: cooperative workloads that publish work pulses and honour `throttle_sleep_ns`.

Out: FOC of someone else’s closed shaders. That needs hooks or an honestly named contending mode. Different plant. We will not blur them.

## How to run

```bash
make gold
./build/gputronic_gold check
./build/gputronic_gold run 30
make transfer && ./build/gputronic_transfer check
```

`include/gputronic.h` · `src/gputronic_gold.cu` · MIT

The interesting claim was never “more FPS by force.” It was that instruction-flow impedance is a feedback variable, and that once the interface is honest the rest looks obvious.

— Ben Coupland
