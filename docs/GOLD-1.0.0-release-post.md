# GPUtronic Gold 1.0.0

> **Superseded.** This describes the original sleep-duration control. Commit `600ed66` (2026-09-23) replaced sleep actuation with SM-count actuation (`active_sms`), and PR #1 removed the remaining sleep API (`gputronic_set_open_loop_sleep`, `gputronic_get_sleep_ns`, the sleep scale settings) and made the controller two-sided. See the README and `docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md` for the current design.

This note is the first public description. At that time the control was how long the kernel slept. The code in the repo now sets how many SMs run. See the README.

Most GPU software still treats the device like a firehose with a thermal fuse. Measure heat. Hope. Throttle clocks.

GPUtronic starts from a different premise: a modern GPU has a work rate you can count, and a response to how hard you let it run. If you can measure that rate and change it without fighting the driver's temperature control, the rest is an ordinary feedback loop.

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

Open-loop sleep did little until about 200 µs and cut about 30–45% at 500 µs. The mapped counter, not the math in the tile, set the loop period.

## Scope

In: cooperative workloads that publish work pulses and honour `throttle_sleep_ns`.

Out: kernels we did not write and that do not read the control block. That is a different problem.

## How to run

```bash
make gold
./build/gputronic_gold check
./build/gputronic_gold run 30
make transfer && ./build/gputronic_transfer check
```

`include/gputronic.h` · `src/gputronic_gold.cu` · MIT

The claim was never “more frames per second.” It was that the rate of finished work can be a feedback signal.

— Ben Coupland
