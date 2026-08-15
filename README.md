# GPUTronic

Closed-loop control of instruction-flow throughput on a GPU.

Reference: **Gold 1.0.0**

![GPUTRONIC](docs/assets/gputronic-mark.jpg)

A modern GPU is one of the last machines we still treat as a firehose. Launch a kernel. Wait. Read a temperature a few times a second. If it gets hot, the driver pulls clocks. If it does not, we assume the work happened. There is no tachometer on useful instruction flow. There is no setpoint. There is no short cable from an observer to an actuator. Thermal management is closed-loop. The work itself is not.

That is strange, once you notice it. Every other plant we take seriously — a shaft, a current, a cabin, a fuel rail — has a measured rate, a hidden state we estimate, and a command we can apply faster than the plant moves. A GPU has all of those pieces. Occupancy, memory stalls, issue slots, and `__nanosleep` are not mysteries. What was missing was an honest interface: one number for useful work, one definition of impedance that is 1 when the device is free-running, and one command that actually changes the rate.

GPUTronic Gold is that interface, frozen, gated, and exercised on two plants.

The specification of record is [`docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md`](docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md). This README is the tour.

## The claim

Instruction-flow throughput is a feedback variable.

If you can observe useful work faster than the plant’s period, form an impedance from it, and actuate without fighting the stock thermal path, then closed-loop control of that rate is ordinary. Not a benchmark hack. Not a per-game special case. The same job a PI controller already does everywhere else.

Gold does not claim to FOC someone else’s closed shaders. It claims something smaller and, if it holds up, more important: the *plant interface* is practical. Once that is boring, the rest — workload-class setpoints, embedding, an honestly named contending mode — is control engineering, not invention.

## Three axes, one loop

The names are borrowed from field-oriented control because the *job* is the same: decouple what you care about from what you must not fight, and estimate the state you cannot read directly.

| Axis | Quantity | Who owns it |
|------|----------|-------------|
| **Q** | Useful work rate | Gold. Aggregate pulse tachometer. |
| **Z** | Impedance proxy, \(Z = r_{\mathrm{ref}} / r\) | Gold. Free-run calibrates to \(Z \approx 1\). |
| **D** | Thermal and power | Stock NVIDIA driver. Not in the fast loop. |

Q is not “utilization.” Utilization can be high while useful work is stalled. Q is a count of completed work units, incremented with `atomicAdd` into one aggregate, `total_work_pulses`. Last-SM overwrite of that counter is forbidden. That single rule is the difference between a governor and a ghost.

Z is not ohms and not phase margin in degrees. \(Z = 1.5\) means the plant is delivering two-thirds of its calibrated free-run rate. Sleep goes up, rate goes down, Z goes up. The plant gain is positive. The PI has no extra minus sign.

D stays with NVSMC. A 25 ms NVML poll cannot sit in a 10 µs path, and fighting the driver’s thermal loop is how you get a worse plant, not a better one. If the hardware throttles, Q falls, Z rises, and the governor backs off. That is collaboration, not domination.

The observer is a 2-state Kalman filter on \([Z, \dot{Z}]\). Gold 1.0 publishes \(\dot{Z}\) and does not yet feed it as a D-term. The estimate is on the cable for the next honest use, not as decoration.

The actuator is `__nanosleep` on a mapped control block. On this RTX 5080 the host pointer equals the device pointer. The kernel re-reads `throttle_sleep_ns` every chunk. No 20–50 µs memcpy. If mapping fails, Gold aborts. A slow cable is a different plant.

## Why “Gold”

There was a long ladder — a GTX 1080 proof of concept in this repo’s history, then persistent kernels, PI, Kalman, game coupling, many forks. The idea survived. The *contract* did not. Counters that did not aggregate. A Z that stuck at 0.25 or 4.0. A loop that looked closed because the calibration had already parked the measurement above the setpoint.

Gold is the cleanup that makes the claim reproducible tomorrow.

Three lies, specifically, that look like control until you measure useful work:

1. **Peak-hold calibration.** Taking the peak of a rate EMA during free-run inflates \(r_{\mathrm{ref}}\). Free-run Z then sits at 1.4–1.7. A one-sided PI, which only adds sleep when \(Z\) is *below* target, never engages. The log says “tracking.” The actuator is at base sleep. Nothing moved.

2. **A 10 µs rate EMA.** The persistent loop on this 5080 is about 300 µs, set by mapped atomics, not by the math inside the tile. Forming rate on a 10 µs tick and holding on empty samples leaves the EMA at free-run while a wall-clock pulse window already shows the cut. The observer is blind to the plant it is supposed to regulate. Gold forms rate over ~8 ms, then a slow EMA.

3. **A scale that commands 8 µs.** `__nanosleep` on this plant is almost invisible below ~200 µs and bites near 500 µs. Stock `sleep_scale = 1.2e5` at a typical error commands about 8 µs. Gold measures the loop period after calibration and sets scale so a 0.3 error maps onto roughly one period — the start of the cliff. Do not retune Kp to hide a dead-zone actuator.

One-sided PI follows from the plant: if Z is already at or above target, hold base sleep (max Q). Do not undersleep below base to “fight” high impedance on a synthetic plant. Thermal remains the driver’s problem.

## What it does on this machine

Hardware of record: RTX 5080 (GB203, 84 SMs), CUDA 12.8.

**Self-test plant** — Gold’s own persistent kernel. Dyno, open-loop sleep sweep, rate from wall \(\Delta p / \Delta t\):

| sleep | pulse/s | Z |
|------:|--------:|--:|
| 0 | ~499k | ≈ 1.00 |
| 200 µs | ~469k | almost flat |
| 500 µs | ~314k | ≈ 1.58 |

Closed-loop, target \(Z = 1.50\): 445k → 324k pulse/s (−27%). Mean Z ≈ 1.37. Sleep ≈ 415 µs, in the authority band.

**Transfer plant** — a second, cooperative kernel. Real 8×8×8 FP32 GEMM tiles, 32 tiles per pulse, 32768 FLOP/pulse. Same Gold controller. No gain override.

Open-loop: 9.03 → 5.15 GFLOP/s at 500 µs, locked to tile rate. Free-run Z ≈ 1.02.

Closed-loop: 8.77 → 6.55 GFLOP/s (−25%). Mean Z ≈ 1.34.

That second number is the one that matters. Pulse rate and counted FLOPs moved together, in the direction the model predicts, on a plant that is not `gputronic_gold.cu`’s synthetic fmaf. If those two ever disagree in sign, the tachometer has stopped measuring the work.

Both gates are one command:

```bash
make gold && ./build/gputronic_gold check
# === GOLD GATE: dyno=PASS closedloop=PASS ===

make transfer && ./build/gputronic_transfer check
# === TRANSFER GATE: dyno=PASS closedloop=PASS ===
```

CSV lands under `results/` every run. Dyno rate is always a wall window, never the lagging EMA alone.

## The cable

Host and device share one struct, `GPUTronicControl`, allocated with `cudaHostAllocMapped`.

Device writes one thing that matters: `total_work_pulses`, via `atomicAdd`. Warp or block leaders only. Fence every 32 pulses, not every pulse — or the fence drowns the actuator.

Host writes one command that matters: `throttle_sleep_ns`. Then `__sync_synchronize()`.

Everything else on the struct is telemetry or flags (STOP / PAUSE / RESET). The kernel must run if the snapshots are stale. If a signal is not in that struct, the device cannot see it. The header *is* the cable.

A cooperative workload — the only mode that is actually Gold — does this:

1. `launch_selftest_kernel = 0`
2. `gputronic_create`, then `cudaHostGetDevicePointer` on `gputronic_get_control()`
3. Launch your persistent kernel before or with `gputronic_start`, so calibration sees pulses
4. `atomicAdd` completed work into `total_work_pulses`
5. Read `throttle_sleep_ns` every chunk and `__nanosleep` it
6. Honour STOP

That is the entire embed contract. The transfer plant is the existence proof. llama.cpp, a custom GEMM, a decode kernel: same five steps. Uninstrumented apps are a different plant. Call that contending, or observe-only, and do not call it this.

## Timescales, because they are the whole story

| Clock | Period | Role |
|-------|--------|------|
| GPU inner work | nanoseconds | fmaf / tile |
| Persistent loop | ~300 µs here | mapped atomic + nanosleep |
| Control *path* | 10 µs | thread wake, not the observer |
| Rate window | ~8 ms | what Z is actually formed from |
| Calibration | 0.7 s warmup + 1.1 s sustain | \(r_{\mathrm{ref}}\) |
| NVML / thermal | ~25 ms | not in this loop |

People hear “100 kHz governor” and imagine the plant is being steered every 10 µs. The path can be that fast. The plant is not. Gold is honest about the mismatch. The observer is sized to the plant. The path is sized to not add another 300 µs of host latency on top.

## What this is not

It is not a magic sidecar that regulates Cyberpunk’s shaders because a process is running. External, uninstrumented kernels have no pulse source and do not honour sleep. That needs hooks or a contending design with different physics. Blurring those is how the v26-era tree got noisy.

It is not a replacement for the NVIDIA driver. Clocks, power, and temperature stay where they are.

It is not 9 GFLOP/s as a performance claim. The transfer tiles exist to be *counted*. The result is that FLOPs moved when Z was commanded. cuBLAS is not the competitor. Open-loop hope is.

It is not universal yet. Two plants on one 5080. The next honest work is another real cooperative path — inference, a bigger GEMM, a title that will take a hook — and a Gold tag if the calibration or scale law has to change.

## Quick start (RTX 5080 / CUDA 12.8)

```bash
make gold
./build/gputronic_gold check
./build/gputronic_gold run 30
make transfer && ./build/gputronic_transfer check
make demo && ./build/gputronic_demo
```

```
include/gputronic.h                      public C API
src/gputronic_gold.cu                    core + CLI
src/gputronic_transfer.cu                second plant
docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md
GOLD.md  STATUS.md  TRANSFER.md
archive/v0.1-github/                     first public PoC (GTX 1080)
```

Read the function frame with the source open, starting at the cable, not at `main`. Function IDs F-01…F-14 in the spec match comments in the code.

## Why it should become ordinary

Closed-loop regulation of a measured work rate is not a new idea. It is what we already do wherever the plant is worth controlling. A GPU execution path is such a plant. The unusual part was getting the tachometer, the calibration, and the actuator into the same room without lying to ourselves.

If that holds on the workloads people actually run, instruction-throughput control belongs in the same category as any other feedback loop we no longer argue about. Gold 1.0 is the interface and the evidence. The rest is embedding, and then — only with a different name — contention.

## License

MIT
