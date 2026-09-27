# GPUTronic

Closed-loop control of instruction-flow throughput on a GPU.

Reference: **GPUTronic 1.0 Gold**

![GPUTRONIC](docs/assets/gputronic-mark.jpg)

A modern GPU is one of the last machines we still treat as a firehose. Launch a kernel. Wait. Read a temperature a few times a second. If it gets hot, the driver pulls clocks. If it does not, we assume the work happened. There is no tachometer on useful instruction flow. There is no setpoint. There is no short cable from an observer to an actuator. Thermal management is closed-loop. The work itself is not.

That is strange, once you notice it. Every other plant we take seriously — a shaft, a current, a cabin, a fuel rail — has a measured rate, a hidden state we estimate, and a command we can apply faster than the plant moves. A GPU has all of those pieces. Occupancy, memory stalls, issue slots, and `__nanosleep` are not mysteries. What was missing was an honest interface: one number for useful work, one definition of impedance that is 1 when the device is free-running, and one command that actually changes the rate.

GPUTronic is that interface, checked on three test kernels. The control in this tree is how many SMs do the work (`active_sms`). An earlier version used sleep duration instead. That source is in `archive/gold-1.0.0-nanosleep/`. The long function-frame doc still describes the sleep version, with a note at the top.

The specification of record is [`docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md`](docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md). This README is the tour.

## The claim

Instruction-flow throughput is a feedback variable.

If you can observe useful work faster than the plant’s period, form an impedance from it, and actuate without fighting the stock thermal path, then closed-loop control of that rate is ordinary. Not a benchmark hack. Not a per-game special case. The same job a PI controller already does everywhere else.

GPUTronic does not claim to control someone else’s closed shaders. It claims something smaller and, if it holds up, more important: the *plant interface* is practical. Once that is boring, the rest — workload-class setpoints, embedding, an honestly named contending mode — is control engineering, not invention.

## Three axes, one loop

The names are borrowed from field-oriented control because the *job* is the same: decouple what you care about from what you must not fight, and estimate the state you cannot read directly.

| Axis | Quantity | Who owns it |
|------|----------|-------------|
| **Q** | Useful work rate | GPUTronic. Aggregate pulse tachometer. |
| **Z** | Impedance proxy, \(Z = r_{\mathrm{ref}} / r\) | GPUTronic. Free-run calibrates to \(Z \approx 1\). |
| **D** | Thermal and power | Stock NVIDIA driver. Not in the fast loop. |

Q is not “utilization.” Utilization can be high while useful work is stalled. Q is a count of completed work units, incremented with `atomicAdd` into one aggregate, `total_work_pulses`. Last-SM overwrite of that counter is forbidden. That single rule is the difference between a governor and a ghost.

Z is not ohms and not a phase margin. \(Z = 1.5\) means the GPU is doing about two-thirds of the speed it had when nothing was holding it back. Turning SMs off lowers that speed and raises Z. There is no extra minus sign in the controller. Full speed is every SM running. The point of the loop is to land near the speed you asked for when conditions are not ideal.

Temperature and power stay with the NVIDIA driver. A 25 ms temperature poll does not belong in the 10 µs control thread. If the driver slows the chip, the work rate falls, Z rises, and this loop eases off. That is the intended split.

A 2-state Kalman filter estimates Z and how fast Z is changing. The controller does not use the rate-of-change yet. It is logged.

The control value is `active_sms` in a shared control block. That many SMs run the tiles and increment the counter. The others skip the work. At least 4 stay on, so the kernel is never empty. On this RTX 5080 the CPU and the GPU see the same memory. The kernel reads the count every pass. There is no 20–50 µs copy. If the mapping fails, startup aborts. A slow link between the two would be a different system. Sleep duration was the first attempt. On this 5080 it does little below about 200 µs, jumps near 500 µs, and then longer sleeps do not slow the chip further. Sleeping some SMs while others kept working did nothing. That is why the count replaced it.

## Why 1.0 is tagged Gold

There was a long ladder — a GTX 1080 proof of concept in this repo’s history, then persistent kernels, PI, Kalman, game coupling, many forks. The idea survived. The *contract* did not. Counters that did not aggregate. A Z that stuck at 0.25 or 4.0. A loop that looked closed because the calibration had already parked the measurement above the setpoint.

1.0 Gold is the cleanup that makes the claim reproducible tomorrow.

Three mistakes that look like a working controller until you measure the work:

1. **Peak-hold calibration.** Taking the peak of a rate EMA during free-run inflates \(r_{\mathrm{ref}}\). Free-run Z then sits at 1.4–1.7. A one-sided PI, which only adds sleep when \(Z\) is *below* target, never engages. The log says “tracking.” The actuator is at base sleep. Nothing moved.

2. **A 10 µs rate EMA.** The persistent loop on this 5080 is about 300 µs, set by mapped atomics, not by the math inside the tile. Forming rate on a 10 µs tick and holding on empty samples leaves the EMA at free-run while a wall-clock pulse window already shows the cut. The observer is blind to the plant it is supposed to regulate. GPUTronic forms rate over ~8 ms, then a slow EMA.

3. **A scale that commands 8 µs.** `__nanosleep` on this plant is almost invisible below ~200 µs and bites near 500 µs. Stock `sleep_scale = 1.2e5` at a typical error commands about 8 µs. After calibration, GPUTronic measures the loop period and sets scale so a 0.3 error maps onto roughly one period — the start of the cliff. Do not retune Kp to hide a dead-zone actuator.

If Z is already at or above the target, every SM runs. If Z is only a little short, the SM count stays put, so a small error does not jump back to full speed. Temperature stays with the driver. The 8 µs scale in the list above is a reason the sleep-duration control was dropped. Adjusting Kp does not fix a control that cannot move the speed smoothly.

## What it does on this machine

Hardware of record: RTX 5080 (GB203, 84 SMs), CUDA 12.8.

The sleep tables below are from the older sleep-duration control, kept because they show the jump. Current numbers are in `STATUS.md`. On 2026-09-23 all three checks passed with the SM count as the control. Closed-loop average Z was about 1.36–1.38. Turning the request back to full speed returned the rate to the unthrottled run.

**Self-test plant** — GPUTronic’s own persistent kernel. Dyno, open-loop sleep sweep, rate from wall \(\Delta p / \Delta t\):

| sleep | pulse/s | Z |
|------:|--------:|--:|
| 0 | ~499k | ≈ 1.00 |
| 200 µs | ~469k | almost flat |
| 500 µs | ~314k | ≈ 1.58 |

Closed-loop, target \(Z = 1.50\): 445k → 324k pulse/s (−27%). Mean Z ≈ 1.37. Sleep ≈ 415 µs, in the authority band.

**Transfer plant** — a second, cooperative kernel. Real 8×8×8 FP32 GEMM tiles, 32 tiles per pulse, 32768 FLOP/pulse. Same GPUTronic controller. No gain override.

Open-loop: 9.03 → 5.15 GFLOP/s at 500 µs, locked to tile rate. Free-run Z ≈ 1.02.

Closed-loop: 8.77 → 6.55 GFLOP/s (−25%). Mean Z ≈ 1.34.

That second number is the one that matters. Pulse rate and counted FLOPs moved together, in the direction the model predicts, on a plant that is not `gputronic_gold.cu`’s synthetic fmaf. If those two ever disagree in sign, the tachometer has stopped measuring the work.

Both gates are one command:

```bash
make gold && ./build/gputronic_gold check
# === GOLD GATE: dyno=PASS closedloop=PASS ===

make transfer && ./build/gputronic_transfer check
# === TRANSFER GATE: dyno=PASS closedloop=PASS ===

make episode && ./build/gputronic_episode check
# === EPISODE GATE: dyno=PASS closedloop=PASS ===
```

CSV lands under `results/` every run. Dyno rate is always a wall window, never the lagging EMA alone.

## The shared control block

The CPU and the GPU share one struct, `GPUTronicControl`, allocated with `cudaHostAllocMapped`.

The GPU writes one thing that matters: `total_work_pulses`, with `atomicAdd`, from the thread that finishes a unit of work. Fence every 32 pulses, not every pulse. A fence on every pulse swamps the copy.

The CPU writes `active_sms`, then `__sync_synchronize()`. That many SMs do the work. The others skip it.

The rest of the struct is readings or flags (STOP / PAUSE / RESET). If a value is not in that struct, the GPU cannot see it.

A kernel we compile ourselves does this:

1. `launch_selftest_kernel = 0`
2. `gputronic_create`, then `cudaHostGetDevicePointer` on `gputronic_get_control()`
3. Launch the kernel before or with `gputronic_start`, so the free-run measurement sees pulses
4. `atomicAdd` finished work into `total_work_pulses`
5. Only the first `active_sms` SMs run that work
6. Stop when asked

Transfer is the test where the kernel stays resident. Episode (`src/gputronic_episode.cu`) is the test where the kernel returns and the 8 ms window still sees Z. A program we do not modify does not do any of this. That is a separate problem.

## Timescales, because they are the whole story

| Clock | Period | Role |
|-------|--------|------|
| GPU inner work | nanoseconds | fmaf / tile |
| Persistent loop | ~300 µs here | mapped atomic, then the SM check |
| Control *path* | 10 µs | thread wake, not the observer |
| Rate window | ~8 ms | what Z is actually formed from |
| Calibration | 0.7 s warmup + 1.1 s sustain | \(r_{\mathrm{ref}}\) |
| NVML / thermal | ~25 ms | not in this loop |

The control thread can wake every 10 µs. That does not mean the rate estimate is that fast. The rate is a count of pulses over about 8 ms. The 10 µs wake is only so the CPU is not late writing the SM count.

## What this is not

It does not control a game or a library you did not build, just because a process is running. Those kernels do not add to the pulse count and do not read `active_sms`.

It does not replace the NVIDIA driver. Clocks, power, and temperature stay there.

The transfer kernel’s few GFLOP/s are not a speed claim. The tiles are there so the FLOPs can be counted. They moved when Z was commanded. cuBLAS is not the comparison.

It is not general yet. Three kernels on one 5080. The next step is the same loop on those kernels when the workload changes, without fitting a new curve by hand.

## Quick start (RTX 5080 / CUDA 12.8)

```bash
make gold
./build/gputronic_gold check
./build/gputronic_gold run 30
make transfer && ./build/gputronic_transfer check
make episode && ./build/gputronic_episode check
make demo && ./build/gputronic_demo
```

```
include/gputronic.h                      public C API
src/gputronic_gold.cu                    core + CLI
src/gputronic_transfer.cu                second plant (persistent)
src/gputronic_episode.cu                 third plant (launch-complete)
docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md
GOLD.md  STATUS.md  TRANSFER.md  EPISODE.md
archive/v0.1-github/                     first public PoC (GTX 1080)
```

Read the function frame with the source open, starting at the control block, not at `main`. Function IDs F-01…F-15 in that file match comments in the code.

## Why it should become ordinary

Closed-loop regulation of a measured work rate is not a new idea. It is what we already do wherever the plant is worth controlling. A GPU execution path is such a plant. The unusual part was getting the tachometer, the calibration, and the actuator into the same room without lying to ourselves.

If that holds on the workloads people actually run, instruction-throughput control belongs in the same category as any other feedback loop we no longer argue about. GPUTronic 1.0 Gold is the interface and the evidence. The rest is embedding, and then — only with a different name — contention.

## License

MIT
