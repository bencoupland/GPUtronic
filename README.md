# GPUTronic

Closed-loop control of instruction-flow throughput on a GPU.

Reference: **GPUTronic 1.0 Gold**

![GPUTRONIC](docs/assets/gputronic-mark.jpg)

A GPU driver will slow the chip if it gets hot. It will not tell you whether the work you asked for is the work that finished. You launch a kernel, wait, and take whatever rate comes out.

Other machines we control have a measured rate, an estimate of what we cannot see directly, and a command that moves the rate. A GPU can have those too. What was missing was a counter of finished work, a way to say “this is full speed,” and a command that actually changes that speed.

GPUTronic is that interface, checked on three test kernels. The control in this tree is how many SMs do the work (`active_sms`). An earlier version used sleep duration instead. That source is in `archive/gold-1.0.0-nanosleep/`. The long function-frame doc still describes the sleep version, with a note at the top.

The function-by-function writeup is [`docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md`](docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md). This README is the short tour.

## The claim

The rate of finished work can be a feedback signal.

Count that work. Compare it with the speed you measured when nothing was holding the GPU back. Change how many SMs are allowed to run, and leave clocks and temperature to the driver. Then a normal PI controller can hold a requested rate. This does not control a program you did not build. It does not beat an unthrottled run. Full speed is every SM on.

## Three axes, one loop

The names are borrowed from motor control because the split is the same: measure the thing you care about, estimate what you cannot read, and do not fight the limit that already owns temperature.

| Name | What it is | Who handles it |
|------|------------|----------------|
| **Q** | Finished work per second | GPUTronic. One running total of pulses. |
| **Z** | \(Z = r_{\mathrm{ref}} / r\) | GPUTronic. Full speed is about 1. |
| **D** | Temperature and power | The NVIDIA driver. Not in this loop. |

Q is not “GPU utilization.” Utilization can read 100% while useful work is stalled. Q is a count of finished work units, added with `atomicAdd` into one total, `total_work_pulses`. Do not let the last SM overwrite that total with its own count. That bug made Z stick at a floor or a ceiling.

Z is not ohms and not a phase margin. \(Z = 1.5\) means the GPU is doing about two-thirds of the speed it had when nothing was holding it back. Turning SMs off lowers that speed and raises Z. There is no extra minus sign in the controller. Full speed is every SM running. The point of the loop is to land near the speed you asked for when conditions are not ideal.

Temperature and power stay with the NVIDIA driver. A 25 ms temperature poll does not belong in the 10 µs control thread. If the driver slows the chip, the work rate falls, Z rises, and this loop eases off. That is the intended split.

A 2-state Kalman filter estimates Z and how fast Z is changing. The controller does not use the rate-of-change yet. It is logged.

The control value is `active_sms` in a shared control block. That many SMs run the tiles and increment the counter. The others skip the work. At least 4 stay on, so the kernel is never empty. On this RTX 5080 the CPU and the GPU see the same memory. The kernel reads the count every pass. There is no 20–50 µs copy. If the mapping fails, startup aborts. A slow link between the two would be a different system. Sleep duration was the first attempt. On this 5080 it does little below about 200 µs, jumps near 500 µs, and then longer sleeps do not slow the chip further. Sleeping some SMs while others kept working did nothing. That is why the count replaced it.

## Why 1.0 is tagged Gold

The repo starts with a GTX 1080 prototype in January 2026, then a run of experiments through May 2026, then this 1.0 tree. Those older trees are still in `archive/`. What failed in the later experiments was the counter: it was not one total, Z stuck near 0.25 or 4.0, or the free-run measurement was a peak instead of a sustained window, so the loop thought it was already at the target and never did anything.

1.0 is the version you can build and check again tomorrow.

Three mistakes that look like a working controller until you measure the work:

1. **Peak-hold calibration.** Taking the peak of a rate EMA during free-run inflates \(r_{\mathrm{ref}}\). Free-run Z then sits at 1.4–1.7. A one-sided PI, which only adds sleep when \(Z\) is *below* target, never engages. The log says “tracking.” The actuator is at base sleep. Nothing moved.

2. **A 10 µs rate EMA.** The persistent loop on this 5080 is about 300 µs, set by mapped atomics, not by the math inside the tile. Forming rate on a 10 µs tick and holding on empty samples leaves the EMA at free-run while a wall-clock pulse window already shows the cut. The observer is blind to the plant it is supposed to regulate. GPUTronic forms rate over ~8 ms, then a slow EMA.

3. **A gain that commands about 8 µs of sleep.** On this 5080, `__nanosleep` does little below about 200 µs and then jumps near 500 µs. The old default gain, at a normal error, asked for about 8 µs. The GPU did not slow down. Changing Kp does not fix a control input the hardware ignores.

The controller works in both directions around a starting SM count measured at calibration: below target it turns SMs off, above target it turns them back on. If Z is within 0.015 of the target, the SM count stays put, so a small error does not jump back to full speed. Temperature stays with the driver. The 8 µs scale in the list above is a reason the sleep-duration control was dropped. Adjusting Kp does not fix a control that cannot move the speed smoothly.

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

That second number is the one that matters. Pulse rate and counted FLOPs moved together, on a kernel that is not the self-test’s dummy math. If those two ever move in opposite directions, the counter is no longer measuring that work.

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

## Two sensors

The Gold kernel can count two kinds of work: CUDA-core FMAs (the original counter) and tensor-core
matrix steps (one extra warp per block, sm_70 and newer). Each has its own reference rate and its own Z.
The SM count is still the only control. The controller follows whichever Z is higher, so it turns
off the fewest SMs that bring either kind of work down to the target. Not yet run on a GPU.

## Test kernels

- **Gold**: GPUTronic's own self-test kernel and the reference controller (`src/gputronic_gold.cu`).
- **Transfer**: a persistent kernel doing small GEMM tiles that reports its own work (`src/gputronic_transfer.cu`).
- **Episode**: a GEMM kernel launched once per work episode, closer to a normal application (`src/gputronic_episode.cu`).

Calibration takes about 3 s from start: a 1.8 s full-speed reference, then four 300 ms steps at fewer SMs. That is about 0.7 s longer than before PR #1 (the old half-width probe took 0.5 s).

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
src/gputronic_transfer.cu                Transfer kernel (persistent)
src/gputronic_episode.cu                 Episode kernel (launch-complete)
docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md
GOLD.md  STATUS.md  TRANSFER.md  EPISODE.md
archive/v0.1-gtx1080-2026-01/            first public PoC (GTX 1080)
```

Read the function frame with the source open, starting at the control block, not at `main`. Function IDs F-01…F-15 in that file match comments in the code.

## Next

A feedback loop on a measured rate is ordinary. The work here was getting the counter, the full-speed measurement, and the SM count to agree with each other.

Next: test on more workloads.

## License

MIT
