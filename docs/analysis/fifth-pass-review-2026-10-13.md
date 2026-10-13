# Review of the fifth pass: the kernel was stalled on PCIe reads

Grok Bot's reading of `docs/analysis/fifth-pass-2026-10-13.md` on `test-results/2026-10-13_fifth-pass` (`1a8fdd3`). No new runs. No code changes. All numbers are Grok Build's, measured on the RTX 5080 on 12 October 2026.

## The finding

GPC 3's lead came from every block reading the mapped control words on every pass. On the 84-SM hold:

| Hold | GPC 3 vs the others | Pulses/s |
|---|---|---|
| Both switches off | 2.14x (781k vs about 367k ops per host bin) | 547k |
| Pulse atomic batched to about once per ms (`GPUTRONIC_PULSE_LOCAL=1`) | 2.07x | 673k |
| Atomic batched and control words cached to once per block per ms (`GPUTRONIC_CTRL_CACHE`) | 1.03x, largest SM-to-SM gap 1.05x | 255M |

255 million pulses a second at 65,536 ops a pulse is about 17 TFLOP/s. The switches-off hold was about 36 GFLOP/s, so the gain is about 470 times. Chip clocks were the same, about 2890 MHz, on all three holds.

The reading: with both switches off, the kernel spent almost all of its time waiting on mapped reads across PCIe. The SMs were not short of compute. They were queued on the host interface, and GPC 3 was simply the GPC that won that queue most often. That also explains the fourth pass: idle-block polling added to the same queue, so quieting it helped the working blocks.

## The rest of the fifth pass

- **Per-GPC bench** (`tools/gpc_path_bench.cu`, one GPC at a time): no path showed a 2x gap. GPCs 1, 2, and 3 were about 11% faster than GPCs 0, 4, 5, and 6 on every test, pure compute included. That is a small, steady plant fact, not the step.
- **Placement inside a GPC** barely matters. Whether two SMs share a TPC, or a set of TPCs is clustered or spread, moves the rate by a few percent. Per-SM rate falls as more SMs in the GPC work.
- **60-SM split:** with GPCs 1 and 2 at about 1.8x GPCs 0 and 4, quiet on, this repeated on all three runs. It is still open, and no explanation is attached.
- **GPC 2's 1.20x** from the fourth pass did not repeat (1.002x on three runs). It was noise.
- Both new switches stay off by default. The Gold, transfer, and episode checks passed with them off. Kp 0.55, Ki 0.08, deadband 0.08, and the scaled clamp (3.36 SMs) were not changed.

## What it means for the project

The caller's points, recorded as stated:
- The per-SM tally is now a working tachometer. Before, the meter was effectively maxed out and choked: every block was pinned against the same PCIe queue, so what it measured was the queue.
- Individually, the pulses were not worth enough work. Each one carried a round trip across PCIe for 65,536 ops.
- The per-SM counter, together with the tuning work, made the stall visible. Without per-SM numbers, the 2x step and its cause would have stayed hidden in a chip-wide average.

Grok Bot's notes:
- The controller has been governing a plant whose throughput was set by PCIe reads, not by SM count. That is why working SMs sped up as others went idle (23 September), and why Z sat below `84/n`.
- With the words cached, the plant should become compute-bound. Then Z against SM count should move toward equal shares (`84/n`). That has to be measured, not assumed.
- At 17 TFLOP/s the card will draw real power and get hot. The fifth-pass holds stayed between 40 and 55 °C at about 36 GFLOP/s. Clocks and thermals will become part of the plant.
- Batching the pulse word makes it lumpy at 1 ms: the 1 ms ratio was 3.09 on the cached hold. It was already 1.0003 at 20 ms. A controller that samples the pulse word every millisecond would see that lumpiness as noise.
- Caching the control words to once per ms means a command change reaches each block up to 1 ms later. That is new latency in the loop.

## Open design question (for Ben and the caller)

Should the real controller cache the control words, and batch the pulse? Doing so changes the free-run rate, the meaning and value of Z at a given SM count, the calibration (the half-SM probe and `sms_per_u`), and the noise floor. The data says the current design measures and throttles a PCIe queue. Further throughput gains probably need the pulse stream redesigned, not more GPU-side tuning. That decision is not made here. Both switches stay off until it is.
