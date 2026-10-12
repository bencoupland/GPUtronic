# Directive for Grok Build: why GPC 3 is faster

Written 12 October 2026. Follows `docs/analysis/fourth-pass-2026-10-13.md` on `test-results/2026-10-13_fourth-pass` (`876c004`). Start from that branch's code: the `%smid` slot, `tools/sm_gpc_map.c`, the pass and SM working-set switches, and `GPUTRONIC_IDLE_QUIET`.

## What we know

- The firmware map is confirmed. Each 12-block launch pass is one whole GPC (pass = `(smid % 14) / 2`), and blocks 36–47 are GPC 3. One block sits on each physical SM.
- GPC 3 is faster wherever it works:
  - 2.11x the other six GPCs at 84 SMs, with nothing idle.
  - 1.54x GPC 0 when each runs alone (801k vs 521k).
  - 1.91x its neighbours when only TPC 0 of GPCs 0–5 works.
- Inside a GPC, the 12 SMs match. The step is between GPCs.
- Idle polling is a second, separate effect. At 36 working SMs with `GPUTRONIC_IDLE_QUIET=1`, GPC 3's lead fell to about 1.03x, and every GPC roughly doubled. At 84 SMs there are no idle blocks, so the quiet switch does nothing, and GPC 3 stays 2.11x. The idle-neighbour idea is ruled out.
- Unexplained from the fourth pass:
  - At 60 SMs with quiet on, GPCs 1 and 2 rose to about 704k, but GPCs 0 and 4 stayed near 373k.
  - In the passes 0, 2, 4 set, GPC 2 ran 1.20x GPCs 0 and 4, with or without quiet.
- Working hypothesis, not a fact: GPC 3 has a shorter or less-contended path to mapped host memory. Every working pass does a mapped PCIe atomic on `total_work_pulses`, and every block reads mapped control words. At full width, that traffic is the only shared, chip-wide path that grows with working SMs.

## Who owns the mapped words (read this first)

The control words are GPUtronic's own. They are fields of `struct GPUTronicControl` in `include/gputronic.h`, written by this project, not by NVIDIA's driver:
- **Commands**, which the host writes and the device reads: `control_flags`, `active_sms` (the throttle cable), and `work_scale_milli` (test load, read only when the LOAD flag is set).
- **Sensors**, which the device writes and the host reads: `total_work_pulses` (atomicAdd every pass), `tensor_work_pulses`, and the separate T11 per-SM tally array.
- **Telemetry**, which only the host writes and reads: Z, `rate_ema`, the integral, and so on. The kernel never reads these.

The throttle cable, the control words, and the sensors all use one method: `cudaHostAlloc(..., cudaHostAllocMapped)` plus `cudaHostGetDevicePointer` in `src/gputronic_gold.cu`. The driver only pins a page of host RAM and maps it into the GPU's address space, as opaque memory. Nothing in the driver interprets these words, and no driver code runs on each access. After setup, every device read of a `volatile` field is a load across PCIe to host memory, and every pulse is a PCIe atomic. Commands and sensors share that one path.

So everything this pass tests is inside GPUtronic's own code. Moving the pulse atomic, or the command reads, off the per-pass hot path is a change to how often our kernel crosses PCIe. It needs no driver change.

## Ground rules

- Put results on `test-results/2026-10-13_fifth-pass`. Never push to or merge into `main`.
- Do not change Kp 0.55, Ki 0.08, the deadband 0.08, or the scaled clamp (3.36 SMs). The tally stays out of the PI.
- Keep every new switch test-only and off by default. Pass the Gold, transfer, and episode checks with every switch off.
- Stop and write a blocker note after three failures.

For every hold, use open loop for 30 s, the tally on for every block, `%smid` recorded, 1 ms host bins and 20 ms analyzer buckets, dropping the first second. Report a per-GPC mean, the CV, the largest step and where it is, chip mean, pulse rate, and the 100 ms tally-to-pulse ratio. Where the pulse counter is replaced (Experiment 1), say how the ratio was computed.

## Experiment 1: take the mapped traffic out of the working path (first priority)

Add one test-only switch, for example `GPUTRONIC_PULSE_LOCAL=1`. Each block counts its pulses in device memory or a register, and only the block leader adds its batch to the mapped `total_work_pulses` once per bin, about 1 ms. This takes only the sensor write off the hot path. Leave the command reads (`control_flags`, `active_sms`, `work_scale_milli`) unchanged at first, so the two parts of the shared path are tested one at a time.

Run 84 SMs, then GPC 0 alone and GPC 3 alone, each with the switch off and then on.

- If GPC 3's lead at 84 falls toward 1x with the switch on, the per-pass mapped atomic is the path that favours GPC 3.
- If it stays near 2x, add a second switch that caches the command words: the leader reads them once per bin and shares them through shared memory. Rerun 84. If the lead falls then, it is the mapped reads.
- If neither moves it, the difference is not mapped traffic. Go to Experiment 2 with that stated plainly.

## Experiment 2: per-GPC memory path microbenchmark

Write a standalone kernel, kept out of the Gold binary, like `tools/sm_gpc_map.c`. Run 12 blocks on one GPC at a time, selected by `%smid`, for each of the 7 GPCs, with every other block exiting immediately. Measure per-SM throughput and mean latency for:

1. Pure compute: the Gold FMA loop with no memory traffic. This is the control. All GPCs should match.
2. Device-memory atomics on one shared address.
3. Device-memory reads that stream through L2.
4. Mapped host-memory reads.
5. Mapped host-memory atomics, the same pattern as the Gold pulse.

Report a 7-GPC by 5-test table. Whichever test reproduces GPC 3's lead names the path. Note that the 5080 has GDDR7 and no HBM or NUMA, so do not try per-GPC NUMA affinity.

## Experiment 3: placement inside one GPC

Using the SM working-set switch, quiet on, run GPC 0 and then GPC 3 with these working sets, and compare per-SM rates:

| Set | SMs per GPC |
|---|---|
| 1 TPC, both SMs | 2 |
| 2 SMs on 2 TPCs, one SM each | 2 |
| TPCs 0, 1, 2 (clustered, by firmware TPC id) | 6 |
| TPCs 0, 2, 4 (spread) | 6 |
| All 6 TPCs | 12 |

This tells us whether SM choice inside a GPC matters, and whether two SMs sharing a TPC slow each other. Use the firmware TPC ids from `tools/sm_gpc_map.c`, not launch order.

## Experiment 4: the two leftovers

- Rerun the 60-SM hold with quiet on, 3 times. Is the split between GPCs 1 and 2 (about 704k) and GPCs 0 and 4 (about 373k) repeatable?
- Rerun passes 0, 2, 4 with quiet on, 3 times. Is GPC 2's 1.20x repeatable?

If they repeat, list them as open. Do not build a theory on them yet.

Physical SM ids are already recorded, and the 36, 46, and 48 holds are done. They are not repeated here except where an experiment above calls for them.

## Report back

- A results note in `docs/analysis/`, with one table per experiment.
- One paragraph at the top naming the path that favours GPC 3, or saying none was found. Name the run that decided it.
- Confirm the three checks passed with every new switch off.
