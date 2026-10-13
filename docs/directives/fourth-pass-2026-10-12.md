# Directive for Grok Build: why the fourth pass is faster

Written 12 October 2026. Follows `docs/analysis/smid-2026-10-13.md` on `test-results/2026-10-13_smid` (`5cd317e`). Use that branch's `%smid` slot and `GPUTRONIC_IDLE_QUIET` switch as the starting code.

## What we know

- One block per physical SM. The launch hands out SMs in pairs, 14 apart. Block pass k (12 blocks) gets SM pairs (2k, 2k+1) + 14j for j = 0..5. Blocks 0–35 are passes 0–2, and blocks 36–47 are pass 3, starting on SM 6.
- 36 SMs: flat (CV 0.8%). 46: step at block 36, 1.85x. 48: same step, 1.74x, with the whole fourth pass fast.
- 46 with `GPUTRONIC_IDLE_QUIET=1`: same step, but 1.27x. The slow side rose from about 426k to 702k ops per host bin. Idle polling is most of the gap at 46. It was not tested at 48.
- Working hypothesis to test, not a fact: a working SM runs faster when the SMs physically next to it are idle. In passes 0–2, every working pair has working pairs on both sides. The pass-3 pairs (6,7) have idle neighbours (8,9). This fits the 23 September archive result that working SMs speed up as others drop out.

## Ground rules

Same as `docs/directives/overnight-2026-10-12.md`:
- Put results on `test-results/2026-10-13_fourth-pass`. Never push to or merge into `main`.
- Do not change Kp 0.55, Ki 0.08, the deadband 0.08, or the scaled clamp.
- Stop and write a blocker note after three failures.
- Keep any new switch test-only and off by default. Pass the three checks with every switch off.

For every hold, use open loop for 30 s, the tally on for every block, `%smid` recorded, 1 ms host bins and 20 ms analyzer buckets, dropping the first second. Report the block-to-SM map, the per-block mean ops per host bin, CV, the largest neighbour step and where it is, chip mean, pulse rate, and the 100 ms tally-to-pulse ratio.

## Lookup first: confirm the SM-to-GPC map

This is a lookup only. It does not change Experiments 1–3.

Read the GPU's own SM-to-GPC-and-TPC table on this 5080, through the driver's control calls. The table comes from the GPU firmware; the open kernel driver only passes it on. The calls are in `src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gr.h` in NVIDIA's open-gpu-kernel-modules:
- `NV2080_CTRL_CMD_GR_GET_SM_TO_GPC_TPC_MAPPINGS`: the GPC and TPC for each SM id.
- `NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER`: the global SM order, with GPC, TPC, and virtual GPC ids.
- `NV2080_CTRL_CMD_GR_GET_GPC_MASK` and `NV2080_CTRL_CMD_GR_GET_TPC_MASK`: which GPCs and TPCs are enabled.

Use headers from the open-gpu-kernel-modules tag that matches the installed driver (595.99.02). Use a small standalone program; do not add it to the Gold binary. Allocate an RM client and subdevice through `/dev/nvidiactl`, then issue the controls.

The question to answer: is each 12-block pass one whole GPC? That means blocks 0–11 are all GPC 0, 12–23 all GPC 1, 24–35 all GPC 2, and 36–47 all GPC 3, with SM ids 14 apart being neighbouring TPCs in the same GPC. Cross-check the table against the `%smid` map from `test-results/2026-10-13_smid`.

If the driver refuses a control (for example, because it is privileged or internal on this build), write a blocker note with the exact control, the status code, and the driver version. Then go on to the experiments. Do not infer the map from timing.

## Experiment 1: the quiet-idle switch across counts

Run 36, 46, 48, 60, and 84 SMs, each with `GPUTRONIC_IDLE_QUIET=1` and without it.

- Does the 48 step drop the way 46 did?
- At 60 (passes 0–4), is only pass 4 fast?
- At 84 (all seven passes, no idle blocks), is everything flat? If 84 is flat, the step comes from the boundary between working and idle SMs, not from SM 6.

## Experiment 2: choose the working set by physical SM

Add a test-only switch, for example `GPUTRONIC_ACTIVE_PASSES=0,1,3`. A block works if its launch pass is in the list, decided from its `%smid` pair position ((smid mod 14) / 2), not from its block index. Run each at 36 working SMs, quiet idle off and then on:

| Set | Tests |
|---|---|
| 0,1,2 | Baseline, should match the 36 hold |
| 0,1,3 | Pass 3 gets working neighbours on one side only. Is pass 3 still fast? |
| 0,2,4 | No working pair has a working neighbour pair. Is everything fast and flat? |
| 1,2,3 | Pass 0 SMs idle. Does pass 1 become the fast one? |
| 4,5,6 | Is a whole other region of the chip flat? |

If the fast pass always follows "idle neighbours" and not the specific physical SMs, the cause is shared resources between nearby SMs, such as L1, TPC, or GPC paths. If SMs 6 and 7 (plus 14j) are fast in every set, something about those physical SMs is special.

## Experiment 3: spacing

Using the same switch, run 12 SMs as one pass (pass 0, then pass 3) and 12 SMs as six adjacent pairs chosen to fill whole groups of 14, for example SMs 0–11. Compare the per-SM rates. This tells us whether the 14 spacing itself matters.

## Report back

- The confirmed SM-to-GPC-and-TPC map from the lookup (or its blocker note), as a table of block, physical SM, GPC, and TPC, next to the experiment results.
- A results note in `docs/analysis/`, with one table per experiment, using the columns above.
- One paragraph at the top saying which explanation the data supports: idle-neighbour contention, special SMs, spacing, or none. Name the run that decided it.
- Confirm that the three checks passed with every new switch off.
