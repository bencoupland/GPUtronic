# The 46-SM tally is two rate groups

Reading of `logs/2026-10-12_t11/load-imbalance-notes.md`, from the same
open-loop log. No new run. No code change. RTX 5080, 84 SMs, driver
595.99.02, CUDA 12.8. Tensor sensor off. Open loop at 46 SMs for 30 s,
tally on for every block. Log: `logs/2026-10-12_t11/sm_all.csv.zst`.

## What the 29% actually is

The coefficient of variation is 29.1% because the per-block means form
two flat groups, not because the blocks scatter.

After the first second (28,981 host rows, the same count for every
block):

| Blocks | Role in this hold | Ops per host bin |
|---|---|---|
| 0–35 | working | 443,816 to 460,120, mean 452,377 |
| 36–45 | working | 824,911 to 833,115, mean 829,062 |
| 46–83 | idle | no ops |

The fast group averages 1.83 times the slow group. The analyzer's range,
443,816 to 833,115, is the slow minimum and the fast maximum. The chip
mean, 534,265 ops per bin, sits between the groups because 36 blocks are
in one and 10 are in the other. Inside each group the spread is a few
percent.

Blocks 46–83 adding nothing is the hold at 46. The gate did what it was
told.

The rank did not move. A block that is slow at the start of the half
minute is slow at the end. A fight for the mapped atomic that rotated
among blocks would have washed out over 30 seconds. It did not.

The sum of the blocks still matches the pulse counter at 100 ms: ratio
1.000, drift 0.01%, correlation 0.995. The two groups are real finished
passes. The pulse word is not inventing the step.

## What this does to the four candidates

The tally slot is `blockIdx.x`. The launch comment says one block per
SM. The programming model does not promise that, so a `%smid` sample is
still the check that block 36 is a different SM from block 35.

`%nsmid` is the number of SMs on the device. This run already knows that
number: 84. It does not say which group a block landed in.

This 5080 is 7 groups of 12 SMs. Thirty-six is three full groups. A hold
at 46 fills those three and puts 10 working blocks into a fourth. The
step in the log is exactly there, and inside each side the rates match.
That is the candidate the log already supports: the working set is the
low block indices, and on this chip those indices line up with groups of
12.

A second block sharing an SM would show up as pairs, scattered or
adjacent. The fast blocks are the solid run 36 through 45.

A path-length gradient, some SMs closer to the PCIe interface than
others, would smear. Both groups are flat.

Idle blocks poll the mapped control words, and 38 of them were polling
during this hold. That traffic is still worth a run with the poll quiet.
It does not explain this step. The fast blocks are the ones whose indices
sit next to the first idle blocks.

Block 0 is in the slow group, at about 0.83 of the chip mean. The earlier
one-block tally, which reported a share of 0.75 for block 0, fenced that
block alone. Block 0 is a member of the slow group, not a separate fault.

The 1 ms correlation of 0.663 is the publish grid. The device posts on
the pass that crosses 1 ms, about every 1.2 ms, and about one host bin in
six is empty. At 20 ms the sum already lines up with the pulses
(correlation 0.954). The per-block means above are the persistence test.
They do not need a finer bin.

## What stays put

Kp is 0.55. Ki is 0.08. The deadband is 0.08. The integral clamp stays
the scaled one, about 3.4 SMs here. The tally stays out of the PI. The
command stays one count for the whole chip. The version string stays
1.0.0-gold.

The closed loop on this card sits near 46 to 48 SMs, which is the same
edge this hold landed on. That is a reason to measure the edge. It is not
a reason to retune.

## Next measurement, when it is asked for

1. Read `%smid` once per block at launch and write the block-to-SM map.
   One sample is enough while the block stays resident.
2. Repeat the open-loop tally at 36, at 46, and at 48. If 36 and 48 are
   flat, and 46 still steps, the partial last group is the plant effect.
3. One hold at 46 with the idle blocks not reading the mapped words, so
   the poll can be separated from the group boundary.

No gain change sits between this note and those three runs.
