# Why the fourth pass is faster

The fast SMs are the whole of GPC 3. The 84-SM hold decided it. Every SM was working, so nothing was idle. GPC 3 ran at 752k ops per host bin. The other six GPCs matched each other at 354k–358k, a 2.11× step at the edge of GPC 3 (block 47 to block 48). Those SMs are physical SMs 6 and 7, then the same pair 14 apart, which the firmware table names GPC 3. One pass of GPC 3 alone was flat at 801k. One pass of GPC 0 alone was flat at 521k. SMs 0–11, one pair from each of six GPCs, stepped 1.91× on the GPC 3 pair, with working SMs on both sides of it. Idle polling widens the gap when some SMs are idle. It is a second measured effect. Quieting it shrank the 46-SM step from 1.85× to 1.27× and the 48-SM step from 1.74× to 1.12×. The full-chip step stayed.

All of the numbers below were measured on this RTX 5080 on 12 October 2026, driver 595.99.02, CUDA 12.8. Nothing here is modelled. Kp stayed 0.55, Ki stayed 0.08, the deadband stayed 0.08, and the scaled integral clamp stayed 3.36 SMs. The tally stayed out of the PI. GPU temperature during the holds stayed between 39 °C and 53 °C.

Logs: `logs/2026-10-13_fourth-pass/`. The three checks, with both new switches unset and the quiet switch unset, passed. Gold window mean Z was 1.454 (free-run 584k pulses/s, closed 401k). Transfer mean Z was 1.447 (free-run 8.12 GFLOP/s, closed 5.60). Episode mean Z was 1.488 (free-run 22.66 GFLOP/s, closed 15.53).

## The map

A standalone program, `tools/sm_gpc_map.c`, opened an RM client on `/dev/nvidiactl` and read the firmware tables. It is not linked into the Gold binary. Driver version check returned `595.99.02`. All four controls returned NV_OK: GPC mask `0x7f` (7 GPCs), TPC mask `0x3f` on each GPC (6 TPCs), 84 SMs, 42 TPCs, two SMs per TPC.

Launch pass `(smid % 14) / 2` is the GPC id. Pass 0 is GPC 0, pass 1 is GPC 1, and so on through pass 6. Each 12-SM pass is one whole GPC, and the passes sit on different GPCs. SM ids 14 apart stay in that same GPC. Their TPC ids along that walk are 0, 2, 4, 1, 3, 5, so they are not consecutive TPC numbers. On GPC 0 the pairs are SMs 0/1 on TPC 0, 14/15 on TPC 2, 28/29 on TPC 4, 42/43 on TPC 1, 56/57 on TPC 3, and 70/71 on TPC 5. The same TPC order repeats on every GPC.

SMs 0–11 are TPC 0 of GPCs 0 through 5. They are six GPCs, not one packed group.

All 23 holds recorded the same `%smid` map. 84 physical SMs, one block each. Block index is not the SM id. Blocks 0–11 are GPC 0, 12–23 are GPC 1, 24–35 are GPC 2, 36–47 are GPC 3, 48–59 are GPC 4, 60–71 are GPC 5, and 72–83 are GPC 6.

| Block | Physical SM | GPC | TPC |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 |
| 2 | 14 | 0 | 2 |
| 3 | 15 | 0 | 2 |
| 4 | 28 | 0 | 4 |
| 5 | 29 | 0 | 4 |
| 6 | 42 | 0 | 1 |
| 7 | 43 | 0 | 1 |
| 8 | 56 | 0 | 3 |
| 9 | 57 | 0 | 3 |
| 10 | 70 | 0 | 5 |
| 11 | 71 | 0 | 5 |
| 12 | 2 | 1 | 0 |
| 13 | 3 | 1 | 0 |
| 14 | 16 | 1 | 2 |
| 15 | 17 | 1 | 2 |
| 16 | 30 | 1 | 4 |
| 17 | 31 | 1 | 4 |
| 18 | 44 | 1 | 1 |
| 19 | 45 | 1 | 1 |
| 20 | 58 | 1 | 3 |
| 21 | 59 | 1 | 3 |
| 22 | 72 | 1 | 5 |
| 23 | 73 | 1 | 5 |
| 24 | 4 | 2 | 0 |
| 25 | 5 | 2 | 0 |
| 26 | 18 | 2 | 2 |
| 27 | 19 | 2 | 2 |
| 28 | 32 | 2 | 4 |
| 29 | 33 | 2 | 4 |
| 30 | 46 | 2 | 1 |
| 31 | 47 | 2 | 1 |
| 32 | 60 | 2 | 3 |
| 33 | 61 | 2 | 3 |
| 34 | 74 | 2 | 5 |
| 35 | 75 | 2 | 5 |
| 36 | 6 | 3 | 0 |
| 37 | 7 | 3 | 0 |
| 38 | 20 | 3 | 2 |
| 39 | 21 | 3 | 2 |
| 40 | 34 | 3 | 4 |
| 41 | 35 | 3 | 4 |
| 42 | 48 | 3 | 1 |
| 43 | 49 | 3 | 1 |
| 44 | 62 | 3 | 3 |
| 45 | 63 | 3 | 3 |
| 46 | 76 | 3 | 5 |
| 47 | 77 | 3 | 5 |
| 48 | 8 | 4 | 0 |
| 49 | 9 | 4 | 0 |
| 50 | 22 | 4 | 2 |
| 51 | 23 | 4 | 2 |
| 52 | 36 | 4 | 4 |
| 53 | 37 | 4 | 4 |
| 54 | 50 | 4 | 1 |
| 55 | 51 | 4 | 1 |
| 56 | 64 | 4 | 3 |
| 57 | 65 | 4 | 3 |
| 58 | 78 | 4 | 5 |
| 59 | 79 | 4 | 5 |
| 60 | 10 | 5 | 0 |
| 61 | 11 | 5 | 0 |
| 62 | 24 | 5 | 2 |
| 63 | 25 | 5 | 2 |
| 64 | 38 | 5 | 4 |
| 65 | 39 | 5 | 4 |
| 66 | 52 | 5 | 1 |
| 67 | 53 | 5 | 1 |
| 68 | 66 | 5 | 3 |
| 69 | 67 | 5 | 3 |
| 70 | 80 | 5 | 5 |
| 71 | 81 | 5 | 5 |
| 72 | 12 | 6 | 0 |
| 73 | 13 | 6 | 0 |
| 74 | 26 | 6 | 2 |
| 75 | 27 | 6 | 2 |
| 76 | 40 | 6 | 4 |
| 77 | 41 | 6 | 4 |
| 78 | 54 | 6 | 1 |
| 79 | 55 | 6 | 1 |
| 80 | 68 | 6 | 3 |
| 81 | 69 | 6 | 3 |
| 82 | 82 | 6 | 5 |
| 83 | 83 | 6 | 5 |

## What was added

`GPUTRONIC_ACTIVE_PASSES` and `GPUTRONIC_ACTIVE_SMS` are test switches, off unless set. After calibration the host sets bit 4 and that list replaces the block-index gate. A pass is `(smid % 14) / 2`. Setting both variables at once is refused. The checks left both unset. `GPUTRONIC_IDLE_QUIET` is unchanged and stayed off for the checks.

Levels moved between holds that have the same shape. The two 84-SM holds differ by about 8%, and the quiet path does not run when nothing is idle. The comparison that counts is the ratio inside one hold. Absolute ops per bin are the measured level of that hold.

Every hold is open loop for 30 s, tally on for every block, 1 ms host bins, first second dropped. The 100 ms tally-to-pulse ratio was 1.000 on every hold (0.9998 on the GPC 3 pass alone). Two very flat holds, passes 0, 1, 2 and passes 4, 5, 6, had 100 ms correlation 0.887 and 0.895. Their ratios were 1.000. The counters match.

## Experiment 1

Block-index gate. Pass means are ops per host bin for the working SMs of that GPC. A blank pass was not in the working set. At 46 SMs, GPC 3 has 10 working SMs. Everywhere else a listed pass has 12.

| Hold | GPC 0 | GPC 1 | GPC 2 | GPC 3 | GPC 4 | GPC 5 | GPC 6 | CV | Largest block step | Chip mean | Pulses/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 36 | 432k | 434k | 438k | | | | | 0.8% | none at 1.2× | 435k | 239k |
| 36 quiet | 827k | 826k | 827k | | | | | 0.3% | none at 1.2× | 827k | 454k |
| 46 | 426k | 426k | 429k | 797k | | | | 30.1% | 1.85× at block 35→36 | 507k | 356k |
| 46 quiet | 679k | 679k | 680k | 868k | | | | 10.9% | 1.27× at block 35→36 | 720k | 505k |
| 48 | 429k | 430k | 432k | 760k | | | | 27.8% | 1.74× at block 35→36 | 513k | 375k |
| 48 quiet | 677k | 677k | 679k | 759k | | | | 5.1% | none at 1.2× (GPC 3 is 1.12×) | 698k | 511k |
| 60 | 376k | 450k | 452k | 795k | 381k | | | 31.7% | 2.10× at block 47→48 | 491k | 449k |
| 60 quiet | 371k | 705k | 703k | 786k | 375k | | | 30.3% | 2.11× at block 47→48 | 588k | 538k |
| 84 | 354k | 356k | 358k | 752k | 358k | 355k | 355k | 33.6% | 2.11× at block 47→48 | 412k | 528k |
| 84 quiet | 380k | 382k | 382k | 810k | 381k | 382k | 381k | 33.9% | 2.13× at block 47→48 | 442k | 567k |

Inside each pass the 12 SMs match. The step is between GPCs.

The 48-SM step drops further than the 46-SM step when idle blocks wait 1 ms between mapped reads. At 48 with quiet, GPC 3 is still the high side, by 1.12×. At 46 with quiet the 10 working SMs of GPC 3 stay 1.27× the others.

At 60 SMs, pass 4 is not the fast one. GPC 3 is. GPC 4 sits with GPC 0. Quieting the idle blocks (GPCs 5 and 6) raised GPCs 1 and 2 from about 450k to about 704k. GPC 0 and GPC 4 stayed near 373k, and GPC 3 stayed near 790k. These runs do not say why the quiet switch picked out GPCs 1 and 2.

At 84 SMs the step is still there, and it is larger. The quiet-labelled rerun has the same shape. With no idle block, that switch does not take the quiet path.

## Experiment 2

Three passes, 36 working SMs, chosen by `%smid` rather than by block index. The grid stays resident at 84 blocks. The question on each row is the one the directive asked.

| Set | Quiet | GPC means | CV | Largest block step | Chip mean | Pulses/s |
|---|---|---|---|---|---|---|
| 0, 1, 2 | off | 492k, 492k, 535k | 4.0% | none at 1.2× (GPC 2 is 1.09×) | 507k | 278k |
| 0, 1, 2 | on | 989k, 988k, 990k | 0.1% | none at 1.2× | 989k | 543k |
| 0, 1, 3 | off | 484k, 476k, 838k | 28.2% | 1.76× at block 23→36 | 599k | 329k |
| 0, 1, 3 | on | 962k, 962k, 994k | 1.6% | none at 1.2× (GPC 3 is 1.03×) | 973k | 534k |
| 0, 2, 4 | off | 411k, 496k, 411k | 9.1% | 1.21× | 439k | 241k |
| 0, 2, 4 | on | 866k, 1038k, 866k | 8.8% | GPC 2 is 1.20× the other two | 923k | 507k |
| 1, 2, 3 | off | 458k, 450k, 792k | 28.1% | 1.76× at block 35→36 | 566k | 311k |
| 1, 2, 3 | on | 971k, 972k, 1001k | 1.4% | none at 1.2× (GPC 3 is 1.03×) | 981k | 539k |
| 4, 5, 6 | off | 479k, 480k, 479k | 0.1% | none at 1.2× | 479k | 263k |
| 4, 5, 6 | on | 977k, 976k, 976k | 0.0% | none at 1.2× | 976k | 536k |

Passes 0, 1, 2 are flat, which is what the 36-SM hold did. The pass-filter hold sat higher than the block-index hold. The shape is the match. With quiet, the three passes agree to 0.1%.

Passes 0, 1, 3: GPC 3 is still the fast one, 1.76×, with a working neighbour on one side only in GPC-index order. Quiet brings GPC 0 and GPC 1 up to it. The leftover is 1.03×.

Passes 0, 2, 4 leave GPC 3 idle, so no working pair has a working GPC neighbour in index order. They are not all fast, and they are not flat at the GPC 3 rate. GPC 2, one neighbour of the idle GPC 3, runs at 1.20× GPCs 0 and 4, and that ratio survives the quiet switch. GPC 4, the neighbour on the other side, stays with GPC 0.

Passes 1, 2, 3 leave GPC 0 idle. Pass 1 does not become the fast one. Pass 3 does, at 1.76×. Quiet again leaves GPC 3 about 1.03× the other two.

Passes 4, 5, 6 are flat. That region of the chip matches itself, at the ordinary rate, not at the GPC 3 rate.

## Experiment 3

Twelve SMs. Quiet stays off. A pass cannot name SMs 0–11, so that hold uses `GPUTRONIC_ACTIVE_SMS=0-11`.

| Hold | What worked | Ops per host bin | CV | Largest step | Chip mean | Pulses/s |
|---|---|---|---|---|---|---|
| Pass 0 | GPC 0, 12 SMs, 14 apart | 521k, the twelve agree | 0.0% | none at 1.2× | 521k | 95k |
| Pass 3 | GPC 3, 12 SMs, 14 apart | 801k, the twelve agree | 0.0% | none at 1.2× | 801k | 147k |
| SMs 0–11 | TPC 0 of GPCs 0–5, two SMs each | GPC 3 pair 1091k; the other pairs 572k–598k | 28.2% | 1.91× between SM 7 (1091k) and SM 8 (572k) | 669k | 123k |

Pass 3 alone is 1.54× pass 0 alone. Both passes are flat, so the 14-apart spacing is an even layout inside one GPC. The gap between the two GPCs is still there when each runs alone. SMs 0–11 put the GPC 3 pair between working pairs of GPC 2 and GPC 4. That pair is the fast one, at 1.91× its neighbour.

## What stays put

Kp is 0.55. Ki is 0.08. The deadband is 0.08. The integral clamp stays the scaled one. The tally stays out of the PI. The command stays one count for the chip. The version string stays 1.0.0-gold. The new switches stay off unless a test sets them.
