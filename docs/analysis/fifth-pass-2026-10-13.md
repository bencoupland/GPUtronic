# Why GPC 3 is faster

The path that favours GPC 3 is the per-pass read of the mapped control words. The run that decided it is `e1-n84-cached`. On the 84-SM hold with the mapped pulse atomic still once a pass, GPC 3 ran at 781k ops per host bin and the other six GPCs ran at 364k–372k, a 2.14× step. Batching that atomic to once a block per millisecond, on `e1-n84-local`, raised every GPC and left the step at 2.07× (938k against 453k–462k). Caching the control words as well, one read per block per millisecond, removed the step. The same 84 SMs then ran at 196M–203M ops per host bin. GPC 3 was 1.03× GPC 0. The largest gap between any two SMs was 1.05×. The pulse rate on that hold was 255 million pulses/s, against 673 thousand on the batched-atomic hold and 547 thousand with both switches off. Chip clocks at the end of those three holds were the same, about 2890 MHz.

The single-GPC bench did not reproduce the 2× step. Compute, device atomics, device streaming, mapped reads, and mapped atomics each stayed inside about 11% across the seven GPCs, and the mapped-atomic row was fastest on GPC 1 and GPC 2. Those tests run one GPC at a time. The 2× step shows up when every GPC reads the same mapped control words on every pass.

All of the numbers below were measured on this RTX 5080 on 12 October 2026, driver 595.99.02, CUDA 12.8. Nothing here is modelled. Kp stayed 0.55, Ki stayed 0.08, the deadband stayed 0.08, and the scaled integral clamp stayed 3.36 SMs. The tally stayed out of the PI. The version string stayed 1.0.0-gold. GPU temperature during the holds stayed between 40 °C and 55 °C.

Logs: `logs/2026-10-13_fifth-pass/`. The three checks, with both new switches unset and the quiet switch unset, passed on this binary.

```
[CHECK] pulses=8576613 z=1.530 z_mean=1.472 rate=339994.1 n=54
[CHECK] free-run 516060.9 pulse/s | closed 358161.3 pulse/s
[CHECK] pulses=4871055 z=1.507 z_mean=1.463 target=1.50 ema=156990.5 sleep=5
[CHECK] free-run 234320.4 pulse/s (7.6782 GFLOP/s) | closed 161125.1 pulse/s (5.2797 GFLOP/s)
[CHECK] pulses=12301048 z=1.438 z_mean=1.464 target=1.50 ema=412411.9 sleep=5
[CHECK] free-run 597498.5 pulse/s (19.5788 GFLOP/s) | closed 401168.5 pulse/s (13.1455 GFLOP/s)
```

Gold, transfer, and episode each printed `GATE: dyno=PASS closedloop=PASS`.

## Experiment 1

Open loop, 30 s, tally on, quiet off. The first two rows launch all 84 SMs. The other rows launch 84 blocks and let one GPC work. `GPUTRONIC_PULSE_LOCAL=1` counts pulses in a register and adds the batch to `total_work_pulses` about once a millisecond. The control words are still read every pass on those rows. Ops per host bin are the per-GPC means. The 100 ms column is the tally-to-pulse ratio.

| Hold | Local pulse | GPC 0 | GPC 1 | GPC 2 | GPC 3 | GPC 4 | GPC 5 | GPC 6 | CV | Largest step | Pulses/s | 100 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-n84 | off | 364k | 367k | 371k | 781k | 372k | 366k | 365k | 33.9% | 2.12× at block 47→48 | 547k | 1.0000 |
| e1-n84-local | on | 453k | 456k | 460k | 938k | 462k | 455k | 454k | 32.1% | 2.05× at block 47→48 | 673k | 1.0000 |
| e1-gpc0 | off | 531k | | | | | | | 0.4% | none at 1.2× | 97k | 1.0000 |
| e1-gpc0-local | on | 629k | | | | | | | 0.4% | none at 1.2× | 115k | 1.0000 |
| e1-gpc3 | off | | | | 670k | | | | 0.5% | none at 1.2× | 123k | 1.0000 |
| e1-gpc3-local | on | | | | 980k | | | | 0.3% | none at 1.2× | 179k | 1.0000 |
| e1-n84-cached | on, and control words cached | 197.2M | 198.3M | 200.0M | 202.9M | 200.6M | 198.1M | 196.5M | 1.2% | none at 1.2× (max/min 1.05×) | 255M | 1.0000 |

GPC 3 alone was 1.26× GPC 0 alone with the switches off (670k against 531k), and 1.56× with the local pulse count (980k against 629k). Inside each of those holds the twelve SMs matched. The full-chip step is the one that stayed near 2× until the control words were cached.

The ratio is the analyzer's usual one. CUDA ops in the bucket, divided by `bin_pulses` times 65,536. `bin_pulses` is the host's read of `total_work_pulses`. With the local pulse count, that word moves once a block per millisecond, by the sum of the warps' register counts, instead of once a warp per pass. A 1 ms host bin then sometimes catches two of those batches and sometimes none. On `e1-n84-local` the 1 ms ratio mean was 1.63. On `e1-n84-cached` it was 3.09. Over 20 ms the cached hold was already 1.0003. Over 100 ms every hold in this table was 1.0000, correlation 0.981 to 0.996.

## Experiment 2

`tools/gpc_path_bench.cu` is a standalone kernel, not linked into the Gold binary. It launches one block per SM, 64 threads. Blocks whose `(smid % 14) / 2` is not the requested GPC return immediately. Each test runs for 400 ms of `%globaltimer` and reports nanoseconds per iteration on thread 0. Thread 0 is a lane that issues the atomic on the two atomic tests. The device-stream buffer is 256 MiB, four times the 64 MiB L2. Twelve SMs reported on every row. The bench printed `fails 0`.

Lower is faster. No row has a 2× gap.

| GPC | Compute | Device atomic | Device stream | Mapped read | Mapped atomic |
|---|---|---|---|---|---|
| 0 | 4316 | 101.7 | 436 | 116.9 | 16784 |
| 1 | 3965 | 91.1 | 387 | 105.4 | 15415 |
| 2 | 3893 | 91.0 | 381 | 105.4 | 15459 |
| 3 | 3893 | 91.0 | 380 | 105.4 | 16236 |
| 4 | 4341 | 100.6 | 421 | 116.4 | 16935 |
| 5 | 4308 | 101.6 | 427 | 116.7 | 16727 |
| 6 | 4350 | 100.8 | 432 | 116.3 | 17137 |

On compute, GPC 0 took 1.11× GPC 3. GPC 2 matched GPC 3. The same grouping, about 11%, shows up on the device atomic, the device stream, and the mapped read. The mapped atomic is fastest on GPC 1 (15415 ns) and GPC 2 (15459 ns). GPC 3 sits in the middle of that row at 16236 ns. GPCs 0, 4, 5, and 6 are the slow group on every test. That is a small, repeatable spread. It is not the full-chip 2× step.

## Experiment 3

Quiet on, both new switches off, 84 blocks launched, the mask chooses who works. Firmware TPC ids. Inside every set the SMs matched: CV 0.0% or 0.1%, max/min 1.000 to 1.002. The 100 ms ratio was 1.0000 on all ten holds.

| Set | SMs | GPC 0 | GPC 3 | GPC 3 / GPC 0 | GPC 0 pulses/s | GPC 3 pulses/s |
|---|---|---|---|---|---|---|
| 1 TPC, both SMs | 2 | 13.06M | 14.67M | 1.12 | 398k | 447k |
| One SM on each of two TPCs | 2 | 13.60M | 14.72M | 1.08 | 415k | 449k |
| TPCs 0, 1, 2 | 6 | 4.65M | 5.25M | 1.13 | 425k | 480k |
| TPCs 0, 2, 4 | 6 | 4.70M | 5.18M | 1.10 | 430k | 474k |
| All 6 TPCs | 12 | 2.49M | 2.63M | 1.06 | 455k | 480k |

On GPC 0 the split pair was 1.04× the pair that shares a TPC. On GPC 3 the two pairs matched, 1.003×. The spread set of six was 1.01× the clustered set on GPC 0 and 0.99× on GPC 3. Which SM, and whether two SMs share a TPC, does not move the rate by much. The per-SM rate falls as more SMs in that GPC work: twelve SMs run at about a fifth of the two-SM rate. GPC 3 stays ahead at every size, by 6% to 13%. That is the size of the single-GPC bench gap, not the full-chip 2× step.

## Experiment 4

Three repeats each, quiet on, new switches off. The 60-SM split repeats. Passes 0, 2, and 4 do not. Both are left as measured. No explanation is attached.

The 60-SM hold is the first 60 blocks, so GPCs 0 through 4 work and GPCs 5 and 6 are idle and quiet. On all three repeats GPCs 0 and 4 match each other, GPCs 1 and 2 match each other, and GPC 3 sits about 5% above GPCs 1 and 2. GPCs 1 and 2 run at about 1.8× GPCs 0 and 4.

| Repeat | GPC 0 | GPC 1 | GPC 2 | GPC 3 | GPC 4 | GPC 1 / GPC 0 | Pulses/s |
|---|---|---|---|---|---|---|---|
| e4-n60-a | 388k | 703k | 704k | 737k | 388k | 1.81 | 534k |
| e4-n60-b | 419k | 751k | 751k | 787k | 419k | 1.79 | 572k |
| e4-n60-c | 415k | 743k | 744k | 779k | 415k | 1.79 | 566k |

The fourth-pass quiet 60-SM hold had the same shape, GPCs 1 and 2 near 704k and GPCs 0 and 4 near 373k. The levels moved. The split did not.

Passes 0, 2, and 4, three repeats. GPC 2 was 1.002× GPC 0 on each one. The fourth-pass 1.20× did not show up. CV was 0.1% and max/min was 1.002. The 100 ms ratio was 1.0000.

| Repeat | GPC 0 | GPC 2 | GPC 4 | GPC 2 / GPC 0 | Pulses/s |
|---|---|---|---|---|---|
| e4-p024-a | 976k | 977k | 976k | 1.002 | 536k |
| e4-p024-b | 890k | 892k | 891k | 1.002 | 489k |
| e4-p024-c | 918k | 920k | 918k | 1.002 | 504k |

## What stays put

Kp is 0.55. Ki is 0.08. The deadband is 0.08. The integral clamp stays the scaled one, printed as 3.36 SMs on these runs. The tally stays out of the PI. The command stays one count for the chip. The version string stays 1.0.0-gold. `GPUTRONIC_PULSE_LOCAL` and `GPUTRONIC_CTRL_CACHE` stay off unless a test sets them.
