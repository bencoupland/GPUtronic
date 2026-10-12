# Traces from the 2026-10-12 RTX 5080 run

The numbers in `docs/analysis/5080-2026-10-12.md` come from these files.
The text logs next to this note are the program's own printout. Two more
forms of the same runs are here so a later session can replot them.

## 10 ms bins

`traces_10ms/*.csv`. One row is 10 ms. Columns:

| Column | Meaning |
|---|---|
| `t_s` | Middle of the bin, seconds from `gputronic_start` |
| `n` | `active_sms` of the last 10 µs sample in the bin |
| `z_mean`, `z_min`, `z_max` | `z_hat` over the bin |
| `z_gt_1_6` | How many 10 µs samples in the bin had `z_hat` > 1.6 |
| `rate`, `rate_t` | Mean `rate_ema` and `rate_tensor_ema`, pulses/s |
| `z_cuda`, `z_tensor` | Mean of the two sensor Z values |
| `integral` | Mean of the PI integral |
| `sensor` | 0 = CUDA-core, 1 = tensor. Last sample in the bin |
| `samples` | 10 µs rows in the bin. A full bin is about 1000 |

`T2_step_1ms.csv` is the open-loop step at 1 ms, with `z_min` and `z_max`,
so the 84-to-42 transition can be read without the 10 µs file. The command
lands at t = 5.001 s. Eight milliseconds later the rate is already down.
Sixteen milliseconds later it is still falling.

`T3_*_closedloop.csv` in this directory is the first pass, the one whose
free-run sample fell inside the table and failed the gate. `T3_*_rerun_closedloop.csv`
is the pass after the 4.5 s wait. The printed reports for both are the
`T3_*rerun*.txt` files beside this note.

## Raw 10 µs samples

Three archives, each under 50 MB. `zstd -d` then `tar -tf`.

| Archive | What is in it |
|---|---|
| `raw_gold_T1_T2.tar.zst` | Dyno off/on, the 40 s Gold run, the closed-loop check, the step, tensor-off, and the Z = 1.0 release |
| `raw_gold_T5.tar.zst` | Reserve runs k = 0, 1, 2, 4 |
| `raw_plants_T3.tar.zst` | First-pass transfer and episode closed loops, plus the reruns named `T3_*_rerun_closedloop.csv` |

Gold columns, written by the controller at about 10 µs:

```
time_s,z_raw,z_hat,dzdt,rate_ema,active_sms,error,integral,ctrl_hz,rate_ref,z_cuda,z_tensor,rate_tensor_ema,rate_ref_tensor,ctrl_sensor
```

The 10 ms files were built from those columns. `z_gt_1_6` is the count that
gave the 32 ms streak above 1.6 on the 40 s Gold run. Sum that column and
multiply by 10 µs.

The copies that stayed only on the 5080 were the uncompressed CSVs, about
930 MB together. They are not required once these archives are here.
