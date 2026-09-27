# GPUTronic 1.0 Gold

Reference closed-loop control for a GPU execution path.

The specification of record is:

`docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md` (GT-FF-GOLD-1.0)

That document is the function frame: signals, functions F-01…F-14, calibration parameters, measured plant, and a guided walk through the source. This file is the short form.

Unpublished until this tree. Transfer-plant lessons are folded in: this *is* 1.0, not a 1.1 tag.

## Why Gold exists

Late v14–v26 forks lost the pulse counter (the last SM overwrote it, and Z stuck at a floor or a ceiling). Gold keeps a few rules:

1. **Aggregate Q** — `atomicAdd` on `total_work_pulses`
2. **Mapped zero-copy** — host_ctrl == device_ctrl on this 5080
3. **Sustained Z cal** — `Z = rate_ref / rate` from a 1.1 s free-run *window*, not peak EMA
4. **Windowed rate** — ~8 ms pulse Δ, not a 10 µs EMA that cannot see a 300 µs plant
5. **SM count** — `active_sms` is how many SMs run the work. The others skip it and do not count a pulse. At least 4 stay on. After free-run, one test at half the SMs sets how strongly the error changes that count. Kp and Ki are unchanged. Sleep duration is not the control. The previous source is in `archive/gold-1.0.0-nanosleep/`.
6. **One-sided PI** — if Z is already at the target, every SM runs. If Z is only a little short, the count stays where it is.
7. **CSV evidence** — every run writes `results/gputronic_gold_*.csv`

## Gate (must stay green)

```bash
make gold && ./build/gputronic_gold check
# === GOLD GATE: dyno=PASS closedloop=PASS ===

make transfer && ./build/gputronic_transfer check
# === TRANSFER GATE: dyno=PASS closedloop=PASS ===
```

## Measured on RTX 5080 / CUDA 12.8 (release run)

**Gold self-test plant**

- Dyno: free-run ~499 kpulse/s, Z≈1.00; 500 µs sleep → ~314 kpulse/s, Z≈1.58; R²≈0.88
- Closed-loop, 2026-09-23, SM count: about 582k → 426k pulse/s, Z mean ≈ 1.38
- Older sleep-duration run: 445k → 324k pulse/s, Z mean ≈ 1.37, sleep ≈ 415 µs

**Transfer plant (cooperative 8×8×8 GEMM tiles, 32768 FLOP/pulse)**

- Closed-loop, 2026-09-23: 7.25 → 5.39 GFLOP/s, Z mean ≈ 1.36
- Older sleep-duration run: 9.03 → 5.15 GFLOP/s at 500 µs; closed loop 8.77 → 6.55 GFLOP/s, Z mean ≈ 1.34

## Files

- `include/gputronic.h` — public C API
- `src/gputronic_gold.cu` — core + CLI
- `src/gputronic_transfer.cu` — second plant
- `examples/gold_demo.c` — embed demo
- `build/gputronic_gold`, `build/libgputronic_gold.a`, `build/gputronic_transfer`

## Plant note

Mapped atomics set a ~300 µs loop on this 5080 when every SM runs. Sleep duration did little below ~200 µs and then jumped. The control is now how many SMs run. After free-run, one half-SM measurement sets how strongly the error changes that count. Do not change Kp or Ki to hide a control that cannot move the rate smoothly.

## Do not

- Reintroduce per-SM last-writer `total_work_pulses = sm_counters[sm]`
- Put NVML thermal in the fast loop
- Use peak-EMA `rate_ref` (it parks free-run Z ≥ target; one-sided PI never engages)
- Claim this controls a program that does not add pulses or read `active_sms`
