# GPUTronic 1.0 Gold

Reference closed-loop control for a GPU execution path.

The specification of record is:

`docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md` (GT-FF-GOLD-1.0)

That document is the function frame: signals, functions F-01…F-14, calibration parameters, measured plant, and a guided walk through the source. This file is the short form.

Unpublished until this tree. Transfer-plant lessons are folded in: this *is* 1.0, not a 1.1 tag.

## Why Gold exists

Late v14–v26 forks lost the tachometer contract (last-SM overwrite, Z stuck at floor/ceiling). Gold freezes the plant interface:

1. **Aggregate Q** — `atomicAdd` on `total_work_pulses`
2. **Mapped zero-copy** — host_ctrl == device_ctrl on this 5080
3. **Sustained Z cal** — `Z = rate_ref / rate` from a 1.1 s free-run *window*, not peak EMA
4. **Windowed rate** — ~8 ms pulse Δ, not a 10 µs EMA that cannot see a 300 µs plant
5. **Auto `sleep_scale`** — sized from measured loop period so PI commands reach the nanosleep cliff
6. **One-sided tracking PI** — sleep only when Z < target
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
- Closed-loop: free-run 445 kpulse/s → 324 kpulse/s (−27%) at target Z=1.50
- Z_mean ≈ 1.37, sleep ≈ 415 µs (in the authority band, not the 8 µs dead zone)

**Transfer plant (cooperative 8×8×8 GEMM tiles, 32768 FLOP/pulse)**

- Dyno: 9.03 → 5.15 GFLOP/s at 500 µs, coupled to tile rate; free-run Z≈1.02
- Closed-loop (stock Gold auto-scale, no override): 8.77 → 6.55 GFLOP/s (−25%)
- Z_mean ≈ 1.34 vs target 1.50

## Files

- `include/gputronic.h` — public C API
- `src/gputronic_gold.cu` — core + CLI
- `src/gputronic_transfer.cu` — second plant
- `examples/gold_demo.c` — embed demo
- `build/gputronic_gold`, `build/libgputronic_gold.a`, `build/gputronic_transfer`

## Plant note

Mapped atomics set a ~300 µs loop period on this 5080. `__nanosleep` is almost invisible below ~200 µs and bites near 500 µs. Gold therefore schedules `sleep_scale` so a typical error (~0.3) commands ~one loop period. Do not retune Kp/Ki to hide a dead-zone actuator.

## Do not

- Reintroduce per-SM last-writer `total_work_pulses = sm_counters[sm]`
- Put NVML thermal in the fast loop
- Use peak-EMA `rate_ref` (it parks free-run Z ≥ target; one-sided PI never engages)
- Claim FOC of uninstrumented third-party kernels — that is a different plant
