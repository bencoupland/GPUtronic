# GPUTronic Transfer Plant

Second plant on the Gold 1.0.0 interface.

```bash
make transfer
./build/gputronic_transfer check
# === TRANSFER GATE: dyno=PASS closedloop=PASS ===
```

Uses stock Gold (`launch_selftest_kernel = 0`, auto `sleep_scale`). No transfer-side gain override.

## Plant

- Persistent kernel, 84×64
- Real FP32 GEMM microtiles: 8×8×8 × 32 = 32768 FLOP per tach pulse
- `atomicAdd` on Gold `total_work_pulses`
- Honours `throttle_sleep_ns`

## Release numbers (RTX 5080)

Open-loop: 9.03 → 5.15 GFLOP/s at 500 µs, coupled to tile rate. Free-run Z≈1.02.

Closed-loop (Gold auto-scale): 8.77 → 6.55 GFLOP/s (−25%), Z_mean≈1.34 vs target 1.50.

That is the transfer test: measure and control Z on something that is not the Gold synthetic kernel, and move a counted FLOP metric in the predicted direction.

## Files

- `src/gputronic_transfer.cu`
- `results/gputronic_transfer_*_report.txt`
