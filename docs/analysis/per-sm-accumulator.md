# T11: per-SM ops accumulator (prototype, not yet run)

## Design
- Each Gold block (one per SM) tallies completed work in shared memory, not with per-pulse global atomics.
  Ops per pass, from the code (FMA = 2 FP32 ops):
  - CUDA-core warp: `GOLD_BATCH` 16 × `GOLD_WORK_ITERS` 8 × 8 FMAs × 2 × 32 lanes = **65,536 ops**
    (two CUDA warps per block at the default 64 threads → 131,072 per block pass).
  - Tensor warp: `GOLD_TENSOR_STEPS` 64 × 2·16·16·16 = **524,288 ops** (0 below sm_70, where wmma is stubbed).
- Every bin (`GPUTRONIC_TRACE_BIN_US`, default 1000 µs when the accumulator is on) the block leader stores one
  32-byte slot `{ops_cuda, ops_tensor, t_ns, seq}` (cumulative 64-bit, plain stores, `seq` last after a
  system fence) to a **mapped host array**. Host reads retry if `seq` changes during the read.
- Timing: `%globaltimer` (nanoseconds, wall-clock). `clock64` counts SM cycles and the SM clock moves with
  boost, so a cycle count is not a fixed time. Globaltimer resolution on some parts is ~1 µs, fine for 1 ms.
- Mapped host array rather than a device buffer copied by the host: the device's write is a one-way posted
  PCIe write of 32 B per SM per bin; a periodic `cudaMemcpyAsync` moves the same bytes plus a DMA setup and
  a host round trip every millisecond. The host reads its own memory locally.
- Opt-in: `GPUTRONIC_SM_ACCUM=1` (all SMs) or `GPUTRONIC_SM_ACCUM_SM=<id>` (only that SM reports; start here).
  Default off: the kernel tests a null pointer argument (a register) per pass; no extra mapped reads or writes.
  Idle blocks (past `active_sms`) do not write; their slot simply stops advancing.
- Host output (trace modes: `noise`, `loadfile`, `loadstep`, `bursts`, `fullwidth`):
  `results/gputronic_gold_smaccum.csv` with `t_ms,sm,ops_cuda,ops_tensor,d_ops_cuda,d_ops_tensor,dev_t_ns,seq,
  active_sms,bin_pulses,rate_ema`. All-SM mode writes 84 rows per ms (about 2.5 M rows in 30 s, ~150 MB).
- Analyzer: `scripts/analyze_smaccum.py` compares per-bin accumulator ops with `bin_pulses × 65,536`
  (single SM scaled by `active_sms`): correlation, ratio mean and drift, best lag; at 1 and 10 bins.
  **Pass:** drift < 1% and correlation ≥ 0.9 at 10 bins.

## Observation rate and PCIe cost
- One value per SM per 1 ms = 84 × 1 kHz = **84,000 samples/s** aggregate.
- "84 streams × 20 kHz = 1.68 MHz" would need a **50 µs** write period per SM: 20× the writes.
- PCIe cost at 1 ms: 84 × 32 B/ms ≈ **2.7 MB/s** of posted writes (8–16 B packed would be 0.7–1.3 MB/s).
  For comparison, today every warp pass already does a mapped atomic (a round-trip PCIe transaction):
  roughly 400–550k atomics/s on Gold.
- Trade-off: a shorter bin gives faster observation but more writes and noisier per-bin counts (fewer
  passes per bin: ~3 passes per SM per ms at ~300 µs/pass). A longer bin is quieter and slower.
