# GPUTronic Gold: expected pulses/s vs SM count, consumer GeForce GTX 10 through RTX 50

File: docs/analysis/pulse-rate-table.csv (generator script not in the repo)
This is a model, not measurements, except the RTX 5080 ANCHOR row.


Specs source (every card row): Wikipedia, "List of Nvidia graphics processing units", desktop
GeForce 10/20/30/40/50 tables, fetched 2026-10-12 PT:
https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units
- SM count = 4th field of "Core Config". Base/boost MHz and FP32 GFLOPS/TFLOPS are copied as published:
  GFLOPS for the 10 series, TFLOPS for the 20-50 series, "base (boost)".
- TechPowerUp's GPU database blocked automated fetch (bot-check page), so it isn't cited. Cards
  with blank fields had none on the source. Nothing was filled in by hand.
- Dies present in the source's desktop tables: GP102/104/106/107, TU102/104/106, GA102/103/104/
  106/107, AD102/103/104/106/107, GB202/203/205/206/207. No GB204 appears: RTX 5070 Ti and 5080 are
  GB203, 5070 is GB205. GTX 16-series (TU116/117) is out of scope.

## Pulse model

- Gold kernel: one block per SM, 64 threads = 2 warps. Each warp leader pulses once per
  GOLD_BATCH x GOLD_WORK_ITERS = 128 work calls x 8 dependent FMAs = 1024 FMAs, so 2 pulses per SM per
  pass.
- Measured anchor (RTX 5080): 582,430 pps at 84 SMs = 6,934 pps/SM = about 288 us per pass. A
  1024-long dependent FMA chain is about 4,096 cycles, about 1.6 us at 2.6 GHz, so math is under 1%
  of the period. The kernel isn't issue-bound at all. The period is set by mapped (PCIe)
  atomics and fences, as the repo docs say, and rate is near-linear in SMs (R² 0.98).
- Two model_kind rows per card:
  atomic_bound (preferred): pps(n) = measured 5080 curve at n (linear interpolation of the
    measured points 4,8,21,28,42,56,63,84). It's card-independent. pps@full above 84 SMs is
    linear extrapolation per SM (no measurement; contention may make it lower).
  clock_scaled (as requested; upper-sensitivity): atomic_bound x boost/2617 MHz x issue factor
    (Pascal 0.67 for ~6-cycle vs ~4-cycle dependent FMA latency; 1.0 for Turing+). Only
    meaningful if the math part dominated, which the anchor says it doesn't.
- Both ignore PCIe generation and host latency (GTX 10 / RTX 20 are PCIe 3, RTX 30 is PCIe 4,
  RTX 50 is PCIe 5; 5060/5050 are x8). Those likely matter more than clocks.
- GTX 1080 can't be anchored: the v0.1 archive dyno (archive/v0.1-gtx1080-2026-01/gputronic_master_dyno.csv)
  is a different kernel (tanf/sqrtf, 128-thread blocks, clock64 throttle, block-count gearshift,
  "RPM" units).
- Pascal can't run Gold as-is (no __nanosleep on sm_61; see gputronic-1080-bringup.txt).

## Where the table could replace runtime calibration

- Seed rate_ref for the first window, so the 0.7-1.8 s free-run calibration could be shortened.
- Seed n_ff (SM count for target Z) and sms_per_u before the probe table finishes. Z(n) is a ratio,
  so only the curve shape matters, and that's the same in both models.
- Sanity alarm: if the measured rate_ref is more than about 2x off the table, flag the mapping or the
  kernel.

## Why runtime calibration is still needed

- Absolute rate depends on PCIe gen/width, host chipset, driver, boost clock, power/thermal state
  (the D axis is left to the driver), and the background load.
- Kernel shape: Transfer and Episode free-run rates differed 7-9% between runs of the same
  kernel. Their Z(n) curves are nonlinear and different from Gold (Episode is flat above ~42 SMs:
  Z=1.5 at ~23 SMs vs Gold ~46).
- A ±5-10% reference error shifts the settled Z by roughly the same fraction (see the GTX 1080
  model; script not in the repo).
