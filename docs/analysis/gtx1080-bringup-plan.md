# GPUTronic on a GTX 1080 (Pascal GP104, sm_61, 20 SMs): bring-up plan

Branch: controller-fixes (PR #1). Nothing here has been run on a GTX 1080. The January v0.1
(archive/v0.1-gtx1080-2026-01/main.cu) did run on a GTX 1080, but it used a
single mapped atomic counter, and kernel relaunch with more/fewer blocks, not this code.

## 1. Toolkit / driver

- CUDA 12.x still targets sm_61. Verified locally: nvcc 12.8 --list-gpu-arch includes compute_61.
- CUDA 13.0 removed offline compilation for pre-Turing (< CC 7.5), including Pascal. Use CUDA
  12.8 or 12.9 (12.9 is the last that targets Pascal).
- Driver: R580 is the last branch that supports Pascal (LTS into 2028). The RTX 5080 results in the
  repo used driver 595.91.07, which won't drive a 1080. Install a 580.xx driver.
- Host compiler: on glibc 2.41 (e.g. Debian 13) CUDA 12.8's crt/math_functions.h conflicts on
  sinpi/cospi noexcept. Use a distro where 12.8/12.9 is supported or patch locally.

## 2. Makefile

NVCC and ARCH are plain assignments, so a command-line override works:

```
make NVCC=/usr/local/cuda-12.9/bin/nvcc ARCH=-arch=sm_61 gold lib demo
```

Or make them `NVCC ?=` / `ARCH ?=` and add a fat binary option, e.g.
  -gencode arch=compute_61,code=sm_61 -gencode arch=compute_120,code=sm_120.

## 3. Code blockers on sm_61

- Actuation is by SM count, so no sleep primitive is part of the design for any kernel or card.
- The open code to-do for sm_61 is tracked in HANDOFF.md, next steps.
- Any legacy sleep-sweep numbers from transfer/episode are not comparable across cards.

## 4. Zero-copy / mapped memory on Pascal

- cudaHostAlloc(cudaHostAllocMapped) + cudaHostGetDevicePointer work on Pascal with UVA (64-bit
  Linux). Pointers are usually equal under UVA. gold_verify_zero_copy() checks type, host and device
  pointer, and a round trip, and aborts if any fail.
- Mapped atomics travel over PCIe 3.0 (1080) rather than Blackwell's PCIe 5 path, so expect a
  longer atomic round trip and a longer persistent-loop period than about 300 us. 64-bit atomicAdd
  on mapped host memory is supported (sm_60+). It's device-scope; host + device concurrent
  atomics need hostNativeAtomicSupported, which is typically 0 on x86 discrete. Don't use
  gputronic_pulse() while the kernel runs.
- __threadfence_system is supported.

## 5. SM floor and resolution

- 20 SMs, floor 4: the actuator range is 4..20 = 17 levels (vs 4..84 = 81 on the 5080).
- Scaling the 5080 Gold curve proportionally (assumption: shape scales by n/N), Z=1.5 sits near
  n ~ 11 of 20 and each SM step moves Z by about 0.1 near target, larger than the 0.015 deadband. The
  loop will limit-cycle between two counts. Expect Z oscillating roughly ±0.05.
- Options: widen deadband to ~0.06 on 20 SMs (gputronic_set_deadband), or dither between n and n+1
  over 8-16 windows (64-128 ms) to get about 0.006-0.012 effective Z resolution. Consider lowering
  the floor to 2 on 20 SMs (it's hard-coded to 4 in the kernel and the control thread).
- Probe counts in gold_build_table become 15, 10, 5, 4. The 5 and 4 points are close together, fine.

## 6. Verification order

 a) Device query: nvidia-smi; deviceQuery or cudaGetDeviceProperties: name, CC 6.1,
    multiProcessorCount 20, canMapHostMemory=1, unifiedAddressing=1, hostNativeAtomicSupported,
    clockRate, pciBusID.
 b) Build only: make gold ARCH=-arch=sm_61 (after the HANDOFF code to-do). Check warnings.
 c) Zero-copy: ./build/gputronic_gold run 5, confirm "[ZC ] type=host host_ptr=OK dev_ptr=OK
    round_trip=OK", pulses increasing, control_hz_ema near 100 kHz.
 d) Open loop: ./build/gputronic_gold dyno, then step. Rate must fall monotonically 20 -> 4 SMs.
    Record R². The step should show the rate change within one 8 ms window.
 e) Calibration table: check [CAL] lines (rate_ref, table n/rate/Z, n_ff, dZ/dn, sms_per_u).
 f) Closed loop: ./build/gputronic_gold check, then run 60. Look at mean Z vs 1.50, the active_sms
    range, and limit cycling.
 g) Only then transfer/episode (which need the spin fallback too).

## 7. Logs to capture

- CSV (results/gputronic_gold*.csv): time_s, z_raw, z_hat, dzdt, rate_ema, active_sms, error,
  integral, ctrl_hz, rate_ref.
- Console: [GPU], [ZC ], [LAW], [CTRL], [CAL] rate_ref + table rows + n_ff/sms_per_u, [TEL] per
  second, [STATS] mean/std Z and active_sms range, dyno/closed-loop reports in results/.
- In parallel: nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,
  utilization.gpu,clocks_throttle_reasons.active --format=csv -lms 100 > smi.csv
  (for D-axis: clocks/power/thermal, and whether the spin fallback changes boost behavior).
- Environment: driver version, CUDA version, nvcc version, kernel/OS, PCIe link gen/width
  (nvidia-smi -q | grep -A3 "Link").
