# Contributor notes

## What this is
GPUtronic is closed-loop control of a GPU's useful-work rate. A persistent kernel counts finished work,
a host thread computes `Z = rate_ref / rate`, and a PI controller sets how many SMs do the work
(`active_sms`). Temperature and power stay with the NVIDIA driver. Details:
`docs/GPUtronic-GOLD-1.0-FUNCTION-FRAME.md`.

## Build
CUDA 12.8, RTX 5080 (`sm_120`) by default:

    make gold lib demo
    make NVCC=/path/to/nvcc ARCH=-arch=sm_XX gold   # other toolkits / GPUs

Pascal (sm_61) does not build yet because the kernels use `__nanosleep`; see
`docs/analysis/gtx1080-bringup-plan.md`.

## Checks
Run on a GPU before merging controller changes:

    ./build/gputronic_gold check
    make transfer-check
    make episode-check

A pass means the counter works and the rate moves the right way. Also report mean Z against the 1.5
target.

## Rules
- `total_work_pulses` is only ever increased with `atomicAdd`; never assign it from one SM.
- Do not form the rate over a single 10 µs tick; use the 8 ms window.
- The actuator is the SM count. Sleep duration is history (`archive/gold-1.0.0-nanosleep/`).
- Keep Kp and Ki at their defaults unless a hardware run justifies a change; record why.
- Put experiments in `archive/` or a branch, not in the main source files.
- Say clearly which results are measured on hardware and which are modelled.
