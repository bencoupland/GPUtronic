# GPUtronic
Closed-loop GPU control: treating the GPU as an Instruction Pressure Engine with PID governor, dynamic block scaling, and thermal knock retard

# GPUtronic v0.1 – Proof-of-Concept: Closed-Loop GPU Control

**Treat the GPU as an "Instruction Pressure Engine" with real-time feedback, adaptive scaling, and safety features.**

Inspired by Bosch Motronic ECU tuning on my Audi B5 S4 (2.7T biturbo, daily driven 12 years) and open-source FOC firmware on my homebuilt 2kW ebikes.

This is an early prototype that proves GPUs can be controlled like physical engines: persistent kernel, atomic tachometer for RPM, PID governor for throttle, dynamic block scaling ("automatic transmission"), thermal knock retard.

**Current state (GTX 1080, Pascal GP104):**
- Ceiling ~140–158k RPM (varies with block count)
- Auto-shifts up/down on sustained high demand
- Idle RPM ~80–130, low util/temp
- Max TDP mode pushes ~100–110W (not full 200W yet)
- Dyno sweeps & CSV logging for repeatable testing

**Core idea:** GPUs are still open-loop carbs — tweak scheduler hints, grid sizes per workload. GPUtronic adds EFI + turbo + knock control.

### Build & Run

```bash
nvcc -o gputronic main.cu \
  -arch=sm_61 -O3 -lineinfo \
  -Xcompiler "-Wall -Wextra" -std=c++11 \
  /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05  # adjust path to your NVML lib
