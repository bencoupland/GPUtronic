# GPUtronic v0.1 – Proof-of-Concept: Closed-Loop GPU as "Instruction Pressure Engine"

GPUtronic is an experimental, self-built closed-loop control system for GPUs.  
It treats the GPU as a physical engine with real-time feedback, adaptive scaling, and safety mechanisms — something current drivers and schedulers do not provide.

![gputronic](https://github.com/user-attachments/assets/10960083-fe38-47f8-b7e2-9794f4d207ca)

### The Big Idea: GPU as an "Instruction Pressure Engine"

Modern GPUs are black-box throughput monsters: launch a kernel, wait, repeat. No real-time "throttle", no feedback loop, no dynamic adjustment of "displacement" (active SMs/blocks).  

GPUtronic changes that. It borrows from automotive ECUs (Bosch Motronic on my Audi B5 S4 2.7T) and field-oriented motor control (open FOC on my homebuilt ebikes) to make the GPU behave like a controllable engine:

- **SMs (Streaming Multiprocessors)** = cylinders  
- **Warps (groups of 32 threads)** = crankshaft pulses  
- **Block count** = variable displacement (automatic gear shifting)  
- **Throttle fraction (0.0–1.0)** = fuel/air mixture (controls active threads per cycle)  
- **Atomic counter** = hall-effect tachometer (RPM = instruction throughput rate)  
- **PID governor** = main ECU loop (adjusts throttle to hold target RPM)  
- **Thermal knock** = safety retard (pull throttle / downshift on high temp)  
- **Persistent kernel** = always-running engine (no repeated launches)

The result: a GPU that self-regulates like a real engine — holding "RPM" (instruction pressure), shifting "gears" (blocks) on demand, and protecting itself from "knock" (overheating).

### Core GPU Concepts Explained (Why This Matters)

- **Occupancy**  
  Percentage of maximum threads running on each SM.  
  High occupancy hides latency (great for memory-bound work — lots of warps waiting).  
  Low occupancy allows more registers per thread (better for compute-bound work like our load).  
  → We dynamically adjust block count to find the sweet spot for current demand.

- **Warp Divergence**  
  When threads in a warp take different paths (if/else), the warp serializes → stalls.  
  → We avoid it with uniform fixed loops (no variable iterations per thread in baseline).

- **Instruction Pressure**  
  Rate of instructions issued per cycle. Our tanf/sqrtf ops are high-latency (10–30+ cycles each) → stresses FPU pipelines, mimicking heavy compute workloads.

- **Memory Pressure**  
  Bandwidth demand on global/shared/L2 caches. Our workload is compute-bound (low memory pressure), but real apps mix it — dynamic scaling helps balance.

- **Stalls**  
  Threads wait for data, dependencies, or resources. Clock64() idle yields without heavy fences; sparse atomics reduce contention stalls.

### Current State (GTX 1080, Pascal GP104)

- Ceiling: ~140–158k RPM (varies with block count)  
- Idle RPM: ~80–130 (low util/temp)  
- Auto-shifts: Triggers on sustained high demand (first real upshifts seen!)  
- Max TDP mode: ~96–108W (not full 200W yet — short bursts vs sustained game load)  
- Dyno sweeps: Repeatable, CSV logging for analysis  
- Thermal safety: Forced downshift at ≥80°C

It's rough — ceiling inconsistent, max TDP not full, no real kernels yet (synthetic load for testing).  
But the loop works. Control is real-time on stock drivers. Pascal limits obvious — Ada/Blackwell will be interesting to see.

### Build & Run

nvcc -o gputronic main.cu \
  -arch=sm_61 -O3 -lineinfo \
  -Xcompiler "-Wall -Wextra" -std=c++11 \
  /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05  # adjust to your NVML lib path

./gputronic

Keys:
w / s : ±10k RPM setpoint
t : Run full-band dyno sweep (0–200k RPM)
q : Quit

Dependencies: NVIDIA driver + CUDA toolkit (tested on Nvidia 580 series driver, CUDA 12.0, Linux Mint 22.2, GTX 1080/GP104 "Pascal" architecture)

How to Integrate Real Workloads:
Replace the power stroke loop with your real compute (e.g., GEMM, conv, inference layer, shader). The control plane (PID, shifts, tach, telemetry) stays the same.

Example:

if (tid < (total_threads * throttle)) {
    // Your real short/long workload here, e.g. batched matmul, reduction, procedural generation
    if (threadIdx.x % 32 == 0) atomicAdd(d_rpm_counter, 1);
}

Short kernels (10–100 cycles) benefit most from dynamic block scaling — upshift for bursts, downshift for idle.

Porting to Newer Architectures:
Volta+ (sm_70+): Use __nanosleep() instead of clock64() busy-wait → lower power, more precise idle.
Ampere+ / Ada / Blackwell: Higher SM count → increase max_blocks (e.g., 200–400), better atomic perf, native persistent threads.

Profiling with Nsight:
nsight-compute --target sm_61 ./gputronic

Future Plans

Real workloads (GEMM, inference, rendering, sims)
Better tach accuracy (hybrid atomic + clock64())
Full knock retard (throttle pull before downshift)
Max TDP to 200W+
Port to Ada/Blackwell

Thanks for checking it out -- let's make GPUs controllable like engines
