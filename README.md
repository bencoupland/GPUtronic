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

This project is self-contained in a single CUDA source file. Tested on Ubuntu/Mint with NVIDIA driver 580 series + CUDA 12 toolkit.

**Compile** (adjust NVML lib path if needed):
```bash
nvcc -o gputronic main.cu \
  -arch=sm_61 \
  -O3 \
  -lineinfo \
  -Xcompiler "-Wall -Wextra" \
  -std=c++11 \
  /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05  # Path to your libnvidia-ml.so (find with `find /usr -name libnvidia-ml.so`)
```
**Run**
./gputronic

**Keys:**
w / s : ±10k RPM setpoint
t : Run full-band dyno sweep (0–200k RPM)
q : Quit

**Dependencies:** NVIDIA driver + CUDA toolkit (tested on Nvidia 580 series driver, CUDA 12.0, Linux Mint 22.2, GTX 1080/GP104 "Pascal" architecture)

### How to Integrate Real Workloads:
Replace the power stroke loop with your real compute (e.g., GEMM, conv, inference layer, shader). The control plane (PID, shifts, tach, telemetry) stays the same.

Example:
```bash
if (tid < (total_threads * throttle)) {

    // Your real short/long workload here, e.g. batched matmul, reduction, procedural generation

    if (threadIdx.x % 32 == 0) atomicAdd(d_rpm_counter, 1);
}
```
Short kernels (10–100 cycles) benefit most from dynamic block scaling — upshift for bursts, downshift for idle.

**Porting to Newer Architectures:**
Volta+ (sm_70+): Use __nanosleep() instead of clock64() busy-wait → lower power, more precise idle.
Ampere / Ada / Blackwell: Higher SM count → increase max_blocks (e.g., 200–400), better atomic perf, native persistent threads.

**Profiling with Nsight:**
nsight-compute --target sm_61 ./gputronic

**Future Plans:**
Real workloads (GEMM, inference, rendering, sims), better tach accuracy (hybrid atomic + clock64()), full knock retard (throttle pull before downshift), max TDP to 200W+, port to Ada/Blackwell.

![gputronic_60block_dyno](https://github.com/user-attachments/assets/258a58b9-c30a-41b9-9b96-22ba652c521f)

### Understanding RPM in GPUtronic: What It Actually Measures

In GPUtronic, **RPM** is **not** literal revolutions per minute — it's a deliberate analogy to make the concept intuitive.  

**RPM = warp completion rate (instruction throughput proxy)**

- Each "tick" in the atomic counter happens when a warp leader (one thread per warp of 32) completes the power stroke loop.
- Every **32 threads completing their work** = **1 tick** (one warp finished).
- The ECU loop runs at **10 Hz** (100 ms period) → we calculate delta ticks per 100 ms.
- To give it an "engine feel", we annualize: `actual_rpm = delta_ticks_per_100ms × 600` (10 loops/sec × 60 sec/min).

So **1 RPM ≈ 32 threads/second** completing their workload (warp throughput).  

**Distinctions & Concepts**:
- **Warp throughput** (what RPM measures): Number of warps (groups of 32 threads) completing work per second. High RPM = high warp completion rate = high overall compute throughput.
- **Thread throughput**: Would be RPM × 32 (threads/second completing work). We could normalize to this for "real" units, but RPM preserves the engine analogy and makes dyno curves intuitive.
- **Instruction throughput**: Actual instructions/second executed. RPM is a proxy — our fixed 400-iter transcendental loop (tanf/sqrtf) creates consistent pressure, but real workloads vary (e.g., GEMM has more FMA, memory-bound has fewer instructions per cycle).
- **Occupancy impact**: Higher block counts increase warps/SM (hides latency), but too many cause register pressure, spills, contention → lower IPC per warp. That's why 40–60 blocks often beats 80+.
- **Divergence & stalls**: Avoided in baseline (uniform 400 iters). Hybrid variable iters caused stalls (threads in same warp finish at different times) → reduced ceiling.
- **Why RPM is useful**: It's a **consistent, workload-independent metric** for tuning control (PID, shifts). In real apps, higher RPM = more work done per second.

**Dyno plot example (60-block run):**
- Near-perfect tracking to ~130k target RPM
- Throttle ramps to 100% at ~140k
- Plateau at ~150–151k RPM (saturation point)

This shows the engine concept in action: requested throughput (target) vs delivered (actual), with throttle as the control input.

![40block](https://github.com/user-attachments/assets/e82cd239-96f5-43a4-98a1-31ab146c0a9d)
![60block](https://github.com/user-attachments/assets/5e7fb933-0163-45a0-84db-f54c39500c23)
![80block](https://github.com/user-attachments/assets/9090bd26-6b57-4500-96e3-9c1295cd6243)

Thanks for checking it out -- let's make GPUs controllable like engines
