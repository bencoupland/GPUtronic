/**
 * GPUtronic v0.1 – Proof-of-Concept Release: Closed-Loop GPU as "Instruction Pressure Engine"
 * =====================================================================================
 * Project Vision & Big Picture
 * ----------------------------
 * GPUtronic is a proof-of-concept that treats the GPU as a controllable "engine" with closed-loop feedback, adaptive scaling, and safety features.
 * Inspired by Bosch Motronic ECU from the Audi B5 S4 (2.7T biturbo tuning) and field-oriented control (FOC) from open-source ebike firmware.
 * The GPU is an "Instruction Pressure Engine" where:
 *   - Streaming Multiprocessors (SMs) are "cylinders" (processing units)
 *   - Warps (32 threads) are "crankshaft pulses" (execution units)
 *   - Block count is "variable displacement" (dynamic gear shifting to adapt occupancy)
 *   - Throttle (0.0-1.0) controls active threads per cycle (fuel mixture)
 *   - Atomic counter is the "hall-effect tachometer" for RPM (instruction throughput feedback)
 *   - PID governor adjusts throttle to hold setpoint RPM
 *   - Thermal knock retard pulls throttle on heat
 *   - Dynamic block scaling shifts "gears" on demand overload or underload
 * - Max TDP mode for burn-in testing (full saturation)

 * Core GPU Concepts Explained:
 *   - Occupancy: % of max threads running on SMs. High occupancy hides latency (good for memory-bound work), low occupancy allows more registers per thread (good for compute-bound like our load). Dynamic scaling adapts occupancy to demand.
 *   - Warp Divergence: When threads in a warp take different paths, execution serializes → stalls. Avoided here with uniform work loops.
 *   - Instruction Pressure: Rate of instructions issued per cycle. Our tanf/sqrtf ops are high-latency (10–30+ cycles) – stresses FPU pipelines.
 *   - Memory Pressure: Bandwidth demand on global/shared/L2. This workload is compute-bound (low pressure), but real apps mix it – dynamic scaling helps balance.
 *   - Stalls: Threads wait for data, dependencies, resources. clock64() idle yields without heavy fences; sparse atomics reduce contention stalls.

 * How to Integrate Real Workloads:
 *   - Replace the power stroke loop with your real kernel (e.g., GEMM, conv, inference layer).
 *   - Example:
 *     if (tid < (total_threads * throttle)) {
 *         // Your real work: e.g., cublasSgemm(handle, op1, op2, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 *         // Or a short reduction, shader, etc.
 *         if (threadIdx.x % 32 == 0) atomicAdd(d_rpm_counter, 1);
 *     }
 *   - For short-thread workloads (10–100 cycles), dynamic scaling shines – upshift for bursts, downshift for idle.
 *   - Port to Newer Arch: On Volta+ (sm_70+), use __nanosleep() for idle instead of clock64() – more precise, lower power. Increase max_blocks for higher SM count (Ada has 144 SMs).
 *   - Nsight Profiling: Use 'nsight-compute --target sm_61 ./gputronic' to profile stalls/occupancy during sweeps. Look for 'Warp State' (stall reasons) and 'Occupancy' charts.

 * License: MIT – fork, improve, share.
 * GitHub: https://github.com/bencoupland/GPUtronic
 */

#include <cuda_runtime.h>           // Core CUDA runtime APIs – kernel launch, memory management, synchronization
#include <device_launch_parameters.h> // Device-side globals: blockIdx, threadIdx, gridDim, blockDim – essential for thread indexing
#include <iostream>                 // Console I/O – used for real-time dashboard gauges (std::cout with \r overwrite)
#include <fstream>                  // File output – CSV black-box logging of dyno data, shifts, knock events
#include <nvml.h>                   // NVIDIA Management Library – real-time GPU sensors (temp, power, clock, util)
#include <unistd.h>                 // usleep – controls the 10Hz ECU loop period
#include <fcntl.h>                  // File control flags – enables non-blocking keyboard input
#include <termios.h>                // Terminal settings – raw mode for instant key detection (gas pedal)

// [INPUTS] Keyboard Gas Pedal Sensor – detects 'w/s/t/q' without Enter
int kbhit() {
    struct termios oldt, newt;  // Terminal state – oldt saves original, newt modifies for raw mode
    int ch, oldf;               // ch = key read, oldf = file flags
    tcgetattr(STDIN_FILENO, &oldt);  // Get current settings
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);  // Raw mode: no buffer, no echo for instant response
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);  // Apply changes now
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);  // Non-blocking read – don't wait for key
    ch = getchar();                 // Attempt read
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);  // Restore settings
    fcntl(STDIN_FILENO, F_SETFL, oldf);       // Restore flags
    if(ch != EOF) { ungetc(ch, stdin); return 1; }  // Key pressed – push back for main to read
    return 0;                       // No key
}

// [KERNEL] Persistent Instruction Pressure Engine – the "combustion chamber"
__global__ void gputronic_engine(float *d_throttle, unsigned long long *d_rpm_counter, int *dummy_out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Unique thread (cylinder) ID
    int total_threads = gridDim.x * blockDim.x;       // Total "displacement" – all threads in grid

    while(true) {
        float throttle = *d_throttle;                 // Zero-copy read from host (PCIe latency ~μs)

        // [SHUTDOWN_SIG] Clutch signal – exit for gear changes or key-off
        if (throttle < -0.5f) break;

        if (tid < (total_threads * throttle)) {
            // [STROKE] Power Stroke: Fixed 400 iterations – maximizes IPC, no divergence
            // tanf & sqrtf are high-latency transcendental ops (10–30+ cycles each) → high instruction pressure
            // Fixed iterations eliminate warp divergence (uniform execution per warp) → better scheduler efficiency
            float val = (float)tid;
            for(int i = 0; i < 400; i++) {
                val = tanf(val) + sqrtf(val);            // Transcendental: 10–30+ cycles each, high instruction pressure
            }
            dummy_out[tid % 16384] = (int)val;           // Global write – light memory pressure, prevents compiler dead-code elim

            // [ENCODER] Tach Pulse: Sparse to minimize atomic stalls/contention
            // Warp leader (1/32 sparsity) – one pulse per warp. Balances accuracy with low overhead
            // AtomicAdd is efficient on Pascal for int, but high frequency can cause serialization stalls – sparsity mitigates
            if (threadIdx.x % 32 == 0) atomicAdd(d_rpm_counter, 1);
        } else {
            // [IDLE] Bypass Valve: clock64() busy-wait – no PCIe/system fence overhead
            // Delay scales with idle (1 - throttle) – longer idle = longer spin → lower util/temp
            // clock64() is per-SM cycle counter – fast local read, no cross-GPU sync. Preferred over __threadfence_system() on Pascal.
            long long start = clock64();                 // Per-SM cycle counter – fast local read
            long long delay = (long long)((1.0f - throttle) * 1500000LL);  // Max ~0.94ms at 1.6GHz – tunable for idle util
            while (clock64() < start + delay) {
                // Yield hint every 1000 cycles – __syncthreads() encourages scheduler to reschedule warps without heavy cost
                if ((clock64() - start) % 1000 == 0) __syncthreads();  // Block sync – low-cost yield
            }
        }
    }
}

int main() {
    // [INIT] NVML – OBD-II diagnostic interface for GPU sensors
    nvmlInit();                                  // Boot NVML – required for all calls
    nvmlDevice_t device;                         // Device handle for GPU 0
    nvmlDeviceGetHandleByIndex(0, &device);      // Assume single GPU – index 0

    // [REC] Black Box Recorder – CSV append mode for multi-session persistence
    std::ofstream logFile("gputronic_master_dyno.csv", std::ios::app);  // CSV for dyno plots/analysis

    // [CAN-BUS] Zero-Copy Telemetry – host/device shared memory for low-latency control
    float *h_throttle, *d_throttle;             // Throttle angle (host/device views)
    unsigned long long *h_rpm_counter, *d_rpm_counter;  // Tachometer counter
    cudaHostAlloc((void**)&h_throttle, sizeof(float), cudaHostAllocMapped);  // Pinned mapped mem – low-latency PCIe
    cudaHostGetDevicePointer((void**)&d_throttle, (void*)h_throttle, 0);
    cudaHostAlloc((void**)&h_rpm_counter, sizeof(unsigned long long), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&d_rpm_counter, (void*)h_rpm_counter, 0);

    int *d_out;                                  // Exhaust buffer – VRAM dump to prevent dead-code elim
    cudaMalloc(&d_out, 16384 * sizeof(int));     // Fixed size – modulo access to avoid out-of-bounds

    // [CAL] Calibration Constants – tuned for GTX 1080 (GP104 Pascal)
    int current_blocks = 40;                     // Golden Ratio start – proven peak for fixed load
    const int min_blocks = 20;                   // Lowest gear – minimal persistence
    const int max_blocks = 80;                   // Max gear – avoids occupancy penalties
    float target_rpm = 10000.0f;                 // Initial idle setpoint
    float Kp = 0.000003f;                        // Proportional gain – immediate response to error
    float Ki = 0.0000008f;                       // Integral gain – eliminates steady-state error
    float integral_error = 0.0f;                 // Accumulated error – clamped to prevent windup
    unsigned long long last_count = 0;           // Previous tach value for delta RPM

    // [SHIFT_CAL] Gearbox Parameters – Debounce + Hysteresis + Thermal Knock
    static int up_lag_counter = 0;               // Counter for upshift confirmation
    static int down_lag_counter = 0;             // Counter for downshift
    const int debounce_polls = 5;                // ~500ms confirmation to avoid false shifts
    const int min_shift_interval = 30;           // ~3s cooldown between shifts (hysteresis)
    static int last_shift_poll = 0;              // Previous shift poll for cooldown
    int poll_count = 0;                          // Global ECU cycle counter

    *h_throttle = 0.1f;                          // Prime throttle
    *h_rpm_counter = 0;                          // Previous tach value for delta RPM

    std::cout << "======================================================" << std::endl;
    std::cout << " GPUtronic Stage 0.1: Closed-Loop Control for GPUs" << std::endl;
    std::cout << " [W/S] ±10k RPM | [T] Dyno Sweep | [Q] Key-Off" << std::endl;
    std::cout << "======================================================" << std::endl;

    // [IGNITION] Launch initial kernel in golden ratio gear
    gputronic_engine<<<current_blocks, 128>>>(d_throttle, d_rpm_counter, d_out);

    while (true) {
        poll_count++;

        // [SENSE_TELEM] Full NVML sensor suite – engine health monitoring
        unsigned int temp = 0;
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
        unsigned int power = 0;
        nvmlDeviceGetPowerUsage(device, &power);
        unsigned int clock = 0;
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock);
        nvmlUtilization_t util;
        nvmlDeviceGetUtilizationRates(device, &util);

        // [TACH] Delta RPM calculation – from atomic counter pulses
        unsigned long long current_count = *h_rpm_counter;
        float actual_rpm = (float)(current_count - last_count);
        last_count = current_count;

        // [LL_RPM] PID Feedback Loop – throttle adjustment to hold setpoint
        float error = target_rpm - actual_rpm;
        integral_error += error;
        if (integral_error > 20.0f) integral_error = 20.0f;  // Anti-windup clamp (prevents runaway)
        if (integral_error < -20.0f) integral_error = -20.0f;
        *h_throttle += (Kp * error) + (Ki * integral_error);

        // Actuator limits – prevent zero or over-throttle
        if (*h_throttle < 0.001f) *h_throttle = 0.001f;
        if (*h_throttle > 1.0f) *h_throttle = 1.0f;

        // [USER] Keyboard Event Handler
        if (kbhit()) {
            char c = getchar();
            if (c == 'w') target_rpm += 10000.0f;
            if (c == 's') target_rpm -= 10000.0f;
            if (c == 'q') break;

            if (c == 't') {
                std::cout << "\n\n[!] FULL BAND DYNO SWEEP (" << current_blocks << " Blocks)\n";
                printf("%-12s | %-12s | %-12s | %-6s\n", "Tgt RPM", "Act RPM", "Thr %", "Temp");
                logFile << "SWEEP_START,Blocks," << current_blocks << "\n";

                for (float sweep_tgt = 0; sweep_tgt <= 200000; sweep_tgt += 10000) {
                    for (int settle = 0; settle < 25; settle++) {
                        unsigned long long s_count = *h_rpm_counter;
                        float s_actual = (float)(s_count - last_count);
                        last_count = s_count;
                        *h_throttle += (Kp * (sweep_tgt - s_actual));
                        if (*h_throttle > 1.0f) *h_throttle = 1.0f;
                        if (*h_throttle < 0.001f) *h_throttle = 0.001f;
                        usleep(100000);
                    }

                    unsigned int s_temp;
                    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &s_temp);
                    unsigned long long f_count = *h_rpm_counter;
                    float f_actual = (float)(f_count - last_count);
                    last_count = f_count;

                    printf("%-12d | %-12d | %-11d%% | %dC\n",
                           (int)sweep_tgt, (int)f_actual, (int)(*h_throttle * 100), s_temp);
                    logFile << "DATA," << sweep_tgt << "," << (int)f_actual << "," << (int)(*h_throttle * 100) << "," << s_temp << "\n";
                }
                std::cout << "-------------------------------------------------------------\n" << std::endl;
            }
        }

        // [TRANS_CTL] Automatic Gearbox – Dynamic Displacement Control
        bool shift_active = false;
        bool is_upshift = false;

        if (poll_count - last_shift_poll >= min_shift_interval) {
            float error_val = target_rpm - actual_rpm;

            // Upshift: saturated throttle + significant lag (debounced)
            if (*h_throttle > 0.98f && error_val > 8000.0f && current_blocks < max_blocks) {
                up_lag_counter++;
                if (up_lag_counter >= debounce_polls) {
                    current_blocks += 20;  // Dramatic step – shows concept well
                    shift_active = true;
                    is_upshift = true;
                    up_lag_counter = 0;
                }
            } else {
                up_lag_counter = 0;
            }

            // Downshift: light load + target still high
            if (*h_throttle < 0.10f && target_rpm > 20000.0f && current_blocks > min_blocks) {
                down_lag_counter++;
                if (down_lag_counter >= debounce_polls) {
                    current_blocks -= 20;  // Dramatic step
                    shift_active = true;
                    down_lag_counter = 0;
                }
            } else {
                down_lag_counter = 0;
            }

            // [KNOCK] Thermal safety – forced downshift on heat
            if (temp >= 80 && current_blocks > min_blocks) {
                current_blocks -= 20;
                shift_active = true;
                std::cout << "\n[!KNOCK!] Thermal limit: Forced downshift to " << current_blocks << "\n";
                logFile << "KNOCK,DOWNSHIFT," << current_blocks << "," << actual_rpm << "," << temp << "\n";
            }
        }

        if (shift_active) {
            std::cout << "\n[!] " << (is_upshift ? "UPSHIFT" : "DOWNSHIFT") << " to " << current_blocks << " blocks\n";

            logFile << "SHIFT," << (is_upshift ? "UP" : "DOWN") << "," << current_blocks << "," << actual_rpm << "," << temp << ","
                    << power << "," << clock << "," << util.gpu << "\n";

            *h_throttle = -1.0f;             // Clutch in – signal kernel exit
            cudaDeviceSynchronize();         // Wait for all SMs to stop
            *h_throttle = 0.1f;              // Reset throttle
            integral_error = 0.0f;           // Reset PID memory to avoid surge

            gputronic_engine<<<current_blocks, 128>>>(d_throttle, d_rpm_counter, d_out);

            last_shift_poll = poll_count;
        }

        // [DISP] Dashboard – real-time engine gauges with expanded metrics
        std::cout << "\r[ECU] Blk:" << current_blocks
                  << " Tgt:" << (int)target_rpm
                  << " Act:" << (int)actual_rpm
                  << " Thr:" << (int)(*h_throttle * 100) << "% "
                  << " Tmp:" << temp << "C "
                  << " Pwr:" << power / 1000 << "W "
                  << " Clk:" << clock << "MHz "
                  << " Utl:" << util.gpu << "% " << std::flush;

        usleep(100000);  // 10 Hz ECU cycle
    }

    // [SHUT] Safe engine shutdown sequence
    std::cout << "\n\n[!] KEY-OFF. COOLING DOWN..." << std::endl;
    logFile << "SESSION_END\n";
    logFile.close();

    *h_throttle = -1.0f;
    cudaDeviceSynchronize();
    nvmlShutdown();
    cudaFreeHost(h_throttle);
    cudaFreeHost(h_rpm_counter);
    cudaFree(d_out);

    std::cout << "Engine Cold. System Safe." << std::endl;
    return 0;
}
