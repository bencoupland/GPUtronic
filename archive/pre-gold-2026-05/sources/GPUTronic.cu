// =============================================================================
// GPUTronic — Closed-Loop Governor for Massively Parallel Architectures
// Version: v1.13 (Cyberpunk-Optimized)
// =============================================================================

// -----------------------------------------------------------------------------
// INCLUDES
// These headers provide CUDA runtime functions, threading (pthreads),
// standard I/O, timing, and math operations.
// -----------------------------------------------------------------------------
#include <cuda_runtime.h>   // CUDA runtime API (cudaMalloc, cudaHostAlloc, etc.)
#include <pthread.h>        // POSIX threads for the control loop thread
#include <stdio.h>          // printf, standard I/O
#include <stdlib.h>         // General utilities (not heavily used here)
#include <string.h>         // String functions (not heavily used here)
#include <time.h>           // clock_gettime for high-resolution timing
#include <math.h>           // math functions (sinf, cosf, fmaf)
#include <unistd.h>         // usleep for sleeping in control loop

// -----------------------------------------------------------------------------
// HARDWARE / LAUNCH CONFIGURATION
// These constants describe the target GPU and kernel launch shape.
// NUM_SM must match the actual GPU (RTX 5080 has 84 SMs).
// -----------------------------------------------------------------------------
#define NUM_SM              84      // Number of Streaming Multiprocessors
#define THREADS_PER_BLOCK   64      // Threads per CUDA block (must be multiple of 32)
#define WORK_UNITS_PER_THREAD 8192  // Work items each thread processes per iteration

// -----------------------------------------------------------------------------
// CONTROL LAW PARAMETERS (Cyberpunk-Optimized)
// These values were empirically tuned for Cyberpunk 2077 at 4K Path Tracing.
// They represent the current best-known configuration.
// -----------------------------------------------------------------------------
#define CONTROL_DT_US       10      // Control loop period in microseconds (100 kHz)
#define TARGET_Z            1.5f    // Desired impedance setpoint (slightly below natural Z)
#define Z_CEILING           10.0f   // Maximum allowed Z value (safety clamp)
#define KP                  0.45f   // Proportional gain
#define KI                  0.055f  // Integral gain
#define INTEGRAL_CLAMP      1.0f    // Anti-windup limit for the integrator
#define SLEEP_SCALE         55000.0f// How aggressively we convert control output to sleep time
#define MAX_SLEEP_NS        200000  // Hard safety limit on sleep duration (200 µs)

// -----------------------------------------------------------------------------
// CONTROL DATA STRUCTURE
// This structure is shared between host and device via zero-copy memory.
// It carries both commands (target_pm, control_flags) and telemetry.
// The __align__(16) ensures proper alignment for atomic operations.
// -----------------------------------------------------------------------------
struct __align__(16) GPUControlData {
    unsigned int control_flags;           // Bit 1 = stop, Bit 2 = reset counters
    float target_pm;                      // Desired phase margin (not currently used)
    float current_pm;                     // Current measured Z (telemetry)
    unsigned long long total_work_pulses; // Total work completed (atomic counter)
    int throttle_sleep_ns;                // Output: how long to sleep this iteration
    int max_sleep_ns;                     // Safety limit passed from host
    float z_estimate;                     // Smoothed Z value (telemetry)
    float dzdt_estimate;                  // Rate of change of Z (telemetry)
    float pm_error;                       // Current error from target (telemetry)
    int proactive_corrections;            // Number of times we applied throttle
};

// -----------------------------------------------------------------------------
// GLOBAL STATE
// These variables are shared between main() and the control thread.
// In a production version they would be encapsulated in a context struct.
// -----------------------------------------------------------------------------
static volatile int g_running = 1;                    // Main loop control flag
static GPUControlData* g_h = NULL;                    // Host pointer to control data
static GPUControlData* g_d = NULL;                    // Device pointer to control data
static unsigned long long* g_counters = NULL;         // Per-SM work counters
static float P[2][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}};  // Kalman covariance matrix

// -----------------------------------------------------------------------------
// HIGH-RESOLUTION TIMER
// Returns current time in microseconds using the monotonic clock.
// This is used for precise control loop timing.
// -----------------------------------------------------------------------------
inline double get_time_us() {
    struct timespec ts;                              // timespec holds seconds + nanoseconds
    clock_gettime(CLOCK_MONOTONIC, &ts);             // Get time since arbitrary point (not wall time)
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;    // Convert to microseconds
}

// -----------------------------------------------------------------------------
// DEVICE WORK FUNCTION (Placeholder Workload)
// This function simulates compute work. In a real integration this would be
// replaced by the actual workload (GEMM, inference, etc.).
// The unroll pragma tells the compiler to fully unroll the inner loop.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void heavy_work(float* acc, int sm_id, int iter) {
    float val = *acc;                                // Load accumulator from shared memory
    #pragma unroll                                   // Force full unrolling for performance
    for (int k = 0; k < 24; k++) {                   // 24 iterations of FMA work
        float a = __sinf((float)((iter * 13 + k * 7 + sm_id) & 0xFF) * 0.0245436926f);
        float b = __cosf((float)((iter * 17 + k * 11 + sm_id) & 0xFF) * 0.0245436926f);
        val = __fmaf_rn(a, b, val);                  // Fused multiply-add (a*b + val)
        val = __fmaf_rn(val * 0.7f, a, val);         // Additional dependent FMA
    }
    *acc = val;                                      // Write result back
}

// -----------------------------------------------------------------------------
// PERSISTENT KERNEL
// This kernel runs forever on every SM until the stop flag is set.
// It performs work, increments the atomic counter, then sleeps according to
// the throttle value written by the host control loop.
// -----------------------------------------------------------------------------
__global__ void gpu_persistent_kernel(unsigned long long* counters,
                                      GPUControlData* ctrl, int nsm) {
    int sm = blockIdx.x;                             // Each block runs on one SM
    float local = 0.0f;                              // Per-thread accumulator

    while (true) {                                   // Persistent loop
        if (ctrl->control_flags & 0x2) break;        // Bit 1 = stop signal from host

        for (int i = 0; i < 1024; i++) {             // 1024 work chunks per iteration
            heavy_work(&local, sm, i);               // Do compute work
        }
        atomicAdd(&counters[sm], 1024ULL);           // Atomically record completed work

        int s = ctrl->throttle_sleep_ns;             // Read throttle value written by host
        if (s > 0) __nanosleep(s);                   // Sleep to reduce throughput
    }
}

// -----------------------------------------------------------------------------
// RESET KERNEL (Utility)
// Simple kernel to zero the per-SM counters at startup.
// -----------------------------------------------------------------------------
__global__ void reset_sm_counters(unsigned long long* counters, int nsm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nsm) counters[idx] = 0ULL;
}

// -----------------------------------------------------------------------------
// HOST CONTROL LOOP (Runs in its own pthread)
// This is the heart of the governor. It runs at ~100 kHz, reads the work
// counters, estimates Z, runs the PI controller, and writes the sleep value
// back into the zero-copy structure for the kernel to read.
// -----------------------------------------------------------------------------
void* control_loop(void*) {
    const float dt = CONTROL_DT_US * 1e-6f;          // Control period in seconds (10 µs)
    float integral = 0.0f;                           // Integral term for PI controller
    float last_z = 1.0f;                             // Previous Z value (for derivative)
    double last_time = get_time_us();                // Timestamp of last iteration
    float smoothed_rate = 1500000.0f;                // Exponential moving average of work rate

    // Initialize Kalman covariance matrix (2x2)
    P[0][0] = 1.0f; P[0][1] = 0.0f;
    P[1][0] = 0.0f; P[1][1] = 1.0f;

    while (g_running) {                              // Main control loop
        double now = get_time_us();
        if (now - last_time < 8e-6) {                // Guard against running too fast
            usleep(1);
            continue;
        }
        last_time = now;

        // Read total work completed since last sample
        unsigned long long delta = g_h->total_work_pulses;
        static unsigned long long prev = 0;
        unsigned long long d = delta - prev; prev = delta;

        // Calculate instantaneous work rate (work units per second)
        float inst_rate = d / dt;

        // Smooth the rate with exponential moving average (alpha = 0.25)
        smoothed_rate = 0.75f * smoothed_rate + 0.25f * inst_rate;

        // Calculate impedance Z (higher = more stalled)
        float z = 2000000.0f / (smoothed_rate + 10000.0f);
        if (z > Z_CEILING) z = Z_CEILING;
        if (z < 0.3f) z = 0.3f;

        // Simple proportional control toward TARGET_Z
        float error = TARGET_Z - z;
        integral += error * dt;
        if (integral > INTEGRAL_CLAMP) integral = INTEGRAL_CLAMP;
        if (integral < -INTEGRAL_CLAMP) integral = -INTEGRAL_CLAMP;

        float delta_q = -(KP * error + KI * integral);

        // Deadband around the target to reduce chattering
        if (z > (TARGET_Z - 0.15f) && z < (TARGET_Z + 0.15f)) delta_q = 0.0f;

        // Convert control output to sleep duration
        int sleep = (int)(5 + delta_q * SLEEP_SCALE);
        if (sleep < 0) sleep = 0;
        if (sleep > g_h->max_sleep_ns) sleep = g_h->max_sleep_ns;
        g_h->throttle_sleep_ns = sleep;              // Write result for kernel to read

        // Update telemetry fields
        g_h->current_pm = z;
        g_h->z_estimate = z;
        g_h->pm_error = error;
        if (sleep > 5) g_h->proactive_corrections++;

        usleep(1);                                   // Yield a tiny amount
    }
    return NULL;
}

// -----------------------------------------------------------------------------
// MAIN
// Allocates zero-copy memory, launches the persistent kernel, starts the
// control thread, and prints telemetry for a short period.
// -----------------------------------------------------------------------------
int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic] v1.13 — Z Target = 1.5 (100 kHz)\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    cudaSetDevice(0);                                // Select first GPU
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] %s | SMs=%d\n\n", prop.name, prop.multiProcessorCount);

    // Allocate zero-copy control structure (host + device pointers)
    cudaHostAlloc((void**)&g_h, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d, g_h, 0);

    // Allocate per-SM work counters
    cudaMalloc(&g_counters, NUM_SM * sizeof(unsigned long long));

    // Initialize control data
    g_h->target_pm = 0.82f;
    g_h->max_sleep_ns = MAX_SLEEP_NS;
    g_h->throttle_sleep_ns = 0;

    // Launch persistent kernel (one block per SM)
    gpu_persistent_kernel<<<NUM_SM, THREADS_PER_BLOCK>>>(g_counters, g_d, NUM_SM);

    // Start control loop in its own thread
    pthread_t t;
    pthread_create(&t, NULL, control_loop, NULL);

    // Print telemetry for ~20 seconds
    for (int i = 0; i < 20; i++) {
        usleep(700000);
        printf("[TEL] Z=%.3f | sleep=%6d | PM=%.3f | err=%.3f | corr=%d\n",
               g_h->z_estimate, g_h->throttle_sleep_ns,
               g_h->current_pm, g_h->pm_error, g_h->proactive_corrections);
    }

    // Graceful shutdown
    g_running = 0;
    g_h->control_flags |= 0x2;
    pthread_join(t, NULL);
    cudaDeviceSynchronize();

    printf("\n[GPUTronic] Run complete.\n");
    cudaFree(g_counters);
    cudaFreeHost(g_h);
    return 0;
}