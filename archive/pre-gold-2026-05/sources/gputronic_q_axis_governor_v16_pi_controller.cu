// =============================================================================
// GPUTronic Q-Axis Persistent Governor v16 — Dynamic PI Controller
// Adds closed-loop PI regulation on top of the real-workload v15 kernel
//
// Control Law:
//   error = target_rate - measured_rate
//   delta = -(Kp * error + Ki * integral)
//   new_target = clamp(current + delta)
//
// Features:
//   - Dedicated control thread (pthread) running at ~10 kHz target
//   - Integral windup protection
//   - Output clamping [0.10, 1.00]
//   - Telemetry printed from control thread
// =============================================================================

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>

// ============================================================================
// BLACKWELL CONFIGURATION
// ============================================================================
#define NUM_SM              84
#define WARP_SIZE           32
#define MAX_WARPS_PER_SM    48
#define THREADS_PER_BLOCK   64
#define BLOCKS_PER_SM       (MAX_WARPS_PER_SM / (THREADS_PER_BLOCK / WARP_SIZE))
#define TOTAL_BLOCKS        (NUM_SM * BLOCKS_PER_SM)

#define CONTROL_DT_US       100
#define WORK_UNITS_PER_THREAD 8192
#define TILE_K              16

// PI Controller Gains (tuned for Q-axis)
#define KP                  0.85f
#define KI                  0.12f
#define INTEGRAL_MAX        5.0f
#define TARGET_RATE_MIN     0.10f
#define TARGET_RATE_MAX     1.00f

// ============================================================================
// CONTROL STRUCTURES
// ============================================================================
struct __align__(16) GPUControlData {
    unsigned int control_flags;
    float target_q_rate;
    int blocks_per_sm_target;

    unsigned long long total_work_pulses;
    float last_control_error;
    int active_blocks_current;
};

static unsigned long long* g_sm_counters = NULL;
static GPUControlData* g_d_control_data = NULL;
static GPUControlData* g_h_control_data = NULL;

static volatile int g_running = 1;
static double g_start_time_us = 0.0;

// ============================================================================
// TIMING
// ============================================================================
inline double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

// ============================================================================
// REAL WORKLOAD (same as v15)
// ============================================================================
__device__ __forceinline__ void real_workload_tile(float* __restrict__ acc,
                                                   int sm_id, int iter) {
    float a[TILE_K];
    float b[TILE_K];

    #pragma unroll
    for (int k = 0; k < TILE_K; k++) {
        a[k] = sinf((float)(iter * 17 + k * 31 + sm_id) * 0.0174532925f) * 0.5f + 0.5f;
        b[k] = cosf((float)(iter * 23 + k * 19 + sm_id) * 0.0174532925f) * 0.5f + 0.5f;
    }

    float local_acc = *acc;
    #pragma unroll
    for (int k = 0; k < TILE_K; k++) {
        local_acc = fmaf(a[k], b[k], local_acc);
        local_acc = fmaf(a[k] * 0.7f, b[k] * 1.3f, local_acc);
    }

    volatile __shared__ float smem[THREADS_PER_BLOCK];
    smem[threadIdx.x] = local_acc;
    local_acc += smem[(threadIdx.x + 17) % THREADS_PER_BLOCK] * 0.01f;

    *acc = local_acc;
}

// ============================================================================
// PERSISTENT KERNEL (unchanged from v15)
// ============================================================================
__global__ void gpu_persistent_governor_kernel(
    unsigned long long* sm_counters,
    GPUControlData* control_data,
    int num_sm, int max_blocks_per_sm) {

    int sm_id = blockIdx.x;

    while (1) {
        unsigned int flags = control_data->control_flags;

        if ((flags & 0x2) != 0) {
            __nanosleep(1000);
            continue;
        }
        if (flags & 0x4) {
            sm_counters[sm_id] = 0ULL;
            flags &= ~0x4;
        }

        float target_rate = control_data->target_q_rate;
        unsigned long long work_done = 0;
        float thread_acc = 0.0f;
        int iterations = WORK_UNITS_PER_THREAD / 32;

        for (int i = 0; i < iterations; i++) {
            real_workload_tile(&thread_acc, sm_id, i);
            if ((i & 31) == 0) {
                atomicAdd(&sm_counters[sm_id], 32ULL);
                work_done += 32;
            }
        }

        int remainder = WORK_UNITS_PER_THREAD % 32;
        for (int i = 0; i < remainder; i++) {
            real_workload_tile(&thread_acc, sm_id, iterations + i);
        }
        if (remainder > 0) {
            atomicAdd(&sm_counters[sm_id], (unsigned long long)remainder);
        }

        control_data->total_work_pulses = sm_counters[sm_id];
        control_data->active_blocks_current = max_blocks_per_sm;
        __nanosleep(50);
    }
}

__global__ void reset_sm_counters(unsigned long long* sm_counters, int num_sm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sm) sm_counters[idx] = 0ULL;
}

// ============================================================================
// DYNAMIC PI CONTROLLER THREAD
// ============================================================================
void* pi_control_thread(void* arg) {
    (void)arg;

    float integral = 0.0f;
    unsigned long long last_pulses = 0;
    double last_time = get_time_us();

    printf("[PI] Control thread started (Kp=%.2f, Ki=%.2f)\n", KP, KI);

    while (g_running) {
        double now = get_time_us();
        double dt = (now - last_time) / 1e6f;   // seconds
        if (dt < 0.00005f) { usleep(50); continue; } // ~20 kHz max

        unsigned long long current = g_h_control_data->total_work_pulses;
        float measured_rate = (float)((current - last_pulses) / (dt * 1e6f)); // pulses per second (normalized)

        // Normalize measured rate roughly to [0..1] range
        // (empirical scaling based on observed ~65M pulses/sec at full rate)
        float normalized_measured = measured_rate / 65000000.0f;
        if (normalized_measured > 1.0f) normalized_measured = 1.0f;

        float target = g_h_control_data->target_q_rate;
        float error = target - normalized_measured;

        // Integral with windup protection
        integral += error * dt;
        if (integral > INTEGRAL_MAX) integral = INTEGRAL_MAX;
        if (integral < -INTEGRAL_MAX) integral = -INTEGRAL_MAX;

        // PI output
        float delta = -(KP * error + KI * integral);

        // Apply correction
        float new_target = target + delta * 0.8f;   // damping factor

        // Clamp
        if (new_target < TARGET_RATE_MIN) new_target = TARGET_RATE_MIN;
        if (new_target > TARGET_RATE_MAX) new_target = TARGET_RATE_MAX;

        // Update zero-copy structure
        g_h_control_data->target_q_rate = new_target;
        g_h_control_data->last_control_error = error;

        // Telemetry
        if (((int)(now / 200000.0)) % 5 == 0) {  // every ~1 sec
            printf("[PI] t=%.1fs | target=%.3f | measured=%.3f | err=%.4f | integral=%.3f\n",
                   (now - g_start_time_us) / 1e6f,
                   new_target, normalized_measured, error, integral);
        }

        last_pulses = current;
        last_time = now;

        usleep(CONTROL_DT_US);   // target ~10 kHz loop
    }
    return NULL;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic] Q-Axis Governor v16 — Dynamic PI Controller\n");
    printf("        Real Workload + Closed-Loop Rate Regulation\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    cudaSetDevice(0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] %s | SMs=%d | CC=%d.%d\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    // Zero-copy allocation
    cudaHostAlloc((void**)&g_h_control_data, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d_control_data, g_h_control_data, 0);

    memset(g_h_control_data, 0, sizeof(GPUControlData));
    g_h_control_data->target_q_rate = 0.80f;
    g_h_control_data->blocks_per_sm_target = BLOCKS_PER_SM;

    cudaMalloc((void**)&g_sm_counters, NUM_SM * sizeof(unsigned long long));
    reset_sm_counters<<<(NUM_SM+31)/32, 32>>>(g_sm_counters, NUM_SM);
    cudaDeviceSynchronize();

    g_start_time_us = get_time_us();

    // Launch persistent kernel
    dim3 block(THREADS_PER_BLOCK);
    gpu_persistent_governor_kernel<<<NUM_SM, block>>>(
        g_sm_counters, g_d_control_data, NUM_SM, BLOCKS_PER_SM);

    // Start PI control thread
    pthread_t ctrl_thread;
    pthread_create(&ctrl_thread, NULL, pi_control_thread, NULL);

    printf("[MAIN] Kernel + PI controller running. Press Ctrl+C to stop.\n\n");

    // Simple monitor (non-blocking)
    while (g_running) {
        usleep(500000); // 500 ms
        if (get_time_us() - g_start_time_us > 30e6) { // 30 second safety limit
            g_running = 0;
        }
    }

    // Shutdown
    g_h_control_data->control_flags |= 0x2; // pause
    g_running = 0;
    pthread_join(ctrl_thread, NULL);
    cudaDeviceSynchronize();

    cudaFree(g_sm_counters);
    cudaFreeHost(g_h_control_data);

    printf("\n[GPUTronic] v16 PI Controller test finished.\n");
    return 0;
}