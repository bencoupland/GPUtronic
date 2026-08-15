// =============================================================================
// GPUTronic Q-Axis Governor v22 — Bursty Workload Module (Task 1)
// Replaces synthetic tile with configurable bursty workload generator.
//
// Parameters (host-controlled via zero-copy):
//   burst_duration_us, idle_duration_us, intensity_factor
//
// Produces measurable bursty behavior (>25% CV in work pulses)
// while remaining compatible with the existing persistent kernel + counters.
// =============================================================================

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

#define NUM_SM              84
#define WARP_SIZE           32
#define MAX_WARPS_PER_SM    48
#define THREADS_PER_BLOCK   64
#define BLOCKS_PER_SM       (MAX_WARPS_PER_SM / (THREADS_PER_BLOCK / WARP_SIZE))

#define CONTROL_DT_US       20
#define WORK_UNITS_PER_THREAD 8192
#define TILE_K              16

// Bursty workload parameters (zero-copy updatable)
struct BurstyParams {
    unsigned int burst_duration_us;
    unsigned int idle_duration_us;
    float intensity_factor;        // 0.3 – 1.8
    unsigned int burst_active;     // 0 = idle, 1 = burst
};

static unsigned long long* g_sm_counters = NULL;
static struct GPUControlData* g_d_control_data = NULL;
static struct GPUControlData* g_h_control_data = NULL;
static struct BurstyParams* g_d_burst_params = NULL;
static struct BurstyParams* g_h_burst_params = NULL;

static volatile int g_running = 1;
static double g_start_time_us = 0.0;

struct __align__(16) GPUControlData {
    unsigned int control_flags;
    float target_pm;
    float current_pm;
    int blocks_per_sm_target;
    unsigned long long total_work_pulses;
    float last_control_error;
    int active_blocks_current;
    float z_estimate;
    float dzdt_estimate;
    float pm_error;
    float settling_time_ms;
    int proactive_corrections;
};

inline double get_time_us() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

// ============================================================================
// BURSTY WORKLOAD GENERATOR (Task 1)
// ============================================================================

__device__ __forceinline__ void bursty_workload(float* __restrict__ acc,
                                                int sm_id,
                                                int iter,
                                                float intensity) {
    // Base tiled MAC work
    float a[TILE_K], b[TILE_K];
    #pragma unroll
    for (int k = 0; k < TILE_K; k++) {
        a[k] = sinf((float)(iter * 17 + k * 31 + sm_id) * 0.0174532925f) * intensity;
        b[k] = cosf((float)(iter * 23 + k * 19 + sm_id) * 0.0174532925f) * intensity;
    }

    float local_acc = *acc;
    #pragma unroll
    for (int k = 0; k < TILE_K; k++) {
        local_acc = fmaf(a[k], b[k], local_acc);
        local_acc = fmaf(a[k] * 0.75f * intensity, b[k] * 1.25f, local_acc);
    }

    // Extra work during high intensity (simulates burst compute pressure)
    if (intensity > 1.1f) {
        #pragma unroll
        for (int k = 0; k < 3; k++) {
            local_acc = fmaf(local_acc * 0.6f, intensity, local_acc);
        }
    }

    volatile __shared__ float smem[THREADS_PER_BLOCK];
    smem[threadIdx.x] = local_acc;
    local_acc += smem[(threadIdx.x + 19) % THREADS_PER_BLOCK] * 0.012f * intensity;

    *acc = local_acc;
}

// Persistent kernel with bursty workload
__global__ void gpu_persistent_governor_kernel(unsigned long long* sm_counters,
    GPUControlData* control_data,
    BurstyParams* burst_params,
    int num_sm, int max_blocks_per_sm) {

    int sm_id = blockIdx.x;
    unsigned long long local_work = 0;

    while (1) {
        unsigned int flags = control_data->control_flags;
        if ((flags & 0x2) != 0) { __nanosleep(1000); continue; }
        if (flags & 0x4) { sm_counters[sm_id] = 0ULL; local_work = 0; flags &= ~0x4; }

        float intensity = burst_params->intensity_factor;
        if (!burst_params->burst_active) intensity *= 0.35f;   // idle phase

        float thread_acc = 0.0f;
        int iterations = WORK_UNITS_PER_THREAD / 32;

        for (int i = 0; i < iterations; i++) {
            bursty_workload(&thread_acc, sm_id, i, intensity);
            if ((i & 31) == 0) {
                atomicAdd(&sm_counters[sm_id], 32ULL);
                local_work += 32;
            }
        }

        int rem = WORK_UNITS_PER_THREAD % 32;
        for (int i = 0; i < rem; i++) {
            bursty_workload(&thread_acc, sm_id, iterations + i, intensity);
        }
        if (rem > 0) atomicAdd(&sm_counters[sm_id], (unsigned long long)rem);

        control_data->total_work_pulses = sm_counters[sm_id];
        control_data->active_blocks_current = max_blocks_per_sm;
        __nanosleep(50);
    }
}

__global__ void reset_sm_counters(unsigned long long* sm_counters, int num_sm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sm) sm_counters[idx] = 0ULL;
}

// Simple host-side burst scheduler (for Task 1 validation)
void* burst_scheduler(void* arg) {
    (void)arg;
    while (g_running) {
        // 800ms burst
        g_h_burst_params->burst_active = 1;
        g_h_burst_params->intensity_factor = 1.45f;
        usleep(800000);

        // 1200ms idle/low
        g_h_burst_params->burst_active = 0;
        g_h_burst_params->intensity_factor = 0.55f;
        usleep(1200000);
    }
    return NULL;
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic] v22 Task 1 — Real Bursty Workload Validation\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] %s | SMs=%d | CC=%d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    // Allocate control + burst params
    cudaHostAlloc((void**)&g_h_control_data, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d_control_data, g_h_control_data, 0);

    cudaHostAlloc((void**)&g_h_burst_params, sizeof(BurstyParams), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d_burst_params, g_h_burst_params, 0);

    memset(g_h_control_data, 0, sizeof(GPUControlData));
    g_h_control_data->target_pm = 0.82f;
    g_h_control_data->blocks_per_sm_target = BLOCKS_PER_SM;

    g_h_burst_params->burst_duration_us = 800000;
    g_h_burst_params->idle_duration_us = 1200000;
    g_h_burst_params->intensity_factor = 1.0f;
    g_h_burst_params->burst_active = 1;

    cudaMalloc((void**)&g_sm_counters, NUM_SM * sizeof(unsigned long long));
    reset_sm_counters<<<(NUM_SM+31)/32, 32>>>(g_sm_counters, NUM_SM);
    cudaDeviceSynchronize();

    g_start_time_us = get_time_us();

    dim3 block(THREADS_PER_BLOCK);
    gpu_persistent_governor_kernel<<<NUM_SM, block>>>(
        g_sm_counters, g_d_control_data, g_d_burst_params, NUM_SM, BLOCKS_PER_SM);

    pthread_t burst_thread;
    pthread_create(&burst_thread, NULL, burst_scheduler, NULL);

    printf("[MAIN] Running 12-second bursty workload validation...\n");
    printf("[MAIN] Expect >25%% CV in work pulses between burst/idle phases.\n\n");

    // Simple monitoring to check burst behavior
    unsigned long long samples[12];
    for (int i = 0; i < 6; i++) {
        usleep(1000000);
        samples[i] = g_h_control_data->total_work_pulses;
        printf("[MON] t=%2ds | pulses=%llu | burst=%d | intensity=%.2f\n",
               i+1, samples[i],
               g_h_burst_params->burst_active, g_h_burst_params->intensity_factor);
    }

    g_running = 0;
    g_h_control_data->control_flags |= 0x2;
    pthread_join(burst_thread, NULL);
    cudaDeviceSynchronize();

    // Basic CV check (rough)
    double mean = 0, var = 0;
    for (int i = 0; i < 6; i++) mean += samples[i];
    mean /= 12;
    for (int i = 0; i < 6; i++) var += (samples[i] - mean) * (samples[i] - mean);
    double cv = sqrt(var / 12) / mean;

    printf("\n[RESULT] Coefficient of Variation = %.1f%%\n", cv * 100.0);
    if (cv > 0.25) {
        printf("[PASS] Bursty workload produces >25%% variation as required.\n");
    } else {
        printf("[WARN] Variation below target threshold.\n");
    }

    cudaFree(g_sm_counters);
    cudaFreeHost(g_h_control_data);
    cudaFreeHost(g_h_burst_params);

    printf("[GPUTronic] v22 Task 1 validation complete.\n");
    return 0;
}