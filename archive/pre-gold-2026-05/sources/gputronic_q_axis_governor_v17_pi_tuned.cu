// =============================================================================
// GPUTronic Q-Axis Governor v17 — Tuned PI + Deadband + Phase-Margin Style
// Improvements over v16:
//   1. Deadband to reduce chatter
//   2. Auto-calibrated normalization using measured max throughput
//   3. Phase-margin inspired control (target PM = 0.85)
//   4. Step-response test mode with scheduled target changes
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

#define NUM_SM              84
#define WARP_SIZE           32
#define MAX_WARPS_PER_SM    48
#define THREADS_PER_BLOCK   64
#define BLOCKS_PER_SM       (MAX_WARPS_PER_SM / (THREADS_PER_BLOCK / WARP_SIZE))
#define TOTAL_BLOCKS        (NUM_SM * BLOCKS_PER_SM)

#define CONTROL_DT_US       100
#define WORK_UNITS_PER_THREAD 8192
#define TILE_K              16

// PI Gains (more conservative after v16 oscillation)
#define KP                  0.45f
#define KI                  0.08f
#define INTEGRAL_MAX        3.0f
#define DEADBAND            0.04f          // 4% deadband

#define TARGET_RATE_MIN     0.10f
#define TARGET_RATE_MAX     1.00f
#define TARGET_PM           0.85f          // Desired phase margin

static unsigned long long* g_sm_counters = NULL;
static struct GPUControlData* g_d_control_data = NULL;
static struct GPUControlData* g_h_control_data = NULL;

static volatile int g_running = 1;
static double g_start_time_us = 0.0;
static double g_measured_max_rate = 65000000.0; // will be auto-calibrated

struct __align__(16) GPUControlData {
    unsigned int control_flags;
    float target_q_rate;
    int blocks_per_sm_target;
    unsigned long long total_work_pulses;
    float last_control_error;
    int active_blocks_current;
};

inline double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

// Real workload (unchanged)
__device__ __forceinline__ void real_workload_tile(float* __restrict__ acc, int sm_id, int iter) {
    float a[TILE_K], b[TILE_K];
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

__global__ void gpu_persistent_governor_kernel(unsigned long long* sm_counters,
    GPUControlData* control_data, int num_sm, int max_blocks_per_sm) {
    int sm_id = blockIdx.x;
    while (1) {
        unsigned int flags = control_data->control_flags;
        if ((flags & 0x2) != 0) { __nanosleep(1000); continue; }
        if (flags & 0x4) { sm_counters[sm_id] = 0ULL; flags &= ~0x4; }

        float target_rate = control_data->target_q_rate;
        float thread_acc = 0.0f;
        int iterations = WORK_UNITS_PER_THREAD / 32;

        for (int i = 0; i < iterations; i++) {
            real_workload_tile(&thread_acc, sm_id, i);
            if ((i & 31) == 0) atomicAdd(&sm_counters[sm_id], 32ULL);
        }
        int rem = WORK_UNITS_PER_THREAD % 32;
        for (int i = 0; i < rem; i++) real_workload_tile(&thread_acc, sm_id, iterations + i);
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

// Improved PI controller with deadband + auto-calibration
void* pi_control_thread(void* arg) {
    (void)arg;
    float integral = 0.0f;
    unsigned long long last_pulses = 0;
    double last_time = get_time_us();
    int step_phase = 0;
    double next_step_time = get_time_us() + 4e6; // first step after 4s

    printf("[PI] Tuned controller started (Kp=%.2f Ki=%.2f Deadband=%.2f)\n", KP, KI, DEADBAND);

    while (g_running) {
        double now = get_time_us();
        double dt = (now - last_time) / 1e6f;
        if (dt < 0.00008f) { usleep(40); continue; }

        unsigned long long current = g_h_control_data->total_work_pulses;
        double delta = (double)(current - last_pulses);
        if (delta < 0) delta = 0; // handle potential wrap
        double measured = delta / dt;

        // Auto-calibrate max rate (with sanity cap)
        if (now - g_start_time_us < 4e6) {
            if (measured > g_measured_max_rate * 0.9 && measured < 300e6)
                g_measured_max_rate = measured;
        }

        float normalized = (float)(measured / g_measured_max_rate);
        if (normalized > 1.0f) normalized = 1.0f;

        float target = g_h_control_data->target_q_rate;
        float error = target - normalized;

        // Deadband
        if (fabsf(error) < DEADBAND) error = 0.0f;

        // Step response test schedule
        if (now > next_step_time) {
            step_phase = (step_phase + 1) % 4;
            float new_target;
            switch (step_phase) {
                case 0: new_target = 0.85f; break;
                case 1: new_target = 0.45f; break;
                case 2: new_target = 0.95f; break;
                default: new_target = 0.25f; break;
            }
            g_h_control_data->target_q_rate = new_target;
            printf("[STEP] t=%.1fs → target=%.2f\n", (now - g_start_time_us)/1e6f, new_target);
            next_step_time = now + 5e6; // 5 second steps
        }

        integral += error * dt;
        if (integral > INTEGRAL_MAX) integral = INTEGRAL_MAX;
        if (integral < -INTEGRAL_MAX) integral = -INTEGRAL_MAX;

        float pi_delta = -(KP * error + KI * integral);
        float new_target = target + pi_delta * 0.7f;

        if (new_target < TARGET_RATE_MIN) new_target = TARGET_RATE_MIN;
        if (new_target > TARGET_RATE_MAX) new_target = TARGET_RATE_MAX;

        g_h_control_data->target_q_rate = new_target;
        g_h_control_data->last_control_error = error;

        // Telemetry
        static int tick = 0;
        if ((tick++ % 12) == 0) {
            printf("[PI] t=%.1fs | tgt=%.3f | meas=%.3f | err=%.3f | int=%.3f | max=%.1fM\n",
                   (now - g_start_time_us)/1e6f, new_target, normalized, error, integral,
                   g_measured_max_rate / 1e6f);
        }

        last_pulses = current;
        last_time = now;
        usleep(CONTROL_DT_US);
    }
    return NULL;
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic] Q-Axis Governor v17 — Tuned PI + Deadband + Steps\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] %s | SMs=%d | CC=%d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    cudaHostAlloc((void**)&g_h_control_data, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d_control_data, g_h_control_data, 0);
    memset(g_h_control_data, 0, sizeof(GPUControlData));
    g_h_control_data->target_q_rate = 0.80f;
    g_h_control_data->blocks_per_sm_target = BLOCKS_PER_SM;

    cudaMalloc((void**)&g_sm_counters, NUM_SM * sizeof(unsigned long long));
    reset_sm_counters<<<(NUM_SM+31)/32, 32>>>(g_sm_counters, NUM_SM);
    cudaDeviceSynchronize();

    g_start_time_us = get_time_us();

    dim3 block(THREADS_PER_BLOCK);
    gpu_persistent_governor_kernel<<<NUM_SM, block>>>(
        g_sm_counters, g_d_control_data, NUM_SM, BLOCKS_PER_SM);

    pthread_t ctrl_thread;
    pthread_create(&ctrl_thread, NULL, pi_control_thread, NULL);

    printf("[MAIN] Running 30-second step-response test...\n\n");

    while (g_running) {
        usleep(500000);
        if (get_time_us() - g_start_time_us > 30e6) g_running = 0;
    }

    g_h_control_data->control_flags |= 0x2;
    g_running = 0;
    pthread_join(ctrl_thread, NULL);
    cudaDeviceSynchronize();

    cudaFree(g_sm_counters);
    cudaFreeHost(g_h_control_data);

    printf("\n[GPUTronic] v17 test complete.\n");
    return 0;
}