// =============================================================================
// GPUTronic Q-Axis Governor v20 — True Phase-Margin PI Controller
// Direct PM error control + Z/dZ/dt feed-forward + Step-response test
//
// Primary error signal: target_pm - current_pm
// Feed-forward: predictive damping using Z and dZ/dt
// Test: Step changes 0.70 → 0.85 → 0.60 with settling time logging
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

#define CONTROL_DT_US       80          // ~12.5 kHz base (we can push later)
#define WORK_UNITS_PER_THREAD 8192
#define TILE_K              16

// PI Gains for direct PM control
#define KP                  0.52f
#define KI                  0.09f
#define INTEGRAL_MAX        1.8f
#define DEADBAND            0.025f

// Feed-forward gains
#define FF_Z_GAIN           0.035f
#define FF_DZDT_GAIN        0.18f

// Kalman
#define KALMAN_Q_PROC       0.00065f
#define KALMAN_R_MEAS       0.028f

// Step test schedule
#define STEP_DURATION_US    6000000ULL   // 6 seconds per step
#define STEP_1_PM           0.70f
#define STEP_2_PM           0.85f
#define STEP_3_PM           0.60f

static unsigned long long* g_sm_counters = NULL;
static struct GPUControlData* g_d_control_data = NULL;
static struct GPUControlData* g_h_control_data = NULL;

static volatile int g_running = 1;
static double g_start_time_us = 0.0;
static double g_measured_max_rate = 65000000.0;

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
};

inline double get_time_us() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

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
        local_acc = fmaf(a[k]*0.7f, b[k]*1.3f, local_acc);
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

// 2-state Kalman
void kalman_update(float* z, float* dzdt, float measured_z, float dt) {
    float z_pred = *z + *dzdt * dt;
    float dzdt_pred = *dzdt;
    static float P[2][2] = {{0.1f, 0.0f}, {0.0f, 0.05f}};
    P[0][0] += KALMAN_Q_PROC;
    P[1][1] += KALMAN_Q_PROC * 0.6f;
    float y = measured_z - z_pred;
    float S = P[0][0] + KALMAN_R_MEAS;
    float K0 = P[0][0] / S;
    float K1 = P[1][0] / S;
    *z    = z_pred + K0 * y;
    *dzdt = dzdt_pred + K1 * y;
    P[0][0] = (1.0f - K0) * P[0][0];
    P[1][0] = (1.0f - K0) * P[1][0];
    P[0][1] = P[1][0];
    P[1][1] = (1.0f - K1) * P[1][1];
}

float compute_pm(float z, float dzdt) {
    float pm = 1.0f / (1.0f + z * 0.58f + fabsf(dzdt) * 2.8f);
    if (pm < 0.32f) pm = 0.32f;
    if (pm > 0.98f) pm = 0.98f;
    return pm;
}

void* control_thread(void* arg) {
    (void)arg;
    float integral = 0.0f;
    unsigned long long last_pulses = 0;
    double last_time = get_time_us();
    float z = 1.08f, dzdt = 0.0f;
    int current_step = 0;
    double next_step_time = get_time_us() + STEP_DURATION_US;
    double step_start_time = get_time_us();
    float last_target = STEP_1_PM;

    g_h_control_data->target_pm = STEP_1_PM;

    printf("[PM] True Phase-Margin PI + Step Test active\n");
    printf("[PM] Steps: 0.70 → 0.85 → 0.60 (6s each)\n\n");

    while (g_running) {
        double now = get_time_us();
        double dt = (now - last_time) / 1e6f;
        if (dt < 0.00006f) { usleep(30); continue; }

        // Step scheduler
        if (now > next_step_time) {
            current_step = (current_step + 1) % 3;
            float new_target;
            if (current_step == 0) new_target = STEP_1_PM;
            else if (current_step == 1) new_target = STEP_2_PM;
            else new_target = STEP_3_PM;

            g_h_control_data->target_pm = new_target;
            step_start_time = now;
            last_target = new_target;
            printf("[STEP] t=%.1fs → target_pm = %.2f\n", (now - g_start_time_us)/1e6f, new_target);
            next_step_time = now + STEP_DURATION_US;
        }

        unsigned long long current = g_h_control_data->total_work_pulses;
        double delta = (double)(current - last_pulses); if (delta < 0) delta = 0;
        double measured_rate = delta / dt;

        if (now - g_start_time_us < 3.5e6 && measured_rate > g_measured_max_rate * 0.9 && measured_rate < 300e6)
            g_measured_max_rate = measured_rate;

        float normalized = (float)(measured_rate / g_measured_max_rate);
        if (normalized > 1.0f) normalized = 1.0f;

        // Kalman Z
        float measured_z = (normalized > 0.035f) ? (1.0f / normalized) : 2.7f;
        kalman_update(&z, &dzdt, measured_z, (float)dt);
        if (z < 0.5f) z = 0.5f; if (z > 3.0f) z = 3.0f;

        float current_pm = compute_pm(z, dzdt);
        g_h_control_data->z_estimate = z;
        g_h_control_data->dzdt_estimate = dzdt;
        g_h_control_data->current_pm = current_pm;

        // === TRUE PHASE-MARGIN PI ===
        float target_pm = g_h_control_data->target_pm;
        float pm_error = target_pm - current_pm;

        // Feed-forward from Z and dZ/dt
        float ff = (z - 1.0f) * FF_Z_GAIN + dzdt * FF_DZDT_GAIN;
        pm_error += ff;

        if (fabsf(pm_error) < DEADBAND) pm_error = 0.0f;

        integral += pm_error * dt;
        if (integral > INTEGRAL_MAX) integral = INTEGRAL_MAX;
        if (integral < -INTEGRAL_MAX) integral = -INTEGRAL_MAX;

        float pi_delta = -(KP * pm_error + KI * integral);
        float new_pm = target_pm + pi_delta;

        if (new_pm < 0.52f) new_pm = 0.52f;
        if (new_pm > 0.96f) new_pm = 0.96f;

        g_h_control_data->target_pm = new_pm;
        g_h_control_data->pm_error = pm_error;
        g_h_control_data->last_control_error = pm_error;

        // Simple settling detection (within 0.03 of target for >300ms)
        static double stable_start = 0;
        if (fabsf(pm_error) < 0.03f) {
            if (stable_start == 0) stable_start = now;
        } else {
            stable_start = 0;
        }
        float settle_ms = (stable_start > 0) ? (float)((now - stable_start)/1000.0) : -1.0f;
        g_h_control_data->settling_time_ms = settle_ms;

        // Logging
        static int tick = 0;
        if ((tick++ % 8) == 0) {
            printf("[PM] t=%.1fs | tgt=%.3f | cur=%.3f | err=%.4f | Z=%.3f | dZ=%.4f | settle=%.0fms\n",
                   (now - g_start_time_us)/1e6f, new_pm, current_pm, pm_error, z, dzdt, settle_ms);
        }

        last_pulses = current;
        last_time = now;
        usleep(CONTROL_DT_US);
    }
    return NULL;
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic] Q-Axis Governor v20 — True PM PI + Step Test\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] %s | SMs=%d | CC=%d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    cudaHostAlloc((void**)&g_h_control_data, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d_control_data, g_h_control_data, 0);
    memset(g_h_control_data, 0, sizeof(GPUControlData));
    g_h_control_data->target_pm = STEP_1_PM;
    g_h_control_data->blocks_per_sm_target = BLOCKS_PER_SM;

    cudaMalloc((void**)&g_sm_counters, NUM_SM * sizeof(unsigned long long));
    reset_sm_counters<<<(NUM_SM+31)/32, 32>>>(g_sm_counters, NUM_SM);
    cudaDeviceSynchronize();

    g_start_time_us = get_time_us();

    dim3 block(THREADS_PER_BLOCK);
    gpu_persistent_governor_kernel<<<NUM_SM, block>>>(
        g_sm_counters, g_d_control_data, NUM_SM, BLOCKS_PER_SM);

    pthread_t ctrl_thread;
    pthread_create(&ctrl_thread, NULL, control_thread, NULL);

    printf("[MAIN] Running 22-second step-response test...\n\n");

    while (g_running) {
        usleep(350000);
        if (get_time_us() - g_start_time_us > 22e6) g_running = 0;
    }

    g_h_control_data->control_flags |= 0x2;
    g_running = 0;
    pthread_join(ctrl_thread, NULL);
    cudaDeviceSynchronize();

    cudaFree(g_sm_counters);
    cudaFreeHost(g_h_control_data);

    printf("\n[GPUTronic] v20 True PM-PI test complete.\n");
    return 0;
}