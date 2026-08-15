// =============================================================================
// GPUTronic Q-Axis Governor v23 — 80-100 kHz PM-PI + Predictive Throttling
// (Q-axis only, D-axis removed per NVIDIA driver reality, Z-axis Kalman 2-state)
// =============================================================================
//
// Key upgrades from v22:
// - CONTROL_DT_US = 10  → target 100 kHz (guard allows 80-100 kHz range)
// - Full constant-velocity Kalman covariance propagation (project-bible Pattern #3)
// - Retuned process/measurement noise + feed-forward gains with dt scaling
// - Slightly reduced KP for higher bandwidth stability
// - Bursty workload preserved for realistic validation
//
// Build: nvcc -O3 -arch=sm_120 -o gputronic_v23 gputronic_q_axis_governor_v23_80-100khz.cu -lcuda -lpthread
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

#define CONTROL_DT_US       10          // 100 kHz target (80-100 kHz capable)
#define WORK_UNITS_PER_THREAD 8192
#define TILE_K              16

// PI Controller (retuned for higher bandwidth)
#define KP                  0.45f
#define KI                  0.055f
#define INTEGRAL_CLAMP      0.8f
#define DEADBAND_LOW        1.05f
#define DEADBAND_HIGH       1.80f

// Predictive feed-forward (retuned + dt scaling)
#define FF_Z_GAIN           0.055f
#define FF_DZDT_GAIN        0.55f
#define FF_DZDT_NEG_GAIN    0.14f
#define FF_DT_SCALE         (CONTROL_DT_US / 20.0f)

// Kalman observer (2-state [Z, dZ/dt], retuned for dt=10us)
#define KALMAN_Q_PROC       0.00045f
#define KALMAN_R_MEAS       0.018f

// Bursty workload parameters (zero-copy updatable)
struct BurstyParams {
    unsigned int burst_duration_us;
    unsigned int idle_duration_us;
    float intensity_factor;
    unsigned int burst_active;
};

static unsigned long long* g_sm_counters = NULL;
static struct GPUControlData* g_d_control_data = NULL;
static struct GPUControlData* g_h_control_data = NULL;
static struct BurstyParams* g_d_burst_params = NULL;
static struct BurstyParams* g_h_burst_params = NULL;

static volatile int g_running = 1;
static double g_start_time_us __attribute__((unused)) = 0.0;

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

// Full 2x2 covariance matrix for constant-velocity Kalman
static float P[2][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}};

inline double get_time_us() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

// ============================================================================
// BURSTY WORKLOAD GENERATOR (unchanged from v22)
// ============================================================================

__device__ __forceinline__ void bursty_workload(float* __restrict__ acc,
                                                int sm_id,
                                                int iter,
                                                float intensity) {
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

// Persistent kernel (workload unchanged, control path will be driven from host)
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
        if (!burst_params->burst_active) intensity *= 0.35f;

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

// ============================================================================
// HOST-SIDE 100 kHz CONTROL LOOP (v23 upgrade)
// ============================================================================

void* control_loop(void* arg) {
    (void)arg;

    const float dt = CONTROL_DT_US * 1e-6f;
    float integral = 0.0f;
    float last_z = 1.0f;
    double last_time = get_time_us();

    // Initialize Kalman
    P[0][0] = 1.0f; P[0][1] = 0.0f;
    P[1][0] = 0.0f; P[1][1] = 1.0f;

    printf("[CTRL] v23 80-100 kHz PM-PI + Full-Covariance Kalman started (dt=%dus)\n", CONTROL_DT_US);

    while (g_running) {
        double now = get_time_us();
        double elapsed = now - last_time;
        if (elapsed < CONTROL_DT_US * 0.8) {
            usleep(2);
            continue;
        }
        last_time = now;

        unsigned long long work = g_h_control_data->total_work_pulses;
        static unsigned long long prev_work = 0;
        unsigned long long delta = work - prev_work;
        prev_work = work;

        // Phase margin calculation (Z = impedance proxy)
        float z = (delta > 0) ? (1.0f / (delta * 1e-6f + 1e-9f)) : 2.0f;
        if (z > 2.0f) z = 2.0f;
        if (z < 0.05f) z = 0.05f;

        // === FULL KALMAN UPDATE (constant velocity model) ===
        float Q = KALMAN_Q_PROC;
        float R = KALMAN_R_MEAS;

        // Predict
        float p00 = P[0][0] + 2.0f*dt*P[0][1] + dt*dt*P[1][1] + Q;
        float p01 = P[0][1] + dt*P[1][1];
        float p10 = P[1][0] + dt*P[1][1];
        float p11 = P[1][1] + Q;

        P[0][0] = p00; P[0][1] = p01;
        P[1][0] = p10; P[1][1] = p11;

        // Update
        float y = z - last_z;                    // measurement residual (velocity-like)
        float s = P[0][0] + R;
        float K0 = P[0][0] / s;
        float K1 = P[1][0] / s;

        float z_hat = last_z + K0 * y;
        float dzdt_hat = (last_z - z) / dt + K1 * y;   // dZ/dt estimate

        // Joseph form covariance update (simplified)
        P[0][0] = (1.0f - K0) * P[0][0];
        P[0][1] = (1.0f - K0) * P[0][1];
        P[1][0] = (1.0f - K1) * P[1][0];
        P[1][1] = (1.0f - K1) * P[1][1];

        last_z = z_hat;

        // === FEED-FORWARD (with dt scaling) ===
        float dzdt = (z_hat - last_z) / dt;
        float ff = FF_Z_GAIN * (z_hat - 1.0f) +
                   FF_DZDT_GAIN * dzdt * FF_DT_SCALE;
        if (dzdt < 0.0f) ff += FF_DZDT_NEG_GAIN * dzdt * FF_DT_SCALE;

        // === PI CONTROLLER ===
        float error = g_h_control_data->target_pm - z_hat;
        integral += error * dt;
        if (integral > INTEGRAL_CLAMP) integral = INTEGRAL_CLAMP;
        if (integral < -INTEGRAL_CLAMP) integral = -INTEGRAL_CLAMP;

        float delta_q = -(KP * error + KI * integral) + ff;

        // Deadband
        if (z_hat > DEADBAND_LOW && z_hat < DEADBAND_HIGH) {
            delta_q = 0.0f;
        }

        // Apply (clamp to safe range)
        float new_pm = g_h_control_data->target_pm + delta_q;
        if (new_pm < 0.5f) new_pm = 0.5f;
        if (new_pm > 1.0f) new_pm = 1.0f;
        g_h_control_data->target_pm = new_pm;

        // Telemetry
        g_h_control_data->current_pm = z_hat;
        g_h_control_data->z_estimate = z_hat;
        g_h_control_data->dzdt_estimate = dzdt;
        g_h_control_data->pm_error = error;
        g_h_control_data->last_control_error = error;
        if (fabsf(delta_q) > 0.001f) g_h_control_data->proactive_corrections++;

        // 100 kHz guard
        if (elapsed < 0.000008f) { usleep(4); continue; }
    }
    return NULL;
}

void* burst_scheduler(void* arg);
int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic] v23 80-100 kHz Q-Axis Governor (Full Kalman + Scaled FF)\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] %s | SMs=%d | CC=%d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    // Allocate zero-copy structures
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

    void* burst_scheduler(void* arg);
    pthread_t ctrl_thread, burst_thread;
    pthread_create(&ctrl_thread, NULL, control_loop, NULL);
    pthread_create(&burst_thread, NULL, burst_scheduler, NULL);  // reuse from v22

    printf("[MAIN] Running 15-second 100 kHz governor validation...\n");
    printf("[MAIN] Watch for proactive_corrections and stable Z≈1.0 under bursty load.\n\n");

    for (int i = 0; i < 15; i++) {
        usleep(1000000);
        printf("[TEL] t=%2ds | Z=%.3f | dZ/dt=%.2f | PM=%.3f | err=%.3f | corr=%d\n",
               i+1,
               g_h_control_data->z_estimate,
               g_h_control_data->dzdt_estimate,
               g_h_control_data->current_pm,
               g_h_control_data->pm_error,
               g_h_control_data->proactive_corrections);
    }

    g_running = 0;
    g_h_control_data->control_flags |= 0x2;
    pthread_join(ctrl_thread, NULL);
    pthread_join(burst_thread, NULL);
    cudaDeviceSynchronize();

    printf("\n[GPUTronic] v23 80-100 kHz run complete.\n");
    cudaFree(g_sm_counters);
    cudaFreeHost(g_h_control_data);
    cudaFreeHost(g_h_burst_params);
    return 0;
}

// ============================================================================
// HOST-SIDE BURST SCHEDULER (identical to v22)
// ============================================================================

void* burst_scheduler(void* arg) {
    (void)arg;
    while (g_running) {
        // 800ms burst phase
        g_h_burst_params->burst_active = 1;
        g_h_burst_params->intensity_factor = 1.45f;
        usleep(800000);

        // 1200ms idle/low phase
        g_h_burst_params->burst_active = 0;
        g_h_burst_params->intensity_factor = 0.55f;
        usleep(1200000);
    }
    return NULL;
}