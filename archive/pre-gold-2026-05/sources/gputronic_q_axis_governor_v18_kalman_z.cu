// =============================================================================
// GPUTronic Q-Axis Governor v18 — Z-Axis Kalman Observer Re-introduced
// Restores the 2-state Kalman [Z, dZ/dt] from v14 architecture while keeping
// all v17 improvements (real workload, tuned PI, deadband, auto-calibration).
//
// Z is estimated from the rate of change of Q-axis work pulses.
// This gives the governor predictive stall/impedance awareness.
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

#define CONTROL_DT_US       100
#define WORK_UNITS_PER_THREAD 8192
#define TILE_K              16

// PI + Deadband
#define KP                  0.42f
#define KI                  0.07f
#define INTEGRAL_MAX        2.8f
#define DEADBAND            0.035f
#define TARGET_RATE_MIN     0.10f
#define TARGET_RATE_MAX     1.00f

// Kalman 2-state [Z, dZ/dt]
#define KALMAN_Q_PROC       0.0008f
#define KALMAN_R_MEAS       0.035f

static unsigned long long* g_sm_counters = NULL;
static struct GPUControlData* g_d_control_data = NULL;
static struct GPUControlData* g_h_control_data = NULL;

static volatile int g_running = 1;
static double g_start_time_us = 0.0;
static double g_measured_max_rate = 65000000.0;

struct __align__(16) GPUControlData {
    unsigned int control_flags;
    float target_q_rate;
    int blocks_per_sm_target;
    unsigned long long total_work_pulses;
    float last_control_error;
    int active_blocks_current;
    float z_estimate;      // Z-axis impedance (new in v18)
    float dzdt_estimate;   // dZ/dt
};

inline double get_time_us() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

// Real workload
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

// 2-state Kalman observer for Z-axis
void kalman_update(float* z, float* dzdt, float measured_impedance, float dt) {
    // Prediction
    float z_pred = *z + *dzdt * dt;
    float dzdt_pred = *dzdt;

    // Covariance prediction (simplified 2-state)
    static float P[2][2] = {{0.1f, 0.0f}, {0.0f, 0.05f}};
    P[0][0] += KALMAN_Q_PROC;
    P[1][1] += KALMAN_Q_PROC * 0.6f;

    // Measurement update
    float y = measured_impedance - z_pred;           // innovation
    float S = P[0][0] + KALMAN_R_MEAS;
    float K0 = P[0][0] / S;
    float K1 = P[1][0] / S;

    *z    = z_pred + K0 * y;
    *dzdt = dzdt_pred + K1 * y;

    // Covariance update
    P[0][0] = (1.0f - K0) * P[0][0];
    P[1][0] = (1.0f - K0) * P[1][0];
    P[0][1] = P[1][0];
    P[1][1] = (1.0f - K1) * P[1][1];
}

// Main control thread with Kalman Z estimation
void* control_thread(void* arg) {
    (void)arg;
    float integral = 0.0f;
    unsigned long long last_pulses = 0;
    double last_time = get_time_us();
    float z = 1.0f, dzdt = 0.0f;

    printf("[CTRL] v18 Kalman Z-Axis Observer active (2-state)\n");

    while (g_running) {
        double now = get_time_us();
        double dt = (now - last_time) / 1e6f;
        if (dt < 0.00008f) { usleep(40); continue; }

        unsigned long long current = g_h_control_data->total_work_pulses;
        double delta = (double)(current - last_pulses);
        if (delta < 0) delta = 0;
        double measured_rate = delta / dt;

        // Auto-calibrate
        if (now - g_start_time_us < 4e6 && measured_rate > g_measured_max_rate * 0.9 && measured_rate < 300e6)
            g_measured_max_rate = measured_rate;

        float normalized = (float)(measured_rate / g_measured_max_rate);
        if (normalized > 1.0f) normalized = 1.0f;

        // === Z-AXIS KALMAN OBSERVER ===
        // Impedance proxy: inverse of normalized throughput (high Z = low throughput)
        float measured_z = (normalized > 0.05f) ? (1.0f / normalized) : 2.5f;
        kalman_update(&z, &dzdt, measured_z, (float)dt);

        // Clamp Z
        if (z < 0.6f) z = 0.6f;
        if (z > 2.8f) z = 2.8f;

        // Update zero-copy telemetry
        g_h_control_data->z_estimate = z;
        g_h_control_data->dzdt_estimate = dzdt;

        // PI control with Z influence (simple phase-margin style)
        float target = g_h_control_data->target_q_rate;
        float error = target - normalized;

        // Z-based adjustment (reduce target when Z is high)
        if (z > 1.6f) error += (z - 1.6f) * 0.08f;

        if (fabsf(error) < DEADBAND) error = 0.0f;

        integral += error * dt;
        if (integral > INTEGRAL_MAX) integral = INTEGRAL_MAX;
        if (integral < -INTEGRAL_MAX) integral = -INTEGRAL_MAX;

        float pi_delta = -(KP * error + KI * integral);
        float new_target = target + pi_delta * 0.65f;

        if (new_target < TARGET_RATE_MIN) new_target = TARGET_RATE_MIN;
        if (new_target > TARGET_RATE_MAX) new_target = TARGET_RATE_MAX;

        g_h_control_data->target_q_rate = new_target;
        g_h_control_data->last_control_error = error;

        // Telemetry
        static int tick = 0;
        if ((tick++ % 10) == 0) {
            printf("[CTRL] t=%.1fs | tgt=%.3f | Z=%.3f | dZ/dt=%.4f | err=%.3f\n",
                   (now - g_start_time_us)/1e6f, new_target, z, dzdt, error);
        }

        last_pulses = current;
        last_time = now;
        usleep(CONTROL_DT_US);
    }
    return NULL;
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic] Q-Axis Governor v18 — Z-Axis Kalman Re-introduced\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] %s | SMs=%d | CC=%d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    cudaHostAlloc((void**)&g_h_control_data, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d_control_data, g_h_control_data, 0);
    memset(g_h_control_data, 0, sizeof(GPUControlData));
    g_h_control_data->target_q_rate = 0.82f;
    g_h_control_data->blocks_per_sm_target = BLOCKS_PER_SM;
    g_h_control_data->z_estimate = 1.0f;

    cudaMalloc((void**)&g_sm_counters, NUM_SM * sizeof(unsigned long long));
    reset_sm_counters<<<(NUM_SM+31)/32, 32>>>(g_sm_counters, NUM_SM);
    cudaDeviceSynchronize();

    g_start_time_us = get_time_us();

    dim3 block(THREADS_PER_BLOCK);
    gpu_persistent_governor_kernel<<<NUM_SM, block>>>(
        g_sm_counters, g_d_control_data, NUM_SM, BLOCKS_PER_SM);

    pthread_t ctrl_thread;
    pthread_create(&ctrl_thread, NULL, control_thread, NULL);

    printf("[MAIN] Running 25s with Z-Kalman observer...\n\n");

    while (g_running) {
        usleep(400000);
        if (get_time_us() - g_start_time_us > 25e6) g_running = 0;
    }

    g_h_control_data->control_flags |= 0x2;
    g_running = 0;
    pthread_join(ctrl_thread, NULL);
    cudaDeviceSynchronize();

    cudaFree(g_sm_counters);
    cudaFreeHost(g_h_control_data);

    printf("\n[GPUTronic] v18 Kalman Z test complete.\n");
    return 0;
}