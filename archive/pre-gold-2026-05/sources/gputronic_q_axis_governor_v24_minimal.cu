// =============================================================================
// GPUTronic Q-Axis Governor v24 — Minimal (Nanosleep Throttle Only)
// =============================================================================
// - No bursty workload
// - No burst scheduler
// - Placeholder workload (simple FMA loop for now)
// - Focus: Get core governor + 100 kHz control loop working on RTX 5080
// =============================================================================

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

#define NUM_SM              8
#define THREADS_PER_BLOCK   64
#define WORK_UNITS_PER_THREAD 8192
#define TILE_K              16

#define CONTROL_DT_US       10
#define KP                  0.45f
#define KI                  0.055f
#define INTEGRAL_CLAMP      0.8f
#define DEADBAND_LOW        1.05f
#define DEADBAND_HIGH       1.80f

#define FF_Z_GAIN           0.055f
#define FF_DZDT_GAIN        0.55f
#define FF_DZDT_NEG_GAIN    0.14f
#define FF_DT_SCALE         (CONTROL_DT_US / 20.0f)

#define KALMAN_Q_PROC       0.00045f
#define KALMAN_R_MEAS       0.018f

#define MAX_SLEEP_NS        800000
#define BASE_SLEEP_NS       50

static unsigned long long* g_sm_counters = NULL;
static struct GPUControlData* g_d_control = NULL;
static struct GPUControlData* g_h_control = NULL;

static volatile int g_running = 1;

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
    int throttle_sleep_ns;
    int max_sleep_ns;
};

static float P[2][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}};

inline double get_time_us() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

__device__ __forceinline__ void placeholder_workload(float* acc, int sm_id, int iter) {
    float a = sinf((float)(iter * 17 + sm_id) * 0.0174532925f);
    float b = cosf((float)(iter * 23 + sm_id) * 0.0174532925f);
    *acc = fmaf(a, b, *acc);
}

__global__ void gpu_persistent_kernel(unsigned long long* sm_counters,
                                      GPUControlData* control_data, int num_sm) {
    int sm_id = blockIdx.x;

    while (true) {
        if (control_data->control_flags & 0x2) { __nanosleep(1000); continue; }

        int sleep_ns = control_data->throttle_sleep_ns;
        if (sleep_ns > control_data->max_sleep_ns) sleep_ns = control_data->max_sleep_ns;

        float thread_acc = 0.0f;
        int iterations = WORK_UNITS_PER_THREAD / 32;

        for (int i = 0; i < iterations; i++) {
            placeholder_workload(&thread_acc, sm_id, i);
            if ((i & 31) == 0) {
                atomicAdd(&sm_counters[sm_id], 32ULL);
            }
        }

        if (sleep_ns > 0) __nanosleep(sleep_ns);

        control_data->total_work_pulses = sm_counters[sm_id];
        control_data->active_blocks_current = num_sm;
    }
}

__global__ void reset_sm_counters(unsigned long long* sm_counters, int num_sm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sm) sm_counters[idx] = 0ULL;
}

void* control_loop(void* arg) {
    (void)arg;
    const float dt = CONTROL_DT_US * 1e-6f;
    float integral = 0.0f;
    float last_z = 1.0f;
    double last_time = get_time_us();

    P[0][0] = 1.0f; P[0][1] = 0.0f;
    P[1][0] = 0.0f; P[1][1] = 1.0f;

    printf("[CTRL] v24 minimal control loop started (100 kHz)\n");

    while (g_running) {
        double now = get_time_us();
        double elapsed = now - last_time;
        if (elapsed < CONTROL_DT_US * 0.8) { usleep(2); continue; }
        last_time = now;

        unsigned long long work = g_h_control->total_work_pulses;
        static unsigned long long prev_work = 0;
        unsigned long long delta = work - prev_work;
        prev_work = work;

        float z = (delta > 0) ? (1.0f / (delta * 1e-6f + 1e-9f)) : 2.0f;
        if (z > 2.0f) z = 2.0f;
        if (z < 0.05f) z = 0.05f;

        // Kalman
        float p00 = P[0][0] + 2.0f*dt*P[0][1] + dt*dt*P[1][1] + KALMAN_Q_PROC;
        float p01 = P[0][1] + dt*P[1][1];
        float p10 = P[1][0] + dt*P[1][1];
        float p11 = P[1][1] + KALMAN_Q_PROC;
        P[0][0] = p00; P[0][1] = p01; P[1][0] = p10; P[1][1] = p11;

        float y = z - last_z;
        float s = P[0][0] + KALMAN_R_MEAS;
        float K0 = P[0][0] / s;
        float K1 = P[1][0] / s;
        float z_hat = last_z + K0 * y;

        P[0][0] = (1.0f - K0) * P[0][0];
        P[0][1] = (1.0f - K0) * P[0][1];
        P[1][0] = (1.0f - K1) * P[1][0];
        P[1][1] = (1.0f - K1) * P[1][1];
        last_z = z_hat;

        float dzdt = (z_hat - last_z) / dt;
        float ff = FF_Z_GAIN * (z_hat - 1.0f) + FF_DZDT_GAIN * dzdt * FF_DT_SCALE;
        if (dzdt < 0.0f) ff += FF_DZDT_NEG_GAIN * dzdt * FF_DT_SCALE;

        float error = g_h_control->target_pm - z_hat;
        integral += error * dt;
        if (integral > INTEGRAL_CLAMP) integral = INTEGRAL_CLAMP;
        if (integral < -INTEGRAL_CLAMP) integral = -INTEGRAL_CLAMP;

        float delta_q = -(KP * error + KI * integral) + ff;
        if (z_hat > DEADBAND_LOW && z_hat < DEADBAND_HIGH) delta_q = 0.0f;

        int new_sleep = (int)(BASE_SLEEP_NS + (delta_q * 120000.0f));
        if (new_sleep < 0) new_sleep = 0;
        if (new_sleep > g_h_control->max_sleep_ns) new_sleep = g_h_control->max_sleep_ns;
        g_h_control->throttle_sleep_ns = new_sleep;

        g_h_control->current_pm = z_hat;
        g_h_control->z_estimate = z_hat;
        g_h_control->dzdt_estimate = dzdt;
        g_h_control->pm_error = error;
        if (new_sleep > BASE_SLEEP_NS) g_h_control->proactive_corrections++;

        if (elapsed < 0.000008f) { usleep(4); continue; }
    }
    return NULL;
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic] v24 Minimal Nanosleep Governor (RTX 5080)\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] %s | SMs=%d | CC=%d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    cudaHostAlloc((void**)&g_h_control, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d_control, g_h_control, 0);

    memset(g_h_control, 0, sizeof(GPUControlData));
    g_h_control->target_pm = 0.82f;
    g_h_control->blocks_per_sm_target = NUM_SM;
    g_h_control->max_sleep_ns = MAX_SLEEP_NS;
    g_h_control->throttle_sleep_ns = 0;

    cudaMalloc((void**)&g_sm_counters, NUM_SM * sizeof(unsigned long long));
    reset_sm_counters<<<(NUM_SM+31)/32, 32>>>(g_sm_counters, NUM_SM);
    cudaDeviceSynchronize();

    dim3 block(THREADS_PER_BLOCK);
    gpu_persistent_kernel<<<NUM_SM, block>>>(g_sm_counters, g_d_control, NUM_SM);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("[ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err)); return 1; }
    printf("[OK] Kernel launched on %d SMs\n", NUM_SM);

    pthread_t ctrl_thread;
    pthread_create(&ctrl_thread, NULL, control_loop, NULL);

    printf("[MAIN] Running 15-second validation...\n\n");

    for (int i = 0; i < 15; i++) {
        usleep(1000000);
        printf("[TEL] t=%2ds | Z=%.3f | sleep=%6d ns | PM=%.3f | err=%.3f | corr=%d\n",
               i+1,
               g_h_control->z_estimate,
               g_h_control->throttle_sleep_ns,
               g_h_control->current_pm,
               g_h_control->pm_error,
               g_h_control->proactive_corrections);
    }

    g_running = 0;
    g_h_control->control_flags |= 0x2;
    pthread_join(ctrl_thread, NULL);
    cudaDeviceSynchronize();

    printf("\n[GPUTronic] v24 minimal validation complete.\n");
    cudaFree(g_sm_counters);
    cudaFreeHost(g_h_control);
    return 0;
}