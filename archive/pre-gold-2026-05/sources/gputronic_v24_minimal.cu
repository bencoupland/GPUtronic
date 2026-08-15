// GPUTronic v24 Minimal — Governor with extended Z range (0.3 → 5.0)

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

#define NUM_SM 84

struct __align__(16) GPUControlData {
    unsigned int control_flags;
    float target_pm;
    float current_pm;
    unsigned long long total_work_pulses;
    int throttle_sleep_ns;
    int max_sleep_ns;
    float z_estimate;
    float dzdt_estimate;
    float pm_error;
    int proactive_corrections;
};

static volatile int g_running = 1;
static GPUControlData* g_h = NULL;
static GPUControlData* g_d = NULL;
static unsigned long long* g_counters = NULL;

static float P[2][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}};

inline double get_time_us() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

__device__ __forceinline__ void heavy_work(float* acc, int sm_id, int iter) {
    float val = *acc;
    #pragma unroll
    for (int k = 0; k < 24; k++) {
        float a = __sinf((float)((iter * 13 + k * 7 + sm_id) & 0xFF) * 0.0245436926f);
        float b = __cosf((float)((iter * 17 + k * 11 + sm_id) & 0xFF) * 0.0245436926f);
        val = __fmaf_rn(a, b, val);
        val = __fmaf_rn(val * 0.7f, a, val);
    }
    *acc = val;
}

__global__ void kernel(unsigned long long* counters, GPUControlData* ctrl, int nsm) {
    int sm = blockIdx.x;
    float local = 0.0f;

    while (true) {
        if (ctrl->control_flags & 0x2) break;

        for (int i = 0; i < 1024; i++) {
            heavy_work(&local, sm, i);
        }
        atomicAdd(&counters[sm], 1024ULL);

        int s = ctrl->throttle_sleep_ns;
        if (s > 0) __nanosleep(s);
    }
}

void* control_loop(void*) {
    const float dt = 10e-6f;
    float integral = 0.0f;
    float last_z = 1.0f;
    double last_time = get_time_us();

    float smoothed_rate = 1500000.0f;
    const float TARGET_RATE = 2000000.0f;

    P[0][0] = 1.0f; P[0][1] = 0.0f;
    P[1][0] = 0.0f; P[1][1] = 1.0f;

    printf("[CTRL] Governor with extended Z range (0.3 → 5.0)\n");

    while (g_running) {
        double now = get_time_us();
        if (now - last_time < 8e-6) { usleep(1); continue; }
        last_time = now;

        unsigned long long delta = g_h->total_work_pulses;
        static unsigned long long prev = 0;
        unsigned long long d = delta - prev; prev = delta;

        float inst_rate = d / dt;
        smoothed_rate = 0.75f * smoothed_rate + 0.25f * inst_rate;

        // Extended Z range (relative)
        float z = TARGET_RATE / (smoothed_rate + 10000.0f);
        if (z > 5.0f) z = 5.0f;
        if (z < 0.3f) z = 0.3f;

        // Kalman
        float p00 = P[0][0] + 2*dt*P[0][1] + dt*dt*P[1][1] + 0.00045f;
        float p01 = P[0][1] + dt*P[1][1];
        float p10 = P[1][0] + dt*P[1][1];
        float p11 = P[1][1] + 0.00045f;
        P[0][0] = p00; P[0][1] = p01; P[1][0] = p10; P[1][1] = p11;

        float y = z - last_z;
        float s = P[0][0] + 0.018f;
        float K0 = P[0][0] / s;
        float z_hat = last_z + K0 * y;
        last_z = z_hat;

        float dzdt = (z_hat - last_z) / dt;
        float ff = 0.055f * (z_hat - 1.0f) + 0.55f * dzdt * 0.5f;
        if (dzdt < 0) ff += 0.14f * dzdt * 0.5f;

        float error = g_h->target_pm - z_hat;
        integral += error * dt;
        if (integral > 0.8f) integral = 0.8f;
        if (integral < -0.8f) integral = -0.8f;

        float delta_q = -(0.45f * error + 0.055f * integral) + ff;

        // Deadband only between 1.05 and 1.80
        if (z_hat > 1.05f && z_hat < 1.80f) delta_q = 0.0f;

        // More aggressive scaling when Z is high
        float sleep_scale = (z_hat > 2.5f) ? 65000.0f : 45000.0f;
        int sleep = (int)(5 + delta_q * sleep_scale);

        if (sleep < 0) sleep = 0;
        if (sleep > g_h->max_sleep_ns) sleep = g_h->max_sleep_ns;
        g_h->throttle_sleep_ns = sleep;

        g_h->current_pm = z_hat;
        g_h->z_estimate = z_hat;
        g_h->dzdt_estimate = dzdt;
        g_h->pm_error = error;
        if (sleep > 5) g_h->proactive_corrections++;

        usleep(1);
    }
    return NULL;
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic] v24 Minimal Governor — Extended Z Range\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] %s | SMs=%d\n\n", prop.name, prop.multiProcessorCount);

    cudaMalloc(&g_counters, NUM_SM * sizeof(unsigned long long));

    cudaHostAlloc((void**)&g_h, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d, g_h, 0);

    g_h->target_pm = 0.82f;
    g_h->max_sleep_ns = 150000;
    g_h->throttle_sleep_ns = 0;

    printf("[MAIN] Launching on full GPU (%d SMs)...\n", NUM_SM);
    kernel<<<NUM_SM, 64>>>(g_counters, g_d, NUM_SM);

    pthread_t t;
    pthread_create(&t, NULL, control_loop, NULL);

    for (int i = 0; i < 20; i++) {
        usleep(700000);
        printf("[TEL] Z=%.3f | sleep=%6d | PM=%.3f | err=%.3f | corr=%d\n",
               g_h->z_estimate, g_h->throttle_sleep_ns,
               g_h->current_pm, g_h->pm_error, g_h->proactive_corrections);
    }

    g_running = 0;
    g_h->control_flags |= 0x2;
    pthread_join(t, NULL);
    cudaDeviceSynchronize();

    printf("\n[GPUTronic] v24 run complete.\n");
    cudaFree(g_counters);
    cudaFreeHost(g_h);
    return 0;
}