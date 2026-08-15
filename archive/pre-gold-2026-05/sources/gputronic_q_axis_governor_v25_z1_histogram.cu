// =============================================================================
// GPUTronic Q-Axis Governor v25 — Z=1.0 Target + Histogram + Warm-up
// =============================================================================
// - 100 kHz control loop
// - 20-second zero-throttle warm-up on startup
// - Command-line Kp Ki target
// - Real-time Z histogram
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
#define THREADS_PER_BLOCK   64
#define WORK_UNITS_PER_THREAD 8192

#define CONTROL_DT_US       10
#define MAX_SLEEP_NS        800000
#define BASE_SLEEP_NS       50

static unsigned long long* g_sm_counters = NULL;
static struct GPUControlData* g_d_control = NULL;
static struct GPUControlData* g_h_control = NULL;

static volatile int g_running = 1;

static float KP = 0.35f;
static float KI = 0.04f;
static float TARGET_PM = 1.0f;
static float DEADBAND_LOW = 0.95f;
static float DEADBAND_HIGH = 1.25f;

#define HIST_BINS 40
static int hist[HIST_BINS] = {0};
static long long hist_samples = 0;
static double sum_z = 0.0, sum_sq_z = 0.0;
static int min_sleep = 999999, max_sleep = 0;

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
            if ((i & 31) == 0) atomicAdd(&sm_counters[sm_id], 32ULL);
        }
        if (sleep_ns > 0) __nanosleep(sleep_ns);
        control_data->total_work_pulses = sm_counters[sm_id];
        control_data->active_blocks_current = num_sm;
    }
}

void handle_signal(int sig) { (void)sig; g_running = 0; if (g_h_control) g_h_control->control_flags |= 0x2; }

void print_histogram() {
    printf("\n=== Z=1.0 Tracking Summary (Kp=%.3f Ki=%.4f Target=%.2f) ===\n", KP, KI, TARGET_PM);
    printf("Samples: %lld\n", hist_samples);
    if (hist_samples > 0) {
        float mean = sum_z / hist_samples;
        float variance = (sum_sq_z / hist_samples) - (mean * mean);
        printf("Mean Z: %.4f   StdDev: %.4f\n", mean, sqrtf(variance));
    }
    printf("Throttle range: %d - %d ns\n\n", min_sleep, max_sleep);
    printf("Z Histogram (0.05–2.00):\n");
    for (int i = 0; i < HIST_BINS; i++) {
        float zc = 0.05f + i * 0.05f;
        int pct = hist_samples ? (int)(hist[i] * 100.0f / hist_samples) : 0;
        const char* tag = (zc >= 0.95f && zc <= 1.25f) ? " <-- target band" : "";
        if (hist[i] > 0 || (zc >= 0.90f && zc <= 1.30f))
            printf("  %.2f: %6d (%3d%%)%s\n", zc, hist[i], pct, tag);
    }
}

void* control_loop(void* arg) {
    (void)arg;
    const float dt = CONTROL_DT_US * 1e-6f;
    float integral = 0.0f;
    float last_z = TARGET_PM;
    double last_time = get_time_us();

    P[0][0] = 1.0f; P[0][1] = 0.0f;
    P[1][0] = 0.0f; P[1][1] = 1.0f;

    printf("[CTRL] v25 Z=1.0 + 20s warm-up | Kp=%.3f Ki=%.4f\n", KP, KI);

    double warmup_until = get_time_us() + 20000000.0;  // 20s warm-up

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
        float p00 = P[0][0] + 2.0f*dt*P[0][1] + dt*dt*P[1][1] + 0.00045f;
        float p01 = P[0][1] + dt*P[1][1];
        float p10 = P[1][0] + dt*P[1][1];
        float p11 = P[1][1] + 0.00045f;
        P[0][0] = p00; P[0][1] = p01; P[1][0] = p10; P[1][1] = p11;

        float y = z - last_z;
        float s = P[0][0] + 0.018f;
        float K0 = P[0][0] / s;
        float K1 = P[1][0] / s;
        float z_hat = last_z + K0 * y;

        P[0][0] = (1.0f - K0) * P[0][0];
        P[0][1] = (1.0f - K0) * P[0][1];
        P[1][0] = (1.0f - K1) * P[1][0];
        P[1][1] = (1.0f - K1) * P[1][1];
        last_z = z_hat;

        float dzdt = (z_hat - last_z) / dt;
        float error = TARGET_PM - z_hat;
        integral += error * dt;
        if (integral > 0.8f) integral = 0.8f;
        if (integral < -0.8f) integral = -0.8f;

        int new_sleep = 0;
        if (now > warmup_until) {
            float delta_q = -(KP * error + KI * integral);
            if (z_hat > DEADBAND_LOW && z_hat < DEADBAND_HIGH) delta_q = 0.0f;
            new_sleep = (int)(BASE_SLEEP_NS + (delta_q * 120000.0f));
            if (new_sleep < 0) new_sleep = 0;
            if (new_sleep > g_h_control->max_sleep_ns) new_sleep = g_h_control->max_sleep_ns;
        }

        g_h_control->throttle_sleep_ns = new_sleep;
        g_h_control->current_pm = z_hat;
        g_h_control->z_estimate = z_hat;
        g_h_control->dzdt_estimate = dzdt;
        g_h_control->pm_error = error;
        if (new_sleep > BASE_SLEEP_NS) g_h_control->proactive_corrections++;

        // Histogram
        int bin = (int)((z_hat - 0.05f) * 20.0f);
        if (bin < 0) bin = 0; if (bin >= HIST_BINS) bin = HIST_BINS-1;
        hist[bin]++; hist_samples++; sum_z += z_hat; sum_sq_z += z_hat * z_hat;
        if (new_sleep < min_sleep) min_sleep = new_sleep;
        if (new_sleep > max_sleep) max_sleep = new_sleep;

        if (elapsed < 0.000008f) { usleep(4); continue; }
    }
    print_histogram();
    return NULL;
}

int main(int argc, char** argv) {
    if (argc >= 4) { KP = atof(argv[1]); KI = atof(argv[2]); TARGET_PM = atof(argv[3]); }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic v25] Z=1.0 + 20s Warm-up Governor\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] %s | SMs=%d\n", prop.name, prop.multiProcessorCount);

    cudaHostAlloc((void**)&g_h_control, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d_control, g_h_control, 0);

    memset(g_h_control, 0, sizeof(GPUControlData));
    g_h_control->target_pm = TARGET_PM;
    g_h_control->blocks_per_sm_target = NUM_SM;
    g_h_control->max_sleep_ns = MAX_SLEEP_NS;
    g_h_control->throttle_sleep_ns = 0;   // Start at zero throttle

    cudaMalloc((void**)&g_sm_counters, NUM_SM * sizeof(unsigned long long));

    dim3 block(THREADS_PER_BLOCK);
    gpu_persistent_kernel<<<NUM_SM, block>>>(g_sm_counters, g_d_control, NUM_SM);
    printf("[OK] Kernel launched | Kp=%.3f Ki=%.4f Target=%.2f | 20s warm-up\n\n", KP, KI, TARGET_PM);

    pthread_t ctrl_thread;
    pthread_create(&ctrl_thread, NULL, control_loop, NULL);

    for (int i = 0; i < 360; i++) {   // 6 minutes
        usleep(1000000);
        printf("[TEL] t=%3ds | Z=%.3f | sleep=%6d | err=%.3f\n",
               i+1, g_h_control->z_estimate, g_h_control->throttle_sleep_ns, g_h_control->pm_error);
    }

    g_running = 0;
    g_h_control->control_flags |= 0x2;
    pthread_join(ctrl_thread, NULL);
    cudaDeviceSynchronize();

    printf("\n[GPUTronic v25] Run complete.\n");
    cudaFree(g_sm_counters);
    cudaFreeHost(g_h_control);
    return 0;
}