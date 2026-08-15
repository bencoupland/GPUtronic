// GPUTronic v26 — Host-side Integration Layer (embeddable governor)

#include "gputronic.h"
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#define NUM_SM              GPUTRONIC_NUM_SM
#define THREADS_PER_BLOCK   64
#define CONTROL_DT_US       GPUTRONIC_CONTROL_DT_US
#define MAX_SLEEP_NS        GPUTRONIC_MAX_SLEEP_NS
#define BASE_SLEEP_NS       50
#define HIST_BINS           40

// Forward declaration of the device kernel (defined in gputronic_kernel.cu)
void launch_persistent_kernel(unsigned long long* counters,
                                         struct GPUTronicControl* control,
                                         int num_sm, int threads_per_block);

struct GPUTronicHandle {
    struct GPUTronicControl* h_control;
    struct GPUTronicControl* d_control;
    unsigned long long*      sm_counters;
    pthread_t                ctrl_thread;
    int                      running;
    float                    kp, ki, target;
};

// Histogram + Kalman state
static int    hist[HIST_BINS] = {0};
static long long hist_samples = 0;
static double sum_z = 0.0, sum_sq_z = 0.0;
static int    min_sleep = 999999, max_sleep = 0;
static float  P[2][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}};

static inline double get_time_us() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

static void print_histogram(float kp, float ki, float target) {
    printf("\n=== GPUTronic v26 Z Tracking (Kp=%.3f Ki=%.4f Target=%.2f) ===\n", kp, ki, target);
    printf("Samples: %lld\n", hist_samples);
    if (hist_samples > 0) {
        float mean = sum_z / hist_samples;
        float variance = (sum_sq_z / hist_samples) - (mean * mean);
        printf("Mean Z: %.4f   StdDev: %.4f\n", mean, sqrtf(variance));
    }
    printf("Throttle range: %d - %d ns\n\n", min_sleep, max_sleep);
}

static void* v26_control_loop(void* arg) {
    GPUTronicHandle* h = (GPUTronicHandle*)arg;
    const float dt = CONTROL_DT_US * 1e-6f;
    float integral = 0.0f;
    float last_z = h->target;
    double last_time = get_time_us();

    P[0][0] = 1.0f; P[0][1] = 0.0f;
    P[1][0] = 0.0f; P[1][1] = 1.0f;

    printf("[CTRL] v26 100kHz | Kp=%.3f Ki=%.4f Target=%.2f | 20s warm-up\n",
           h->kp, h->ki, h->target);

    double warmup_until = get_time_us() + 20000000.0;

    while (h->running) {
        double now = get_time_us();
        double elapsed = now - last_time;
        if (elapsed < CONTROL_DT_US * 0.8) { usleep(2); continue; }
        last_time = now;

        unsigned long long work = h->h_control->total_work_pulses;
        static unsigned long long prev_work = 0;
        unsigned long long delta = work - prev_work;
        prev_work = work;

        float z = (delta > 0) ? fminf(3.0f, 0.1f + (1200000.0f / (delta + 1.0f))) : 2.0f;
        if (z > 3.0f) z = 3.0f;
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
        float error = h->target - z_hat;
        integral += error * dt;
        if (integral > 0.8f) integral = 0.8f;
        if (integral < -0.8f) integral = -0.8f;

        int new_sleep = 0;
        if (now > warmup_until) {
            float delta_q = -(h->kp * error + h->ki * integral);
            if (z_hat > 0.95f && z_hat < 1.25f) delta_q = 0.0f;
            new_sleep = (int)(BASE_SLEEP_NS + (delta_q * 120000.0f));
            if (new_sleep < 0) new_sleep = 0;
            if (new_sleep > h->h_control->max_sleep_ns) new_sleep = h->h_control->max_sleep_ns;
        }

        h->h_control->throttle_sleep_ns = new_sleep;
        h->h_control->current_pm = z_hat;
        h->h_control->z_estimate = z_hat;
        h->h_control->dzdt_estimate = dzdt;
        h->h_control->pm_error = error;
        if (new_sleep > BASE_SLEEP_NS) h->h_control->proactive_corrections++;

        int bin = (int)((z_hat - 0.05f) * 20.0f);
        if (bin < 0) bin = 0; if (bin >= HIST_BINS) bin = HIST_BINS-1;
        hist[bin]++; hist_samples++; sum_z += z_hat; sum_sq_z += z_hat * z_hat;
        if (new_sleep < min_sleep) min_sleep = new_sleep;
        if (new_sleep > max_sleep) max_sleep = new_sleep;

        if (elapsed < 0.000008f) { usleep(4); continue; }
    }
    print_histogram(h->kp, h->ki, h->target);
    return NULL;
}

GPUTronicHandle* gputronic_init(float kp, float ki, float target_z) {
    GPUTronicHandle* h = (GPUTronicHandle*)calloc(1, sizeof(GPUTronicHandle));
    h->kp = kp; h->ki = ki; h->target = target_z;

    cudaSetDevice(0);
    cudaHostAlloc((void**)&h->h_control, sizeof(struct GPUTronicControl), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&h->d_control, h->h_control, 0);

    memset(h->h_control, 0, sizeof(struct GPUTronicControl));
    h->h_control->target_pm = target_z;
    h->h_control->max_sleep_ns = MAX_SLEEP_NS;
    h->h_control->blocks_per_sm_target = NUM_SM;
    h->h_control->throttle_sleep_ns = 0;

    cudaMalloc((void**)&h->sm_counters, NUM_SM * sizeof(unsigned long long));

    // Launch kernel (implemented in gputronic_kernel.cu)
    launch_persistent_kernel(h->sm_counters, h->d_control, NUM_SM, THREADS_PER_BLOCK);

    printf("[GPUTronic v26] Initialized + kernel launched | target=%.2f Kp=%.3f Ki=%.4f\n",
           target_z, kp, ki);
    return h;
}

int gputronic_start(GPUTronicHandle* h) {
    if (!h) return -1;
    h->running = 1;
    pthread_create(&h->ctrl_thread, NULL, v26_control_loop, h);
    return 0;
}

void gputronic_stop(GPUTronicHandle* h) {
    if (!h) return;
    h->running = 0;
    h->h_control->control_flags |= 0x2;
    pthread_join(h->ctrl_thread, NULL);
    cudaDeviceSynchronize();
    cudaFree(h->sm_counters);
    cudaFreeHost(h->h_control);
    free(h);
    printf("[GPUTronic v26] Stopped.\n");
}

float gputronic_get_z(GPUTronicHandle* h) {
    return (h && h->h_control) ? h->h_control->z_estimate : 0.0f;
}

int gputronic_get_sleep_ns(GPUTronicHandle* h) {
    return (h && h->h_control) ? h->h_control->throttle_sleep_ns : 0;
}

void gputronic_set_target(GPUTronicHandle* h, float target_z) {
    if (h && h->h_control) h->h_control->target_pm = target_z;
}

struct GPUTronicControl* gputronic_get_control(GPUTronicHandle* h) {
    return h ? h->h_control : NULL;
}

// External mode: governor runs but does not launch any kernel.
// The application (Cyberpunk, inference engine, etc.) is responsible for its own work.
// The governor still provides Z observation and throttle control via zero-copy.
GPUTronicHandle* gputronic_init_external(float kp, float ki, float target_z) {
    GPUTronicHandle* h = (GPUTronicHandle*)calloc(1, sizeof(GPUTronicHandle));
    h->kp = kp; h->ki = ki; h->target = target_z;

    cudaSetDevice(0);
    cudaHostAlloc((void**)&h->h_control, sizeof(struct GPUTronicControl), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&h->d_control, h->h_control, 0);

    memset(h->h_control, 0, sizeof(struct GPUTronicControl));
    h->h_control->target_pm = target_z;
    h->h_control->max_sleep_ns = MAX_SLEEP_NS;
    h->h_control->blocks_per_sm_target = NUM_SM;
    h->h_control->throttle_sleep_ns = 0;

    // No sm_counters allocation or kernel launch in external mode
    h->sm_counters = NULL;

    printf("[GPUTronic v26] External mode initialized | target=%.2f Kp=%.3f Ki=%.4f\n",
           target_z, kp, ki);
    printf("  (No internal kernel — suitable for Cyberpunk / external workloads)\n");
    return h;
}
