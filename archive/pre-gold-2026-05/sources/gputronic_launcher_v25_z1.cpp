// GPUTronic Launcher v25 — Z=1.0 target with Kp/Ki args + histogram
// Usage: ./gputronic_launcher_v25_z1 <kp> <ki> <governor_binary> [args...]

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <time.h>
#include <math.h>
#include <string.h>

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
static pid_t g_child_pid = -1;
static FILE* g_log = NULL;

static float P[2][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}};
static const float Q = 0.0008f;
static const float R = 0.018f;
static float last_z = 1.0f;

static float user_kp = 2.0f;
static float user_ki = 0.25f;

// Histogram (0.80 ... 1.35+)
#define HIST_BINS 12
static int hist[HIST_BINS] = {0};
static long long hist_samples = 0;
static double sum_z = 0.0, sum_sq = 0.0;
static int min_sleep_seen = 999999;
static int max_sleep_seen = 0;

void handle_signal(int sig) {
    g_running = 0;
    if (g_child_pid > 0) kill(g_child_pid, SIGTERM);
}

void kalman_update(float measured_z, float dt, float* z_hat, float* dzdt_hat) {
    float z_pred = last_z + dt * (*dzdt_hat);
    float dzdt_pred = *dzdt_hat;

    float p00 = P[0][0] + 2.0f * dt * P[0][1] + dt * dt * P[1][1] + Q;
    float p01 = P[0][1] + dt * P[1][1];
    float p10 = P[1][0] + dt * P[1][1];
    float p11 = P[1][1] + Q;

    P[0][0] = p00; P[0][1] = p01;
    P[1][0] = p10; P[1][1] = p11;

    float y = measured_z - z_pred;
    float s = P[0][0] + R;
    float K0 = P[0][0] / s;
    float K1 = P[1][0] / s;

    *z_hat = z_pred + K0 * y;
    *dzdt_hat = dzdt_pred + K1 * y;

    P[0][0] = (1.0f - K0) * P[0][0];
    P[0][1] = (1.0f - K0) * P[0][1];
    P[1][0] = (1.0f - K1) * P[1][0];
    P[1][1] = (1.0f - K1) * P[1][1];

    last_z = *z_hat;
}

void* control_loop(void*) {
    const float dt = 10e-6f;
    float integral = 0.0f;
    float z_hat = 1.0f;
    float dzdt_hat = 0.0f;
    double last_time = 0;

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    last_time = ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;

    P[0][0] = 1.0f; P[0][1] = 0.0f;
    P[1][0] = 0.0f; P[1][1] = 1.0f;

    printf("[CTRL] Z=1.0 target | Kp=%.3f Ki=%.4f | ceiling=100\n", user_kp, user_ki);

    while (g_running) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double now = ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
        if (now - last_time < 8) { usleep(1); continue; }
        last_time = now;

        unsigned long long delta = g_h->total_work_pulses;
        static unsigned long long prev = 0;
        unsigned long long d = delta - prev; prev = delta;

        float inst_rate = d / dt;
        float measured_z = 2000000.0f / (inst_rate + 10000.0f);
        if (measured_z > 100.0f) measured_z = 100.0f;
        if (measured_z < 0.3f) measured_z = 0.3f;

        kalman_update(measured_z, dt, &z_hat, &dzdt_hat);

        float z = z_hat;
        if (z > 100.0f) z = 100.0f;
        if (z < 0.3f) z = 0.3f;

        float error = 1.0f - z;
        integral += error * dt;
        if (integral > 1.0f) integral = 1.0f;
        if (integral < -1.0f) integral = -1.0f;

        float delta_q = -(user_kp * error + user_ki * integral);
        if (z > 0.95f && z < 1.05f) delta_q = 0.0f;

        int sleep = (int)(5 + delta_q * 55000.0f);
        if (sleep < 0) sleep = 0;
        if (sleep > g_h->max_sleep_ns) sleep = g_h->max_sleep_ns;
        g_h->throttle_sleep_ns = sleep;

        g_h->current_pm = z;
        g_h->z_estimate = z;
        g_h->dzdt_estimate = dzdt_hat;
        g_h->pm_error = error;
        if (sleep > 5) g_h->proactive_corrections++;

        // Histogram update
        int bin = (int)((z - 0.80f) * 20.0f);
        if (bin < 0) bin = 0;
        if (bin >= HIST_BINS) bin = HIST_BINS-1;
        hist[bin]++;
        hist_samples++;
        sum_z += z;
        sum_sq += z * z;
        if (sleep < min_sleep_seen) min_sleep_seen = sleep;
        if (sleep > max_sleep_seen) max_sleep_seen = sleep;

        if (g_log) {
            fprintf(g_log, "%.3f,%.3f,%d,%.3f,%.3f,%d\n",
                    now/1e6, z, sleep, z, error, g_h->proactive_corrections);
        }
        usleep(1);
    }
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <kp> <ki> <governor_binary> [args...]\n", argv[0]);
        return 1;
    }
    user_kp = atof(argv[1]);
    user_ki = atof(argv[2]);

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    time_t t = time(NULL);
    struct tm* tm_info = localtime(&t);
    char logname[128];
    strftime(logname, sizeof(logname), "gputronic_z1_%Y%m%d_%H%M%S.log", tm_info);
    g_log = fopen(logname, "w");
    if (g_log) fprintf(g_log, "time,Z,sleep,PM,error,corr\n");

    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic v25] Z=1.0 target | Kp=%.3f Ki=%.4f\n", user_kp, user_ki);
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[LOG] %s\n\n", logname);

    cudaSetDevice(0);
    cudaHostAlloc((void**)&g_h, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d, g_h, 0);
    cudaMalloc(&g_counters, NUM_SM * sizeof(unsigned long long));

    g_h->target_pm = 1.0f;
    g_h->max_sleep_ns = 200000;
    g_h->throttle_sleep_ns = 0;

    pthread_t ctrl_thread;
    pthread_create(&ctrl_thread, NULL, control_loop, NULL);

    usleep(200000);

    pid_t pid = fork();
    if (pid == 0) {
        execvp(argv[3], &argv[3]);
        perror("execvp failed");
        exit(1);
    }
    g_child_pid = pid;

    int tick = 0;
    while (waitpid(pid, NULL, WNOHANG) == 0 && g_running) {
        printf("[TEL %3d] Z=%.3f | sleep=%6d | err=%.3f\n",
               tick++, g_h->z_estimate, g_h->throttle_sleep_ns, g_h->pm_error);
        usleep(700000);
    }

    g_running = 0;
    pthread_join(ctrl_thread, NULL);
    if (g_log) fclose(g_log);

    // Final histogram report
    printf("\n=== Z=1.0 Tracking Summary (Kp=%.3f Ki=%.4f) ===\n", user_kp, user_ki);
    printf("Samples: %lld\n", hist_samples);
    if (hist_samples > 0) {
        float mean = sum_z / hist_samples;
        float var = (sum_sq / hist_samples) - (mean * mean);
        printf("Mean Z: %.4f   StdDev: %.4f\n", mean, sqrtf(var));
    }
    printf("Throttle range: %d - %d ns\n\n", min_sleep_seen, max_sleep_seen);

    printf("Z Histogram (0.80-1.35+):\n");
    for (int i = 0; i < HIST_BINS; i++) {
        float zc = 0.80f + i * 0.05f;
        int pct = hist_samples ? (int)(hist[i] * 100.0f / hist_samples) : 0;
        const char* tag = (zc >= 0.95f && zc <= 1.05f) ? " <-- target band" : "";
        printf("  %.2f: %5d (%3d%%)%s\n", zc, hist[i], pct, tag);
    }

    printf("\n[LAUNCHER v25] Done.\n");
    cudaFreeHost(g_h);
    cudaFree(g_counters);
    return 0;
}