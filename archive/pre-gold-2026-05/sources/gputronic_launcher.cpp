// GPUTronic Launcher v1.14 — 2-State Kalman (Q=0.0008, Z target=1.0, Z ceiling=100)

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
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
static pid_t g_child_pid = -1;
static FILE* g_log = NULL;

static float P[2][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}};
static const float Q = 0.0008f;
static const float R = 0.018f;
static float last_z = 1.0f;

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

    printf("[CTRL] 2-State Kalman (Q=0.0008, Z target=1.0, Z ceiling=100)\n");

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
        if (measured_z > 100.0f) measured_z = 100.0f;   // Raised ceiling
        if (measured_z < 0.3f) measured_z = 0.3f;

        kalman_update(measured_z, dt, &z_hat, &dzdt_hat);

        float z = z_hat;
        if (z > 100.0f) z = 100.0f;   // Raised ceiling
        if (z < 0.3f) z = 0.3f;

        float error = 1.0f - z;
        integral += error * dt;
        if (integral > 1.0f) integral = 1.0f;
        if (integral < -1.0f) integral = -1.0f;

        float delta_q = -(0.45f * error + 0.055f * integral);
        if (z > 0.9f && z < 1.1f) delta_q = 0.0f;

        int sleep = (int)(5 + delta_q * 55000.0f);
        if (sleep < 0) sleep = 0;
        if (sleep > g_h->max_sleep_ns) sleep = g_h->max_sleep_ns;
        g_h->throttle_sleep_ns = sleep;

        g_h->current_pm = z;
        g_h->z_estimate = z;
        g_h->dzdt_estimate = dzdt_hat;
        g_h->pm_error = error;
        if (sleep > 5) g_h->proactive_corrections++;

        if (g_log) {
            fprintf(g_log, "%.3f,%.3f,%d,%.3f,%.3f,%d\n",
                    now/1e6, z, sleep, z, error, g_h->proactive_corrections);
        }
        usleep(1);
    }
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: %s <kp> <ki> <command...>\n", argv[0]); return 1; }
    float user_kp = atof(argv[1]);
    float user_ki = atof(argv[2]);
    // shift argv so the rest is the child command
    argv += 2;
    argc -= 2;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    time_t t = time(NULL);
    struct tm* tm_info = localtime(&t);
    char logname[128];
    strftime(logname, sizeof(logname), "gputronic_%Y%m%d_%H%M%S.log", tm_info);
    g_log = fopen(logname, "w");
    if (g_log) fprintf(g_log, "time,Z,sleep,PM,error,corr\n");

    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic] Launcher v1.14 — Z ceiling=100\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[LOG] %s\n\n", logname);

    cudaSetDevice(0);
    cudaHostAlloc((void**)&g_h, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d, g_h, 0);
    cudaMalloc(&g_counters, NUM_SM * sizeof(unsigned long long));

    g_h->target_pm = 0.82f;
    g_h->max_sleep_ns = 200000;
    g_h->throttle_sleep_ns = 0;

    pthread_t ctrl_thread;
    pthread_create(&ctrl_thread, NULL, control_loop, NULL);

    usleep(200000);

    pid_t pid = fork();
    if (pid == 0) {
        execvp(argv[1], &argv[1]);
        perror("execvp failed");
        exit(1);
    }
    g_child_pid = pid;

    int tick = 0;
    while (waitpid(pid, NULL, WNOHANG) == 0 && g_running) {
        printf("[TEL %3d] Z=%.3f | sleep=%6d | PM=%.3f | err=%.3f | corr=%d\n",
               tick++, g_h->z_estimate, g_h->throttle_sleep_ns,
               g_h->current_pm, g_h->pm_error, g_h->proactive_corrections);
        usleep(700000);
    }

    g_running = 0;
    pthread_join(ctrl_thread, NULL);
    if (g_log) fclose(g_log);

    printf("\n[LAUNCHER] Done.\n");
    cudaFreeHost(g_h);
    cudaFree(g_counters);
    return 0;
}