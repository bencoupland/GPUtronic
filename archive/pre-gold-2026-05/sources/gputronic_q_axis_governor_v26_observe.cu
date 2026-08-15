// =============================================================================
// GPUTronic Q-Axis Governor v26.5 — Observation Mode (No Cap)
// =============================================================================
// - Adaptive floating capacity model
// - NO artificial rate cap
// - Intended for use alongside real workloads (not placeholder)
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

#define MAX_SLEEP_NS        800000
#define BASE_SLEEP_NS       50
#define FLAG_STOP           0x1
#define FLAG_PAUSE          0x2
#define WARMUP_TIME_US      20000000.0

static unsigned long long* g_sm_counters = NULL;
static struct GPUControlData* g_d_control = NULL;
static struct GPUControlData* g_h_control = NULL;

static volatile int g_running = 1;

static float KP = 0.35f;
static float KI = 0.04f;
static float TARGET_PM = 1.0f;

static float initial_max_rate = 0.0f;
static float current_capacity = 0.0f;
static int calibration_done = 0;

struct __align__(16) GPUControlData {
    unsigned int control_flags;
    float target_pm;
    float current_pm;
    unsigned long long total_work_pulses;
    float z_estimate;
    float dzdt_estimate;
    float pm_error;
    int throttle_sleep_ns;
    int max_sleep_ns;
};

static float P[2][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}};

inline double get_time_us() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

__global__ void gpu_persistent_kernel(unsigned long long* sm_counters,
                                      GPUControlData* control_data) {
    int sm_id = blockIdx.x;
    int tid = threadIdx.x;

    while (true) {
        if (control_data->control_flags & FLAG_STOP) break;
        if (control_data->control_flags & FLAG_PAUSE) { __nanosleep(1000); continue; }

        int sleep_ns = control_data->throttle_sleep_ns;
        if (sleep_ns < BASE_SLEEP_NS) sleep_ns = BASE_SLEEP_NS;

        // Minimal work just to keep the kernel alive and count cycles
        atomicAdd(&sm_counters[sm_id], 1ULL);
        __nanosleep(sleep_ns);

        if (tid == 0) {
            control_data->total_work_pulses = sm_counters[sm_id];
        }
    }
}

void handle_signal(int sig) {
    (void)sig; g_running = 0;
    if (g_h_control) g_h_control->control_flags |= FLAG_STOP;
}

static void kalman_update(float measured_z, float dt, float* z_hat, float* dzdt_hat) {
    float z_pred = *z_hat + *dzdt_hat * dt;
    float dzdt_pred = *dzdt_hat;

    const float Q = 0.0006f;
    P[0][0] += 2.0f * dt * P[0][1] + dt * dt * P[1][1] + Q;
    P[0][1] += dt * P[1][1];
    P[1][0] = P[0][1];
    P[1][1] += Q * 0.7f;

    float y = measured_z - z_pred;
    const float R = 0.022f;
    float S = P[0][0] + R;
    float K0 = P[0][0] / S;
    float K1 = P[1][0] / S;

    *z_hat = z_pred + K0 * y;
    *dzdt_hat = dzdt_pred + K1 * y;

    P[0][0] = (1.0f - K0) * P[0][0];
    P[0][1] = (1.0f - K0) * P[0][1];
    P[1][0] = P[0][1];
    P[1][1] = (1.0f - K1) * P[1][1];
}

void* control_loop(void* arg) {
    (void)arg;
    float integral = 0.0f;
    float z_hat = 1.0f;
    float dzdt_hat = 0.0f;
    double last_time = get_time_us();
    unsigned long long prev_work = 0;
    static float rate_ema = 0.0f;
    const float RATE_EMA_ALPHA = 0.23f;

    P[0][0] = 1.0f; P[0][1] = 0.0f;
    P[1][0] = 0.0f; P[1][1] = 1.0f;

    FILE* csv = fopen("gputronic_log.csv", "w");
    if (csv) fprintf(csv, "time_s,Z,rate_mps,expected_rate,sleep_ns,error\n");

    printf("[CTRL] v26.5 OBSERVE | no rate cap + adaptive capacity\n");

    double warmup_until = get_time_us() + WARMUP_TIME_US;

    while (g_running) {
        double now = get_time_us();
        double dt = (now - last_time) * 1e-6;
        if (dt < 0.0000095) { usleep(1); continue; }
        last_time = now;

        unsigned long long work = g_h_control->total_work_pulses;
        unsigned long long delta = work - prev_work;
        prev_work = work;

        float inst_rate = (dt > 0) ? (delta / dt) : 0.0f;

        rate_ema = RATE_EMA_ALPHA * inst_rate + (1.0f - RATE_EMA_ALPHA) * rate_ema;

        // === Adaptive Capacity ===
        if (now < warmup_until) {
            if (rate_ema > initial_max_rate) initial_max_rate = rate_ema;
            current_capacity = initial_max_rate;
        } else {
            calibration_done = 1;
            if (rate_ema > current_capacity * 0.95f) {
                current_capacity = current_capacity * 0.995f + rate_ema * 0.005f;
            }
        }

        float expected_rate = current_capacity;
        float current_sleep = g_h_control->throttle_sleep_ns;

        float sleep_factor = 1.0f - (current_sleep / (float)MAX_SLEEP_NS * 0.92f);
        if (sleep_factor < 0.05f) sleep_factor = 0.05f;
        expected_rate = current_capacity * sleep_factor;

        float z = expected_rate / (rate_ema + 1.0f);
        if (z > 4.0f) z = 4.0f;
        if (z < 0.2f) z = 0.2f;

        int new_sleep = BASE_SLEEP_NS;
        float error = 0.0f;

        if (now > warmup_until && z > TARGET_PM) {
            error = TARGET_PM - z;
            integral += error * (float)dt;
            if (integral > 1.2f) integral = 1.2f;
            if (integral < -1.2f) integral = -1.2f;

            float delta_q = -(KP * error + KI * integral);
            new_sleep = (int)(BASE_SLEEP_NS + delta_q * 115000.0f);
            if (new_sleep < BASE_SLEEP_NS) new_sleep = BASE_SLEEP_NS;
            if (new_sleep > g_h_control->max_sleep_ns) new_sleep = g_h_control->max_sleep_ns;
        } else {
            integral = 0.0f;
        }

        g_h_control->throttle_sleep_ns = new_sleep;
        g_h_control->current_pm = z;
        g_h_control->z_estimate = z;
        g_h_control->dzdt_estimate = dzdt_hat;
        g_h_control->pm_error = error;

        if (csv) {
            fprintf(csv, "%.3f,%.4f,%.1f,%.1f,%d,%.4f\n",
                    (now - warmup_until)/1e6, z, rate_ema, expected_rate, new_sleep, error);
        }
    }

    if (csv) fclose(csv);
    printf("[INFO] initial_max=%.1f  final_capacity=%.1f\n", initial_max_rate, current_capacity);
    return NULL;
}

int main(int argc, char** argv) {
    if (argc >= 4) { KP = atof(argv[1]); KI = atof(argv[2]); TARGET_PM = atof(argv[3]); }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic v26.5 OBSERVE] Real workload mode - no cap\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    cudaSetDevice(0);
    cudaHostAlloc((void**)&g_h_control, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d_control, g_h_control, 0);

    memset(g_h_control, 0, sizeof(GPUControlData));
    g_h_control->target_pm = TARGET_PM;
    g_h_control->max_sleep_ns = MAX_SLEEP_NS;
    g_h_control->throttle_sleep_ns = BASE_SLEEP_NS;

    cudaMalloc((void**)&g_sm_counters, NUM_SM * sizeof(unsigned long long));

    dim3 block(THREADS_PER_BLOCK);
    gpu_persistent_kernel<<<NUM_SM, block>>>(g_sm_counters, g_d_control);

    pthread_t ctrl_thread;
    pthread_create(&ctrl_thread, NULL, control_loop, NULL);

    for (int i = 0; i < 300; i++) {
        usleep(1000000);
        if (!g_running) break;
        printf("[TEL] t=%3ds | Z=%.3f | sleep=%6d | capacity=%.0f\n",
               i+1, g_h_control->z_estimate, g_h_control->throttle_sleep_ns, current_capacity);
    }

    g_running = 0;
    g_h_control->control_flags |= FLAG_STOP;
    usleep(100000);
    pthread_join(ctrl_thread, NULL);
    cudaDeviceSynchronize();

    printf("\n[GPUTronic v26.5 OBSERVE] Run complete. CSV: gputronic_log.csv\n");
    cudaFree(g_sm_counters);
    cudaFreeHost(g_h_control);
    return 0;
}