// z_margin_harness.cpp
// Small diagnostic harness for Z=1.0 tracking with histogram + Q command logging
// Compile: g++ -o z_margin_harness z_margin_harness.cpp -pthread
// Usage: ./z_margin_harness <kp> <ki> <duration_seconds>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <string.h>

volatile int g_running = 1;

void handle_sig(int) { g_running = 0; }

struct GPUControlData {
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

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <kp> <ki> <seconds>\n", argv[0]);
        return 1;
    }

    float kp = atof(argv[1]);
    float ki = atof(argv[2]);
    int duration = atoi(argv[3]);

    signal(SIGINT, handle_sig);
    signal(SIGTERM, handle_sig);

    printf("[HARNESS] Z=1.0 margin test | Kp=%.3f Ki=%.4f | %ds run\n", kp, ki, duration);
    printf("[HARNESS] Logging Z histogram (bins 0.8-1.4) and throttle_sleep_ns (Q proxy)\n\n");

    // Simulated control loop (replace with real shared memory attach when integrated)
    const float dt = 10e-6f;
    float integral = 0.0f;
    float z_hat = 1.0f;
    float dzdt_hat = 0.0f;
    float target = 1.0f;

    const int bins = 12;           // 0.80, 0.85 ... 1.35+
    int histogram[bins] = {0};
    long long total_samples = 0;
    double sum_z = 0.0;
    double sum_sq_z = 0.0;
    int min_sleep = 999999;
    int max_sleep = 0;

    double start = 0;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    start = ts.tv_sec + ts.tv_nsec / 1e9;

    while (g_running) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double now = ts.tv_sec + ts.tv_nsec / 1e9;
        if (now - start > duration) break;

        // In real version: read g_h->z_estimate and throttle_sleep_ns from shared mem
        // For now simulate realistic Z around 1.0 with noise
        float measured_z = 1.0f + 0.08f * sinf(now * 3.7f) + 0.03f * ((rand() % 1000) - 500) / 500.0f;
        if (measured_z < 0.7f) measured_z = 0.7f;
        if (measured_z > 1.6f) measured_z = 1.6f;

        // Simple PI update (mirrors governor logic)
        float error = target - measured_z;
        integral += error * dt * 100000.0f;
        if (integral > 0.8f) integral = 0.8f;
        if (integral < -0.8f) integral = -0.8f;
        float delta_q = -(kp * error + ki * integral);
        int sleep_ns = (int)(delta_q * 800.0f);
        if (sleep_ns < 0) sleep_ns = 0;
        if (sleep_ns > 800000) sleep_ns = 800000;

        // Histogram
        int bin = (int)((measured_z - 0.80f) * 20.0f);
        if (bin < 0) bin = 0;
        if (bin >= bins) bin = bins - 1;
        histogram[bin]++;
        total_samples++;
        sum_z += measured_z;
        sum_sq_z += measured_z * measured_z;

        if (sleep_ns < min_sleep) min_sleep = sleep_ns;
        if (sleep_ns > max_sleep) max_sleep = sleep_ns;

        usleep(200);  // ~5 kHz sample rate for harness
    }

    // Results
    printf("\n=== Z=1.0 Tracking Results (Kp=%.3f, Ki=%.4f) ===\n", kp, ki);
    printf("Samples: %lld\n", total_samples);
    if (total_samples > 0) {
        float mean = sum_z / total_samples;
        float variance = (sum_sq_z / total_samples) - (mean * mean);
        printf("Mean Z: %.4f   StdDev: %.4f\n", mean, sqrtf(variance));
    }
    printf("Throttle range: %d - %d ns\n\n", min_sleep, max_sleep);

    printf("Z Histogram (0.80-1.35+):\n");
    for (int i = 0; i < bins; i++) {
        float z_center = 0.80f + i * 0.05f;
        int pct = (int)(histogram[i] * 100.0f / total_samples);
        printf("  %.2f: %5d (%3d%%) %s\n", z_center, histogram[i], pct, 
               (z_center >= 0.95f && z_center <= 1.05f) ? "<-- target band" : "");
    }

    printf("\n[INFO] Closer tracking to Z=1.0 requires higher Kp/Ki until stability limit.\n");
    return 0;
}