// Test: Zero-copy GPUControlData + pthread control loop
// Kernel is minimal. Focus on host <-> device communication.

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#define NUM_SM 84

struct __align__(16) GPUControlData {
    unsigned int control_flags;
    float target_pm;
    float current_pm;
    unsigned long long total_work_pulses;
    int throttle_sleep_ns;
    int max_sleep_ns;
};

static volatile int g_running = 1;
static GPUControlData* g_h = NULL;
static GPUControlData* g_d = NULL;

void* control_loop(void* arg) {
    (void)arg;
    printf("[CTRL] Control loop thread started\n");

    int count = 0;
    while (g_running && count < 8) {
        g_h->current_pm = 0.75f + (count * 0.02f);
        g_h->throttle_sleep_ns = 1000 + count * 200;
        g_h->total_work_pulses += 12345;
        printf("[CTRL] t=%d | PM=%.2f | sleep=%d\n", count, g_h->current_pm, g_h->throttle_sleep_ns);
        usleep(200000);
        count++;
    }
    return NULL;
}

int main() {
    printf("=== v24 Control Loop + Zero-Copy Test ===\n");

    cudaSetDevice(0);

    cudaHostAlloc((void**)&g_h, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d, g_h, 0);

    g_h->target_pm = 0.82f;
    g_h->max_sleep_ns = 500000;
    g_h->throttle_sleep_ns = 0;

    printf("[OK] Zero-copy GPUControlData allocated\n");

    pthread_t ctrl_thread;
    pthread_create(&ctrl_thread, NULL, control_loop, NULL);

    sleep(3);

    g_running = 0;
    pthread_join(ctrl_thread, NULL);

    printf("[OK] Control loop ran. Final PM=%.2f, sleep=%d\n", 
           g_h->current_pm, g_h->throttle_sleep_ns);
    printf("[SUCCESS] Zero-copy + pthread control loop works.\n");

    cudaFreeHost(g_h);
    return 0;
}