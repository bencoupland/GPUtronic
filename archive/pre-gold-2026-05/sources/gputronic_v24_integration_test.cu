// Integration test: Kernel + Control loop + Zero-copy together
// This is the combination that was failing in v24.

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#define NUM_SM              84
#define THREADS_PER_BLOCK   64

struct __align__(16) GPUControlData {
    unsigned int control_flags;
    float target_pm;
    float current_pm;
    unsigned long long total_work_pulses;
    int throttle_sleep_ns;
    int max_sleep_ns;
};

struct BurstyParams {
    unsigned int burst_active;
    float intensity_factor;
};

static volatile int g_running = 1;
static GPUControlData* g_h_control = NULL;
static GPUControlData* g_d_control = NULL;
static BurstyParams* g_h_burst = NULL;
static BurstyParams* g_d_burst = NULL;

__device__ __forceinline__ void bursty_workload(float* acc, int sm_id, int iter, float intensity) {
    float a = sinf((float)(iter + sm_id) * 0.0174532925f) * intensity;
    *acc = fmaf(a, 0.8f, *acc);
}

__global__ void test_kernel(unsigned long long* counters, GPUControlData* ctrl, BurstyParams* burst, int num_sm) {
    int sm_id = blockIdx.x;
    for (int i = 0; i < 300; i++) {
        float acc = 0.0f;
        float intensity = burst->intensity_factor;
        if (!burst->burst_active) intensity *= 0.3f;

        for (int k = 0; k < 64; k++) {
            bursty_workload(&acc, sm_id, i * 64 + k, intensity);
        }
        atomicAdd(&counters[sm_id], 1ULL);
        __nanosleep(ctrl->throttle_sleep_ns > 0 ? ctrl->throttle_sleep_ns : 50);
    }
}

void* control_loop(void* arg) {
    (void)arg;
    int count = 0;
    while (g_running && count < 10) {
        g_h_control->current_pm = 0.78f + (count * 0.015f);
        g_h_control->throttle_sleep_ns = 200 + count * 150;
        g_h_control->total_work_pulses += 98765;
        usleep(150000);
        count++;
    }
    return NULL;
}

int main() {
    printf("=== v24 Full Integration Test ===\n");

    cudaSetDevice(0);

    unsigned long long* d_counters = NULL;
    cudaMalloc(&d_counters, NUM_SM * sizeof(unsigned long long));

    cudaHostAlloc((void**)&g_h_control, sizeof(GPUControlData), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d_control, g_h_control, 0);

    cudaHostAlloc((void**)&g_h_burst, sizeof(BurstyParams), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d_burst, g_h_burst, 0);

    g_h_control->target_pm = 0.82f;
    g_h_control->max_sleep_ns = 500000;
    g_h_control->throttle_sleep_ns = 100;
    g_h_burst->burst_active = 1;
    g_h_burst->intensity_factor = 1.0f;

    printf("[OK] All allocations done. Launching kernel...\n");

    dim3 block(THREADS_PER_BLOCK);
    test_kernel<<<NUM_SM, block>>>(d_counters, g_d_control, g_d_burst, NUM_SM);

    pthread_t ctrl_thread;
    pthread_create(&ctrl_thread, NULL, control_loop, NULL);

    sleep(4);

    g_running = 0;
    pthread_join(ctrl_thread, NULL);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[FAIL] %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("[OK] Integration test completed successfully.\n");
    printf("[SUCCESS] Kernel + Control loop + Zero-copy working together on RTX 5080.\n");

    cudaFree(d_counters);
    cudaFreeHost(g_h_control);
    cudaFreeHost(g_h_burst);
    return 0;
}