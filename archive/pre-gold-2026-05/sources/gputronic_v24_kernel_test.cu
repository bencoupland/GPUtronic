// Kernel launch + bursty workload test for v24 on RTX 5080
// No control loop, no nanosleep throttling yet.

#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

#define NUM_SM              84
#define THREADS_PER_BLOCK   64
#define WORK_UNITS_PER_THREAD 8192
#define TILE_K              16

struct BurstyParams {
    unsigned int burst_active;
    float intensity_factor;
};

__device__ __forceinline__ void bursty_workload(float* __restrict__ acc, int sm_id, int iter, float intensity) {
    float a[TILE_K], b[TILE_K];
    #pragma unroll
    for (int k = 0; k < TILE_K; k++) {
        a[k] = sinf((float)(iter * 17 + k * 31 + sm_id) * 0.0174532925f) * intensity;
        b[k] = cosf((float)(iter * 23 + k * 19 + sm_id) * 0.0174532925f) * intensity;
    }
    float local_acc = *acc;
    #pragma unroll
    for (int k = 0; k < TILE_K; k++) {
        local_acc = fmaf(a[k], b[k], local_acc);
    }
    *acc = local_acc;
}

__global__ void test_kernel(unsigned long long* counters, BurstyParams* params, int num_sm) {
    int sm_id = blockIdx.x;
    unsigned long long local = 0;

    for (int iter = 0; iter < 500; iter++) {   // limited iterations for test
        float acc = 0.0f;
        float intensity = params->intensity_factor;
        if (!params->burst_active) intensity *= 0.3f;

        for (int i = 0; i < WORK_UNITS_PER_THREAD / 64; i++) {
            bursty_workload(&acc, sm_id, i, intensity);
        }
        atomicAdd(&counters[sm_id], 1ULL);
        local++;
        __nanosleep(100);   // light sleep to avoid spinning too hard
    }
}

int main() {
    printf("=== v24 Kernel Launch Test ===\n");

    cudaSetDevice(0);

    unsigned long long* d_counters = NULL;
    cudaMalloc(&d_counters, NUM_SM * sizeof(unsigned long long));

    BurstyParams h_params = {1, 1.0f};
    BurstyParams* d_params = NULL;
    cudaMalloc(&d_params, sizeof(BurstyParams));
    cudaMemcpy(d_params, &h_params, sizeof(BurstyParams), cudaMemcpyHostToDevice);

    printf("[OK] Memory allocated. Launching kernel on 84 SMs...\n");

    dim3 block(THREADS_PER_BLOCK);
    test_kernel<<<NUM_SM, block>>>(d_counters, d_params, NUM_SM);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[FAIL] Kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("[OK] Kernel launched.\n");

    // Let it run for a few seconds
    sleep(4);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[FAIL] After sync: %s\n", cudaGetErrorString(err));
        return 1;
    }

    unsigned long long h_counters[NUM_SM];
    cudaMemcpy(h_counters, d_counters, NUM_SM * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long total = 0;
    for (int i = 0; i < NUM_SM; i++) total += h_counters[i];
    printf("[OK] Kernel completed. Total work pulses = %llu\n", total);
    printf("[SUCCESS] Persistent kernel + bursty workload runs on RTX 5080.\n");

    cudaFree(d_counters);
    cudaFree(d_params);
    return 0;
}