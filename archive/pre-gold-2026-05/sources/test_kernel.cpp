#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gpu_workload_kernel(ThrottleControl* throttle) {
    int tid = threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    
    double q_fraction = throttle->q_fraction;
    bool should_work = (fmod((double)tid, 1000.0) / 1000.0 < q_fraction);
    
    if (!should_work) {
        __nanosleep(100);
        return;
    }
    
    atomicAdd(&total_work_completed, WORK_UNITS_PER_THREAD);
}

int main() {
    ThrottleControl* h_throttle = (ThrottleControl*)malloc(sizeof(ThrottleControl));
    ThrottleControl* d_throttle;
    cudaMallocHost(&h_throttle, sizeof(ThrottleControl));
    cudaHostRegister(h_throttle, sizeof(ThrottleControl), 0);
    cudaMalloc(&d_throttle, sizeof(ThrottleControl));
    
    h_throttle->q_fraction = 0.5f;
    cudaMemcpy(d_throttle, h_throttle, sizeof(ThrottleControl), cudaMemcpyHostToDevice);
    h_throttle->running_flag = 1;
    
    printf("Launching kernel...\n");
    dim3 block(256);
    gpu_workload_kernel<<<1, block>>>(d_throttle);
    
    cudaError_t err = cudaGetLastError();
    printf("Kernel error code: %d (%s)\n", err, cudaGetErrorString(err));
    
    cudaFreeHost(h_throttle);
    cudaFree(d_throttle);
    return 0;
}
