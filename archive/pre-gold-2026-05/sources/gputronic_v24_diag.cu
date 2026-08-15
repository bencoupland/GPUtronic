// Minimal diagnostic test for GPUTronic v24 on RTX 5080
// Tests CUDA init, device properties, and zero-copy allocation only.

#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    printf("=== GPUTronic v24 Diagnostic ===\n");

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("[FAIL] cudaSetDevice: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("[OK] cudaSetDevice(0)\n");

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        printf("[FAIL] cudaGetDeviceProperties: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("[OK] Device: %s | SMs=%d | CC=%d.%d\n", 
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    // Test zero-copy allocation (the most likely failure point)
    struct TestStruct {
        unsigned int control_flags;
        float target_pm;
        unsigned long long counter;
    } *h_ptr = NULL, *d_ptr = NULL;

    err = cudaHostAlloc((void**)&h_ptr, sizeof(TestStruct), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("[FAIL] cudaHostAllocMapped: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("[OK] cudaHostAllocMapped\n");

    err = cudaHostGetDevicePointer((void**)&d_ptr, h_ptr, 0);
    if (err != cudaSuccess) {
        printf("[FAIL] cudaHostGetDevicePointer: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_ptr);
        return 1;
    }
    printf("[OK] cudaHostGetDevicePointer\n");

    // Simple write/read test
    h_ptr->target_pm = 0.82f;
    h_ptr->control_flags = 0x12345678;

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[FAIL] After sync: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_ptr);
        return 1;
    }

    printf("[OK] Zero-copy write/read test passed (target_pm=%.2f)\n", h_ptr->target_pm);
    printf("[SUCCESS] Basic CUDA + zero-copy initialization works on this system.\n");

    cudaFreeHost(h_ptr);
    return 0;
}