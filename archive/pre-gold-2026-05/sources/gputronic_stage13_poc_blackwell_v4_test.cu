/**
 * GPUtronic Stage 13 — Blackwell PoC v4 Test (Zero-copy disabled for constrained systems)
 * Warp Scheduling: 2 blocks/SM × 64 threads = 4 warps SM (~8% of max 48)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>
#include <fstream>

#ifdef __linux__
#include <unistd.h>
#endif

// Blackwell cc 12.0: Max 48 concurrent warps/SM
constexpr int   BLOCK_THREADS     = 64;
constexpr int   BLOCKS_PER_SM     = 2;

__global__ void g_scag_engine_test(
    float q,
    uint64_t* __restrict__ d_global_work,
    uint64_t* __restrict__ d_local_work,
    int workload_mode
) {
    int sm_idx = blockIdx.x;
    int tid = threadIdx.x;

    uint64_t local_accumulator = 0;
    const int flush_interval = 2000; 
    int iteration = 0;

    while (true) {
        // Simple loop counter to simulate work
        float dummy = q * 0.5f + sinf((float)iteration);
        for(int i=0; i<100; ++i) {
            dummy += cosf((float)(sm_idx + tid));
        }
        
        local_accumulator++;
        d_local_work[sm_idx] = local_accumulator;

        if (iteration % flush_interval == 0) {
            atomicAdd((unsigned long long*)d_global_work, (unsigned long long)flush_interval);
        }

        // Simple shutdown check
        if (iteration > 1000000) break;
        iteration++;
    }
}

int main() {
    std::cout << "=== GPUtronic v4 Test Mode ===\\n";
    std::cout << "Warp Scheduling: " << BLOCKS_PER_SM << " blocks/SM × " 
              << BLOCK_THREADS << " threads = " << (BLOCKS_PER_SM * 2) 
              << " warps SM\\n\\n";

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "[ERROR] No CUDA devices found!\\n";
        return 1;
    }
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    int sm_count = props.multiProcessorCount;
    
    std::cout << "GPU: " << props.name << "\\n";
    std::cout << "SM Count: " << sm_count << "\\n\\n";

    // Simple test with regular host memory
    float h_q = 1.0f;
    uint64_t* h_global_work = new uint64_t;
    *h_global_work = 0;

    uint64_t* d_global_work, *d_local_work;
    cudaError_t err = cudaMalloc(&d_global_work, sizeof(uint64_t));
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc global failed: " << cudaGetErrorString(err) << "\\n";
        delete h_global_work;
        return 1;
    }
    
    err = cudaMalloc(&d_local_work, sm_count * sizeof(uint64_t));
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc local failed: " << cudaGetErrorString(err) << "\\n";
        cudaFree(d_global_work);
        delete h_global_work;
        return 1;
    }

    int total_blocks = sm_count * BLOCKS_PER_SM;
    std::cout << "[LAUNCH] " << total_blocks << " blocks (" << (BLOCKS_PER_SM * 2) 
              << " warps/SM)...\\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    g_scag_engine_test<<<total_blocks, BLOCK_THREADS>>>(
        h_q, d_global_work, d_local_work, 0
    );
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double dt = std::chrono::duration<double>(end - start).count();
    std::cout << "[TEST] Work completed: " << *h_global_work 
              << " in " << (dt*1000) << " ms\\n";

    cudaFree(d_global_work);
    cudaFree(d_local_work);
    delete h_global_work;

    std::cout << "[OK] Test complete\\n";
    return 0;
}
