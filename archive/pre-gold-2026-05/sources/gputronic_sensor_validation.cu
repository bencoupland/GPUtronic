// =============================================================================
// GPUTronic Sensor Validation Test — Per-SM Atomic Counters + Host Accumulator
// Target: sm_90 (Ada Lovelace / Blackwell) with __nanosleep support
// Purpose: Validate sensor (work counters) works correctly before control loop
//
// Blackwell sm_120 Occupancy Analysis:
// - Max concurrent warps per SM = 48 (cc 12.0)
// - Warp size = 32 threads
// - With 64 threads/block → 2 warps/block
// - Max blocks/SM = 48 ÷ 2 = 24 blocks
// - Total blocks = 84 SMs × 24 = 2016 blocks
// =============================================================================
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ============================================================================
// BLACKWELL sm_120 CONFIGURATION (RTX 5080)
// ============================================================================

#define NUM_SM              84        // RTX 5080 has 84 SMs
#define WARP_SIZE           32        // Standard warp size for all NVIDIA GPUs
#define MAX_WARPS_PER_SM    48        // Blackwell cc12.0 max concurrent warps/SM

// Occupancy calculation: with 64 threads/block = 2 warps/block
// max_blocks_per_sm = MAX_WARPS_PER_SM / (threads_per_block / WARP_SIZE)
#define THREADS_PER_BLOCK   64        // Optimal for occupancy + __nanosleep support
#define BLOCKS_PER_SM       (MAX_WARPS_PER_SM / (THREADS_PER_BLOCK / WARP_SIZE))  // = 24
#define TOTAL_BLOCKS        (NUM_SM * BLOCKS_PER_SM)                              // = 2016

// Fixed iterations per thread for validation
#define WORK_UNITS_PER_THREAD 20000

// Timing helper
inline double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

// Per-SM atomic counter array (device memory)
static unsigned long long* g_sm_counters = NULL;

// Host mirror for reading counters back
static unsigned long long g_host_sm_counters[NUM_SM] = {0};

// =============================================================================
// Kernels
// =============================================================================

// Workload kernel with per-SM atomic counters
__global__ void gpu_workload_kernel(unsigned long long* sm_counters, int num_sm) {
    // Map this block to its SM index (blockIdx.x % num_sm)
    int sm_id = blockIdx.x % num_sm;
    
    // Each thread does WORK_UNITS_PER_THREAD iterations
    // Increment atomic counter every few units to batch work
    for (int i = 0; i < WORK_UNITS_PER_THREAD; i++) {
        // Do simple work (prevent compiler optimization)
        volatile double result = sin(0.1) * cos(0.2);
        
        // AtomicAdd per-SM counter every 32 iterations (warp-aligned batching)
        if ((i + 1) % 32 == 0) {
            atomicAdd(&sm_counters[sm_id], 32ULL);
        }
    }
    
    // Handle remaining work
    int remainder = WORK_UNITS_PER_THREAD % 32;
    if (remainder > 0) {
        // Do final batch
        for (int i = 0; i < remainder; i++) {
            volatile double result = sin(0.1) * cos(0.2);
        }
        atomicAdd(&sm_counters[sm_id], (unsigned long long)remainder);
    }
}

// Reset all per-SM counters to zero
__global__ void reset_sm_counters(unsigned long long* sm_counters, int num_sm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sm) {
        sm_counters[idx] = 0ULL;
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic] Sensor Validation Test — Per-SM Atomic Counters\n");
    printf("[GPUTronic] Blackwell sm_90 Configuration:\n");
    printf("        SMs: %d | Max warps/SM: %d | Warp size: %d\n", 
           NUM_SM, MAX_WARPS_PER_SM, WARP_SIZE);
    printf("        Threads/block: %d (%d warps) | Blocks/SM: %d\n",
           THREADS_PER_BLOCK, THREADS_PER_BLOCK/WARP_SIZE, BLOCKS_PER_SM);
    printf("[GPUTronic] Total blocks: %d | Total threads: %d\n",
           TOTAL_BLOCKS, TOTAL_BLOCKS * THREADS_PER_BLOCK);
    
    unsigned long long expected_pulses = (unsigned long long)TOTAL_BLOCKS * 
                                         (unsigned long long)THREADS_PER_BLOCK * 
                                         (unsigned long long)WORK_UNITS_PER_THREAD;
    printf("[GPUTronic] Expected pulses (Q=1.0): %llu\n", expected_pulses);
    printf("═══════════════════════════════════════════════════════════\n\n");

    // Initialize CUDA
    cudaSetDevice(0);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[GPUTronic] GPU: %s | SMs detected: %d | Compute capability: %d.%d\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    // Allocate per-SM counters on device (84 × 8 bytes = 672 bytes — tiny!)
    cudaMalloc((void**)&g_sm_counters, NUM_SM * sizeof(unsigned long long));
    
    // Initialize all SM counters to zero
    printf("[GPUTronic] Allocating %d × uint64 per-SM counter array...\n", NUM_SM);
    reset_sm_counters<<<(NUM_SM + 31) / 32, 32>>>(g_sm_counters, NUM_SM);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] Counter initialization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaDeviceSynchronize();
    printf("[GPUTronic] Per-SM counter array initialized and zeroed.\n\n");

    // Launch workload kernel
    printf("[GPUTronic] Launching workload kernel (%d blocks, %d threads/block)...\n",
           TOTAL_BLOCKS, THREADS_PER_BLOCK);
    
    dim3 block(THREADS_PER_BLOCK);
    double start_time = get_time_us();
    
    gpu_workload_kernel<<<TOTAL_BLOCKS, block>>>(g_sm_counters, NUM_SM);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Wait for kernel to complete
    printf("[GPUTronic] Waiting for kernel completion...\n");
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] Kernel sync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    double end_time = get_time_us();
    double duration_ms = (end_time - start_time) / 1000.0;

    // Read back per-SM counters
    printf("[GPUTronic] Reading per-SM counters from device to host...\n");
    cudaMemcpy(g_host_sm_counters, g_sm_counters, 
               NUM_SM * sizeof(unsigned long long), 
               cudaMemcpyDeviceToHost);
    
    // Sum all SM counters
    unsigned long long total_work = 0;
    for (int i = 0; i < NUM_SM; i++) {
        total_work += g_host_sm_counters[i];
    }

    // Print results
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("[RESULTS] Kernel execution time: %.2f ms\n", duration_ms);
    printf("[RESULTS] Per-SM counter breakdown:\n");
    
    // Print first 16 SMs
    for (int i = 0; i < 16 && i < NUM_SM; i++) {
        if (g_host_sm_counters[i] > 0) {
            printf("        SM %2d: %llu pulses\n", i, g_host_sm_counters[i]);
        }
    }
    
    // Print summary
    double efficiency = (double)total_work / expected_pulses * 100.0;
    
    printf("\n[RESULTS] Total pulses across all SMs: %llu\n", total_work);
    printf("[RESULTS] Expected pulses: %llu\n", expected_pulses);
    printf("[RESULTS] Efficiency: %.2f%%\n", efficiency);
    printf("[RESULTS] Throughput: %.2f M pulses/sec\n", 
           (double)total_work / duration_ms / 1000.0);
    
    // Validation
    printf("\n═══════════════════════════════════════════════════════════\n");
    if (efficiency >= 95.0) {
        printf("[VALIDATION] ✓ SENSOR VALIDATED — Efficiency >95%%\n");
        printf("[VALIDATION] Per-SM counters working correctly!\n");
    } else if (efficiency >= 80.0) {
        printf("[VALIDATION] ⚠ SENSOR OPERATIONAL — Efficiency %.2f%%\n", efficiency);
        printf("[VALIDATION] Slight inefficiency detected, but functional.\n");
    } else {
        printf("[VALIDATION] ✗ SENSOR ISSUE — Efficiency only %.2f%%\n", efficiency);
        printf("[VALIDATION] Expected ~100M pulses, got %llu\n", total_work);
    }
    printf("═══════════════════════════════════════════════════════════\n");

    // Cleanup
    cudaFree(g_sm_counters);

    return (efficiency >= 80.0) ? 0 : 1;
}
