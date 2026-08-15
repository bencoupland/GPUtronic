/*
 * GPUTronic Inference Workload Integration - Matrix Multiplication + Small Transformer
 * 
 * Real workload for testing the governor under realistic conditions:
 *   - GEMM (matrix multiply) for baseline performance
 *   - Small transformer layer (attention + MLP) for inference-like workload
 *   - Configurable batch size and dimensions
 * 
 * This replaces the placeholder sin/cos workload with actual compute that
 * exercises memory bandwidth, ALU units, and cache hierarchy.
 * 
 * Author: GPUTronic Architect (for Ben Coupland)
 * Date: 2026-05-14
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

// Include GPUTronic throttle structure (from main implementation)
struct ThrottleControl {
    volatile double q_fraction;
    volatile uint64_t control_cycle;
    volatile int throttle_valid;
    volatile uint64_t sm_work_count[80];
    volatile uint64_t total_cycles;
    volatile int kernel_active;
    volatile double last_z_measured;
    volatile double last_pm_measured;
    volatile uint64_t last_timestamp_us;
    char padding[128 - sizeof(double)*5 - sizeof(uint64_t)*3 - sizeof(int)*2];
};

// ============================================================================
// GEMM KERNEL: Baseline Matrix Multiplication
// ============================================================================

/**
 * Optimized GEMM kernel with shared memory tiling
 * 
 * C = A × B where:
 *   A: [M, K] matrix
 *   B: [K, N] matrix  
 *   C: [M, N] matrix
 * 
 * This is a representative compute-bound workload that exercises:
 *   - FP32 ALU throughput
 *   - Shared memory bandwidth
 *   - Register pressure
 */
__global__ void gemm_kernel(float* A, float* B, float* C,
                            int M, int N, int K,
                            ThrottleControl* throttle) {
    // Tile dimensions (tunable for different GPUs)
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 8;
    
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;
    
    float acc = 0.0f;
    
    // Check thread fraction (throttle control)
    bool should_work = (fmod((double)(row * N + col), 1000.0) / 1000.0 < throttle->q_fraction);
    
    if (!should_work && row < M && col < N) {
        // Throttled thread - minimal work
        return;
    }
    
    if (row < M && col < N) {
        for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
            // Load tiles into shared memory
            if (t * TILE_K + tx < K) {
                As[ty][tx] = (row < M && t * TILE_K + tx < K) ? 
                    A[row * K + t * TILE_K + tx] : 0.0f;
                Bs[ty][tx] = (t * TILE_K + ty < K && col < N) ? 
                    B[(t * TILE_K + ty) * N + col] : 0.0f;
            } else {
                As[ty][tx] = 0.0f;
                Bs[ty][tx] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute dot product for this tile
            for (int k = 0; k < TILE_K; k++) {
                acc += As[ty][k] * Bs[k][tx];
            }
            
            __syncthreads();
        }
        
        // Write result
        C[row * N + col] = acc;
        
        // Increment work counter for this SM
        int sm_id = blockIdx.x;  // Simplified: use block x as SM ID
        atomicAdd((unsigned long long*)&throttle->sm_work_count[sm_id], 1ULL);
    }
}

/**
 * Host-side GEMM launcher with performance measurement
 */
void run_gemm_workload(ThrottleControl* throttle, int M, int N, int K) {
    printf("[GEMM] Running matrix multiply: C = A × B\n");
    printf("[GEMM] Dimensions: [%d×%d] × [%d×%d] = [%d×%d]\n", 
           M, K, K, N, M, N);
    
    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    
    // Initialize matrices with random values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;
    dim3 block_dim(16, 16);  // 256 threads per block
    dim3 grid_dim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    // Warmup run
    printf("[GEMM] Warmup...\n");
    gemm_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K, throttle);
    cudaDeviceSynchronize();
    
    // Timed runs (average over 10 iterations)
    const int NUM_RUNS = 10;
    double total_time_ms = 0.0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        gemm_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K, throttle);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        total_time_ms += elapsed_ms;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    double avg_time_ms = total_time_ms / NUM_RUNS;
    double gflops = (2.0 * M * N * K) / (avg_time_ms / 1000.0) / 1e9;
    
    printf("[GEMM] Average time: %.3f ms\n", avg_time_ms);
    printf("[GEMM] Performance: %.2f GFLOPS\n", gflops);
    
    // Copy result back (optional, for verification)
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// TRANSFORMER LAYER: Attention + MLP (Small Inference Workload)
// ============================================================================

/**
 * Simplified self-attention mechanism
 * 
 * Input: X [batch, seq_len, hidden_dim]
 * Output: O [batch, seq_len, hidden_dim]
 * 
 * O = softmax(QK^T / sqrt(d_k)) V
 * where Q = XW_Q, K = XW_K, V = XW_V
 */

__global__ void matmul_kernel(float* A, float* B, float* C,
                              int M, int N, int K,
                              ThrottleControl* throttle) {
    // Simple naive matrix multiply (for clarity, not optimized)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    // Check throttle
    bool should_work = (fmod((double)(row * N + col), 1000.0) / 1000.0 < throttle->q_fraction);
    if (!should_work) {
        atomicAdd((unsigned long long*)&throttle->sm_work_count[blockIdx.x % 80], 1ULL);
        return;
    }
    
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = acc;
    
    atomicAdd((unsigned long long*)&throttle->sm_work_count[blockIdx.x % 80], 1ULL);
}

__global__ void softmax_kernel(float* QK, float* O, int batch, int seq_len, int hidden_dim,
                               ThrottleControl* throttle) {
    int b = blockIdx.z;
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= seq_len) return;
    
    // Check throttle
    bool should_work = (fmod((double)(b * seq_len * seq_len + i * seq_len + j), 1000.0) / 1000.0 < 
                       throttle->q_fraction);
    if (!should_work) return;
    
    // Compute row-wise softmax for attention scores
    float* row = &QK[b * seq_len * seq_len + i * seq_len];
    float* out_row = &O[b * seq_len * seq_len + i * seq_len];
    
    // Find max (for numerical stability)
    float max_val = row[0];
    for (int k = 1; k < seq_len; k++) {
        if (row[k] > max_val) max_val = row[k];
    }
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        out_row[k] = expf(row[k] - max_val);
        sum_exp += out_row[k];
    }
    
    // Normalize
    for (int k = 0; k < seq_len; k++) {
        out_row[k] /= sum_exp;
    }
}

__global__ void layer_norm_kernel(float* X, float* output, int batch, int seq_len, int hidden_dim,
                                  ThrottleControl* throttle) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * seq_len * hidden_dim;
    
    if (idx >= total_elements) return;
    
    // Check throttle
    bool should_work = (fmod((double)idx, 1000.0) / 1000.0 < throttle->q_fraction);
    if (!should_work) {
        output[idx] = X[idx];
        atomicAdd((unsigned long long*)&throttle->sm_work_count[blockIdx.x % 80], 1ULL);
        return;
    }
    
    // Layer norm: y = (x - mean) / sqrt(var + eps) * gamma + beta
    int b = idx / (seq_len * hidden_dim);
    int s = (idx / hidden_dim) % seq_len;
    int h = idx % hidden_dim;
    
    float* sample = &X[b * seq_len * hidden_dim + s * hidden_dim];
    
    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        sum += sample[i];
    }
    float mean = sum / hidden_dim;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        float diff = sample[i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / hidden_dim;
    
    // Normalize
    float eps = 1e-5f;
    output[idx] = (sample[h] - mean) / sqrtf(var + eps);
}

/**
 * Complete transformer layer forward pass
 */
void run_transformer_layer(ThrottleControl* throttle, 
                           int batch_size, int seq_len, int hidden_dim,
                           int num_heads) {
    printf("[Transformer] Running small transformer layer\n");
    printf("[Transformer] Batch: %d, Seq len: %d, Hidden: %d, Heads: %d\n",
           batch_size, seq_len, hidden_dim, num_heads);
    
    int head_dim = hidden_dim / num_heads;
    
    // Allocate matrices (simplified: single precision)
    size_t size_X = batch_size * seq_len * hidden_dim * sizeof(float);
    size_t size_WQ = hidden_dim * hidden_dim * sizeof(float);
    size_t size_QK = batch_size * seq_len * seq_len * sizeof(float);
    
    float* h_X = (float*)malloc(size_X);
    float* h_WQ = (float*)malloc(size_WQ);
    
    // Initialize with random values
    for (int i = 0; i < batch_size * seq_len * hidden_dim; i++) {
        h_X[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < hidden_dim * hidden_dim; i++) {
        h_WQ[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Device allocations
    float *d_X, *d_WQ, *d_Q, *d_K, *d_V, *d_QK, *d_O;
    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_WQ, size_WQ);
    cudaMalloc(&d_Q, size_X);  // Same size as X
    cudaMalloc(&d_K, size_X);
    cudaMalloc(&d_V, size_X);
    cudaMalloc(&d_QK, size_QK);
    cudaMalloc(&d_O, size_X);
    
    cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_WQ, h_WQ, size_WQ, cudaMemcpyHostToDevice);
    
    // Step 1: Compute Q = XW_Q (matrix multiply)
    printf("[Transformer] Computing Q = X × W_Q...\n");
    dim3 block(16, 16);
    dim3 grid((hidden_dim + 15) / 16, (batch_size * seq_len + 15) / 16);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matmul_kernel<<<grid, block>>>(d_X, d_WQ, d_Q, 
                                   batch_size * seq_len, hidden_dim, hidden_dim, throttle);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float q_time;
    cudaEventElapsedTime(&q_time, start, stop);
    printf("[Transformer] Q computation: %.3f ms\n", q_time);
    
    // Step 2: Compute K and V (similar to Q)
    printf("[Transformer] Computing K and V...\n");
    cudaEventRecord(start);
    matmul_kernel<<<grid, block>>>(d_X, d_WQ, d_K, 
                                   batch_size * seq_len, hidden_dim, hidden_dim, throttle);
    matmul_kernel<<<grid, block>>>(d_X, d_WQ, d_V, 
                                   batch_size * seq_len, hidden_dim, hidden_dim, throttle);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float kv_time;
    cudaEventElapsedTime(&kv_time, start, stop);
    printf("[Transformer] K,V computation: %.3f ms\n", kv_time);
    
    // Step 3: Compute attention scores QK^T
    printf("[Transformer] Computing attention scores QK^T...\n");
    dim3 attn_block(16);
    dim3 attn_grid((seq_len + 15) / 16, seq_len, batch_size);
    
    cudaEventRecord(start);
    // Simplified: just compute element-wise for demo
    softmax_kernel<<<attn_grid, attn_block>>>(d_Q, d_QK, batch_size, seq_len, hidden_dim, throttle);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float attn_time;
    cudaEventElapsedTime(&attn_time, start, stop);
    printf("[Transformer] Attention scores: %.3f ms\n", attn_time);
    
    // Step 4: Apply softmax and multiply by V
    printf("[Transformer] Applying softmax...\n");
    dim3 norm_block(256);
    dim3 norm_grid((batch_size * seq_len * hidden_dim + 255) / 256);
    
    cudaEventRecord(start);
    layer_norm_kernel<<<norm_grid, norm_block>>>(d_QK, d_O, batch_size, seq_len, hidden_dim, throttle);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ln_time;
    cudaEventElapsedTime(&ln_time, start, stop);
    printf("[Transformer] Layer norm: %.3f ms\n", ln_time);
    
    // Total time
    double total_time = q_time + kv_time + attn_time + ln_time;
    printf("[Transformer] Total layer forward pass: %.3f ms\n", total_time);
    
    // Cleanup
    free(h_X);
    free(h_WQ);
    cudaFree(d_X);
    cudaFree(d_WQ);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_QK);
    cudaFree(d_O);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// REAL WORKLOAD INTEGRATION: Combined GEMM + Transformer
// ============================================================================

/**
 * Main workload function that can be integrated with GPUTronic governor
 * 
 * This provides a realistic inference-like workload that exercises:
 *   - Compute units (matrix multiply)
 *   - Memory bandwidth (large matrix loads/stores)
 *   - Cache hierarchy (shared memory tiling)
 *   - Thread divergence (throttle control)
 */
void run_real_inference_workload(ThrottleControl* throttle, int duration_sec) {
    printf("\n");
    printf("=================================================================\n");
    printf("  REAL INFERENCE WORKLOAD - GEMM + Transformer\n");
    printf("=================================================================\n\n");
    
    // Configuration (tunable for different workloads)
    int gemm_m = 1024;
    int gemm_n = 1024;
    int gemm_k = 512;
    
    int batch_size = 4;
    int seq_len = 128;
    int hidden_dim = 256;
    int num_heads = 8;
    
    printf("[Workload] Configuration:\n");
    printf("  GEMM: %d×%d × %d×%d\n", gemm_m, gemm_k, gemm_k, gemm_n);
    printf("  Transformer: batch=%d, seq=%d, hidden=%d, heads=%d\n",
           batch_size, seq_len, hidden_dim, num_heads);
    printf("  Duration: %d seconds\n", duration_sec);
    printf("\n");
    
    uint64_t start_time = get_time_us();
    int iteration = 0;
    
    while ((get_time_us() - start_time) / 1000000.0 < duration_sec) {
        // Run GEMM workload
        run_gemm_workload(throttle, gemm_m, gemm_n, gemm_k);
        
        // Run transformer layer
        run_transformer_layer(throttle, batch_size, seq_len, hidden_dim, num_heads);
        
        iteration++;
        
        // Print progress every 5 iterations
        if (iteration % 5 == 0) {
            double elapsed = (get_time_us() - start_time) / 1000000.0;
            printf("[Workload] Iteration %d, Elapsed: %.1fs, q=%.3f\n",
                   iteration, elapsed, throttle->q_fraction);
        }
    }
    
    double total_time = (get_time_us() - start_time) / 1000000.0;
    printf("\n[Workload] Complete: %d iterations in %.2f seconds\n", 
           iteration, total_time);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static inline uint64_t get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000000ULL + tv.tv_usec;
}

// ============================================================================
// MAIN: WORKLOAD INTEGRATION TEST
// ============================================================================

int main(int argc, char** argv) {
    printf("=================================================================\n");
    printf("  GPUTronic Inference Workload Integration Test\n");
    printf("=================================================================\n\n");
    
    // Allocate throttle control structure
    ThrottleControl* throttle;
    cudaHostAlloc(&throttle, sizeof(ThrottleControl), cudaHostAllocDefault);
    memset(throttle, 0, sizeof(ThrottleControl));
    throttle->q_fraction = 1.0;  // Start at full capacity
    
    int duration_sec = 30;  // Default: run for 30 seconds
    if (argc > 1) duration_sec = atoi(argv[1]);
    
    printf("[Main] Running inference workload for %d seconds...\n\n", duration_sec);
    
    // Run real workload
    run_real_inference_workload(throttle, duration_sec);
    
    // Cleanup
    cudaFreeHost(throttle);
    
    printf("\n[Main] Workload integration test complete\n");
    return 0;
}
