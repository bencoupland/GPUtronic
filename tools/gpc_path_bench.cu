/* Standalone per-GPC path microbenchmark. Not linked into the Gold binary.
 * Compile: nvcc -arch=sm_120 -O3 -o build/gpc_path_bench tools/gpc_path_bench.cu
 *
 * Launches one block per SM. Blocks whose launch pass is not the requested GPC
 * return immediately. Pass is (smid % 14) / 2, which on this 5080 is the GPC id.
 * Each test runs a fixed wall-time budget and reports nanoseconds per iteration.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GOLD_WORK_ITERS 8
#define GOLD_BATCH 16

__device__ float bench_sink;

__device__ __forceinline__ unsigned long long bench_now(void) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void bench_fma(float* acc, int sm, int iter) {
    float val = *acc;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        float a = 0.11f + 0.002f * (float)((iter + k + sm) & 31);
        float b = 0.19f + 0.003f * (float)((iter * 3 + k) & 31);
        val = __fmaf_rn(a, b, val);
    }
    *acc = val;
}

/* test: 0 compute, 1 device atomic, 2 device stream, 3 mapped read, 4 mapped atomic */
__global__ void bench_kernel(int target_gpc, int test, unsigned long long budget_ns,
                             unsigned long long* d_atomic, const float* d_stream, size_t stream_floats,
                             unsigned long long* h_words, unsigned long long* h_atomic,
                             unsigned long long* out_iters, unsigned long long* out_ns,
                             unsigned int* out_sm) {
    __shared__ float s_keep[64];
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (threadIdx.x == 0) out_sm[blockIdx.x] = smid;
    /* Uniform in the block: every thread on an SM has the same smid. */
    if (((smid % 14u) / 2u) != (unsigned)target_gpc) return;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float acc = 0.1f + 0.001f * (float)smid;
    unsigned long long n = 0;
    unsigned long long t0 = bench_now();
    unsigned long long deadline = t0 + budget_ns;
    size_t cursor = ((size_t)smid * 104729u + (size_t)threadIdx.x) % (stream_floats > 0 ? stream_floats : 1);

    while (bench_now() < deadline) {
        if (test == 0) {
            for (int b = 0; b < GOLD_BATCH; b++)
                for (int i = 0; i < GOLD_WORK_ITERS; i++)
                    bench_fma(&acc, (int)smid, i + b * GOLD_WORK_ITERS);
        } else if (test == 1) {
            if (lane == 0) atomicAdd(d_atomic, 1ULL);
        } else if (test == 2) {
            float v;
            asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(v) : "l"(d_stream + cursor));
            acc += v;
            cursor += 32;
            if (cursor >= stream_floats) cursor = threadIdx.x;
        } else if (test == 3) {
            /* One mapped word, read by every thread. Same shape as the control words. */
            unsigned long long v;
            asm volatile("ld.global.u64 %0, [%1];" : "=l"(v) : "l"(h_words));
            acc += (float)v;
        } else {
            if (lane == 0) atomicAdd(h_atomic, 1ULL);
        }
        n++;
        (void)warp;
    }
    unsigned long long t1 = bench_now();
    /* Every thread's acc has to land somewhere or the compiler drops the FMAs and loads. */
    s_keep[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int i = 0; i < 64; i++) s += s_keep[i];
        bench_sink = s;
        /* Lane 0 is the thread that issues the atomic on tests 1 and 4. */
        out_iters[blockIdx.x] = n;
        out_ns[blockIdx.x] = t1 - t0;
    }
}

static double sqrt_local(double x) {
    if (x <= 0.0) return 0.0;
    double g = x;
    for (int i = 0; i < 12; i++) g = 0.5 * (g + x / g);
    return g;
}

static const char* test_name(int t) {
    switch (t) {
    case 0: return "compute";
    case 1: return "datomic";
    case 2: return "dstream";
    case 3: return "hread";
    default: return "hatomic";
    }
}

static int run_one(int gpc, int test, unsigned long long budget_ns,
                   unsigned long long* d_atomic, const float* d_stream, size_t stream_floats,
                   unsigned long long* h_words, unsigned long long* h_atomic,
                   unsigned long long* d_iters, unsigned long long* d_ns, unsigned int* d_sm,
                   unsigned long long* h_iters, unsigned long long* h_ns, unsigned int* h_sm,
                   int blocks) {
    cudaMemset(d_iters, 0, (size_t)blocks * sizeof(unsigned long long));
    cudaMemset(d_ns, 0, (size_t)blocks * sizeof(unsigned long long));
    cudaMemset(d_sm, 0, (size_t)blocks * sizeof(unsigned int));
    if (d_atomic) cudaMemset(d_atomic, 0, sizeof(unsigned long long));
    bench_kernel<<<blocks, 64>>>(gpc, test, budget_ns, d_atomic, d_stream, stream_floats,
                                 h_words, h_atomic, d_iters, d_ns, d_sm);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpc %d %s: %s\n", gpc, test_name(test), cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(h_iters, d_iters, (size_t)blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ns, d_ns, (size_t)blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sm, d_sm, (size_t)blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    double sum_ns = 0.0;
    int nsm = 0;
    double min_ns = 0, max_ns = 0;
    printf("[sms]");
    for (int b = 0; b < blocks; b++) {
        if (h_iters[b] == 0 || h_ns[b] == 0) continue;
        double ns = (double)h_ns[b] / (double)h_iters[b];
        if (nsm == 0 || ns < min_ns) min_ns = ns;
        if (nsm == 0 || ns > max_ns) max_ns = ns;
        sum_ns += ns;
        nsm++;
        printf(" %u:%.1f", h_sm[b], ns);
    }
    printf("\n");
    if (nsm == 0) {
        printf("[row] gpc %d %s n 0\n", gpc, test_name(test));
        return 1;
    }
    double mean = sum_ns / nsm;
    double var = 0.0;
    for (int b = 0; b < blocks; b++) {
        if (h_iters[b] == 0 || h_ns[b] == 0) continue;
        double ns = (double)h_ns[b] / (double)h_iters[b];
        double d = ns - mean;
        var += d * d;
    }
    double cv = (mean > 0.0) ? (sqrt_local(var / nsm) / mean) : 0.0;
    printf("[row] gpc %d %s n %d mean_ns %.2f min_ns %.2f max_ns %.2f cv %.4f\n",
           gpc, test_name(test), nsm, mean, min_ns, max_ns, cv);
    return 0;
}

int main(int argc, char** argv) {
    unsigned long long budget_ns = 400000000ULL;
    if (argc >= 2) budget_ns = strtoull(argv[1], NULL, 10);
    if (budget_ns < 50000000ULL) budget_ns = 50000000ULL;

    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int l2 = 0;
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
    int blocks = prop.multiProcessorCount;
    printf("[bench] %s SMs=%d L2=%d bytes budget_ns=%llu\n", prop.name, blocks, l2, budget_ns);

    size_t stream_bytes = (size_t)l2 * 4u;
    if (stream_bytes < (size_t)256 * 1024 * 1024) stream_bytes = (size_t)256 * 1024 * 1024;
    float* d_stream = NULL;
    cudaMalloc(&d_stream, stream_bytes);
    cudaMemset(d_stream, 1, stream_bytes);
    size_t stream_floats = stream_bytes / sizeof(float);

    unsigned long long* d_atomic = NULL;
    cudaMalloc(&d_atomic, sizeof(unsigned long long));

    unsigned long long *h_words = NULL, *h_atomic = NULL;
    cudaHostAlloc(&h_words, 64 * sizeof(unsigned long long), cudaHostAllocMapped);
    cudaHostAlloc(&h_atomic, sizeof(unsigned long long), cudaHostAllocMapped);
    memset(h_words, 0, 64 * sizeof(unsigned long long));
    memset(h_atomic, 0, sizeof(unsigned long long));
    unsigned long long *d_words = NULL, *d_hatomic = NULL;
    cudaHostGetDevicePointer(&d_words, h_words, 0);
    cudaHostGetDevicePointer(&d_hatomic, h_atomic, 0);

    unsigned long long *d_iters = NULL, *d_ns = NULL, *h_iters = NULL, *h_ns = NULL;
    unsigned int *d_sm = NULL, *h_sm = NULL;
    cudaMalloc(&d_iters, (size_t)blocks * sizeof(unsigned long long));
    cudaMalloc(&d_ns, (size_t)blocks * sizeof(unsigned long long));
    cudaMalloc(&d_sm, (size_t)blocks * sizeof(unsigned int));
    h_iters = (unsigned long long*)calloc((size_t)blocks, sizeof(unsigned long long));
    h_ns = (unsigned long long*)calloc((size_t)blocks, sizeof(unsigned long long));
    h_sm = (unsigned int*)calloc((size_t)blocks, sizeof(unsigned int));

    printf("[stream] bytes %zu\n", stream_bytes);
    int fails = 0;
    for (int gpc = 0; gpc < 7; gpc++) {
        for (int test = 0; test < 5; test++) {
            fails += run_one(gpc, test, budget_ns, d_atomic, d_stream, stream_floats,
                             d_words, d_hatomic, d_iters, d_ns, d_sm, h_iters, h_ns, h_sm, blocks);
        }
    }
    printf("[bench] fails %d\n", fails);

    cudaFree(d_stream);
    cudaFree(d_atomic);
    cudaFree(d_iters);
    cudaFree(d_ns);
    cudaFree(d_sm);
    cudaFreeHost(h_words);
    cudaFreeHost(h_atomic);
    free(h_iters);
    free(h_ns);
    free(h_sm);
    return fails ? 1 : 0;
}
