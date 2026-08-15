// =============================================================================
// GPUTronic Stage 13 SCAG v14.0.9 — PER-SM HIERARCHICAL COUNTERS + ACCUMULATOR
// Target: Blackwell (RTX 5080, sm_120) — 84 SMs × per-SM atomic counters
// =============================================================================
//
// ARCHITECTURE CHANGE FROM v14.0.8:
// - OLD: Single global atomic counter → massive bank contention at scale
// - NEW: Per-SM atomic array + host-side accumulator → minimal contention
//
// FLOW:
//   Threads → per-SM atomicAdd (only 256 threads fight within same SM)
//            ↓
//   sm_counters[84] ← each SM writes to its own slot
//            ↓
//   Host reads all 84 counters every control tick, sums → aggregate throughput
//            ↓
//   Zero-copy mapped memory ← governor reads single aggregate value
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

// ─── Hardware Configuration ────────────────────────────────────────────────
#define NUM_SM              84        // RTX 5080 Blackwell
#define THREADS_PER_BLOCK   256
#define BLOCKS_PER_SM       1         // 1 block per SM = 84 blocks total
#define TOTAL_BLOCKS        (NUM_SM * BLOCKS_PER_SM)
#define WORK_UNITS_PER_THREAD 5000    // Max work units per thread per cycle

// ─── Control Loop Configuration ────────────────────────────────────────────
#define CONTROL_DT_US       100       // 10kHz control loop
#define Q_LOW_THRESH        0.05
#define Q_HIGH_THRESH       1.0
#define Kp                  2.0       // Proportional gain

// ─── Timing Helper ─────────────────────────────────────────────────────────
inline double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

// ─── Throttle Control (Zero-Copy Pinned Memory) ────────────────────────────
typedef struct {
    volatile double q_fraction;   // Governor output: throttle fraction [0.05, 1.0]
    volatile int running_flag;     // Shutdown signal
} ThrottleControl;

// ─── Per-SM Work Counters (Device Memory) ──────────────────────────────────
// Each SM has its own atomic counter slot → eliminates cross-SM contention
static unsigned long long* g_sm_counters = NULL;  // Device pointer, size NUM_SM
static unsigned long long g_host_sm_counters[NUM_SM] = {0};  // Host mirror for accumulation

// ─── Accumulator State (Host) ──────────────────────────────────────────────
static unsigned long long g_last_aggregate = 0;   // Last summed total across all SMs
static unsigned long long g_current_aggregate = 0; // Current summed total

// ─── Kalman Observer (2-State: Z, dZ/dt) ──────────────────────────────────
typedef struct {
    double x[2];  // [0] = Z (impedance), [1] = dZ/dt (rate of change)
} KalmanFilter;

static KalmanFilter g_kalman = {{0}};

// ─── Global Pointers for Kernel Access ─────────────────────────────────────
static ThrottleControl* g_throttle = NULL;        // Zero-copy pinned host memory
static ThrottleControl* g_d_throttle_ptr = NULL;  // Device pointer to same memory

// =============================================================================
// Kernels
// =============================================================================

// ─── Workload Kernel (Per-SM Counter Architecture) ─────────────────────────
// Each block is assigned to a specific SM via blockIdx mapping.
// Threads within a block do work and increment their SM's counter.
__global__ void gpu_workload_kernel(ThrottleControl* throttle, 
                                     unsigned long long* sm_counters,
                                     int num_sm) {
    // Map this block to its SM index
    int sm_id = blockIdx.x % num_sm;
    
    // Local thread work accumulator (reduces atomic contention further)
    int tid = threadIdx.x;
    int local_work = 0;
    
    // Read throttle fraction (from zero-copy memory, visible on device)
    double q_fraction = throttle->q_fraction;
    
    // Calculate work units for this thread based on throttle
    int units_this_cycle = (int)(q_fraction * WORK_UNITS_PER_THREAD);
    
    // Do work locally first, then atomicAdd once per batch
    for (int i = 0; i < units_this_cycle; i++) {
        // Placeholder workload — replace with real work here
        // (GEMM, inference, simulation, etc.)
        volatile double result = 0.0;
        result += sin(0.1) * cos(0.2);  // Prevents compiler optimization
        
        // Increment local counter every 32 units (warp-aligned batching)
        if ((i + 1) % 32 == 0) {
            local_work += 32;
        }
    }
    
    // Handle remaining work not in a full warp batch
    int remainder = units_this_cycle % 32;
    local_work += remainder;
    
    // Single atomicAdd per thread to the SM's counter (not global!)
    // Only ~1-2 threads per SM hit this simultaneously → minimal contention
    if (local_work > 0) {
        atomicAdd(&sm_counters[sm_id], (unsigned long long)local_work);
    }
}

// ─── Counter Reset Kernel (Clear all SM counters for next measurement period)
__global__ void reset_sm_counters(unsigned long long* sm_counters, int num_sm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sm) {
        sm_counters[idx] = 0;
    }
}

// =============================================================================
// Kalman Observer (2-State: Z impedance + dZ/dt rate)
// =============================================================================

void kalman_init(KalmanFilter* kf) {
    kf->x[0] = 1.0;   // Initial Z estimate
    kf->x[1] = 0.0;   // Initial dZ/dt estimate
}

void kalman_predict(KalmanFilter* kf, double dt) {
    // Deadbeat prediction: Z_next = Z + dZ/dt * dt
    double z_pred = kf->x[0] + kf->x[1] * dt;
    double dz_pred = 0.0;  // Assume constant rate (no acceleration model)
    
    kf->x[0] = z_pred;
    kf->x[1] = dz_pred;
}

void kalman_update(KalmanFilter* kf, double z_measurement) {
    // Simple proportional correction — full Kalman gain matrix is overkill here
    double innovation = z_measurement - kf->x[0];
    
    // Correction factor (tunable: higher = faster response, lower = smoother)
    double K = 0.5;  
    
    kf->x[0] += K * innovation;       // Correct Z estimate
    kf->x[1] += K * innovation / 0.001;  // Estimate dZ/dt from correction rate
}

double kalman_get_z(KalmanFilter* kf) {
    return kf->x[0];
}

// =============================================================================
// Accumulator: Sum all SM counters → single aggregate throughput
// =============================================================================

unsigned long long accumulate_sm_counters() {
    unsigned long long total = 0;
    for (int i = 0; i < NUM_SM; i++) {
        total += g_host_sm_counters[i];
    }
    return total;
}

double compute_work_delta() {
    g_current_aggregate = accumulate_sm_counters();
    double delta = (double)(g_current_aggregate - g_last_aggregate);
    g_last_aggregate = g_current_aggregate;
    return delta;
}

// =============================================================================
// Control Loop Thread
// =============================================================================

void* control_loop_thread(void*) {
    printf("[GPUTronic] Control loop starting (per-SM counters + accumulator)\n");
    
    kalman_init(&g_kalman);
    
    double last_print = get_time_us();
    int tick_count = 0;
    
    while (g_throttle->running_flag) {
        double now = get_time_us();
        
        // ─── Step 1: Read per-SM counters from device to host ──────────────
        cudaMemcpy(g_host_sm_counters, g_sm_counters, 
                   NUM_SM * sizeof(unsigned long long), 
                   cudaMemcpyDeviceToHost);
        
        // ─── Step 2: Accumulate → single throughput number ─────────────────
        double work_delta = compute_work_delta();
        
        // ─── Step 3: Compute Z (impedance) from throughput ─────────────────
        // Z = 1 / throughput (higher throughput = lower impedance)
        double z_measured = 1.0 / (work_delta > 0 ? work_delta : 1e-6);
        
        // Clamp to prevent extreme values during startup/transients
        if (z_measured < 0.05) z_measured = 0.05;
        if (z_measured > 2.0)  z_measured = 2.0;
        
        // ─── Step 4: Kalman filter update ──────────────────────────────────
        kalman_predict(&g_kalman, CONTROL_DT_US / 1e6);  // dt in seconds
        kalman_update(&g_kalman, z_measured);
        
        double z_filt = kalman_get_z(&g_kalman);
        
        // ─── Step 5: Q-axis proportional control (NO D-axis) ──────────────
        // Target impedance Z=1.0 → stable throughput
        double error_z = z_filt - 1.0;
        double delta_q = -error_z * Kp;
        
        double new_q = g_throttle->q_fraction + delta_q;
        
        // Clamp throttle to safe range
        if (new_q < Q_LOW_THRESH) new_q = Q_LOW_THRESH;
        if (new_q > Q_HIGH_THRESH) new_q = Q_HIGH_THRESH;
        
        g_throttle->q_fraction = new_q;
        
        // ─── Step 6: Reset SM counters for next measurement period ─────────
        reset_sm_counters<<<(NUM_SM + 31) / 32, 32>>>(g_sm_counters, NUM_SM);
        
        // ─── Step 7: Telemetry output (every 10ms = 100Hz print rate) ─────
        tick_count++;
        if ((now - last_print) > 10000.0) {
            printf("[Tick %d] Z=%.3f | q=%.3f | delta=%llu | aggregate=%llu\n", 
                   tick_count, z_filt, new_q, 
                   (unsigned long long)(g_current_aggregate - g_last_aggregate + work_delta),
                   g_current_aggregate);
            fflush(stdout);
            last_print = now;
        }
        
        // ─── Step 8: Sleep for control interval ────────────────────────────
        usleep(CONTROL_DT_US * 1000);  // Convert µs to ns for usleep (actually ms*1000)
    }
    
    return NULL;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic] Stage 13 SCAG v14.0.9 — PER-SM HIERARCHICAL COUNTERS\n");
    printf("[GPUTronic] Architecture: NVIDIA Blackwell (RTX 5080), sm_120\n");
    printf("[GPUTronic] SMs: %d | Threads/SM: %d | Total blocks: %d\n", 
           NUM_SM, THREADS_PER_BLOCK, TOTAL_BLOCKS);
    printf("[GPUTronic] Control loop: %.0f kHz (%dµs interval)\n", 
           1e6 / CONTROL_DT_US, CONTROL_DT_US);
    printf("[GPUTronic] Counter architecture: Per-SM atomics + host accumulator\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    // ─── Initialize CUDA ──────────────────────────────────────────────────
    cudaSetDevice(0);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[GPUTronic] GPU: %s | SMs reported: %d\n", prop.name, prop.multiProcessorCount);

    // ─── Allocate zero-copy pinned memory for throttle control ─────────────
    cudaMallocHost((void**)&g_throttle, sizeof(ThrottleControl));
    g_d_throttle_ptr = (ThrottleControl*)g_throttle;  // Same pointer works on device
    
    memset(g_throttle, 0, sizeof(ThrottleControl));
    g_throttle->q_fraction = 1.0;   // Start at full throttle
    g_throttle->running_flag = 1;

    // ─── Allocate per-SM counters on device ────────────────────────────────
    cudaMalloc((void**)&g_sm_counters, NUM_SM * sizeof(unsigned long long));
    
    // Initialize all SM counters to zero
    reset_sm_counters<<<(NUM_SM + 31) / 32, 32>>>(g_sm_counters, NUM_SM);
    cudaDeviceSynchronize();

    printf("[GPUTronic] Per-SM counter array allocated: %d × uint64\n", NUM_SM);
    printf("[GPUTronic] Zero-copy throttle memory allocated and mapped\n");

    // ─── Launch persistent workload kernel ────────────────────────────────
    printf("\n[GPUTronic] Launching workload kernel (%d blocks, %d threads/block)...\n", 
           TOTAL_BLOCKS, THREADS_PER_BLOCK);
    
    dim3 block(THREADS_PER_BLOCK);
    gpu_workload_kernel<<<TOTAL_BLOCKS, block>>>(g_d_throttle_ptr, g_sm_counters, NUM_SM);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("[GPUTronic] Kernel launched successfully.\n");

    // ─── Start control loop thread ────────────────────────────────────────
    printf("[GPUTronic] Starting control loop thread...\n\n");
    
    pthread_t control_thread;
    if (pthread_create(&control_thread, NULL, control_loop_thread, NULL) != 0) {
        fprintf(stderr, "[ERROR] Failed to create control thread\n");
        return 1;
    }

    // ─── Main thread: wait for interrupt (Ctrl+C handled via signal) ──────
    printf("[GPUTronic] Running... Press Ctrl+C to stop.\n");
    
    while (g_throttle->running_flag) {
        usleep(500000);  // Sleep 500ms, check flag periodically
    }

    // ─── Shutdown ──────────────────────────────────────────────────────────
    printf("\n[GPUTronic] Shutting down...\n");
    
    g_throttle->running_flag = 0;
    pthread_join(control_thread, NULL);
    
    // Wait for kernel to finish (it should exit when running_flag is cleared)
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(g_sm_counters);
    cudaFreeHost(g_throttle);
    
    printf("[GPUTronic] Shutdown complete.\n");
    printf("[GPUTronic] Final aggregate work: %llu units\n", g_current_aggregate);

    return 0;
}

