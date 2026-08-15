// =============================================================================
// GPUTronic Stage 13 SCAG v14.0.8 — Q-AXIS-ONLY MODE (D-AXIS REMOVED) - ALL THREADS ACTIVE
// Target: Blackwell (RTX 5080, sm_120) — Pure work counter feedback control
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

inline double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

#define THREADS_PER_BLOCK 256
#define WORK_UNITS_PER_THREAD 5000
#define CONTROL_DT_US 1000   // 1ms = 1kHz control loop (slower for testing atomic visibility)
#define Q_LOW_THRESH    0.05
#define Q_HIGH_THRESH   1.0

typedef struct {
    volatile double q_fraction;
    volatile int running_flag;  
} ThrottleControl;

static ThrottleControl *g_throttle = NULL;           
static ThrottleControl *g_d_throttle_ptr = NULL;     
static __device__ unsigned long long total_work_completed = 0;
static __host__ unsigned long long last_work_counter = 0;

typedef struct {
    double x[3];
    double P[9];  
} KalmanFilter;

static KalmanFilter g_kalman = {{0}};

// Kernel with work counter feedback - all threads always active for simplicity
__global__ void gpu_workload_kernel(ThrottleControl* throttle) {
    int tid = threadIdx.x;  
    
    // Every thread increments work counter (all 256 threads do work per cycle)
    // q_fraction controls how many work units each thread produces
    volatile double result = 0.0;
    
    // Each thread does WORK_UNITS_PER_THREAD iterations scaled by throttle
    int units_this_cycle = (int)(throttle->q_fraction * WORK_UNITS_PER_THREAD);
    
    for (int i = 0; i < units_this_cycle; i++) {
        atomicAdd(&total_work_completed, 1);
    }
}

void kalman_init(KalmanFilter* kf) {
    kf->x[0] = 1.0;   
    kf->x[1] = 0.0;
    kf->x[2] = 0.0;  
    for (int i = 0; i < 9; i++) kf->P[i] = 0.1;
}

void kalman_predict(KalmanFilter* kf) {
    // Deadbeat predictor - assumes constant work rate
    kf->x[0] = 1.0;  
}

void kalman_update(KalmanFilter* kf, double measurement) {
    // Proportional control update for Q-axis governor
    double y = measurement - kf->x[0];
    kf->x[0] += y * 0.5;  
}

double kalman_get_z(KalmanFilter* kf) {
    return kf->x[0];  
}

void* control_loop_thread(void*) {
    printf("[GPUTronic] Control loop starting (Q-axis only, work counter feedback)\n");
    
    kalman_init(&g_kalman);  
    
    // Regular control loop with proportional governor
    while (g_throttle->running_flag) {
        double now = get_time_us();
        
        cudaMemcpy(&total_work_completed, &total_work_completed, 
                   sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        
        if (last_work_counter == 0) last_work_counter = total_work_completed;
        
        double work_delta = (double)(total_work_completed - last_work_counter);
        double z_measured = 1.0 / (work_delta > 0 ? work_delta : 1e-6);
        
        // Clamp impedance measurement to prevent extreme values
        if (z_measured < 0.5) z_measured = 0.5;
        if (z_measured > 2.0) z_measured = 2.0;

        kalman_predict(&g_kalman);
        kalman_update(&g_kalman, z_measured);
        
        double z_filt = kalman_get_z(&g_kalman);
        
        // Pure Proportional Control (Q-AXIS ONLY) - NO D-AXIS INVOLVED
        double error_z = z_filt - 1.0;
        double delta_q = -error_z * 2.0;
        double new_q = g_throttle->q_fraction + delta_q;
        
        if (new_q < Q_LOW_THRESH) new_q = Q_LOW_THRESH;
        if (new_q > Q_HIGH_THRESH) new_q = Q_HIGH_THRESH;

        g_throttle->q_fraction = new_q;
        
        // Print status every 10ms
        static double last_print = get_time_us();
        if ((now - last_print) > 10000) {
            printf("[Status] Z=%.3f | q=%.3f | work_delta=%lu\n", 
                   z_filt, new_q, (unsigned long)(total_work_completed - last_work_counter));
            fflush(stdout);
            last_print = now;
        }
        
        // Sleep for control interval at 10kHz frequency
        usleep(CONTROL_DT_US * 1000);
    }  
}

int main() {
    
    printf("[GPUTronic] Stage 13 SCAG v14.0.8 — Q-AXIS-ONLY MODE (D-AXIS DISABLED)\n");
    printf("[GPUTronic] Architecture: NVIDIA Blackwell (RTX 5080), compute sm_120\n");
    printf("[GPUTronic] nanosleep supported: YES (Volta+)\n");
    printf("[GPUTronic] Control frequency: %.0f kHz (%dus)\n", 1e6/CONTROL_DT_US, CONTROL_DT_US);

    
    cudaSetDevice(0);

    cudaMallocHost((void**)&g_throttle, sizeof(ThrottleControl));
    g_d_throttle_ptr = (ThrottleControl*)g_throttle;

    printf("[GPUTronic] Launching workload kernel...\n");
    printf("[GPUTronic] Block size: %d threads\n", THREADS_PER_BLOCK);
    
    dim3 block(THREADS_PER_BLOCK);
    gpu_workload_kernel<<<1, block>>>(g_d_throttle_ptr);
    
    // Ensure kernel completes before main thread exits
    cudaDeviceSynchronize();
    
    printf("[GPUTronic] Kernel launched, waiting for threads to initialize...\n");
    usleep(10000);  // Let kernel launch warm up
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    
    memset(g_throttle, 0, sizeof(ThrottleControl));
    g_throttle->q_fraction = 1.0;  // Start at full speed
    g_throttle->running_flag = 1;
  

    printf("[GPUTronic] Starting control loop thread...\n");
    
    pthread_t control_thread;
    if (pthread_create(&control_thread, NULL, control_loop_thread, NULL) != 0) {
        fprintf(stderr, "[ERROR] Failed to create control thread\n");
        return 1;
    }

    // Main thread sleeps waiting for interrupt
    while (g_throttle->running_flag) {
        usleep(100000);
    }
  

    printf("\n[GPUTronic] Shutting down...\n");
    
    g_throttle->running_flag = 0;

    pthread_join(control_thread, NULL);
  
    cudaFreeHost(g_throttle);
    printf("[GPUTronic] Shutdown complete\n\n");

    return 0;
}