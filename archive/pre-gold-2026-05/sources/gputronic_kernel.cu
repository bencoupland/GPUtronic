// GPUTronic v26 — Device kernel + launch helper
// Very heavy synthetic workload designed to push Z well above 1.0
// when throttle is low, so the governor has something to control.

#include <cuda_runtime.h>

#define WORK_UNITS_PER_THREAD 32768   // Significantly increased

__device__ __forceinline__ void heavy_workload(float* acc, int sm_id, int iter) {
    float x = (float)(iter * 17 + sm_id) * 0.0174532925f;
    float y = (float)(iter * 23 + sm_id) * 0.0174532925f;

    // Heavy inner work — many more operations per unit
    #pragma unroll 8
    for (int k = 0; k < 8; k++) {
        float a = sinf(x + k);
        float b = cosf(y * (k + 1));
        float c = sinf(y * 1.3f + k * 0.7f);
        float d = cosf(x * 0.9f + k * 1.1f);

        *acc = fmaf(a, b, *acc);
        *acc = fmaf(c, d, *acc);
        *acc = fmaf(a * 0.6f, c, *acc);
        *acc = fmaf(b * 0.4f, d, *acc);
        *acc = fmaf(*acc, 0.97f, 0.03f);   // feedback to increase register pressure
    }
}

__global__ void gpu_persistent_kernel(unsigned long long* sm_counters,
                                      void* control_data_void, int num_sm) {
    struct GPUTronicControl {
        unsigned int control_flags;
        float target_pm;
        float current_pm;
        int throttle_sleep_ns;
        int max_sleep_ns;
        unsigned long long total_work_pulses;
        float z_estimate;
        float dzdt_estimate;
        float pm_error;
        int proactive_corrections;
        int blocks_per_sm_target;
        float P[2][2];
        int active_blocks_current;
    } *control_data = (struct GPUTronicControl*)control_data_void;

    int sm_id = blockIdx.x;
    while (true) {
        if (control_data->control_flags & 0x2) { __nanosleep(1000); continue; }
        int sleep_ns = control_data->throttle_sleep_ns;
        if (sleep_ns > control_data->max_sleep_ns) sleep_ns = control_data->max_sleep_ns;

        float thread_acc = 0.0f;
        int iterations = WORK_UNITS_PER_THREAD / 32;
        for (int i = 0; i < iterations; i++) {
            heavy_workload(&thread_acc, sm_id, i);
            if ((i & 31) == 0) atomicAdd(&sm_counters[sm_id], 32ULL);
        }
        if (sleep_ns > 0) __nanosleep(sleep_ns);
        control_data->total_work_pulses = sm_counters[sm_id];
        control_data->active_blocks_current = num_sm;
    }
}

extern "C" void launch_persistent_kernel(unsigned long long* counters,
                                         void* control,
                                         int num_sm,
                                         int threads_per_block) {
    dim3 block(threads_per_block);
    gpu_persistent_kernel<<<num_sm, block>>>(counters, control, num_sm);
}
