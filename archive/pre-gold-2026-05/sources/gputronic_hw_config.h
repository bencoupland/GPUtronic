/**
 * ============================================================================
 * GPUtronic Hardware Configuration — NVIDIA GPU Variant Specifications
 * ============================================================================
 * 
 * This header contains hardcoded specifications for different NVIDIA GPU dies.
 * Used for initial governor setup and calibration parameters.
 * 
 * Format: [GPU_NAME] = {SM_COUNT, WARP_SIZE, MAX_THREADS_PER_SM, ARCH_VERSION}
 * 
 * ============================================================================
 */

#ifndef GPUTRONIC_HW_CONFIG_H
#define GPUTRONIC_HW_CONFIG_H

// ============================================================================
// GPU VARIANT DEFINITIONS
// ============================================================================

/**
 * NVIDIA GeForce RTX 5080 (Blackwell GB208)
 * - SM Count: 84
 * - Warp Size: 32
 * - Compute Capability: 12.0 (sm_120)
 * - Memory: 16 GB GDDR6
 */
struct GPUConfig {
    const char* name;
    int sm_count;
    int warp_size;
    int max_threads_per_sm;
    float compute_capability;
};

constexpr GPUConfig HW_RTX_5080 = {
    .name = "NVIDIA GeForce RTX 5080",
    .sm_count = 84,
    .warp_size = 32,
    .max_threads_per_sm = 1536,  // Typical for Blackwell
    .compute_capability = 12.0f
};

/**
 * NVIDIA GeForce RTX 4090 (Ada Lovelace GH100)
 * - SM Count: 84 (same as RTX 4080 but full die)
 * - Warp Size: 32
 * - Compute Capability: 9.0 (sm_90)
 */
constexpr GPUConfig HW_RTX_4090 = {
    .name = "NVIDIA GeForce RTX 4090",
    .sm_count = 84,
    .warp_size = 32,
    .max_threads_per_sm = 1536,
    .compute_capability = 9.0f
};

/**
 * NVIDIA GeForce RTX 4080 (Ada Lovelace GH100)
 * - SM Count: 72 (cut from GH100)
 * - Warp Size: 32
 * - Compute Capability: 9.0 (sm_90)
 */
constexpr GPUConfig HW_RTX_4080 = {
    .name = "NVIDIA GeForce RTX 4080",
    .sm_count = 72,
    .warp_size = 32,
    .max_threads_per_sm = 1536,
    .compute_capability = 9.0f
};

/**
 * NVIDIA A100 (Hopper H100)
 * - SM Count: 104
 * - Warp Size: 32
 * - Compute Capability: 9.0 (sm_90)
 */
constexpr GPUConfig HW_A100 = {
    .name = "NVIDIA A100",
    .sm_count = 104,
    .warp_size = 32,
    .max_threads_per_sm = 1536,
    .compute_capability = 9.0f
};

/**
 * NVIDIA H100 (Hopper H100)
 * - SM Count: 72
 * - Warp Size: 32
 * - Compute Capability: 9.0 (sm_90)
 */
constexpr GPUConfig HW_H100 = {
    .name = "NVIDIA H100",
    .sm_count = 72,
    .warp_size = 32,
    .max_threads_per_sm = 1536,
    .compute_capability = 9.0f
};

/**
 * NVIDIA Tesla V100 (Volta GV100)
 * - SM Count: 80
 * - Warp Size: 32
 * - Compute Capability: 7.0 (sm_70)
 */
constexpr GPUConfig HW_V100 = {
    .name = "NVIDIA Tesla V100",
    .sm_count = 80,
    .warp_size = 32,
    .max_threads_per_sm = 1024,
    .compute_capability = 7.0f
};

/**
 * NVIDIA Tesla T4 (Turing TU102)
 * - SM Count: 35
 * - Warp Size: 32
 * - Compute Capability: 7.5 (sm_75)
 */
constexpr GPUConfig HW_T4 = {
    .name = "NVIDIA Tesla T4",
    .sm_count = 35,
    .warp_size = 32,
    .max_threads_per_sm = 1024,
    .compute_capability = 7.5f
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get GPU config by SM count (fallback detection)
 */
__device__ __host__ inline const GPUConfig* get_gpu_config(int sm_count) {
    switch (sm_count) {
        case 84: return &HW_RTX_5080;  // RTX 5080 or RTX 4090
        case 72: return &HW_RTX_4080;  // RTX 4080 or H100
        case 104: return &HW_A100;     // A100
        case 35: return &HW_T4;        // T4
        default: return nullptr;       // Unknown GPU
    }
}

/**
 * Calculate optimal block size for a given GPU
 */
__device__ __host__ inline int get_optimal_block_size(int sm_count) {
    // Heuristic: 256 threads per SM for most GPUs
    if (sm_count >= 80) return 128;  // High-end GPUs prefer larger blocks
    if (sm_count >= 35) return 64;   // Mid-range GPUs
    return 32;                       // Entry-level GPUs
}

// ============================================================================
// CALIBRATION PARAMETERS BY GPU VARIANT
// ============================================================================

/**
 * Kalman observer parameters tuned per GPU variant
 */
struct KalmanParams {
    float q_proc;     // Process noise covariance
    float r_meas;     // Measurement noise variance
};

constexpr KalmanParams KALMAN_RTX_5080 = {0.002f, 0.015f};
constexpr KalmanParams KALMAN_RTX_4090 = {0.003f, 0.020f};
constexpr KalmanParams KALMAN_RTX_4080 = {0.003f, 0.020f};
constexpr KalmanParams KALMAN_A100 = {0.001f, 0.010f};
constexpr KalmanParams KALMAN_H100 = {0.001f, 0.010f};
constexpr KalmanParams KALMAN_V100 = {0.005f, 0.030f};
constexpr KalmanParams KALMAN_T4 = {0.008f, 0.040f};

/**
 * Phase margin parameters tuned per GPU variant
 */
struct PhaseMarginParams {
    float target_pm;      // Target phase margin (headroom)
    float kp;             // Proportional gain
    float ki;             // Integral gain
    float integral_max;   // Integral windup protection
};

constexpr PhaseMarginParams PM_RTX_5080 = {1.10f, 0.50f, 0.12f, 8.0f};
constexpr PhaseMarginParams PM_RTX_4090 = {1.15f, 0.45f, 0.15f, 10.0f};
constexpr PhaseMarginParams PM_RTX_4080 = {1.15f, 0.45f, 0.15f, 10.0f};
constexpr PhaseMarginParams PM_A100 = {1.20f, 0.60f, 0.10f, 12.0f};
constexpr PhaseMarginParams PM_H100 = {1.20f, 0.60f, 0.10f, 12.0f};

// ============================================================================
// END OF HARDWARE CONFIGURATION
// ============================================================================

#endif // GPUTRONIC_HW_CONFIG_H