/* =============================================================================
 * GPUTronic Gold — Public API
 * Version: 1.0.0-gold
 *
 * Closed-loop FOC-style governor for massively parallel GPUs.
 * Q-axis = useful work rate | Z-axis = impedance proxy | D-axis = stock driver
 *
 * Modes:
 *   Cooperative — workload publishes pulses + honours throttle_sleep_ns
 *   Self-test   — internal persistent kernel (dyno / step / free-run)
 * ============================================================================= */

#ifndef GPUTRONIC_H
#define GPUTRONIC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GPUTRONIC_VERSION_MAJOR 1
#define GPUTRONIC_VERSION_MINOR 0
#define GPUTRONIC_VERSION_PATCH 0
#define GPUTRONIC_VERSION_STRING "1.0.0-gold"

/* RTX 5080 (GB203) defaults — override via config.num_sm after query */
#define GPUTRONIC_DEFAULT_NUM_SM 84
#define GPUTRONIC_DEFAULT_THREADS_PER_BLOCK 64
#define GPUTRONIC_DEFAULT_CONTROL_DT_US 10
#define GPUTRONIC_DEFAULT_MAX_SLEEP_NS 500000
#define GPUTRONIC_DEFAULT_BASE_SLEEP_NS 5

/* Gold control law (Cyberpunk-proven v1.13 equilibrium) */
#define GPUTRONIC_GOLD_TARGET_Z 1.5f
#define GPUTRONIC_GOLD_KP 0.55f
#define GPUTRONIC_GOLD_KI 0.08f
#define GPUTRONIC_GOLD_SLEEP_SCALE 120000.0f
#define GPUTRONIC_GOLD_INTEGRAL_CLAMP 1.0f
#define GPUTRONIC_GOLD_DEADBAND 0.08f
#define GPUTRONIC_GOLD_Z_CEILING 10.0f
#define GPUTRONIC_GOLD_Z_FLOOR 0.3f

/* Zero-copy control block shared host <-> device (mapped pinned memory). */
struct GPUTronicControl {
    volatile unsigned int control_flags; /* bit0=STOP bit1=PAUSE bit2=RESET_COUNTERS */
    volatile int throttle_sleep_ns;      /* actuator command written by host */
    int max_sleep_ns;
    int base_sleep_ns;
    /* Tachometer: single aggregate incremented by device (atomicAdd). */
    volatile unsigned long long total_work_pulses;
    /* Telemetry snapshots (host-written; device may read target only). */
    float target_z;
    float z_raw;
    float z_hat;
    float dzdt_hat;
    float rate_ema;
    float pm_error;
    float integral;
    int proactive_corrections;
    int control_hz_ema;
    int num_sm;
};

#define GPUTRONIC_FLAG_STOP 0x1u
#define GPUTRONIC_FLAG_PAUSE 0x2u
#define GPUTRONIC_FLAG_RESET 0x4u

typedef struct GPUTronicHandle GPUTronicHandle;

typedef struct GPUTronicConfig {
    int num_sm;              /* 0 = query device */
    int threads_per_block;   /* default 64 */
    int control_dt_us;       /* default 10 (=100 kHz) */
    int max_sleep_ns;
    int base_sleep_ns;
    float target_z;
    float kp;
    float ki;
    float sleep_scale;
    float integral_clamp;
    float deadband;
    float z_ceiling;
    float z_floor;
    int one_sided;           /* 1 = only add sleep when z < target (Gold default) */
    int enable_kalman;       /* 1 = 2-state Kalman on Z (Gold default) */
    int launch_selftest_kernel; /* 1 = persistent synthetic plant */
    int auto_sleep_scale;    /* 1 = set sleep_scale from measured loop period after cal */
    const char* csv_path;    /* NULL = results/gputronic_gold.csv */
} GPUTronicConfig;

/* Fill config with Gold defaults (Cyberpunk v1.13 law + one-sided + Kalman). */
void gputronic_config_gold(GPUTronicConfig* cfg);

/* Lifecycle */
GPUTronicHandle* gputronic_create(const GPUTronicConfig* cfg);
int gputronic_start(GPUTronicHandle* h);   /* starts control thread (+ kernel if configured) */
void gputronic_stop(GPUTronicHandle* h);   /* joins thread, signals kernel, frees */
void gputronic_destroy(GPUTronicHandle* h);

/* Runtime */
void gputronic_set_target(GPUTronicHandle* h, float target_z);
void gputronic_set_gains(GPUTronicHandle* h, float kp, float ki);
void gputronic_set_open_loop_sleep(GPUTronicHandle* h, int sleep_ns); /* <0 = closed loop */
float gputronic_get_z(const GPUTronicHandle* h);
float gputronic_get_z_raw(const GPUTronicHandle* h);
float gputronic_get_rate(const GPUTronicHandle* h);
int gputronic_get_sleep_ns(const GPUTronicHandle* h);
struct GPUTronicControl* gputronic_get_control(GPUTronicHandle* h);

/* Cooperative workloads: publish completed work units into the tachometer. */
void gputronic_pulse(GPUTronicHandle* h, unsigned long long units);

/* Diagnostics */
int gputronic_zero_copy_ok(const GPUTronicHandle* h); /* host_ptr == device_ptr */
void gputronic_print_status(const GPUTronicHandle* h, int elapsed_s);

#ifdef __cplusplus
}
#endif

#endif /* GPUTRONIC_H */
