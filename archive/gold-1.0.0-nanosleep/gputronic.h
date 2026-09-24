/* =============================================================================
 * GPUTronic Gold — Public API
 * Version: 1.0.0-gold
 *
 * This header *is* the cable. If a signal is not in GPUTronicControl, the
 * device cannot see it. Full function frame:
 *   docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md   (GT-FF-GOLD-1.0)
 *
 * Axes (names, not a claim that the GPU is a PMSM):
 *   Q — useful work rate     (F-04 tachometer)
 *   Z — impedance proxy      (F-07)  Z = rate_ref / rate, free-run ≈ 1
 *   D — thermal/power        stock driver; not in this header on purpose
 *
 * Modes:
 *   Cooperative — foreign kernel publishes pulses + honours throttle_sleep_ns
 *                 (F-03). Transfer plant is the existence proof.
 *   Self-test   — Gold launches gold_persistent_kernel (F-02).
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

/* RTX 5080 (GB203) defaults — override via config.num_sm after query.
 * Calibration constants: function frame §3. Changing these is a Gold tag. */
#define GPUTRONIC_DEFAULT_NUM_SM 84
#define GPUTRONIC_DEFAULT_THREADS_PER_BLOCK 64
#define GPUTRONIC_DEFAULT_CONTROL_DT_US 10      /* path period, not observer period (F-05) */
#define GPUTRONIC_DEFAULT_MAX_SLEEP_NS 500000   /* must reach the nanosleep cliff (F-11) */
#define GPUTRONIC_DEFAULT_BASE_SLEEP_NS 5

/* Tracking law defaults. sleep_scale is a *fallback*; F-09 overwrites it
 * after calibration when auto_sleep_scale=1. Kp/Ki came from Cyberpunk-era
 * v1.13 and only actuate once scale is honest. */
#define GPUTRONIC_GOLD_TARGET_Z 1.5f
#define GPUTRONIC_GOLD_KP 0.55f
#define GPUTRONIC_GOLD_KI 0.08f
#define GPUTRONIC_GOLD_SLEEP_SCALE 120000.0f
#define GPUTRONIC_GOLD_INTEGRAL_CLAMP 1.0f
#define GPUTRONIC_GOLD_DEADBAND 0.08f
#define GPUTRONIC_GOLD_Z_CEILING 10.0f
#define GPUTRONIC_GOLD_Z_FLOOR 0.3f

/* F-01 cable. Mapped pinned memory. Host writes actuators; device atomicAdds Q.
 * Telemetry snapshots are host-written; the kernel must run without them. */
struct GPUTronicControl {
    volatile unsigned int control_flags; /* F-02: bit0=STOP bit1=PAUSE bit2=RESET */
    volatile int throttle_sleep_ns;      /* F-11 actuator command (host → device) */
    int max_sleep_ns;                    /* clamp for F-11 */
    int base_sleep_ns;                   /* free-run / one-sided hold */
    /* F-04 tachometer. Device: atomicAdd only. Never last-SM assign. */
    volatile unsigned long long total_work_pulses;
    /* Host snapshots (F-07, F-08, F-10). Device may read target_z. */
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
    int control_dt_us;       /* default 10 (=100 kHz *path*) */
    int max_sleep_ns;
    int base_sleep_ns;
    float target_z;
    float kp;
    float ki;
    float sleep_scale;       /* fallback; F-09 overwrites if auto_sleep_scale */
    float integral_clamp;
    float deadband;
    float z_ceiling;
    float z_floor;
    int one_sided;           /* F-10: only add sleep when z < target */
    int enable_kalman;       /* F-08 */
    int launch_selftest_kernel; /* 1 = F-02; 0 = F-03 foreign plant */
    int auto_sleep_scale;    /* F-09 after F-06 */
    const char* csv_path;    /* NULL = results/gputronic_gold.csv */
} GPUTronicConfig;

/* Fill config with Gold calibration defaults. */
void gputronic_config_gold(GPUTronicConfig* cfg);

/* F-12 lifecycle */
GPUTronicHandle* gputronic_create(const GPUTronicConfig* cfg);
int gputronic_start(GPUTronicHandle* h);   /* control thread (+ F-02 if configured) */
void gputronic_stop(GPUTronicHandle* h);
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

/* F-03 / F-04 host path. Single-writer. Do not mix with device atomicAdd. */
void gputronic_pulse(GPUTronicHandle* h, unsigned long long units);

int gputronic_zero_copy_ok(const GPUTronicHandle* h);
void gputronic_print_status(const GPUTronicHandle* h, int elapsed_s);

#ifdef __cplusplus
}
#endif

#endif /* GPUTRONIC_H */
