/* =============================================================================
 * GPUTronic Gold — Public API
 * Version: 1.0.0-gold
 *
 * If a signal is not in GPUTronicControl, the device cannot see it.
 * Function list: docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md
 *
 * Names, not a claim that the GPU is a motor:
 *   Q — useful work rate, the pulse counter [F-04]
 *   Z — rate_ref / rate, about 1 at full speed [F-07]
 *   D — temperature and power, left to the stock driver
 *
 * Modes:
 *   Another kernel adds pulses and runs only active_sms SMs [F-03].
 *   Self-test: Gold launches gold_persistent_kernel [F-02].
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
 * Calibration constants are listed in docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md, section 4.
 * Changing them needs a new version number. */
#define GPUTRONIC_DEFAULT_NUM_SM 84
#define GPUTRONIC_DEFAULT_THREADS_PER_BLOCK 64
#define GPUTRONIC_DEFAULT_CONTROL_DT_US 10      /* control tick, not the rate window [F-05] */
/* Legacy sleep values. Gold no longer reads or writes them after create; they
 * are copied into the control block only for the transfer/episode plants,
 * whose kernels still honour throttle_sleep_ns. */
#define GPUTRONIC_DEFAULT_MAX_SLEEP_NS 500000
#define GPUTRONIC_DEFAULT_BASE_SLEEP_NS 5

/* Controller defaults. Kp and Ki unchanged. Calibration [F-09] measures a rate-vs-SM table at
 * calibration and sets the feed-forward SM count (n_ff) and sms_per_u (local
 * slope) from it. The law is two-sided around n_ff:
 *   u = Kp*e + Ki*I - Kd*dZ/dt,  n = n_ff - u*sms_per_u,  e = target - Z     */
#define GPUTRONIC_GOLD_TARGET_Z 1.5f
#define GPUTRONIC_GOLD_KP 0.55f
#define GPUTRONIC_GOLD_KI 0.08f
#define GPUTRONIC_GOLD_KD 0.0f   /* see function frame note; 0 = dZ/dt logged only */
/* Integral clamp. 0 = scale after calibration so the integral's largest correction,
 * Ki * clamp * sms_per_u (in SMs), equals GPUTRONIC_GOLD_INTEGRAL_SMS_FRAC of the device
 * SM count (84 SMs -> ~3.4 SMs, 20 SMs -> 0.8 SM). Set integral_clamp > 0 in the config
 * to override with a fixed value. Was a fixed 1.0. */
#define GPUTRONIC_GOLD_INTEGRAL_CLAMP 0.0f
#define GPUTRONIC_GOLD_INTEGRAL_SMS_FRAC 0.04f
#define GPUTRONIC_GOLD_DEADBAND 0.08f   /* NEXT.md A3; 0.015 let the count walk on the 5080. GPUTRONIC_DEADBAND overrides */
#define GPUTRONIC_GOLD_PROBE_MS 300   /* per calibration point */
#define GPUTRONIC_GOLD_SENSOR_HYSTERESIS 0.02f  /* [F-16] */
#define GPUTRONIC_GOLD_Z_CEILING 10.0f
#define GPUTRONIC_GOLD_Z_FLOOR 0.3f

/* ---- Shared control block [F-01] ----
 * Mapped pinned memory. Host writes the commands; the device adds pulses.
 * Telemetry snapshots are host-written; the kernel must run without them. */
struct GPUTronicControl {
    volatile unsigned int control_flags; /* bit0 STOP, bit1 PAUSE, bit2 RESET, bit3 LOAD [F-02] */
    volatile int throttle_sleep_ns;      /* legacy: read only by transfer/episode kernels */
    int max_sleep_ns;                    /* legacy: transfer/episode clamp */
    int base_sleep_ns;                   /* legacy: initial throttle_sleep_ns */
    /* Working-SM count. Lowest active_sms blocks issue and pulse.
     * 0 or >= num_sm means every SM. Never commanded below 4. */
    volatile int active_sms;
    /* Pulse total [F-04]. Device: atomicAdd only; never assign it from one SM. */
    volatile unsigned long long total_work_pulses;
    /* Host-written telemetry [F-07, F-08, F-10]. The kernel does not read these. */
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
    /* ---- Tensor-core sensor [F-16] ----
     * Appended at the end so earlier offsets (used by Transfer/Episode) do
     * not move. Device: atomicAdd only on tensor_work_pulses. */
    volatile unsigned long long tensor_work_pulses;
    float z_cuda;            /* host: CUDA-core Z (rate_ref / rate) */
    float z_tensor;          /* host: tensor-core Z */
    float rate_tensor_ema;   /* host: tensor pulse rate, 8 ms window + EMA */
    int ctrl_sensor;         /* host: sensor driving the PI, 0 = CUDA, 1 = tensor */
    /* Synthetic load for tests T6/T7 (Gold kernel only). Appended at the end.
     * 1000 (or 0) = no load. At S > 1000 each pulsing warp emits 1000 pulses per
     * S units of work, so the pulse rate drops to 1000/S of normal: work per pulse
     * rises by S/1000 as seen by the sensors. Host writes, device reads. */
    volatile unsigned int work_scale_milli;
};

/* ---- Flags and handle ---- */
#define GPUTRONIC_FLAG_STOP 0x1u
#define GPUTRONIC_FLAG_PAUSE 0x2u
#define GPUTRONIC_FLAG_RESET 0x4u
#define GPUTRONIC_FLAG_LOAD 0x8u   /* T6/T7: kernel reads work_scale_milli only while set */
/* Gold test only. Off unless the host sets it after calibration.
 * GPUTRONIC_ACTIVE_PASSES or GPUTRONIC_ACTIVE_SMS then replaces the block-index gate. */
#define GPUTRONIC_FLAG_SELECT 0x10u

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
    float kd;                /* dZ/dt gain (Kalman slope); default 0 */
    float integral_clamp;    /* > 0 fixed; <= 0 scaled to SM count after calibration */
    float deadband;
    float z_ceiling;
    float z_floor;
    int enable_kalman;       /* Kalman filter on Z [F-08] */
    int launch_selftest_kernel; /* 1 = launch self-test kernel [F-02]; 0 = another kernel [F-03] */
    int probe_ms;            /* ms per calibration point, default 300 [F-09] */
    int enable_tensor_sensor; /* 1 = add a wmma warp and a second Z [F-16]. Gold default is 0.
                               * GPUTRONIC_TENSOR_SENSOR=1 turns it on. Forced off for other
                               * kernels and for GPUs below sm_70. */
    float sensor_hysteresis; /* Z margin before switching sensors, default 0.02 */
    int sensor_rule;         /* 0 = follow the higher Z (default), 1 = follow the lower Z [F-16] */
    int reserve_sms;         /* extra SMs kept on above the calibrated starting count, default 0 */
    const char* csv_path;    /* NULL = results/gputronic_gold.csv */
} GPUTronicConfig;

/* ---- Configuration ---- */

/* Fill config with Gold calibration defaults. */
void gputronic_config_gold(GPUTronicConfig* cfg);

/* ---- Lifecycle [F-12] ---- */
GPUTronicHandle* gputronic_create(const GPUTronicConfig* cfg);
int gputronic_start(GPUTronicHandle* h);   /* control thread, plus self-test kernel if configured [F-02] */
void gputronic_stop(GPUTronicHandle* h);
void gputronic_destroy(GPUTronicHandle* h);

/* ---- Runtime ---- */
void gputronic_set_target(GPUTronicHandle* h, float target_z);
void gputronic_set_gains(GPUTronicHandle* h, float kp, float ki);
void gputronic_set_kd(GPUTronicHandle* h, float kd);
void gputronic_set_deadband(GPUTronicHandle* h, float deadband);
void gputronic_set_open_loop_sms(GPUTronicHandle* h, int n); /* <0 closed loop, 0 all, else working SMs */
/* Zero total_work_pulses, per-SM counters and host rate state. Blocks up to
 * ~50 ms. rate_ref and the calibration table are kept. Returns 0 on success. */
int gputronic_reset_counters(GPUTronicHandle* h);
float gputronic_get_z(const GPUTronicHandle* h);
float gputronic_get_z_raw(const GPUTronicHandle* h);
float gputronic_get_rate(const GPUTronicHandle* h);
int gputronic_get_active_sms(const GPUTronicHandle* h);
/* Two-sensor telemetry [F-16]. Tensor values are 0 when the sensor is off. */
float gputronic_get_z_cuda(const GPUTronicHandle* h);
float gputronic_get_z_tensor(const GPUTronicHandle* h);
float gputronic_get_rate_tensor(const GPUTronicHandle* h);
int gputronic_get_ctrl_sensor(const GPUTronicHandle* h);   /* 0 CUDA, 1 tensor */
struct GPUTronicControl* gputronic_get_control(GPUTronicHandle* h);

/* Host-side pulse [F-03, F-04]. Host atomic add (__sync_fetch_and_add). Only
 * coherent with concurrent device atomicAdd on platforms with
 * hostNativeAtomicSupported; otherwise do not mix with a pulsing kernel. */
void gputronic_pulse(GPUTronicHandle* h, unsigned long long units);

int gputronic_zero_copy_ok(const GPUTronicHandle* h);
void gputronic_print_status(const GPUTronicHandle* h, int elapsed_s);

#ifdef __cplusplus
}
#endif

#endif /* GPUTRONIC_H */
