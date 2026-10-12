/* =============================================================================
 * GPUTronic Gold 1.0.0
 * -----------------------------------------------------------------------------
 * Read with: docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md  (GT-FF-GOLD-1.0)
 *
 * Walk order (do not start at main):
 *   gold_work / gold_persistent_kernel        device kernel [F-02]
 *   gputronic_create                          mapping + zero-copy check [F-01]
 *   gputronic_start / stop                    lifecycle [F-12]
 *   control_thread                            rate, Z, Kalman, controller [F-05..F-10]
 *   gold_build_table                          calibration table [F-09]
 *   kalman_update                             Z and dZ/dt filter [F-08]
 *   mode_dyno / mode_closedloop_check         checks [F-13]
 *
 * Contracts that define Gold (lost in late v26, restored here):
 *   1. One pulse total, only ever increased with atomicAdd [F-04]
 *   2. Periodic __threadfence_system; host barrier after writing active_sms [F-01]
 *   3. rate_ref from a sustained full-speed window, not a peak [F-06]
 *   4. Z = rate_ref / rate; fewer SMs raise Z [F-07]
 *   5. Two-sided PI(D) around a starting SM count [F-10]
 *   6. Two-state Kalman with P_pred = F P F^T + Q [F-08]
 *   7. Starting SM count and sms_per_u from a measured rate-vs-SM table [F-09]
 *   8. Batched device work between mapped atomics [F-02]
 *
 * Build: make gold
 * Run:   ./build/gputronic_gold check | dyno | step | run [sec] [Z] [Kp] [Ki]
 * ============================================================================= */

#include "gputronic.h"

#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ---- Device kernel: work, pulse counter, SM-count actuator [F-02, F-04, F-11] ---- */

/* Mapped atomics set the loop period (~300 µs on this 5080). The fmaf batch
 * is real ALU so DCE cannot eat the loop; it is not a FLOP claim [see F-14]. */
#define GOLD_WORK_ITERS 8
#define GOLD_BATCH 16

__device__ __forceinline__ void gold_work(float* acc, int sm, int iter) {
    float val = *acc;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        float a = 0.11f + 0.002f * (float)((iter + k + sm) & 31);
        float b = 0.19f + 0.003f * (float)((iter * 3 + k) & 31);
        val = __fmaf_rn(a, b, val);
    }
    *acc = val;
}

/* Tensor-core sensor work [F-16]: one 16x16x16 FP16 -> FP32 wmma per step,
 * accumulating into the same C fragment so the chain cannot be removed.
 * wmma needs sm_70+; older targets compile a stub that does no work. */
#define GOLD_TENSOR_STEPS 64

__device__ __forceinline__ float gold_tensor_work(int sm) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    using namespace nvcuda;
    __shared__ __align__(32) half a_tile[16 * 16];
    __shared__ __align__(32) half b_tile[16 * 16];
    const int lane = threadIdx.x & 31;
    for (int i = lane; i < 256; i += 32) {
        a_tile[i] = __float2half(0.01f * (float)((i + sm) & 15));
        b_tile[i] = __float2half(0.02f * (float)((i * 3) & 15));
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::load_matrix_sync(fa, a_tile, 16);
    wmma::load_matrix_sync(fb, b_tile, 16);
    wmma::fill_fragment(fc, 0.0f);
    for (int k = 0; k < GOLD_TENSOR_STEPS; k++) wmma::mma_sync(fc, fa, fb, fc);
    return fc.x[0];
#else
    (void)sm;
    return 0.0f;
#endif
}

/*
 * Persistent self-test kernel [F-02].
 *   grid = num_sm (one block / SM)
 *   warp leaders (lane==0) add to the pulse total and the per-SM diagnostic [F-04]
 *   Actuator: the lowest active_sms blocks do the work and pulse.
 *   Idle blocks backoff 5 µs and do not pulse. Duration is not the command.
 *   Two sensors [F-16]: warps below cuda_warps run the CUDA-core FMA batch
 *   and pulse total_work_pulses (unchanged). When the tensor sensor is on,
 *   one extra warp per block runs wmma and pulses tensor_work_pulses with
 *   the same fence cadence. active_sms gates both.
 * Period is atomic-dominated. Do not "tune Kp" to hide that.
 */
__global__ void gold_persistent_kernel(unsigned long long* sm_counters,
                                       GPUTronicControl* ctrl,
                                       int num_sm,
                                       int cuda_warps) {
    const int sm = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int is_tensor_warp = (tid >> 5) >= cuda_warps;
    float local = 0.0f;
    unsigned int local_pulses = 0;

    if (sm >= num_sm) return;

    while (true) {
        unsigned int flags = ctrl->control_flags;
        if (flags & GPUTRONIC_FLAG_STOP) break;
        if (flags & GPUTRONIC_FLAG_PAUSE) {
            __nanosleep(5000);
            continue;
        }
        if (flags & GPUTRONIC_FLAG_RESET) {
            if (tid == 0) sm_counters[sm] = 0ULL;
            __syncthreads();
            __nanosleep(1000);
            continue;
        }

        int active = ctrl->active_sms;
        if (active <= 0 || active > num_sm) active = num_sm;
        if (active < 4) active = 4;
        if (sm >= active) {
            __nanosleep(5000);
            continue;
        }

        if (is_tensor_warp) {
            float t = gold_tensor_work(sm);
            if (t > 1.0e30f) local += t;   /* keep the result live */
            if (lane == 0) {
                atomicAdd((unsigned long long*)&ctrl->tensor_work_pulses, 1ULL);
                local_pulses++;
                if ((local_pulses & 31u) == 0u) __threadfence_system();
            }
            continue;
        }

        for (int b = 0; b < GOLD_BATCH; b++) {
            for (int i = 0; i < GOLD_WORK_ITERS; i++)
                gold_work(&local, sm, i + b * GOLD_WORK_ITERS);
        }

        /* Prevent compiler from DCE-ing the work */
        if (local > 1.0e30f) local = 0.0f;

        if (lane == 0) {
            atomicAdd(&sm_counters[sm], 1ULL);
            atomicAdd((unsigned long long*)&ctrl->total_work_pulses, 1ULL);
            local_pulses++;
            /* Fence every 32 pulses — enough for host visibility. */
            if ((local_pulses & 31u) == 0u) __threadfence_system();
        }
    }
}

extern "C" void gold_launch_kernel(unsigned long long* sm_counters,
                                   GPUTronicControl* d_ctrl,
                                   int num_sm,
                                   int threads_per_block,
                                   int tensor_sensor) {
    /* CUDA-core warps stay as configured; the tensor sensor adds one warp. */
    int cuda_warps = (threads_per_block + 31) / 32;
    int threads = cuda_warps * 32 + (tensor_sensor ? 32 : 0);
    gold_persistent_kernel<<<num_sm, threads>>>(sm_counters, d_ctrl, num_sm, cuda_warps);
}

/* ---- Host handle and control thread ---- */

#define GOLD_MAX_PROBE 8

struct GPUTronicHandle {
    GPUTronicConfig cfg;
    GPUTronicControl* h_ctrl;
    GPUTronicControl* d_ctrl;
    unsigned long long* d_sm;   /* per-SM diagnostic counters (device memory) */
    pthread_t thr;
    int running;
    int started;
    int open_loop_sms;   /* <0 closed loop, 0 all SMs, >0 working-SM count */
    int held_sms;
    int zero_copy_ok;
    /* Calibration table: rate measured at each probe SM count [F-09]. */
    int probing;
    int probe_idx;
    int n_probe;
    int probe_n[GOLD_MAX_PROBE];
    double probe_rate[GOLD_MAX_PROBE];
    double probe_until_us;
    double probe_measure_from_us;
    unsigned long long probe_mark;
    double probe_mark_us;
    int probe_marked;
    float n_ff;          /* feed-forward working-SM count for target_z */
    float sms_per_u;     /* SMs per unit of u (= 1 / local dZ/dn) */
    /* Counter reset request, host API to control thread [F-04] */
    volatile int reset_req;   /* 1 = requested, 2 = flag raised, 0 = done */
    double reset_t_us;
    /* Kalman / PI state (control thread only) */
    float P[2][2];
    float z_hat;
    float dzdt_hat;
    float integral;
    float rate_ema;
    float rate_ref; /* free-run rate for Z = rate_ref / rate */
    double rate_win_dt;
    unsigned long long rate_win_pulses;
    int calibrated;
    double calibrate_until_us;
    double cal_sustain_from_us;
    unsigned long long cal_pulse_mark;
    double cal_mark_us;
    int cal_marked;
    unsigned long long prev_pulses;
    /* Tensor-core sensor [F-16]: same window/EMA/reference as the CUDA path. */
    int tensor_on;
    unsigned long long prev_pulses_t;
    unsigned long long rate_win_pulses_t;
    float rate_ema_t;
    float rate_ref_t;
    unsigned long long cal_pulse_mark_t;
    unsigned long long probe_mark_t;
    double probe_rate_t[GOLD_MAX_PROBE];
    float iclamp;         /* integral clamp in use, see gold_set_iclamp */
    float n_ff_s[2];      /* per-sensor starting SM count, 0 = CUDA, 1 = tensor */
    float spu_s[2];       /* per-sensor sms_per_u */
    int sel;              /* sensor driving the PI */
    float z_c, z_t;       /* latest per-sensor Z */
    double t0_us;
    FILE* csv;
    long long csv_rows;
    /* stats */
    double sum_z, sum_z2;
    long long n_z;
    int min_sms, max_sms;
};

static volatile int g_signal_stop = 0;

static inline double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
}

static void on_signal(int sig) {
    (void)sig;
    g_signal_stop = 1;
}

void gputronic_config_gold(GPUTronicConfig* cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->num_sm = 0; /* query */
    cfg->threads_per_block = GPUTRONIC_DEFAULT_THREADS_PER_BLOCK;
    cfg->control_dt_us = GPUTRONIC_DEFAULT_CONTROL_DT_US;
    cfg->max_sleep_ns = GPUTRONIC_DEFAULT_MAX_SLEEP_NS;
    cfg->base_sleep_ns = GPUTRONIC_DEFAULT_BASE_SLEEP_NS;
    cfg->target_z = GPUTRONIC_GOLD_TARGET_Z;
    cfg->kp = GPUTRONIC_GOLD_KP;
    cfg->ki = GPUTRONIC_GOLD_KI;
    cfg->kd = GPUTRONIC_GOLD_KD;
    cfg->integral_clamp = GPUTRONIC_GOLD_INTEGRAL_CLAMP;
    cfg->deadband = GPUTRONIC_GOLD_DEADBAND;
    cfg->z_ceiling = GPUTRONIC_GOLD_Z_CEILING;
    cfg->z_floor = GPUTRONIC_GOLD_Z_FLOOR;
    cfg->enable_kalman = 1;
    cfg->launch_selftest_kernel = 1;
    cfg->probe_ms = GPUTRONIC_GOLD_PROBE_MS;
    cfg->enable_tensor_sensor = 1;
    cfg->sensor_hysteresis = GPUTRONIC_GOLD_SENSOR_HYSTERESIS;
    cfg->reserve_sms = 0;
    cfg->sensor_rule = 0;
    /* Test overrides (defaults unchanged when unset):
     *   GPUTRONIC_TENSOR_SENSOR=0|1  turn the tensor-core sensor off or on
     *   GPUTRONIC_RESERVE_SMS=k      keep k extra SMs on above the calibrated start */
    const char* ev = getenv("GPUTRONIC_TENSOR_SENSOR");
    if (ev && *ev) cfg->enable_tensor_sensor = atoi(ev) ? 1 : 0;
    ev = getenv("GPUTRONIC_SENSOR_RULE");   /* max (default) | min */
    if (ev && *ev) cfg->sensor_rule = (strcmp(ev, "min") == 0) ? 1 : 0;
    ev = getenv("GPUTRONIC_RESERVE_SMS");
    if (ev && *ev) cfg->reserve_sms = atoi(ev) < 0 ? 0 : atoi(ev);
    cfg->csv_path = "results/gputronic_gold.csv";
}

static void kalman_update(GPUTronicHandle* h, float z_meas, float dt) {
    /* Constant-velocity model on [Z, dZ/dt] [F-08].
     * P_pred is the explicit F P F^T + Q expansion. Do not replace with P[i]+=Q. */
    float z_pred = h->z_hat + h->dzdt_hat * dt;
    float dz_pred = h->dzdt_hat;

    const float Q = 0.0008f;
    const float R = 0.018f;

    float p00 = h->P[0][0] + 2.0f * dt * h->P[0][1] + dt * dt * h->P[1][1] + Q;
    float p01 = h->P[0][1] + dt * h->P[1][1];
    float p10 = h->P[1][0] + dt * h->P[1][1];
    float p11 = h->P[1][1] + Q;

    float y = z_meas - z_pred;
    float S = p00 + R;
    float K0 = p00 / S;
    float K1 = p10 / S;

    h->z_hat = z_pred + K0 * y;
    h->dzdt_hat = dz_pred + K1 * y;

    h->P[0][0] = (1.0f - K0) * p00;
    h->P[0][1] = (1.0f - K0) * p01;
    h->P[1][0] = p10 - K1 * p00;
    h->P[1][1] = p11 - K1 * p01;
}

/* Calibration table: rate vs SM count gives the starting count and local slope at target_z [F-09].
 * Points are (n, Z(n) = rate_ref / rate(n)), n descending. Interpolate to
 * target_z; sms_per_u = 1 / |dZ/dn| on the bracketing segment, so u is in Z
 * units. Expectation from the committed dyno CSVs (sanity only, not used):
 * Z(n) is concave in n on the episode plant (flat above ~42 SMs), close to
 * 1/n-shaped on gold/transfer, so a single half-width point mis-scales it. */
static void gold_fit_table(GPUTronicHandle* h, const char* label, double rate_ref,
                           const double* rates, float* n_ff_out, float* spu_out) {
    const int N = h->cfg.num_sm;
    const float tz = h->cfg.target_z;
    float zs[GOLD_MAX_PROBE + 1];
    int ns[GOLD_MAX_PROBE + 1];
    int m = 0;
    ns[m] = N; zs[m] = 1.0f; m++;
    for (int i = 0; i < h->n_probe; i++) {
        if (rates[i] < 1.0) continue;
        ns[m] = h->probe_n[i];
        zs[m] = (float)(rate_ref / rates[i]);
        printf("[CAL] %s table n=%3d rate=%.1f Z=%.3f\n", label, ns[m], rates[i], zs[m]);
        m++;
    }
    float n_ff = (float)N, g = 0.0f;
    int found = 0;
    for (int i = 0; i + 1 < m; i++) {
        float za = zs[i], zb = zs[i + 1];   /* ns[i] > ns[i+1] */
        if ((tz >= za && tz <= zb) || (tz <= za && tz >= zb)) {
            float span = zb - za;
            float f = (fabsf(span) > 1e-6f) ? (tz - za) / span : 0.0f;
            n_ff = (float)ns[i] + f * (float)(ns[i + 1] - ns[i]);
            g = span / (float)(ns[i] - ns[i + 1]);
            found = 1;
            break;
        }
    }
    if (!found) {
        /* Target outside the measured range: pin to the nearest end and use
         * the slope of the overall table. */
        n_ff = (tz <= zs[0]) ? (float)N : (float)ns[m - 1];
        g = (m > 1) ? (zs[m - 1] - zs[0]) / (float)(ns[0] - ns[m - 1]) : 0.0f;
    }
    if (g < 1e-3f) g = 1e-3f;            /* flat segment: cap gain at 1000 SM/u */
    float spu = 1.0f / g;
    if (spu > (float)N) spu = (float)N;
    *n_ff_out = n_ff;
    *spu_out = spu;
    printf("[CAL] %s target Z=%.2f → n_ff=%.1f dZ/dn=%.4f sms_per_u=%.1f%s\n",
           label, tz, n_ff, g, spu, found ? "" : " (target outside table)");
}

/* Both sensors [F-09, F-16]. The PI starts on the sensor that reaches the
 * target with the most SMs still on (larger n_ff), i.e. the one whose Z is
 * higher at that count; this matches the max selector in control_thread. */
/* Integral clamp in use: the config value if > 0, otherwise scaled so that
 * Ki * clamp * sms_per_u = GPUTRONIC_GOLD_INTEGRAL_SMS_FRAC * num_sm SMs. Falls back
 * to 1.0 if Ki or sms_per_u is not positive. Recomputed when sms_per_u changes. */
static void gold_set_iclamp(GPUTronicHandle* h) {
    if (h->cfg.integral_clamp > 0.0f) { h->iclamp = h->cfg.integral_clamp; return; }
    float denom = h->cfg.ki * h->sms_per_u;
    h->iclamp = (denom > 1e-6f && h->cfg.num_sm > 0) ? GPUTRONIC_GOLD_INTEGRAL_SMS_FRAC * (float)h->cfg.num_sm / denom : 1.0f;
}

static void gold_build_table(GPUTronicHandle* h) {
    gold_fit_table(h, "cuda", h->rate_ref, h->probe_rate, &h->n_ff_s[0], &h->spu_s[0]);
    h->sel = 0;
    if (h->tensor_on) {
        gold_fit_table(h, "tensor", h->rate_ref_t, h->probe_rate_t, &h->n_ff_s[1], &h->spu_s[1]);
        /* max rule: start on the sensor that needs fewer SMs off; min rule: the other one. */
        if (h->cfg.sensor_rule == 0 ? (h->n_ff_s[1] > h->n_ff_s[0]) : (h->n_ff_s[1] < h->n_ff_s[0])) h->sel = 1;
    }
    /* Reserve: shift both starting counts toward more SMs (lower Z). */
    for (int i = 0; i < 2; i++) {
        h->n_ff_s[i] += (float)h->cfg.reserve_sms;
        if (h->n_ff_s[i] > (float)h->cfg.num_sm) h->n_ff_s[i] = (float)h->cfg.num_sm;
    }
    if (h->cfg.reserve_sms > 0) printf("[CAL] reserve: +%d SMs on the starting count\n", h->cfg.reserve_sms);
    h->n_ff = h->n_ff_s[h->sel];
    h->sms_per_u = h->spu_s[h->sel];
    printf("[CAL] controller starts on the %s sensor\n", h->sel ? "tensor" : "cuda");
    gold_set_iclamp(h);
    printf("[CAL] integral clamp=%.3f (%s, max integral correction %.2f SMs of %d)\n", h->iclamp,
           h->cfg.integral_clamp > 0.0f ? "config" : "scaled", h->cfg.ki * h->iclamp * h->sms_per_u, h->cfg.num_sm);
}

static void gold_reset_rate_state(GPUTronicHandle* h) {
    h->prev_pulses = h->h_ctrl->total_work_pulses;
    h->rate_win_pulses = 0;
    h->rate_win_dt = 0.0;
    h->rate_ema = 0.0f;
    h->prev_pulses_t = h->h_ctrl->tensor_work_pulses;
    h->rate_win_pulses_t = 0;
    h->rate_ema_t = 0.0f;
}

static void* control_thread(void* arg) {
    /* Control thread [F-12]. Each tick: rate [F-05], Z [F-06, F-07], Kalman [F-08], controller [F-10], write active_sms. */
    GPUTronicHandle* h = (GPUTronicHandle*)arg;
    const float dt_target = h->cfg.control_dt_us * 1e-6f;
    double last = now_us();
    double hz_ema = 1.0 / dt_target;
    int tick = 0;
    const int N = h->cfg.num_sm;

    h->P[0][0] = 1.0f;
    h->P[0][1] = 0.0f;
    h->P[1][0] = 0.0f;
    h->P[1][1] = 1.0f;
    h->z_hat = 1.0f;
    h->dzdt_hat = 0.0f;
    h->integral = 0.0f;
    h->rate_ref = 0.0f;
    h->rate_ref_t = 0.0f;
    h->sel = 0;
    h->calibrated = 0;
    /* Warmup 0.7 s, then a ~1.1 s sustained window (not peak EMA). */
    h->calibrate_until_us = now_us() + 1800000.0;
    h->cal_sustain_from_us = now_us() + 700000.0;
    h->cal_pulse_mark = 0;
    h->cal_mark_us = 0.0;
    h->cal_marked = 0;
    gold_reset_rate_state(h);
    h->min_sms = N;
    h->max_sms = 0;
    h->n_ff = (float)N;

    /* Probe counts: 3/4, 1/2, 1/4, floor. Full width is the rate_ref point. */
    {
        int c[4] = {N * 3 / 4, N / 2, N / 4, 4};
        h->n_probe = 0;
        for (int i = 0; i < 4; i++) {
            int n = c[i] < 4 ? 4 : c[i];
            if (n >= N) continue;
            if (h->n_probe > 0 && h->probe_n[h->n_probe - 1] == n) continue;
            h->probe_n[h->n_probe] = n;
            h->probe_rate[h->n_probe] = 0.0;
            h->probe_rate_t[h->n_probe] = 0.0;
            h->n_probe++;
        }
    }
    const double probe_us = (h->cfg.probe_ms > 0 ? h->cfg.probe_ms : GPUTRONIC_GOLD_PROBE_MS) * 1000.0;

    printf("[CTRL] Gold | target=%.2f Kp=%.3f Ki=%.4f Kd=%.4f deadband=%.3f kalman=%d two-sided | cal 1.8s + %d-point table\n",
           h->cfg.target_z, h->cfg.kp, h->cfg.ki, h->cfg.kd, h->cfg.deadband,
           h->cfg.enable_kalman, h->n_probe);

    while (h->running && !g_signal_stop) {
        double t = now_us();
        double elapsed_us = t - last;
        if (elapsed_us < h->cfg.control_dt_us * 0.85) {
            usleep(1);
            continue;
        }
        float dt = (float)(elapsed_us * 1e-6);
        if (dt < 1e-7f) dt = dt_target;
        if (dt > 0.01f) dt = 0.01f;
        last = t;

        /* ---- Counter reset, requested by gputronic_reset_counters [F-04] ----
         * Raise RESET so blocks stop pulsing, wait out an in-flight batch
         * (~300 µs period; 2 ms margin), then zero the mapped total from the
         * host while no device atomics are issuing, and restart rate state. */
        if (h->reset_req == 1) {
            h->h_ctrl->control_flags |= GPUTRONIC_FLAG_RESET;
            __sync_synchronize();
            h->reset_t_us = t;
            h->reset_req = 2;
            continue;
        }
        if (h->reset_req == 2) {
            if (t - h->reset_t_us < 2000.0) continue;
            h->h_ctrl->total_work_pulses = 0ULL;
            __sync_synchronize();
            h->h_ctrl->control_flags &= ~GPUTRONIC_FLAG_RESET;
            __sync_synchronize();
            gold_reset_rate_state(h);
            h->reset_req = 0;
            continue;
        }

        unsigned long long pulses = h->h_ctrl->total_work_pulses;
        unsigned long long dpulse = pulses - h->prev_pulses;
        h->prev_pulses = pulses;

        /* Rate over an ~8 ms window, not one 10 µs tick [F-05]. */
        unsigned long long pulses_t = h->h_ctrl->tensor_work_pulses;
        h->rate_win_pulses_t += pulses_t - h->prev_pulses_t;
        h->prev_pulses_t = pulses_t;
        h->rate_win_pulses += dpulse;
        h->rate_win_dt += (double)dt;
        if (h->rate_win_dt >= 0.008) {
            float inst_rate = (float)((double)h->rate_win_pulses / h->rate_win_dt);
            float inst_t = (float)((double)h->rate_win_pulses_t / h->rate_win_dt);
            h->rate_win_pulses = 0;
            h->rate_win_pulses_t = 0;
            h->rate_win_dt = 0.0;
            if (h->rate_ema <= 1.0f)
                h->rate_ema = inst_rate;
            else
                h->rate_ema = 0.70f * h->rate_ema + 0.30f * inst_rate;
            if (h->rate_ema_t <= 1.0f)
                h->rate_ema_t = inst_t;
            else
                h->rate_ema_t = 0.70f * h->rate_ema_t + 0.30f * inst_t;
        }

        /* ---- Calibration: reference rate, then table [F-06, F-09] ---- */
        float z_raw;
        int n_cmd = N;
        if (!h->calibrated) {
            if (!h->cal_marked && t >= h->cal_sustain_from_us) {
                h->cal_pulse_mark = pulses;
                h->cal_pulse_mark_t = pulses_t;
                h->cal_mark_us = t;
                h->cal_marked = 1;
            }
            if (!h->probing && t >= h->calibrate_until_us && h->cal_marked) {
                double dt_cal = (t - h->cal_mark_us) * 1e-6;
                unsigned long long dp_cal = pulses - h->cal_pulse_mark;
                if (dt_cal > 0.2 && dp_cal > 1000ULL)
                    h->rate_ref = (float)((double)dp_cal / dt_cal);
                else
                    h->rate_ref = h->rate_ema;
                printf("[CAL] rate_ref=%.1f pulses/s (sustained %.2fs window)\n", h->rate_ref, dt_cal);
                if (h->tensor_on) {
                    unsigned long long dp_t = pulses_t - h->cal_pulse_mark_t;
                    h->rate_ref_t = (dt_cal > 0.2) ? (float)((double)dp_t / dt_cal) : h->rate_ema_t;
                    printf("[CAL] rate_ref_tensor=%.1f pulses/s\n", h->rate_ref_t);
                    if (dp_t < 100ULL) {
                        printf("[CAL] tensor sensor produced no pulses; continuing with CUDA only\n");
                        h->tensor_on = 0;
                    }
                }
                h->probing = 1;
                h->probe_idx = 0;
                h->probe_marked = 0;
                h->probe_measure_from_us = t + probe_us / 3.0;  /* settle first third */
                h->probe_until_us = t + probe_us;
            }
            if (h->probing) {
                int i = h->probe_idx;
                n_cmd = h->probe_n[i];
                if (!h->probe_marked && t >= h->probe_measure_from_us) {
                    h->probe_mark = pulses;
                    h->probe_mark_t = pulses_t;
                    h->probe_mark_us = t;
                    h->probe_marked = 1;
                }
                if (t >= h->probe_until_us) {
                    double dt_p = (t - h->probe_mark_us) * 1e-6;
                    unsigned long long dp_p = pulses - h->probe_mark;
                    h->probe_rate[i] = (h->probe_marked && dt_p > 0.02) ? (double)dp_p / dt_p : 0.0;
                    h->probe_rate_t[i] = (h->probe_marked && dt_p > 0.02)
                                             ? (double)(pulses_t - h->probe_mark_t) / dt_p : 0.0;
                    h->probe_idx++;
                    if (h->probe_idx >= h->n_probe) {
                        gold_build_table(h);
                        h->probing = 0;
                        h->calibrated = 1;
                        h->integral = 0.0f;
                        h->held_sms = (int)(h->n_ff + 0.5f);
                        gold_reset_rate_state(h);   /* EMA restarts at the new width */
                        n_cmd = h->held_sms;
                    } else {
                        h->probe_marked = 0;
                        h->probe_measure_from_us = t + probe_us / 3.0;
                        h->probe_until_us = t + probe_us;
                        n_cmd = h->probe_n[h->probe_idx];
                    }
                }
            }
            z_raw = 1.0f; /* nominal during cal and the table probe */
        } else if (h->rate_ema <= 1.0f) {
            z_raw = h->z_hat;   /* no window since reset/probe: hold */
        } else {
            /*
             * Two sensors, one actuator [F-16]. Both Z rise as SMs drop, so
             * the PI runs on the larger one (max selector): the target is met
             * by removing the fewest SMs, and neither path is slowed further
             * than requested. Hysteresis stops chatter when the two are close.
             * On a switch the integral is recomputed so the commanded SM count
             * does not jump (bumpless transfer), and the Kalman state is moved
             * to the new sensor's value.
             */
            h->z_c = h->rate_ref / (h->rate_ema + 1.0f);
            h->z_t = (h->tensor_on && h->rate_ema_t > 1.0f) ? h->rate_ref_t / (h->rate_ema_t + 1.0f) : 0.0f;
            int want = h->sel;
            /*
             * Rule comparison (model, RTX 5080 dyno curves, 2026-10-12; not GPU-tested).
             * Default is max (follow the higher Z). The min rule (GPUTRONIC_SENSOR_RULE=min)
             * lost: with a spec-ideal tensor curve it lets the other sensor's Z go past 1.5
             * (Gold 1.83, Transfer 1.59, Episode 3.65 peak); with the realistic atomic-bound
             * curve it settles lower than max (Z error Gold -0.052, Transfer -0.106,
             * Episode -0.011 vs max -0.023 / -0.052 / +0.003). Neither rule recovers to
             * within 0.02 of target after a sustained -20% load step (integral clamp).
             */
            if (h->tensor_on && h->cfg.sensor_rule == 0) {          /* follow higher Z */
                if (h->sel == 0 && h->z_t > h->z_c + h->cfg.sensor_hysteresis) want = 1;
                if (h->sel == 1 && h->z_c > h->z_t + h->cfg.sensor_hysteresis) want = 0;
            } else if (h->tensor_on) {                              /* follow lower Z */
                if (h->sel == 0 && h->z_t < h->z_c - h->cfg.sensor_hysteresis) want = 1;
                if (h->sel == 1 && h->z_c < h->z_t - h->cfg.sensor_hysteresis) want = 0;
            }
            float z_new = want ? h->z_t : h->z_c;
            if (want != h->sel) {
                h->sel = want;
                h->n_ff = h->n_ff_s[want];
                h->sms_per_u = h->spu_s[want];
                gold_set_iclamp(h);
                float e_new = h->cfg.target_z - z_new;
                float i_new = (h->cfg.ki > 1e-6f && h->sms_per_u > 1e-6f)
                                  ? ((h->n_ff - (float)h->held_sms) / h->sms_per_u - h->cfg.kp * e_new) / h->cfg.ki
                                  : 0.0f;
                if (i_new > h->iclamp) i_new = h->iclamp;
                if (i_new < -h->iclamp) i_new = -h->iclamp;
                h->integral = i_new;
                h->z_hat = z_new;
                h->dzdt_hat = 0.0f;
            }
            z_raw = z_new;
        }
        if (z_raw > h->cfg.z_ceiling) z_raw = h->cfg.z_ceiling;
        if (z_raw < h->cfg.z_floor) z_raw = h->cfg.z_floor;

        float z = z_raw;
        if (h->cfg.enable_kalman && h->calibrated) {
            kalman_update(h, z_raw, dt);
            z = h->z_hat;
            if (z > h->cfg.z_ceiling) z = h->cfg.z_ceiling;
            if (z < h->cfg.z_floor) z = h->cfg.z_floor;
        } else {
            h->z_hat = z_raw;
            h->dzdt_hat = 0.0f;
        }

        float error = 0.0f;
        if (h->probing || !h->calibrated) {
            /* n_cmd set by the calibration block above */
            h->integral = 0.0f;
        } else if (h->open_loop_sms > 0) {
            n_cmd = h->open_loop_sms;
            h->integral = 0.0f;
        } else if (h->open_loop_sms == 0) {
            n_cmd = N;
            h->integral = 0.0f;
        } else {
            /*
             * Two-sided PI(D) around the starting count [F-10]:
             *   u = Kp*e + Ki*I - Kd*dZ/dt,  n = n_ff - u*sms_per_u.
             * e > 0 (Z below target) drops SMs; e < 0 adds them back
             * gradually. Inside the deadband the count is held and the
             * integral frozen. Anti-windup: clamp I, and do not integrate
             * further into a saturated actuator.
             */
            error = h->cfg.target_z - z;
            n_cmd = h->held_sms;
            if (n_cmd < 4 || n_cmd > N) n_cmd = N;
            if (fabsf(error) >= h->cfg.deadband) {
                float u_p = h->cfg.kp * error - h->cfg.kd * h->dzdt_hat;
                float i_new = h->integral + error * dt;
                if (i_new > h->iclamp) i_new = h->iclamp;
                if (i_new < -h->iclamp) i_new = -h->iclamp;
                float n_try = h->n_ff - (u_p + h->cfg.ki * i_new) * h->sms_per_u;
                int saturated = (n_try < 4.0f && error > 0.0f) ||
                                (n_try > (float)N && error < 0.0f);
                if (!saturated) h->integral = i_new;
                float u = u_p + h->cfg.ki * h->integral;
                float nf = h->n_ff - u * h->sms_per_u;
                n_cmd = (int)floorf(nf + 0.5f);
            }
        }

        if (n_cmd < 4) n_cmd = 4;
        if (n_cmd > N) n_cmd = N;
        h->held_sms = n_cmd;
        h->h_ctrl->active_sms = n_cmd;
        __sync_synchronize();

        h->h_ctrl->target_z = h->cfg.target_z;
        h->h_ctrl->z_raw = z_raw;
        h->h_ctrl->z_hat = z;
        h->h_ctrl->dzdt_hat = h->dzdt_hat;
        h->h_ctrl->rate_ema = h->rate_ema;
        h->h_ctrl->z_cuda = h->z_c;
        h->h_ctrl->z_tensor = h->z_t;
        h->h_ctrl->rate_tensor_ema = h->rate_ema_t;
        h->h_ctrl->ctrl_sensor = h->sel;
        h->h_ctrl->pm_error = error;
        h->h_ctrl->integral = h->integral;
        if (n_cmd < N) h->h_ctrl->proactive_corrections++;

        double inst_hz = 1.0 / dt;
        hz_ema = 0.95 * hz_ema + 0.05 * inst_hz;
        h->h_ctrl->control_hz_ema = (int)(hz_ema + 0.5);

        if (h->calibrated) {
            if (n_cmd < h->min_sms) h->min_sms = n_cmd;
            if (n_cmd > h->max_sms) h->max_sms = n_cmd;
            h->sum_z += z;
            h->sum_z2 += (double)z * (double)z;
            h->n_z++;
        }

        if (h->csv) {
            double t_s = (t - h->t0_us) * 1e-6;
            fprintf(h->csv, "%.6f,%.5f,%.5f,%.5f,%.1f,%d,%.5f,%.5f,%d,%.1f,%.5f,%.5f,%.1f,%.1f,%d\n",
                    t_s, z_raw, z, h->dzdt_hat, h->rate_ema, n_cmd,
                    error, h->integral, h->h_ctrl->control_hz_ema, h->rate_ref,
                    h->z_c, h->z_t, h->rate_ema_t, h->rate_ref_t, h->sel);
            h->csv_rows++;
            if ((tick & 1023) == 0) fflush(h->csv);
        }
        tick++;
    }
    return NULL;
}

/* Zero-copy round trip [F-01]: device reads a host-written word through the mapped
 * pointer and writes back its complement. */
__global__ void gold_zc_probe_kernel(GPUTronicControl* ctrl) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int v = *(volatile int*)&ctrl->proactive_corrections;
        *(volatile int*)&ctrl->control_hz_ema = ~v;
        __threadfence_system();
    }
}

static int gold_verify_zero_copy(GPUTronicHandle* h) {
    cudaPointerAttributes attr;
    memset(&attr, 0, sizeof(attr));
    cudaError_t err = cudaPointerGetAttributes(&attr, h->h_ctrl);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ZC ] cudaPointerGetAttributes failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    int type_ok = (attr.type == cudaMemoryTypeHost);
    int host_ok = (attr.hostPointer == (void*)h->h_ctrl);
    int dev_ok = (attr.devicePointer == (void*)h->d_ctrl) && h->d_ctrl != NULL;
    const int magic = 0x5A17C0DE;
    *(volatile int*)&h->h_ctrl->proactive_corrections = magic;
    *(volatile int*)&h->h_ctrl->control_hz_ema = 0;
    __sync_synchronize();
    gold_zc_probe_kernel<<<1, 1>>>(h->d_ctrl);
    err = cudaDeviceSynchronize();
    int rt_ok = (err == cudaSuccess) &&
                (*(volatile int*)&h->h_ctrl->control_hz_ema == ~magic);
    h->h_ctrl->proactive_corrections = 0;
    h->h_ctrl->control_hz_ema = 0;
    printf("[ZC ] type=%s host_ptr=%s dev_ptr=%s round_trip=%s (host %p, device %p%s)\n",
           type_ok ? "host" : "NOT-HOST", host_ok ? "OK" : "MISMATCH",
           dev_ok ? "OK" : "MISMATCH", rt_ok ? "OK" : "FAIL",
           (void*)h->h_ctrl, (void*)h->d_ctrl,
           (h->h_ctrl == h->d_ctrl) ? ", same address (UVA)" : "");
    return type_ok && host_ok && dev_ok && rt_ok;
}
GPUTronicHandle* gputronic_create(const GPUTronicConfig* cfg_in) {
    /* Map the shared control block and query the SM count. No copy-based fallback [F-01]. */
    GPUTronicHandle* h = (GPUTronicHandle*)calloc(1, sizeof(GPUTronicHandle));
    if (!h) return NULL;

    if (cfg_in)
        h->cfg = *cfg_in;
    else
        gputronic_config_gold(&h->cfg);

    g_signal_stop = 0;
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPUTronic] cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        free(h);
        return NULL;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (h->cfg.num_sm <= 0) h->cfg.num_sm = prop.multiProcessorCount;
    if (h->cfg.threads_per_block <= 0) h->cfg.threads_per_block = 64;
    if (h->cfg.control_dt_us <= 0) h->cfg.control_dt_us = 10;

    /* Mapped control block */
    err = cudaHostAlloc((void**)&h->h_ctrl, sizeof(GPUTronicControl), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPUTronic] cudaHostAlloc control failed: %s\n", cudaGetErrorString(err));
        free(h);
        return NULL;
    }
    err = cudaHostGetDevicePointer((void**)&h->d_ctrl, h->h_ctrl, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPUTronic] cudaHostGetDevicePointer failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h->h_ctrl);
        free(h);
        return NULL;
    }

    memset(h->h_ctrl, 0, sizeof(GPUTronicControl));
    h->zero_copy_ok = gold_verify_zero_copy(h);
    if (!h->zero_copy_ok) {
        fprintf(stderr, "[GPUTronic] zero-copy verification failed; no PCIe fallback\n");
        cudaFreeHost(h->h_ctrl);
        free(h);
        return NULL;
    }
    h->h_ctrl->max_sleep_ns = h->cfg.max_sleep_ns;
    h->h_ctrl->base_sleep_ns = h->cfg.base_sleep_ns;
    h->h_ctrl->throttle_sleep_ns = h->cfg.base_sleep_ns;
    h->h_ctrl->target_z = h->cfg.target_z;
    h->h_ctrl->num_sm = h->cfg.num_sm;

    /* Per-SM counters in device memory */
    err = cudaMalloc((void**)&h->d_sm, (size_t)h->cfg.num_sm * sizeof(unsigned long long));
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPUTronic] cudaMalloc sm counters failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h->h_ctrl);
        free(h);
        return NULL;
    }
    cudaMemset(h->d_sm, 0, (size_t)h->cfg.num_sm * sizeof(unsigned long long));

    h->open_loop_sms = -1;
    h->held_sms = h->cfg.num_sm;
    h->n_ff = (float)h->cfg.num_sm;
    h->sms_per_u = 1.0f / 0.03f;   /* placeholder until the calibration table [F-09] */
    gold_set_iclamp(h);
    /* Tensor sensor only runs inside the self-test kernel, and wmma needs sm_70+. */
    if (!h->cfg.launch_selftest_kernel || prop.major < 7) h->cfg.enable_tensor_sensor = 0;
    h->tensor_on = h->cfg.enable_tensor_sensor;
    printf("[SNS] tensor sensor %s\n", h->tensor_on ? "on (one wmma warp per block)" : "off");
    h->h_ctrl->active_sms = h->cfg.num_sm;
    h->running = 0;
    h->started = 0;

    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic Gold %s]\n", GPUTRONIC_VERSION_STRING);
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPU] %s | SMs=%d (using %d) | thr/block=%d\n",
           prop.name, prop.multiProcessorCount, h->cfg.num_sm, h->cfg.threads_per_block);
    printf("[LAW] target_z=%.2f Kp=%.3f Ki=%.4f Kd=%.4f deadband=%.3f width-actuator two-sided\n",
           h->cfg.target_z, h->cfg.kp, h->cfg.ki, h->cfg.kd, h->cfg.deadband);
    printf("[LOOP] dt=%d us (%.0f kHz) active_sms=%d floor=4\n\n",
           h->cfg.control_dt_us, 1000.0 / h->cfg.control_dt_us, h->cfg.num_sm);

    return h;
}

int gputronic_start(GPUTronicHandle* h) {
    if (!h || h->started) return -1;

    /* CSV */
    const char* path = h->cfg.csv_path ? h->cfg.csv_path : "results/gputronic_gold.csv";
    {
        /* best-effort results dir */
        int mk = system("mkdir -p results");
        (void)mk;
    }
    h->csv = fopen(path, "w");
    if (h->csv) {
        fprintf(h->csv, "time_s,z_raw,z_hat,dzdt,rate_ema,active_sms,error,integral,ctrl_hz,rate_ref,"
                         "z_cuda,z_tensor,rate_tensor_ema,rate_ref_tensor,ctrl_sensor\n");
        printf("[CSV] %s\n", path);
    } else {
        fprintf(stderr, "[WARN] could not open CSV %s\n", path);
    }

    h->t0_us = now_us();
    h->running = 1;

    if (h->cfg.launch_selftest_kernel) {
        gold_launch_kernel(h->d_sm, h->d_ctrl, h->cfg.num_sm, h->cfg.threads_per_block,
                           h->cfg.enable_tensor_sensor);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[GPUTronic] kernel launch failed: %s\n", cudaGetErrorString(err));
            h->running = 0;
            if (h->csv) fclose(h->csv);
            h->csv = NULL;
            return -1;
        }
        printf("[OK] Persistent self-test kernel launched (%d blocks x %d threads)\n",
               h->cfg.num_sm, h->cfg.threads_per_block);
    }

    if (pthread_create(&h->thr, NULL, control_thread, h) != 0) {
        fprintf(stderr, "[GPUTronic] pthread_create failed\n");
        h->h_ctrl->control_flags |= GPUTRONIC_FLAG_STOP;
        h->running = 0;
        return -1;
    }

    h->started = 1;
    return 0;
}

void gputronic_stop(GPUTronicHandle* h) {
    if (!h || !h->started) return;
    h->running = 0;
    g_signal_stop = 1;
    if (h->h_ctrl) h->h_ctrl->control_flags |= GPUTRONIC_FLAG_STOP;
    pthread_join(h->thr, NULL);
    cudaDeviceSynchronize();

    if (h->csv) {
        fflush(h->csv);
        fclose(h->csv);
        h->csv = NULL;
    }

    if (h->n_z > 0) {
        double mean = h->sum_z / (double)h->n_z;
        double var = h->sum_z2 / (double)h->n_z - mean * mean;
        if (var < 0) var = 0;
        printf("\n[STATS] samples=%lld mean_Z=%.4f std_Z=%.4f active_sms=%d..%d rows_csv=%lld\n",
               (long long)h->n_z, mean, sqrt(var), h->min_sms, h->max_sms,
               (long long)h->csv_rows);
    }
    h->started = 0;
    printf("[GPUTronic Gold] Stopped cleanly.\n");
}

void gputronic_destroy(GPUTronicHandle* h) {
    if (!h) return;
    if (h->started) gputronic_stop(h);
    if (h->d_sm) cudaFree(h->d_sm);
    if (h->h_ctrl) cudaFreeHost(h->h_ctrl);
    free(h);
}

void gputronic_set_target(GPUTronicHandle* h, float target_z) {
    if (!h) return;
    h->cfg.target_z = target_z;
    if (h->h_ctrl) h->h_ctrl->target_z = target_z;
}

void gputronic_set_gains(GPUTronicHandle* h, float kp, float ki) {
    if (!h) return;
    h->cfg.kp = kp;
    h->cfg.ki = ki;
}

void gputronic_set_kd(GPUTronicHandle* h, float kd) {
    if (!h) return;
    h->cfg.kd = kd;
}

void gputronic_set_deadband(GPUTronicHandle* h, float deadband) {
    if (!h) return;
    h->cfg.deadband = (deadband < 0.0f) ? 0.0f : deadband;
}

void gputronic_set_open_loop_sms(GPUTronicHandle* h, int n) {
    if (!h) return;
    h->open_loop_sms = n;
}

int gputronic_reset_counters(GPUTronicHandle* h) {
    if (!h || !h->h_ctrl) return -1;
    if (!h->started) {
        /* No kernel/thread running: plain host writes are safe. */
        h->h_ctrl->total_work_pulses = 0ULL;
        cudaMemset(h->d_sm, 0, (size_t)h->cfg.num_sm * sizeof(unsigned long long));
        return 0;
    }
    h->reset_req = 1;
    for (int i = 0; i < 5000 && h->reset_req != 0; i++) usleep(10);
    return h->reset_req == 0 ? 0 : -1;
}

float gputronic_get_z(const GPUTronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->z_hat : 0.0f;
}

float gputronic_get_z_raw(const GPUTronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->z_raw : 0.0f;
}

float gputronic_get_rate(const GPUTronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->rate_ema : 0.0f;
}

float gputronic_get_z_cuda(const GPUTronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->z_cuda : 0.0f;
}

float gputronic_get_z_tensor(const GPUTronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->z_tensor : 0.0f;
}

float gputronic_get_rate_tensor(const GPUTronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->rate_tensor_ema : 0.0f;
}

int gputronic_get_ctrl_sensor(const GPUTronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->ctrl_sensor : 0;
}

int gputronic_get_active_sms(const GPUTronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->active_sms : 0;
}

struct GPUTronicControl* gputronic_get_control(GPUTronicHandle* h) {
    return h ? h->h_ctrl : NULL;
}

void gputronic_pulse(GPUTronicHandle* h, unsigned long long units) {
    if (!h || !h->h_ctrl || units == 0) return;
    /* Host atomic RMW (was a plain +=, which lost updates against itself and
     * against device atomics). See header for the device-coherence caveat. */
    __sync_fetch_and_add((unsigned long long*)&h->h_ctrl->total_work_pulses, units);
}

int gputronic_zero_copy_ok(const GPUTronicHandle* h) {
    return h ? h->zero_copy_ok : 0;
}

void gputronic_print_status(const GPUTronicHandle* h, int elapsed_s) {
    if (!h || !h->h_ctrl) return;
    printf("[TEL] t=%4ds | Z_raw=%.3f Z_hat=%.3f dZ=%.3f | Zc=%.3f Zt=%.3f sel=%s | rate=%.0f rate_t=%.0f | n=%3d | hz~%d | pulses=%llu\n",
           elapsed_s,
           h->h_ctrl->z_raw,
           h->h_ctrl->z_hat,
           h->h_ctrl->dzdt_hat,
           h->h_ctrl->z_cuda,
           h->h_ctrl->z_tensor,
           h->h_ctrl->ctrl_sensor ? "tensor" : "cuda",
           h->h_ctrl->rate_ema,
           h->h_ctrl->rate_tensor_ema,
           h->h_ctrl->active_sms,
           h->h_ctrl->control_hz_ema,
           (unsigned long long)h->h_ctrl->total_work_pulses);
}

/* ---- Command-line modes [F-13] ---- */

static int mode_run(int duration_s, float target, float kp, float ki) {
    GPUTronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.target_z = target;
    cfg.kp = kp;
    cfg.ki = ki;
    cfg.csv_path = "results/gputronic_gold_run.csv";

    GPUTronicHandle* h = gputronic_create(&cfg);
    if (!h) return 1;
    if (gputronic_start(h) != 0) {
        gputronic_destroy(h);
        return 1;
    }

    for (int s = 0; s < duration_s && !g_signal_stop; s++) {
        usleep(1000000);
        gputronic_print_status(h, s + 1);
    }

    gputronic_stop(h);
    gputronic_destroy(h);
    return 0;
}

/* Dyno: open-loop working-SM sweep. Rate = wall Δpulses/Δt. */
static int mode_dyno(void) {
    GPUTronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_gold_dyno.csv";

    GPUTronicHandle* h = gputronic_create(&cfg);
    if (!h) return 1;
    if (gputronic_start(h) != 0) {
        gputronic_destroy(h);
        return 1;
    }

    int num_sm = h->cfg.num_sm;
    int counts[] = {num_sm, num_sm * 3 / 4, num_sm * 2 / 3, num_sm / 2,
                    num_sm / 3, num_sm / 4, 8, 4};
    const int n = (int)(sizeof(counts) / sizeof(counts[0]));
    double xs[16], ys[16];

    /* Free-run settle covers calibration and the table probe (~1.8 s + 4×0.3 s). */
    gputronic_set_open_loop_sms(h, 0);
    usleep(3500000);

    printf("\n=== DYNO SWEEP (open-loop working SMs → rate) ===\n");
    printf("%10s %14s %10s %10s\n", "n_sm", "rate", "Z_hat", "pulses_d");
    printf("--------------------------------------------------------\n");

    FILE* df = fopen("results/gputronic_gold_dyno_summary.csv", "w");
    if (df) fprintf(df, "active_sms,rate,z_hat,delta_pulses\n");

    for (int i = 0; i < n && !g_signal_stop; i++) {
        int nsm = counts[i];
        if (nsm < 4) nsm = 4;
        gputronic_set_open_loop_sms(h, nsm);
        usleep(800000);
        unsigned long long p0 = h->h_ctrl->total_work_pulses;
        double t0 = now_us();
        double z_acc = 0.0;
        const int samples = 20;
        for (int k = 0; k < samples; k++) {
            usleep(40000);
            z_acc += h->h_ctrl->z_hat;
        }
        unsigned long long p1 = h->h_ctrl->total_work_pulses;
        double t1 = now_us();
        double dt_s = (t1 - t0) * 1e-6;
        if (dt_s < 1e-6) dt_s = 1e-6;
        double rate = (double)(p1 - p0) / dt_s;
        double z = z_acc / samples;
        unsigned long long dp = p1 - p0;
        xs[i] = (double)nsm;
        ys[i] = rate;
        printf("%10d %14.1f %10.4f %10llu\n", nsm, rate, z, (unsigned long long)dp);
        if (df) fprintf(df, "%d,%.3f,%.5f,%llu\n", nsm, rate, z, (unsigned long long)dp);
    }
    if (df) fclose(df);

    double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (int i = 0; i < n; i++) {
        sx += xs[i];
        sy += ys[i];
        sxx += xs[i] * xs[i];
        sxy += xs[i] * ys[i];
        syy += ys[i] * ys[i];
    }
    double denom = n * sxx - sx * sx;
    double b = (fabs(denom) > 1e-12) ? (n * sxy - sx * sy) / denom : 0.0;
    double a = (sy - b * sx) / n;
    double ss_tot = syy - (sy * sy) / n;
    double ss_res = 0.0;
    for (int i = 0; i < n; i++) {
        double e = ys[i] - (a + b * xs[i]);
        ss_res += e * e;
    }
    double r2 = (ss_tot > 1e-12) ? (1.0 - ss_res / ss_tot) : 0.0;

    /* Plant OK if the narrowest count cuts rate by >= 25% vs full width. */
    int plant_ok = (ys[0] > 1000.0) && (ys[n - 1] < ys[0] * 0.75);
    int decreasing = 1;
    for (int i = 1; i < n; i++) {
        if (ys[i] > ys[i - 1] * 1.15) decreasing = 0;
    }

    printf("--------------------------------------------------------\n");
    printf("fit: rate ≈ %.1f + (%.6f)*active_sms\n", a, b);
    printf("R² = %.4f  | plant_ok=%s | mostly_decreasing=%s\n",
           r2, plant_ok ? "YES" : "NO", decreasing ? "YES" : "NO");
    printf("PASS: fewer SMs cut rate >= 25%% AND (R²>=0.85 OR mostly_decreasing)\n");
    int pass = plant_ok && (r2 >= 0.85 || decreasing);
    printf("RESULT: %s\n\n", pass ? "PASS" : "CHECK");

    FILE* rf = fopen("results/gputronic_gold_dyno_report.txt", "w");
    if (rf) {
        fprintf(rf, "GPUTronic Gold Dyno Report\n");
        fprintf(rf, "R2=%.4f slope=%.6f intercept=%.1f plant_ok=%d decreasing=%d PASS=%d\n",
                r2, b, a, plant_ok, decreasing, pass);
        for (int i = 0; i < n; i++)
            fprintf(rf, "active_sms=%d rate=%.1f\n", (int)xs[i], ys[i]);
        fclose(rf);
    }

    gputronic_stop(h);
    gputronic_destroy(h);
    return pass ? 0 : 2;
}

/* Step: open-loop working SMs N → N/2 → N, watch rate/Z transient */
static int mode_step(void) {
    GPUTronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_gold_step.csv";

    GPUTronicHandle* h = gputronic_create(&cfg);
    if (!h) return 1;
    if (gputronic_start(h) != 0) {
        gputronic_destroy(h);
        return 1;
    }

    int n_full = h->cfg.num_sm;
    int n_step = n_full / 2 < 4 ? 4 : n_full / 2;
    printf("\n=== STEP RESPONSE (open-loop working SMs) ===\n");
    gputronic_set_open_loop_sms(h, 0);
    for (int s = 0; s < 5 && !g_signal_stop; s++) {
        usleep(1000000);
        gputronic_print_status(h, s + 1);
    }
    printf("--- step active_sms %d -> %d ---\n", n_full, n_step);
    gputronic_set_open_loop_sms(h, n_step);
    for (int s = 0; s < 5 && !g_signal_stop; s++) {
        usleep(1000000);
        gputronic_print_status(h, s + 6);
    }
    printf("--- step active_sms %d -> %d ---\n", n_step, n_full);
    gputronic_set_open_loop_sms(h, 0);
    for (int s = 0; s < 5 && !g_signal_stop; s++) {
        usleep(1000000);
        gputronic_print_status(h, s + 11);
    }

    gputronic_stop(h);
    gputronic_destroy(h);
    return 0;
}

/* Closed-loop short validation after dyno */
static int mode_closedloop_check(int seconds) {
    GPUTronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_gold_closedloop.csv";

    GPUTronicHandle* h = gputronic_create(&cfg);
    if (!h) return 1;
    if (gputronic_start(h) != 0) {
        gputronic_destroy(h);
        return 1;
    }

    printf("\n=== CLOSED-LOOP (target Z=%.2f) ===\n", cfg.target_z);
    double z_acc = 0.0;
    int z_n = 0;
    double fr_rate = 0.0, cl_rate_acc = 0.0;
    int cl_n = 0;
    int fr_done = 0;
    for (int s = 0; s < seconds && !g_signal_stop; s++) {
        usleep(1000000);
        gputronic_print_status(h, s + 1);
        /* Free-run useful-work baseline while cal still holds full width. */
        if (!fr_done && s == 0) {
            unsigned long long p0 = h->h_ctrl->total_work_pulses;
            double t0 = now_us();
            usleep(400000);
            unsigned long long p1 = h->h_ctrl->total_work_pulses;
            double dt = (now_us() - t0) * 1e-6;
            if (dt < 1e-6) dt = 1e-6;
            fr_rate = (double)(p1 - p0) / dt;
            fr_done = 1;
            printf("[BASE] free-run window  pulse/s=%.1f\n", fr_rate);
        }
        if (s >= 4) {
            z_acc += gputronic_get_z(h);
            z_n++;
            unsigned long long p0 = h->h_ctrl->total_work_pulses;
            double t0 = now_us();
            usleep(200000);
            unsigned long long p1 = h->h_ctrl->total_work_pulses;
            double dt = (now_us() - t0) * 1e-6;
            if (dt < 1e-6) dt = 1e-6;
            cl_rate_acc += (double)(p1 - p0) / dt;
            cl_n++;
        }
    }

    float z = gputronic_get_z(h);
    float rate = gputronic_get_rate(h);
    int n_now = h->h_ctrl->active_sms;
    int pulses_ok = (h->h_ctrl->total_work_pulses > 1000ULL);
    int z_ok = (z >= cfg.z_floor && z <= cfg.z_ceiling);
    int rate_ok = (rate > 100.0f);
    float z_mean = (z_n > 0) ? (float)(z_acc / z_n) : z;
    double cl_rate = (cl_n > 0) ? (cl_rate_acc / cl_n) : (double)rate;
    float z_err = fabsf(z_mean - cfg.target_z);
    int track_ok = (z_err < 0.25f);
    int useful_drop = (fr_rate > 1000.0) && (cl_rate < fr_rate * 0.90);
    int engaged = (n_now >= 4 && n_now < h->cfg.num_sm) || useful_drop;
    int model_ok = useful_drop && engaged && track_ok;

    printf("\n[CHECK] pulses=%llu z=%.3f z_mean=%.3f rate=%.1f n=%d\n",
           (unsigned long long)h->h_ctrl->total_work_pulses, z, z_mean, rate, n_now);
    printf("[CHECK] free-run %.1f pulse/s | closed %.1f pulse/s\n", fr_rate, cl_rate);
    printf("[CHECK] tach=%s z_sane=%s rate_sane=%s track=%s useful_drop=%s (err=%.3f)\n",
           pulses_ok ? "OK" : "FAIL", z_ok ? "OK" : "FAIL", rate_ok ? "OK" : "FAIL",
           track_ok ? "OK" : "WEAK", useful_drop ? "YES" : "NO", z_err);

    FILE* rf = fopen("results/gputronic_gold_closedloop_report.txt", "w");
    if (rf) {
        fprintf(rf, "pulses=%llu z=%.4f z_mean=%.4f rate=%.1f active_sms=%d tach=%d track=%d "
                    "fr_rate=%.1f cl_rate=%.1f useful_drop=%d PASS=%d\n",
                (unsigned long long)h->h_ctrl->total_work_pulses, z, z_mean, rate, n_now,
                pulses_ok, track_ok, fr_rate, cl_rate, useful_drop, model_ok && pulses_ok);
        fclose(rf);
    }

    gputronic_stop(h);
    gputronic_destroy(h);
    return (pulses_ok && z_ok && rate_ok && model_ok) ? 0 : 2;
}

static void usage(const char* argv0) {
    printf("GPUTronic Gold %s\n\n", GPUTRONIC_VERSION_STRING);
    printf("Usage:\n");
    printf("  %s                 # 30s closed-loop free-run (Gold defaults)\n", argv0);
    printf("  %s run [sec] [Z] [Kp] [Ki]\n", argv0);
    printf("  %s dyno            # open-loop working-SM sweep + R²\n", argv0);
    printf("  %s step            # open-loop SM-count step response\n", argv0);
    printf("  %s check           # dyno + 20s closed-loop gate\n", argv0);
    printf("  %s all             # dyno + step + closed-loop\n", argv0);
}

#ifndef GPUTRONIC_NO_MAIN
int main(int argc, char** argv) {
    const char* mode = (argc >= 2) ? argv[1] : "run";

    if (strcmp(mode, "-h") == 0 || strcmp(mode, "--help") == 0 || strcmp(mode, "help") == 0) {
        usage(argv[0]);
        return 0;
    }

    if (strcmp(mode, "dyno") == 0) return mode_dyno();
    if (strcmp(mode, "step") == 0) return mode_step();
    if (strcmp(mode, "check") == 0) {
        int rd = mode_dyno();
        int rc = mode_closedloop_check(20);
        printf("\n=== GOLD GATE: dyno=%s closedloop=%s ===\n",
               rd == 0 ? "PASS" : "FAIL", rc == 0 ? "PASS" : "FAIL");
        return (rd == 0 && rc == 0) ? 0 : 2;
    }
    if (strcmp(mode, "all") == 0) {
        int rd = mode_dyno();
        int rs = mode_step();
        int rc = mode_closedloop_check(20);
        printf("\n=== GOLD ALL: dyno=%d step=%d closed=%d ===\n", rd, rs, rc);
        return (rd == 0 && rs == 0 && rc == 0) ? 0 : 2;
    }

    /* run [sec] [Z] [Kp] [Ki]  OR bare defaults when mode omitted */
    int sec = 30;
    float z = GPUTRONIC_GOLD_TARGET_Z;
    float kp = GPUTRONIC_GOLD_KP;
    float ki = GPUTRONIC_GOLD_KI;

    if (strcmp(mode, "run") == 0) {
        if (argc >= 3) sec = atoi(argv[2]);
        if (argc >= 4) z = (float)atof(argv[3]);
        if (argc >= 5) kp = (float)atof(argv[4]);
        if (argc >= 6) ki = (float)atof(argv[5]);
    } else {
        /* first arg might be seconds */
        int maybe = atoi(mode);
        if (maybe > 0) {
            sec = maybe;
            if (argc >= 3) z = (float)atof(argv[2]);
            if (argc >= 4) kp = (float)atof(argv[3]);
            if (argc >= 5) ki = (float)atof(argv[4]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (sec <= 0) sec = 30;
    return mode_run(sec, z, kp, ki);
}
#endif /* GPUTRONIC_NO_MAIN */
