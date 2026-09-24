/* =============================================================================
 * GPUTronic Gold 1.0.0
 * -----------------------------------------------------------------------------
 * Read with: docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md  (GT-FF-GOLD-1.0)
 *
 * Walk order (do not start at main):
 *   F-02 gold_work / gold_persistent_kernel
 *   F-01 gputronic_create
 *   F-12 gputronic_start / stop
 *   F-05…F-10 control_thread
 *   F-09 gold_apply_auto_scale
 *   F-08 kalman_update
 *   F-13 mode_dyno / mode_closedloop_check
 *
 * Contracts that define Gold (lost in late v26, restored here):
 *   1. F-04  Aggregate Q via atomicAdd on total_work_pulses
 *   2. F-01  Periodic __threadfence_system; host __sync_synchronize after sleep
 *   3. F-06  rate_ref from a sustained free-run window (not peak EMA)
 *   4. F-07  Z = rate_ref / rate; plant gain sleep↑ ⇒ Z↑
 *   5. F-10  One-sided PI: sleep only when Z < target
 *   6. F-08  2-state Kalman with P_pred = F P F^T + Q
 *   7. F-09  sleep_scale from measured loop period (nanosleep cliff, not 8 µs)
 *   8. F-02  Batched device work between mapped atomics
 *
 * Build: make gold
 * Run:   ./build/gputronic_gold check | dyno | step | run [sec] [Z] [Kp] [Ki]
 * ============================================================================= */

#include "gputronic.h"

#include <cuda_runtime.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* -------------------- F-02 device plant + F-04 tach + F-11 actuator -------------------- */

/* Mapped atomics set the loop period (~300 µs on this 5080). The fmaf batch
 * is real ALU so DCE cannot eat the loop; it is not a FLOP claim (see F-14). */
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

/*
 * F-02 persistent plant.
 *   grid = num_sm (one block / SM)
 *   warp leaders (lane==0) pulse F-04 aggregate + per-SM diagnostic
 *   F-11: __nanosleep(throttle_sleep_ns) after the work batch
 * Period is atomic-dominated. Do not "tune Kp" to hide that.
 */
__global__ void gold_persistent_kernel(unsigned long long* sm_counters,
                                       GPUTronicControl* ctrl,
                                       int num_sm) {
    const int sm = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
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
            /* Fence every 32 pulses — enough for host visibility, cheap enough
               that it does not drown the sleep actuator in system traffic. */
            if ((local_pulses & 31u) == 0u) __threadfence_system();
        }

        /* Volatile-qualified field: re-read command every chunk */
        int sleep_ns = ctrl->throttle_sleep_ns;
        int max_s = ctrl->max_sleep_ns;
        if (sleep_ns < 0) sleep_ns = 0;
        if (max_s > 0 && sleep_ns > max_s) sleep_ns = max_s;
        if (sleep_ns > 0) __nanosleep((unsigned int)sleep_ns);
    }
}

extern "C" void gold_launch_kernel(unsigned long long* sm_counters,
                                   GPUTronicControl* d_ctrl,
                                   int num_sm,
                                   int threads_per_block) {
    gold_persistent_kernel<<<num_sm, threads_per_block>>>(sm_counters, d_ctrl, num_sm);
}

/* -------------------- handle / host -------------------- */

struct GPUTronicHandle {
    GPUTronicConfig cfg;
    GPUTronicControl* h_ctrl;
    GPUTronicControl* d_ctrl;
    unsigned long long* d_sm;
    unsigned long long* h_sm; /* optional staging for host sum */
    pthread_t thr;
    int running;
    int started;
    int open_loop_sleep; /* >=0 forces open-loop sleep; -1 = closed loop */
    int zero_copy_ok;
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
    double t0_us;
    FILE* csv;
    long long csv_rows;
    /* stats */
    double sum_z, sum_z2;
    long long n_z;
    int min_sleep, max_sleep_seen;
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
    cfg->sleep_scale = GPUTRONIC_GOLD_SLEEP_SCALE;
    cfg->integral_clamp = GPUTRONIC_GOLD_INTEGRAL_CLAMP;
    cfg->deadband = GPUTRONIC_GOLD_DEADBAND;
    cfg->z_ceiling = GPUTRONIC_GOLD_Z_CEILING;
    cfg->z_floor = GPUTRONIC_GOLD_Z_FLOOR;
    cfg->one_sided = 1;
    cfg->enable_kalman = 1;
    cfg->launch_selftest_kernel = 1;
    cfg->auto_sleep_scale = 1;
    cfg->csv_path = "results/gputronic_gold.csv";
}

static void kalman_update(GPUTronicHandle* h, float z_meas, float dt) {
    /* F-08: constant-velocity model on [Z, dZ/dt].
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

static void gold_apply_auto_scale(GPUTronicHandle* h) {
    /* F-09: map e_des=0.30 onto ~1.25 loop periods so PI reaches the cliff. */
    if (!h->cfg.auto_sleep_scale || h->rate_ref < 1000.0f || h->cfg.num_sm <= 0)
        return;
    /* Pulse sources per loop: self-test has one warp-leader pulse per warp.
     * Cooperative plants typically pulse once per block ≈ once per SM. */
    int sources = h->cfg.num_sm;
    if (h->cfg.launch_selftest_kernel && h->cfg.threads_per_block >= 32)
        sources = h->cfg.num_sm * (h->cfg.threads_per_block / 32);
    if (sources < 1) sources = 1;
    double period_ns = 1e9 * (double)sources / (double)h->rate_ref;
    float kp = (h->cfg.kp > 1e-6f) ? h->cfg.kp : 0.55f;
    const float e_des = 0.30f;
    float scale = (float)(1.25 * period_ns / ((double)kp * (double)e_des));
    if (scale < 200000.0f) scale = 200000.0f;
    if (scale > 8000000.0f) scale = 8000000.0f;
    h->cfg.sleep_scale = scale;
    printf("[CAL] auto sleep_scale=%.0f (period~%.0f ns, %d sources, e=%.2f → ~%.0f ns)\n",
           scale, period_ns, sources, e_des, (double)kp * e_des * scale);
}

static void* control_thread(void* arg) {
    /* F-12 ECU thread. Per accepted tick: F-05 rate → F-06/F-07 Z → F-08 → F-10 → cable. */
    GPUTronicHandle* h = (GPUTronicHandle*)arg;
    const float dt_target = h->cfg.control_dt_us * 1e-6f;
    double last = now_us();
    double hz_ema = 1.0 / dt_target;
    int tick = 0;

    h->P[0][0] = 1.0f;
    h->P[0][1] = 0.0f;
    h->P[1][0] = 0.0f;
    h->P[1][1] = 1.0f;
    h->z_hat = 1.0f;
    h->dzdt_hat = 0.0f;
    h->integral = 0.0f;
    h->rate_ema = 0.0f;
    h->rate_ref = 0.0f;
    h->rate_win_dt = 0.0;
    h->rate_win_pulses = 0;
    h->calibrated = 0;
    /* Warmup 0.7 s, then a ~1.1 s sustained window (not peak EMA). */
    h->calibrate_until_us = now_us() + 1800000.0;
    h->cal_sustain_from_us = now_us() + 700000.0;
    h->cal_pulse_mark = 0;
    h->cal_mark_us = 0.0;
    h->cal_marked = 0;
    h->prev_pulses = h->h_ctrl->total_work_pulses;
    h->min_sleep = 999999999;
    h->max_sleep_seen = 0;

    printf("[CTRL] Gold | target=%.2f Kp=%.3f Ki=%.4f one_sided=%d kalman=%d auto_scale=%d | cal 1.8s sustain\n",
           h->cfg.target_z, h->cfg.kp, h->cfg.ki, h->cfg.one_sided, h->cfg.enable_kalman,
           h->cfg.auto_sleep_scale);

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

        unsigned long long pulses = h->h_ctrl->total_work_pulses;
        unsigned long long dpulse = pulses - h->prev_pulses;
        h->prev_pulses = pulses;

        float inst_rate = (dt > 0.0f) ? ((float)dpulse / dt) : 0.0f;
        /* Form rate over ~8 ms, not a 10 µs tick. Short ticks + empty-hold
         * left EMA stuck at free-run while window rate had already dropped. */
        h->rate_win_pulses += dpulse;
        h->rate_win_dt += (double)dt;
        if (h->rate_win_dt >= 0.008) {
            inst_rate = (float)((double)h->rate_win_pulses / h->rate_win_dt);
            h->rate_win_pulses = 0;
            h->rate_win_dt = 0.0;
            if (h->rate_ema <= 1.0f)
                h->rate_ema = inst_rate;
            else
                h->rate_ema = 0.70f * h->rate_ema + 0.30f * inst_rate;
        } else if (h->rate_ema <= 1.0f && inst_rate > 1.0f) {
            h->rate_ema = inst_rate;
        }

        /* ---- Z measurement ----
         * Calibrated impedance proxy: Z = rate_ref / rate
         * Free-run (max Q) → Z ≈ 1.0; sleep raises Z as rate falls.
         * rate_ref is a sustained window mean, not a peak hold.
         */
        float z_raw;
        if (!h->calibrated) {
            if (h->open_loop_sleep < 0) {
                h->h_ctrl->throttle_sleep_ns = h->cfg.base_sleep_ns;
            } else {
                h->h_ctrl->throttle_sleep_ns = h->open_loop_sleep;
            }
            if (!h->cal_marked && t >= h->cal_sustain_from_us) {
                h->cal_pulse_mark = pulses;
                h->cal_mark_us = t;
                h->cal_marked = 1;
            }
            if (t >= h->calibrate_until_us && h->cal_marked) {
                double dt_cal = (t - h->cal_mark_us) * 1e-6;
                unsigned long long dp_cal = pulses - h->cal_pulse_mark;
                if (dt_cal > 0.2 && dp_cal > 1000ULL) {
                    h->rate_ref = (float)((double)dp_cal / dt_cal);
                    h->calibrated = 1;
                    printf("[CAL] rate_ref=%.1f pulses/s (sustained %.2fs window, %llu pulses)\n",
                           h->rate_ref, dt_cal, (unsigned long long)dp_cal);
                    gold_apply_auto_scale(h);
                } else if (h->rate_ema > 1000.0f) {
                    /* Fallback if the window was empty (should be rare). */
                    h->rate_ref = h->rate_ema;
                    h->calibrated = 1;
                    printf("[CAL] rate_ref=%.1f pulses/s (EMA fallback)\n", h->rate_ref);
                    gold_apply_auto_scale(h);
                }
            }
            z_raw = 1.0f; /* nominal during cal */
        } else {
            z_raw = h->rate_ref / (h->rate_ema + 1.0f);
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

        int new_sleep = h->cfg.base_sleep_ns;
        float error = 0.0f;

        if (h->open_loop_sleep >= 0) {
            new_sleep = h->open_loop_sleep;
            h->integral = 0.0f;
        } else if (!h->calibrated) {
            new_sleep = h->cfg.base_sleep_ns;
            h->integral = 0.0f;
        } else {
            /*
             * Tracking PI on plant with positive gain (sleep ↑ ⇒ Z ↑):
             *   e = target - z
             *   sleep = base + scale * (Kp*e + Ki*∫e)
             * When z < target, e>0 → add sleep → raise Z toward target.
             *
             * One-sided Gold policy: if z >= target, stay at base sleep
             * (max Q). Do not fight high impedance by undersleeping below base.
             */
            int engage = 1;
            if (h->cfg.one_sided && z >= h->cfg.target_z) engage = 0;

            if (engage) {
                error = h->cfg.target_z - z;
                h->integral += error * dt;
                if (h->integral > h->cfg.integral_clamp) h->integral = h->cfg.integral_clamp;
                if (h->integral < -h->cfg.integral_clamp) h->integral = -h->cfg.integral_clamp;

                float u = h->cfg.kp * error + h->cfg.ki * h->integral;
                if (fabsf(error) < h->cfg.deadband) u = 0.0f;

                new_sleep = (int)(h->cfg.base_sleep_ns + u * h->cfg.sleep_scale);
            } else {
                h->integral = 0.0f;
                new_sleep = h->cfg.base_sleep_ns;
            }
        }

        if (new_sleep < 0) new_sleep = 0;
        if (new_sleep > h->cfg.max_sleep_ns) new_sleep = h->cfg.max_sleep_ns;

        h->h_ctrl->throttle_sleep_ns = new_sleep;
        /* Ensure host store is visible to mapped device mappings promptly */
        __sync_synchronize();

        h->h_ctrl->target_z = h->cfg.target_z;
        h->h_ctrl->z_raw = z_raw;
        h->h_ctrl->z_hat = z;
        h->h_ctrl->dzdt_hat = h->dzdt_hat;
        h->h_ctrl->rate_ema = h->rate_ema;
        h->h_ctrl->pm_error = error;
        h->h_ctrl->integral = h->integral;
        if (new_sleep > h->cfg.base_sleep_ns) h->h_ctrl->proactive_corrections++;

        double inst_hz = 1.0 / dt;
        hz_ema = 0.95 * hz_ema + 0.05 * inst_hz;
        h->h_ctrl->control_hz_ema = (int)(hz_ema + 0.5);

        if (new_sleep < h->min_sleep) h->min_sleep = new_sleep;
        if (new_sleep > h->max_sleep_seen) h->max_sleep_seen = new_sleep;
        h->sum_z += z;
        h->sum_z2 += (double)z * (double)z;
        h->n_z++;

        if (h->csv) {
            double t_s = (t - h->t0_us) * 1e-6;
            fprintf(h->csv, "%.6f,%.5f,%.5f,%.5f,%.1f,%d,%.5f,%.5f,%d,%.1f\n",
                    t_s, z_raw, z, h->dzdt_hat, h->rate_ema, new_sleep,
                    error, h->integral, h->h_ctrl->control_hz_ema, h->rate_ref);
            h->csv_rows++;
            if ((tick & 1023) == 0) fflush(h->csv);
        }
        tick++;
    }
    return NULL;
}

GPUTronicHandle* gputronic_create(const GPUTronicConfig* cfg_in) {
    /* F-01: map the cable. Query SM count. No PCIe fallback. */
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

    h->zero_copy_ok = ((uintptr_t)h->h_ctrl == (uintptr_t)h->d_ctrl) || (h->d_ctrl != NULL);
    /* On many platforms mapped host ptr != device ptr value but still zero-copy.
       Record pointer equality separately; presence of device pointer is required. */
    h->zero_copy_ok = (h->d_ctrl != NULL);

    memset(h->h_ctrl, 0, sizeof(GPUTronicControl));
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
    h->h_sm = (unsigned long long*)calloc((size_t)h->cfg.num_sm, sizeof(unsigned long long));

    h->open_loop_sleep = -1;
    h->running = 0;
    h->started = 0;

    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPUTronic Gold %s]\n", GPUTRONIC_VERSION_STRING);
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPU] %s | SMs=%d (using %d) | thr/block=%d\n",
           prop.name, prop.multiProcessorCount, h->cfg.num_sm, h->cfg.threads_per_block);
    printf("[ZC ] host_ctrl=%p device_ctrl=%p %s\n",
           (void*)h->h_ctrl, (void*)h->d_ctrl,
           h->d_ctrl ? "OK (mapped)" : "FAIL");
    printf("[LAW] target_z=%.2f Kp=%.3f Ki=%.4f scale=%.0f one_sided=%d\n",
           h->cfg.target_z, h->cfg.kp, h->cfg.ki, h->cfg.sleep_scale, h->cfg.one_sided);
    printf("[LOOP] dt=%d us (%.0f kHz) base_sleep=%d max_sleep=%d\n\n",
           h->cfg.control_dt_us, 1000.0 / h->cfg.control_dt_us,
           h->cfg.base_sleep_ns, h->cfg.max_sleep_ns);

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
        fprintf(h->csv, "time_s,z_raw,z_hat,dzdt,rate_ema,sleep_ns,error,integral,ctrl_hz,rate_ref\n");
        printf("[CSV] %s\n", path);
    } else {
        fprintf(stderr, "[WARN] could not open CSV %s\n", path);
    }

    h->t0_us = now_us();
    h->running = 1;

    if (h->cfg.launch_selftest_kernel) {
        gold_launch_kernel(h->d_sm, h->d_ctrl, h->cfg.num_sm, h->cfg.threads_per_block);
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
        printf("\n[STATS] samples=%lld mean_Z=%.4f std_Z=%.4f sleep_range=%d..%d ns rows_csv=%lld\n",
               (long long)h->n_z, mean, sqrt(var), h->min_sleep, h->max_sleep_seen,
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
    free(h->h_sm);
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

void gputronic_set_open_loop_sleep(GPUTronicHandle* h, int sleep_ns) {
    if (!h) return;
    h->open_loop_sleep = sleep_ns;
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

int gputronic_get_sleep_ns(const GPUTronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->throttle_sleep_ns : 0;
}

struct GPUTronicControl* gputronic_get_control(GPUTronicHandle* h) {
    return h ? h->h_ctrl : NULL;
}

void gputronic_pulse(GPUTronicHandle* h, unsigned long long units) {
    if (!h || !h->h_ctrl || units == 0) return;
    /* Host-side cooperative pulse (single-writer assumed for host path). */
    h->h_ctrl->total_work_pulses += units;
}

int gputronic_zero_copy_ok(const GPUTronicHandle* h) {
    return h ? h->zero_copy_ok : 0;
}

void gputronic_print_status(const GPUTronicHandle* h, int elapsed_s) {
    if (!h || !h->h_ctrl) return;
    printf("[TEL] t=%4ds | Z_raw=%.3f Z_hat=%.3f dZ=%.3f | rate=%.0f | sleep=%6d | hz~%d | pulses=%llu\n",
           elapsed_s,
           h->h_ctrl->z_raw,
           h->h_ctrl->z_hat,
           h->h_ctrl->dzdt_hat,
           h->h_ctrl->rate_ema,
           h->h_ctrl->throttle_sleep_ns,
           h->h_ctrl->control_hz_ema,
           (unsigned long long)h->h_ctrl->total_work_pulses);
}

/* -------------------- CLI modes -------------------- */

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

/* Dyno F-13: open-loop sleep sweep. Rate = wall Δpulses/Δt, not rate_ema alone. */
static int mode_dyno(void) {
    GPUTronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_gold_dyno.csv";
    /* Raise max sleep so dyno can explore deep throttle */
    cfg.max_sleep_ns = 2000000;

    GPUTronicHandle* h = gputronic_create(&cfg);
    if (!h) return 1;
    if (gputronic_start(h) != 0) {
        gputronic_destroy(h);
        return 1;
    }

    /* Free-run settle + allow calibration window */
    gputronic_set_open_loop_sleep(h, 0);
    usleep(2500000);

    const int sleeps[] = {0, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000};
    const int n = (int)(sizeof(sleeps) / sizeof(sleeps[0]));
    double xs[16], ys[16];

    printf("\n=== DYNO SWEEP (open-loop sleep → rate) ===\n");
    printf("%10s %14s %10s %10s\n", "sleep_ns", "rate_ema", "Z_hat", "pulses_d");
    printf("--------------------------------------------------------\n");

    FILE* df = fopen("results/gputronic_gold_dyno_summary.csv", "w");
    if (df) fprintf(df, "sleep_ns,rate_ema,z_hat,delta_pulses\n");

    for (int i = 0; i < n && !g_signal_stop; i++) {
        gputronic_set_open_loop_sleep(h, sleeps[i]);
        usleep(1500000); /* settle (EMA + plant) */
        unsigned long long p0 = h->h_ctrl->total_work_pulses;
        double t0 = now_us();
        double z_acc = 0.0;
        const int samples = 40;
        for (int k = 0; k < samples; k++) {
            usleep(50000);
            z_acc += h->h_ctrl->z_hat;
        }
        unsigned long long p1 = h->h_ctrl->total_work_pulses;
        double t1 = now_us();
        double dt_s = (t1 - t0) * 1e-6;
        if (dt_s < 1e-6) dt_s = 1e-6;
        double rate = (double)(p1 - p0) / dt_s; /* true pulse rate over window */
        double z = z_acc / samples;
        unsigned long long dp = p1 - p0;
        xs[i] = (double)sleeps[i];
        ys[i] = rate;
        printf("%10d %14.1f %10.4f %10llu\n", sleeps[i], rate, z, (unsigned long long)dp);
        if (df) fprintf(df, "%d,%.3f,%.5f,%llu\n", sleeps[i], rate, z, (unsigned long long)dp);
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

    /* Plant OK if deep sleep cuts rate by >= 25% vs free-run */
    int plant_ok = (ys[0] > 1000.0) && (ys[n - 1] < ys[0] * 0.75);
    /* Also require mostly non-increasing rate as sleep grows */
    int decreasing = 1;
    for (int i = 1; i < n; i++) {
        if (ys[i] > ys[i - 1] * 1.15) decreasing = 0;
    }

    printf("--------------------------------------------------------\n");
    printf("fit: rate ≈ %.1f + (%.6f)*sleep_ns\n", a, b);
    printf("R² = %.4f  | plant_ok=%s | mostly_decreasing=%s\n",
           r2, plant_ok ? "YES" : "NO", decreasing ? "YES" : "NO");
    printf("PASS: plant_ok AND (R²>=0.85 OR mostly_decreasing)\n");
    int pass = plant_ok && (r2 >= 0.85 || decreasing);
    printf("RESULT: %s\n\n", pass ? "PASS" : "CHECK");

    FILE* rf = fopen("results/gputronic_gold_dyno_report.txt", "w");
    if (rf) {
        fprintf(rf, "GPUTronic Gold Dyno Report\n");
        fprintf(rf, "R2=%.4f slope=%.6f intercept=%.1f plant_ok=%d decreasing=%d PASS=%d\n",
                r2, b, a, plant_ok, decreasing, pass);
        for (int i = 0; i < n; i++)
            fprintf(rf, "sleep=%d rate=%.1f\n", sleeps[i], ys[i]);
        fclose(rf);
    }

    gputronic_stop(h);
    gputronic_destroy(h);
    return pass ? 0 : 2;
}

/* Step: open-loop sleep 0 → 20us → 0, watch rate/Z transient */
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

    printf("\n=== STEP RESPONSE (open-loop sleep) ===\n");
    gputronic_set_open_loop_sleep(h, 0);
    for (int s = 0; s < 5 && !g_signal_stop; s++) {
        usleep(1000000);
        gputronic_print_status(h, s + 1);
    }
    printf("--- step sleep -> 20000 ns ---\n");
    gputronic_set_open_loop_sleep(h, 20000);
    for (int s = 0; s < 5 && !g_signal_stop; s++) {
        usleep(1000000);
        gputronic_print_status(h, s + 6);
    }
    printf("--- step sleep -> 0 ns ---\n");
    gputronic_set_open_loop_sleep(h, 0);
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
    cfg.max_sleep_ns = 500000;

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
        /* Free-run useful-work baseline while cal still holds base sleep. */
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
    int sleep_ns = gputronic_get_sleep_ns(h);
    int pulses_ok = (h->h_ctrl->total_work_pulses > 1000ULL);
    int z_ok = (z >= cfg.z_floor && z <= cfg.z_ceiling);
    int rate_ok = (rate > 100.0f);
    float z_mean = (z_n > 0) ? (float)(z_acc / z_n) : z;
    double cl_rate = (cl_n > 0) ? (cl_rate_acc / cl_n) : (double)rate;
    float z_err = fabsf(z_mean - cfg.target_z);
    int track_ok = (z_err < 0.40f) || (z_mean > 1.05f && sleep_ns > cfg.base_sleep_ns);
    int useful_drop = (fr_rate > 1000.0) && (cl_rate < fr_rate * 0.90);
    int engaged = (sleep_ns > 50000) || useful_drop;
    int model_ok = useful_drop && engaged;

    printf("\n[CHECK] pulses=%llu z=%.3f z_mean=%.3f rate=%.1f sleep=%d\n",
           (unsigned long long)h->h_ctrl->total_work_pulses, z, z_mean, rate, sleep_ns);
    printf("[CHECK] free-run %.1f pulse/s | closed %.1f pulse/s\n", fr_rate, cl_rate);
    printf("[CHECK] tach=%s z_sane=%s rate_sane=%s track=%s useful_drop=%s (err=%.3f)\n",
           pulses_ok ? "OK" : "FAIL", z_ok ? "OK" : "FAIL", rate_ok ? "OK" : "FAIL",
           track_ok ? "OK" : "WEAK", useful_drop ? "YES" : "NO", z_err);

    FILE* rf = fopen("results/gputronic_gold_closedloop_report.txt", "w");
    if (rf) {
        fprintf(rf, "pulses=%llu z=%.4f z_mean=%.4f rate=%.1f sleep=%d tach=%d track=%d "
                    "fr_rate=%.1f cl_rate=%.1f useful_drop=%d PASS=%d\n",
                (unsigned long long)h->h_ctrl->total_work_pulses, z, z_mean, rate, sleep_ns,
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
    printf("  %s dyno            # open-loop sleep sweep + R²\n", argv0);
    printf("  %s step            # open-loop sleep step response\n", argv0);
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
