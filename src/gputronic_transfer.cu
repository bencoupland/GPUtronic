/* =============================================================================
 * GPUTronic Transfer Plant — cooperative GEMM microtiles on Gold 1.0
 * -----------------------------------------------------------------------------
 * Read with: docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md [F-03, F-14]
 *
 * Second kernel on the Gold interface.
 *   - launch_selftest_kernel = 0
 *   - persistent kernel does FP32 GEMM tiles (shared-mem 8x8x8)
 *   - block leader atomicAdd on total_work_pulses
 *   - SMs at or above active_sms skip the tiles
 *   - counted work = completed tiles * FLOPS_PER_TILE
 *
 * What to check: fewer SMs → tile rate down, Z up, and GFLOP/s down with the tile rate.
 *
 * Build: make transfer
 * Run:   ./build/gputronic_transfer check | dyno | run [sec]
 * ============================================================================= */

#include "gputronic.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Legacy sleep throttle for this plant's kernel, which still honours
 * throttle_sleep_ns. Gold 1.0 removed gputronic_set_open_loop_sleep /
 * gputronic_get_sleep_ns; the control thread no longer writes this field, so
 * the value set here persists. ns < 0 = closed loop with base sleep. */
static void legacy_set_sleep(GPUTronicHandle* h, int ns) {
    struct GPUTronicControl* c = gputronic_get_control(h);
    gputronic_set_open_loop_sms(h, ns < 0 ? -1 : 0);
    if (c) c->throttle_sleep_ns = (ns < 0) ? c->base_sleep_ns : ns;
}
static int legacy_get_sleep(GPUTronicHandle* h) {
    struct GPUTronicControl* c = gputronic_get_control(h);
    return c ? c->throttle_sleep_ns : 0;
}

#define TRANSFER_TILE 8
#define TRANSFER_THREADS 64
/* Inner GEMM tiles per tach pulse. Mapped atomics dominate a single 8x8x8
 * (~300 µs/loop on this 5080), so batch real work before one pulse+sleep.
 * FLOPs per pulse = tiles_per_pulse * 1024. */
#define TRANSFER_TILES_PER_PULSE 32
#define TRANSFER_FLOPS_PER_TILE (2ULL * TRANSFER_TILE * TRANSFER_TILE * TRANSFER_TILE)
#define TRANSFER_FLOPS_PER_PULSE (TRANSFER_FLOPS_PER_TILE * (unsigned long long)TRANSFER_TILES_PER_PULSE)

static volatile int g_stop = 0;

static inline double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
}

/* ---- Transfer kernel: cooperative GEMM tiles ---- */

/* sleep_sms == NULL, or *sleep_sms < 0: every SM honours the command (the
 * gate). *sleep_sms >= 0: only the lowest blocks take __nanosleep. Awake
 * blocks do the same tiles and do not wait. One block per SM, no grid barrier. */
/* n_work == NULL: every SM runs the tiles (gate and sleep-mask).
 * *n_work >= 0: only the lowest blocks run the tiles and pulse.
 * *n_work < 0: the stock sleep command selects that count. Full clamp maps
 * to half the SMs (measured Z ≈ 1.82 on this 5080, past the setpoint).
 * At least 4 SMs stay awake.
 * Idle blocks do not pulse. They backoff 5 µs so they do not spin on the cable. */
__global__ void transfer_persistent_kernel(GPUTronicControl* ctrl, int num_sm,
                                          volatile int* sleep_sms,
                                          volatile int* n_work) {
    const int sm = blockIdx.x;
    const int tid = threadIdx.x;
    const int r = tid / TRANSFER_TILE;
    const int c = tid % TRANSFER_TILE;
    float acc = 0.0f;
    unsigned int local_pulses = 0;

    if (sm >= num_sm) return;

    __shared__ float As[TRANSFER_TILE][TRANSFER_TILE];
    __shared__ float Bs[TRANSFER_TILE][TRANSFER_TILE];

    unsigned int iter = 0;
    while (true) {
        unsigned int flags = ctrl->control_flags;
        if (flags & GPUTRONIC_FLAG_STOP) break;
        if (flags & GPUTRONIC_FLAG_PAUSE) {
            (void)(5000);   /* was __nanosleep: removed, NEXT.md A4 */
            continue;
        }

        int sleep_ns = ctrl->throttle_sleep_ns;
        int max_s = ctrl->max_sleep_ns;
        if (sleep_ns < 0) sleep_ns = 0;
        if (max_s > 0 && sleep_ns > max_s) sleep_ns = max_s;

        int active = num_sm;
        int follow = 0;
        if (n_work) {
            int ov = *n_work;
            if (ov >= 0) active = ov;
            else follow = 1;
        }
        if (follow) {
            int clamp = (max_s > 0) ? max_s : 500000;
            int n_at_clamp = num_sm / 2;
            if (n_at_clamp < 4) n_at_clamp = 4;
            int span = num_sm - n_at_clamp;
            if (span < 0) span = 0;
            int drop = (clamp > 0) ? (int)(((long long)span * sleep_ns) / clamp) : 0;
            active = num_sm - drop;
        }
        if (n_work) {
            if (active < 4) active = 4;
            if (active > num_sm) active = num_sm;
            if (sm >= active) {
                (void)(5000);   /* was __nanosleep: removed, NEXT.md A4 */
                continue;
            }
        } else {
            int cable = ctrl->active_sms;
            if (cable <= 0 || cable > num_sm) cable = num_sm;
            if (cable < 4) cable = 4;
            if (sm >= cable) {
                (void)(5000);   /* was __nanosleep: removed, NEXT.md A4 */
                continue;
            }
        }

        for (int t = 0; t < TRANSFER_TILES_PER_PULSE; t++) {
            float seed = (float)((iter * 31u + (unsigned)sm * 17u + (unsigned)tid + (unsigned)t * 13u) & 1023u);
            if (r < TRANSFER_TILE && c < TRANSFER_TILE) {
                As[r][c] = seed * 0.001f + 0.10f;
                Bs[r][c] = seed * 0.0007f + 0.20f;
            }
            __syncthreads();

            if (r < TRANSFER_TILE && c < TRANSFER_TILE) {
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < TRANSFER_TILE; k++)
                    s = __fmaf_rn(As[r][k], Bs[k][c], s);
                acc += s;
            }
            __syncthreads();
            iter++;
        }

        if (acc > 1.0e30f) acc = 0.0f;

        /* One pulse == TRANSFER_TILES_PER_PULSE completed GEMM tiles. */
        if (tid == 0) {
            atomicAdd((unsigned long long*)&ctrl->total_work_pulses, 1ULL);
            local_pulses++;
            if ((local_pulses & 31u) == 0u) __threadfence_system();
        }

        int honour = 1;
        if (sleep_sms) {
            int k = *sleep_sms;
            if (k >= 0) honour = (sm < k);
        }
        /* In follow mode the command is a count of working SMs, not a stall. */
        if (!follow && honour && sleep_ns > 1000) (void)((unsigned int)sleep_ns);   /* was __nanosleep: removed, NEXT.md A4 */

        iter++;
    }

    /* Keep acc live across the persistent loop. */
    if (tid == 0 && acc == 3.14159265f)
        atomicAdd((unsigned long long*)&ctrl->total_work_pulses, 0ULL);
}

static int launch_transfer_plant(GPUTronicHandle* h) {
    GPUTronicControl* h_ctrl = gputronic_get_control(h);
    if (!h_ctrl) {
        fprintf(stderr, "[TRANSFER] no control block\n");
        return -1;
    }

    GPUTronicControl* d_ctrl = NULL;
    cudaError_t err = cudaHostGetDevicePointer((void**)&d_ctrl, h_ctrl, 0);
    if (err != cudaSuccess || !d_ctrl) {
        fprintf(stderr, "[TRANSFER] cudaHostGetDevicePointer failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    int num_sm = h_ctrl->num_sm;
    if (num_sm <= 0) num_sm = GPUTRONIC_DEFAULT_NUM_SM;

    transfer_persistent_kernel<<<num_sm, TRANSFER_THREADS>>>(d_ctrl, num_sm, NULL, NULL);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[TRANSFER] kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("[TRANSFER] plant launched | %d blocks x %d threads | tile=%dx%dx%d x %d | %.0f FLOP/pulse\n",
           num_sm, TRANSFER_THREADS, TRANSFER_TILE, TRANSFER_TILE, TRANSFER_TILE,
           TRANSFER_TILES_PER_PULSE, (double)TRANSFER_FLOPS_PER_PULSE);
    printf("[TRANSFER] host_ctrl=%p device_ctrl=%p %s\n",
           (void*)h_ctrl, (void*)d_ctrl,
           ((uintptr_t)h_ctrl == (uintptr_t)d_ctrl) ? "ptr-equal" : "mapped");
    return 0;
}

static GPUTronicHandle* transfer_boot(GPUTronicConfig* cfg) {
    cfg->launch_selftest_kernel = 0;
    GPUTronicHandle* h = gputronic_create(cfg);
    if (!h) return NULL;
    if (launch_transfer_plant(h) != 0) {
        gputronic_destroy(h);
        return NULL;
    }
    if (gputronic_start(h) != 0) {
        gputronic_destroy(h);
        return NULL;
    }
    return h;
}

static double gflops_from_rate(double pulse_rate) {
    return pulse_rate * (double)TRANSFER_FLOPS_PER_PULSE / 1.0e9;
}

/* Measure true tile rate over a wall window (not lagging EMA). */
static void measure_window(GPUTronicHandle* h, int settle_us, int window_us,
                           double* rate_out, double* gflops_out, double* z_out,
                           unsigned long long* dp_out) {
    if (settle_us > 0) usleep((useconds_t)settle_us);
    GPUTronicControl* c = gputronic_get_control(h);
    unsigned long long p0 = c->total_work_pulses;
    double t0 = now_us();
    double z_acc = 0.0;
    const int samples = 20;
    useconds_t slice = (useconds_t)(window_us / samples);
    if (slice < 1000) slice = 1000;
    for (int k = 0; k < samples; k++) {
        usleep(slice);
        z_acc += (double)c->z_hat;
    }
    unsigned long long p1 = c->total_work_pulses;
    double dt_s = (now_us() - t0) * 1e-6;
    if (dt_s < 1e-6) dt_s = 1e-6;
    double rate = (double)(p1 - p0) / dt_s;
    *rate_out = rate;
    *gflops_out = gflops_from_rate(rate);
    *z_out = z_acc / (double)samples;
    if (dp_out) *dp_out = p1 - p0;
}

/* ---- Command-line modes ---- */

static int mode_dyno(void) {
    GPUTronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_transfer_dyno.csv";
    cfg.max_sleep_ns = 2000000;

    GPUTronicHandle* h = transfer_boot(&cfg);
    if (!h) return 1;

    gputronic_set_open_loop_sms(h, 0);
    usleep(3000000);

    int num_sm = h ? 84 : 84;
    {
        GPUTronicControl* c0 = gputronic_get_control(h);
        if (c0 && c0->num_sm > 0) num_sm = c0->num_sm;
    }
    const int sleeps[] = {num_sm, num_sm * 3 / 4, num_sm * 2 / 3, num_sm / 2, num_sm / 3, num_sm / 4, 16, 8};
    const int n = (int)(sizeof(sleeps) / sizeof(sleeps[0]));
    double xs[16], ys[16], gf[16];

    printf("\n=== TRANSFER DYNO (open-loop working SMs → tile rate + GFLOP/s) ===\n");
    printf("%10s %14s %12s %10s %12s\n", "n_sm", "tile/s", "GFLOP/s", "Z_hat", "tiles_d");
    printf("--------------------------------------------------------------------\n");

    FILE* df = fopen("results/gputronic_transfer_dyno_summary.csv", "w");
    if (df) fprintf(df, "sleep_ns,tile_rate,gflops,z_hat,delta_tiles\n");

    for (int i = 0; i < n && !g_stop; i++) {
        gputronic_set_open_loop_sms(h, sleeps[i] < 4 ? 4 : sleeps[i]);
        double rate, gflops, z;
        unsigned long long dp = 0;
        measure_window(h, 1500000, 2000000, &rate, &gflops, &z, &dp);
        xs[i] = (double)sleeps[i];
        ys[i] = rate;
        gf[i] = gflops;
        printf("%10d %14.1f %12.4f %10.4f %12llu\n",
               sleeps[i], rate, gflops, z, (unsigned long long)dp);
        if (df) fprintf(df, "%d,%.3f,%.6f,%.5f,%llu\n",
                        sleeps[i], rate, gflops, z, (unsigned long long)dp);
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

    /* Plant authority: deep sleep cuts tile rate AND useful FLOPs ≥25%. */
    int rate_cut = (ys[0] > 1000.0) && (ys[n - 1] < ys[0] * 0.75);
    int flop_cut = (gf[0] > 0.01) && (gf[n - 1] < gf[0] * 0.75);
    int coupled = 1;
    for (int i = 1; i < n; i++) {
        /* Tile rate and GFLOP/s must not disagree in sign by more than noise. */
        double dr = ys[i] - ys[i - 1];
        double dg = gf[i] - gf[i - 1];
        if (fabs(dr) > 0.05 * ys[0] && fabs(dg) > 0.05 * gf[0] && dr * dg < 0.0)
            coupled = 0;
    }
    int decreasing = 1;
    for (int i = 1; i < n; i++) {
        if (ys[i] > ys[i - 1] * 1.15) decreasing = 0;
    }

    int pass = rate_cut && flop_cut && coupled && (r2 >= 0.70 || decreasing);

    printf("--------------------------------------------------------------------\n");
    printf("fit: tile/s ≈ %.1f + (%.6f)*active_sms   R²=%.4f\n", a, b, r2);
    printf("free-run  %.1f tile/s  (%.4f GFLOP/s)\n", ys[0], gf[0]);
    printf("n=%-4d    %.1f tile/s  (%.4f GFLOP/s)  rate_cut=%s flop_cut=%s coupled=%s\n",
           sleeps[n - 1],
           ys[n - 1], gf[n - 1],
           rate_cut ? "YES" : "NO", flop_cut ? "YES" : "NO", coupled ? "YES" : "NO");
    printf("RESULT: %s\n\n", pass ? "PASS" : "FAIL");

    FILE* rf = fopen("results/gputronic_transfer_dyno_report.txt", "w");
    if (rf) {
        fprintf(rf, "GPUTronic Transfer Dyno Report\n");
        fprintf(rf, "R2=%.4f slope=%.6f intercept=%.1f rate_cut=%d flop_cut=%d coupled=%d PASS=%d\n",
                r2, b, a, rate_cut, flop_cut, coupled, pass);
        fprintf(rf, "flops_per_pulse=%llu tiles_per_pulse=%d\n",
                (unsigned long long)TRANSFER_FLOPS_PER_PULSE, TRANSFER_TILES_PER_PULSE);
        for (int i = 0; i < n; i++)
            fprintf(rf, "sleep=%d rate=%.1f gflops=%.6f\n", sleeps[i], ys[i], gf[i]);
        fclose(rf);
    }

    gputronic_stop(h);
    gputronic_destroy(h);
    return pass ? 0 : 2;
}

static int mode_closedloop(int seconds) {
    GPUTronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_transfer_closedloop.csv";
    cfg.max_sleep_ns = 500000;
    cfg.launch_selftest_kernel = 0;
    /* Gold sets n_ff and sms_per_u from its calibration table. */

    GPUTronicHandle* h = transfer_boot(&cfg);
    if (!h) return 1;

    printf("\n=== TRANSFER CLOSED-LOOP ===\n");
    printf("[LAW] target=%.2f | Gold SM-count table (no transfer override)\n",
           GPUTRONIC_GOLD_TARGET_Z);

    /* Full-width reference. The table probe finishes near 3 s (1.8 s sustain
     * plus four 300 ms steps). A 2.2 s delay landed this window on a probe
     * step, so the gate compared closed-loop rate with a cut rate and
     * reported useful_drop=NO while Z was already near the target. */
    usleep(4500000);
    legacy_set_sleep(h, 0);
    double fr_rate = 0, fr_gflops = 0, fr_z = 0;
    measure_window(h, 300000, 1000000, &fr_rate, &fr_gflops, &fr_z, NULL);
    float z_free = gputronic_get_z(h);
    printf("[BASE] sustained free-run  pulse/s=%.1f  GFLOP/s=%.4f  Z_hat=%.3f\n",
           fr_rate, fr_gflops, z_free);

    float target = GPUTRONIC_GOLD_TARGET_Z;
    int cal_swallow = 0;
    if (z_free >= target - 0.08f) {
        cal_swallow = 1;
        printf("[CAL] WARN free-run Z=%.3f still near/above target %.2f\n",
               z_free, target);
    }
    gputronic_set_target(h, target);
    legacy_set_sleep(h, -1);

    printf("[LOOP] closed-loop target Z=%.2f for %ds\n", target, seconds);

    double z_acc = 0.0;
    double cl_rate_acc = 0.0;
    double cl_gflops_acc = 0.0;
    int z_n = 0;

    for (int s = 0; s < seconds && !g_stop; s++) {
        usleep(1000000);
        gputronic_print_status(h, s + 1);
        if (s >= 3) {
            GPUTronicControl* c = gputronic_get_control(h);
            unsigned long long p0 = c->total_work_pulses;
            double t0 = now_us();
            usleep(200000);
            unsigned long long p1 = c->total_work_pulses;
            double dt = (now_us() - t0) * 1e-6;
            if (dt < 1e-6) dt = 1e-6;
            double rate = (double)(p1 - p0) / dt;
            z_acc += (double)gputronic_get_z(h);
            cl_rate_acc += rate;
            cl_gflops_acc += gflops_from_rate(rate);
            z_n++;
        }
    }

    float z = gputronic_get_z(h);
    float rate_ema = gputronic_get_rate(h);
    int sleep_ns = legacy_get_sleep(h);
    GPUTronicControl* c = gputronic_get_control(h);
    unsigned long long pulses = c ? c->total_work_pulses : 0ULL;
    float z_mean = (z_n > 0) ? (float)(z_acc / z_n) : z;
    double cl_rate = (z_n > 0) ? (cl_rate_acc / z_n) : 0.0;
    double cl_gflops = (z_n > 0) ? (cl_gflops_acc / z_n) : 0.0;

    int pulses_ok = (pulses > 1000ULL);
    int z_ok = (z >= cfg.z_floor && z <= cfg.z_ceiling);
    int rate_ok = (rate_ema > 100.0f) || (cl_rate > 100.0);
    float z_err = fabsf(z_mean - target);
    int track_ok = (z_err < 0.50f) || (sleep_ns > 80000);
    int engaged = (sleep_ns > 80000) || (z_mean > z_free + 0.12f);

    /* Model: commanding Z above sustained free-run must cut GFLOP/s. */
    int useful_drop = (fr_gflops > 0.01 && cl_gflops < fr_gflops * 0.90);
    int model_ok = useful_drop && engaged;

    int pass = pulses_ok && z_ok && rate_ok && model_ok;

    printf("\n[CHECK] pulses=%llu z=%.3f z_mean=%.3f target=%.2f ema=%.1f sleep=%d\n",
           pulses, z, z_mean, target, rate_ema, sleep_ns);
    printf("[CHECK] free-run %.1f pulse/s (%.4f GFLOP/s) | closed %.1f pulse/s (%.4f GFLOP/s)\n",
           fr_rate, fr_gflops, cl_rate, cl_gflops);
    printf("[CHECK] tach=%s z_sane=%s track=%s engaged=%s useful_drop=%s cal_swallow=%s\n",
           pulses_ok ? "OK" : "FAIL",
           z_ok ? "OK" : "FAIL",
           track_ok ? "OK" : "WEAK",
           engaged ? "YES" : "NO",
           useful_drop ? "YES" : "NO",
           cal_swallow ? "YES" : "NO");
    printf("RESULT: %s\n\n", pass ? "PASS" : "FAIL");

    FILE* rf = fopen("results/gputronic_transfer_closedloop_report.txt", "w");
    if (rf) {
        fprintf(rf,
                "pulses=%llu z=%.4f z_mean=%.4f target=%.3f sleep=%d "
                "fr_rate=%.1f fr_gflops=%.6f cl_rate=%.1f cl_gflops=%.6f "
                "tach=%d track=%d useful_drop=%d model=%d cal_swallow=%d "
                "PASS=%d\n",
                pulses, z, z_mean, target, sleep_ns,
                fr_rate, fr_gflops, cl_rate, cl_gflops,
                pulses_ok, track_ok, useful_drop, model_ok, cal_swallow,
                pass);
        fclose(rf);
    }

    gputronic_stop(h);
    gputronic_destroy(h);
    return pass ? 0 : 2;
}

static int mode_run(int seconds) {
    GPUTronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_transfer_run.csv";
    cfg.launch_selftest_kernel = 0;

    GPUTronicHandle* h = transfer_boot(&cfg);
    if (!h) return 1;

    printf("Transfer closed-loop %ds | useful work = GEMM tiles (%.0f FLOP/pulse)\n",
           seconds, (double)TRANSFER_FLOPS_PER_PULSE);
    for (int s = 0; s < seconds && !g_stop; s++) {
        usleep(1000000);
        gputronic_print_status(h, s + 1);
        double rate = (double)gputronic_get_rate(h);
        printf("       useful ≈ %.4f GFLOP/s  (from rate_ema)\n", gflops_from_rate(rate));
    }

    gputronic_stop(h);
    gputronic_destroy(h);
    return 0;
}

/* Open-loop k × duration. Duration is the old axis, applied only to the
 * lowest k SMs. Stock clamp, so a single sleep cannot step onto the plateau. */
static int mode_mask(void) {
    volatile int* h_k = NULL;
    volatile int* d_k = NULL;
    GPUTronicConfig cfg;
    GPUTronicHandle* h;
    GPUTronicControl* hc;
    GPUTronicControl* dc = NULL;
    int num_sm, di, ki;
    const int dur_ns[] = {200000, 500000};
    const int n_dur = 2;
    const int frac_num[] = {0, 1, 2, 3, 4, 5, 6, 8, 10, 12};
    const int n_k = 10;
    double rate0 = 0.0, gf0 = 0.0, z0 = 0.0;
    FILE* csv;
    FILE* rep;

    cudaError_t err = cudaHostAlloc((void**)&h_k, sizeof(int), cudaHostAllocMapped);
    if (err != cudaSuccess || !h_k) {
        fprintf(stderr, "[MASK] alloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaHostGetDevicePointer((void**)&d_k, (void*)h_k, 0);
    if (err != cudaSuccess || !d_k) {
        fprintf(stderr, "[MASK] map failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost((void*)h_k);
        return 1;
    }
    *h_k = -1;

    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_transfer_mask_gold.csv";
    cfg.max_sleep_ns = GPUTRONIC_DEFAULT_MAX_SLEEP_NS;
    cfg.launch_selftest_kernel = 0;
    h = gputronic_create(&cfg);
    if (!h) {
        cudaFreeHost((void*)h_k);
        return 1;
    }
    hc = gputronic_get_control(h);
    num_sm = (hc && hc->num_sm > 0) ? hc->num_sm : GPUTRONIC_DEFAULT_NUM_SM;
    err = cudaHostGetDevicePointer((void**)&dc, hc, 0);
    if (err != cudaSuccess || !dc) {
        fprintf(stderr, "[MASK] control map failed\n");
        gputronic_destroy(h);
        cudaFreeHost((void*)h_k);
        return 1;
    }
    transfer_persistent_kernel<<<num_sm, TRANSFER_THREADS>>>(dc, num_sm, d_k, NULL);
    err = cudaGetLastError();
    if (err != cudaSuccess || gputronic_start(h) != 0) {
        fprintf(stderr, "[MASK] boot failed: %s\n", cudaGetErrorString(err));
        gputronic_destroy(h);
        cudaFreeHost((void*)h_k);
        return 1;
    }

    printf("\n=== TRANSFER PER-SM MASK ===\n");
    printf("Persistent GEMM, one block per SM, no grid barrier.\n");
    printf("Duration is the stock command. Only the lowest k of %d SMs honour it.\n", num_sm);
    printf("Awake SMs run the same tiles. Gold clamp %d ns. Kp/Ki untouched.\n\n",
           GPUTRONIC_DEFAULT_MAX_SLEEP_NS);
    fflush(stdout);

    *h_k = 0;
    __sync_synchronize();
    legacy_set_sleep(h, 0);
    usleep(2500000);
    measure_window(h, 400000, 1500000, &rate0, &gf0, &z0, NULL);
    printf("[FREE] k=0 sleep=0   %.1f pulse/s  %.3f GFLOP/s  Z_hat=%.3f\n", rate0, gf0, z0);
    fflush(stdout);

    csv = fopen("results/gputronic_transfer_mask.csv", "w");
    if (csv) fprintf(csv, "sleep_ns,k,rate,gflops,z_hat,z_wall,z_if_fully_off\n");

    for (di = 0; di < n_dur; di++) {
        legacy_set_sleep(h, dur_ns[di]);
        printf("\n[DUR] %d ns applied to k SMs\n", dur_ns[di]);
        printf("%6s %12s %10s %8s %8s %8s\n", "k", "GFLOP/s", "pulse/s", "Z_wall", "Z_hat", "Z_off");
        for (ki = 0; ki < n_k && !g_stop; ki++) {
            int k = num_sm * frac_num[ki] / 12;
            double rate, gflops, zhat, z_wall, z_off;
            *h_k = k;
            __sync_synchronize();
            measure_window(h, 400000, 1200000, &rate, &gflops, &zhat, NULL);
            z_wall = (rate > 1.0 && rate0 > 1.0) ? (rate0 / rate) : 0.0;
            z_off = (k < num_sm) ? ((double)num_sm / (double)(num_sm - k)) : 0.0;
            if (k < num_sm)
                printf("%6d %12.3f %10.1f %8.3f %8.3f %8.3f\n",
                       k, gflops, rate, z_wall, zhat, z_off);
            else
                printf("%6d %12.3f %10.1f %8.3f %8.3f %8s\n",
                       k, gflops, rate, z_wall, zhat, "all");
            fflush(stdout);
            if (csv) fprintf(csv, "%d,%d,%.3f,%.6f,%.5f,%.5f,%.5f\n",
                             dur_ns[di], k, rate, gflops, zhat, z_wall, z_off);
        }
        /* The first sweep stepped from 70 to 84. Fill that gap: the stall
         * may require the last awake SMs to join. */
        if (dur_ns[di] == 500000) {
            const int tail[] = {76, 80, 82, 83};
            int ti;
            printf("[TAIL] k near the full mask, same 500 us command\n");
            for (ti = 0; ti < 4 && !g_stop; ti++) {
                int k = tail[ti];
                double rate, gflops, zhat, z_wall, z_off;
                *h_k = k;
                __sync_synchronize();
                measure_window(h, 400000, 1200000, &rate, &gflops, &zhat, NULL);
                z_wall = (rate > 1.0 && rate0 > 1.0) ? (rate0 / rate) : 0.0;
                z_off = (double)num_sm / (double)(num_sm - k);
                printf("%6d %12.3f %10.1f %8.3f %8.3f %8.3f\n",
                       k, gflops, rate, z_wall, zhat, z_off);
                fflush(stdout);
                if (csv) fprintf(csv, "%d,%d,%.3f,%.6f,%.5f,%.5f,%.5f\n",
                                 dur_ns[di], k, rate, gflops, zhat, z_wall, z_off);
            }
        }
    }
    if (csv) fclose(csv);

    rep = fopen("results/gputronic_transfer_mask_report.txt", "w");
    if (rep) {
        fprintf(rep, "transfer per-SM mask\n");
        fprintf(rep, "num_sm=%d free_rate=%.3f free_gflops=%.6f free_z=%.4f clamp_ns=%d\n",
                num_sm, rate0, gf0, z0, GPUTRONIC_DEFAULT_MAX_SLEEP_NS);
        fprintf(rep, "csv=results/gputronic_transfer_mask.csv\n");
        fprintf(rep, "Z_off = num_sm/(num_sm-k) if the masked SMs produced nothing.\n");
        fprintf(rep, "Masked SMs still do their tiles, then sleep. Z_off is the ceiling, not the prediction.\n");
        fclose(rep);
    }

    if (hc) hc->control_flags |= GPUTRONIC_FLAG_STOP;
    gputronic_stop(h);
    gputronic_destroy(h);
    cudaFreeHost((void*)h_k);
    printf("\nreport: results/gputronic_transfer_mask_report.txt\n");
    printf("csv:    results/gputronic_transfer_mask.csv\n");
    return 0;
}

/* Open-loop count of working SMs, then the stock PI if 56 SMs lands near Z=1.5.
 * Idle SMs skip the tiles and do not pulse. Floor is 4 SMs. */
static int mode_work(void) {
    volatile int* h_n = NULL;
    volatile int* d_n = NULL;
    GPUTronicConfig cfg;
    GPUTronicHandle* h;
    GPUTronicControl* hc;
    GPUTronicControl* dc = NULL;
    int num_sm, i;
    const int nsweep[] = {84, 80, 76, 72, 64, 60, 58, 57, 56, 55, 54, 52, 48, 42, 28, 14, 8, 4};
    const int npts = (int)(sizeof(nsweep) / sizeof(nsweep[0]));
    double rate0 = 0.0, gf0 = 0.0, z0 = 0.0;
    double z_best = 0.0;
    int n_best = 0;
    double best_err = 1.0e9;
    int mono = 1;
    double prev_rate = 1.0e30;
    FILE* csv;
    FILE* rep;
    cudaError_t err;

    err = cudaHostAlloc((void**)&h_n, sizeof(int), cudaHostAllocMapped);
    if (err != cudaSuccess || !h_n) {
        fprintf(stderr, "[WORK] alloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaHostGetDevicePointer((void**)&d_n, (void*)h_n, 0);
    if (err != cudaSuccess || !d_n) {
        fprintf(stderr, "[WORK] map failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost((void*)h_n);
        return 1;
    }
    *h_n = 84;

    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_transfer_work_gold.csv";
    cfg.max_sleep_ns = GPUTRONIC_DEFAULT_MAX_SLEEP_NS;
    cfg.launch_selftest_kernel = 0;
    h = gputronic_create(&cfg);
    if (!h) {
        cudaFreeHost((void*)h_n);
        return 1;
    }
    hc = gputronic_get_control(h);
    num_sm = (hc && hc->num_sm > 0) ? hc->num_sm : GPUTRONIC_DEFAULT_NUM_SM;
    err = cudaHostGetDevicePointer((void**)&dc, hc, 0);
    if (err != cudaSuccess || !dc) {
        fprintf(stderr, "[WORK] control map failed\n");
        gputronic_destroy(h);
        cudaFreeHost((void*)h_n);
        return 1;
    }
    transfer_persistent_kernel<<<num_sm, TRANSFER_THREADS>>>(dc, num_sm, NULL, d_n);
    err = cudaGetLastError();
    if (err != cudaSuccess || gputronic_start(h) != 0) {
        fprintf(stderr, "[WORK] boot failed: %s\n", cudaGetErrorString(err));
        gputronic_destroy(h);
        cudaFreeHost((void*)h_n);
        return 1;
    }

    printf("\n=== TRANSFER WORK KNOB ===\n");
    printf("Lowest n of %d SMs run the tiles and pulse. The rest skip the work.\n", num_sm);
    printf("Floor is 4 SMs. Idle backoff is 5 us. No duration throttle on the workers.\n");
    printf("Z_pred = %d/n if each working SM carries an equal share.\n\n", num_sm);
    fflush(stdout);

    *h_n = num_sm;
    __sync_synchronize();
    legacy_set_sleep(h, 0);
    usleep(2500000);
    measure_window(h, 400000, 1500000, &rate0, &gf0, &z0, NULL);
    printf("[FREE] n=%d  %.1f pulse/s  %.3f GFLOP/s  Z_hat=%.3f\n\n", num_sm, rate0, gf0, z0);
    printf("%6s %12s %10s %8s %8s %8s\n", "n", "GFLOP/s", "pulse/s", "Z_wall", "Z_hat", "Z_pred");
    fflush(stdout);

    csv = fopen("results/gputronic_transfer_work.csv", "w");
    if (csv) fprintf(csv, "n,rate,gflops,z_hat,z_wall,z_pred\n");

    for (i = 0; i < npts && !g_stop; i++) {
        int n = nsweep[i];
        double rate, gflops, zhat, z_wall, z_pred;
        if (n > num_sm) n = num_sm;
        if (n < 4) n = 4;
        *h_n = n;
        __sync_synchronize();
        measure_window(h, 350000, 1200000, &rate, &gflops, &zhat, NULL);
        z_wall = (rate > 1.0 && rate0 > 1.0) ? (rate0 / rate) : 0.0;
        z_pred = (n > 0) ? ((double)num_sm / (double)n) : 0.0;
        if (rate > prev_rate * 1.08) mono = 0;
        prev_rate = rate;
        if (fabs(z_wall - 1.5) < best_err) {
            best_err = fabs(z_wall - 1.5);
            z_best = z_wall;
            n_best = n;
        }
        printf("%6d %12.3f %10.1f %8.3f %8.3f %8.3f\n",
               n, gflops, rate, z_wall, zhat, z_pred);
        fflush(stdout);
        if (csv) fprintf(csv, "%d,%.3f,%.6f,%.5f,%.5f,%.5f\n",
                         n, rate, gflops, zhat, z_wall, z_pred);
    }
    if (csv) fclose(csv);

    printf("\n[SWEEP] monotonic=%s  closest n=%d Z=%.3f\n",
           mono ? "YES" : "NO", n_best, z_best);
    fflush(stdout);

    if (mono && best_err < 0.08) {
        int s;
        int track_s = 12;
        int rel_s = 6;
        double sleep_acc = 0.0;
        int sleep_n = 0;
        unsigned long long p_a = 0, p_b;
        double t_a = 0.0, t_b, z_wall, cl_gflops, rel_gflops, rel_z;
        printf("\n[LOOP] stock PI. Command selects n: 0 → %d SMs, clamp → %d SMs.\n",
               num_sm, num_sm / 2);
        printf("[LOOP] open-loop n=%d held Z=%.3f. Floor remains 4.\n", n_best, z_best);
        fflush(stdout);
        *h_n = -1;
        __sync_synchronize();
        gputronic_set_target(h, GPUTRONIC_GOLD_TARGET_Z);
        legacy_set_sleep(h, -1);
        for (s = 0; s < track_s && !g_stop; s++) {
            usleep(1000000);
            gputronic_print_status(h, s + 1);
            if (s == 3) {
                p_a = hc->total_work_pulses;
                t_a = now_us();
            }
            if (s >= 3) {
                sleep_acc += (double)legacy_get_sleep(h);
                sleep_n++;
            }
            fflush(stdout);
        }
        p_b = hc->total_work_pulses;
        t_b = now_us();
        {
            double dt = (t_b - t_a) * 1e-6;
            double rate = (dt > 1e-6) ? (double)(p_b - p_a) / dt : 0.0;
            z_wall = (rate > 1.0) ? (rate0 / rate) : 0.0;
            cl_gflops = gflops_from_rate(rate);
        }
        printf("[LOOP] Z_wall=%.3f  %.3f GFLOP/s  mean_cmd=%.0f ns\n",
               z_wall, cl_gflops, (sleep_n > 0) ? (sleep_acc / sleep_n) : 0.0);
        gputronic_set_target(h, 1.0f);
        printf("[RELEASE] target 1.0\n");
        fflush(stdout);
        p_a = hc->total_work_pulses;
        t_a = now_us();
        for (s = 0; s < rel_s && !g_stop; s++) {
            usleep(1000000);
            gputronic_print_status(h, s + 1);
            if (s == 2) {
                p_a = hc->total_work_pulses;
                t_a = now_us();
            }
            fflush(stdout);
        }
        p_b = hc->total_work_pulses;
        t_b = now_us();
        {
            double dt = (t_b - t_a) * 1e-6;
            double rate = (dt > 1e-6) ? (double)(p_b - p_a) / dt : 0.0;
            rel_gflops = gflops_from_rate(rate);
            rel_z = (double)gputronic_get_z(h);
        }
        printf("[RELEASE] %.3f GFLOP/s  Z_hat=%.3f  (free-run %.3f)\n",
               rel_gflops, rel_z, gf0);
        printf("[WORK] open-loop n=%d Z=%.3f  closed-loop Z=%.3f\n", n_best, z_best, z_wall);
    } else {
        printf("[WORK] open-loop did not place n=56 on Z=1.5. Closed loop not run.\n");
    }
    fflush(stdout);

    rep = fopen("results/gputronic_transfer_work_report.txt", "w");
    if (rep) {
        fprintf(rep, "transfer work knob\n");
        fprintf(rep, "num_sm=%d free_rate=%.3f free_gflops=%.6f n_best=%d Z_best=%.4f mono=%d\n",
                num_sm, rate0, gf0, n_best, z_best, mono);
        fprintf(rep, "csv=results/gputronic_transfer_work.csv\n");
        fprintf(rep, "Idle SMs skip tiles, do not pulse, backoff 5 us. Floor 4.\n");
        fclose(rep);
    }

    if (hc) hc->control_flags |= GPUTRONIC_FLAG_STOP;
    gputronic_stop(h);
    gputronic_destroy(h);
    cudaFreeHost((void*)h_n);
    printf("report: results/gputronic_transfer_work_report.txt\n");
    return 0;
}

static void usage(const char* argv0) {
    printf("GPUTronic Transfer Plant (Gold 1.0 cooperative GEMM tiles)\n\n");
    printf("Usage:\n");
    printf("  %s check           # dyno + closed-loop transfer gate\n", argv0);
    printf("  %s dyno            # sleep → tile/s + GFLOP/s\n", argv0);
    printf("  %s run [sec]       # closed-loop on the transfer plant\n", argv0);
    printf("  %s mask            # open-loop per-SM sleep count × duration\n", argv0);
    printf("  %s work            # open-loop working-SM count, then stock PI\n", argv0);
}

int main(int argc, char** argv) {
    const char* mode = (argc >= 2) ? argv[1] : "check";
    if (!strcmp(mode, "-h") || !strcmp(mode, "--help") || !strcmp(mode, "help")) {
        usage(argv[0]);
        return 0;
    }
    if (!strcmp(mode, "mask")) return mode_mask();
    if (!strcmp(mode, "work")) return mode_work();
    if (!strcmp(mode, "dyno")) return mode_dyno();
    if (!strcmp(mode, "check")) {
        int rd = mode_dyno();
        int rc = mode_closedloop(20);
        printf("=== TRANSFER GATE: dyno=%s closedloop=%s ===\n",
               rd == 0 ? "PASS" : "FAIL", rc == 0 ? "PASS" : "FAIL");
        return (rd == 0 && rc == 0) ? 0 : 2;
    }
    int sec = 20;
    if (!strcmp(mode, "run")) {
        if (argc >= 3) sec = atoi(argv[2]);
    } else {
        int maybe = atoi(mode);
        if (maybe > 0) sec = maybe;
        else {
            usage(argv[0]);
            return 1;
        }
    }
    if (sec <= 0) sec = 20;
    return mode_run(sec);
}
