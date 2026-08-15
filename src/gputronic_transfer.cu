/* =============================================================================
 * GPUTronic Transfer Plant — cooperative GEMM microtiles on Gold 1.0
 * -----------------------------------------------------------------------------
 * Second plant on the Gold 1.0 interface (stock auto sleep_scale).
 *   - launch_selftest_kernel = 0
 *   - persistent kernel does real FP32 GEMM microtiles (shared-mem 8x8x8)
 *   - block leader atomicAdd on Gold's total_work_pulses
 *   - honours throttle_sleep_ns via __nanosleep
 *   - useful work = completed tiles * FLOPS_PER_TILE
 *
 * Transfer claim (the only one that matters):
 *   sleep ↑ → tile rate ↓ → Z ↑  AND  GFLOP/s ↓
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

/* -------------------- cooperative plant -------------------- */

__global__ void transfer_persistent_kernel(GPUTronicControl* ctrl, int num_sm) {
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
            __nanosleep(5000);
            continue;
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

        int sleep_ns = ctrl->throttle_sleep_ns;
        int max_s = ctrl->max_sleep_ns;
        if (sleep_ns < 0) sleep_ns = 0;
        if (max_s > 0 && sleep_ns > max_s) sleep_ns = max_s;
        if (sleep_ns > 0) __nanosleep((unsigned int)sleep_ns);

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

    transfer_persistent_kernel<<<num_sm, TRANSFER_THREADS>>>(d_ctrl, num_sm);
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

/* -------------------- modes -------------------- */

static int mode_dyno(void) {
    GPUTronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_transfer_dyno.csv";
    cfg.max_sleep_ns = 2000000;

    GPUTronicHandle* h = transfer_boot(&cfg);
    if (!h) return 1;

    gputronic_set_open_loop_sleep(h, 0);
    usleep(2500000);

    const int sleeps[] = {0, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000};
    const int n = (int)(sizeof(sleeps) / sizeof(sleeps[0]));
    double xs[16], ys[16], gf[16];

    printf("\n=== TRANSFER DYNO (open-loop sleep → tile rate + GFLOP/s) ===\n");
    printf("%10s %14s %12s %10s %12s\n", "sleep_ns", "tile/s", "GFLOP/s", "Z_hat", "tiles_d");
    printf("--------------------------------------------------------------------\n");

    FILE* df = fopen("results/gputronic_transfer_dyno_summary.csv", "w");
    if (df) fprintf(df, "sleep_ns,tile_rate,gflops,z_hat,delta_tiles\n");

    for (int i = 0; i < n && !g_stop; i++) {
        gputronic_set_open_loop_sleep(h, sleeps[i]);
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
    printf("fit: tile/s ≈ %.1f + (%.6f)*sleep_ns   R²=%.4f\n", a, b, r2);
    printf("free-run  %.1f tile/s  (%.4f GFLOP/s)\n", ys[0], gf[0]);
    printf("500 us    %.1f tile/s  (%.4f GFLOP/s)  rate_cut=%s flop_cut=%s coupled=%s\n",
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
    /* Gold 1.0 auto-schedules sleep_scale from the measured loop period. */

    GPUTronicHandle* h = transfer_boot(&cfg);
    if (!h) return 1;

    printf("\n=== TRANSFER CLOSED-LOOP ===\n");
    printf("[LAW] target=%.2f | Gold auto_sleep_scale (no transfer override)\n",
           GPUTRONIC_GOLD_TARGET_Z);

    usleep(2200000);
    gputronic_set_open_loop_sleep(h, 0);
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
    gputronic_set_open_loop_sleep(h, -1);

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
    int sleep_ns = gputronic_get_sleep_ns(h);
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
                "sleep_scale=%.0f PASS=%d\n",
                pulses, z, z_mean, target, sleep_ns,
                fr_rate, fr_gflops, cl_rate, cl_gflops,
                pulses_ok, track_ok, useful_drop, model_ok, cal_swallow,
                cfg.sleep_scale, pass);
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

static void usage(const char* argv0) {
    printf("GPUTronic Transfer Plant (Gold 1.0 cooperative GEMM tiles)\n\n");
    printf("Usage:\n");
    printf("  %s check           # dyno + closed-loop transfer gate\n", argv0);
    printf("  %s dyno            # sleep → tile/s + GFLOP/s\n", argv0);
    printf("  %s run [sec]       # closed-loop on the transfer plant\n", argv0);
}

int main(int argc, char** argv) {
    const char* mode = (argc >= 2) ? argv[1] : "check";
    if (!strcmp(mode, "-h") || !strcmp(mode, "--help") || !strcmp(mode, "help")) {
        usage(argv[0]);
        return 0;
    }
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
