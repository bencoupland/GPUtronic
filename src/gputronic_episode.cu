/* =============================================================================
 * GPUTronic Episode Plant — launch-complete GEMM on Gold 1.0
 * -----------------------------------------------------------------------------
 * Gold 1.0 source is untouched. Runtime knobs (env) are for observer/throw
 * campaigns; defaults preserve the original episode gate.
 *
 *   GPUTRONIC_EPISODE_PULSES     pulses per launch (default 4)
 *   GPUTRONIC_EPISODE_HONOUR     once | host | chunkN   (default once)
 *   GPUTRONIC_EPISODE_PREFIX     result path prefix
 *   GPUTRONIC_EPISODE_SLEEPS     comma list of open-loop sleeps (ns)
 *   GPUTRONIC_EPISODE_SETTLE_US / WINDOW_US / CL_SEC
 *
 * honour=once   : one __nanosleep(cmd) at epilogue (llama-shaped default)
 * honour=chunkN : N device nanosleeps of the commanded value (throw without
 *                 raising Gold max_sleep)
 * honour=host   : kernel does not sleep; host usleep(cmd) after sync
 *
 * `proof` is the demo on this same plant (Gold untouched, stock Kp/Ki/clamp):
 *   1 uninstrumented twin
 *   2 tachometer on, actuator at base — the tax
 *   3 open-loop stall sized to the launch, near Z=1.5, on a slope
 *   4 stock PI, then release, only if phase 3 could already sit there
 * ============================================================================= */

#include "gputronic.h"

#include <cuda_runtime.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define EPISODE_TILE 8
#define EPISODE_THREADS 64
#define EPISODE_TILES_PER_PULSE 32
#define EPISODE_FLOPS_PER_TILE (2ULL * EPISODE_TILE * EPISODE_TILE * EPISODE_TILE)
#define EPISODE_FLOPS_PER_PULSE (EPISODE_FLOPS_PER_TILE * (unsigned long long)EPISODE_TILES_PER_PULSE)
#define EPISODE_MAX_SLEEPS 32

static volatile int g_stop = 0;
static int g_pulses = 4;
static int g_sleep_shots = 1; /* 0 = device does not sleep */
static int g_host_wait = 0;
static char g_honour[32] = "once";
static char g_prefix[256] = "results/gputronic_episode";
static int g_settle_us = 1500000;
static int g_window_us = 2000000;
static int g_cl_sec = 20;

static inline double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
}

static int env_int(const char* key, int fallback) {
    const char* v = getenv(key);
    if (!v || !v[0]) return fallback;
    return atoi(v);
}

static void episode_load_env(void) {
    g_pulses = env_int("GPUTRONIC_EPISODE_PULSES", 4);
    if (g_pulses < 1) g_pulses = 1;
    if (g_pulses > 256) g_pulses = 256;
    g_settle_us = env_int("GPUTRONIC_EPISODE_SETTLE_US", 1500000);
    g_window_us = env_int("GPUTRONIC_EPISODE_WINDOW_US", 2000000);
    g_cl_sec = env_int("GPUTRONIC_EPISODE_CL_SEC", 20);
    if (g_cl_sec < 5) g_cl_sec = 5;

    const char* hon = getenv("GPUTRONIC_EPISODE_HONOUR");
    if (hon && hon[0]) {
        strncpy(g_honour, hon, sizeof(g_honour) - 1);
        g_honour[sizeof(g_honour) - 1] = 0;
    }
    g_sleep_shots = 1;
    g_host_wait = 0;
    if (!strcmp(g_honour, "once")) {
        g_sleep_shots = 1;
    } else if (!strcmp(g_honour, "host")) {
        g_sleep_shots = 0;
        g_host_wait = 1;
    } else if (!strncmp(g_honour, "chunk", 5)) {
        int n = atoi(g_honour + 5);
        if (n < 1) n = 1;
        if (n > 16) n = 16;
        g_sleep_shots = n;
    }

    const char* pref = getenv("GPUTRONIC_EPISODE_PREFIX");
    if (pref && pref[0]) {
        strncpy(g_prefix, pref, sizeof(g_prefix) - 1);
        g_prefix[sizeof(g_prefix) - 1] = 0;
    }
}

/* -------------------- launch-complete plant -------------------- */

__global__ void episode_kernel(GPUTronicControl* ctrl, int num_sm,
                               int pulses_per_episode, int sleep_shots) {
    const int sm = blockIdx.x;
    const int tid = threadIdx.x;
    const int r = tid / EPISODE_TILE;
    const int c = tid % EPISODE_TILE;
    float acc = 0.0f;

    if (sm >= num_sm) return;

    int active = ctrl->active_sms;
    if (active <= 0 || active > num_sm) active = num_sm;
    if (active < 4) active = 4;
    if (sm >= active) return;

    unsigned int flags = ctrl->control_flags;
    if (flags & GPUTRONIC_FLAG_STOP) return;
    if (flags & GPUTRONIC_FLAG_PAUSE) {
        __nanosleep(5000);
        return;
    }

    __shared__ float As[EPISODE_TILE][EPISODE_TILE];
    __shared__ float Bs[EPISODE_TILE][EPISODE_TILE];

    unsigned int iter = 0;
    for (int p = 0; p < pulses_per_episode; p++) {
        for (int t = 0; t < EPISODE_TILES_PER_PULSE; t++) {
            float seed = (float)((iter * 31u + (unsigned)sm * 17u + (unsigned)tid + (unsigned)t * 13u) & 1023u);
            if (r < EPISODE_TILE && c < EPISODE_TILE) {
                As[r][c] = seed * 0.001f + 0.10f;
                Bs[r][c] = seed * 0.0007f + 0.20f;
            }
            __syncthreads();

            if (r < EPISODE_TILE && c < EPISODE_TILE) {
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < EPISODE_TILE; k++)
                    s = __fmaf_rn(As[r][k], Bs[k][c], s);
                acc += s;
            }
            __syncthreads();
            iter++;
        }

        if (acc > 1.0e30f) acc = 0.0f;
        if (tid == 0)
            atomicAdd((unsigned long long*)&ctrl->total_work_pulses, 1ULL);
    }

    if (tid == 0) __threadfence_system();
    int sleep_ns = ctrl->throttle_sleep_ns;
    int max_s = ctrl->max_sleep_ns;
    if (sleep_ns < 0) sleep_ns = 0;
    if (max_s > 0 && sleep_ns > max_s) sleep_ns = max_s;
    if (sleep_ns > 1000 && sleep_shots > 0) {
        for (int s = 0; s < sleep_shots; s++)
            __nanosleep((unsigned int)sleep_ns);
    }

    if (tid == 0 && acc == 3.14159265f)
        atomicAdd((unsigned long long*)&ctrl->total_work_pulses, 0ULL);
}

struct EpisodePump {
    GPUTronicControl* h_ctrl;
    GPUTronicControl* d_ctrl;
    int num_sm;
    pthread_t thr;
    int running;
    unsigned long long launches;
    double work_us_acc;
};

static void* episode_pump_thread(void* arg) {
    EpisodePump* p = (EpisodePump*)arg;
    cudaSetDevice(0);
    while (!g_stop && p->running) {
        unsigned int flags = p->h_ctrl->control_flags;
        if (flags & GPUTRONIC_FLAG_STOP) break;
        if (flags & GPUTRONIC_FLAG_PAUSE) {
            usleep(200);
            continue;
        }
        double t0 = now_us();
        episode_kernel<<<p->num_sm, EPISODE_THREADS>>>(p->d_ctrl, p->num_sm, g_pulses, g_sleep_shots);
        cudaError_t err = cudaDeviceSynchronize();
        if (g_host_wait) {
            int sleep_ns = p->h_ctrl->throttle_sleep_ns;
            int max_s = p->h_ctrl->max_sleep_ns;
            if (sleep_ns < 0) sleep_ns = 0;
            if (max_s > 0 && sleep_ns > max_s) sleep_ns = max_s;
            if (sleep_ns >= 1000) usleep((useconds_t)(sleep_ns / 1000));
        }
        p->work_us_acc += now_us() - t0;
        p->launches++;
        if (err != cudaSuccess) {
            fprintf(stderr, "[EPISODE] launch/sync failed: %s\n", cudaGetErrorString(err));
            break;
        }
    }
    p->running = 0;
    return NULL;
}

static int episode_pump_start(EpisodePump* p, GPUTronicHandle* h) {
    memset(p, 0, sizeof(*p));
    p->h_ctrl = gputronic_get_control(h);
    if (!p->h_ctrl) {
        fprintf(stderr, "[EPISODE] no control block\n");
        return -1;
    }
    cudaError_t err = cudaHostGetDevicePointer((void**)&p->d_ctrl, p->h_ctrl, 0);
    if (err != cudaSuccess || !p->d_ctrl) {
        fprintf(stderr, "[EPISODE] cudaHostGetDevicePointer failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    p->num_sm = p->h_ctrl->num_sm;
    if (p->num_sm <= 0) p->num_sm = GPUTRONIC_DEFAULT_NUM_SM;
    p->running = 1;
    if (pthread_create(&p->thr, NULL, episode_pump_thread, p) != 0) {
        fprintf(stderr, "[EPISODE] pump thread failed\n");
        p->running = 0;
        return -1;
    }
    printf("[EPISODE] pump started | %d blocks x %d threads | tile=%dx%dx%d x %d x %d | %.0f FLOP/pulse\n",
           p->num_sm, EPISODE_THREADS, EPISODE_TILE, EPISODE_TILE, EPISODE_TILE,
           EPISODE_TILES_PER_PULSE, g_pulses, (double)EPISODE_FLOPS_PER_PULSE);
    printf("[EPISODE] honour=%s sleep_shots=%d host_wait=%d | host_ctrl=%p device_ctrl=%p %s\n",
           g_honour, g_sleep_shots, g_host_wait, (void*)p->h_ctrl, (void*)p->d_ctrl,
           ((uintptr_t)p->h_ctrl == (uintptr_t)p->d_ctrl) ? "ptr-equal" : "mapped");
    return 0;
}

static void episode_pump_stop(EpisodePump* p) {
    if (!p) return;
    p->running = 0;
    if (p->h_ctrl) p->h_ctrl->control_flags |= GPUTRONIC_FLAG_STOP;
    if (p->thr) pthread_join(p->thr, NULL);
    p->thr = 0;
}

static GPUTronicHandle* episode_boot(GPUTronicConfig* cfg, EpisodePump* pump) {
    cfg->launch_selftest_kernel = 0;
    GPUTronicHandle* h = gputronic_create(cfg);
    if (!h) return NULL;
    if (episode_pump_start(pump, h) != 0) {
        gputronic_destroy(h);
        return NULL;
    }
    if (gputronic_start(h) != 0) {
        episode_pump_stop(pump);
        gputronic_destroy(h);
        return NULL;
    }
    return h;
}

static void episode_shutdown(GPUTronicHandle* h, EpisodePump* pump) {
    episode_pump_stop(pump);
    if (h) {
        gputronic_stop(h);
        gputronic_destroy(h);
    }
}

static double gflops_from_rate(double pulse_rate) {
    return pulse_rate * (double)EPISODE_FLOPS_PER_PULSE / 1.0e9;
}

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

static int parse_sleeps(int* sleeps, int maxn) {
    const char* spec = getenv("GPUTRONIC_EPISODE_SLEEPS");
    if (!spec || !spec[0]) {
        const int def[] = {0, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000};
        int n = (int)(sizeof(def) / sizeof(def[0]));
        if (n > maxn) n = maxn;
        memcpy(sleeps, def, (size_t)n * sizeof(int));
        return n;
    }
    int n = 0;
    const char* p = spec;
    while (*p && n < maxn) {
        sleeps[n++] = atoi(p);
        const char* c = strchr(p, ',');
        if (!c) break;
        p = c + 1;
    }
    return n;
}

static int find_sleep(const int* sleeps, int n, int target) {
    int best = 0;
    int best_d = abs(sleeps[0] - target);
    for (int i = 1; i < n; i++) {
        int d = abs(sleeps[i] - target);
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    return best;
}

static void path_join(char* out, size_t n, const char* suffix) {
    snprintf(out, n, "%s%s", g_prefix, suffix);
}

/* -------------------- modes -------------------- */

static int mode_dyno(void) {
    GPUTronicConfig cfg;
    gputronic_config_gold(&cfg);
    char csv[320], sum[320], rep[320];
    path_join(csv, sizeof(csv), "_dyno.csv");
    path_join(sum, sizeof(sum), "_dyno_summary.csv");
    path_join(rep, sizeof(rep), "_dyno_report.txt");
    cfg.csv_path = csv;
    cfg.max_sleep_ns = 2000000;

    EpisodePump pump;
    GPUTronicHandle* h = episode_boot(&cfg, &pump);
    if (!h) return 1;

    gputronic_set_open_loop_sms(h, 0);
    usleep(3000000);

    int num_sm = pump.num_sm > 0 ? pump.num_sm : GPUTRONIC_DEFAULT_NUM_SM;
    int counts[8] = {num_sm, num_sm * 3 / 4, num_sm / 2, num_sm / 3, num_sm / 4, 16, 8, 4};
    int n = 8;
    int i500 = 2;
    int i2000 = n - 1;
    double xs[8], ys[8], gf[8], zs[8];

    printf("\n=== EPISODE DYNO (open-loop working SMs → tile rate + GFLOP/s) ===\n");
    printf("%10s %14s %12s %10s %12s\n", "n_sm", "tile/s", "GFLOP/s", "Z_hat", "tiles_d");
    printf("--------------------------------------------------------------------\n");

    FILE* df = fopen(sum, "w");
    if (df) fprintf(df, "active_sms,tile_rate,gflops,z_hat,delta_tiles,pulses,honour\n");

    for (int i = 0; i < n && !g_stop; i++) {
        int nsm = counts[i] < 4 ? 4 : counts[i];
        counts[i] = nsm;
        gputronic_set_open_loop_sms(h, nsm);
        double rate, gflops, z;
        unsigned long long dp = 0;
        measure_window(h, g_settle_us, g_window_us, &rate, &gflops, &z, &dp);
        xs[i] = (double)counts[i];
        ys[i] = rate;
        gf[i] = gflops;
        zs[i] = z;
        printf("%10d %14.1f %12.4f %10.4f %12llu\n",
               counts[i], rate, gflops, z, (unsigned long long)dp);
        if (df) fprintf(df, "%d,%.3f,%.6f,%.5f,%llu,%d,%s\n",
                        counts[i], rate, gflops, z, (unsigned long long)dp, g_pulses, g_honour);
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

    int tach_ok = ys[0] > 1000.0;
    int cut500 = tach_ok && (ys[i500] < ys[0] * 0.75);
    int flop500 = (gf[0] > 0.01) && (gf[i500] < gf[0] * 0.75);
    int cut2ms = tach_ok && (ys[i2000] < ys[0] * 0.75);
    int flop2ms = (gf[0] > 0.01) && (gf[i2000] < gf[0] * 0.75);
    int z_sees = (zs[i2000] > zs[0] + 0.12) || (zs[i500] > zs[0] + 0.08);
    int coupled = 1;
    for (int i = 1; i < n; i++) {
        double dr = ys[i] - ys[i - 1];
        double dg = gf[i] - gf[i - 1];
        if (fabs(dr) > 0.05 * ys[0] && fabs(dg) > 0.05 * gf[0] && dr * dg < 0.0)
            coupled = 0;
    }
    int decreasing = 1;
    for (int i = 1; i < n; i++) {
        if (ys[i] > ys[i - 1] * 1.15) decreasing = 0;
    }

    int window_ok = tach_ok && z_sees && coupled;
    int plant_ok = window_ok && cut2ms && flop2ms && (r2 >= 0.55 || decreasing);
    int gold_act = cut500 && flop500;
    int pass = plant_ok;

    double mean_launch_us = (pump.launches > 0) ? (pump.work_us_acc / (double)pump.launches) : 0.0;
    double f09_period_ns = 0.0;
    if (ys[0] > 1.0 && pump.num_sm > 0)
        f09_period_ns = 1e9 * (double)pump.num_sm / ys[0];

    printf("--------------------------------------------------------------------\n");
    printf("fit: tile/s ≈ %.1f + (%.6f)*active_sms   R²=%.4f\n", a, b, r2);
    printf("free-run  %.1f tile/s  (%.4f GFLOP/s)  Z=%.3f\n", ys[0], gf[0], zs[0]);
    printf("n=%-4d    %.1f tile/s  (%.4f GFLOP/s)  Z=%.3f\n",
           counts[i500], ys[i500], gf[i500], zs[i500]);
    printf("n=%-4d    %.1f tile/s  (%.4f GFLOP/s)  Z=%.3f  plant_gain=%s\n",
           counts[i2000], ys[i2000], gf[i2000], zs[i2000], (cut2ms && flop2ms) ? "YES" : "NO");
    printf("window_sees_Z=%s coupled=%s mean_episode=%.0f us launches=%llu\n",
           z_sees ? "YES" : "NO", coupled ? "YES" : "NO",
           mean_launch_us, (unsigned long long)pump.launches);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    printf("JSON:{\"mode\":\"dyno\",\"pulses\":%d,\"honour\":\"%s\",\"sleep_shots\":%d,\"host_wait\":%d,"
           "\"r2\":%.4f,\"rate0\":%.1f,\"gflops0\":%.4f,\"z0\":%.4f,"
           "\"rate500\":%.1f,\"gflops500\":%.4f,\"z500\":%.4f,"
           "\"rate2ms\":%.1f,\"gflops2ms\":%.4f,\"z2ms\":%.4f,"
           "\"mean_episode_us\":%.1f,\"f09_period_ns\":%.1f,\"launches\":%llu,"
           "\"window_ok\":%d,\"plant_ok\":%d,\"gold_act\":%d,\"pass\":%d}\n\n",
           g_pulses, g_honour, g_sleep_shots, g_host_wait, r2,
           ys[0], gf[0], zs[0], ys[i500], gf[i500], zs[i500],
           ys[i2000], gf[i2000], zs[i2000], mean_launch_us, f09_period_ns,
           (unsigned long long)pump.launches, window_ok, plant_ok, gold_act, pass);

    FILE* rf = fopen(rep, "w");
    if (rf) {
        fprintf(rf, "GPUTronic Episode Dyno Report\n");
        fprintf(rf, "pulses=%d honour=%s sleep_shots=%d host_wait=%d\n",
                g_pulses, g_honour, g_sleep_shots, g_host_wait);
        fprintf(rf, "R2=%.4f slope=%.6f intercept=%.1f window=%d plant=%d gold_act_500us=%d PASS=%d\n",
                r2, b, a, window_ok, plant_ok, gold_act, pass);
        fprintf(rf, "flops_per_pulse=%llu tiles_per_pulse=%d pulses_per_episode=%d mean_episode_us=%.1f f09_period_ns=%.1f launches=%llu\n",
                (unsigned long long)EPISODE_FLOPS_PER_PULSE, EPISODE_TILES_PER_PULSE,
                g_pulses, mean_launch_us, f09_period_ns, (unsigned long long)pump.launches);
        for (int i = 0; i < n; i++)
            fprintf(rf, "active_sms=%d rate=%.1f gflops=%.6f z=%.4f\n", counts[i], ys[i], gf[i], zs[i]);
        fclose(rf);
    }

    episode_shutdown(h, &pump);
    return pass ? 0 : 2;
}

static int mode_closedloop(int seconds) {
    GPUTronicConfig cfg;
    gputronic_config_gold(&cfg);
    char csv[320], rep[320];
    path_join(csv, sizeof(csv), "_closedloop.csv");
    path_join(rep, sizeof(rep), "_closedloop_report.txt");
    cfg.csv_path = csv;
    cfg.max_sleep_ns = 500000;
    cfg.launch_selftest_kernel = 0;

    EpisodePump pump;
    GPUTronicHandle* h = episode_boot(&cfg, &pump);
    if (!h) return 1;

    printf("\n=== EPISODE CLOSED-LOOP ===\n");
    printf("[LAW] target=%.2f | Gold auto_sleep_scale | max_sleep=500us (stock) | honour=%s shots=%d host=%d\n",
           GPUTRONIC_GOLD_TARGET_Z, g_honour, g_sleep_shots, g_host_wait);

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
        printf("[CAL] WARN free-run Z=%.3f still near/above target %.2f\n", z_free, target);
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
    double mean_launch_us = (pump.launches > 0) ? (pump.work_us_acc / (double)pump.launches) : 0.0;
    double pulses_per_8ms = cl_rate * 0.008;

    int pulses_ok = (pulses > 1000ULL);
    int z_ok = (z >= cfg.z_floor && z <= cfg.z_ceiling);
    int rate_ok = (rate_ema > 100.0f) || (cl_rate > 100.0);
    float z_err = fabsf(z_mean - target);
    int track_ok = (z_err < 0.50f) || (sleep_ns > 80000);
    int engaged = (sleep_ns > 80000) || (z_mean > z_free + 0.12f);
    int useful_drop = (fr_gflops > 0.01 && cl_gflops < fr_gflops * 0.90);
    int track_tight = (z_err < 0.20f);
    int window_ok = pulses_ok && z_ok && rate_ok && !cal_swallow;
    int model_ok = useful_drop && engaged;
    int pass = window_ok && model_ok;

    printf("\n[CHECK] pulses=%llu z=%.3f z_mean=%.3f target=%.2f ema=%.1f sleep=%d\n",
           pulses, z, z_mean, target, rate_ema, sleep_ns);
    printf("[CHECK] free-run %.1f pulse/s (%.4f GFLOP/s) | closed %.1f pulse/s (%.4f GFLOP/s)\n",
           fr_rate, fr_gflops, cl_rate, cl_gflops);
    printf("[CHECK] tach=%s z_sane=%s track=%s engaged=%s useful_drop=%s cal_swallow=%s window=%s\n",
           pulses_ok ? "OK" : "FAIL",
           z_ok ? "OK" : "FAIL",
           track_ok ? "OK" : "WEAK",
           engaged ? "YES" : "NO",
           useful_drop ? "YES" : "NO",
           cal_swallow ? "YES" : "NO",
           window_ok ? "OK" : "FAIL");
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    printf("JSON:{\"mode\":\"closedloop\",\"pulses\":%d,\"honour\":\"%s\",\"sleep_shots\":%d,\"host_wait\":%d,"
           "\"z\":%.4f,\"z_mean\":%.4f,\"z_free\":%.4f,\"target\":%.3f,\"sleep_ns\":%d,"
           "\"fr_rate\":%.1f,\"fr_gflops\":%.6f,\"cl_rate\":%.1f,\"cl_gflops\":%.6f,"
           "\"mean_episode_us\":%.1f,\"pulses_per_8ms\":%.2f,\"launches\":%llu,"
           "\"tach\":%d,\"track\":%d,\"track_tight\":%d,\"useful_drop\":%d,\"engaged\":%d,"
           "\"cal_swallow\":%d,\"window\":%d,\"pass\":%d}\n\n",
           g_pulses, g_honour, g_sleep_shots, g_host_wait,
           z, z_mean, z_free, target, sleep_ns,
           fr_rate, fr_gflops, cl_rate, cl_gflops,
           mean_launch_us, pulses_per_8ms, (unsigned long long)pump.launches,
           pulses_ok, track_ok, track_tight, useful_drop, engaged,
           cal_swallow, window_ok, pass);

    FILE* rf = fopen(rep, "w");
    if (rf) {
        fprintf(rf,
                "pulses_cfg=%d honour=%s shots=%d host=%d "
                "pulses=%llu z=%.4f z_mean=%.4f target=%.3f sleep=%d "
                "fr_rate=%.1f fr_gflops=%.6f cl_rate=%.1f cl_gflops=%.6f "
                "tach=%d track=%d useful_drop=%d model=%d cal_swallow=%d "
                "window=%d mean_episode_us=%.1f pulses_per_8ms=%.2f launches=%llu PASS=%d\n",
                g_pulses, g_honour, g_sleep_shots, g_host_wait,
                pulses, z, z_mean, target, sleep_ns,
                fr_rate, fr_gflops, cl_rate, cl_gflops,
                pulses_ok, track_ok, useful_drop, model_ok, cal_swallow,
                window_ok, mean_launch_us, pulses_per_8ms,
                (unsigned long long)pump.launches, pass);
        fclose(rf);
    }

    episode_shutdown(h, &pump);
    return pass ? 0 : 2;
}

static int mode_run(int seconds) {
    GPUTronicConfig cfg;
    gputronic_config_gold(&cfg);
    char csv[320];
    path_join(csv, sizeof(csv), "_run.csv");
    cfg.csv_path = csv;
    cfg.launch_selftest_kernel = 0;

    EpisodePump pump;
    GPUTronicHandle* h = episode_boot(&cfg, &pump);
    if (!h) return 1;

    printf("Episode closed-loop %ds | launch-complete GEMM (%.0f FLOP/pulse, %d pulses/launch, honour=%s)\n",
           seconds, (double)EPISODE_FLOPS_PER_PULSE, g_pulses, g_honour);
    for (int s = 0; s < seconds && !g_stop; s++) {
        usleep(1000000);
        gputronic_print_status(h, s + 1);
        double rate = (double)gputronic_get_rate(h);
        printf("       useful ≈ %.4f GFLOP/s  (from rate_ema)\n", gflops_from_rate(rate));
    }

    episode_shutdown(h, &pump);
    return 0;
}

/* =============================================================================
 * Proof demo. Same episode math, Gold frozen.
 *
 * Phase 1 — twin. Same tiles, no mapped atomic, no fence, no sleep.
 * Phase 2 — tachometer on, actuator at base. Tax is phase2 / phase1.
 * Phase 3 — open-loop stall sized to the measured launch T.
 *   Host: clock wait after sync (real wall time, not __nanosleep).
 *   Device: N sleeps of the stock 500 µs clamp. max_sleep stays 500 µs,
 *   so a single call cannot step onto the Z≈2.3 plateau.
 *   A point is usable only when wall Z = rate_free/rate is near 1.5 and a
 *   shorter stall sits lower. The setpoint has to lie on a slope.
 * Phase 4 — stock PI, stock clamp, only after phase 3. The plant maps the
 *   commanded nanoseconds onto that stall: 500 µs of command corresponds to
 *   the open-loop wait that held Z nearest 1.5. Kp, Ki, max_sleep unchanged.
 *   Target then returns to 1.0 and the one-sided law releases.
 *   A second mapping — treat the mean emitted command as full scale — was
 *   measured and dropped. The command was banging between base and the clamp,
 *   so enlarging every kick moved the mean toward 1.5 and widened the hunt.
 * ============================================================================= */

#define PROOF_SWEEP_US 700000
#define PROOF_POINT_US 1400000
#define PROOF_SETTLE_US 450000
#define PROOF_CLAMP_NS 500000
#define PROOF_MAX_TILE_STEP 8
#define PROOF_MAX_CANDIDATES 3

static volatile int g_proof_tiles = EPISODE_TILES_PER_PULSE;
static volatile int g_proof_shots = 0;
static volatile int g_proof_stretch = 0;
static volatile long long g_proof_dfull_ns = 0;
static volatile long long g_proof_extra_ns = 0;

struct ProofPump {
    GPUTronicControl* h_ctrl;
    GPUTronicControl* d_ctrl;
    float* sink;
    int num_sm;
    pthread_t thr;
    volatile int running;
    unsigned long long launches;
    double work_us_acc;
    long long wait_act_acc;
    unsigned long long wait_n;
};

static float* g_sink = NULL;
static unsigned long long* g_slot_h = NULL;
static unsigned long long* g_slot_d = NULL;
static int g_bench_sm = 0;

static long long wait_ns(long long ns) {
    struct timespec t0, t1, req, rem;
    if (ns <= 0) return 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    req.tv_sec = (time_t)(ns / 1000000000LL);
    req.tv_nsec = (long)(ns % 1000000000LL);
    while (nanosleep(&req, &rem) == -1 && errno == EINTR)
        req = rem;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);
}

/* Same 8×8×8 tiles as episode_kernel. Runtime tile count is the batch knob.
 * do_sense=0 is the twin: the math runs, the cable is not touched.
 * Tiny sleeps are not honoured — base (5 ns) must stay free-run. */
__global__ void episode_var_kernel(unsigned long long* pulses, float* sink,
                                   int num_sm, int n_pulses, int n_tiles,
                                   int sleep_shots, int sleep_ns, int do_sense) {
    const int sm = blockIdx.x;
    const int tid = threadIdx.x;
    const int r = tid / EPISODE_TILE;
    const int c = tid % EPISODE_TILE;
    float acc = 0.0f;
    unsigned int iter = 0;

    if (sm >= num_sm) return;

    __shared__ float As[EPISODE_TILE][EPISODE_TILE];
    __shared__ float Bs[EPISODE_TILE][EPISODE_TILE];

    for (int p = 0; p < n_pulses; p++) {
        for (int t = 0; t < n_tiles; t++) {
            float seed = (float)((iter * 31u + (unsigned)sm * 17u + (unsigned)tid + (unsigned)t * 13u) & 1023u);
            if (r < EPISODE_TILE && c < EPISODE_TILE) {
                As[r][c] = seed * 0.001f + 0.10f;
                Bs[r][c] = seed * 0.0007f + 0.20f;
            }
            __syncthreads();
            if (r < EPISODE_TILE && c < EPISODE_TILE) {
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < EPISODE_TILE; k++)
                    s = __fmaf_rn(As[r][k], Bs[k][c], s);
                acc += s;
            }
            __syncthreads();
            iter++;
        }
        if (acc > 1.0e30f) acc = 0.0f;
        if (do_sense && tid == 0 && pulses)
            atomicAdd(pulses, 1ULL);
    }

    if (do_sense && tid == 0 && pulses)
        __threadfence_system();
    if (do_sense && sleep_shots > 0 && sleep_ns > 1000) {
        unsigned int sn = (unsigned int)sleep_ns;
        for (int s = 0; s < sleep_shots; s++)
            __nanosleep(sn);
    }
    if (tid == 0 && sink && acc == 3.14159265f)
        sink[sm] = acc;
}

__global__ void episode_proof_kernel(GPUTronicControl* ctrl, float* sink,
                                    int num_sm, int n_pulses, int n_tiles,
                                    int sleep_shots) {
    const int sm = blockIdx.x;
    const int tid = threadIdx.x;
    const int r = tid / EPISODE_TILE;
    const int c = tid % EPISODE_TILE;
    float acc = 0.0f;
    unsigned int iter = 0;
    int sleep_ns = 0;

    if (sm >= num_sm) return;

    unsigned int flags = ctrl->control_flags;
    if (flags & GPUTRONIC_FLAG_STOP) return;
    if (flags & GPUTRONIC_FLAG_PAUSE) {
        __nanosleep(5000);
        return;
    }

    __shared__ float As[EPISODE_TILE][EPISODE_TILE];
    __shared__ float Bs[EPISODE_TILE][EPISODE_TILE];

    for (int p = 0; p < n_pulses; p++) {
        for (int t = 0; t < n_tiles; t++) {
            float seed = (float)((iter * 31u + (unsigned)sm * 17u + (unsigned)tid + (unsigned)t * 13u) & 1023u);
            if (r < EPISODE_TILE && c < EPISODE_TILE) {
                As[r][c] = seed * 0.001f + 0.10f;
                Bs[r][c] = seed * 0.0007f + 0.20f;
            }
            __syncthreads();
            if (r < EPISODE_TILE && c < EPISODE_TILE) {
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < EPISODE_TILE; k++)
                    s = __fmaf_rn(As[r][k], Bs[k][c], s);
                acc += s;
            }
            __syncthreads();
            iter++;
        }
        if (acc > 1.0e30f) acc = 0.0f;
        if (tid == 0)
            atomicAdd((unsigned long long*)&ctrl->total_work_pulses, 1ULL);
    }

    if (tid == 0) __threadfence_system();
    sleep_ns = ctrl->throttle_sleep_ns;
    if (sleep_ns < 0) sleep_ns = 0;
    if (ctrl->max_sleep_ns > 0 && sleep_ns > ctrl->max_sleep_ns)
        sleep_ns = ctrl->max_sleep_ns;
    if (sleep_shots > 0 && sleep_ns > 1000) {
        unsigned int sn = (unsigned int)sleep_ns;
        for (int s = 0; s < sleep_shots; s++)
            __nanosleep(sn);
    }
    if (tid == 0 && sink && acc == 3.14159265f)
        sink[sm] = acc;
}

static int bench_init(void) {
    cudaDeviceProp prop;
    cudaError_t err;
    if (g_sink) return 0;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[PROOF] device query failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    g_bench_sm = prop.multiProcessorCount;
    if (g_bench_sm <= 0) g_bench_sm = GPUTRONIC_DEFAULT_NUM_SM;
    err = cudaMalloc((void**)&g_sink, (size_t)g_bench_sm * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[PROOF] sink alloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaHostAlloc((void**)&g_slot_h, sizeof(unsigned long long), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "[PROOF] slot alloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaHostGetDevicePointer((void**)&g_slot_d, g_slot_h, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[PROOF] slot map failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    g_slot_h[0] = 0ULL;
    printf("[PROOF] device sm=%d name=%s\n", g_bench_sm, prop.name);
    fflush(stdout);
    return 0;
}

static double flops_of(int pulses, int tiles, unsigned long long launches, int sm) {
    return (double)launches * (double)pulses * (double)sm *
           (double)EPISODE_FLOPS_PER_TILE * (double)tiles;
}

/* Host-timed launch+sync. sense=0 twin, sense=1 mapped atomic per pulse. */
static int bench_window(int sense, int tiles, int pulses, int window_us,
                        double* gflops, double* launch_us, double* pulses_8ms) {
    unsigned long long n = 0;
    double acc_us = 0.0;
    double t_warm = now_us();
    cudaError_t err = cudaSuccess;
    while (now_us() - t_warm < 200000.0) {
        double a = now_us();
        episode_var_kernel<<<g_bench_sm, EPISODE_THREADS>>>(
            sense ? g_slot_d : NULL, g_sink, g_bench_sm, pulses, tiles, 0, 0, sense);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) break;
        if (now_us() - a > 80000.0) {
            fprintf(stderr, "[PROOF] warmup launch %.0f us — stopping this batch\n", now_us() - a);
            return -1;
        }
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "[PROOF] warmup failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    double t0 = now_us();
    double tlim = t0 + (double)window_us;
    while (now_us() < tlim) {
        double a = now_us();
        episode_var_kernel<<<g_bench_sm, EPISODE_THREADS>>>(
            sense ? g_slot_d : NULL, g_sink, g_bench_sm, pulses, tiles, 0, 0, sense);
        err = cudaDeviceSynchronize();
        double b = now_us();
        if (err != cudaSuccess) {
            fprintf(stderr, "[PROOF] bench launch failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
        acc_us += (b - a);
        n++;
        if (n == 1 && (b - a) > 80000.0) {
            fprintf(stderr, "[PROOF] launch %.0f us — stopping this batch\n", b - a);
            break;
        }
    }
    double wall_s = (now_us() - t0) * 1e-6;
    if (wall_s < 1e-6) wall_s = 1e-6;
    double flops = flops_of(pulses, tiles, n, g_bench_sm);
    *gflops = flops / wall_s / 1.0e9;
    *launch_us = (n > 0) ? (acc_us / (double)n) : 0.0;
    *pulses_8ms = (double)n * (double)pulses * (double)g_bench_sm / wall_s * 0.008;
    return (n > 0) ? 0 : -1;
}

static void* proof_pump_thread(void* arg) {
    ProofPump* p = (ProofPump*)arg;
    cudaSetDevice(0);
    while (!g_stop && p->running) {
        unsigned int flags = p->h_ctrl->control_flags;
        int shots;
        long long extra;
        int stretch;
        long long dfull;
        if (flags & GPUTRONIC_FLAG_STOP) break;
        if (flags & GPUTRONIC_FLAG_PAUSE) {
            usleep(200);
            continue;
        }
        shots = g_proof_shots;
        extra = g_proof_extra_ns;
        stretch = g_proof_stretch;
        dfull = g_proof_dfull_ns;
        double t0 = now_us();
        episode_proof_kernel<<<p->num_sm, EPISODE_THREADS>>>(
            p->d_ctrl, p->sink, p->num_sm, g_pulses, g_proof_tiles, shots);
        cudaError_t err = cudaDeviceSynchronize();
        if (stretch && dfull > 0) {
            int cmd = p->h_ctrl->throttle_sleep_ns;
            long long d;
            if (cmd < 0) cmd = 0;
            if (cmd > PROOF_CLAMP_NS) cmd = PROOF_CLAMP_NS;
            d = (long long)((double)cmd / (double)PROOF_CLAMP_NS * (double)dfull);
            if (d > 0) {
                long long act = wait_ns(d);
                p->wait_act_acc += act;
                p->wait_n++;
            }
        } else if (extra > 0) {
            long long act = wait_ns(extra);
            p->wait_act_acc += act;
            p->wait_n++;
        }
        p->work_us_acc += now_us() - t0;
        p->launches++;
        if (err != cudaSuccess) {
            fprintf(stderr, "[PROOF] launch/sync failed: %s\n", cudaGetErrorString(err));
            break;
        }
    }
    p->running = 0;
    return NULL;
}

static int proof_pump_start(ProofPump* p, GPUTronicHandle* h) {
    memset(p, 0, sizeof(*p));
    p->h_ctrl = gputronic_get_control(h);
    if (!p->h_ctrl) return -1;
    if (cudaHostGetDevicePointer((void**)&p->d_ctrl, p->h_ctrl, 0) != cudaSuccess || !p->d_ctrl)
        return -1;
    p->sink = g_sink;
    p->num_sm = p->h_ctrl->num_sm > 0 ? p->h_ctrl->num_sm : g_bench_sm;
    p->running = 1;
    if (pthread_create(&p->thr, NULL, proof_pump_thread, p) != 0) {
        p->running = 0;
        return -1;
    }
    return 0;
}

static void proof_pump_stop(ProofPump* p) {
    if (!p) return;
    p->running = 0;
    if (p->h_ctrl) p->h_ctrl->control_flags |= GPUTRONIC_FLAG_STOP;
    if (p->thr) pthread_join(p->thr, NULL);
    p->thr = 0;
}

struct SweepRow {
    int tiles;
    double twin_gflops;
    double sense_gflops;
    double ratio;
    double twin_launch_us;
    double sense_launch_us;
    double pulses_8ms;
    int ok;
};

struct ThrowPoint {
    int kind; /* 1 = host wait, 2 = device N × 500 µs */
    double frac;
    int shots;
    long long req_ns;
    double act_us;
    double z_wall;
    double z_hat;
    double gflops;
    double rate;
};

static int proof_window(GPUTronicHandle* h, ProofPump* pump, int settle_us, int window_us,
                        double* rate, double* gflops, double* z_hat, double* launch_us,
                        double* act_wait_us) {
    GPUTronicControl* c;
    unsigned long long l0, p0, wn0;
    double wacc0, t0, dt, z_acc;
    int samples = 10;
    int k;
    if (settle_us > 0) usleep((useconds_t)settle_us);
    c = gputronic_get_control(h);
    l0 = pump->launches;
    p0 = c->total_work_pulses;
    wacc0 = (double)pump->work_us_acc;
    wn0 = pump->wait_n;
    {
        long long w0 = pump->wait_act_acc;
        t0 = now_us();
        z_acc = 0.0;
        for (k = 0; k < samples; k++) {
            usleep((useconds_t)(window_us / samples));
            z_acc += (double)c->z_hat;
        }
        dt = (now_us() - t0) * 1e-6;
        if (dt < 1e-6) dt = 1e-6;
        *rate = (double)(c->total_work_pulses - p0) / dt;
        *gflops = (*rate) * (double)EPISODE_FLOPS_PER_TILE * (double)g_proof_tiles / 1.0e9;
        *z_hat = z_acc / (double)samples;
        {
            unsigned long long dl = pump->launches - l0;
            *launch_us = (dl > 0) ? ((pump->work_us_acc - wacc0) / (double)dl) : 0.0;
        }
        if (act_wait_us) {
            unsigned long long wn = pump->wait_n - wn0;
            *act_wait_us = (wn > 0) ? ((double)(pump->wait_act_acc - w0) / (double)wn) / 1.0e3 : 0.0;
        }
    }
    return 0;
}

static GPUTronicHandle* proof_boot(ProofPump* pump, int tiles, int path_us) {
    GPUTronicConfig cfg;
    GPUTronicHandle* h;
    gputronic_config_gold(&cfg);
    cfg.launch_selftest_kernel = 0;
    cfg.max_sleep_ns = PROOF_CLAMP_NS;
    /* Stock path is 10 µs. A longer path still feeds the 8 ms window; it
     * only changes how often the host re-reads the mapped counter. */
    if (path_us >= 10) cfg.control_dt_us = path_us;
    cfg.csv_path = "results/episode_proof_gold.csv";
    g_proof_tiles = tiles;
    g_proof_shots = 0;
    g_proof_stretch = 0;
    g_proof_dfull_ns = 0;
    g_proof_extra_ns = 0;
    h = gputronic_create(&cfg);
    if (!h) return NULL;
    if (proof_pump_start(pump, h) != 0) {
        gputronic_destroy(h);
        return NULL;
    }
    if (gputronic_start(h) != 0) {
        proof_pump_stop(pump);
        gputronic_destroy(h);
        return NULL;
    }
    return h;
}

static void proof_shutdown(GPUTronicHandle* h, ProofPump* pump) {
    proof_pump_stop(pump);
    if (h) {
        gputronic_stop(h);
        gputronic_destroy(h);
    }
}

static int point_near(const struct ThrowPoint* p) {
    return p->z_wall > 1.35 && p->z_wall < 1.65;
}

/* Shorter stall on the same curve must sit clearly lower. */
static int point_graded(const struct ThrowPoint* pts, int n, int idx) {
    int i;
    double best = -1.0;
    if (idx < 0 || pts[idx].z_wall < 1.35 || pts[idx].z_wall > 1.65) return 0;
    for (i = 0; i < n; i++) {
        if (pts[i].kind != pts[idx].kind) continue;
        if (i == idx) continue;
        if (pts[i].kind == 1 && pts[i].frac >= pts[idx].frac) continue;
        if (pts[i].kind == 2 && pts[i].shots >= pts[idx].shots) continue;
        if (pts[i].z_wall > best) best = pts[i].z_wall;
    }
    if (best < 0.0) return 0;
    return (pts[idx].z_wall - best) >= 0.08;
}

static int mode_proof(void) {
    struct SweepRow rows[PROOF_MAX_TILE_STEP];
    int n_rows = 0;
    int tiles_ladder[PROOF_MAX_TILE_STEP] = {32, 128, 512, 2048, 8192, 32768, 131072, 262144};
    int i, c;
    int best_i = -1;
    double best_gflops = -1.0;
    double best_ratio = -1.0;
    int cand[PROOF_MAX_CANDIDATES];
    int n_cand = 0;
    FILE* csv;
    FILE* rep;
    int tax_ok = 0;
    int throw_ok = 0;
    int loop_ok = 0;
    int released_ok = 0;
    int winner = -1;
    int law = 0; /* 1 host clamp-scaled, 2 host reached-scaled, 3 device */
    int path_us = 10;
    double twin_g = 0, sense_g = 0, full_g = 0, tax = 0;
    double fr_rate = 0, fr_z = 0, fr_launch = 0;
    double cl_z_wall = 0, cl_z_hat = 0, cl_gflops = 0, cl_sleep = 0;
    double rel_gflops = 0, rel_z = 0, rel_sleep = 0;
    double dfull_us = 0, cmd_mean = 0;
    struct ThrowPoint host_pts[8];
    struct ThrowPoint dev_pts[10];
    int n_host = 0, n_dev = 0;
    int host_pick = -1, dev_pick = -1;
    const double fracs[] = {0.0, 0.25, 0.50, 0.75, 1.00, 1.25};
    const int n_fracs = 6;
    char law_name[64] = "none";

    if (bench_init() != 0) return 1;
    printf("\n=== EPISODE PROOF ===\n");
    printf("Stock driver, Gold frozen, Kp=%.2f Ki=%.2f clamp=%d ns, pulses/launch=%d\n\n",
           GPUTRONIC_GOLD_KP, GPUTRONIC_GOLD_KI, PROOF_CLAMP_NS, g_pulses);
    fflush(stdout);

    csv = fopen("results/episode_proof_sweep.csv", "w");
    if (csv) fprintf(csv, "tiles,twin_gflops,sense_gflops,ratio,twin_launch_us,sense_launch_us,pulses_per_8ms\n");

    printf("[PHASE 1-2 sweep] twin vs mapped tachometer, no Gold thread\n");
    printf("%8s %12s %12s %8s %12s %12s\n", "tiles", "twin", "sense", "ratio", "T_twin_us", "pulse/8ms");
    for (i = 0; i < PROOF_MAX_TILE_STEP; i++) {
        double tg, sg, tlaunch, slaunch, p8;
        int tiles = tiles_ladder[i];
        if (bench_window(0, tiles, g_pulses, PROOF_SWEEP_US, &tg, &tlaunch, &p8) != 0) break;
        if (tlaunch > 60000.0) {
            printf("%8d  launch %.0f us — past the useful batch\n", tiles, tlaunch);
            break;
        }
        if (bench_window(1, tiles, g_pulses, PROOF_SWEEP_US, &sg, &slaunch, &p8) != 0) break;
        rows[n_rows].tiles = tiles;
        rows[n_rows].twin_gflops = tg;
        rows[n_rows].sense_gflops = sg;
        rows[n_rows].ratio = (tg > 1e-6) ? (sg / tg) : 0.0;
        rows[n_rows].twin_launch_us = tlaunch;
        rows[n_rows].sense_launch_us = slaunch;
        rows[n_rows].pulses_8ms = p8;
        rows[n_rows].ok = (rows[n_rows].ratio >= 0.90 && p8 >= 100.0);
        printf("%8d %12.2f %12.2f %8.3f %12.1f %12.1f%s\n",
               tiles, tg, sg, rows[n_rows].ratio, tlaunch, p8,
               rows[n_rows].ok ? "  tax-ok" : "");
        fflush(stdout);
        if (csv) fprintf(csv, "%d,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f\n",
                         tiles, tg, sg, rows[n_rows].ratio, tlaunch, slaunch, p8);
        if (rows[n_rows].ratio > best_ratio) best_ratio = rows[n_rows].ratio;
        if (rows[n_rows].ok && tg > best_gflops) {
            best_gflops = tg;
            best_i = n_rows;
        }
        n_rows++;
        if (tlaunch > 20000.0 && rows[n_rows - 1].ratio >= 0.90) break;
        if (n_rows >= 2 && tg < rows[n_rows - 2].twin_gflops * 0.70 && rows[n_rows - 2].twin_gflops > 50.0)
            break;
    }
    if (csv) fclose(csv);

    if (n_rows == 0) {
        fprintf(stderr, "[PROOF] sweep produced no rows\n");
        return 1;
    }
    if (best_i < 0) {
        /* No row cleared 0.90. Keep the best ratio that still feeds the window. */
        best_ratio = -1.0;
        for (i = 0; i < n_rows; i++) {
            if (rows[i].pulses_8ms < 50.0) continue;
            if (rows[i].ratio > best_ratio) {
                best_ratio = rows[i].ratio;
                best_i = i;
            }
        }
        if (best_i < 0) best_i = n_rows - 1;
    }

    /* Fastest tax-ok batches first. The walk stops at the first one whose
     * open-loop host wait can sit near Z=1.5. */
    for (i = 0; i < n_rows; i++) {
        int j;
        int pick = -1;
        double peak = -1.0;
        for (j = 0; j < n_rows; j++) {
            int seen = 0;
            int k;
            if (!rows[j].ok) continue;
            for (k = 0; k < n_cand; k++)
                if (cand[k] == j) seen = 1;
            if (seen) continue;
            if (rows[j].twin_gflops > peak) {
                peak = rows[j].twin_gflops;
                pick = j;
            }
        }
        if (pick < 0) break;
        cand[n_cand++] = pick;
        if (n_cand >= PROOF_MAX_CANDIDATES) break;
    }
    if (n_cand == 0) {
        cand[n_cand++] = best_i;
    }

    printf("\n[PHASE 3] host-wait grid on up to %d batches, fastest tax-ok first\n", n_cand);
    fflush(stdout);

    for (c = 0; c < n_cand && !throw_ok; c++) {
        ProofPump pump;
        GPUTronicHandle* h;
        double rate, gflops, zhat, launch_us, act_us;
        double T_us;
        int fi;
        double fresh_twin = 0.0, fresh_launch = 0.0, fresh_p8 = 0.0;
        memset(&pump, 0, sizeof(pump));
        printf("\n[BATCH] tiles=%d sweep twin=%.2f sense=%.2f ratio=%.3f\n",
               rows[cand[c]].tiles, rows[cand[c]].twin_gflops,
               rows[cand[c]].sense_gflops, rows[cand[c]].ratio);
        fflush(stdout);
        /* Twin measured immediately before the Gold boot, same window. */
        if (bench_window(0, rows[cand[c]].tiles, g_pulses, PROOF_POINT_US,
                         &fresh_twin, &fresh_launch, &fresh_p8) != 0) {
            fprintf(stderr, "[PROOF] fresh twin failed\n");
            continue;
        }
        h = proof_boot(&pump, rows[cand[c]].tiles, 10);
        if (!h) {
            fprintf(stderr, "[PROOF] boot failed\n");
            continue;
        }
        /* Cal at base, then force open-loop zero so phase 2 is sleep 0. */
        usleep(2500000);
        gputronic_set_open_loop_sleep(h, 0);
        proof_window(h, &pump, PROOF_SETTLE_US, PROOF_POINT_US,
                     &rate, &gflops, &zhat, &launch_us, &act_us);
        T_us = launch_us;
        fr_rate = rate;
        fr_z = zhat;
        fr_launch = launch_us;
        full_g = gflops;
        twin_g = fresh_twin;
        sense_g = rows[cand[c]].sense_gflops;
        tax = (twin_g > 1e-6) ? (full_g / twin_g) : 0.0;
        path_us = 10;
        printf("[PHASE 2] path=10us  full %.2f GFLOP/s  twin %.2f  Z_hat=%.3f  T=%.1f us  tax=%.3f\n",
               full_g, twin_g, fr_z, T_us, tax);
        fflush(stdout);
        /* The 10 µs path re-reads the mapped counter far faster than the
         * 8 ms window uses. On this PCIe 3.0 link that read is the tax.
         * 250 µs is still ~32 samples per window. Kp, Ki, clamp unchanged. */
        if (tax < 0.90) {
            printf("[PATH] tax %.3f at 10 us. Retry with control_dt=250 us\n", tax);
            fflush(stdout);
            proof_shutdown(h, &pump);
            memset(&pump, 0, sizeof(pump));
            h = proof_boot(&pump, rows[cand[c]].tiles, 250);
            if (!h) continue;
            usleep(2500000);
            gputronic_set_open_loop_sleep(h, 0);
            proof_window(h, &pump, PROOF_SETTLE_US, PROOF_POINT_US,
                         &rate, &gflops, &zhat, &launch_us, &act_us);
            T_us = launch_us;
            fr_rate = rate;
            fr_z = zhat;
            fr_launch = launch_us;
            full_g = gflops;
            tax = (twin_g > 1e-6) ? (full_g / twin_g) : 0.0;
            path_us = 250;
            printf("[PHASE 2] path=250us full %.2f GFLOP/s  twin %.2f  Z_hat=%.3f  T=%.1f us  tax=%.3f\n",
                   full_g, twin_g, fr_z, T_us, tax);
            fflush(stdout);
        }
        if (tax < 0.90 && (c + 1) < n_cand) {
            printf("[PATH] tax still %.3f — try a heavier batch\n", tax);
            proof_shutdown(h, &pump);
            continue;
        }
        if (T_us < 50.0) {
            printf("[PHASE 3] launch too short to size a wait (T=%.1f us)\n", T_us);
            proof_shutdown(h, &pump);
            continue;
        }

        n_host = 0;
        g_proof_shots = 0;
        g_proof_stretch = 0;
        for (fi = 0; fi < n_fracs && n_host < 8; fi++) {
            long long req = (long long)(fracs[fi] * T_us * 1000.0);
            double z_wall;
            g_proof_extra_ns = req;
            proof_window(h, &pump, PROOF_SETTLE_US, PROOF_POINT_US,
                         &rate, &gflops, &zhat, &launch_us, &act_us);
            z_wall = (rate > 1.0) ? (fr_rate / rate) : 0.0;
            host_pts[n_host].kind = 1;
            host_pts[n_host].frac = fracs[fi];
            host_pts[n_host].shots = 0;
            host_pts[n_host].req_ns = req;
            host_pts[n_host].act_us = act_us;
            host_pts[n_host].z_wall = z_wall;
            host_pts[n_host].z_hat = zhat;
            host_pts[n_host].gflops = gflops;
            host_pts[n_host].rate = rate;
            printf("[HOST] D/T=%.2f req=%.0f us act=%.0f us  Z_wall=%.3f Z_hat=%.3f  %.2f GFLOP/s\n",
                   fracs[fi], (double)req / 1.0e3, act_us, z_wall, zhat, gflops);
            fflush(stdout);
            n_host++;
        }
        g_proof_extra_ns = 0;

        host_pick = -1;
        {
            double best_e = 1e9;
            for (fi = 0; fi < n_host; fi++) {
                double e = fabs(host_pts[fi].z_wall - 1.5);
                if (e < best_e) {
                    best_e = e;
                    host_pick = fi;
                }
            }
            if (host_pick >= 0 && !point_graded(host_pts, n_host, host_pick))
                host_pick = -1;
            else if (host_pick >= 0 && !point_near(&host_pts[host_pick]))
                host_pick = -1;
        }

        /* Device quanta at the stock clamp. Recorded on every candidate so the
         * report shows whether N × 500 µs accumulates. Scored only if host missed. */
        n_dev = 0;
        gputronic_set_open_loop_sleep(h, PROOF_CLAMP_NS);
        for (fi = 1; fi <= 48 && n_dev < 10; fi *= 2) {
            double z_wall;
            if ((double)fi * 500.0 > T_us * 2.0 && fi > 4) break;
            g_proof_shots = fi;
            proof_window(h, &pump, PROOF_SETTLE_US, PROOF_POINT_US,
                         &rate, &gflops, &zhat, &launch_us, &act_us);
            z_wall = (rate > 1.0) ? (fr_rate / rate) : 0.0;
            dev_pts[n_dev].kind = 2;
            dev_pts[n_dev].frac = 0.0;
            dev_pts[n_dev].shots = fi;
            dev_pts[n_dev].req_ns = (long long)fi * (long long)PROOF_CLAMP_NS;
            dev_pts[n_dev].act_us = 0.0;
            dev_pts[n_dev].z_wall = z_wall;
            dev_pts[n_dev].z_hat = zhat;
            dev_pts[n_dev].gflops = gflops;
            dev_pts[n_dev].rate = rate;
            printf("[DEV] N=%d × 500 us  Z_wall=%.3f Z_hat=%.3f  %.2f GFLOP/s  T_now=%.0f us\n",
                   fi, z_wall, zhat, gflops, launch_us);
            fflush(stdout);
            n_dev++;
            if (z_wall > 2.2) break;
        }
        g_proof_shots = 0;
        gputronic_set_open_loop_sleep(h, 0);

        dev_pick = -1;
        {
            double best_e = 1e9;
            for (fi = 0; fi < n_dev; fi++) {
                double e = fabs(dev_pts[fi].z_wall - 1.5);
                if (e < best_e) {
                    best_e = e;
                    dev_pick = fi;
                }
            }
            if (dev_pick >= 0 && !point_graded(dev_pts, n_dev, dev_pick))
                dev_pick = -1;
            else if (dev_pick >= 0 && !point_near(&dev_pts[dev_pick]))
                dev_pick = -1;
        }

        if (host_pick >= 0 || dev_pick >= 0) {
            int track_s = 12;
            int rel_s = 6;
            int s;
            double sleep_acc = 0.0;
            int sleep_n = 0;
            double p0_rate_anchor = fr_rate;
            unsigned long long p_a, p_b;
            double t_a, t_b;
            winner = cand[c];
            throw_ok = 1;
            tax_ok = (tax >= 0.90);
            /* Prefer the wall-time slope. Device N is the fallback. */
            if (host_pick >= 0) {
                law = 1;
                snprintf(law_name, sizeof(law_name), "host clamp-scaled");
                g_proof_dfull_ns = host_pts[host_pick].req_ns;
                dfull_us = (double)host_pts[host_pick].req_ns / 1.0e3;
                g_proof_stretch = 1;
                g_proof_shots = 0;
                g_proof_extra_ns = 0;
            } else {
                law = 3;
                snprintf(law_name, sizeof(law_name), "device N of live cmd");
                g_proof_shots = dev_pts[dev_pick].shots;
                g_proof_stretch = 0;
                g_proof_dfull_ns = 0;
                dfull_us = 500.0 * (double)dev_pts[dev_pick].shots;
            }
            printf("\n[PHASE 3] hold  Z_wall=%.3f via %s\n",
                   (law == 3) ? dev_pts[dev_pick].z_wall : host_pts[host_pick].z_wall,
                   law_name);
            printf("[PHASE 4] stock PI for %ds, clamp %d ns, then target 1.0 for %ds\n",
                   track_s, PROOF_CLAMP_NS, rel_s);
            fflush(stdout);

            gputronic_set_target(h, GPUTRONIC_GOLD_TARGET_Z);
            gputronic_set_open_loop_sleep(h, -1);
            p_a = 0;
            t_a = 0.0;
            for (s = 0; s < track_s && !g_stop; s++) {
                usleep(1000000);
                gputronic_print_status(h, s + 1);
                if (s == 3) {
                    p_a = gputronic_get_control(h)->total_work_pulses;
                    t_a = now_us();
                }
                if (s >= 3) {
                    sleep_acc += (double)gputronic_get_sleep_ns(h);
                    sleep_n++;
                }
                fflush(stdout);
            }
            p_b = gputronic_get_control(h)->total_work_pulses;
            t_b = now_us();
            {
                double dt = (t_b - t_a) * 1e-6;
                double rate_cl = (dt > 1e-6) ? (double)(p_b - p_a) / dt : 0.0;
                cl_z_wall = (rate_cl > 1.0) ? (p0_rate_anchor / rate_cl) : 0.0;
                cl_gflops = rate_cl * (double)EPISODE_FLOPS_PER_TILE * (double)g_proof_tiles / 1.0e9;
            }
            cl_z_hat = (double)gputronic_get_z(h);
            cmd_mean = (sleep_n > 0) ? (sleep_acc / (double)sleep_n) : 0.0;
            cl_sleep = cmd_mean;
            printf("[PHASE 4] Z_wall=%.3f Z_hat=%.3f  %.2f GFLOP/s  mean_cmd=%.0f ns\n",
                   cl_z_wall, cl_z_hat, cl_gflops, cmd_mean);
            fflush(stdout);

            gputronic_set_target(h, 1.0f);
            printf("[RELEASE] target 1.0 — one-sided law back to base\n");
            fflush(stdout);
            sleep_acc = 0.0;
            sleep_n = 0;
            p_a = gputronic_get_control(h)->total_work_pulses;
            t_a = now_us();
            for (s = 0; s < rel_s && !g_stop; s++) {
                usleep(1000000);
                gputronic_print_status(h, s + 1);
                if (s == 2) {
                    p_a = gputronic_get_control(h)->total_work_pulses;
                    t_a = now_us();
                }
                if (s >= 2) {
                    sleep_acc += (double)gputronic_get_sleep_ns(h);
                    sleep_n++;
                }
            }
            p_b = gputronic_get_control(h)->total_work_pulses;
            t_b = now_us();
            {
                double dt = (t_b - t_a) * 1e-6;
                double rate_rel = (dt > 1e-6) ? (double)(p_b - p_a) / dt : 0.0;
                rel_gflops = rate_rel * (double)EPISODE_FLOPS_PER_TILE * (double)g_proof_tiles / 1.0e9;
                rel_z = (double)gputronic_get_z(h);
            }
            rel_sleep = (sleep_n > 0) ? (sleep_acc / (double)sleep_n) : 0.0;
            printf("[RELEASE] %.2f GFLOP/s  Z_hat=%.3f  mean_cmd=%.0f ns  (free-run was %.2f)\n",
                   rel_gflops, rel_z, rel_sleep, full_g);
            fflush(stdout);

            loop_ok = (cl_z_wall >= 1.30 && cl_z_wall <= 1.70);
            released_ok = (full_g > 1.0 && rel_gflops > full_g * 0.90 && rel_sleep < 50000.0);
            proof_shutdown(h, &pump);
            break;
        }

        printf("[PHASE 3] this batch has no graded point near Z=1.5\n");
        proof_shutdown(h, &pump);
    }

    if (!throw_ok) {
        tax_ok = (best_i >= 0 && rows[best_i].ok);
        twin_g = rows[best_i].twin_gflops;
        sense_g = rows[best_i].sense_gflops;
        tax = rows[best_i].ratio;
    }

    rep = fopen("results/episode_proof_report.txt", "w");
    if (rep) {
        fprintf(rep, "GPUTronic Episode Proof\n");
        fprintf(rep, "gold_frozen=1 kp=%.2f ki=%.2f clamp_ns=%d path_us=%d pulses=%d\n",
                GPUTRONIC_GOLD_KP, GPUTRONIC_GOLD_KI, PROOF_CLAMP_NS, path_us, g_pulses);
        fprintf(rep, "tax_ok=%d throw_ok=%d loop_ok=%d release_ok=%d law=%s\n",
                tax_ok, throw_ok, loop_ok, released_ok, law_name);
        if (winner >= 0)
            fprintf(rep, "tiles=%d twin=%.4f sense=%.4f full_free=%.4f tax_full=%.4f T_us=%.1f Zfree=%.4f\n",
                    rows[winner].tiles, twin_g, sense_g, full_g, tax, fr_launch, fr_z);
        else if (best_i >= 0)
            fprintf(rep, "tiles=%d twin=%.4f sense=%.4f ratio=%.4f (no throw)\n",
                    rows[best_i].tiles, twin_g, sense_g, tax);
        fprintf(rep, "closed Z_wall=%.4f Z_hat=%.4f gflops=%.4f cmd_ns=%.0f dfull_us=%.1f\n",
                cl_z_wall, cl_z_hat, cl_gflops, cl_sleep, dfull_us);
        fprintf(rep, "release gflops=%.4f Z_hat=%.4f cmd_ns=%.0f\n", rel_gflops, rel_z, rel_sleep);
        fprintf(rep, "sweep:\n");
        for (i = 0; i < n_rows; i++)
            fprintf(rep, "  tiles=%d twin=%.3f sense=%.3f ratio=%.3f T=%.1f p8=%.1f ok=%d\n",
                    rows[i].tiles, rows[i].twin_gflops, rows[i].sense_gflops, rows[i].ratio,
                    rows[i].twin_launch_us, rows[i].pulses_8ms, rows[i].ok);
        fprintf(rep, "host:\n");
        for (i = 0; i < n_host; i++)
            fprintf(rep, "  frac=%.2f req_us=%.0f act_us=%.0f Z=%.4f Zhat=%.4f gflops=%.3f\n",
                    host_pts[i].frac, (double)host_pts[i].req_ns / 1e3, host_pts[i].act_us,
                    host_pts[i].z_wall, host_pts[i].z_hat, host_pts[i].gflops);
        fprintf(rep, "device:\n");
        for (i = 0; i < n_dev; i++)
            fprintf(rep, "  N=%d Z=%.4f Zhat=%.4f gflops=%.3f\n",
                    dev_pts[i].shots, dev_pts[i].z_wall, dev_pts[i].z_hat, dev_pts[i].gflops);
        fclose(rep);
    }

    printf("\n=== EPISODE PROOF RESULT ===\n");
    if (winner >= 0) {
        printf("  tiles/pulse     %d\n", rows[winner].tiles);
        printf("  phase 1 twin    %.2f GFLOP/s\n", twin_g);
        printf("  phase 2 tach    %.2f GFLOP/s   tax %.3f   path %d us   Z_hat %.3f   %s\n",
               full_g, tax, path_us, fr_z, tax_ok ? "CHEAP" : "TAX HIGH");
    } else if (best_i >= 0) {
        printf("  best sensor     tiles=%d twin %.2f sense %.2f ratio %.3f\n",
               rows[best_i].tiles, twin_g, sense_g, tax);
    }
    printf("  phase 3 throw   %s\n", throw_ok ? "GRADED NEAR 1.50" : "NO GRADED HOLD");
    if (throw_ok)
        printf("  phase 4 loop    Z_wall %.3f  %s  %s\n",
               cl_z_wall, law_name, loop_ok ? "HOLD" : "MISSED");
    if (throw_ok)
        printf("  release         %.2f GFLOP/s (free-run %.2f)  %s\n",
               rel_gflops, full_g, released_ok ? "RETURNED" : "DID NOT RETURN");
    if (tax_ok && throw_ok && loop_ok && released_ok)
        printf("DEMO: phases 1-4\n");
    else if (throw_ok && loop_ok && released_ok)
        printf("DEMO: loop held and released, free-run tax %.3f is still above a tenth\n", tax);
    else if (throw_ok)
        printf("DEMO: open-loop hold only — closed loop did not sit and release\n");
    else if (best_i >= 0 && rows[best_i].ok)
        printf("DEMO: sensor is cheap, setpoint is not on a slope\n");
    else
        printf("DEMO: sensor tax not yet boring\n");
    printf("report: results/episode_proof_report.txt\n");
    fflush(stdout);

    if (g_sink) cudaFree(g_sink);
    g_sink = NULL;
    if (g_slot_h) cudaFreeHost(g_slot_h);
    g_slot_h = NULL;
    g_slot_d = NULL;
    return (tax_ok && throw_ok && loop_ok && released_ok) ? 0 : 2;
}

static void usage(const char* argv0) {
    printf("GPUTronic Episode Plant (Gold 1.0 launch-complete GEMM)\n\n");
    printf("Usage:\n");
    printf("  %s check           # dyno + closed-loop episode gate\n", argv0);
    printf("  %s dyno            # sleep → tile/s + GFLOP/s\n", argv0);
    printf("  %s run [sec]       # closed-loop on the episode plant\n", argv0);
    printf("  %s closedloop [s]  # closed-loop only\n", argv0);
    printf("  %s proof           # twin, tax, open-loop hold, stock PI, release\n", argv0);
    printf("\nEnv: GPUTRONIC_EPISODE_PULSES GPUTRONIC_EPISODE_HONOUR={once,host,chunkN}\n");
}

int main(int argc, char** argv) {
    episode_load_env();
    const char* mode = (argc >= 2) ? argv[1] : "check";
    if (!strcmp(mode, "-h") || !strcmp(mode, "--help") || !strcmp(mode, "help")) {
        usage(argv[0]);
        return 0;
    }
    if (!strcmp(mode, "proof")) return mode_proof();
    if (!strcmp(mode, "dyno")) return mode_dyno();
    if (!strcmp(mode, "closedloop")) {
        int sec = (argc >= 3) ? atoi(argv[2]) : g_cl_sec;
        if (sec <= 0) sec = g_cl_sec;
        return mode_closedloop(sec);
    }
    if (!strcmp(mode, "check")) {
        int rd = mode_dyno();
        int rc = mode_closedloop(g_cl_sec);
        printf("=== EPISODE GATE: dyno=%s closedloop=%s ===\n",
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
