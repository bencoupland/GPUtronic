/* =============================================================================
 * GPUTronic Show: one cooperative kernel, one strip chart
 *
 * Phases: full-speed rate, SM-count map, the same step again, stock
 * controller, release back to full speed. Full speed (all SMs) is the
 * fastest this interface gets.
 * ============================================================================= */
#include "gputronic.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define SHOW_TILE 8
#define SHOW_THREADS 64
#define SHOW_TILES 8192
#define SHOW_FLOPS_PER_TILE (2 * SHOW_TILE * SHOW_TILE * SHOW_TILE)
#define SHOW_MAX_SAMPLES 800

enum {
    PH_CAL = 0,
    PH_CEIL,
    PH_MAP,
    PH_AGAIN,
    PH_LOOP,
    PH_RELEASE
};

static const char* phase_name[] = {
    "CAL", "CEILING", "MAP", "AGAIN", "LOOP", "RELEASE"
};

struct Sample {
    double t;
    int n;
    int phase;
    double z;
    double gflops;
};

/* ---- Kernels ---- */

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
}

__global__ void show_twin(unsigned long long* pulses, int num_sm, int tiles, volatile int* run) {
    const int sm = blockIdx.x;
    const int tid = threadIdx.x;
    const int r = tid / SHOW_TILE;
    const int c = tid % SHOW_TILE;
    float acc = 0.0f;
    unsigned int iter = 0;
    unsigned long long local_n = 0;
    if (sm >= num_sm) return;
    __shared__ float As[SHOW_TILE][SHOW_TILE];
    __shared__ float Bs[SHOW_TILE][SHOW_TILE];
    while (*run) {
        for (int t = 0; t < tiles; t++) {
            float seed = (float)((iter * 31u + (unsigned)sm * 17u + (unsigned)tid + (unsigned)t * 13u) & 1023u);
            if (r < SHOW_TILE && c < SHOW_TILE) {
                As[r][c] = seed * 0.001f + 0.10f;
                Bs[r][c] = seed * 0.0007f + 0.20f;
            }
            __syncthreads();
            if (r < SHOW_TILE && c < SHOW_TILE) {
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < SHOW_TILE; k++)
                    s = __fmaf_rn(As[r][k], Bs[k][c], s);
                acc += s;
            }
            __syncthreads();
            iter++;
        }
        if (acc > 1.0e30f) acc = 0.0f;
        local_n++;
    }
    if (tid == 0) pulses[sm] = local_n;
    if (tid == 0 && acc == 3.14159265f) pulses[sm] = local_n;
}

__global__ void show_kernel(GPUTronicControl* ctrl, int num_sm, int tiles) {
    const int sm = blockIdx.x;
    const int tid = threadIdx.x;
    const int r = tid / SHOW_TILE;
    const int c = tid % SHOW_TILE;
    float acc = 0.0f;
    unsigned int iter = 0;
    unsigned int local_pulses = 0;
    if (sm >= num_sm) return;
    __shared__ float As[SHOW_TILE][SHOW_TILE];
    __shared__ float Bs[SHOW_TILE][SHOW_TILE];
    while (true) {
        unsigned int flags = ctrl->control_flags;
        if (flags & GPUTRONIC_FLAG_STOP) break;
        if (flags & GPUTRONIC_FLAG_PAUSE) {
            (void)(5000);   /* was __nanosleep: removed, NEXT.md A4 */
            continue;
        }
        int active = ctrl->active_sms;
        if (active <= 0 || active > num_sm) active = num_sm;
        if (active < 4) active = 4;
        if (sm >= active) {
            (void)(5000);   /* was __nanosleep: removed, NEXT.md A4 */
            continue;
        }
        for (int t = 0; t < tiles; t++) {
            float seed = (float)((iter * 31u + (unsigned)sm * 17u + (unsigned)tid + (unsigned)t * 13u) & 1023u);
            if (r < SHOW_TILE && c < SHOW_TILE) {
                As[r][c] = seed * 0.001f + 0.10f;
                Bs[r][c] = seed * 0.0007f + 0.20f;
            }
            __syncthreads();
            if (r < SHOW_TILE && c < SHOW_TILE) {
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < SHOW_TILE; k++)
                    s = __fmaf_rn(As[r][k], Bs[k][c], s);
                acc += s;
            }
            __syncthreads();
            iter++;
        }
        if (acc > 1.0e30f) acc = 0.0f;
        if (tid == 0) {
            atomicAdd((unsigned long long*)&ctrl->total_work_pulses, 1ULL);
            local_pulses++;
            if ((local_pulses & 31u) == 0u) __threadfence_system();
        }
    }
    if (tid == 0 && acc == 3.14159265f)
        atomicAdd((unsigned long long*)&ctrl->total_work_pulses, 0ULL);
}

/* ---- Sampling and chart output ---- */

static int sample_at(GPUTronicHandle* h, struct Sample* s, double t0, int phase,
                     unsigned long long* p_prev, double* t_prev) {
    GPUTronicControl* c = gputronic_get_control(h);
    double t = now_us();
    unsigned long long p = c->total_work_pulses;
    double dt = (t - *t_prev) * 1e-6;
    double rate = (dt > 1e-4) ? (double)(p - *p_prev) / dt : 0.0;
    s->t = (t - t0) * 1e-6;
    s->n = c->active_sms;
    s->phase = phase;
    s->z = (double)c->z_hat;
    s->gflops = rate * (double)SHOW_FLOPS_PER_TILE * (double)SHOW_TILES / 1.0e9;
    *p_prev = p;
    *t_prev = t;
    printf("  %6.1fs  %-8s  n=%3d  Z=%5.2f  %7.1f GFLOP/s\n",
           s->t, phase_name[phase], s->n, s->z, s->gflops);
    fflush(stdout);
    return 0;
}

static void hold(GPUTronicHandle* h, struct Sample* sam, int* nsam, double t0,
                 int phase, double seconds, unsigned long long* p_prev, double* t_prev) {
    double end = now_us() + seconds * 1e6;
    while (now_us() < end && *nsam < SHOW_MAX_SAMPLES) {
        usleep(200000);
        sample_at(h, &sam[*nsam], t0, phase, p_prev, t_prev);
        (*nsam)++;
    }
}

static double mean_gflops(const struct Sample* s, int n, int phase) {
    double a = 0.0;
    int k = 0;
    int i;
    for (i = 0; i < n; i++) {
        if (s[i].phase != phase) continue;
        if (s[i].gflops < 1.0) continue;
        a += s[i].gflops;
        k++;
    }
    return (k > 0) ? a / k : 0.0;
}

static double mean_z(const struct Sample* s, int n, int phase) {
    double a = 0.0;
    int k = 0;
    int i;
    for (i = 0; i < n; i++) {
        if (s[i].phase != phase || s[i].z < 0.2) continue;
        a += s[i].z;
        k++;
    }
    return (k > 0) ? a / k : 0.0;
}

static void phase_span(const struct Sample* s, int n, int phase, double* t0, double* t1) {
    int i, seen = 0;
    *t0 = 0;
    *t1 = 0;
    for (i = 0; i < n; i++) {
        if (s[i].phase != phase) continue;
        if (!seen) {
            *t0 = s[i].t;
            seen = 1;
        }
        *t1 = s[i].t;
    }
}

static void write_chart(const struct Sample* s, int n, double twin_gflops,
                        int n_star, double z_star, double z_again,
                        const char* path_svg, const char* path_html) {
    double tmax = (n > 0) ? s[n - 1].t : 1.0;
    double zmax = 1.6, gmax = 1.0;
    int i;
    FILE* fp;
    const int W = 1100, H = 760;
    const int L = 64, R = 24, T = 48;
    const int plot_w = W - L - R;
    const int plot_h = 168;
    const int gap = 36;
    int ph;
    if (tmax < 1.0) tmax = 1.0;
    for (i = 0; i < n; i++) {
        if (s[i].z > zmax && s[i].z < 4.0) zmax = s[i].z;
        if (s[i].gflops > gmax) gmax = s[i].gflops;
    }
    zmax *= 1.08;
    gmax *= 1.08;

    fp = fopen(path_svg, "w");
    if (!fp) return;
    fprintf(fp, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n",
            W, H, W, H);
    fprintf(fp, "<rect width=\"100%%\" height=\"100%%\" fill=\"#101114\"/>\n");
    fprintf(fp, "<text x=\"64\" y=\"32\" fill=\"#e8e4dc\" font-family=\"ui-sans-serif,sans-serif\" font-size=\"18\">GPUTronic — aggregate rate</text>\n");
    fprintf(fp, "<text x=\"64\" y=\"50\" fill=\"#8d897f\" font-family=\"ui-sans-serif,sans-serif\" font-size=\"12\">free-run is the ceiling · width is the command · Z = rate_ref / rate</text>\n");

    for (ph = 0; ph < 6; ph++) {
        double a, b;
        double x0, x1;
        const char* band[] = {"#16171b", "#141a18", "#1c1914", "#1a1c14", "#14181c", "#16141a"};
        phase_span(s, n, ph, &a, &b);
        if (b <= a) continue;
        x0 = L + (a / tmax) * plot_w;
        x1 = L + (b / tmax) * plot_w;
        fprintf(fp, "<rect x=\"%.1f\" y=\"%d\" width=\"%.1f\" height=\"%d\" fill=\"%s\"/>\n",
                x0, T + 16, x1 - x0, plot_h * 3 + gap * 2, band[ph]);
        fprintf(fp, "<text x=\"%.1f\" y=\"%d\" fill=\"#6e6a62\" font-family=\"ui-sans-serif,sans-serif\" font-size=\"11\">%s</text>\n",
                x0 + 6, T + 30, phase_name[ph]);
    }

    /* three panels: Z, GFLOP/s, n */
    {
        const char* ylab[] = {"Z", "GFLOP/s", "SMs"};
        double ymax[] = {zmax, gmax, 84.0};
        const char* col[] = {"#e0b15a", "#8fb9c4", "#d7d3cb"};
        int panel;
        for (panel = 0; panel < 3; panel++) {
            int y0 = T + 16 + panel * (plot_h + gap);
            int gy;
            fprintf(fp, "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"none\" stroke=\"#2a2c31\"/>\n",
                    L, y0, plot_w, plot_h);
            for (gy = 0; gy <= 4; gy++) {
                double v = ymax[panel] * gy / 4.0;
                double y = y0 + plot_h - (v / ymax[panel]) * plot_h;
                fprintf(fp, "<line x1=\"%d\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\" stroke=\"#22242a\"/>\n",
                        L, y, L + plot_w, y);
                fprintf(fp, "<text x=\"%d\" y=\"%.1f\" fill=\"#6e6a62\" font-family=\"ui-monospace,monospace\" font-size=\"10\" text-anchor=\"end\">%.2f</text>\n",
                        L - 6, y + 3, v);
            }
            fprintf(fp, "<text x=\"%d\" y=\"%d\" fill=\"%s\" font-family=\"ui-sans-serif,sans-serif\" font-size=\"12\">%s</text>\n",
                    L + 8, y0 + 16, col[panel], ylab[panel]);
            if (panel == 0) {
                double y = y0 + plot_h - (1.5 / ymax[0]) * plot_h;
                fprintf(fp, "<line x1=\"%d\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\" stroke=\"#e0b15a\" stroke-dasharray=\"4 4\" stroke-opacity=\"0.7\"/>\n",
                        L, y, L + plot_w, y);
                fprintf(fp, "<text x=\"%d\" y=\"%.1f\" fill=\"#e0b15a\" font-family=\"ui-monospace,monospace\" font-size=\"10\">1.50</text>\n",
                        L + plot_w - 36, y - 4);
            }
            fprintf(fp, "<polyline fill=\"none\" stroke=\"%s\" stroke-width=\"1.6\" points=\"", col[panel]);
            for (i = 0; i < n; i++) {
                double v = (panel == 0) ? s[i].z : (panel == 1) ? s[i].gflops : (double)s[i].n;
                double x, y;
                if (panel == 0 && v > ymax[0]) v = ymax[0];
                if (v < 0) v = 0;
                x = L + (s[i].t / tmax) * plot_w;
                y = y0 + plot_h - (v / ymax[panel]) * plot_h;
                fprintf(fp, "%.1f,%.1f ", x, y);
            }
            fprintf(fp, "\"/>\n");
        }
    }

    fprintf(fp, "<text x=\"64\" y=\"%d\" fill=\"#8d897f\" font-family=\"ui-sans-serif,sans-serif\" font-size=\"12\">twin %.0f GFLOP/s · ceiling %.0f · map n=%d Z=%.2f · again Z=%.2f · loop Z=%.2f · release %.0f</text>\n",
            H - 28,
            twin_gflops,
            mean_gflops(s, n, PH_CEIL),
            n_star, z_star, z_again,
            mean_z(s, n, PH_LOOP),
            mean_gflops(s, n, PH_RELEASE));
    fprintf(fp, "</svg>\n");
    fclose(fp);

    fp = fopen(path_html, "w");
    if (!fp) return;
    fprintf(fp, "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>GPUTronic</title>\n");
    fprintf(fp, "<style>body{margin:0;background:#101114;color:#e8e4dc;font-family:ui-sans-serif,sans-serif;}");
    fprintf(fp, "main{max-width:1100px;margin:32px auto;padding:0 16px;} p{color:#8d897f;line-height:1.5;}</style></head><body><main>\n");
    fprintf(fp, "<object data=\"gputronic_show.svg\" type=\"image/svg+xml\" width=\"1100\"></object>\n");
    fprintf(fp, "<p>Free-run is the ceiling. The width is the command. The loop holds a band under 1.50 and then gives the chip back. ");
    fprintf(fp, "Twin %.0f GFLOP/s. Ceiling %.0f. The map step nearest 1.50 was n=%d at Z=%.2f, and the repeat was Z=%.2f. ",
            twin_gflops, mean_gflops(s, n, PH_CEIL), n_star, z_star, z_again);
    fprintf(fp, "Closed loop mean Z=%.2f. Release %.0f GFLOP/s.</p>\n",
            mean_z(s, n, PH_LOOP), mean_gflops(s, n, PH_RELEASE));
    fprintf(fp, "</main></body></html>\n");
    fclose(fp);
}

/* ---- Phases and sweep ---- */

static double run_twin(int num_sm, int tiles, double seconds) {
    volatile int* run_h = NULL;
    volatile int* run_d = NULL;
    unsigned long long* dev = NULL;
    unsigned long long* host = NULL;
    double t0, t1, gflops = 0.0;
    unsigned long long sum = 0;
    int i;
    if (cudaHostAlloc((void**)&run_h, sizeof(int), cudaHostAllocMapped) != cudaSuccess) return 0;
    if (cudaMalloc((void**)&dev, (size_t)num_sm * sizeof(unsigned long long)) != cudaSuccess) return 0;
    cudaHostGetDevicePointer((void**)&run_d, (void*)run_h, 0);
    cudaMemset(dev, 0, (size_t)num_sm * sizeof(unsigned long long));
    *run_h = 1;
    show_twin<<<num_sm, SHOW_THREADS>>>(dev, num_sm, tiles, run_d);
    t0 = now_us();
    usleep((useconds_t)(seconds * 1e6));
    *run_h = 0;
    cudaDeviceSynchronize();
    t1 = now_us();
    host = (unsigned long long*)calloc((size_t)num_sm, sizeof(unsigned long long));
    cudaMemcpy(host, dev, (size_t)num_sm * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    for (i = 0; i < num_sm; i++) sum += host[i];
    gflops = (double)sum * (double)SHOW_FLOPS_PER_TILE * (double)tiles / ((t1 - t0) * 1e-6) / 1.0e9;
    free(host);
    cudaFree(dev);
    cudaFreeHost((void*)run_h);
    return gflops;
}

static GPUTronicHandle* boot_show(int num_sm, int tiles) {
    GPUTronicConfig cfg;
    GPUTronicHandle* h;
    GPUTronicControl* hc;
    GPUTronicControl* dc = NULL;
    gputronic_config_gold(&cfg);
    cfg.launch_selftest_kernel = 0;
    cfg.csv_path = "results/gputronic_sweep_gold.csv";
    h = gputronic_create(&cfg);
    if (!h) return NULL;
    hc = gputronic_get_control(h);
    if (cudaHostGetDevicePointer((void**)&dc, hc, 0) != cudaSuccess || !dc) return NULL;
    show_kernel<<<num_sm, SHOW_THREADS>>>(dc, num_sm, tiles);
    if (gputronic_start(h) != 0) return NULL;
    gputronic_set_open_loop_sms(h, 0);
    usleep(3400000);
    return h;
}

static void shutdown_show(GPUTronicHandle* h) {
    GPUTronicControl* c;
    if (!h) return;
    c = gputronic_get_control(h);
    if (c) c->control_flags |= GPUTRONIC_FLAG_STOP;
    gputronic_stop(h);
    gputronic_destroy(h);
}

static void one_window(GPUTronicHandle* h, int tiles, int settle_us, int window_us,
                      double* gflops, double* z, int* n_out) {
    GPUTronicControl* c = gputronic_get_control(h);
    unsigned long long p0, p1;
    double t0, dt;
    if (settle_us > 0) usleep((useconds_t)settle_us);
    p0 = c->total_work_pulses;
    t0 = now_us();
    usleep((useconds_t)window_us);
    p1 = c->total_work_pulses;
    dt = (now_us() - t0) * 1e-6;
    if (dt < 1e-4) dt = 1e-4;
    *gflops = (double)(p1 - p0) / dt * (double)SHOW_FLOPS_PER_TILE * (double)tiles / 1.0e9;
    *z = (double)c->z_hat;
    if (n_out) *n_out = c->active_sms;
}

static int sweep_main(int num_sm) {
    const int tiles_list[] = {256, 1024, 4096, 8192, 16384, 32768, 65536};
    const int ntiles = 7;
    double twin_g[8], cable_g[8], ratio[8];
    int ti, knee = 4;
    double best_cable = -1.0;
    FILE* csv;
    FILE* svg;
    int sweep_tiles[2];
    double z_at[2][128];
    double g_at[2][128];
    int n_star[2];
    double z_star[2];
    int pass;

    printf("\nTILES    saturation of work per pulse, full width\n");
    printf("%8s %12s %12s %8s\n", "tiles", "twin", "cable", "ratio");
    csv = fopen("results/gputronic_sweep_tiles.csv", "w");
    if (csv) fprintf(csv, "tiles,twin_gflops,cable_gflops,ratio\n");
    for (ti = 0; ti < ntiles; ti++) {
        GPUTronicHandle* h;
        double z;
        int nnow;
        twin_g[ti] = run_twin(num_sm, tiles_list[ti], 1.2);
        h = boot_show(num_sm, tiles_list[ti]);
        if (!h) return 1;
        one_window(h, tiles_list[ti], 200000, 800000, &cable_g[ti], &z, &nnow);
        shutdown_show(h);
        ratio[ti] = (twin_g[ti] > 1.0) ? cable_g[ti] / twin_g[ti] : 0.0;
        printf("%8d %12.1f %12.1f %8.3f\n", tiles_list[ti], twin_g[ti], cable_g[ti], ratio[ti]);
        fflush(stdout);
        if (csv) fprintf(csv, "%d,%.4f,%.4f,%.4f\n", tiles_list[ti], twin_g[ti], cable_g[ti], ratio[ti]);
        if (cable_g[ti] > best_cable) {
            best_cable = cable_g[ti];
            knee = ti;
        }
    }
    if (csv) fclose(csv);
    printf("cable ceiling at %d tiles/pulse\n", tiles_list[knee]);

    /* 1024 is still paying the cable. The knee is where more tiles stop buying rate. */
    sweep_tiles[0] = 1024;
    sweep_tiles[1] = tiles_list[knee];
    if (sweep_tiles[1] == 1024) sweep_tiles[1] = 32768;

    csv = fopen("results/gputronic_sweep_sms.csv", "w");
    if (csv) fprintf(csv, "tiles,n,gflops,z\n");
    for (pass = 0; pass < 2; pass++) {
        GPUTronicHandle* h = boot_show(num_sm, sweep_tiles[pass]);
        int n;
        double best_e = 1e9;
        if (!h) return 1;
        n_star[pass] = num_sm;
        z_star[pass] = 0;
        printf("\nWIDTH    every SM, %d tiles/pulse\n", sweep_tiles[pass]);
        printf("%6s %12s %8s\n", "n", "GFLOP/s", "Z");
        for (n = num_sm; n >= 4; n--) {
            double g, z;
            gputronic_set_open_loop_sms(h, n);
            one_window(h, sweep_tiles[pass], 180000, 420000, &g, &z, NULL);
            g_at[pass][n] = g;
            z_at[pass][n] = z;
            if (fabs(z - 1.5) < best_e && z > 0.5 && z < 9.5) {
                best_e = fabs(z - 1.5);
                n_star[pass] = n;
                z_star[pass] = z;
            }
            if (n == num_sm || n % 8 == 0 || n == n_star[pass] || n <= 8)
                printf("%6d %12.1f %8.2f\n", n, g, z);
            if (csv) fprintf(csv, "%d,%d,%.4f,%.4f\n", sweep_tiles[pass], n, g, z);
        }
        printf("nearest 1.50: n=%d Z=%.2f\n", n_star[pass], z_star[pass]);
        fflush(stdout);
        shutdown_show(h);
    }
    if (csv) fclose(csv);

    svg = fopen("results/gputronic_sweep.svg", "w");
    if (svg) {
        const int W = 1100, H = 760, L = 72, plot_w = 980, plot_h = 260;
        double gmax = 1.0;
        int ti2, n, y0;
        const char* cols[] = {"#e0b15a", "#8fb9c4"};
        fprintf(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n", W, H, W, H);
        fprintf(svg, "<rect width=\"100%%\" height=\"100%%\" fill=\"#101114\"/>\n");
        fprintf(svg, "<text x=\"72\" y=\"36\" fill=\"#e8e4dc\" font-family=\"ui-sans-serif,sans-serif\" font-size=\"18\">Two axes, two calibrations</text>\n");
        fprintf(svg, "<text x=\"72\" y=\"56\" fill=\"#8d897f\" font-family=\"ui-sans-serif,sans-serif\" font-size=\"12\">grey twin, blue cable · gold is %d tiles/pulse, blue is %d</text>\n",
                sweep_tiles[0], sweep_tiles[1]);
        for (ti2 = 0; ti2 < ntiles; ti2++) {
            if (twin_g[ti2] > gmax) gmax = twin_g[ti2];
            if (cable_g[ti2] > gmax) gmax = cable_g[ti2];
        }
        gmax *= 1.08;
        y0 = 80;
        fprintf(svg, "<text x=\"80\" y=\"%d\" fill=\"#8d897f\" font-size=\"12\" font-family=\"ui-sans-serif,sans-serif\">GFLOP/s vs tiles/pulse, 84 SMs</text>\n", y0 + 18);
        fprintf(svg, "<polyline fill=\"none\" stroke=\"#6e6a62\" stroke-width=\"1.6\" points=\"");
        for (ti2 = 0; ti2 < ntiles; ti2++) {
            double x = L + (double)ti2 / (double)(ntiles - 1) * plot_w;
            double y = y0 + plot_h - (twin_g[ti2] / gmax) * plot_h;
            fprintf(svg, "%.1f,%.1f ", x, y);
        }
        fprintf(svg, "\"/>\n");
        fprintf(svg, "<polyline fill=\"none\" stroke=\"#8fb9c4\" stroke-width=\"1.6\" points=\"");
        for (ti2 = 0; ti2 < ntiles; ti2++) {
            double x = L + (double)ti2 / (double)(ntiles - 1) * plot_w;
            double y = y0 + plot_h - (cable_g[ti2] / gmax) * plot_h;
            fprintf(svg, "%.1f,%.1f ", x, y);
        }
        fprintf(svg, "\"/>\n");
        for (ti2 = 0; ti2 < ntiles; ti2++) {
            double x = L + (double)ti2 / (double)(ntiles - 1) * plot_w;
            fprintf(svg, "<text x=\"%.0f\" y=\"%d\" fill=\"#6e6a62\" font-size=\"10\" font-family=\"ui-monospace,monospace\" text-anchor=\"middle\">%d</text>\n",
                    x, y0 + plot_h + 16, tiles_list[ti2]);
        }
        y0 = 420;
        fprintf(svg, "<text x=\"80\" y=\"%d\" fill=\"#e0b15a\" font-size=\"12\" font-family=\"ui-sans-serif,sans-serif\">Z vs working SMs</text>\n", y0 + 18);
        fprintf(svg, "<line x1=\"%d\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\" stroke=\"#e0b15a\" stroke-dasharray=\"4 4\" stroke-opacity=\"0.6\"/>\n",
                L, y0 + plot_h - (1.5 / 4.0) * plot_h, L + plot_w, y0 + plot_h - (1.5 / 4.0) * plot_h);
        for (pass = 0; pass < 2; pass++) {
            fprintf(svg, "<polyline fill=\"none\" stroke=\"%s\" stroke-width=\"1.6\" points=\"", cols[pass]);
            for (n = 4; n <= num_sm; n++) {
                double zv = z_at[pass][n];
                double x, y;
                if (zv > 4.0) zv = 4.0;
                if (zv < 0.0) zv = 0.0;
                x = L + (double)(n - 4) / (double)(num_sm - 4) * plot_w;
                y = y0 + plot_h - (zv / 4.0) * plot_h;
                fprintf(svg, "%.1f,%.1f ", x, y);
            }
            fprintf(svg, "\"/>\n");
        }
        fprintf(svg, "<text x=\"72\" y=\"%d\" fill=\"#8d897f\" font-size=\"12\" font-family=\"ui-sans-serif,sans-serif\">%d tiles → n=%d at Z %.2f · %d tiles → n=%d at Z %.2f</text>\n",
                H - 24, sweep_tiles[0], n_star[0], z_star[0], sweep_tiles[1], n_star[1], z_star[1]);
        fprintf(svg, "</svg>\n");
        fclose(svg);
    }
    printf("\ncsv  results/gputronic_sweep_tiles.csv  results/gputronic_sweep_sms.csv\n");
    return 0;
}

/* ---- Command-line entry ---- */

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    int num_sm, errc;
    volatile int* run_h = NULL;
    volatile int* run_d = NULL;
    unsigned long long* twin_d = NULL;
    unsigned long long* twin_h = NULL;
    double twin_gflops = 0.0;
    double t_twin0, t_twin1;
    GPUTronicConfig cfg;
    GPUTronicHandle* h;
    GPUTronicControl* hc;
    GPUTronicControl* dc = NULL;
    struct Sample sam[SHOW_MAX_SAMPLES];
    int nsam = 0;
    double t0, tprev;
    unsigned long long pprev;
    const int map_n[] = {84, 72, 64, 56, 48, 32, 16};
    int mi, n_star = 64;
    double best_err = 1e9, z_star = 0, z_again = 0;
    double step_z[7];
    cudaError_t err;

    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "no device: %s\n", cudaGetErrorString(err));
        return 1;
    }
    num_sm = prop.multiProcessorCount;
    if (num_sm <= 0) num_sm = 84;
    if (argc > 1 && strcmp(argv[1], "sweep") == 0) return sweep_main(num_sm);
    printf("\nGPUTronic show  ·  %s  ·  %d SMs  ·  %d tiles/pulse\n", prop.name, num_sm, SHOW_TILES);
    printf("Free-run is the ceiling. Width is the command.\n\n");
    fflush(stdout);

    err = cudaHostAlloc((void**)&run_h, sizeof(int), cudaHostAllocMapped);
    errc = cudaMalloc((void**)&twin_d, (size_t)num_sm * sizeof(unsigned long long));
    if (err != cudaSuccess || errc != cudaSuccess) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    cudaHostGetDevicePointer((void**)&run_d, (void*)run_h, 0);
    cudaMemset(twin_d, 0, (size_t)num_sm * sizeof(unsigned long long));
    *run_h = 1;
    printf("TWIN     uninstrumented, same tiles\n");
    fflush(stdout);
    show_twin<<<num_sm, SHOW_THREADS>>>(twin_d, num_sm, SHOW_TILES, run_d);
    t_twin0 = now_us();
    sleep(2);
    *run_h = 0;
    cudaDeviceSynchronize();
    t_twin1 = now_us();
    twin_h = (unsigned long long*)calloc((size_t)num_sm, sizeof(unsigned long long));
    cudaMemcpy(twin_h, twin_d, (size_t)num_sm * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    {
        unsigned long long sum = 0;
        int si;
        for (si = 0; si < num_sm; si++) sum += twin_h[si];
        twin_gflops = (double)sum * (double)SHOW_FLOPS_PER_TILE * (double)SHOW_TILES /
                      ((t_twin1 - t_twin0) * 1e-6) / 1.0e9;
    }
    free(twin_h);
    printf("         %.1f GFLOP/s\n\n", twin_gflops);
    fflush(stdout);

    gputronic_config_gold(&cfg);
    cfg.launch_selftest_kernel = 0;
    cfg.csv_path = "results/gputronic_show_gold.csv";
    h = gputronic_create(&cfg);
    if (!h) return 1;
    hc = gputronic_get_control(h);
    if (hc && hc->num_sm > 0) num_sm = hc->num_sm;
    err = cudaHostGetDevicePointer((void**)&dc, hc, 0);
    if (err != cudaSuccess || !dc) {
        fprintf(stderr, "control map failed\n");
        return 1;
    }
    show_kernel<<<num_sm, SHOW_THREADS>>>(dc, num_sm, SHOW_TILES);
    if (gputronic_start(h) != 0) {
        fprintf(stderr, "start failed\n");
        return 1;
    }
    gputronic_set_open_loop_sms(h, 0);
    t0 = now_us();
    pprev = hc->total_work_pulses;
    tprev = t0;
    printf("CAL      free-run window, then half the SMs, then the scale\n");
    hold(h, sam, &nsam, t0, PH_CAL, 3.6, &pprev, &tprev);

    printf("\nCEILING  all %d SMs\n", num_sm);
    gputronic_set_open_loop_sms(h, 0);
    hold(h, sam, &nsam, t0, PH_CEIL, 4.0, &pprev, &tprev);

    printf("\nMAP      width steps\n");
    for (mi = 0; mi < 7; mi++) {
        int n = map_n[mi];
        int begin, j;
        double zacc = 0.0;
        int zk = 0;
        if (n > num_sm) n = num_sm;
        if (n < 4) n = 4;
        gputronic_set_open_loop_sms(h, n);
        begin = nsam;
        hold(h, sam, &nsam, t0, PH_MAP, 2.4, &pprev, &tprev);
        for (j = begin + (nsam - begin) / 2; j < nsam; j++) {
            zacc += sam[j].z;
            zk++;
        }
        step_z[mi] = (zk > 0) ? zacc / zk : 0.0;
        if (fabs(step_z[mi] - 1.5) < best_err) {
            best_err = fabs(step_z[mi] - 1.5);
            n_star = n;
            z_star = step_z[mi];
        }
        printf("         step n=%d  Z=%.2f\n", n, step_z[mi]);
    }

    printf("\nAGAIN    n=%d, the step nearest 1.50\n", n_star);
    gputronic_set_open_loop_sms(h, n_star);
    {
        int begin = nsam, j, zk = 0;
        double zacc = 0.0;
        hold(h, sam, &nsam, t0, PH_AGAIN, 3.0, &pprev, &tprev);
        for (j = begin + (nsam - begin) / 2; j < nsam; j++) {
            zacc += sam[j].z;
            zk++;
        }
        z_again = (zk > 0) ? zacc / zk : 0.0;
    }

    printf("\nLOOP     stock PI, target 1.50\n");
    gputronic_set_target(h, GPUTRONIC_GOLD_TARGET_Z);
    gputronic_set_open_loop_sms(h, -1);
    hold(h, sam, &nsam, t0, PH_LOOP, 10.0, &pprev, &tprev);

    printf("\nRELEASE  target 1.0\n");
    gputronic_set_target(h, 1.0f);
    hold(h, sam, &nsam, t0, PH_RELEASE, 5.0, &pprev, &tprev);

    hc->control_flags |= GPUTRONIC_FLAG_STOP;
    gputronic_stop(h);
    gputronic_destroy(h);
    cudaFree(twin_d);
    cudaFreeHost((void*)run_h);

    write_chart(sam, nsam, twin_gflops, n_star, z_star, z_again,
                "results/gputronic_show.svg", "results/gputronic_show.html");

    printf("\n────────────────────────────────────────\n");
    printf("twin        %7.1f GFLOP/s\n", twin_gflops);
    printf("ceiling     %7.1f GFLOP/s   Z %.2f   tax %.2f\n",
           mean_gflops(sam, nsam, PH_CEIL), mean_z(sam, nsam, PH_CEIL),
           (twin_gflops > 1.0) ? mean_gflops(sam, nsam, PH_CEIL) / twin_gflops : 0.0);
    printf("map         n=%d  Z %.2f\n", n_star, z_star);
    printf("again       n=%d  Z %.2f\n", n_star, z_again);
    printf("loop        Z %.2f\n", mean_z(sam, nsam, PH_LOOP));
    printf("release     %7.1f GFLOP/s   Z %.2f\n",
           mean_gflops(sam, nsam, PH_RELEASE), mean_z(sam, nsam, PH_RELEASE));
    printf("chart       results/gputronic_show.html\n");
    fflush(stdout);
    return 0;
}
