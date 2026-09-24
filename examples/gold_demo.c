/* GPUTronic Gold — watchable closed-loop embed demo
 *
 * Self-test plant only (Gold persistent kernel). Not a FLOP benchmark.
 * Protocol: ~8 s free-run (open-loop sleep=0, includes F-06 cal),
 * then ~22 s one-sided PI tracking target Z=1.50.
 *
 *   make demo && ./build/gputronic_demo
 */
#include "gputronic.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define FREE_S 8.0
#define TRACK_S 22.0
#define TICK_US 200000
#define BAR_W 20
#define Z_BAR_LO 0.50f
#define Z_BAR_HI 2.50f
#define LIVE_CSV "results/gputronic_gold_watch.csv"
#define LOOP_CSV "results/gputronic_gold_watch_loop.csv"
#define REPORT_PATH "results/gputronic_gold_watch_report.txt"

enum { PHASE_FREE = 0, PHASE_TRACK = 1 };

struct Sample {
    double t;
    int phase;
    float z_hat;
    float z_raw;
    float rate;
    int sleep_ns;
    int hz;
    unsigned long long pulses;
};

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void bar_z(char* out, int n, float z, float target) {
    int i;
    int fill, tgt;
    if (n < 8) {
        out[0] = '\0';
        return;
    }
    if (z < Z_BAR_LO) z = Z_BAR_LO;
    if (z > Z_BAR_HI) z = Z_BAR_HI;
    fill = (int)((z - Z_BAR_LO) / (Z_BAR_HI - Z_BAR_LO) * (float)BAR_W + 0.5f);
    tgt = (int)((target - Z_BAR_LO) / (Z_BAR_HI - Z_BAR_LO) * (float)BAR_W + 0.5f);
    if (fill < 0) fill = 0;
    if (fill > BAR_W) fill = BAR_W;
    if (tgt < 0) tgt = 0;
    if (tgt > BAR_W - 1) tgt = BAR_W - 1;
    out[0] = '[';
    for (i = 0; i < BAR_W; i++) {
        char c = (i < fill) ? '#' : '.';
        if (i == tgt) c = '|';
        out[1 + i] = c;
    }
    out[1 + BAR_W] = ']';
    out[2 + BAR_W] = '\0';
}

static void window_stats(const struct Sample* s, int n, int phase, double last_s,
                         double* z_mean, double* pulse_s, double* sleep_mean,
                         int* n_used) {
    int i;
    int i0 = -1, i1 = -1;
    double z_acc = 0.0;
    double sl_acc = 0.0;
    int k = 0;
    double t_end, t_cut;
    if (n <= 1) {
        *z_mean = 0.0;
        *pulse_s = 0.0;
        *sleep_mean = 0.0;
        *n_used = 0;
        return;
    }
    t_end = -1.0;
    for (i = 0; i < n; i++) {
        if (s[i].phase == phase) t_end = s[i].t;
    }
    if (t_end < 0.0) {
        *z_mean = 0.0;
        *pulse_s = 0.0;
        *sleep_mean = 0.0;
        *n_used = 0;
        return;
    }
    t_cut = t_end - last_s;
    for (i = 0; i < n; i++) {
        if (s[i].phase != phase) continue;
        if (s[i].t < t_cut) continue;
        if (i0 < 0) i0 = i;
        i1 = i;
        z_acc += (double)s[i].z_hat;
        sl_acc += (double)s[i].sleep_ns;
        k++;
    }
    *n_used = k;
    *z_mean = (k > 0) ? z_acc / (double)k : 0.0;
    *sleep_mean = (k > 0) ? sl_acc / (double)k : 0.0;
    if (i0 >= 0 && i1 > i0) {
        double dt = s[i1].t - s[i0].t;
        if (dt < 1e-6) dt = 1e-6;
        *pulse_s = (double)(s[i1].pulses - s[i0].pulses) / dt;
    } else {
        *pulse_s = 0.0;
    }
}

static void write_report(FILE* fp, int zc_ok, double z_fr, double rate_fr, double sl_fr,
                         int n_fr, double z_cl, double rate_cl, double sl_cl, int n_cl,
                         float target) {
    double err = z_cl - (double)target;
    if (err < 0.0) err = -err;
    double drop = 0.0;
    if (rate_fr > 1.0) drop = 1.0 - rate_cl / rate_fr;
    fprintf(fp, "============================================================\n");
    fprintf(fp, "GPUTronic Gold 1.0 — watch report (self-test plant)\n");
    fprintf(fp, "============================================================\n");
    fprintf(fp, "Cable:  zero-copy mapped  %s\n", zc_ok ? "YES" : "NO");
    fprintf(fp, "Loop:   %s  (control-path rows)\n", LOOP_CSV);
    fprintf(fp, "Live:   %s  (200 ms samples)\n", LIVE_CSV);
    fprintf(fp, "Law:    one-sided PI  target Z=%.2f  Kp=%.2f  Ki=%.2f\n",
            target, GPUTRONIC_GOLD_KP, GPUTRONIC_GOLD_KI);
    fprintf(fp, "\n");
    fprintf(fp, "Free-run (last 3.0 s of open-loop sleep=0, after cal)\n");
    fprintf(fp, "  samples     %d\n", n_fr);
    fprintf(fp, "  Z_hat mean  %.3f\n", z_fr);
    fprintf(fp, "  pulse/s     %.0f\n", rate_fr);
    fprintf(fp, "  sleep mean  %.0f ns\n", sl_fr);
    fprintf(fp, "\n");
    fprintf(fp, "Closed-loop (last 8.0 s, target Z=%.2f)\n", target);
    fprintf(fp, "  samples     %d\n", n_cl);
    fprintf(fp, "  Z_hat mean  %.3f   |Z-target|=%.3f\n", z_cl, err);
    fprintf(fp, "  pulse/s     %.0f\n", rate_cl);
    fprintf(fp, "  sleep mean  %.0f ns\n", sl_cl);
    if (rate_fr > 1.0)
        fprintf(fp, "  useful drop %.1f%%  (pulse/s vs free-run)\n", 100.0 * drop);
    else
        fprintf(fp, "  useful drop n/a  (free-run rate missing)\n");
    fprintf(fp, "\n");
    fprintf(fp, "Not a FLOP benchmark. Q moved because Z was commanded.\n");
    fprintf(fp, "Z = rate_ref / rate after sustained-window cal. Free-run ~ 1.0.\n");
    fprintf(fp, "============================================================\n");
}

int main(void) {
    GPUTronicConfig cfg;
    GPUTronicHandle* gov;
    FILE* live;
    FILE* report;
    struct GPUTronicControl* ctrl;
    struct Sample samples[200];
    int n = 0;
    int phase = PHASE_FREE;
    int switched = 0;
    double t0, t;
    float target;
    char bar[BAR_W + 8];
    double z_fr, rate_fr, sl_fr, z_cl, rate_cl, sl_cl;
    int n_fr, n_cl;
    int zc_ok;

    gputronic_config_gold(&cfg);
    cfg.csv_path = LOOP_CSV;
    cfg.target_z = GPUTRONIC_GOLD_TARGET_Z;
    target = cfg.target_z;

    gov = gputronic_create(&cfg);
    if (!gov) return 1;
    if (gputronic_start(gov) != 0) {
        gputronic_destroy(gov);
        return 1;
    }

    zc_ok = gputronic_zero_copy_ok(gov);
    gputronic_set_open_loop_sleep(gov, 0);

    live = fopen(LIVE_CSV, "w");
    if (live)
        fprintf(live, "time_s,phase,z_raw,z_hat,rate_ema,sleep_ns,ctrl_hz,pulses\n");

    printf("\n");
    printf("Watch: Gold self-test  |  %.0fs free-run  then  %.0fs track Z=%.2f\n",
           FREE_S, TRACK_S, target);
    printf("Bar:  %.2f .. %.2f   '|' = target   '#' = Z_hat\n", Z_BAR_LO, Z_BAR_HI);
    printf("------------------------------------------------------------\n");
    fflush(stdout);

    t0 = now_s();
    for (;;) {
        t = now_s() - t0;
        if (t >= FREE_S + TRACK_S) break;

        if (!switched && t >= FREE_S) {
            gputronic_set_open_loop_sleep(gov, -1);
            phase = PHASE_TRACK;
            switched = 1;
            printf("--- closed-loop ON  target Z=%.2f ---\n", target);
            fflush(stdout);
        }

        ctrl = gputronic_get_control(gov);
        if (ctrl && n < (int)(sizeof(samples) / sizeof(samples[0]))) {
            struct Sample s;
            s.t = t;
            s.phase = phase;
            s.z_hat = ctrl->z_hat;
            s.z_raw = ctrl->z_raw;
            s.rate = ctrl->rate_ema;
            s.sleep_ns = ctrl->throttle_sleep_ns;
            s.hz = ctrl->control_hz_ema;
            s.pulses = (unsigned long long)ctrl->total_work_pulses;
            samples[n++] = s;
            bar_z(bar, (int)sizeof(bar), s.z_hat, target);
            printf(" t=%5.1fs  %-5s  Z=%5.3f %s  rate=%9.0f  sleep=%7d  hz~%d\n",
                   s.t, phase == PHASE_FREE ? "FREE" : "TRACK", s.z_hat, bar, s.rate,
                   s.sleep_ns, s.hz);
            fflush(stdout);
            if (live)
                fprintf(live, "%.3f,%s,%.5f,%.5f,%.1f,%d,%d,%llu\n", s.t,
                        phase == PHASE_FREE ? "free" : "track", s.z_raw, s.z_hat, s.rate,
                        s.sleep_ns, s.hz, s.pulses);
        }
        usleep(TICK_US);
    }

    if (live) {
        fflush(live);
        fclose(live);
    }

    window_stats(samples, n, PHASE_FREE, 3.0, &z_fr, &rate_fr, &sl_fr, &n_fr);
    window_stats(samples, n, PHASE_TRACK, 8.0, &z_cl, &rate_cl, &sl_cl, &n_cl);

    printf("\n");
    write_report(stdout, zc_ok, z_fr, rate_fr, sl_fr, n_fr, z_cl, rate_cl, sl_cl, n_cl,
                 target);

    report = fopen(REPORT_PATH, "w");
    if (report) {
        write_report(report, zc_ok, z_fr, rate_fr, sl_fr, n_fr, z_cl, rate_cl, sl_cl, n_cl,
                     target);
        fclose(report);
        printf("[REPORT] %s\n", REPORT_PATH);
    }

    gputronic_stop(gov);
    gputronic_destroy(gov);
    return 0;
}
