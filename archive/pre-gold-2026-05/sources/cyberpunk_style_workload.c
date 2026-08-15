// Example: GPUTronic v26 library usage
// Supports two modes:
//   1. Simulated workload (default) — uses internal kernel, good for tuning
//   2. External / Real game mode — uses gputronic_init_external(), no internal kernel
//
// Compile:
//   nvcc -I../include -L../build -lgputronic_v26 cyberpunk_style_workload.c -o workload_demo
//
// Usage:
//   ./workload_demo                 # 30s simulated (internal kernel)
//   ./workload_demo 600             # 10-minute simulated run
//   ./workload_demo cyberpunk       # External mode (for real Cyberpunk / external workloads)

#include "gputronic.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

int main(int argc, char** argv) {
    int duration_sec = 30;
    int external_mode = 0;

    if (argc >= 2) {
        if (strcmp(argv[1], "cyberpunk") == 0 || strcmp(argv[1], "game") == 0 ||
            strcmp(argv[1], "external") == 0) {
            external_mode = 1;
        } else {
            duration_sec = atoi(argv[1]);
            if (duration_sec <= 0) duration_sec = 30;
        }
    }

    if (external_mode) {
        printf("=== GPUTronic v26 — External Workload Mode ===\n");
        printf("Using gputronic_init_external() — no internal kernel launched.\n");
        printf("Duration: %d seconds\n\n", duration_sec);

        GPUTronicHandle* gov = gputronic_init_external(0.35f, 0.04f, 1.0f);
        if (!gov) {
            fprintf(stderr, "Failed to initialize GPUTronic (external mode)\n");
            return 1;
        }

        gputronic_start(gov);

        printf("External governor active. Ready for real workload (Cyberpunk, etc.).\n");
        printf("Z target = 1.0 | Press Ctrl+C to abort early.\n\n");

        for (int sec = 0; sec < duration_sec; sec++) {
            usleep(1000000);
            float z = gputronic_get_z(gov);
            int sleep_ns = gputronic_get_sleep_ns(gov);
            printf("[Ext] t=%3ds | Z=%.3f | throttle=%6d ns\n", sec+1, z, sleep_ns);
        }

        gputronic_stop(gov);
        printf("\n[Demo] External governor stopped cleanly.\n");
        return 0;
    }

    // Default: simulated internal-kernel mode
    printf("=== GPUTronic v26 — Simulated Workload Mode ===\n");
    printf("Duration: %d seconds (great for tuning)\n\n", duration_sec);

    GPUTronicHandle* gov = gputronic_init(0.35f, 0.04f, 1.0f);
    if (!gov) {
        fprintf(stderr, "Failed to initialize GPUTronic\n");
        return 1;
    }

    gputronic_start(gov);

    printf("Governor active. Running simulated heavy workload...\n");
    printf("Z target = 1.0 | Press Ctrl+C to abort early.\n\n");

    for (int sec = 0; sec < duration_sec; sec++) {
        usleep(1000000);
        float z = gputronic_get_z(gov);
        int sleep_ns = gputronic_get_sleep_ns(gov);
        printf("[Sim] t=%3ds | Z=%.3f | throttle=%6d ns\n", sec+1, z, sleep_ns);
    }

    gputronic_stop(gov);
    printf("\n[Demo] Governor stopped. Simulated run complete.\n");
    return 0;
}
