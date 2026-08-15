// =============================================================================
// GPUTronic Stage 13 SCAG v14.0.0 + PER-SM COUNTER LOGGING ENABLED
// Target: Blackwell (RTX 5080) — captures per-SM workload distribution
// Author: GPUTronic Architect for Ben Coupland
// =============================================================================

#define USE_PER_SM_LOGGING 1

static FILE *g_sm_log = NULL;  // Per-SM CSV log file

inline void start_sm_logging(const char* filename) {
    if (g_sm_log == NULL) {
        g_sm_log = fopen(filename, "w");
        if (g_sm_log) {
            fprintf(g_sm_log, "#timestamp_us,sm_id,work_delta\n");
        }
    }
}

inline void log_per_sm_counters() {
    static double last_print_us = 0;
    
    // Log per-SM counters every 10ms (~100Hz) to avoid I/O bottleneck
    if ((now - last_print_us) > 10000.0 && g_sm_log != NULL) {
        fprintf(g_sm_log, "%.0f", now);
        
        for (int i = 0; i < NUM_SM; ++i) {
            unsigned long long sm_work = atomicLoad(&sm_work_counters[i]);
            unsigned long long last_work = g_last_work_per_sm[i];
            
            if (sm_log_initialized && i == 0) {
                // Header already written, now just data
            } else {
                fprintf(g_sm_log, ",%d", i);
            }
        }
        
        fprintf(g_sm_log, "\n");
        fflush(g_sm_log);
        
        last_print_us = now;
    }
}

// Add this after all counter declarations in global scope:
static unsigned long long g_last_work_per_sm[NUM_SM] = {0};
static int sm_log_initialized = 0;
