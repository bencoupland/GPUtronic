/* =============================================================================
 * GPUtronic Gold 1.0.0
 * -----------------------------------------------------------------------------
 * Read with: docs/GPUtronic-GOLD-1.0-FUNCTION-FRAME.md  (GT-FF-GOLD-1.0)
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

/* Test only. GPUTRONIC_PEAK_REPS and GPUTRONIC_PEAK_CHAINS. The result is
 * stored because ptxas deletes the nest otherwise. One slot per thread.
 * The unset kernel does not call this. Calling it from the same kernel
 * makes ptxas keep the default loop's FMAs. */
__device__ volatile float gold_peak_keep[84 * 1024];

__device__ void gold_work_peak(float* local, int sm, int reps, int chains) {
    int rN = reps > 1 ? reps : 1;
    int cN = chains > 1 ? chains : 1;
    if (rN > 256) rN = 256;
    if (cN > 16) cN = 16;
    float acc[16];
    for (int c = 0; c < 16; c++) acc[c] = *local + 0.001f * (float)c;
    for (int r = 0; r < rN; r++) {
        for (int b = 0; b < GOLD_BATCH; b++) {
            for (int i = 0; i < GOLD_WORK_ITERS; i++) {
                int iter = i + b * GOLD_WORK_ITERS;
                for (int c = 0; c < cN; c++)
                    gold_work(&acc[c], sm, iter + r * 64 + c * 17);
            }
        }
    }
    float sum = acc[0];
    for (int c = 1; c < cN; c++) sum += acc[c];
    *local = sum;
    int slot = sm * 1024 + (int)threadIdx.x;
    if ((unsigned)slot < 84u * 1024u)
        gold_peak_keep[slot] = sum;
}

/* Test only. GPUTRONIC_PEAK_ACC is 4, 8, 16, or 32 independent chains.
 * GPUTRONIC_PEAK_ROUNDS repeats that block (default 32, cap 4096). One pulse
 * is still one finished block. The volatile FMA stays in SASS; a plain
 * C loop of acc=acc*a+b can be closed over by ptxas. This is not called
 * from the unset kernel. */
template<int ACC>
__device__ void gold_work_indep(float* local, int sm, int rounds, volatile float* keep) {
    float acc[ACC];
    float a = 1.0000001f + 0.0000001f * (float)(sm & 3);
    float b = 0.9999999f;
#pragma unroll
    for (int i = 0; i < ACC; i++)
        acc[i] = *local + 0.01f * (float)(i + 1);
    int rN = rounds > 0 ? rounds : 1;
    if (rN > 4096) rN = 4096;
    /* One iteration must stay one block of ACC FMAs, so the pulse quantum
     * is ACC * rounds and not an unrolled fragment. */
#pragma unroll 1
    for (int r = 0; r < rN; r++) {
#pragma unroll
        for (int i = 0; i < ACC; i++) {
            float d;
            asm volatile("fma.rn.ftz.f32 %0, %1, %2, %3;"
                         : "=f"(d) : "f"(a), "f"(b), "f"(acc[i]));
            acc[i] = d;
        }
    }
    float sum = acc[0];
#pragma unroll
    for (int i = 1; i < ACC; i++) sum += acc[i];
    *local = sum;
    int slot = sm * 1024 + (int)threadIdx.x;
    if (keep && (unsigned)slot < 84u * 1024u)
        keep[slot] = sum;
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

/* ---- T11 per-SM ops accumulator (opt-in) ----
 * Exact ops from the code above, FMA counted as 2 FP32 ops:
 *   CUDA warp pass  = GOLD_BATCH 16 x GOLD_WORK_ITERS 8 x 8 FMAs x 2 x 32 lanes = 65,536 ops
 *   tensor warp pass = GOLD_TENSOR_STEPS 64 x 2*16*16*16 = 524,288 ops (0 below sm_70: stub)
 * Warps add to a shared-memory tally (no global atomics). Every bin_ns of %globaltimer
 * the block leader stores one 40-byte slot {ops_cuda, ops_tensor, t_ns, phys_sm, seq}
 * to a mapped host array. phys_sm is the %%smid read once at block start. seq is last,
 * after a system fence (host retries if seq changes mid-read). The slot stays keyed by
 * blockIdx.x. accum == NULL (default) costs one register test per pass and no memory
 * traffic.
 * GPUTRONIC_IDLE_QUIET=1 is a test switch, off by default. Idle blocks then wait on
 * %%globaltimer for 1 ms between samples of the mapped control words, instead of
 * polling those words every trip around the loop.
 * GPUTRONIC_ACTIVE_PASSES and GPUTRONIC_ACTIVE_SMS are test switches, off by
 * default. After calibration the host sets GPUTRONIC_FLAG_SELECT and the block
 * index gate is replaced: a block works if its launch pass ((%%smid %% 14) / 2)
 * is in the list, or if its physical SM id is in the list.
 * GPUTRONIC_PULSE_LOCAL=1 is a test switch, off by default. Each warp counts
 * pulses in a register. The block leader adds that batch to the mapped pulse
 * counter about once a millisecond. Control-word reads stay every pass.
 * GPUTRONIC_CTRL_CACHE=1 is a separate test switch, off by default. The block
 * leader reads the mapped control words, including work_scale_milli, once per
 * GPUTRONIC_CTRL_CACHE_US (default 1000, floor 50) and the block uses that
 * shared copy. Unset, every pass reads the mapped words, and work_scale_milli
 * is still read only while the load flag is set. The pulse batch stays at 1 ms.
 * Tensor-on blocks do not take this cache: a barrier there is not uniform.
 * GPUTRONIC_ICLAMP_SMS sets the integral clamp in SMs for one run. Unset
 * keeps the scaled law. GPUTRONIC_RATE_WIN_US sets the rate window in
 * microseconds for one run. Unset keeps 8000 (8 ms). Floor 500, cap 100000.
 * The 0.70/0.30 mix stays. GPUTRONIC_OPEN_SCHED is a host test, off by
 * default. fullwidth and noise then read a loadfile-format schedule
 * (t_ms scale, optional active_sms). Unset keeps scale 1.0 and the launch
 * count. A count at or above the launch means every block. Closed-loop
 * loadfile ignores the third column. GPUTRONIC_IDLE_WORK=1 is a test, off by default.
 * Blocks past active_sms then run the same body and atomicAdd into
 * total_work_pulses. Unset, those blocks do not. GPUTRONIC_IDLE_EXIT=1
 * makes an idle block return.
 * The host launches fresh blocks when the count rises. GPUTRONIC_RELAUNCH=1
 * retires the grid and launches the commanded count, after a hold so a
 * one-tick flicker does not relaunch. GPUTRONIC_PEAK_REPS, GPUTRONIC_PEAK_CHAINS,
 * and GPUTRONIC_THREADS raise work per pass. All of these stay off when unset. */
#define GOLD_OPS_CUDA_WARP_PASS (2ULL * GOLD_BATCH * GOLD_WORK_ITERS * 8ULL * 32ULL)
#define GOLD_OPS_TENSOR_WARP_PASS (2ULL * 16 * 16 * 16 * GOLD_TENSOR_STEPS)
#define GOLD_ACCUM_WORDS 5
#define GOLD_IDLE_QUIET_NS 1000000ULL
#define GOLD_PULSE_LOCAL_NS 1000000ULL

__device__ __forceinline__ unsigned long long gold_globaltimer_ns(void) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

/* Test only. The guest kernel is a second grid in this process. It does not
 * read active_sms and it does not add to total_work_pulses. One pulse here is
 * one warp finishing one pass, the same quantum as the Gold kernel with the
 * pulse batch on. The count is published once a millisecond. Off unless
 * GPUTRONIC_GUEST_BLOCKS is set. keep is separate so the two grids do not
 * store into the same sink.
 * marks is 4 words per block, indexed by blockIdx: enter_ns, smid,
 * first_pulse_ns, last_ns. enter_ns is the GPU timer when the block begins.
 * first_pulse_ns is the timer of its first published pulse. */
__device__ volatile float guest_keep[84 * 1024];

template<int ACC>
__global__ void gold_guest_kernel(unsigned long long* pulses, int* stop, int* live,
                                  unsigned long long* marks,
                                  int rounds, int skip0, int skip1) {
    const int sm = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warps = (int)blockDim.x >> 5;
    float local = 0.0f;
    unsigned int unpublished = 0;
    unsigned long long pulse_last = 0ULL;
    __shared__ unsigned int s_warp_pulses[32];
    __shared__ int s_flush;
    __shared__ int s_stop;
    int indep_rounds = rounds > 0 ? rounds : 1;
    if (indep_rounds > 4096) indep_rounds = 4096;
    if (tid == 0) {
        unsigned int phys = 0;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(phys));
        int skip = ((int)phys == skip0 || (int)phys == skip1);
        s_stop = skip ? 1 : 0;
        if (!skip && live) atomicAdd(live, 1);
        if (!skip && marks && sm >= 0 && sm < 84) {
            unsigned long long base = (unsigned long long)sm * 4ULL;
            marks[base + 0] = gold_globaltimer_ns();
            marks[base + 1] = (unsigned long long)phys;
        }
        __threadfence_system();
    }
    __syncthreads();
    if (s_stop) return;
    while (s_stop == 0) {
        gold_work_indep<ACC>(&local, sm, indep_rounds, guest_keep);
        if (local > 1.0e30f) local = 0.0f;
        if (lane == 0) {
            unpublished++;
            s_warp_pulses[tid >> 5] = unpublished;
        }
        __syncthreads();
        if (tid == 0) {
            unsigned long long now = gold_globaltimer_ns();
            s_flush = 0;
            if (pulse_last == 0ULL || now - pulse_last >= GOLD_PULSE_LOCAL_NS) {
                unsigned int sum = 0;
                int wlim = warps < 32 ? warps : 32;
                for (int w = 0; w < wlim; w++) sum += s_warp_pulses[w];
                if (sum && pulses) atomicAdd(pulses, (unsigned long long)sum);
                if (marks && sum && sm >= 0 && sm < 84) {
                    unsigned long long base = (unsigned long long)sm * 4ULL;
                    if (marks[base + 2] == 0ULL) marks[base + 2] = now;
                    marks[base + 3] = now;
                }
                __threadfence_system();
                pulse_last = now;
                s_flush = 1;
                if (stop && *stop) s_stop = 1;
            }
        }
        __syncthreads();
        if (s_flush && lane == 0) unpublished = 0;
    }
}

/*
 * Persistent self-test kernel [F-02].
 *   grid = num_sm (one block per launched SM; two device SMs stay unlaunched)
 *   warp leaders (lane==0) add to the pulse total and the per-SM diagnostic [F-04]
 *   Actuator: the lowest active_sms blocks do the work and pulse.
 *   Idle blocks poll the mapped words and do not pulse. Duration is not the command.
 *   Two sensors [F-16]: warps below cuda_warps run the CUDA-core FMA batch
 *   and pulse total_work_pulses (unchanged). When the tensor sensor is on,
 *   one extra warp per block runs wmma and pulses tensor_work_pulses with
 *   the same fence cadence. active_sms gates both.
 * Period is atomic-dominated. Do not "tune Kp" to hide that.
 */
/* Synthetic load [T6/T7]: with work_scale_milli = S > 1000, a warp pulses once per
 * S/1000 batches (credit counter), so measured work per pulse rises by S/1000 and
 * the pulse rate falls to 1000/S. A literal extra-FMA multiplier would not work:
 * the pass is about 288 us of mapped atomics and only ~1.6 us of arithmetic, so 20%
 * more arithmetic barely moves the rate. Default 1000/0 always pulses.
 * That sentence is the unset kernel. GPUTRONIC_WORK_LOAD=1 is the other test,
 * off unless the run sets it, and only with the independent-FMA body: the same
 * scale changes the round count, and every batch still pulses. */
__device__ __forceinline__ int gold_scaled_rounds(int base, unsigned int scale_milli) {
    unsigned int sc = scale_milli ? scale_milli : 1000u;
    int r = (int)(((unsigned long long)base * sc + 500ull) / 1000ull);
    if (r < 1) r = 1;
    if (r > 4096) r = 4096;
    return r;
}

__device__ __forceinline__ int gold_load_pulse(unsigned int flags, GPUtronicControl* ctrl,
                                               unsigned int* credit) {
    /* work_scale_milli is only read while the host has set GPUTRONIC_FLAG_LOAD in
     * control_flags, which every pass already reads. Normal runs add no mapped read.
     * This body is the switch-off path. The cache uses gold_load_from_scale. */
    if (!(flags & GPUTRONIC_FLAG_LOAD)) return 1;
    unsigned int s = ctrl->work_scale_milli;
    if (s <= 1000u) return 1;
    *credit += 1000u;
    if (*credit >= s) { *credit -= s; return 1; }
    return 0;
}

__device__ __forceinline__ int gold_load_from_scale(unsigned int flags, unsigned int s,
                                                    unsigned int* credit) {
    /* s is the shared copy of work_scale_milli. No mapped read. */
    if (!(flags & GPUTRONIC_FLAG_LOAD)) return 1;
    if (s <= 1000u) return 1;
    *credit += 1000u;
    if (*credit >= s) { *credit -= s; return 1; }
    return 0;
}

__device__ __forceinline__ int gold_sm_selected(unsigned int phys,
                                                     unsigned int pass_mask,
                                                     unsigned long long sm_lo,
                                                     unsigned long long sm_hi) {
    /* Physical-SM filter wins over the pass list. Pass is (smid % 14) / 2,
     * which on this 5080 is the GPC id. Both are test-only. */
    if (sm_lo | sm_hi) {
        if (phys < 64u) return (int)((sm_lo >> phys) & 1ull);
        if (phys < 128u) return (int)((sm_hi >> (phys - 64u)) & 1ull);
        return 0;
    }
    unsigned int pass = (phys % 14u) / 2u;
    if (pass >= 32u) return 0;
    return (int)((pass_mask >> pass) & 1u);
}

/* Peak is a compile-time switch. The unset kernel must not contain the
 * peak nest, or ptxas keeps the default FMAs and the pulse rate falls.
 * Acc is 0 on that kernel and on the dependent peak nest. A positive Acc
 * is the independent-FMA test kernel. */
template<int Peak, int Acc>
__global__ void gold_persistent_kernel(unsigned long long* sm_counters,
                                       GPUtronicControl* ctrl,
                                       int num_sm,
                                       int cuda_warps,
                                       unsigned long long* work_count,
                                       unsigned long long* accum,
                                       int accum_sm,
                                       unsigned long long accum_bin_ns,
                                       int idle_quiet,
                                       unsigned int pass_mask,
                                       unsigned long long sm_mask_lo,
                                       unsigned long long sm_mask_hi,
                                       int pulse_local,
                                       int ctrl_cache,
                                       unsigned long long ctrl_cache_ns,
                                       unsigned int* smid_out,
                                       int idle_exit,
                                       int idle_work,
                                       unsigned int* slot_gen,
                                       int block_base,
                                       int peak_reps,
                                       int peak_chains,
                                       int peak_rounds,
                                       int work_load) {
    const int sm = blockIdx.x + block_base;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int is_tensor_warp = (tid >> 5) >= cuda_warps;
    unsigned int phys_sm = 0;
    float local = 0.0f;
    unsigned int local_pulses = 0;
    unsigned int unpublished = 0;   /* pulses not yet added to the mapped counter */
    unsigned long long pulse_last = 0ULL;
    unsigned int load_credit = 0;   /* T6/T7 synthetic load, see work_scale_milli */
    unsigned long long work_done = 0; /* T9: completed batches, this warp */
    unsigned int my_gen = 0;          /* idle-exit ownership of this logical SM */

    __shared__ unsigned long long s_ops[2];   /* T11: CUDA ops, tensor ops. Block-wide. */
    __shared__ unsigned int s_phys;
    __shared__ unsigned int s_warp_pulses[32];
    __shared__ int s_flush;
    __shared__ unsigned int s_flags;
    __shared__ int s_active;
    __shared__ unsigned int s_scale;
    __shared__ unsigned int s_work_scale;
    __shared__ unsigned long long s_ctrl_last;
    if (sm >= num_sm) return;
    if (tid == 0) {
        s_ops[0] = 0ULL;
        s_ops[1] = 0ULL;
        s_flags = 0;
        s_active = num_sm;
        s_scale = 1000u;
        s_ctrl_last = 0ULL;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(phys_sm));
        s_phys = phys_sm;
        /* Once, before the loop, including on a block that will sit idle.
         * The two device SMs with no block never write, so the host can
         * name them by absence. */
        if (smid_out) {
            ((volatile unsigned int*)smid_out)[sm] = phys_sm;
            __threadfence_system();
        }
    }
    __syncthreads();
    const int accum_on = (accum != NULL) && (accum_sm < 0 || accum_sm == sm);
    unsigned long long accum_last = 0ULL, accum_seq = 0ULL;
    int peak_on = (peak_reps > 1 || peak_chains > 1);
    int peak_r = peak_reps > 1 ? peak_reps : 1;
    int peak_c = peak_chains > 1 ? peak_chains : 1;
    if (peak_r > 256) peak_r = 256;
    if (peak_c > 16) peak_c = 16;
    unsigned long long pass_ops = GOLD_OPS_CUDA_WARP_PASS;
    int indep_rounds = peak_rounds > 0 ? peak_rounds : 1;
    if (indep_rounds > 4096) indep_rounds = 4096;
    if constexpr (Peak != 0 && Acc > 0) {
        /* One FMA is 2 FLOP, one warp is 32 lanes, Acc chains, indep_rounds deep. */
        pass_ops = (unsigned long long)Acc * (unsigned long long)indep_rounds * 2ULL * 32ULL;
    } else if (peak_on) {
        pass_ops *= (unsigned long long)peak_r * (unsigned long long)peak_c;
    }
    if (tid == 0) {
        /* One write before the loop so an idle block, which never publishes,
         * still leaves its physical SM in the slot. seq stays 0 until a publish. */
        if (accum_on) {
            volatile unsigned long long* slot = accum + GOLD_ACCUM_WORDS * sm;
            slot[3] = phys_sm;
            __threadfence_system();
        }
        /* Host stored the generation before launch. Exit clears it only if
         * this block still owns the slot. Off unless GPUTRONIC_IDLE_EXIT=1. */
        if (idle_exit && slot_gen) my_gen = ((volatile unsigned int*)slot_gen)[sm];
    }

    while (true) {
        /* ctrl_cache is a launch argument, so the branch matches across the block.
         * Off unless GPUTRONIC_CTRL_CACHE=1. Tensor-on blocks keep the per-pass
         * reads: a barrier there would not be uniform. */
        unsigned int flags;
        int use_cache = ctrl_cache && (blockDim.x == cuda_warps * 32);
        if (use_cache) {
            if (tid == 0) {
                unsigned long long now = gold_globaltimer_ns();
                unsigned long long cache_ns = ctrl_cache_ns ? ctrl_cache_ns : GOLD_PULSE_LOCAL_NS;
                if (s_ctrl_last == 0ULL || now - s_ctrl_last >= cache_ns) {
                    s_flags = ctrl->control_flags;
                    s_active = ctrl->active_sms;
                    s_scale = ctrl->work_scale_milli;
                    s_ctrl_last = now;
                }
            }
            __syncthreads();
            flags = s_flags;
        } else {
            flags = ctrl->control_flags;
        }
        if (flags & GPUTRONIC_FLAG_STOP) break;
        if (flags & GPUTRONIC_FLAG_RETIRE) break;
        if (flags & GPUTRONIC_FLAG_PAUSE) {
            (void)(5000);   /* was __nanosleep: removed, NEXT.md A4 */
            continue;
        }
        if (flags & GPUTRONIC_FLAG_RESET) {
            if (tid == 0) sm_counters[sm] = 0ULL;
            __syncthreads();
            (void)(1000);   /* was __nanosleep: removed, NEXT.md A4 */
            continue;
        }

        int active = use_cache ? s_active : ctrl->active_sms;
        if (active <= 0 || active > num_sm) active = num_sm;
        if (active < GPUTRONIC_SM_FLOOR) active = GPUTRONIC_SM_FLOOR;
        /* FLAG_SELECT is a test switch. The host sets it only after calibration,
         * and only when a pass list or a physical-SM list was requested. */
        int selected = (flags & GPUTRONIC_FLAG_SELECT) &&
                       (pass_mask || sm_mask_lo || sm_mask_hi);
        int idle = selected ? !gold_sm_selected(s_phys, pass_mask, sm_mask_lo, sm_mask_hi)
                            : (sm >= active);
        /* GPUTRONIC_IDLE_WORK=1 is a test, off by default. An idle block then
         * runs the same body below and the same atomicAdd into
         * total_work_pulses. The SM command cannot turn that work off. */
        if (idle && !idle_work) {
            if (idle_exit) {
                /* The block ends. The SM is free for the desktop or a later
                 * launch. Off by default. GPUTRONIC_IDLE_EXIT=1. */
                if (tid == 0 && slot_gen && my_gen != 0u) {
                    atomicCAS((unsigned int*)slot_gen + sm, my_gen, 0u);
                    __threadfence_system();
                }
                return;
            }
            if (idle_quiet) {
                /* Local wait. The mapped words are read once per trip, at the
                 * top of the loop, so STOP and a new active_sms still land.
                 * Off by default. GPUTRONIC_IDLE_QUIET=1. */
                unsigned long long t0 = gold_globaltimer_ns();
                while (gold_globaltimer_ns() - t0 < GOLD_IDLE_QUIET_NS) { }
            }
            continue;
        }

        if (is_tensor_warp) {
            float t = gold_tensor_work(sm);
            if (t > 1.0e30f) local += t;   /* keep the result live */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
            if (accum_on && lane == 0) atomicAdd(&s_ops[1], GOLD_OPS_TENSOR_WARP_PASS);
#endif
            if (lane == 0 && gold_load_pulse(flags, ctrl, &load_credit)) {
                atomicAdd((unsigned long long*)&ctrl->tensor_work_pulses, 1ULL);
                local_pulses++;
                if ((local_pulses & 31u) == 0u) __threadfence_system();
            }
            continue;
        }

        int rounds_now = indep_rounds;
        unsigned long long ops_add = pass_ops;
        if constexpr (Peak != 0 && Acc > 0) {
            /* GPUTRONIC_WORK_LOAD=1. The cached scale is already uniform.
             * Without the cache, one thread reads the mapped word and the
             * block shares it. Tensor-on blocks do not take that barrier. */
            if (work_load && (flags & GPUTRONIC_FLAG_LOAD)) {
                unsigned int sc = 1000u;
                if (use_cache) {
                    sc = s_scale ? s_scale : 1000u;
                } else if (blockDim.x == cuda_warps * 32) {
                    if (tid == 0)
                        s_work_scale = ctrl->work_scale_milli ? ctrl->work_scale_milli : 1000u;
                    __syncthreads();
                    sc = s_work_scale;
                }
                rounds_now = gold_scaled_rounds(indep_rounds, sc);
                ops_add = (unsigned long long)Acc * (unsigned long long)rounds_now * 2ULL * 32ULL;
            }
            gold_work_indep<Acc>(&local, sm, rounds_now, gold_peak_keep);
        } else if constexpr (Peak != 0) {
            gold_work_peak(&local, sm, peak_r, peak_c);
        } else {
            for (int b = 0; b < GOLD_BATCH; b++) {
                for (int i = 0; i < GOLD_WORK_ITERS; i++)
                    gold_work(&local, sm, i + b * GOLD_WORK_ITERS);
            }
        }
        /* Same anti-DCE the default loop has always had. */
        if (local > 1.0e30f) local = 0.0f;

        /* T9 work counter: off (NULL) by default. Plain device memory, one slot
         * per warp, one non-atomic store per pass: no PCIe traffic. */
        if (work_count != NULL && lane == 0) {
            work_done += GOLD_BATCH;
            work_count[sm * 32 + (tid >> 5)] = work_done;
        }

        /* Tensor warps already continued, so a barrier is only legal when this
         * block is CUDA warps alone. That is the normal run: the tensor sensor
         * is off. The barrier lets the leader add both warps into one snapshot. */
        if (accum_on && !is_tensor_warp) {
            if (lane == 0) atomicAdd(&s_ops[0], ops_add);
            const int cuda_only = (blockDim.x == cuda_warps * 32);
            if (cuda_only) __syncthreads();
            if (tid == 0) {
                unsigned long long now = gold_globaltimer_ns();
                if (accum_last == 0ULL || now - accum_last >= accum_bin_ns) {
                    volatile unsigned long long* slot = accum + GOLD_ACCUM_WORDS * sm;
                    slot[0] = s_ops[0];
                    slot[1] = s_ops[1];
                    slot[2] = now;
                    slot[3] = phys_sm;
                    __threadfence_system();
                    slot[4] = ++accum_seq;
                    accum_last = now;
                }
            }
            if (cuda_only) __syncthreads();
        }

        /* pulse_local is a launch argument, so the branch is the same on every
         * thread. It stays off unless GPUTRONIC_PULSE_LOCAL=1. Tensor-on blocks
         * keep the per-pass mapped atomic: a barrier there would not be uniform. */
        int use_local = pulse_local && (blockDim.x == cuda_warps * 32);
        /* use_cache is the same on every thread. Off, this is the old mapped
         * read, and only lane 0 takes it. On, lane 0 uses the shared scale. */
        int do_pulse = 0;
        if (lane == 0) {
            /* Real-work load: every batch publishes. The scale already
             * changed the round count above. Unset, this is the pulse skip. */
            if (work_load)
                do_pulse = 1;
            else if (use_cache)
                do_pulse = gold_load_from_scale(flags, s_scale, &load_credit);
            else
                do_pulse = gold_load_pulse(flags, ctrl, &load_credit);
        }
        if (do_pulse) {
            if (use_local) {
                unpublished++;
            } else {
                atomicAdd(&sm_counters[sm], 1ULL);
                atomicAdd((unsigned long long*)&ctrl->total_work_pulses, 1ULL);
                local_pulses++;
                /* Fence every 32 pulses — enough for host visibility. */
                if ((local_pulses & 31u) == 0u) __threadfence_system();
            }
        }
        if (use_local) {
            if (lane == 0) s_warp_pulses[tid >> 5] = unpublished;
            __syncthreads();
            if (tid == 0) {
                unsigned long long now = gold_globaltimer_ns();
                s_flush = 0;
                if (pulse_last == 0ULL || now - pulse_last >= GOLD_PULSE_LOCAL_NS) {
                    unsigned int sum = 0;
                    for (int w = 0; w < cuda_warps; w++) sum += s_warp_pulses[w];
                    if (sum) {
                        atomicAdd(&sm_counters[sm], (unsigned long long)sum);
                        atomicAdd((unsigned long long*)&ctrl->total_work_pulses,
                                  (unsigned long long)sum);
                        __threadfence_system();
                    }
                    pulse_last = now;
                    s_flush = 1;
                }
            }
            __syncthreads();
            if (s_flush && lane == 0) unpublished = 0;
        }
    }
}

extern "C" void gold_launch_kernel(unsigned long long* sm_counters,
                                   GPUtronicControl* d_ctrl,
                                   int num_sm,
                                   int threads_per_block,
                                   int tensor_sensor,
                                   unsigned long long* work_count,
                                   unsigned long long* accum,
                                   int accum_sm,
                                   unsigned long long accum_bin_ns,
                                   int idle_quiet,
                                   unsigned int pass_mask,
                                   unsigned long long sm_mask_lo,
                                   unsigned long long sm_mask_hi,
                                   int pulse_local,
                                   int ctrl_cache,
                                   unsigned long long ctrl_cache_ns,
                                   unsigned int* smid_out,
                                   int idle_exit,
                                   int idle_work,
                                   unsigned int* slot_gen,
                                   int nblocks,
                                   int block_base,
                                   int peak_reps,
                                   int peak_chains,
                                   int peak_acc,
                                   int peak_rounds,
                                   int use_peak,
                                   int work_load,
                                   cudaStream_t stream) {
    /* CUDA-core warps stay as configured; the tensor sensor adds one warp.
     * nblocks is this launch. num_sm is the logical ceiling (the OS-reserve
     * launch count). block_base is the first logical SM in this launch. */
    int cuda_warps = (threads_per_block + 31) / 32;
    int threads = cuda_warps * 32 + (tensor_sensor ? 32 : 0);
    if (nblocks < 1) nblocks = 1;
    /* use_peak selects a peak kernel. Acc 4/8/16/32 is the independent nest.
     * Acc 0 is the dependent nest. The unset launch stays on the kernel
     * whose FMA nest the compiler removes. */
    int peak = use_peak ? 1 : 0;
    int acc = peak ? peak_acc : 0;
#define GOLD_LAUNCH(PEAK, ACC) \
    gold_persistent_kernel<PEAK, ACC><<<nblocks, threads, 0, stream>>>( \
        sm_counters, d_ctrl, num_sm, cuda_warps, work_count, accum, accum_sm, accum_bin_ns, \
        idle_quiet, pass_mask, sm_mask_lo, sm_mask_hi, pulse_local, ctrl_cache, ctrl_cache_ns, \
        smid_out, idle_exit, idle_work, slot_gen, block_base, peak_reps, peak_chains, peak_rounds, work_load)
    if (!peak)
        GOLD_LAUNCH(0, 0);
    else if (acc == 4)
        GOLD_LAUNCH(1, 4);
    else if (acc == 8)
        GOLD_LAUNCH(1, 8);
    else if (acc == 16)
        GOLD_LAUNCH(1, 16);
    else if (acc == 32)
        GOLD_LAUNCH(1, 32);
    else
        GOLD_LAUNCH(1, 0);
#undef GOLD_LAUNCH
}

/* ---- Host handle and control thread ---- */

#define GOLD_MAX_PROBE 8

struct GPUtronicHandle {
    GPUtronicConfig cfg;
    GPUtronicControl* h_ctrl;
    GPUtronicControl* d_ctrl;
    unsigned long long* d_sm;   /* per-SM diagnostic counters (device memory) */
    int device_sms;             /* multiProcessorCount, before the OS reserve */
    unsigned int* h_smid;       /* mapped %smid of each launched block */
    unsigned int* d_smid;
    unsigned int* smid_boot;    /* host copy of the first launch's map */
    unsigned long long* d_work; /* T9 work counter, num_sm*32 slots; NULL = off (default) */
    unsigned long long* h_accum; /* T11 mapped slots, GOLD_ACCUM_WORDS x u64 per block; NULL = off */
    unsigned long long* d_accum;
    int accum_sm;                /* -1 all SMs, else the one SM that reports */
    unsigned long long accum_bin_ns;
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
    double rate_win_s;   /* seconds; 0.008 unless GPUTRONIC_RATE_WIN_US is set */
    double rate_win_dt;
    unsigned long long rate_win_pulses;
    /* Loop telemetry (control thread writes; trace/stats read; diagnostic only). */
    unsigned long long st_loop;      /* control-loop iterations, all phases */
    unsigned long long st_ticks;     /* closed-loop PI ticks after calibration */
    unsigned long long st_db_skip;   /* ticks held inside the deadband */
    unsigned long long st_int_upd;   /* ticks where the integral changed (> 1e-6) */
    unsigned long long st_rail;      /* ticks with |integral| at the clamp */
    unsigned long long st_moves;     /* ticks where the SM count changed */
    unsigned long long st_p_only;    /* SM-count changes with the integral unchanged */
    unsigned long long last_win_pulses; /* pulses in the last closed 8 ms window */
    float last_p_term;               /* Kp*e - Kd*dZ/dt on the last PI tick */
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
    /* Test-only working-set filter. Zero unless the env is set. The kernel
     * ignores it until the host raises GPUTRONIC_FLAG_SELECT after calibration. */
    unsigned int select_pass_mask;
    unsigned long long select_sm_lo;
    unsigned long long select_sm_hi;
    /* Test only. GPUTRONIC_PULSE_LOCAL=1 batches the mapped pulse atomic. */
    int pulse_local;
    /* Test only. GPUTRONIC_CTRL_CACHE=1 batches the mapped control-word reads,
     * including work_scale_milli. ctrl_cache_ns is that period. Default 1 ms.
     * The pulse batch stays 1 ms. */
    int ctrl_cache;
    unsigned long long ctrl_cache_ns;
    /* Test only. See gold_apply_test_env. Zero unless that run sets them. */
    int idle_quiet;
    int idle_exit;
    int idle_work;
    int relaunch;
    long relaunch_hold_us;
    double launched_at_us;
    int launched_n;
    int smid_expect;
    unsigned int gen_tick;
    unsigned int* h_slot;
    unsigned int* d_slot;
    cudaStream_t side_stream;
    int side_on;
    /* One stream per idle-exit refill. A second launch on the same stream
     * waits until the first refill kernel returns, and that kernel stays
     * resident, so the refill never starts. */
    cudaStream_t wave_stream[64];
    int n_waves;
    cudaStream_t kstream;
    int kstream_on;
    int peak_reps;
    int peak_chains;
    int peak_acc;    /* 4/8/16/32 independent FMAs per thread per round; 0 = off */
    int peak_rounds; /* repeats of that independent block */
    int peak_live; /* 1 when a peak test switch is set, including reps=1 */
    int work_load; /* GPUTRONIC_WORK_LOAD=1: scale changes FMA rounds, every batch pulses */
    unsigned int spawn_count;
    unsigned int relaunch_count;
    /* Test only. GPUTRONIC_GUEST_BLOCKS. Zero unless that run sets it. */
    int guest_blocks;
    int guest_acc;
    int guest_rounds;
    int guest_skip0;
    int guest_skip1;
    unsigned long long* h_guest_pulses;
    unsigned long long* d_guest_pulses;
    int* h_guest_stop;
    int* d_guest_stop;
    int* h_guest_live;
    int* d_guest_live;
    cudaStream_t guest_stream;
    int guest_stream_on;
    int guest_launched;
    unsigned long long guest_prev_pulses;
    double guest_prev_us;
    float guest_rate;
    /* 4 mapped words per block: enter_ns, smid, first_pulse_ns, last_ns. */
    unsigned long long* h_guest_mark;
    unsigned long long* d_guest_mark;
    double guest_run_t0_us;
    double gold_stop_us;
    double guest_enter_seen_us[84];
    double guest_pulse_seen_us[84];
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

void gputronic_config_gold(GPUtronicConfig* cfg) {
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
    cfg->enable_tensor_sensor = 0;   /* NEXT.md A2. GPUTRONIC_TENSOR_SENSOR=1 turns it on. */
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
    ev = getenv("GPUTRONIC_INTEGRAL_CLAMP");   /* > 0 fixed clamp, e.g. 1.0 for the pre-fix baseline */
    if (ev && *ev) cfg->integral_clamp = (float)atof(ev);
    ev = getenv("GPUTRONIC_DEADBAND");   /* hold the count when |Z - target| is inside z (port of 42b88c9) */
    if (ev && *ev) {
        float db = (float)atof(ev);
        if (db >= 0.0f && db < 1.0f) cfg->deadband = db;
    }
    ev = getenv("GPUTRONIC_RESERVE_SMS");
    if (ev && *ev) cfg->reserve_sms = atoi(ev) < 0 ? 0 : atoi(ev);
    /* Test only. Unset keeps 64 threads. Capped at 1024 so the pulse
     * batch still fits in s_warp_pulses[32]. */
    ev = getenv("GPUTRONIC_THREADS");
    if (ev && *ev) {
        int t = atoi(ev);
        if (t < 32) t = 32;
        if (t > 1024) t = 1024;
        t &= ~31;
        if (t < 32) t = 32;
        cfg->threads_per_block = t;
    }
    cfg->csv_path = "results/gputronic_gold.csv";
}

static void kalman_update(GPUtronicHandle* h, float z_meas, float dt) {
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
static void gold_fit_table(GPUtronicHandle* h, const char* label, double rate_ref,
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
/* GPUTRONIC_ICLAMP_SMS is a test switch. It sets the clamp in SMs for one
 * run: Ki * clamp * sms_per_u = that many SMs. Unset keeps the scaled law,
 * or the fixed config value when integral_clamp > 0. */
static int gold_iclamp_sms_env(float* sms_out) {
    const char* ev = getenv("GPUTRONIC_ICLAMP_SMS");
    if (!ev || !*ev) return 0;
    float sms = (float)atof(ev);
    if (sms <= 0.0f) return 0;
    if (sms_out) *sms_out = sms;
    return 1;
}

static void gold_set_iclamp(GPUtronicHandle* h) {
    float sms = 0.0f;
    float denom = h->cfg.ki * h->sms_per_u;
    if (gold_iclamp_sms_env(&sms)) {
        h->iclamp = (denom > 1e-6f) ? sms / denom : 1.0f;
        return;
    }
    if (h->cfg.integral_clamp > 0.0f) { h->iclamp = h->cfg.integral_clamp; return; }
    h->iclamp = (denom > 1e-6f && h->cfg.num_sm > 0) ? GPUTRONIC_GOLD_INTEGRAL_SMS_FRAC * (float)h->cfg.num_sm / denom : 1.0f;
}

static void gold_build_table(GPUtronicHandle* h) {
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
    const char* how = "scaled";
    float sms_req = 0.0f;
    if (gold_iclamp_sms_env(&sms_req)) how = "iclamp_sms";
    else if (h->cfg.integral_clamp > 0.0f) how = "config";
    printf("[CAL] integral clamp=%.3f (%s, max integral correction %.2f SMs of %d)\n", h->iclamp,
           how, h->cfg.ki * h->iclamp * h->sms_per_u, h->cfg.num_sm);
}

static void gold_launch_this(GPUtronicHandle* h, int nblocks, int block_base, cudaStream_t stream) {
    gold_launch_kernel(h->d_sm, h->d_ctrl, h->cfg.num_sm, h->cfg.threads_per_block,
                       h->cfg.enable_tensor_sensor, h->d_work,
                       h->d_accum, h->accum_sm, h->accum_bin_ns, h->idle_quiet,
                       h->select_pass_mask, h->select_sm_lo, h->select_sm_hi,
                       h->pulse_local, h->ctrl_cache, h->ctrl_cache_ns, h->d_smid,
                       h->idle_exit, h->idle_work, h->d_slot, nblocks, block_base,
                       h->peak_reps > 0 ? h->peak_reps : 1,
                       h->peak_chains > 0 ? h->peak_chains : 1,
                       h->peak_acc,
                       h->peak_rounds > 0 ? h->peak_rounds : 1,
                       h->peak_live,
                       h->work_load,
                       stream);
}

/* Short map of one launch. Waits at most 2 ms for %smid, then prints the ends. */
static void gold_note_smids(GPUtronicHandle* h, int nblocks, int block_base) {
    if (!h->h_smid || nblocks <= 0) return;
    int ready = 0;
    for (int spin = 0; spin < 20 && ready < nblocks; spin++) {
        ready = 0;
        for (int i = 0; i < nblocks; i++) {
            int idx = block_base + i;
            if (idx >= 0 && idx < h->cfg.num_sm && h->h_smid[idx] != 0xFFFFFFFFu) ready++;
        }
        if (ready >= nblocks) break;
        usleep(100);
    }
    int same = 0, moved = 0;
    printf("[SMID] launch base=%d n=%d ready=%d", block_base, nblocks, ready);
    int show = nblocks < 8 ? nblocks : 4;
    for (int i = 0; i < nblocks; i++) {
        int idx = block_base + i;
        unsigned int id = (idx >= 0 && idx < h->cfg.num_sm) ? h->h_smid[idx] : 0xFFFFFFFFu;
        if (i < show || i >= nblocks - 2)
            printf(" %d:%u", idx, id);
        else if (i == show)
            printf(" ...");
        if (h->smid_boot && idx >= 0 && idx < h->cfg.num_sm && id != 0xFFFFFFFFu) {
            if (id == h->smid_boot[idx]) same++;
            else moved++;
        }
    }
    if (h->smid_boot) printf(" same=%d moved=%d", same, moved);
    printf("\n");
    fflush(stdout);
}

/* Idle-exit only. A logical SM whose slot is 0 is not running. Claim it and
 * launch a block. Contiguous holes go out as one grid. */
static void gold_spawn_holes(GPUtronicHandle* h, int want) {
    if (!h->idle_exit || !h->h_slot || !h->side_on) return;
    if (h->h_ctrl->control_flags & GPUTRONIC_FLAG_STOP) return;
    if (want < GPUTRONIC_SM_FLOOR) want = GPUTRONIC_SM_FLOOR;
    if (want > h->cfg.num_sm) want = h->cfg.num_sm;
    int s = 0;
    while (s < want) {
        if (h->h_slot[s] != 0u) { s++; continue; }
        int lo = s;
        while (s < want && h->h_slot[s] == 0u) s++;
        int n = s - lo;
        for (int i = lo; i < lo + n; i++) {
            unsigned int g = ++h->gen_tick;
            if (g == 0u) g = ++h->gen_tick;
            h->h_slot[i] = g;
            if (h->h_smid) h->h_smid[i] = 0xFFFFFFFFu;
        }
        __sync_synchronize();
        double t0 = now_us();
        cudaStream_t wave;
        cudaError_t err = cudaSuccess;
        const char* how = "ok";
        if (h->n_waves >= 64) {
            err = cudaErrorUnknown;
            how = "wave cap";
        } else if ((err = cudaStreamCreateWithFlags(&wave, cudaStreamNonBlocking)) != cudaSuccess) {
            how = cudaGetErrorString(err);
        } else {
            h->wave_stream[h->n_waves++] = wave;
            gold_launch_this(h, n, lo, wave);
            err = cudaGetLastError();
            if (err != cudaSuccess) how = cudaGetErrorString(err);
        }
        h->spawn_count++;
        printf("[SPAWN] base=%d n=%d us=%.0f count=%u %s\n",
               lo, n, now_us() - t0, h->spawn_count, how);
        fflush(stdout);
        if (err != cudaSuccess) {
            for (int i = lo; i < lo + n; i++) h->h_slot[i] = 0u;
            return;
        }
        gold_note_smids(h, n, lo);
    }
}

/* Test switches. Unset, this returns without a launch. Relaunch retires the
 * whole grid. Idle-exit only fills holes below the command. */
static void gold_service_width(GPUtronicHandle* h, int n_cmd) {
    if (!h || !h->h_ctrl) return;
    if (h->h_ctrl->control_flags & GPUTRONIC_FLAG_STOP) return;
    if (n_cmd < GPUTRONIC_SM_FLOOR) n_cmd = GPUTRONIC_SM_FLOOR;
    if (n_cmd > h->cfg.num_sm) n_cmd = h->cfg.num_sm;
    if (h->relaunch) {
        if (!h->kstream_on || n_cmd == h->launched_n) return;
        double now = now_us();
        if (h->launched_at_us > 0.0 && now - h->launched_at_us < (double)h->relaunch_hold_us)
            return;
        int from = h->launched_n;
        double t0 = now_us();
        h->h_ctrl->control_flags |= GPUTRONIC_FLAG_RETIRE;
        __sync_synchronize();
        cudaError_t err = cudaStreamSynchronize(h->kstream);
        double t1 = now_us();
        h->h_ctrl->control_flags &= ~GPUTRONIC_FLAG_RETIRE;
        __sync_synchronize();
        if (err != cudaSuccess) {
            printf("[RELAUNCH] sync failed: %s\n", cudaGetErrorString(err));
            fflush(stdout);
            return;
        }
        if (h->h_smid) {
            for (int i = 0; i < h->cfg.num_sm; i++) h->h_smid[i] = 0xFFFFFFFFu;
        }
        gold_launch_this(h, n_cmd, 0, h->kstream);
        err = cudaGetLastError();
        h->launched_n = n_cmd;
        h->launched_at_us = now_us();
        h->smid_expect = n_cmd;
        h->relaunch_count++;
        printf("[RELAUNCH] %d -> %d sync_us=%.0f launch_us=%.0f count=%u %s\n",
               from, n_cmd, t1 - t0, h->launched_at_us - t1, h->relaunch_count,
               err == cudaSuccess ? "ok" : cudaGetErrorString(err));
        fflush(stdout);
        if (err == cudaSuccess) gold_note_smids(h, n_cmd, 0);
        return;
    }
    if (h->idle_exit) gold_spawn_holes(h, n_cmd);
}

static void gold_reset_rate_state(GPUtronicHandle* h) {
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
    GPUtronicHandle* h = (GPUtronicHandle*)arg;
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
        int c[4] = {N * 3 / 4, N / 2, N / 4, GPUTRONIC_SM_FLOOR};
        h->n_probe = 0;
        for (int i = 0; i < 4; i++) {
            int n = c[i] < GPUTRONIC_SM_FLOOR ? GPUTRONIC_SM_FLOOR : c[i];
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

        /* Rate over the window, not one 10 µs tick [F-05]. Unset keeps 8 ms. */
        unsigned long long pulses_t = h->h_ctrl->tensor_work_pulses;
        h->rate_win_pulses_t += pulses_t - h->prev_pulses_t;
        h->prev_pulses_t = pulses_t;
        h->rate_win_pulses += dpulse;
        h->rate_win_dt += (double)dt;
        if (h->rate_win_dt >= h->rate_win_s) {
            float inst_rate = (float)((double)h->rate_win_pulses / h->rate_win_dt);
            float inst_t = (float)((double)h->rate_win_pulses_t / h->rate_win_dt);
            h->last_win_pulses = h->rate_win_pulses;
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
            if (n_cmd < GPUTRONIC_SM_FLOOR || n_cmd > N) n_cmd = N;
            float i_before = h->integral;
            h->st_ticks++;
            if (fabsf(error) < h->cfg.deadband) h->st_db_skip++;
            if (fabsf(error) >= h->cfg.deadband) {
                float u_p = h->cfg.kp * error - h->cfg.kd * h->dzdt_hat;
                h->last_p_term = u_p;
                float i_new = h->integral + error * dt;
                if (i_new > h->iclamp) i_new = h->iclamp;
                if (i_new < -h->iclamp) i_new = -h->iclamp;
                float n_try = h->n_ff - (u_p + h->cfg.ki * i_new) * h->sms_per_u;
                int saturated = (n_try < (float)GPUTRONIC_SM_FLOOR && error > 0.0f) ||
                                (n_try > (float)N && error < 0.0f);
                if (!saturated) h->integral = i_new;
                float u = u_p + h->cfg.ki * h->integral;
                float nf = h->n_ff - u * h->sms_per_u;
                n_cmd = (int)floorf(nf + 0.5f);
            }
            int i_changed = fabsf(h->integral - i_before) > 1e-6f;
            if (i_changed) h->st_int_upd++;
            if (h->iclamp > 0.0f && fabsf(h->integral) >= 0.999f * h->iclamp) h->st_rail++;
            int nc = n_cmd < GPUTRONIC_SM_FLOOR ? GPUTRONIC_SM_FLOOR : (n_cmd > N ? N : n_cmd);
            if (nc != h->held_sms) { h->st_moves++; if (!i_changed) h->st_p_only++; }
        }
        h->st_loop++;

        if (n_cmd < GPUTRONIC_SM_FLOOR) n_cmd = GPUTRONIC_SM_FLOOR;
        if (n_cmd > N) n_cmd = N;
        h->held_sms = n_cmd;
        h->h_ctrl->active_sms = n_cmd;
        __sync_synchronize();
        gold_service_width(h, n_cmd);

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
__global__ void gold_zc_probe_kernel(GPUtronicControl* ctrl) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int v = *(volatile int*)&ctrl->proactive_corrections;
        *(volatile int*)&ctrl->control_hz_ema = ~v;
        __threadfence_system();
    }
}

static int gold_verify_zero_copy(GPUtronicHandle* h) {
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
unsigned int* gputronic_smid_device(GPUtronicHandle* h) {
    return (h && h->d_smid) ? h->d_smid : NULL;
}

void gputronic_log_smids(GPUtronicHandle* h) {
    if (!h || !h->h_smid || h->cfg.num_sm <= 0) {
        printf("[SMID] no map\n");
        return;
    }
    int n = h->cfg.num_sm;
    int nwait = (h->smid_expect > 0 && h->smid_expect < n) ? h->smid_expect : n;
    int dev = h->device_sms > 0 ? h->device_sms : n;
    if (dev > 256) dev = 256;
    volatile unsigned int* map = h->h_smid;
    for (int spin = 0; spin < 500; spin++) {
        int ready = 1;
        for (int i = 0; i < nwait; i++) {
            if (map[i] == 0xFFFFFFFFu) { ready = 0; break; }
        }
        if (ready) break;
        usleep(1000);
    }
    unsigned char seen[256];
    memset(seen, 0, sizeof seen);
    int ready_n = 0;
    printf("[SMID] launched %d blocks on a %d-SM device\n", n, h->device_sms);
    printf("[SMID] block:smid");
    for (int i = 0; i < n; i++) {
        unsigned int id = map[i];
        printf(" %d:%u", i, id);
        if (id != 0xFFFFFFFFu) {
            ready_n++;
            if (id < 256) seen[id] = 1;
        }
    }
    printf("\n");
    printf("[SMID] left out:");
    int left = 0;
    for (int id = 0; id < dev; id++) {
        if (!seen[id]) { printf(" %d", id); left++; }
    }
    if (left == 0) printf(" none");
    printf("\n");
    printf("[SMID] recorded %d of %d\n", ready_n, n);
    fflush(stdout);
}

GPUtronicHandle* gputronic_create(const GPUtronicConfig* cfg_in) {
    /* Map the shared control block and query the SM count. No copy-based fallback [F-01]. */
    GPUtronicHandle* h = (GPUtronicHandle*)calloc(1, sizeof(GPUtronicHandle));
    if (!h) return NULL;
    h->rate_win_s = 0.008;

    if (cfg_in)
        h->cfg = *cfg_in;
    else
        gputronic_config_gold(&h->cfg);

    g_signal_stop = 0;
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPUtronic] cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        free(h);
        return NULL;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    h->device_sms = prop.multiProcessorCount;
    if (h->cfg.num_sm <= 0) h->cfg.num_sm = h->device_sms;
    /* Default leaves two SMs with no block. GPUTRONIC_OS_RESERVE overrides
     * that for one run. Unset keeps the default. */
    int reserve = GPUTRONIC_OS_RESERVE_SMS;
    const char* reserve_env = getenv("GPUTRONIC_OS_RESERVE");
    if (reserve_env && *reserve_env) reserve = atoi(reserve_env);
    if (reserve < 0) reserve = 0;
    if (h->device_sms > GPUTRONIC_SM_FLOOR && reserve > h->device_sms - GPUTRONIC_SM_FLOOR)
        reserve = h->device_sms - GPUTRONIC_SM_FLOOR;
    if (h->device_sms > reserve + GPUTRONIC_SM_FLOOR
        && h->cfg.num_sm == h->device_sms)
        h->cfg.num_sm = h->device_sms - reserve;
    if (reserve_env && *reserve_env)
        printf("[GPU] GPUTRONIC_OS_RESERVE=%d for this run\n", reserve);
    h->h_smid = NULL;
    h->d_smid = NULL;
    if (h->cfg.num_sm > 0 &&
        cudaHostAlloc((void**)&h->h_smid,
                      (size_t)h->cfg.num_sm * sizeof(unsigned int),
                      cudaHostAllocMapped) == cudaSuccess) {
        for (int i = 0; i < h->cfg.num_sm; i++) h->h_smid[i] = 0xFFFFFFFFu;
        if (cudaHostGetDevicePointer((void**)&h->d_smid, h->h_smid, 0) != cudaSuccess) {
            cudaFreeHost(h->h_smid);
            h->h_smid = NULL;
            h->d_smid = NULL;
        }
    }
    if (h->cfg.threads_per_block <= 0) h->cfg.threads_per_block = 64;
    if (h->cfg.control_dt_us <= 0) h->cfg.control_dt_us = 10;

    /* Mapped control block */
    err = cudaHostAlloc((void**)&h->h_ctrl, sizeof(GPUtronicControl), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPUtronic] cudaHostAlloc control failed: %s\n", cudaGetErrorString(err));
        if (h->h_smid) cudaFreeHost(h->h_smid);
        free(h);
        return NULL;
    }
    err = cudaHostGetDevicePointer((void**)&h->d_ctrl, h->h_ctrl, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPUtronic] cudaHostGetDevicePointer failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h->h_ctrl);
        if (h->h_smid) cudaFreeHost(h->h_smid);
        free(h);
        return NULL;
    }

    memset(h->h_ctrl, 0, sizeof(GPUtronicControl));
    h->zero_copy_ok = gold_verify_zero_copy(h);
    if (!h->zero_copy_ok) {
        fprintf(stderr, "[GPUtronic] zero-copy verification failed; no PCIe fallback\n");
        cudaFreeHost(h->h_ctrl);
        if (h->h_smid) cudaFreeHost(h->h_smid);
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
        fprintf(stderr, "[GPUtronic] cudaMalloc sm counters failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h->h_ctrl);
        if (h->h_smid) cudaFreeHost(h->h_smid);
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
    printf("[GPUtronic Gold %s]\n", GPUTRONIC_VERSION_STRING);
    printf("═══════════════════════════════════════════════════════════\n");
    printf("[GPU] %s | SMs=%d (launching %d, %d reserved for the OS) | thr/block=%d\n",
           prop.name, h->device_sms, h->cfg.num_sm, h->device_sms - h->cfg.num_sm,
           h->cfg.threads_per_block);
    printf("[LAW] target_z=%.2f Kp=%.3f Ki=%.4f Kd=%.4f deadband=%.3f width-actuator two-sided\n",
           h->cfg.target_z, h->cfg.kp, h->cfg.ki, h->cfg.kd, h->cfg.deadband);
    printf("[LOOP] dt=%d us (%.0f kHz) active_sms=%d floor=%d\n\n",
           h->cfg.control_dt_us, 1000.0 / h->cfg.control_dt_us, h->cfg.num_sm,
           GPUTRONIC_SM_FLOOR);

    return h;
}

/* Test-only. GPUTRONIC_ACTIVE_PASSES=0,1,3 or GPUTRONIC_ACTIVE_SMS=0-11.
 * Both at once is refused. The mask stays dark until FLAG_SELECT. */
static void gold_parse_select(GPUtronicHandle* h) {
    h->select_pass_mask = 0;
    h->select_sm_lo = 0;
    h->select_sm_hi = 0;
    const char* ps = getenv("GPUTRONIC_ACTIVE_PASSES");
    const char* ss = getenv("GPUTRONIC_ACTIVE_SMS");
    int have_p = ps && *ps;
    int have_s = ss && *ss;
    if (have_p && have_s) {
        printf("[SELECT] both ACTIVE_PASSES and ACTIVE_SMS are set; using neither\n");
        return;
    }
    if (have_p) {
        char buf[128];
        strncpy(buf, ps, sizeof buf - 1);
        buf[sizeof buf - 1] = 0;
        unsigned int mask = 0;
        for (char* t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
            int v = atoi(t);
            if (v >= 0 && v < 7) mask |= 1u << v;
            else printf("[SELECT] ignoring pass %d\n", v);
        }
        h->select_pass_mask = mask;
        printf("[SELECT] passes '%s' mask 0x%x (off until calibration ends)\n", ps, mask);
    }
    if (have_s) {
        char buf[256];
        strncpy(buf, ss, sizeof buf - 1);
        buf[sizeof buf - 1] = 0;
        for (char* t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
            int a = 0, b = 0, n = 0;
            if (sscanf(t, "%d-%d", &a, &b) == 2) {
                if (a > b) { int tmp = a; a = b; b = tmp; }
                n = 2;
            } else if (sscanf(t, "%d", &a) == 1) {
                b = a;
                n = 1;
            }
            if (n == 0) continue;
            for (int s = a; s <= b; s++) {
                if (s < 0 || s >= 128) continue;
                if (s < 64) h->select_sm_lo |= 1ull << s;
                else h->select_sm_hi |= 1ull << (s - 64);
            }
        }
        printf("[SELECT] physical SMs '%s' (off until calibration ends)\n", ss);
    }
}

int gputronic_start(GPUtronicHandle* h) {
    if (!h || h->started) return -1;

    /* CSV. GPUTRONIC_NO_CSV=1 skips the 10 us file for one run. */
    const char* no_csv = getenv("GPUTRONIC_NO_CSV");
    const char* path = h->cfg.csv_path ? h->cfg.csv_path : "results/gputronic_gold.csv";
    if (no_csv && strcmp(no_csv, "1") == 0) {
        h->csv = NULL;
        printf("[CSV] off for this run\n");
    } else {
        /* best-effort results dir */
        int mk = system("mkdir -p results");
        (void)mk;
        h->csv = fopen(path, "w");
        if (h->csv) {
            fprintf(h->csv, "time_s,z_raw,z_hat,dzdt,rate_ema,active_sms,error,integral,ctrl_hz,rate_ref,"
                             "z_cuda,z_tensor,rate_tensor_ema,rate_ref_tensor,ctrl_sensor\n");
            printf("[CSV] %s\n", path);
        } else {
            fprintf(stderr, "[WARN] could not open CSV %s\n", path);
        }
    }

    h->t0_us = now_us();
    h->running = 1;

    if (h->cfg.launch_selftest_kernel) {
        const char* wc = getenv("GPUTRONIC_WORK_COUNTER");   /* T9; off unless "1" */
        if (wc && strcmp(wc, "1") == 0 && !h->d_work) {
            size_t bytes = (size_t)h->cfg.num_sm * 32 * sizeof(unsigned long long);
            if (cudaMalloc((void**)&h->d_work, bytes) == cudaSuccess) cudaMemset(h->d_work, 0, bytes);
            else h->d_work = NULL;
            printf("[T9] work counter %s\n", h->d_work ? "on (device memory)" : "allocation failed, off");
        }
        /* T11 accumulator: GPUTRONIC_SM_ACCUM=1 (all SMs) or GPUTRONIC_SM_ACCUM_SM=<id> (one SM). */
        const char* ea = getenv("GPUTRONIC_SM_ACCUM");
        const char* es = getenv("GPUTRONIC_SM_ACCUM_SM");
        if (((ea && strcmp(ea, "1") == 0) || (es && *es)) && !h->h_accum) {
            size_t bytes = (size_t)h->cfg.num_sm * GOLD_ACCUM_WORDS * sizeof(unsigned long long);
            if (cudaHostAlloc((void**)&h->h_accum, bytes, cudaHostAllocMapped) == cudaSuccess &&
                cudaHostGetDevicePointer((void**)&h->d_accum, h->h_accum, 0) == cudaSuccess) {
                memset(h->h_accum, 0, bytes);
                h->accum_sm = (es && *es) ? atoi(es) : -1;
                if (h->accum_sm >= h->cfg.num_sm) {
                    printf("[T11] SM %d is past the device count %d; reporting all SMs\n",
                           h->accum_sm, h->cfg.num_sm);
                    h->accum_sm = -1;
                }
                const char* bv = getenv("GPUTRONIC_TRACE_BIN_US");
                long bus = (bv && atoi(bv) >= 100) ? atol(bv) : 1000;
                h->accum_bin_ns = (unsigned long long)bus * 1000ULL;
                if (h->accum_sm < 0)
                    printf("[T11] per-SM accumulator on: all SMs, bin %ld us\n", bus);
                else
                    printf("[T11] per-SM accumulator on: SM %d, bin %ld us\n", h->accum_sm, bus);
            } else {
                if (h->h_accum) cudaFreeHost(h->h_accum);
                h->h_accum = NULL; h->d_accum = NULL;
                printf("[T11] accumulator allocation failed, off\n");
            }
        }
        if (!h->h_accum) { h->d_accum = NULL; h->accum_sm = -1; h->accum_bin_ns = 1000000ULL; }
        h->idle_quiet = 0;
        {
            const char* iq = getenv("GPUTRONIC_IDLE_QUIET");
            if (iq && strcmp(iq, "1") == 0) h->idle_quiet = 1;
        }
        if (h->idle_quiet)
            printf("[IDLE] quiet on: idle blocks sample the mapped words once per 1 ms\n");
        {
            const char* ex = getenv("GPUTRONIC_IDLE_EXIT");
            const char* rl = getenv("GPUTRONIC_RELAUNCH");
            h->idle_exit = (ex && strcmp(ex, "1") == 0) ? 1 : 0;
            h->relaunch = (rl && strcmp(rl, "1") == 0) ? 1 : 0;
            if (h->idle_exit && h->relaunch) {
                printf("[IDLE] both IDLE_EXIT and RELAUNCH are set; using neither\n");
                h->idle_exit = 0;
                h->relaunch = 0;
            }
        }
        h->idle_work = 0;
        {
            const char* iw = getenv("GPUTRONIC_IDLE_WORK");
            if (iw && strcmp(iw, "1") == 0) h->idle_work = 1;
        }
        if (h->idle_work) {
            printf("[IDLE] work on: blocks above active_sms do the work and add to total_work_pulses\n");
            if (h->idle_exit || h->relaunch || h->idle_quiet) {
                printf("[IDLE] IDLE_WORK is set; exit, relaunch, and quiet are off for this run\n");
                h->idle_exit = 0;
                h->relaunch = 0;
                h->idle_quiet = 0;
            }
        }
        if (h->relaunch) {
            long hold = 5000;
            const char* hs = getenv("GPUTRONIC_RELAUNCH_HOLD_US");
            if (hs && *hs) hold = atol(hs);
            if (hold < 500) hold = 500;
            if (hold > 1000000) hold = 1000000;
            h->relaunch_hold_us = hold;
            printf("[IDLE] relaunch on: new grid at the commanded count, hold %ld us\n", hold);
        }
        if (h->idle_exit) {
            printf("[IDLE] exit on: an idle block returns; the host launches replacements\n");
            if (cudaHostAlloc((void**)&h->h_slot,
                              (size_t)h->cfg.num_sm * sizeof(unsigned int),
                              cudaHostAllocMapped) == cudaSuccess &&
                cudaHostGetDevicePointer((void**)&h->d_slot, h->h_slot, 0) == cudaSuccess) {
                h->gen_tick = 1u;
                for (int i = 0; i < h->cfg.num_sm; i++) h->h_slot[i] = 1u;
            } else {
                if (h->h_slot) cudaFreeHost(h->h_slot);
                h->h_slot = NULL;
                h->d_slot = NULL;
                h->idle_exit = 0;
                printf("[IDLE] slot map failed; exit switch is off for this run\n");
            }
        }
        h->peak_reps = 1;
        h->peak_chains = 1;
        h->peak_acc = 0;
        h->peak_rounds = 32;
        h->peak_live = 0;
        h->work_load = 0;
        {
            const char* pr = getenv("GPUTRONIC_PEAK_REPS");
            const char* pc = getenv("GPUTRONIC_PEAK_CHAINS");
            const char* pa = getenv("GPUTRONIC_PEAK_ACC");
            const char* po = getenv("GPUTRONIC_PEAK_ROUNDS");
            if (pr && *pr) { h->peak_reps = atoi(pr); h->peak_live = 1; }
            if (pc && *pc) { h->peak_chains = atoi(pc); h->peak_live = 1; }
            if (h->peak_reps < 1) h->peak_reps = 1;
            if (h->peak_chains < 1) h->peak_chains = 1;
            if (h->peak_reps > 256) h->peak_reps = 256;
            if (h->peak_chains > 16) h->peak_chains = 16;
            if (pa && *pa) {
                int a = atoi(pa);
                if (a == 4 || a == 8 || a == 16 || a == 32) {
                    h->peak_acc = a;
                    h->peak_live = 1;
                } else {
                    printf("[PEAK] acc %d is not 4, 8, 16, or 32; independent nest is off\n", a);
                }
            }
            if (po && *po) {
                h->peak_rounds = atoi(po);
                if (h->peak_acc <= 0)
                    printf("[PEAK] rounds apply only with PEAK_ACC; this run uses the other nest\n");
            }
            if (h->peak_rounds < 1) h->peak_rounds = 1;
            if (h->peak_rounds > 4096) h->peak_rounds = 4096;
        }
        if (h->peak_acc > 0)
            printf("[PEAK] indep acc=%d rounds=%d threads=%d\n",
                   h->peak_acc, h->peak_rounds, h->cfg.threads_per_block);
        else if (h->peak_live || h->cfg.threads_per_block != GPUTRONIC_DEFAULT_THREADS_PER_BLOCK)
            printf("[PEAK] reps=%d chains=%d threads=%d\n",
                   h->peak_reps, h->peak_chains, h->cfg.threads_per_block);
        {
            const char* wl = getenv("GPUTRONIC_WORK_LOAD");
            if (wl && strcmp(wl, "1") == 0) {
                if (h->peak_acc > 0) {
                    h->work_load = 1;
                    printf("[WORK] real-work load: scale sets independent-FMA rounds, every batch pulses\n");
                } else {
                    printf("[WORK] GPUTRONIC_WORK_LOAD needs PEAK_ACC; switch is off for this run\n");
                }
            }
        }
        if (h->idle_exit || h->relaunch) {
            if (cudaStreamCreateWithFlags(&h->side_stream, cudaStreamNonBlocking) == cudaSuccess)
                h->side_on = 1;
        }
        if (h->relaunch) {
            if (cudaStreamCreateWithFlags(&h->kstream, cudaStreamNonBlocking) == cudaSuccess)
                h->kstream_on = 1;
            else
                printf("[IDLE] relaunch stream failed; this run keeps the resident grid\n");
            if (!h->kstream_on) h->relaunch = 0;
        }
        if (h->idle_exit && !h->side_on) {
            printf("[IDLE] side stream failed; exit switch is off for this run\n");
            h->idle_exit = 0;
        }
        gold_parse_select(h);
        {
            const char* pl = getenv("GPUTRONIC_PULSE_LOCAL");
            h->pulse_local = (pl && strcmp(pl, "1") == 0) ? 1 : 0;
        }
        if (h->pulse_local)
            printf("[PULSE] local count: one mapped atomic per block per 1 ms\n");
        h->ctrl_cache_ns = GOLD_PULSE_LOCAL_NS;
        {
            const char* cc = getenv("GPUTRONIC_CTRL_CACHE");
            h->ctrl_cache = (cc && strcmp(cc, "1") == 0) ? 1 : 0;
        }
        if (h->ctrl_cache) {
            /* Test only. Default 1 ms. Floor 50 us so a typo cannot hide STOP.
             * The pulse batch period is still GOLD_PULSE_LOCAL_NS. */
            const char* us = getenv("GPUTRONIC_CTRL_CACHE_US");
            long v = 1000;
            if (us && *us) {
                v = atol(us);
                if (v < 50) v = 50;
                if (v > 100000) v = 100000;
            }
            h->ctrl_cache_ns = (unsigned long long)v * 1000ULL;
            printf("[CTRL] cached control words, including work_scale_milli: leader reads once per %ld us\n", v);
        }
        /* Test only. Unset, the persistent kernel stays on the default stream,
         * as in the tenth-pass binary. GPUTRONIC_GOLD_NONBLOCK=1 launches it
         * on a non-blocking stream so a second kernel is not queued behind a
         * grid that does not return. */
        if (!h->kstream_on) {
            const char* nb = getenv("GPUTRONIC_GOLD_NONBLOCK");
            if (nb && strcmp(nb, "1") == 0) {
                if (cudaStreamCreateWithFlags(&h->kstream, cudaStreamNonBlocking) == cudaSuccess) {
                    h->kstream_on = 1;
                    printf("[GPU] persistent kernel on a non-blocking stream\n");
                } else {
                    printf("[GPU] non-blocking launch stream failed; using the default stream\n");
                }
            }
        }
        cudaStream_t launch_stream = h->kstream_on ? h->kstream : 0;
        gold_launch_this(h, h->cfg.num_sm, 0, launch_stream);
        h->launched_n = h->cfg.num_sm;
        h->launched_at_us = now_us();
        h->smid_expect = h->cfg.num_sm;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[GPUtronic] kernel launch failed: %s\n", cudaGetErrorString(err));
            h->running = 0;
            if (h->csv) fclose(h->csv);
            h->csv = NULL;
            return -1;
        }
        printf("[OK] Persistent self-test kernel launched (%d blocks x %d threads)\n",
               h->cfg.num_sm, h->cfg.threads_per_block);
        gputronic_log_smids(h);
        if (h->h_smid && h->cfg.num_sm > 0) {
            h->smid_boot = (unsigned int*)malloc((size_t)h->cfg.num_sm * sizeof(unsigned int));
            if (h->smid_boot)
                memcpy(h->smid_boot, h->h_smid, (size_t)h->cfg.num_sm * sizeof(unsigned int));
        }
    }

    /* Test only. Unset keeps the 8 ms literal. A set value is clamped. */
    h->rate_win_s = 0.008;
    {
        const char* rw = getenv("GPUTRONIC_RATE_WIN_US");
        if (rw && *rw) {
            long v = atol(rw);
            if (v < 500) v = 500;
            if (v > 100000) v = 100000;
            h->rate_win_s = (double)v * 1e-6;
            printf("[RATE] window %ld us (unset keeps 8000)\n", v);
        }
    }

    if (pthread_create(&h->thr, NULL, control_thread, h) != 0) {
        fprintf(stderr, "[GPUtronic] pthread_create failed\n");
        h->h_ctrl->control_flags |= GPUTRONIC_FLAG_STOP;
        h->running = 0;
        return -1;
    }

    h->started = 1;
    return 0;
}

static void gold_guest_halt(GPUtronicHandle* h);

void gputronic_stop(GPUtronicHandle* h) {
    if (!h || !h->started) return;
    h->running = 0;
    g_signal_stop = 1;
    h->gold_stop_us = now_us();
    if (h->h_ctrl) h->h_ctrl->control_flags |= GPUTRONIC_FLAG_STOP;
    gold_guest_halt(h);
    pthread_join(h->thr, NULL);
    cudaDeviceSynchronize();
    if (h->idle_exit || h->relaunch) {
        printf("[IDLE] spawns=%u relaunches=%u\n", h->spawn_count, h->relaunch_count);
        if (h->idle_exit && h->h_slot && h->h_smid) {
            unsigned char seen[256];
            memset(seen, 0, sizeof seen);
            int live = 0;
            int dev = h->device_sms > 0 && h->device_sms < 256 ? h->device_sms : 84;
            printf("[SMID] live");
            for (int i = 0; i < h->cfg.num_sm; i++) {
                if (h->h_slot[i] == 0u) continue;
                unsigned int id = h->h_smid[i];
                printf(" %d:%u", i, id);
                live++;
                if (id < 256) seen[id] = 1;
            }
            printf("\n[SMID] live_count=%d left out:", live);
            int left = 0;
            for (int id = 0; id < dev; id++) {
                if (!seen[id]) { printf(" %d", id); left++; }
            }
            if (left == 0) printf(" none");
            printf("\n");
        }
        gputronic_log_smids(h);
    }

    if (h->csv) {
        fflush(h->csv);
        fclose(h->csv);
        h->csv = NULL;
    }

    if (h->n_z > 0) {
        double mean = h->sum_z / (double)h->n_z;
        double var = h->sum_z2 / (double)h->n_z - mean * mean;
        if (var < 0) var = 0;
        printf("\n[STATS] loop_iters=%llu pi_ticks=%llu deadband_held=%llu integral_updates=%llu "
               "integral_on_rail=%llu sm_moves=%llu p_only_moves=%llu\n",
               h->st_loop, h->st_ticks, h->st_db_skip, h->st_int_upd, h->st_rail, h->st_moves, h->st_p_only);
        printf("\n[STATS] samples=%lld mean_Z=%.4f std_Z=%.4f active_sms=%d..%d rows_csv=%lld\n",
               (long long)h->n_z, mean, sqrt(var), h->min_sms, h->max_sms,
               (long long)h->csv_rows);
    }
    h->started = 0;
    printf("[GPUtronic Gold] Stopped cleanly.\n");
}

void gputronic_destroy(GPUtronicHandle* h) {
    if (!h) return;
    if (h->started) gputronic_stop(h);
    if (h->smid_boot) free(h->smid_boot);
    for (int i = 0; i < h->n_waves && i < 64; i++) cudaStreamDestroy(h->wave_stream[i]);
    if (h->guest_stream_on) cudaStreamDestroy(h->guest_stream);
    if (h->h_guest_pulses) cudaFreeHost(h->h_guest_pulses);
    if (h->h_guest_stop) cudaFreeHost(h->h_guest_stop);
    if (h->h_guest_live) cudaFreeHost(h->h_guest_live);
    if (h->h_guest_mark) cudaFreeHost(h->h_guest_mark);
    if (h->side_on) cudaStreamDestroy(h->side_stream);
    if (h->kstream_on) cudaStreamDestroy(h->kstream);
    if (h->h_slot) cudaFreeHost(h->h_slot);
    if (h->d_sm) cudaFree(h->d_sm);
    if (h->h_smid) cudaFreeHost(h->h_smid);
    if (h->d_work) cudaFree(h->d_work);
    if (h->h_accum) cudaFreeHost(h->h_accum);
    if (h->h_ctrl) cudaFreeHost(h->h_ctrl);
    free(h);
}

void gputronic_set_target(GPUtronicHandle* h, float target_z) {
    if (!h) return;
    h->cfg.target_z = target_z;
    if (h->h_ctrl) h->h_ctrl->target_z = target_z;
}

void gputronic_set_gains(GPUtronicHandle* h, float kp, float ki) {
    if (!h) return;
    h->cfg.kp = kp;
    h->cfg.ki = ki;
}

void gputronic_set_kd(GPUtronicHandle* h, float kd) {
    if (!h) return;
    h->cfg.kd = kd;
}

void gputronic_set_deadband(GPUtronicHandle* h, float deadband) {
    if (!h) return;
    h->cfg.deadband = (deadband < 0.0f) ? 0.0f : deadband;
}

void gputronic_set_open_loop_sms(GPUtronicHandle* h, int n) {
    if (!h) return;
    h->open_loop_sms = n;
}

int gputronic_reset_counters(GPUtronicHandle* h) {
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

float gputronic_get_z(const GPUtronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->z_hat : 0.0f;
}

float gputronic_get_z_raw(const GPUtronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->z_raw : 0.0f;
}

float gputronic_get_rate(const GPUtronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->rate_ema : 0.0f;
}

float gputronic_get_z_cuda(const GPUtronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->z_cuda : 0.0f;
}

float gputronic_get_z_tensor(const GPUtronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->z_tensor : 0.0f;
}

float gputronic_get_rate_tensor(const GPUtronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->rate_tensor_ema : 0.0f;
}

int gputronic_get_ctrl_sensor(const GPUtronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->ctrl_sensor : 0;
}

int gputronic_get_active_sms(const GPUtronicHandle* h) {
    return (h && h->h_ctrl) ? h->h_ctrl->active_sms : 0;
}

struct GPUtronicControl* gputronic_get_control(GPUtronicHandle* h) {
    return h ? h->h_ctrl : NULL;
}

void gputronic_pulse(GPUtronicHandle* h, unsigned long long units) {
    if (!h || !h->h_ctrl || units == 0) return;
    /* Host atomic RMW (was a plain +=, which lost updates against itself and
     * against device atomics). See header for the device-coherence caveat. */
    __sync_fetch_and_add((unsigned long long*)&h->h_ctrl->total_work_pulses, units);
}

int gputronic_zero_copy_ok(const GPUtronicHandle* h) {
    return h ? h->zero_copy_ok : 0;
}

/* Test only. Unset, this returns -1 and launches nothing. The guest pulse
 * word is its own mapped allocation, not a field of GPUtronicControl. */
static int gold_guest_open(GPUtronicHandle* h, int blocks) {
    if (!h || blocks <= 0) return -1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[GUEST] concurrentKernels=%d\n", prop.concurrentKernels);
    if (blocks > h->device_sms) blocks = h->device_sms;
    if (blocks > 84) blocks = 84;
    int acc = 32;
    int rounds = 512;
    const char* pa = getenv("GPUTRONIC_PEAK_ACC");
    const char* po = getenv("GPUTRONIC_PEAK_ROUNDS");
    if (h->peak_acc == 4 || h->peak_acc == 8 || h->peak_acc == 16 || h->peak_acc == 32)
        acc = h->peak_acc;
    else if (pa && *pa) {
        int a = atoi(pa);
        if (a == 4 || a == 8 || a == 16 || a == 32) acc = a;
        else printf("[GUEST] acc %d is not 4, 8, 16, or 32; using 32\n", a);
    }
    if (h->peak_acc > 0 && h->peak_rounds > 0) rounds = h->peak_rounds;
    else if (po && *po) rounds = atoi(po);
    if (rounds < 1) rounds = 1;
    if (rounds > 4096) rounds = 4096;
    if (cudaHostAlloc((void**)&h->h_guest_pulses, sizeof(unsigned long long),
                      cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**)&h->d_guest_pulses, h->h_guest_pulses, 0) != cudaSuccess ||
        cudaHostAlloc((void**)&h->h_guest_stop, sizeof(int), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**)&h->d_guest_stop, h->h_guest_stop, 0) != cudaSuccess ||
        cudaHostAlloc((void**)&h->h_guest_live, sizeof(int), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**)&h->d_guest_live, h->h_guest_live, 0) != cudaSuccess ||
        cudaHostAlloc((void**)&h->h_guest_mark, 84ULL * 4ULL * sizeof(unsigned long long),
                      cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**)&h->d_guest_mark, h->h_guest_mark, 0) != cudaSuccess) {
        printf("[GUEST] mapped counter failed; guest is off for this run\n");
        return -1;
    }
    *h->h_guest_pulses = 0ULL;
    *h->h_guest_stop = 0;
    *h->h_guest_live = 0;
    memset(h->h_guest_mark, 0, 84ULL * 4ULL * sizeof(unsigned long long));
    {
        const char* gp = getenv("GPUTRONIC_GUEST_PRIORITY");
        if (gp && strcmp(gp, "1") == 0) {
            int least = 0, greatest = 0;
            cudaError_t pr = cudaDeviceGetStreamPriorityRange(&least, &greatest);
            cudaError_t cr = (pr == cudaSuccess)
                ? cudaStreamCreateWithPriority(&h->guest_stream, cudaStreamNonBlocking, greatest)
                : pr;
            if (cr != cudaSuccess) {
                printf("[GUEST] priority stream failed; guest is off for this run\n");
                return -1;
            }
            printf("[GUEST] stream priority=%d (highest; range %d..%d)\n",
                   greatest, greatest, least);
        } else if (cudaStreamCreateWithFlags(&h->guest_stream, cudaStreamNonBlocking) != cudaSuccess) {
            printf("[GUEST] stream failed; guest is off for this run\n");
            return -1;
        }
    }
    h->guest_stream_on = 1;
    h->guest_blocks = blocks;
    h->guest_acc = acc;
    h->guest_rounds = rounds;
    h->guest_skip0 = h->device_sms >= 2 ? h->device_sms - 2 : -1;
    h->guest_skip1 = h->device_sms >= 1 ? h->device_sms - 1 : -1;
    printf("[GUEST] blocks=%d acc=%d rounds=%d threads=%d skip_sm=%d,%d\n",
           blocks, acc, rounds, h->cfg.threads_per_block, h->guest_skip0, h->guest_skip1);
    return 0;
}

static int gold_guest_launch(GPUtronicHandle* h) {
    if (!h || h->guest_launched || h->guest_blocks <= 0 || !h->guest_stream_on) return -1;
    *h->h_guest_pulses = 0ULL;
    *h->h_guest_stop = 0;
    *h->h_guest_live = 0;
    if (h->h_guest_mark)
        memset(h->h_guest_mark, 0, 84ULL * 4ULL * sizeof(unsigned long long));
    memset(h->guest_enter_seen_us, 0, sizeof h->guest_enter_seen_us);
    memset(h->guest_pulse_seen_us, 0, sizeof h->guest_pulse_seen_us);
    __sync_synchronize();
    int blocks = h->guest_blocks;
    int threads = h->cfg.threads_per_block;
    int rounds = h->guest_rounds;
    int acc = h->guest_acc;
    unsigned long long* marks = h->d_guest_mark;
    if (acc == 4)
        gold_guest_kernel<4><<<blocks, threads, 0, h->guest_stream>>>(
            h->d_guest_pulses, h->d_guest_stop, h->d_guest_live, marks, rounds,
            h->guest_skip0, h->guest_skip1);
    else if (acc == 8)
        gold_guest_kernel<8><<<blocks, threads, 0, h->guest_stream>>>(
            h->d_guest_pulses, h->d_guest_stop, h->d_guest_live, marks, rounds,
            h->guest_skip0, h->guest_skip1);
    else if (acc == 16)
        gold_guest_kernel<16><<<blocks, threads, 0, h->guest_stream>>>(
            h->d_guest_pulses, h->d_guest_stop, h->d_guest_live, marks, rounds,
            h->guest_skip0, h->guest_skip1);
    else
        gold_guest_kernel<32><<<blocks, threads, 0, h->guest_stream>>>(
            h->d_guest_pulses, h->d_guest_stop, h->d_guest_live, marks, rounds,
            h->guest_skip0, h->guest_skip1);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[GUEST] launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    h->guest_prev_pulses = 0ULL;
    h->guest_prev_us = now_us();
    h->guest_rate = 0.0f;
    h->guest_launched = 1;
    return 0;
}

/* Host view of the mapped mark words. The first time a word is non-zero is
 * the host clock of that observation, relative to guest_run_t0_us later. */
static void gold_guest_observe_marks(GPUtronicHandle* h, int* entered_out, int* first_out) {
    int nenter = 0, nfirst = 0;
    if (h && h->h_guest_mark && h->guest_launched) {
        int n = h->guest_blocks > 84 ? 84 : h->guest_blocks;
        double now = now_us();
        for (int i = 0; i < n; i++) {
            if (h->h_guest_mark[i * 4]) {
                nenter++;
                if (h->guest_enter_seen_us[i] == 0.0) h->guest_enter_seen_us[i] = now;
            }
            if (h->h_guest_mark[i * 4 + 2]) {
                nfirst++;
                if (h->guest_pulse_seen_us[i] == 0.0) h->guest_pulse_seen_us[i] = now;
            }
        }
    }
    if (entered_out) *entered_out = nenter;
    if (first_out) *first_out = nfirst;
}

static void gold_guest_note_residents(const GPUtronicHandle* h) {
    if (!h || !h->h_slot || !h->h_smid) {
        printf("[GUEST] gold resident smids unavailable\n");
        return;
    }
    printf("[GUEST] gold resident smids");
    int n = 0;
    for (int i = 0; i < h->cfg.num_sm; i++) {
        if (h->h_slot[i] == 0u) continue;
        printf(" %u", h->h_smid[i]);
        n++;
    }
    printf(" count=%d\n", n);
}

static void gold_guest_print_marks(const GPUtronicHandle* h) {
    if (!h || !h->h_guest_mark) return;
    double t0 = h->guest_run_t0_us;
    if (h->gold_stop_us > 0.0 && t0 > 0.0)
        printf("[GUEST] gold stop at %.0f ms\n", (h->gold_stop_us - t0) / 1000.0);
    else
        printf("[GUEST] gold stop at n/a\n");
    int n = h->guest_blocks > 84 ? 84 : h->guest_blocks;
    int entered = 0, published = 0, host_enter = 0, host_pulse = 0;
    for (int i = 0; i < n; i++) {
        unsigned long long enter = h->h_guest_mark[i * 4];
        unsigned long long smid = h->h_guest_mark[i * 4 + 1];
        unsigned long long first = h->h_guest_mark[i * 4 + 2];
        unsigned long long last = h->h_guest_mark[i * 4 + 3];
        double em = (h->guest_enter_seen_us[i] > 0.0 && t0 > 0.0)
                        ? (h->guest_enter_seen_us[i] - t0) / 1000.0 : -1.0;
        double pm = (h->guest_pulse_seen_us[i] > 0.0 && t0 > 0.0)
                        ? (h->guest_pulse_seen_us[i] - t0) / 1000.0 : -1.0;
        if (enter) entered++;
        if (first) published++;
        if (h->guest_enter_seen_us[i] > 0.0) host_enter++;
        if (h->guest_pulse_seen_us[i] > 0.0) host_pulse++;
        printf("[GUEST] mark block=%d smid=%llu enter_ns=%llu first_ns=%llu last_ns=%llu enter_seen_ms=%.0f pulse_seen_ms=%.0f\n",
               i, smid, enter, first, last, em, pm);
    }
    printf("[GUEST] marks entered=%d published=%d host_enter=%d host_pulse=%d of %d\n",
           entered, published, host_enter, host_pulse, n);
}

static void gold_guest_halt(GPUtronicHandle* h) {
    if (!h || !h->guest_launched) return;
    if (h->h_guest_stop) {
        *h->h_guest_stop = 1;
        __sync_synchronize();
    }
    if (h->guest_stream_on) cudaStreamSynchronize(h->guest_stream);
    gold_guest_observe_marks(h, NULL, NULL);
    gold_guest_print_marks(h);
    h->guest_launched = 0;
    printf("[GUEST] stopped pulses=%llu live=%d\n",
           h->h_guest_pulses ? (unsigned long long)*h->h_guest_pulses : 0ULL,
           h->h_guest_live ? *h->h_guest_live : 0);
}

static void gold_guest_sample(GPUtronicHandle* h) {
    if (!h || !h->guest_launched || !h->h_guest_pulses) return;
    unsigned long long p = *h->h_guest_pulses;
    double now = now_us();
    double dt = (now - h->guest_prev_us) * 1e-6;
    if (h->guest_prev_us > 0.0 && dt > 1e-4)
        h->guest_rate = (float)((double)(p - h->guest_prev_pulses) / dt);
    h->guest_prev_pulses = p;
    h->guest_prev_us = now;
    int nenter = 0, nfirst = 0;
    gold_guest_observe_marks(h, &nenter, &nfirst);
    printf("[GUEST] visible enter=%d first=%d of %d\n", nenter, nfirst, h->guest_blocks);
}

void gputronic_print_status(const GPUtronicHandle* h, int elapsed_s) {
    if (!h || !h->h_ctrl) return;
    if (h->guest_launched) {
        printf("[TEL] t=%4ds | Z_raw=%.3f Z_hat=%.3f dZ=%.3f | Zc=%.3f Zt=%.3f sel=%s | rate=%.0f rate_t=%.0f | n=%3d | hz~%d | pulses=%llu | guest=%.0f\n",
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
               (unsigned long long)h->h_ctrl->total_work_pulses,
               h->guest_rate);
    } else {
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
}

/* ---- Command-line modes [F-13] ---- */

static int mode_run(int duration_s, float target, float kp, float ki) {
    GPUtronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.target_z = target;
    cfg.kp = kp;
    cfg.ki = ki;
    cfg.csv_path = "results/gputronic_gold_run.csv";

    GPUtronicHandle* h = gputronic_create(&cfg);
    if (!h) return 1;

    int want_guest = 0;
    int guest_at_ms = 8000;
    int stream_first = 0;
    {
        const char* gb = getenv("GPUTRONIC_GUEST_BLOCKS");
        const char* ga = getenv("GPUTRONIC_GUEST_AT_MS");
        const char* sf = getenv("GPUTRONIC_GUEST_STREAM_FIRST");
        if (gb && *gb) want_guest = atoi(gb);
        if (ga && *ga) guest_at_ms = atoi(ga);
        if (sf && strcmp(sf, "1") == 0) stream_first = 1;
    }
    /* Test only. Unset, the guest stream is created after Gold's launch. */
    if (stream_first && want_guest > 0) {
        if (gold_guest_open(h, want_guest) != 0) want_guest = 0;
        else printf("[GUEST] stream created before the Gold launch\n");
    }
    if (gputronic_start(h) != 0) {
        gputronic_destroy(h);
        return 1;
    }
    if (want_guest > 0 && !h->guest_stream_on && gold_guest_open(h, want_guest) != 0)
        want_guest = 0;
    double run_t0 = now_us();
    h->guest_run_t0_us = run_t0;

    for (int s = 0; s < duration_s && !g_signal_stop; s++) {
        usleep(1000000);
        if (want_guest > 0 && !h->guest_launched &&
            (now_us() - run_t0) / 1000.0 >= (double)guest_at_ms) {
            double elapsed_ms = (now_us() - run_t0) / 1000.0;
            if (gold_guest_launch(h) == 0) {
                printf("[GUEST] launched at %.0f ms\n", elapsed_ms);
                gold_guest_note_residents(h);
                int nenter = 0, nfirst = 0;
                gold_guest_observe_marks(h, &nenter, &nfirst);
                printf("[GUEST] visible enter=%d first=%d of %d\n",
                       nenter, nfirst, h->guest_blocks);
            } else
                want_guest = 0;
        }
        gold_guest_sample(h);
        gputronic_print_status(h, s + 1);
    }

    gputronic_stop(h);
    gputronic_destroy(h);
    return 0;
}

/* Guest grid only. The Gold persistent kernel is not launched. */
static int mode_guest(int sec, int blocks) {
    if (sec <= 0) sec = 10;
    if (blocks <= 0) {
        const char* gb = getenv("GPUTRONIC_GUEST_BLOCKS");
        blocks = (gb && *gb) ? atoi(gb) : 28;
    }
    GPUtronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.launch_selftest_kernel = 0;
    cfg.csv_path = NULL;
    GPUtronicHandle* h = gputronic_create(&cfg);
    if (!h) return 1;
    printf("[GUEST] gold resident=0\n");
    h->guest_run_t0_us = now_us();
    if (gold_guest_open(h, blocks) != 0 || gold_guest_launch(h) != 0) {
        printf("[GUEST] launch failed\n");
        gputronic_destroy(h);
        return 1;
    }
    printf("[GUEST] launched at 0 ms\n");
    gold_guest_note_residents(h);
    for (int s = 0; s < sec && !g_signal_stop; s++) {
        usleep(1000000);
        gold_guest_sample(h);
        printf("[GUEST] t=%4ds | rate=%.0f | pulses=%llu | blocks=%d live=%d\n",
               s + 1, h->guest_rate,
               h->h_guest_pulses ? (unsigned long long)*h->h_guest_pulses : 0ULL,
               h->guest_blocks,
               h->h_guest_live ? *h->h_guest_live : 0);
    }
    gold_guest_halt(h);
    gputronic_destroy(h);
    return 0;
}

/* Dyno: open-loop working-SM sweep. Rate = wall Δpulses/Δt. */
static int mode_dyno(void) {
    GPUtronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_gold_dyno.csv";

    GPUtronicHandle* h = gputronic_create(&cfg);
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
        if (nsm < GPUTRONIC_SM_FLOOR) nsm = GPUTRONIC_SM_FLOOR;
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
        fprintf(rf, "GPUtronic Gold Dyno Report\n");
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
    GPUtronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_gold_step.csv";

    GPUtronicHandle* h = gputronic_create(&cfg);
    if (!h) return 1;
    if (gputronic_start(h) != 0) {
        gputronic_destroy(h);
        return 1;
    }

    int n_full = h->cfg.num_sm;
    int n_step = n_full / 2 < GPUTRONIC_SM_FLOOR ? GPUTRONIC_SM_FLOOR : n_full / 2;
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

/* Closed-loop short validation after dyno. Commands 1.5 so the gate still
 * requires a closed rate under 90% of free run. Unset runs keep the macro. */
static int mode_closedloop_check(int seconds) {
    GPUtronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.target_z = 1.5f;
    cfg.csv_path = "results/gputronic_gold_closedloop.csv";

    GPUtronicHandle* h = gputronic_create(&cfg);
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


#ifndef GPUTRONIC_NO_MAIN
/* ---- Synthetic load tests T6/T7 and full-width co-run for T8 ---- */

#define GOLD_LOAD_MAX 4096
typedef struct {
    double t_ms;
    unsigned int scale_milli;
    int sms; /* -1: leave the open-loop count alone. Closed loop ignores this. */
} GoldLoadPoint;

/* Schedule file: "t_ms scale" per line, optional third column active_sms.
 * Scale is a factor (1.2). '#' starts a comment. t_ms counts from the end
 * of calibration. A count below the SM floor is raised to that floor. */
static int gold_load_read(const char* path, GoldLoadPoint* p, int maxn) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[LOAD] cannot open %s\n", path); return -1; }
    char line[256];
    int n = 0;
    while (n < maxn && fgets(line, sizeof line, f)) {
        double t, sc;
        int sms = -1;
        int got;
        if (line[0] == '#') continue;
        got = sscanf(line, "%lf %lf %d", &t, &sc, &sms);
        if (got >= 2 && sc > 0.0) {
            if (got == 2 || sms < 0) sms = -1;
            else if (sms < GPUTRONIC_SM_FLOOR) sms = GPUTRONIC_SM_FLOOR;
            p[n].t_ms = t;
            p[n].scale_milli = (unsigned int)(sc * 1000.0 + 0.5);
            p[n].sms = sms;
            n++;
        }
    }
    fclose(f);
    return n;
}

/* Unset: *out is NULL, *np is 0, and nothing is printed. Set: read the file
 * or fail. An empty or unreadable file must not run as scale 1.0. */
static int gold_open_sched_points(GoldLoadPoint** out, int* np) {
    static GoldLoadPoint pts[GOLD_LOAD_MAX];
    const char* path = getenv("GPUTRONIC_OPEN_SCHED");
    *out = NULL;
    *np = 0;
    if (!path || !path[0]) return 0;
    int n = gold_load_read(path, pts, GOLD_LOAD_MAX);
    if (n <= 0) {
        if (n == 0) fprintf(stderr, "[SCHED] %s has no points\n", path);
        return 1;
    }
    printf("[SCHED] open-loop schedule %s, %d points (unset keeps scale 1.0 and the launch count)\n",
           path, n);
    *out = pts;
    *np = n;
    return 0;
}

/* Bursts of 2, 4, 8, 16, 50, 200 ms at `scale`, `reps` times each in a seeded
 * shuffled order, each followed by a 1.5-2.5 s recovery gap. Starts at 10 s. */
static int gold_load_bursts(GoldLoadPoint* p, int maxn, double scale, unsigned int seed, int reps) {
    const int dur[6] = {2, 4, 8, 16, 50, 200};
    int order[6 * 16];
    int m = 0;
    if (reps > 16) reps = 16;
    for (int r = 0; r < reps; r++) for (int i = 0; i < 6; i++) order[m++] = dur[i];
    srand(seed);
    for (int i = m - 1; i > 0; i--) { int j = rand() % (i + 1); int t = order[i]; order[i] = order[j]; order[j] = t; }
    double t = 10000.0;
    int n = 0;
    for (int i = 0; i < m && n + 2 <= maxn; i++) {
        p[n].t_ms = t; p[n].scale_milli = (unsigned int)(scale * 1000.0 + 0.5); p[n].sms = -1; n++;
        t += order[i];
        p[n].t_ms = t; p[n].scale_milli = 1000u; p[n].sms = -1; n++;
        t += 1500.0 + (double)(rand() % 1001);
    }
    return n;
}

/* Run closed loop at the compiled target (or open-loop full width if full_width), wait for
 * calibration, then apply the schedule and log one row per 10 ms bin. */
/* full_width: 0 closed loop, 1 open loop all SMs, >1 open loop at that many SMs (T10).
 * Bin size: GPUTRONIC_TRACE_BIN_US (default 1 ms with the tally, else 10 ms, minimum
 * 100). Rows are streamed to the file, so memory use does not grow with run length.
 * GPUTRONIC_STEP_AT_MS and GPUTRONIC_STEP_SMS, and the STEP2 pair, are test only.
 * During this timed loop they change the open-loop SM count. Unset, the count stays. */
static unsigned long long gold_work_sum(GPUtronicHandle* h, cudaStream_t st, unsigned long long* buf);

static int gold_load_run(const char* csv_bin, const GoldLoadPoint* p, int np, int duration_s, int full_width) {
    GPUtronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.csv_path = full_width ? "results/gputronic_gold_fullwidth.csv" : "results/gputronic_gold_load.csv";
    /* Test only. loadfile has no Z argument. Unset keeps the compiled target.
     * `run` takes Z on the command line and does not read this. */
    {
        const char* tz = getenv("GPUTRONIC_TARGET_Z");
        if (tz && *tz) {
            float v = (float)atof(tz);
            if (v > 0.0f && v < 20.0f) {
                cfg.target_z = v;
                printf("[TARGET] GPUTRONIC_TARGET_Z=%.3f (unset keeps %.1f)\n",
                       v, (double)GPUTRONIC_GOLD_TARGET_Z);
            }
        }
    }
    GPUtronicHandle* h = gputronic_create(&cfg);
    if (!h) return 1;
    if (gputronic_start(h) != 0) { gputronic_destroy(h); return 1; }
    h->h_ctrl->work_scale_milli = 1000u;
    while (!h->calibrated && !g_signal_stop) usleep(1000);
    if (full_width) gputronic_set_open_loop_sms(h, full_width > 1 ? full_width : 0);
    if (h->select_pass_mask || h->select_sm_lo || h->select_sm_hi) {
        __sync_fetch_and_or(&h->h_ctrl->control_flags, GPUTRONIC_FLAG_SELECT);
        printf("[SELECT] flag on. The block-index gate is off for this hold.\n");
    }
    double bin_ms = h->h_accum ? 1.0 : 10.0;   /* 1 ms default when the T11 accumulator is on */
    { const char* bv = getenv("GPUTRONIC_TRACE_BIN_US"); if (bv && atoi(bv) >= 100) bin_ms = atoi(bv) / 1000.0; }
    int step_sms = -1, step2_sms = -1, step_done = 0, step2_done = 0;
    double step_at = -1.0, step2_at = -1.0;
    {
        const char* a = getenv("GPUTRONIC_STEP_AT_MS");
        const char* s = getenv("GPUTRONIC_STEP_SMS");
        if (a && *a && s && *s) {
            step_at = atof(a);
            step_sms = atoi(s);
            if (step_sms < GPUTRONIC_SM_FLOOR) step_sms = GPUTRONIC_SM_FLOOR;
            printf("[STEP] will set active_sms %d at %.3f ms\n", step_sms, step_at);
        }
        a = getenv("GPUTRONIC_STEP2_AT_MS");
        s = getenv("GPUTRONIC_STEP2_SMS");
        if (a && *a && s && *s) {
            step2_at = atof(a);
            step2_sms = atoi(s);
            if (step2_sms < GPUTRONIC_SM_FLOOR) step2_sms = GPUTRONIC_SM_FLOOR;
            printf("[STEP] will set active_sms %d at %.3f ms\n", step2_sms, step2_at);
        }
    }
    FILE* fa = NULL;
    unsigned long long* a_prev = NULL;
    unsigned long long pulse_prev = h->h_ctrl->total_work_pulses;
    int accum_primed = 0;
    if (h->h_accum) {
        fa = fopen("results/gputronic_gold_smaccum.csv", "w");
        a_prev = (unsigned long long*)calloc((size_t)h->cfg.num_sm * 2, sizeof(unsigned long long));
        if (fa) fprintf(fa, "t_ms,sm,ops_cuda,ops_tensor,d_ops_cuda,d_ops_tensor,dev_t_ns,seq,"
                            "phys_sm,active_sms,bin_pulses,rate_ema\n");
    }
    struct timespec rt;
    clock_gettime(CLOCK_REALTIME, &rt);
    printf("[LOAD] calibrated; schedule t=0 at wall %ld.%09ld (CLOCK_REALTIME), %d points, %d s\n",
           (long)rt.tv_sec, rt.tv_nsec, np, duration_s);
    const char* no_trace = getenv("GPUTRONIC_NO_CSV");
    FILE* f = (no_trace && strcmp(no_trace, "1") == 0) ? NULL : fopen(csv_bin, "w");
    if (f) fprintf(f, "wall_s,t_ms,scale,z_raw,z_hat,active_sms,integral,clamp,rate_ema,sensor,"
                      "win_pulses,p_term,loop_iters,loop_hz,pi_ticks,deadband_held,integral_updates,"
                      "rail_ticks,sm_moves,p_only_moves\n");
    unsigned long long pv[7] = {h->st_loop, h->st_ticks, h->st_db_skip, h->st_int_upd,
                                h->st_rail, h->st_moves, h->st_p_only};
    /* T9 counter, only when the run asked for it. One sample a second, on its
     * own stream, so the copy does not wait for the resident kernel. */
    cudaStream_t wr_st = 0;
    unsigned long long* wr_buf = NULL;
    unsigned long long wr_p = 0, wr_w = 0;
    int wr_on = 0;
    if (h->d_work &&
        cudaStreamCreateWithFlags(&wr_st, cudaStreamNonBlocking) == cudaSuccess) {
        wr_buf = (unsigned long long*)malloc((size_t)h->cfg.num_sm * 32 * sizeof(unsigned long long));
        if (wr_buf) {
            wr_on = 1;
            wr_p = h->h_ctrl->total_work_pulses;
            wr_w = gold_work_sum(h, wr_st, wr_buf);
        } else {
            cudaStreamDestroy(wr_st);
        }
    }
    double t0 = now_us();
    double next_bin = 0.0;
    int k = 0, last_print = -1;
    unsigned int cur = 1000u;
    while (!g_signal_stop) {
        double t_ms = (now_us() - t0) / 1000.0;
        if (t_ms >= duration_s * 1000.0) break;
        if (!step_done && step_at >= 0.0 && t_ms >= step_at) {
            printf("[STEP] t_ms=%.3f active_sms -> %d\n", t_ms, step_sms);
            fflush(stdout);
            gputronic_set_open_loop_sms(h, step_sms);
            step_done = 1;
        }
        if (!step2_done && step2_at >= 0.0 && t_ms >= step2_at) {
            printf("[STEP] t_ms=%.3f active_sms -> %d\n", t_ms, step2_sms);
            fflush(stdout);
            gputronic_set_open_loop_sms(h, step2_sms);
            step2_done = 1;
        }
        while (k < np && p[k].t_ms <= t_ms) {
            cur = p[k].scale_milli;
            /* The count column is open-loop only. Closed-loop loadfile keeps
             * its two-column behaviour. A count at or above the launch is
             * every block (open_loop_sms 0). */
            if (full_width && p[k].sms >= 0) {
                int nsm = p[k].sms;
                if (nsm >= h->cfg.num_sm) nsm = 0;
                else if (nsm < GPUTRONIC_SM_FLOOR) nsm = GPUTRONIC_SM_FLOOR;
                gputronic_set_open_loop_sms(h, nsm);
            }
            k++;
        }
        if (cur != h->h_ctrl->work_scale_milli) {
            h->h_ctrl->work_scale_milli = cur;
            __sync_synchronize();
            /* Pulse-skip engages only above 1.0. The real-work load also
             * engages below 1.0, so a lighter scale still reaches the kernel. */
            int load_on = h->work_load ? (cur != 1000u) : (cur > 1000u);
            if (load_on) __sync_fetch_and_or(&h->h_ctrl->control_flags, GPUTRONIC_FLAG_LOAD);
            else __sync_fetch_and_and(&h->h_ctrl->control_flags, ~GPUTRONIC_FLAG_LOAD);
        }
        int sampled = 0;
        if (t_ms >= next_bin) {
            sampled = 1;
            clock_gettime(CLOCK_REALTIME, &rt);
            unsigned long long cv[7] = {h->st_loop, h->st_ticks, h->st_db_skip, h->st_int_upd,
                                        h->st_rail, h->st_moves, h->st_p_only};
            if (f) fprintf(f, "%ld.%06ld,%.2f,%.3f,%.5f,%.5f,%d,%.5f,%.5f,%.1f,%s,%llu,%.5f,%llu,%.0f,%llu,%llu,%llu,%llu,%llu,%llu\n",
                           (long)rt.tv_sec, rt.tv_nsec / 1000, t_ms, cur / 1000.0,
                           h->h_ctrl->z_raw, h->h_ctrl->z_hat, h->h_ctrl->active_sms,
                           h->integral, h->iclamp, h->h_ctrl->rate_ema,
                           h->h_ctrl->ctrl_sensor ? "tensor" : "cuda",
                           h->last_win_pulses, h->last_p_term,
                           cv[0] - pv[0], (cv[0] - pv[0]) * 1000.0 / bin_ms,
                           cv[1] - pv[1], cv[2] - pv[2], cv[3] - pv[3], cv[4] - pv[4], cv[5] - pv[5], cv[6] - pv[6]);
            for (int q = 0; q < 7; q++) pv[q] = cv[q];
            if (fa && a_prev) {
                unsigned long long pn = h->h_ctrl->total_work_pulses;
                unsigned long long bp = pn - pulse_prev;
                int s0 = h->accum_sm < 0 ? 0 : h->accum_sm;
                int s1 = h->accum_sm < 0 ? h->cfg.num_sm : h->accum_sm + 1;
                if (!accum_primed) {
                    /* The slot is cumulative from kernel start, which includes
                     * calibration. The first sample only sets the baseline. */
                    for (int q = s0; q < s1 && q < h->cfg.num_sm; q++) {
                        volatile unsigned long long* sl = h->h_accum + GOLD_ACCUM_WORDS * q;
                        unsigned long long sq, sq2;
                        int tries = 0;
                        do { sq = sl[4]; a_prev[2 * q] = sl[0]; a_prev[2 * q + 1] = sl[1]; sq2 = sl[4]; }
                        while (sq != sq2 && ++tries < 8);
                    }
                    pulse_prev = pn;
                    accum_primed = 1;
                } else {
                    pulse_prev = pn;
                    for (int q = s0; q < s1 && q < h->cfg.num_sm; q++) {
                        volatile unsigned long long* sl = h->h_accum + GOLD_ACCUM_WORDS * q;
                        unsigned long long a, b, tn, phys, sq, sq2;
                        int tries = 0;
                        do { sq = sl[4]; a = sl[0]; b = sl[1]; tn = sl[2]; phys = sl[3]; sq2 = sl[4]; }
                        while (sq != sq2 && ++tries < 8);
                        fprintf(fa, "%.2f,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%d,%llu,%.1f\n", t_ms, q, a, b,
                                a - a_prev[2 * q], b - a_prev[2 * q + 1], tn, sq, phys,
                                h->h_ctrl->active_sms, bp, h->h_ctrl->rate_ema);
                        a_prev[2 * q] = a; a_prev[2 * q + 1] = b;
                    }
                }
            }
            next_bin += bin_ms;
            while (next_bin <= t_ms) next_bin += bin_ms;
        }
        int sec = (int)(t_ms / 1000.0);
        if (sec != last_print) {
            last_print = sec;
            if (sec > 0) {
                gold_guest_sample(h);
                gputronic_print_status(h, sec);
                if (wr_on) {
                    unsigned long long pn = h->h_ctrl->total_work_pulses;
                    unsigned long long wn = gold_work_sum(h, wr_st, wr_buf);
                    double dp = (double)(pn - wr_p), dw = (double)(wn - wr_w);
                    double ratio = dw > 0.0 ? dp * (double)GOLD_BATCH / dw : 0.0;
                    printf("[WORK] t=%ds pulses=%.0f batches=%.0f ratio=%.5f\n", sec, dp, dw, ratio);
                    wr_p = pn;
                    wr_w = wn;
                }
            }
        }
        if (!sampled) usleep(200);
    }
    __sync_fetch_and_and(&h->h_ctrl->control_flags,
                         ~(GPUTRONIC_FLAG_LOAD | GPUTRONIC_FLAG_SELECT));
    h->h_ctrl->work_scale_milli = 1000u;
    if (f) fclose(f);
    if (fa) fclose(fa);
    free(a_prev);
    if (wr_on) {
        cudaStreamDestroy(wr_st);
        free(wr_buf);
    }
    gputronic_stop(h);
    gputronic_destroy(h);
    printf("[LOAD] wrote %s\n", csv_bin);
    return 0;
}


/* ---- T9 pulse-to-work ratio ----
 * Open loop at each SM count (default N, 3N/4, N/2, N/4: 84, 63, 42, 21 on the
 * 5080), settle 1 s, hold `hold_s`, and compare CUDA-core pulses with completed
 * batches read from the device work counter (separate non-blocking stream, so the
 * copy does not wait for the persistent kernel). ratio = pulses * GOLD_BATCH / work. */
static unsigned long long gold_work_sum(GPUtronicHandle* h, cudaStream_t st, unsigned long long* buf) {
    size_t n = (size_t)h->cfg.num_sm * 32;
    if (cudaMemcpyAsync(buf, h->d_work, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost, st) != cudaSuccess) return 0;
    cudaStreamSynchronize(st);
    unsigned long long s = 0;
    for (size_t i = 0; i < n; i++) s += buf[i];
    return s;
}

static int mode_ratio(int hold_s, const char* list) {
    setenv("GPUTRONIC_WORK_COUNTER", "1", 1);
    setenv("GPUTRONIC_TENSOR_SENSOR", "0", 1);
    GPUtronicConfig cfg;
    gputronic_config_gold(&cfg);
    cfg.csv_path = "results/gputronic_gold_ratio_trace.csv";
    GPUtronicHandle* h = gputronic_create(&cfg);
    if (!h) return 1;
    if (gputronic_start(h) != 0) { gputronic_destroy(h); return 1; }
    if (!h->d_work) { gputronic_stop(h); gputronic_destroy(h); return 1; }
    while (!h->calibrated && !g_signal_stop) usleep(1000);
    int N = h->cfg.num_sm, counts[16], nc = 0;
    if (list && *list) {
        char tmp[256]; strncpy(tmp, list, sizeof tmp - 1); tmp[sizeof tmp - 1] = 0;
        for (char* t = strtok(tmp, ","); t && nc < 16; t = strtok(NULL, ",")) {
            int v = atoi(t); if (v >= 4 && v <= N) counts[nc++] = v;
        }
    }
    if (nc == 0) { counts[0] = N; counts[1] = N * 3 / 4; counts[2] = N / 2; counts[3] = N / 4 < GPUTRONIC_SM_FLOOR ? GPUTRONIC_SM_FLOOR : N / 4; nc = 4; }
    cudaStream_t st; cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    unsigned long long* buf = (unsigned long long*)malloc((size_t)N * 32 * sizeof(unsigned long long));
    FILE* f = fopen("results/gputronic_gold_ratio.csv", "w");
    if (f) fprintf(f, "active_sms,hold_s,pulses,work_batches,ratio,drift_pct_vs_first\n");
    double r0 = 0.0;
    printf("\n=== T9 PULSE-TO-WORK RATIO (open loop, tensor off) ===\n");
    for (int i = 0; i < nc && !g_signal_stop; i++) {
        gputronic_set_open_loop_sms(h, counts[i] == N ? 0 : counts[i]);
        usleep(1000000);
        unsigned long long p0 = h->h_ctrl->total_work_pulses, w0 = gold_work_sum(h, st, buf);
        double t0 = now_us();
        usleep((useconds_t)hold_s * 1000000u);
        unsigned long long p1 = h->h_ctrl->total_work_pulses, w1 = gold_work_sum(h, st, buf);
        double held = (now_us() - t0) / 1e6;
        double dp = (double)(p1 - p0), dw = (double)(w1 - w0);
        double ratio = dw > 0 ? dp * GOLD_BATCH / dw : 0.0;
        if (i == 0) r0 = ratio;
        double drift = r0 > 0 ? 100.0 * (ratio - r0) / r0 : 0.0;
        printf("[T9] sms=%d hold=%.2fs pulses=%.0f work=%.0f ratio=%.5f drift=%+.3f%%\n", counts[i], held, dp, dw, ratio, drift);
        if (f) fprintf(f, "%d,%.3f,%.0f,%.0f,%.6f,%.4f\n", counts[i], held, dp, dw, ratio, drift);
    }
    if (f) fclose(f);
    free(buf);
    cudaStreamDestroy(st);
    gputronic_stop(h);
    gputronic_destroy(h);
    return 0;
}
#endif /* GPUTRONIC_NO_MAIN */

static void usage(const char* argv0) {
    printf("GPUtronic Gold %s\n\n", GPUTRONIC_VERSION_STRING);
    printf("Usage:\n");
    printf("  %s                 # 30s closed-loop free-run (Gold defaults)\n", argv0);
    printf("  %s run [sec] [Z] [Kp] [Ki]\n", argv0);
    printf("  %s dyno            # open-loop working-SM sweep + R²\n", argv0);
    printf("  %s step            # open-loop SM-count step response\n", argv0);
    printf("  %s check           # dyno + 20s closed-loop gate\n", argv0);
    printf("  %s all             # dyno + step + closed-loop\n", argv0);
    printf("  %s loadstep [sec] [scale] [t_on_s] [t_off_s]  # T6: compiled target, load step (default 60 1.2 20 0=hold)\n", argv0);
    printf("  %s bursts [scale] [seed] [reps]              # T7: 2..200 ms bursts (default 1.2 1 5)\n", argv0);
    printf("  %s loadfile <file> [sec]                      # schedule file, lines \"t_ms scale\"\n", argv0);
    printf("  %s noise [sms] [sec]                         # T10: open loop at fixed SMs, Z noise floor (default 46 30)\n", argv0);
    printf("    test only, off by default: GPUTRONIC_IDLE_QUIET=1\n");
    printf("    GPUTRONIC_ACTIVE_PASSES=0,1,3   GPUTRONIC_ACTIVE_SMS=0-11\n");
    printf("    GPUTRONIC_PULSE_LOCAL=1\n");
    printf("    GPUTRONIC_CTRL_CACHE=1\n");
    printf("    GPUTRONIC_CTRL_CACHE_US=1000   # cache period, only with CTRL_CACHE=1\n");
    printf("    GPUTRONIC_RATE_WIN_US=8000     # rate window, us; unset keeps 8 ms\n");
    printf("    GPUTRONIC_ICLAMP_SMS=6         # clamp in SMs; unset keeps the scaled law\n");
    printf("    GPUTRONIC_IDLE_WORK=1          # idle blocks do the work and add to the one total\n");
    printf("    GPUTRONIC_TARGET_Z=<z>         # loadfile/loadstep only; unset keeps 1.0\n");
    printf("    GPUTRONIC_IDLE_EXIT=1          # idle block returns; host launches the rest\n");
    printf("    GPUTRONIC_RELAUNCH=1           # new grid at the commanded count\n");
    printf("    GPUTRONIC_RELAUNCH_HOLD_US=5000\n");
    printf("    GPUTRONIC_PEAK_REPS / GPUTRONIC_PEAK_CHAINS / GPUTRONIC_THREADS\n");
    printf("    GPUTRONIC_PEAK_ACC (4|8|16|32) / GPUTRONIC_PEAK_ROUNDS (default 32)\n");
    printf("    GPUTRONIC_WORK_LOAD=1          # scale changes FMA rounds; every batch pulses\n");
    printf("    GPUTRONIC_OPEN_SCHED=<path>    # fullwidth/noise: t_ms scale [sms]; unset keeps scale 1.0\n");
    printf("    GPUTRONIC_GUEST_BLOCKS=<n>     # second kernel, own pulse word, off if unset\n");
    printf("    GPUTRONIC_GUEST_AT_MS=8000     # launch the guest this many ms after run starts\n");
    printf("    GPUTRONIC_GUEST_STREAM_FIRST=1 # test: create the guest stream before Gold's\n");
    printf("    GPUTRONIC_GUEST_PRIORITY=1     # test: highest stream priority, off if unset\n");
    printf("    GPUTRONIC_GOLD_NONBLOCK=1      # test: Gold kernel on a non-blocking stream\n");
    printf("  %s guest [sec] [blocks]                       # guest only, Gold kernel not launched\n", argv0);
    printf("    GPUTRONIC_STEP_AT_MS / GPUTRONIC_STEP_SMS   # test only, during noise or loadstep\n");
    printf("  %s ratio [hold_s] [n1,n2,...]                 # T9: pulse-to-work ratio (default 5, N,3N/4,N/2,N/4)\n", argv0);
    printf("  %s fullwidth [sec]                            # T8: open loop, all SMs, no controller\n", argv0);
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

    if (strcmp(mode, "loadstep") == 0) {
        static GoldLoadPoint pts[2];
        int sec = argc >= 3 ? atoi(argv[2]) : 60;
        double sc = argc >= 4 ? atof(argv[3]) : 1.2;
        double ton = argc >= 5 ? atof(argv[4]) : 20.0;
        double toff = argc >= 6 ? atof(argv[5]) : 0.0;
        if (sec <= 0) sec = 60;
        int n = 0;
        pts[n].t_ms = ton * 1000.0; pts[n].scale_milli = (unsigned int)(sc * 1000.0 + 0.5); pts[n].sms = -1; n++;
        if (toff > ton) { pts[n].t_ms = toff * 1000.0; pts[n].scale_milli = 1000u; pts[n].sms = -1; n++; }
        return gold_load_run("results/gputronic_gold_loadstep_10ms.csv", pts, n, sec, 0);
    }
    if (strcmp(mode, "bursts") == 0) {
        static GoldLoadPoint pts[GOLD_LOAD_MAX];
        double sc = argc >= 3 ? atof(argv[2]) : 1.2;
        unsigned int seed = argc >= 4 ? (unsigned int)atoi(argv[3]) : 1u;
        int reps = argc >= 5 ? atoi(argv[4]) : 5;
        if (reps <= 0) reps = 5;
        int n = gold_load_bursts(pts, GOLD_LOAD_MAX, sc, seed, reps);
        int sec = (int)(pts[n - 1].t_ms / 1000.0) + 5;
        return gold_load_run("results/gputronic_gold_bursts_10ms.csv", pts, n, sec, 0);
    }
    if (strcmp(mode, "loadfile") == 0) {
        static GoldLoadPoint pts[GOLD_LOAD_MAX];
        if (argc < 3) { usage(argv[0]); return 1; }
        int n = gold_load_read(argv[2], pts, GOLD_LOAD_MAX);
        if (n < 0) return 1;
        int sec = argc >= 4 ? atoi(argv[3]) : (n > 0 ? (int)(pts[n - 1].t_ms / 1000.0) + 10 : 30);
        return gold_load_run("results/gputronic_gold_loadfile_10ms.csv", pts, n, sec, 0);
    }
    if (strcmp(mode, "noise") == 0) {   /* T10: open loop at a fixed SM count, no control */
        int nsm = argc >= 3 ? atoi(argv[2]) : 46;
        int sec = argc >= 4 ? atoi(argv[3]) : 30;
        if (nsm < GPUTRONIC_SM_FLOOR) nsm = GPUTRONIC_SM_FLOOR;
        if (sec <= 0) sec = 30;
        {
            GoldLoadPoint* pts = NULL;
            int np = 0;
            if (gold_open_sched_points(&pts, &np) != 0) return 1;
            return gold_load_run("results/gputronic_gold_noise.csv", pts, np, sec, nsm);
        }
    }
    if (strcmp(mode, "ratio") == 0) {
        int hold = argc >= 3 ? atoi(argv[2]) : 5;
        return mode_ratio(hold > 0 ? hold : 5, argc >= 4 ? argv[3] : NULL);
    }
    if (strcmp(mode, "guest") == 0) {
        int sec = argc >= 3 ? atoi(argv[2]) : 10;
        int blocks = argc >= 4 ? atoi(argv[3]) : 0;
        return mode_guest(sec, blocks);
    }
    if (strcmp(mode, "fullwidth") == 0) {
        int sec = argc >= 3 ? atoi(argv[2]) : 120;
        if (sec <= 0) sec = 120;
        {
            GoldLoadPoint* pts = NULL;
            int np = 0;
            if (gold_open_sched_points(&pts, &np) != 0) return 1;
            return gold_load_run("results/gputronic_gold_fullwidth_10ms.csv", pts, np, sec, 1);
        }
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
