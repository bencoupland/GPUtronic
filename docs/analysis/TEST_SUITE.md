# GPUTronic test suite

Run in order. Commands are in `GROK_BUILD_INSTRUCTIONS.md`. The "Model prediction" lines were
written before the 2026-10-12 RTX 5080 run. T1, T2, T3, and T5 were measured that day. The score
is `5080-2026-10-12.md` on `test-results/2026-10-12_5080`. T4 has not been run. T6, T7, and T8
are still proposals. The order from here is `NEXT.md`.

## T1. Tensor-sensor overhead (run first)
**What:** `gputronic_gold dyno` with `GPUTRONIC_TENSOR_SENSOR=0`, then `=1`. The dyno holds fixed SM
counts (84, 63, 56, 42, 28, 21, 8, 4).
**Measure:**
- CUDA-core pulse rate at each SM count, on vs off (`*_summary.csv`, column `rate`).
- Tensor pulse rate at each count: `rate_tensor_ema` in `gputronic_gold_dyno.csv` (rows with that
  `active_sms`). Pulses per 8 ms window = rate × 0.008.
**Model prediction:** with the tensor warp on, the CUDA-core rate drops by up to about a third (three warps
share the mapped-atomic path instead of two). Tensor pulses per 8 ms window: about 1,000 at 46 SMs,
about 100 at 4 SMs. The tensor rate follows the same shape as the CUDA rate.
**Pass:** both counters increase at every SM count; the tensor rate falls as SMs drop; at least 50
tensor pulses per 8 ms window at 4 SMs. **Report** the CUDA-rate loss in percent at 84 and 46 SMs.

## T2. Gold closed loop at Z = 1.5, 40 s
**What:** `gputronic_gold run 40 1.5`, plus `check` and `step`.
**Logged** (`gputronic_gold_run.csv`): `rate_ema`, `active_sms`, `z_raw`, `z_hat`, `z_cuda`, `z_tensor`,
`rate_tensor_ema`, `ctrl_sensor`, `integral`. Calibration lines (`[CAL]`) are in the stdout file.
**PCIe / system-memory atomic traffic:** the code has no counter for it. Use `nvidia-smi dmon -s t`
(PCIe rx/tx MB/s), and optionally Nsight Systems GPU metrics; state which was used.
**Model prediction:** mean Z about 1.49 (both sensors about 1.49, largely redundant); SM count settles
around 46–47 with occasional ±1 changes; calibration about 3 s.
**Pass:** mean Z over the last 30 s within 1.45–1.55; Z never above 1.6 for more than 100 ms after
calibration; `step` shows the rate change within two 8 ms windows.

## T3. Transfer and Episode
**What:** `gputronic_transfer check`, `gputronic_episode check` (tensor sensor is off for these).
**Model prediction:** Transfer mean Z about 1.46 (its curve bends between 63 and 42 SMs); Episode
about 1.50.
**Pass:** both report PASS; mean Z within 1.40–1.55.

## T4. GTX 1080 bring-up (separate machine)
- Build: `make NVCC=<cuda-12.x>/bin/nvcc ARCH=-arch=sm_61 gold` (CUDA 13 cannot target sm_61; use the
  R580 driver branch).
- The tensor path is compiled out below sm_70 and switched off at runtime.
**Model prediction:** with runtime calibration, mean Z about 1.48 with frequent 13↔14 SM switching
(one SM ≈ 0.1 Z on 20 SMs).
**Pass:** zero-copy check OK; dyno rate falls monotonically 20 → 4 SMs; closed loop holds mean Z
within 1.40–1.60.

## T5. Feed-forward reserve: 0, 1, 2, 4 SMs
**What:** `GPUTRONIC_RESERVE_SMS=k gputronic_gold run 30 1.5` for k = 0, 1, 2, 4. The reserve adds k SMs
to the calibrated starting count (default 0).
**Bursts and load steps:** there is no built-in load generator. The closest options are running a
second GPU program during the run (an external load) or `gputronic_gold step` (an SM-count step, not a
load step). Note what was used.
**Model prediction:** steady Z about 1.48 at k=0 and about 1.43 at k=4; throughput about 0.5–1% higher
per reserve SM. Bursts shorter than 8 ms are not seen by the controller. A sustained −20% load leaves Z
near 1.65 because the integral clamp limits correction.
**Pass:** steady Z decreases as k increases; report peak Z and time above 1.5 for any external load.

## Proposed: synthetic load generator (not implemented)

The order for this work is `NEXT.md`. The plant-gain step (C1 there) is an in-kernel
change to the work per pulse, run once before any edit to the integral clamp. A
second CUDA process is the coexistence question, not this one. The 1.62–1.74 figure
below is the simulator. It has not been measured.
Nothing below exists in the code yet. It is a plan for a companion program, `gputronic_load`, used by
T6 and T7. Purpose: reproduce and then tune the integral-clamp recovery bug (after a sustained 20% load
step Z hangs about 1.62–1.74; the clamp is hard-coded 1.0 in `include/gputronic.h`, which gives only
about 4 SMs of authority).

**Mechanisms and trade-offs**
- *Separate competing CUDA process* (proposed default): a second program runs a persistent kernel that
  occupies a set number of SMs or streams memory/PCIe traffic. Realistic (it is how a real co-tenant
  behaves) and needs no change to GPUTronic, but the scheduler decides how SMs are shared, so the
  effective load is not exact; time-slicing between contexts adds jitter.
- *Separate stream in the same process*: tighter timing, same caveat on sharing; needs code inside
  the GPUTronic binaries.
- *In-kernel work multiplier on the pulse batch* (e.g. 1.2× FMAs per pulse via a control-block field):
  exact and repeatable, maps directly to "the plant got 20% slower", but changes the owned kernel, so it
  does not test interference from outside work.

**Proposed interface**
```
gputronic_load --mode sms|membw|pcie --schedule FILE [--seed N] [--log FILE] [--device 0]
gputronic_load --step 0.20 --at 10 --hold 20           # one sustained step
gputronic_load --bursts 2,4,8,16,50,200 --gap 500 --seed 7
```
Schedule file (CSV, one line per segment, times in seconds from start):
`t_start_s,duration_s,level,shape` with `level` a fraction of full load and `shape` = `step` or `burst`.
A seed controls any random gaps, so a schedule replays exactly.
Log columns: `wall_ns` (CLOCK_REALTIME), `t_s`, `level_cmd`, `level_applied`, `sms_busy`, `segment_id`.
`wall_ns` aligns with GPUTronic's 10 µs samples by recording CLOCK_REALTIME at GPUTronic start.

## T6. Sustained load steps (proposed; needs `gputronic_load`)
Gold, Transfer, Episode at Z 1.5. After calibration plus 10 s, apply +20% load for 20 s, release,
then −20% for 20 s. Run once before the clamp fix (**baseline, record it**) and once after.
**Model prediction (pre-fix):** Z holds near 1.62–1.74 and does not return within 0.02 of 1.5.
**Pass (post-fix):** Z back within 0.02 of 1.5 within **2 s** of each step (X = 2 s, revisit after the
first baseline), no integral sample on the clamp for longer than 1 s.

## T7. Bursts (proposed; needs `gputronic_load`)
Bursts of 2, 4, 8, 16, 50 and 200 ms, 500 ms apart, seeded, 10 of each.
**Model prediction:** bursts under 8 ms are not seen by the controller; 16 ms and longer cause an SM
change within 8–16 ms.
**Pass:** peak Z and time above 1.5 recorded per burst length; for ≥50 ms bursts Z returns within 0.02
within 200 ms after the burst ends.

## T8. Cyberpunk 2077 co-run (proposed)
See `cyberpunk-benchmark-plan.md`.
