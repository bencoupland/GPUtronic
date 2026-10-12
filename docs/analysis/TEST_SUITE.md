# GPUTronic test suite

Run in order. Commands are in `GROK_BUILD_INSTRUCTIONS.md`. "Model prediction" values come from
simulations in `HANDOFF.md`; none has been measured on hardware.

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

## T4. GTX 1080 bring-up (separate machine, needs a code change)
- `__nanosleep` needs compute 7.0+. In `src/gputronic_gold.cu` it is called at lines **119, 125 and
  133** (pause 5 µs, reset 1 µs, idle-SM 5 µs). Replace with a `clock64()` spin under
  `#if __CUDA_ARCH__ < 700`. Transfer and Episode also call it.
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
