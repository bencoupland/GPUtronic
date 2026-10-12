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

## Synthetic load (implemented, not yet run on hardware)
Mechanism: field `work_scale_milli`, appended at the end of `GPUTronicControl` (default 0/1000 = no load).
At S > 1000 the Gold kernel's pulsing warps emit one pulse per S/1000 batches, so the pulse rate falls to
1000/S and the sensors see S/1000 times more work per pulse. Why not literally more FMAs: a Gold pass is
~288 µs of mapped atomics and ~1.6 µs of arithmetic, so 20% more arithmetic would barely move the rate.
Limitation: this is a sensor-side load (the plant "gets slower" as seen by Z); it is not interference
from another program. Only the Gold kernel reads it; Transfer and Episode ignore it.
Schedules start at the end of calibration. Every mode writes one row per 10 ms with
`wall_s` (CLOCK_REALTIME) `,t_ms,scale,z_raw,z_hat,active_sms,integral,clamp,rate_ema,sensor`; the start
wall time is printed in the `[LOAD] calibrated` line.

| Command | What |
|---|---|
| `./build/gputronic_gold loadstep [sec] [scale] [t_on_s] [t_off_s]` | default `60 1.2 20 0` (hold to end); writes `results/gputronic_gold_loadstep_10ms.csv` |
| `./build/gputronic_gold bursts [scale] [seed] [reps]` | 2, 4, 8, 16, 50, 200 ms bursts, seeded shuffle, 1.5–2.5 s gaps, from 10 s; default `1.2 1 5`; `results/gputronic_gold_bursts_10ms.csv` |
| `./build/gputronic_gold loadfile <file> [sec]` | schedule file, one `t_ms scale` per line (`#` comments); `results/gputronic_gold_loadfile_10ms.csv` |

## T6. Sustained 20% load step (implemented, not yet run)
```
timeout 120 ./build/gputronic_gold loadstep 60 1.2 20 40      # on at 20 s, off at 40 s
python3 scripts/analyze_loadstep.py results/gputronic_gold_loadstep_10ms.csv --recover-s 2
GPUTRONIC_INTEGRAL_CLAMP=1.0 timeout 120 ./build/gputronic_gold loadstep 60 1.2 20 40   # pre-fix baseline
```
Reports Z before the step, peak Z, time to stay within 0.02 of 1.5 (or "never"), % of bins with the
integral at the clamp, SM-count distribution. For the **pre-fix baseline** run the same command with
`GPUTRONIC_INTEGRAL_CLAMP=1.0` (the old fixed clamp), then without it (scaled clamp). Keep both logs.
Scales below 1.0 are not supported (no 0.8× step).
**Model prediction (fixed clamp):** Z parks near 1.62–1.74.
**Pass:** recovered within 2 s. If not, and the integral is at the clamp most of the step, the analyzer
prints "parked at clamp": that reproduces the bug.

## T7. Bursts (implemented, not yet run)
```
timeout 200 ./build/gputronic_gold bursts 1.2 1 5
python3 scripts/analyze_loadstep.py results/gputronic_gold_bursts_10ms.csv
```
Per burst: peak Z, ms above target, whether the controller moved SMs; summary per burst length.
**Model prediction:** 2–8 ms bursts are not seen (no SM change); 16 ms and longer move SMs within 8–16 ms.
**Pass:** recorded for all lengths; for ≥50 ms bursts Z back within 0.02 within 200 ms after the burst.

## T8. Game co-run (implemented as a script, not yet run)
Game on this box: Warhammer 40,000: Space Marine 2 (Steam app id 2183900). Cyberpunk 2077 remains the
alternative (`cyberpunk-benchmark-plan.md`). GPUTronic throttles only its own kernel, never the game.
```
scripts/t8_game_corun.sh alone 1        # repeat runs 1..5 per condition, discard run 1
scripts/t8_game_corun.sh fullwidth 1    # Gold open loop, all SMs (`gputronic_gold fullwidth`)
scripts/t8_game_corun.sh closed 1       # Gold closed loop Z 1.5, tensor sensor off
python3 scripts/analyze_mangohud.py logs/t8/<cond>_run<k>/mangohud/*.csv
```
Gold starts and calibrates before the game is launched. Space Marine 2 has **no built-in benchmark**, so
each run is a manual fixed route; MangoHud starts logging after `WARMUP` s and logs for `DUR` s.
Steam launch options for the game must be `mangohud %command%`. Fixed settings: one preset, fixed
resolution, upscaler fixed, frame generation off, vsync off.

| Fact | Status | Source |
|---|---|---|
| Space Marine 2 Steam app id 2183900 | verified | https://store.steampowered.com/app/2183900/ |
| No built-in benchmark in Space Marine 2 | verified (press) | https://www.dsogaming.com/pc-performance-analyses/warhammer-40k-space-marine-2-benchmarks-pc-performance-analysis/ |
| MangoHud `output_folder`, `autostart_log`, `log_duration`, `MANGOHUD_CONFIG`, `mangohud %command%` | verified | https://github.com/flightlessmango/MangoHud (README) |
| `steam -applaunch <appid>` passes through env vars to the game | **unverified** (an already-running Steam client may not inherit `MANGOHUD_CONFIG`; the launch option `mangohud %command%` plus a MangoHud config file is the fallback) | — |
| MangoHud log column named `frametime` (ms) | **unverified** for the installed version; the analyzer stops with an error if absent | — |
| EasyAntiCheat behaviour under Proton with MangoHud | **unverified** | — |

**Metrics:** average FPS, 1% and 0.1% lows (1000 / mean of the slowest 1% / 0.1% of frame times),
frame-time standard deviation, 99th-percentile frame time; Gold Z, SM count and rate from its CSV.
**Pass (proposed):** closed loop's 1% low within 5% of game-alone while Gold delivers ≥50% of its
free-run rate; full-width visibly worse than closed loop.

## T9. Pulse-to-work ratio (implemented, not yet run)
```
timeout 120 ./build/gputronic_gold ratio 5                  # default counts N, 3N/4, N/2, N/4 (84,63,42,21 on the 5080)
timeout 120 ./build/gputronic_gold ratio 5 84,63,42,21      # explicit list
python3 scripts/analyze_ratio.py results/gputronic_gold_ratio.csv
```
Open loop, tensor sensor off, 1 s settle, then `hold_s` per count. Work is counted independently of the
pulses: a per-warp counter of completed batches in plain device memory (not mapped, no PCIe traffic),
one store per pass, read by `cudaMemcpyAsync` on a non-blocking stream. It is enabled only by this mode
(`GPUTRONIC_WORK_COUNTER=1`); default runs pass a null pointer and do not touch it.
`ratio = pulses × GOLD_BATCH / work_batches` (1.0 means one pulse per completed pass).
**Note:** in Gold the pulse and the work happen in the same pass, so this checks that the mapped
counter does not lose or delay pulses at different widths (the host's view vs the device's), not
whether the arithmetic per pass changes. Snapshot skew between the two reads is a small error source.
**Pass:** |drift vs 84 SMs| ≤ 1% at every count. If it drifts, the Z math is wrong and `rate_ref` needs
revisiting.

## T10. Noise floor, open loop (implemented, not yet run)
Is the Z noise (about ±0.1 bin to bin on 2026-10-12) made by the loop, or is it already in the sensor/plant?
```
GPUTRONIC_TENSOR_SENSOR=0 GPUTRONIC_TRACE_BIN_US=1000 timeout 60 ./build/gputronic_gold noise 46 30
GPUTRONIC_TENSOR_SENSOR=0 GPUTRONIC_TRACE_BIN_US=1000 timeout 60 ./build/gputronic_gold loadfile /dev/null 30
python3 scripts/analyze_loadstep.py results/gputronic_gold_loadfile_10ms.csv --noise results/gputronic_gold_noise.csv
```
`noise [sms] [sec]` calibrates, then holds the SM count fixed (no controller) and writes the trace.
`loadfile /dev/null [sec]` is a plain closed-loop run with the same trace. (The file name says 10ms;
the bin follows `GPUTRONIC_TRACE_BIN_US`.)
**Pass/verdict:** closed/open bin-to-bin noise ratio < 1.5 means the noise is upstream (sensor/plant);
a larger ratio means the loop adds it.

## Trace options and loop telemetry
- `GPUTRONIC_TRACE_BIN_US=1000` gives 1 ms bins for `loadstep`, `bursts`, `loadfile`, `noise`, `fullwidth`
  (default 10000; minimum 500). Rows are streamed to disk.
- Columns: `wall_s,t_ms,scale,z_raw,z_hat,active_sms,integral,clamp,rate_ema,sensor,win_pulses,p_term,
  loop_iters,loop_hz,pi_ticks,deadband_held,integral_updates,rail_ticks,sm_moves,p_only_moves`
  (counts are per bin; `loop_hz` is the actual control-loop rate in that bin).
- The same totals print as a `[STATS] loop_iters=... pi_ticks=...` line at stop.
- Note: the control thread also writes its own CSV row every tick; that file I/O is a likely reason the
  loop runs at ~16–24 kHz rather than 100 kHz. Not changed here.

## Deadband probe (0.015 vs 0.08)
The compiled default is now **0.08** (NEXT.md A3). `GPUTRONIC_DEADBAND` overrides it.
```
GPUTRONIC_TENSOR_SENSOR=0 GPUTRONIC_DEADBAND=0.015 timeout 120 ./build/gputronic_gold loadstep 60 1.2 20 40
cp results/gputronic_gold_loadstep_10ms.csv $LOG/T6_db015.csv
GPUTRONIC_TENSOR_SENSOR=0 GPUTRONIC_DEADBAND=0.08  timeout 120 ./build/gputronic_gold loadstep 60 1.2 20 40
cp results/gputronic_gold_loadstep_10ms.csv $LOG/T6_db08.csv
python3 scripts/analyze_loadstep.py $LOG/T6_db015.csv; python3 scripts/analyze_loadstep.py $LOG/T6_db08.csv
```
The analyzer prints the SM walk (percent of bins where the count changed, distinct counts, std), Z noise,
loop rate, and the PI counters (deadband held %, integral updated %, on rail %, SM moves, P-only moves).

## T11. Per-SM ops accumulator (prototype, implemented, not yet run)
Design and PCIe math: `per-sm-accumulator.md`.
```
GPUTRONIC_TENSOR_SENSOR=0 GPUTRONIC_SM_ACCUM_SM=0 GPUTRONIC_TRACE_BIN_US=1000 timeout 60 ./build/gputronic_gold noise 46 30
python3 scripts/analyze_smaccum.py results/gputronic_gold_smaccum.csv
```
Then all SMs with `GPUTRONIC_SM_ACCUM=1`. **Pass:** ratio drift < 1% and correlation ≥ 0.9 (10-bin level).
