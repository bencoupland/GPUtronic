# Test runbook: RTX 5080, CUDA 12.8

Step-by-step instructions for an automated tester running GPUtronic on Ben's RTX 5080. Every
binary, mode, environment variable and file path below exists on branch `controller-fixes`.
Test definitions and pass/fail criteria are in `TEST_SUITE.md`.

## 0. Read first
Read `docs/analysis/HANDOFF.md` (what is expected, what is unverified) and `docs/analysis/TEST_SUITE.md`.

## 1. Rules
- **Do not push `main` unless Ben Coupland has said to.** On 2026-10-12 he did, and that push is the law in this tree. Never force-push.
- Put a `timeout` on every GPU run (values below). A timeout counts as a failure.
- **Stop on any compile error, any non-zero exit, or GPU temperature above 85 °C.** Write the exact
  error text into `logs/<stamp>/STOPPED.txt`, commit what you have (step 8), and do nothing else.
- Run **T1 (tensor-sensor overhead) before any other test.**
- Do not edit source files during a test run.

## 2. Get the code
```bash
cd ~/GPUtronic            # or wherever the clone lives
git fetch origin
git checkout controller-fixes
git pull --ff-only origin controller-fixes
git log --oneline -1 | tee /tmp/gputronic_head.txt
```

## 3. Build
The Makefile hard-codes `NVCC=/usr/local/cuda-12.8/bin/nvcc` and `ARCH=-arch=sm_120`. Override if
the toolkit lives elsewhere:
```bash
STAMP=$(date +%Y-%m-%d_%H%M); LOG=logs/$STAMP; mkdir -p $LOG
nvidia-smi > $LOG/nvidia-smi.txt; nvcc --version > $LOG/nvcc.txt 2>&1
make clean
make NVCC=/usr/local/cuda-12.8/bin/nvcc ARCH=-arch=sm_120 all transfer episode build/gputronic_show \
  > $LOG/build.log 2>&1 || { echo "build failed" > $LOG/STOPPED.txt; exit 1; }
```
`all` builds `build/gputronic_gold`, `build/libgputronic_gold.a` and `build/gputronic_demo`.

## 4. Start the monitor (keep it running for all tests)
```bash
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,pcie.link.gen.current \
  --format=csv -lms 100 > $LOG/smi.csv &
SMI=$!
nvidia-smi dmon -s t -d 1 > $LOG/smi_pcie.txt &     # PCIe rx/tx MB/s, 1 s samples
DMON=$!
```
Temperature guard, run between tests:
```bash
T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); \
[ "$T" -gt 85 ] && { echo "GPU ${T}C > 85C" > $LOG/STOPPED.txt; kill $SMI $DMON; exit 1; }
```

## 5. Switches that exist
Environment variables read by `gputronic_config_gold` (Gold, Transfer and Episode all call it):

| Variable | Values | Default | Effect |
|---|---|---|---|
| `GPUTRONIC_TENSOR_SENSOR` | 0 / 1 | 1 | Tensor-core sensor off/on (Gold only; always off for Transfer/Episode and below sm_70) |
| `GPUTRONIC_SENSOR_RULE` | `max` / `min` | `max` | Follow the higher Z (default) or the lower Z |
| `GPUTRONIC_RESERVE_SMS` | integer ≥ 0 | 0 | Extra SMs kept on above the calibrated starting count |

There are **no** command-line flags for these. Gold modes: `run [sec] [Z] [Kp] [Ki]`, `dyno`, `step`,
`check`, `all`. Transfer modes: `run [sec]`, `dyno`, `check`, `mask`, `work`. Episode modes:
`run [sec]`, `dyno`, `closedloop [sec]`, `check`, `proof`. Show: no argument, or `sweep`.

Things that **do not exist** in the code: an induced-load or burst generator, a load-step mode, and PCIe or
system-memory traffic counters. Closest real options: `gputronic_gold step` (open-loop SM-count step),
running a second GPU program at the same time as an external load, `nvidia-smi dmon -s t` (PCIe
throughput), and Nsight Systems `nsys profile --gpu-metrics-devices=all` (PCIe throughput metrics, if your
Nsight version supports this GPU). Nsight Compute (`ncu`) is not suitable: the kernels run until
stopped, and ncu replays kernels.

## 6. Tests (run in this order; copy results after each)
Each mode overwrites its files in `results/`, so copy them into `$LOG` straight after each run.

```bash
# T1 tensor-sensor overhead (FIRST). Open-loop SM sweep, sensor off then on.
GPUTRONIC_TENSOR_SENSOR=0 timeout 120 ./build/gputronic_gold dyno > $LOG/T1_off.txt 2>&1 || echo FAIL >> $LOG/T1_off.txt
cp results/gputronic_gold_dyno.csv $LOG/T1_off_dyno.csv; cp results/gputronic_gold_dyno_summary.csv $LOG/T1_off_summary.csv
GPUTRONIC_TENSOR_SENSOR=1 timeout 120 ./build/gputronic_gold dyno > $LOG/T1_on.txt 2>&1 || echo FAIL >> $LOG/T1_on.txt
cp results/gputronic_gold_dyno.csv $LOG/T1_on_dyno.csv; cp results/gputronic_gold_dyno_summary.csv $LOG/T1_on_summary.csv

# T2 Gold closed loop, 40 s, Z 1.5, both sensors
timeout 90 ./build/gputronic_gold run 40 1.5 > $LOG/T2_run.txt 2>&1
cp results/gputronic_gold_run.csv $LOG/T2_run.csv
timeout 120 ./build/gputronic_gold check > $LOG/T2_check.txt 2>&1
cp results/gputronic_gold_closedloop.csv results/gputronic_gold_closedloop_report.txt $LOG/ 2>/dev/null
timeout 60 ./build/gputronic_gold step > $LOG/T2_step.txt 2>&1; cp results/gputronic_gold_step.csv $LOG/

# T3 Transfer and Episode
timeout 180 ./build/gputronic_transfer check > $LOG/T3_transfer.txt 2>&1
cp results/gputronic_transfer_*.csv results/gputronic_transfer_*report.txt $LOG/ 2>/dev/null
timeout 300 ./build/gputronic_episode check > $LOG/T3_episode.txt 2>&1
cp results/episode_* $LOG/ 2>/dev/null

# T5 reserve (0,1,2,4), 30 s each
for K in 0 1 2 4; do
  GPUTRONIC_RESERVE_SMS=$K timeout 70 ./build/gputronic_gold run 30 1.5 > $LOG/T5_k$K.txt 2>&1
  cp results/gputronic_gold_run.csv $LOG/T5_k$K.csv
done
kill $SMI $DMON
```
Check the exit status (`$?`) after every command; stop as in section 1 on non-zero (124 = timeout).
T4 (GTX 1080) needs a different machine; it is not part of this run.

## 7. What to record in the summary
`$LOG/SUMMARY.md`: commit hash, driver, CUDA version, peak temperature, and for each test the numbers
listed in `TEST_SUITE.md` next to the predicted values in `HANDOFF.md`.

## 8. Commit results (never main)
`*.log` and `results/*.csv` are git-ignored, so add with `-f`:
```bash
git checkout -b test-results/$STAMP
git add -f $LOG
git commit -m "Test results $STAMP (RTX 5080, $(cat /tmp/gputronic_head.txt | cut -c1-7))"
git push origin test-results/$STAMP
```

## 9. Report as a GitHub issue
```bash
gh issue create --repo bencoupland/GPUtronic \
  --title "RTX 5080 test results $STAMP" \
  --body-file $LOG/SUMMARY.md
# later runs: gh issue comment <number> --repo bencoupland/GPUtronic --body-file $LOG/SUMMARY.md
```
Compare each result with the "Model prediction" lines in `TEST_SUITE.md` and say which matched.

## 10. Queued work

The order is `docs/analysis/NEXT.md`. Section A of that file is the PR cleanup
(free-run wait, tensor sensor off, deadband 0.08, `__nanosleep` removed). The
integral clamp is not scaled until the in-kernel plant-gain step in that file has
been measured. (Since then the user decided to scale it; see HANDOFF.)

Implemented, not yet run (commands in `TEST_SUITE.md`):
- **T9** pulse-to-work ratio: `timeout 120 ./build/gputronic_gold ratio 5`, then `scripts/analyze_ratio.py`.
  Run it right after T1.
- **T6** load step: `timeout 120 ./build/gputronic_gold loadstep 60 1.2 20 40`, once with
  `GPUTRONIC_INTEGRAL_CLAMP=1.0` (baseline) and once without; `scripts/analyze_loadstep.py`.
- **T7** bursts: `timeout 200 ./build/gputronic_gold bursts 1.2 1 5`; `scripts/analyze_loadstep.py`.
- **T8** Space Marine 2 co-run: `scripts/t8_game_corun.sh alone|fullwidth|closed <run>` (manual route,
  needs a person at the keyboard); `scripts/analyze_mangohud.py`. A second-process load and the
  Cyberpunk co-run stay proposed.
Copy each `results/gputronic_gold_*_10ms.csv` / `gputronic_gold_ratio.csv` into `$LOG` after each run.

## 11. After 2026-10-12 code changes (not yet run)
Default deadband is now 0.08 and device `__nanosleep` is gone. Run in this order, tensor sensor off:
1. `GPUTRONIC_TENSOR_SENSOR=0 timeout 120 ./build/gputronic_gold check` (free-run rate vs 533k; A1–A4 gate).
2. T10 noise floor and the closed-loop comparison at 1 ms (commands in `TEST_SUITE.md`, T10).
3. Deadband probe 0.015 vs 0.08 with `loadstep` (TEST_SUITE, "Deadband probe").
4. `timeout 180 ./build/gputronic_transfer check`, `timeout 300 ./build/gputronic_episode check` (free-run wait port).

## 12. Next run after `7400273` (per-SM accumulator)
Free-run is settled (543k-585k vs 533k; idle polling kept). Tensor sensor off (now the default).
1. T10 at 46 SMs, 30 s, `GPUTRONIC_TRACE_BIN_US=1000`; analyze with `scripts/analyze_loadstep.py`.
2. T11 one SM (`GPUTRONIC_SM_ACCUM_SM=0`), then all SMs (`GPUTRONIC_SM_ACCUM=1`); `scripts/analyze_smaccum.py`.
   If the accumulator agrees with the pulse counter (drift < 1%, corr >= 0.9) the per-SM idea holds before
   wiring all 84 SMs into control; if the ops stream is steady while pulses swing, the ~0.05 Z spread is
   in the sensing path.
Exact commands: `TEST_SUITE.md`, T11. Commit outputs (`results/gputronic_gold_noise.csv`,
`results/gputronic_gold_smaccum.csv`, stdout, analyzer output) to `test-results/YYYY-MM-DD_t11` with
`git add -f`. The all-SM CSV is ~150 MB for 30 s: compress it or keep the single-SM one.
