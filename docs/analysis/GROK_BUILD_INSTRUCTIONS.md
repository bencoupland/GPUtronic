# Test runbook: RTX 5080, CUDA 12.8

Step-by-step instructions for an automated tester running GPUTronic on Ben's RTX 5080. Every
binary, mode, environment variable and file path below exists on branch `controller-fixes`.
Test definitions and pass/fail criteria are in `TEST_SUITE.md`.

## 0. Read first
Read `docs/analysis/HANDOFF.md` (what is expected, what is unverified) and `docs/analysis/TEST_SUITE.md`.

## 1. Rules
- **Never commit to `main`. Never force-push. Never merge.**
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
been measured. T6 as a second process, T7, and the Cyberpunk co-run stay proposed.
