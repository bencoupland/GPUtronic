#!/bin/bash
# Seventh pass, part 2. Reserved SMs.
# Launch is 82 blocks. Floor is 2. The quiet sweep still starts at 4.
# Kp, Ki, and the deadband are not touched. The clamp formula is not touched.
# The earlier 84-SM holds are not repeated.
set -u
cd "$(dirname "$0")/.."
LOG="logs/2026-10-13_seventh-pass-part2"
mkdir -p "$LOG" results
echo $$ > "$LOG/suite.pid"
FAILS=0
if [ -f "$LOG/BLOCKER.txt" ]; then
    echo "BLOCKER file present, not continuing"
    cat "$LOG/BLOCKER.txt"
    exit 2
fi
DMON_PID=""
BIN="./build/gputronic_gold"

gpu_temp() {
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d ' '
}

stop_dmon() {
    if [ -n "${DMON_PID}" ]; then
        kill "$DMON_PID" 2>/dev/null || true
        wait "$DMON_PID" 2>/dev/null || true
        DMON_PID=""
    fi
}
trap stop_dmon EXIT

wait_cool() {
    local t i
    t=$(gpu_temp)
    echo "temp ${t} C"
    if [ "$t" -lt 83 ]; then
        return 0
    fi
    echo "GPU ${t} C, waiting"
    i=0
    while [ "$i" -lt 6 ]; do
        sleep 20
        t=$(gpu_temp)
        echo "temp ${t} C"
        if [ "$t" -lt 80 ]; then
            return 0
        fi
        i=$((i + 1))
    done
    echo "BLOCKER: GPU stayed at ${t} C before the next hold" | tee "$LOG/BLOCKER.txt"
    exit 2
}

clear_switches() {
    unset GPUTRONIC_PULSE_LOCAL GPUTRONIC_CTRL_CACHE GPUTRONIC_CTRL_CACHE_US
    unset GPUTRONIC_STEP_AT_MS GPUTRONIC_STEP_SMS GPUTRONIC_STEP2_AT_MS GPUTRONIC_STEP2_SMS
    unset GPUTRONIC_ACTIVE_PASSES GPUTRONIC_ACTIVE_SMS GPUTRONIC_IDLE_QUIET
    unset GPUTRONIC_TENSOR_SENSOR GPUTRONIC_TRACE_BIN_US GPUTRONIC_SM_ACCUM
    unset GPUTRONIC_SM_ACCUM_SM GPUTRONIC_DEADBAND GPUTRONIC_INTEGRAL_CLAMP
}

drop_fast_csv() {
    # The control thread writes one row per 10 us tick. That file is not a result.
    # Leave the trace CSVs and the tally. The caller compresses those.
    rm -f results/gputronic_gold_fullwidth.csv results/gputronic_gold_load.csv \
          results/gputronic_gold.csv
}

summarize_dmon() {
    local name="$1"
    python3 - "$LOG/${name}.dmon.txt" << 'PY'
import sys
path = sys.argv[1]
rows = []
try:
    lines = open(path)
except OSError:
    print("dmon missing")
    sys.exit(0)
for line in lines:
    if not line.strip() or line.startswith("#"):
        continue
    p = line.split()
    if len(p) < 16:
        continue
    def num(i):
        try:
            return float(p[i])
        except ValueError:
            return None
    pwr, gtemp, sm, mclk, pclk = num(1), num(2), num(4), num(10), num(11)
    pviol, tviol, rx, tx = num(12), num(13), num(14), num(15)
    if None in (pwr, gtemp, sm, mclk, pclk, pviol, tviol, rx, tx):
        continue
    rows.append((pwr, gtemp, sm, mclk, pclk, pviol, tviol, rx, tx))
if not rows:
    print("dmon empty")
    sys.exit(0)
use = rows
def mean(i):
    return sum(r[i] for r in use) / len(use)
print("dmon n=%d temp %.0f..%.0f mean %.1f pwr mean %.1f max %.1f "
      "pclk mean %.0f min %.0f max %.0f mclk mean %.0f sm%% mean %.0f "
      "pviol max %.0f tviol max %.0f rx mean %.0f tx mean %.0f" % (
        len(use), min(r[1] for r in use), max(r[1] for r in use), mean(1),
        mean(0), max(r[0] for r in use), mean(4), min(r[4] for r in use),
        max(r[4] for r in use), mean(3), mean(2),
        max(r[5] for r in use), max(r[6] for r in use), mean(7), mean(8)))
if max(r[1] for r in use) >= 83 or max(r[6] for r in use) >= 1:
    sys.exit(2)
# Busy, but the clock left the step this card held near 2900 MHz.
if mean(0) > 60 and mean(2) > 50 and mean(4) < 2200:
    sys.exit(2)
PY
}

# run_cmd NAME COMMAND...
run_cmd() {
    local name="$1"
    shift
    local rc=0 bpid marked=0 n=0 t
    if [ "$FAILS" -ge 3 ]; then
        echo "STOP: three hard failures"
        exit 3
    fi
    wait_cool
    echo "=== ${name} $* temp=$(gpu_temp) $(date -Is) ==="
    stdbuf -oL nvidia-smi dmon -s pucvt -d 1 -c 180 > "$LOG/${name}.dmon.txt" &
    DMON_PID=$!
    stdbuf -oL -eL "$@" > "$LOG/${name}.txt" 2>&1 &
    bpid=$!
    while kill -0 "$bpid" 2>/dev/null; do
        if [ "$marked" -eq 0 ] && grep -q '\[LOAD\] calibrated' "$LOG/${name}.txt" 2>/dev/null; then
            awk 'NF && $1 !~ /^#/ { n++ } END { print n+0 }' "$LOG/${name}.dmon.txt" > "$LOG/${name}.t0"
            marked=1
        fi
        n=$((n + 1))
        if [ $((n % 10)) -eq 0 ]; then
            t=$(gpu_temp)
            if [ "$t" -ge 83 ]; then
                echo "BLOCKER: ${name} reached ${t} C" | tee "$LOG/BLOCKER.txt"
                kill "$bpid" 2>/dev/null || true
                wait "$bpid" 2>/dev/null || true
                stop_dmon
                exit 2
            fi
        fi
        sleep 0.2
    done
    wait "$bpid" || rc=$?
    echo "temp $(gpu_temp) C  exit ${rc}" >> "$LOG/${name}.txt"
    echo "exit ${rc}" >> "$LOG/${name}.txt"
    echo "DONE" >> "$LOG/${name}.txt"
    stop_dmon
    drop_fast_csv
    echo "temp $(gpu_temp) C  exit ${rc}"
    if [ "$rc" -ne 0 ]; then
        echo "FAIL ${name}: process exit ${rc}"
        FAILS=$((FAILS + 1))
    fi
    if ! summarize_dmon "$name"; then
        echo "BLOCKER: ${name} hit a thermal or power limit" | tee "$LOG/BLOCKER.txt"
        exit 2
    fi
    return 0
}

finished_ok() {
    local name="$1"
    [ -f "$LOG/${name}.txt" ] && grep -q '^exit 0$' "$LOG/${name}.txt" && grep -q '^DONE$' "$LOG/${name}.txt"
}

require_on() {
    local name="$1" us="$2"
    if ! grep -q '\[PULSE\] local count: one mapped atomic per block per 1 ms' "$LOG/${name}.txt"; then
        echo "FAIL ${name}: pulse-local line missing"
        FAILS=$((FAILS + 1))
    fi
    if ! grep -q "cached control words, including work_scale_milli: leader reads once per ${us} us" "$LOG/${name}.txt"; then
        echo "FAIL ${name}: cache line for ${us} us missing"
        FAILS=$((FAILS + 1))
    fi
}

require_reserve() {
    local name="$1"
    if ! grep -q 'launching 82, 2 reserved for the OS' "$LOG/${name}.txt"; then
        echo "FAIL ${name}: OS reserve line missing"
        FAILS=$((FAILS + 1))
    fi
    if ! grep -q 'floor=2' "$LOG/${name}.txt"; then
        echo "FAIL ${name}: floor is not 2"
        FAILS=$((FAILS + 1))
    fi
    if ! grep -q '\[SMID\] left out:' "$LOG/${name}.txt"; then
        echo "FAIL ${name}: smid map missing"
        FAILS=$((FAILS + 1))
    fi
    if ! grep -q 'max integral correction 3.28 SMs of 82' "$LOG/${name}.txt"; then
        echo "FAIL ${name}: scaled clamp is not 3.28 SMs of 82"
        FAILS=$((FAILS + 1))
    fi
}

require_off() {
    local name="$1"
    if grep -q '\[PULSE\] local count' "$LOG/${name}.txt" \
        || grep -q 'including work_scale_milli' "$LOG/${name}.txt" \
        || grep -q '\[IDLE\] quiet on' "$LOG/${name}.txt"; then
        echo "FAIL ${name}: a test switch printed while it should be off"
        FAILS=$((FAILS + 1))
    fi
}

require_quiet() {
    local name="$1" want="$2"
    if [ "$want" = on ]; then
        if ! grep -q '\[IDLE\] quiet on: idle blocks sample the mapped words once per 1 ms' "$LOG/${name}.txt"; then
            echo "FAIL ${name}: quiet-idle line missing"
            FAILS=$((FAILS + 1))
        fi
    else
        if grep -q '\[IDLE\] quiet on' "$LOG/${name}.txt"; then
            echo "FAIL ${name}: quiet idle printed while it should be off"
            FAILS=$((FAILS + 1))
        fi
    fi
}

save_trace() {
    local name="$1" src="$2"
    if [ ! -f "$src" ]; then
        echo "FAIL ${name}: missing ${src}"
        FAILS=$((FAILS + 1))
        return
    fi
    zstd -q -f --rm -o "$LOG/${name}.trace.csv.zst" "$src"
    echo "OK ${name} trace saved"
}

echo "SEVENTH PASS PART 2 START $(date -Is) temp=$(gpu_temp)"
nvidia-smi > "$LOG/nvidia-smi-start.txt"
echo "--- nvidia-smi start ---"
cat "$LOG/nvidia-smi-start.txt"
make NVCC=/usr/local/cuda-12.8/bin/nvcc ARCH=-arch=sm_120 gold lib transfer episode

# Three checks, switches unset. A failed gate stops the suite.
for spec in "gold-check|${BIN}|check" "transfer-check|./build/gputronic_transfer|check" "episode-check|./build/gputronic_episode|check"; do
    name="${spec%%|*}"
    rest="${spec#*|}"
    cmd="${rest%%|*}"
    mode="${rest##*|}"
    if finished_ok "$name" && grep -q 'dyno=PASS closedloop=PASS' "$LOG/${name}.txt"; then
        echo "SKIP ${name}"
        continue
    fi
    clear_switches
    run_cmd "$name" "$cmd" "$mode"
    require_off "$name"
    require_reserve "$name"
    if ! grep -q 'dyno=PASS closedloop=PASS' "$LOG/${name}.txt"; then
        echo "BLOCKER: ${name} gate did not pass" | tee "$LOG/BLOCKER.txt"
        exit 2
    fi
    if ! grep -q 'Kp=0.550 Ki=0.0800' "$LOG/${name}.txt" || ! grep -q 'deadband=0.080' "$LOG/${name}.txt"; then
        echo "BLOCKER: ${name} gains are not the stock law" | tee "$LOG/BLOCKER.txt"
        exit 2
    fi
done

# Piece 2. T6, scale 1.2 from 20 s, 60 s. Baseline first, then both switches on.
if finished_ok "t6-off" && [ -f "$LOG/t6-off.trace.csv.zst" ]; then
    echo "SKIP t6-off"
else
    clear_switches
    export GPUTRONIC_TRACE_BIN_US=1000
    run_cmd "t6-off" "$BIN" loadstep 60 1.2 20
    require_off "t6-off"
    require_reserve "t6-off"
    save_trace "t6-off" results/gputronic_gold_loadstep_10ms.csv
    clear_switches
fi

if finished_ok "t6-on" && [ -f "$LOG/t6-on.trace.csv.zst" ]; then
    echo "SKIP t6-on"
else
    clear_switches
    export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_CTRL_CACHE_US=1000
    export GPUTRONIC_TRACE_BIN_US=1000
    run_cmd "t6-on" "$BIN" loadstep 60 1.2 20
    require_on "t6-on" 1000
    require_quiet "t6-on" off
    require_reserve "t6-on"
    save_trace "t6-on" results/gputronic_gold_loadstep_10ms.csv
    clear_switches
fi

# Piece 3. Open loop, both switches on, tally off. Quiet on, then three quiet-off repeats.
for n in 4 12 24 36 48 56 72 82; do
    name="q-n${n}"
    if finished_ok "$name" && [ -f "$LOG/${name}.trace.csv.zst" ]; then
        echo "SKIP ${name}"
        continue
    fi
    clear_switches
    export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_CTRL_CACHE_US=1000
    export GPUTRONIC_IDLE_QUIET=1
    run_cmd "$name" "$BIN" noise "$n" 30
    require_on "$name" 1000
    require_quiet "$name" on
    require_reserve "$name"
    if grep -q '\[T11\] per-SM accumulator on' "$LOG/${name}.txt"; then
        echo "FAIL ${name}: tally was on"
        FAILS=$((FAILS + 1))
    fi
    save_trace "$name" results/gputronic_gold_noise.csv
    clear_switches
done

for n in 4 36 82; do
    name="q-n${n}-spin"
    if finished_ok "$name" && [ -f "$LOG/${name}.trace.csv.zst" ]; then
        echo "SKIP ${name}"
        continue
    fi
    clear_switches
    export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_CTRL_CACHE_US=1000
    run_cmd "$name" "$BIN" noise "$n" 30
    require_on "$name" 1000
    require_quiet "$name" off
    require_reserve "$name"
    save_trace "$name" results/gputronic_gold_noise.csv
    clear_switches
done

# Command latency at the 1 ms cache, quiet on. Same step as the sixth pass.
if finished_ok "e3-quiet" && [ -f "$LOG/e3-quiet.csv.zst" ]; then
    echo "SKIP e3-quiet"
else
    clear_switches
    export GPUTRONIC_SM_ACCUM=1 GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1
    export GPUTRONIC_CTRL_CACHE_US=1000 GPUTRONIC_TRACE_BIN_US=100 GPUTRONIC_IDLE_QUIET=1
    export GPUTRONIC_STEP_AT_MS=1000 GPUTRONIC_STEP_SMS=42
    export GPUTRONIC_STEP2_AT_MS=2000 GPUTRONIC_STEP2_SMS=82
    run_cmd "e3-quiet" "$BIN" noise 82 3
    require_on "e3-quiet" 1000
    require_quiet "e3-quiet" on
    require_reserve "e3-quiet"
    if [ "$(grep -c '\[STEP\] t_ms=' "$LOG/e3-quiet.txt" || true)" -lt 2 ]; then
        echo "FAIL e3-quiet: expected two step lines"
        FAILS=$((FAILS + 1))
    fi
    if [ -f results/gputronic_gold_noise.csv ]; then
        zstd -q -f --rm -o "$LOG/e3-quiet.noise.csv.zst" results/gputronic_gold_noise.csv
    fi
    if [ ! -f results/gputronic_gold_smaccum.csv ]; then
        echo "FAIL e3-quiet: no tally csv"
        FAILS=$((FAILS + 1))
    else
        zstd -q -f --rm -o "$LOG/e3-quiet.csv.zst" results/gputronic_gold_smaccum.csv
        echo "OK e3-quiet tally saved"
    fi
    clear_switches
fi

nvidia-smi > "$LOG/nvidia-smi-end.txt"
echo "--- nvidia-smi end ---"
cat "$LOG/nvidia-smi-end.txt"
echo "SUITE DONE fails=${FAILS} $(date -Is) temp=$(gpu_temp)"
exit "$FAILS"
