#!/bin/bash
# Sixth-pass holds. Both switches stay off unless a hold sets them.
# Kp, Ki, the deadband, and the clamp are not touched.
set -u
cd "$(dirname "$0")/.."
LOG="logs/2026-10-13_sixth-pass"
mkdir -p "$LOG" results
FAILS=0
LAST_RC=0
BIN="./build/gputronic_gold"
DMON_PID=""

gpu_temp() {
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d ' '
}

stop_dmon() {
    if [ -n "$DMON_PID" ]; then
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
    echo "ABORT: GPU stayed at ${t} C"
    echo "BLOCKER: GPU stayed at ${t} C before the next hold" > "$LOG/BLOCKER.txt"
    exit 2
}

clear_switches() {
    unset GPUTRONIC_PULSE_LOCAL GPUTRONIC_CTRL_CACHE GPUTRONIC_CTRL_CACHE_US
    unset GPUTRONIC_STEP_AT_MS GPUTRONIC_STEP_SMS GPUTRONIC_STEP2_AT_MS GPUTRONIC_STEP2_SMS
    unset GPUTRONIC_ACTIVE_PASSES GPUTRONIC_ACTIVE_SMS GPUTRONIC_IDLE_QUIET
    unset GPUTRONIC_TENSOR_SENSOR GPUTRONIC_TRACE_BIN_US GPUTRONIC_SM_ACCUM
}

drop_fast_csv() {
    # The control thread writes one row per 10 us tick. That file is not a result.
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
use = [r for r in rows if r[0] >= 50 or r[2] >= 40]
if len(use) < 3:
    use = rows[2:] or rows
def mean(i):
    return sum(r[i] for r in use) / len(use)
pclk = mean(4)
print("dmon n=%d temp %.0f..%.0f mean %.1f pwr mean %.1f max %.1f "
      "pclk mean %.0f min %.0f mclk mean %.0f sm%% mean %.0f "
      "pviol max %.0f tviol max %.0f rx mean %.0f tx mean %.0f" % (
        len(use), min(r[1] for r in use), max(r[1] for r in use), mean(1),
        mean(0), max(r[0] for r in use), pclk, min(r[4] for r in use), mean(3),
        mean(2), max(r[5] for r in use), max(r[6] for r in use), mean(7), mean(8)))
# A single ramp sample can be slow. Stop only if the busy clock sits low,
# the temperature stays hot, or a thermal violation is recorded.
if max(r[1] for r in use) >= 83 or max(r[6] for r in use) >= 1 or (mean(0) > 60 and pclk < 2200):
    sys.exit(2)
PY
}

run_cmd() {
    local name="$1"
    shift
    local rc
    if [ "$FAILS" -ge 3 ]; then
        echo "STOP: three hard failures"
        exit 3
    fi
    wait_cool
    echo "=== ${name} $* temp=$(gpu_temp) $(date -Is) ==="
    nvidia-smi dmon -s pucvt -d 1 -c 90 > "$LOG/${name}.dmon.txt" &
    DMON_PID=$!
    "$BIN" "$@" > "$LOG/${name}.txt" 2>&1
    rc=$?
    echo "temp $(gpu_temp) C  exit ${rc}" >> "$LOG/${name}.txt"
    echo "DONE" >> "$LOG/${name}.txt"
    stop_dmon
    drop_fast_csv
    echo "temp $(gpu_temp) C  exit ${rc}"
    LAST_RC=$rc
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

require_on() {
    local name="$1" us="$2"
    if ! grep -q '\[PULSE\] local count: one mapped atomic per block per 1 ms' "$LOG/${name}.txt"; then
        echo "FAIL ${name}: pulse-local line missing"
        FAILS=$((FAILS + 1))
    fi
    if ! grep -q "\[CTRL\] cached control words: leader reads once per ${us} us" "$LOG/${name}.txt"; then
        echo "FAIL ${name}: cache period ${us} us missing"
        FAILS=$((FAILS + 1))
    fi
}

require_off() {
    local name="$1"
    if grep -q '\[PULSE\] local count' "$LOG/${name}.txt" || grep -q '\[CTRL\] cached control words' "$LOG/${name}.txt"; then
        echo "FAIL ${name}: a test switch printed while it should be off"
        FAILS=$((FAILS + 1))
    fi
}

take_tally() {
    local name="$1" expect="$2"
    local arc work t11 ratio ratio_ok
    if [ ! -f results/gputronic_gold_smaccum.csv ]; then
        echo "FAIL ${name}: no tally csv"
        FAILS=$((FAILS + 1))
        return
    fi
    python3 scripts/analyze_smaccum.py results/gputronic_gold_smaccum.csv > "$LOG/${name}_analysis.txt" 2>&1
    arc=$?
    if [ -f results/gputronic_gold_noise.csv ]; then
        zstd -q -f --rm -o "$LOG/${name}.noise.csv.zst" results/gputronic_gold_noise.csv
    fi
    zstd -q -f --rm -o "$LOG/${name}.csv.zst" results/gputronic_gold_smaccum.csv
    work=$(sed -n 's/.*with work \([0-9][0-9]*\).*/\1/p' "$LOG/${name}_analysis.txt" | head -1)
    t11=$(grep -c 'T11 PASS' "$LOG/${name}_analysis.txt" || true)
    ratio=$(sed -n 's/.*\[100 ms\].*ratio mean \([0-9.][0-9.]*\).*/\1/p' "$LOG/${name}_analysis.txt" | head -1)
    ratio_ok=$(awk -v r="${ratio:-0}" 'BEGIN { print (r >= 0.95 && r <= 1.05) ? 1 : 0 }')
    echo "--- ${name} analyzer exit ${arc} work=${work:-none} ratio=${ratio:-none} ---"
    grep -E '^\[(blocks|jump|pass|phys|pulses|100 ms|T11)\]|^  [0-9]+ [0-9]+ ' "$LOG/${name}_analysis.txt" || true
    if [ "${work:-0}" != "$expect" ] || [ "$ratio_ok" -ne 1 ]; then
        echo "FAIL ${name}: work=${work:-none} expect=${expect} ratio=${ratio:-none}"
        FAILS=$((FAILS + 1))
        if [ "${work:-0}" != "$expect" ]; then
            echo "STOP: working-set mismatch on ${name}"
            exit 4
        fi
    elif [ "$t11" -lt 1 ]; then
        echo "OK ${name} work=${work} ratio=${ratio} (correlation under the T11 bar)"
    else
        echo "OK ${name} work=${work} ratio=${ratio}"
    fi
}

take_trace() {
    local name="$1" src="$2"
    if [ ! -f "$src" ]; then
        echo "FAIL ${name}: no trace ${src}"
        FAILS=$((FAILS + 1))
        return
    fi
    zstd -q -f --rm -o "$LOG/${name}.trace.csv.zst" "$src"
}

echo "sixth-pass suite $(date -Is) $(git rev-parse --short HEAD) temp=$(gpu_temp)"
if ! strings "$BIN" | grep -q 'leader reads once per'; then
    echo "binary is missing the cache-period line"
    exit 1
fi

# Experiment 1. Both switches on. Open loop, 30 s. Block-index gate.
for n in 4 12 24 36 42 48 56 64 72 84; do
    name="e1-n${n}"
    if [ -f "$LOG/${name}.csv.zst" ]; then
        echo "SKIP ${name}"
        continue
    fi
    clear_switches
    export GPUTRONIC_SM_ACCUM=1 GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1
    export GPUTRONIC_CTRL_CACHE_US=1000
    run_cmd "$name" noise "$n" 30
    if [ "$LAST_RC" -eq 0 ]; then
        require_on "$name" 1000
        take_tally "$name" "$n"
    fi
    clear_switches
done

# Same evening, 84 and 48 with both switches off.
for n in 84 48; do
    name="e1-n${n}-off"
    if [ -f "$LOG/${name}.csv.zst" ]; then
        echo "SKIP ${name}"
        continue
    fi
    clear_switches
    export GPUTRONIC_SM_ACCUM=1
    run_cmd "$name" noise "$n" 30
    if [ "$LAST_RC" -eq 0 ]; then
        require_off "$name"
        take_tally "$name" "$n"
    fi
    clear_switches
done

# Experiment 2. Closed loop, both switches on, stock gains. Then one T6 load step.
# loadstep 60 1.0 999 never enters the load window, so the 60 s stay closed-loop.
if [ ! -f "$LOG/e2-closed.trace.csv.zst" ]; then
    clear_switches
    export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_CTRL_CACHE_US=1000
    export GPUTRONIC_TRACE_BIN_US=1000
    run_cmd "e2-closed" loadstep 60 1.0 999
    if [ "$LAST_RC" -eq 0 ]; then
        require_on "e2-closed" 1000
        take_trace "e2-closed" results/gputronic_gold_loadstep_10ms.csv
        if ! grep -q '\[CAL\] integral clamp=' "$LOG/e2-closed.txt"; then
            echo "FAIL e2-closed: calibration line missing"
            FAILS=$((FAILS + 1))
        fi
    fi
    clear_switches
else
    echo "SKIP e2-closed"
fi

if [ ! -f "$LOG/e2-t6.trace.csv.zst" ]; then
    clear_switches
    export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_CTRL_CACHE_US=1000
    export GPUTRONIC_TRACE_BIN_US=1000
    run_cmd "e2-t6" loadstep 60 1.2 20
    if [ "$LAST_RC" -eq 0 ]; then
        require_on "e2-t6" 1000
        take_trace "e2-t6" results/gputronic_gold_loadstep_10ms.csv
    fi
    clear_switches
else
    echo "SKIP e2-t6"
fi

# Experiment 3. Command latency at three cache periods. Tally bin 100 us.
# The working set changes on purpose, so the steady-state T11 gate does not apply.
for us in 250 1000 4000; do
    name="e3-${us}us"
    if [ -f "$LOG/${name}.csv.zst" ]; then
        echo "SKIP ${name}"
        continue
    fi
    clear_switches
    export GPUTRONIC_SM_ACCUM=1 GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1
    export GPUTRONIC_CTRL_CACHE_US="$us" GPUTRONIC_TRACE_BIN_US=100
    export GPUTRONIC_STEP_AT_MS=1000 GPUTRONIC_STEP_SMS=42
    export GPUTRONIC_STEP2_AT_MS=2000 GPUTRONIC_STEP2_SMS=84
    run_cmd "$name" noise 84 3
    if [ "$LAST_RC" -eq 0 ]; then
        require_on "$name" "$us"
        if [ "$(grep -c '\[STEP\] t_ms=' "$LOG/${name}.txt" || true)" -lt 2 ]; then
            echo "FAIL ${name}: expected two step lines"
            FAILS=$((FAILS + 1))
        fi
        if [ ! -f results/gputronic_gold_smaccum.csv ]; then
            echo "FAIL ${name}: no tally csv"
            FAILS=$((FAILS + 1))
        else
            if [ -f results/gputronic_gold_noise.csv ]; then
                zstd -q -f --rm -o "$LOG/${name}.noise.csv.zst" results/gputronic_gold_noise.csv
            fi
            zstd -q -f --rm -o "$LOG/${name}.csv.zst" results/gputronic_gold_smaccum.csv
            echo "OK ${name} tally saved"
        fi
    fi
    clear_switches
done

echo "SUITE DONE fails=${FAILS} $(date -Is) temp=$(gpu_temp)"
exit "$FAILS"
