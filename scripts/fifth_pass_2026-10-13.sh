#!/bin/bash
# Fifth-pass open-loop holds. New switches stay off unless a hold sets them.
# Kp, Ki, the deadband, and the clamp are not touched.
set -u
cd "$(dirname "$0")/.."
LOG="logs/2026-10-13_fifth-pass"
mkdir -p "$LOG" results
FAILS=0
export GPUTRONIC_SM_ACCUM=1
unset GPUTRONIC_TENSOR_SENSOR GPUTRONIC_ACTIVE_PASSES GPUTRONIC_ACTIVE_SMS
unset GPUTRONIC_IDLE_QUIET GPUTRONIC_PULSE_LOCAL

gpu_temp() {
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d ' '
}

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
    exit 2
}

run_hold() {
    local name="$1" nsm="$2" quiet="$3" passes="$4" pulse="$5" expect="$6"
    local t0 rc arc work t11 ratio ratio_ok
    if [ -f "$LOG/${name}_analysis.txt" ]; then
        echo "SKIP ${name} (already analyzed)"
        return
    fi
    if [ "$FAILS" -ge 3 ]; then
        echo "STOP: three hard failures"
        exit 3
    fi
    wait_cool
    unset GPUTRONIC_ACTIVE_PASSES GPUTRONIC_ACTIVE_SMS GPUTRONIC_IDLE_QUIET GPUTRONIC_PULSE_LOCAL
    if [ "$quiet" = "1" ]; then export GPUTRONIC_IDLE_QUIET=1; fi
    if [ -n "$passes" ]; then export GPUTRONIC_ACTIVE_PASSES="$passes"; fi
    if [ "$pulse" = "1" ]; then export GPUTRONIC_PULSE_LOCAL=1; fi
    t0=$(gpu_temp)
    echo "=== ${name} nsm=${nsm} quiet=${quiet} passes='${passes}' pulse=${pulse} expect=${expect} temp=${t0} $(date -Is) ==="
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader > "$LOG/${name}.clock.txt" || true
    ./build/gputronic_gold noise "$nsm" 30 > "$LOG/${name}.txt" 2>&1
    rc=$?
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader >> "$LOG/${name}.clock.txt" || true
    echo "temp $(gpu_temp) C  exit ${rc}" >> "$LOG/${name}.txt"
    if [ -f results/gputronic_gold_smaccum.csv ]; then
        mv results/gputronic_gold_smaccum.csv "$LOG/${name}.csv"
    fi
    if [ -f results/gputronic_gold_noise.csv ]; then
        zstd -q -f --rm -o "$LOG/${name}.noise.csv.zst" results/gputronic_gold_noise.csv
    fi
    unset GPUTRONIC_ACTIVE_PASSES GPUTRONIC_ACTIVE_SMS GPUTRONIC_IDLE_QUIET GPUTRONIC_PULSE_LOCAL
    if [ ! -f "$LOG/${name}.csv" ]; then
        echo "FAIL ${name}: no tally csv (exit ${rc})"
        FAILS=$((FAILS + 1))
        return
    fi
    python3 scripts/analyze_smaccum.py "$LOG/${name}.csv" > "$LOG/${name}_analysis.txt" 2>&1
    arc=$?
    zstd -q -f --rm -o "$LOG/${name}.csv.zst" "$LOG/${name}.csv"
    echo "--- ${name} analyzer exit ${arc} ---"
    grep -E '^\[(blocks|jump|pass|phys-jump|pulses|placement|100 ms|T11)\]|^  [0-9]+ [0-9]+ [0-9]+ [0-9]+' "$LOG/${name}_analysis.txt" || true
    work=$(sed -n 's/.*with work \([0-9][0-9]*\).*/\1/p' "$LOG/${name}_analysis.txt" | head -1)
    t11=$(grep -c 'T11 PASS' "$LOG/${name}_analysis.txt" || true)
    ratio=$(sed -n 's/.*\[100 ms\].*ratio mean \([0-9.][0-9.]*\).*/\1/p' "$LOG/${name}_analysis.txt" | head -1)
    ratio_ok=$(awk -v r="${ratio:-0}" 'BEGIN { print (r >= 0.95 && r <= 1.05) ? 1 : 0 }')
    if [ "$rc" -ne 0 ] || [ ! -s "$LOG/${name}.csv.zst" ] || [ "${work:-0}" != "$expect" ] || [ "$ratio_ok" -ne 1 ]; then
        echo "FAIL ${name}: exit=${rc} analyzer=${arc} work=${work:-none} expect=${expect} ratio=${ratio:-none} t11=${t11}"
        FAILS=$((FAILS + 1))
        if [ "${work:-0}" != "$expect" ]; then
            echo "STOP: working-set mismatch on ${name}"
            exit 4
        fi
    else
        if [ "$t11" -lt 1 ]; then
            echo "OK ${name} work=${work} ratio=${ratio} (correlation under the T11 bar)"
        else
            echo "OK ${name} work=${work}"
        fi
    fi
}

echo "fifth-pass suite $(date -Is) $(git rev-parse --short HEAD) temp=$(gpu_temp)"

# Experiment 1. Mapped pulse atomic batched to about 1 ms. Control reads stay every pass.
run_hold "e1-n84" 84 0 "" 0 84
run_hold "e1-n84-local" 84 0 "" 1 84
run_hold "e1-gpc0" 84 0 "0" 0 12
run_hold "e1-gpc0-local" 84 0 "0" 1 12
run_hold "e1-gpc3" 84 0 "3" 0 12
run_hold "e1-gpc3-local" 84 0 "3" 1 12

echo "SUITE DONE fails=${FAILS} $(date -Is) temp=$(gpu_temp)"
exit "$FAILS"
