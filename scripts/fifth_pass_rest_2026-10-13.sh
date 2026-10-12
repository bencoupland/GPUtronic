#!/bin/bash
# Fifth-pass follow-ups. Run after Experiment 1, on a binary that has
# GPUTRONIC_CTRL_CACHE. Kp, Ki, the deadband, and the clamp are not touched.
set -u
cd "$(dirname "$0")/.."
LOG="logs/2026-10-13_fifth-pass"
mkdir -p "$LOG" results
FAILS=0
export GPUTRONIC_SM_ACCUM=1
unset GPUTRONIC_TENSOR_SENSOR GPUTRONIC_ACTIVE_PASSES GPUTRONIC_ACTIVE_SMS
unset GPUTRONIC_IDLE_QUIET GPUTRONIC_PULSE_LOCAL GPUTRONIC_CTRL_CACHE

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

# run_hold name nsm quiet passes sms pulse cache expect
run_hold() {
    local name="$1" nsm="$2" quiet="$3" passes="$4" sms="$5" pulse="$6" cache="$7" expect="$8"
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
    unset GPUTRONIC_ACTIVE_PASSES GPUTRONIC_ACTIVE_SMS GPUTRONIC_IDLE_QUIET
    unset GPUTRONIC_PULSE_LOCAL GPUTRONIC_CTRL_CACHE
    if [ "$quiet" = "1" ]; then export GPUTRONIC_IDLE_QUIET=1; fi
    if [ -n "$passes" ]; then export GPUTRONIC_ACTIVE_PASSES="$passes"; fi
    if [ -n "$sms" ]; then export GPUTRONIC_ACTIVE_SMS="$sms"; fi
    if [ "$pulse" = "1" ]; then export GPUTRONIC_PULSE_LOCAL=1; fi
    if [ "$cache" = "1" ]; then export GPUTRONIC_CTRL_CACHE=1; fi
    t0=$(gpu_temp)
    echo "=== ${name} nsm=${nsm} quiet=${quiet} passes='${passes}' sms='${sms}' pulse=${pulse} cache=${cache} expect=${expect} temp=${t0} $(date -Is) ==="
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
    unset GPUTRONIC_ACTIVE_PASSES GPUTRONIC_ACTIVE_SMS GPUTRONIC_IDLE_QUIET
    unset GPUTRONIC_PULSE_LOCAL GPUTRONIC_CTRL_CACHE
    if [ "$cache" = "1" ] && ! grep -q '\[CTRL\] cached control words' "$LOG/${name}.txt"; then
        echo "FAIL ${name}: control-word cache did not engage"
        FAILS=$((FAILS + 1))
        return
    fi
    if [ "$pulse" = "1" ] && ! grep -q '\[PULSE\] local count' "$LOG/${name}.txt"; then
        echo "FAIL ${name}: local pulse count did not engage"
        FAILS=$((FAILS + 1))
        return
    fi
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

echo "fifth-pass rest $(date -Is) $(git rev-parse --short HEAD) temp=$(gpu_temp)"

# Experiment 1 follow-up. Both test switches on: the mapped atomic is already
# batched, and the control words are read once per millisecond. 84 SMs.
run_hold "e1-n84-cached" 84 0 "" "" 1 1 84

# Experiment 3. Firmware TPC ids, quiet on, noise 84 so the grid stays resident.
# GPC 0 pairs: TPC0 0/1, TPC2 14/15, TPC4 28/29, TPC1 42/43, TPC3 56/57, TPC5 70/71.
# GPC 3 pairs: TPC0 6/7, TPC2 20/21, TPC4 34/35, TPC1 48/49, TPC3 62/63, TPC5 76/77.
run_hold "e3-gpc0-tpc0"    84 1 "" "0,1"                         0 0 2
run_hold "e3-gpc0-split"   84 1 "" "0,14"                        0 0 2
run_hold "e3-gpc0-cluster" 84 1 "" "0,1,14,15,42,43"             0 0 6
run_hold "e3-gpc0-spread"  84 1 "" "0,1,14,15,28,29"             0 0 6
run_hold "e3-gpc0-all"     84 1 "" "0,1,14,15,28,29,42,43,56,57,70,71" 0 0 12
run_hold "e3-gpc3-tpc0"    84 1 "" "6,7"                         0 0 2
run_hold "e3-gpc3-split"   84 1 "" "6,20"                        0 0 2
run_hold "e3-gpc3-cluster" 84 1 "" "6,7,20,21,48,49"             0 0 6
run_hold "e3-gpc3-spread"  84 1 "" "6,7,20,21,34,35"             0 0 6
run_hold "e3-gpc3-all"     84 1 "" "6,7,20,21,34,35,48,49,62,63,76,77" 0 0 12

# Experiment 4. Three repeats. Do not treat a repeat as a new explanation.
run_hold "e4-n60-a" 60 1 "" "" 0 0 60
run_hold "e4-n60-b" 60 1 "" "" 0 0 60
run_hold "e4-n60-c" 60 1 "" "" 0 0 60
run_hold "e4-p024-a" 84 1 "0,2,4" "" 0 0 36
run_hold "e4-p024-b" 84 1 "0,2,4" "" 0 0 36
run_hold "e4-p024-c" 84 1 "0,2,4" "" 0 0 36

echo "SUITE DONE fails=${FAILS} $(date -Is) temp=$(gpu_temp)"
exit "$FAILS"
