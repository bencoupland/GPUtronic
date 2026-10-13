#!/bin/bash
# Follow-up holds after the first eighth-pass suite.
# Each idle-exit refill has its own stream. The peak FMA result stays live.
# Clamp, spin, and relaunch are not repeated.
# One GPU job. Stop at 83 C or after three failures.
set -u
cd "$(dirname "$0")/.."
LOG="logs/2026-10-13_eighth-pass"
BIN="./build/gputronic_gold"
FAILS=0
DMON_PID=""

if [ -f "$LOG/suite.pid" ]; then
    old=$(cat "$LOG/suite.pid")
    if kill -0 "$old" 2>/dev/null; then
        echo "BLOCKER: first suite still running pid=$old"
        exit 2
    fi
fi

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

note_fail() {
    echo "FAIL $1"
    FAILS=$((FAILS + 1))
    if [ "$FAILS" -ge 3 ]; then
        echo "BLOCKER: three failures"
        stop_dmon
        exit 2
    fi
}

wait_cool() {
    local t
    t=$(gpu_temp)
    echo "temp ${t} C"
    if [ "$t" -ge 83 ]; then
        echo "BLOCKER: GPU at ${t} C"
        stop_dmon
        exit 2
    fi
}

clear_switches() {
    unset GPUTRONIC_PULSE_LOCAL GPUTRONIC_CTRL_CACHE GPUTRONIC_CTRL_CACHE_US
    unset GPUTRONIC_STEP_AT_MS GPUTRONIC_STEP_SMS GPUTRONIC_STEP2_AT_MS GPUTRONIC_STEP2_SMS
    unset GPUTRONIC_ACTIVE_PASSES GPUTRONIC_ACTIVE_SMS GPUTRONIC_IDLE_QUIET
    unset GPUTRONIC_TENSOR_SENSOR GPUTRONIC_TRACE_BIN_US GPUTRONIC_SM_ACCUM
    unset GPUTRONIC_SM_ACCUM_SM GPUTRONIC_DEADBAND GPUTRONIC_INTEGRAL_CLAMP
    unset GPUTRONIC_OS_RESERVE GPUTRONIC_NO_CSV GPUTRONIC_ICLAMP_SMS
    unset GPUTRONIC_IDLE_EXIT GPUTRONIC_RELAUNCH GPUTRONIC_RELAUNCH_HOLD_US
    unset GPUTRONIC_PEAK_REPS GPUTRONIC_PEAK_CHAINS GPUTRONIC_THREADS
    unset GPUTRONIC_WORK_COUNTER
}

drop_fast_csv() {
    rm -f results/gputronic_gold_fullwidth.csv results/gputronic_gold_load.csv \
          results/gputronic_gold.csv results/gputronic_gold_noise.csv \
          results/gputronic_gold_fullwidth_10ms.csv results/gputronic_gold_loadstep_10ms.csv \
          results/gputronic_gold_smaccum.csv
}

trap 'stop_dmon' EXIT

mkdir -p "$LOG/before-fix"
echo "First idle-exit refills after the first wave shared one stream and did not start. The peak repeat loops were not live. These logs are that binary." > "$LOG/before-fix/WHY.txt"
shopt -s nullglob
mv -t "$LOG/before-fix" \
    "$LOG"/pwr-exit-n*.txt "$LOG"/pwr-exit-n*.dmon.txt "$LOG"/pwr-exit-n*.clock \
    "$LOG"/step-exit.txt "$LOG"/step-exit.csv "$LOG"/step-exit.clock "$LOG"/step-exit.dmon.txt \
    "$LOG"/step-exit.smaccum.csv "$LOG"/step-exit.smaccum.csv.zst \
    "$LOG"/step-exit-probe.txt "$LOG"/step-exit-probe.dmon.txt "$LOG"/step-exit-probe.clock \
    "$LOG"/present-exit.csv "$LOG"/present-exit.txt \
    "$LOG"/peak-r*.txt "$LOG"/peak-r*.dmon.txt "$LOG"/peak-r*.clock \
    "$LOG"/peak-hold60.txt "$LOG"/peak-hold60.csv "$LOG"/peak-hold60.dmon.txt \
    "$LOG"/peak-hold60.clock "$LOG"/peak-hold60-query.txt \
    "$LOG"/peak-hour.txt "$LOG"/peak-hour.dmon.txt "$LOG"/peak-hour.clock \
    "$LOG"/peak-search.txt "$LOG"/peak-best.txt
shopt -u nullglob

echo $$ > "$LOG/follow.pid"
echo "FOLLOW START $(date -Is) temp=$(gpu_temp)"
if ! make NVCC=/usr/local/cuda-12.8/bin/nvcc ARCH=-arch=sm_120 gold lib transfer episode; then
    echo "BLOCKER: build failed"
    exit 2
fi

run_gold() {
    local name="$1" to="$2" samples="$3"
    shift 3
    wait_cool
    echo "=== ${name} temp=$(gpu_temp) $(date -Is) ==="
    echo "dmon_start $(date +%s.%N)" > "$LOG/${name}.clock"
    stdbuf -oL nvidia-smi dmon -s pucvt -d 1 -c "$samples" > "$LOG/${name}.dmon.txt" &
    DMON_PID=$!
    stdbuf -oL -eL timeout "$to" "$BIN" "$@" > "$LOG/${name}.txt" 2>&1
    local rc=$?
    stop_dmon
    echo "dmon_end $(date +%s.%N) rc=${rc}" >> "$LOG/${name}.clock"
    if [ -f results/gputronic_gold_loadstep_10ms.csv ]; then
        mv results/gputronic_gold_loadstep_10ms.csv "$LOG/${name}.csv"
    fi
    if [ -f results/gputronic_gold_noise.csv ]; then
        mv results/gputronic_gold_noise.csv "$LOG/${name}.csv"
    fi
    if [ -f results/gputronic_gold_smaccum.csv ]; then
        zstd -q -f results/gputronic_gold_smaccum.csv -o "$LOG/${name}.smaccum.csv.zst" || \
            mv results/gputronic_gold_smaccum.csv "$LOG/${name}.smaccum.csv"
        rm -f results/gputronic_gold_smaccum.csv
    fi
    drop_fast_csv
    if [ "$rc" -ne 0 ]; then
        note_fail "${name}: exit ${rc}"
    fi
    if ! grep -q "Stopped cleanly" "$LOG/${name}.txt"; then
        note_fail "${name}: did not stop cleanly"
    fi
    wait_cool
}

clear_switches
echo "=== check2 gold temp=$(gpu_temp) $(date -Is) ==="
timeout 180 "$BIN" check > "$LOG/check2-gold.txt" 2>&1 || note_fail "check2-gold: exit $?"
echo "=== check2 transfer ==="
timeout 180 ./build/gputronic_transfer check > "$LOG/check2-transfer.txt" 2>&1 || note_fail "check2-transfer: exit $?"
echo "=== check2 episode ==="
timeout 240 ./build/gputronic_episode check > "$LOG/check2-episode.txt" 2>&1 || note_fail "check2-episode: exit $?"
drop_fast_csv
echo "PHASE follow-checks done fails=${FAILS} temp=$(gpu_temp)"

for n in 4 12 24 36 48 56 72 82; do
    clear_switches
    export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_NO_CSV=1
    export GPUTRONIC_IDLE_EXIT=1
    run_gold "pwr-exit-n${n}" 70 55 noise "$n" 30
    clear_switches
done
echo "PHASE follow-exit done fails=${FAILS} temp=$(gpu_temp)"

clear_switches
export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1
export GPUTRONIC_IDLE_EXIT=1
export GPUTRONIC_STEP_AT_MS=2000 GPUTRONIC_STEP_SMS=42
export GPUTRONIC_STEP2_AT_MS=5000 GPUTRONIC_STEP2_SMS=82
export GPUTRONIC_SM_ACCUM=1 GPUTRONIC_TRACE_BIN_US=500
run_gold step-exit 40 30 noise 82 8
clear_switches

clear_switches
export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_NO_CSV=1
export GPUTRONIC_IDLE_EXIT=1
export GPUTRONIC_STEP_AT_MS=2000 GPUTRONIC_STEP_SMS=42
export GPUTRONIC_STEP2_AT_MS=5000 GPUTRONIC_STEP2_SMS=82
wait_cool
echo "=== step-exit-probe temp=$(gpu_temp) $(date -Is) ==="
echo "dmon_start $(date +%s.%N)" > "$LOG/step-exit-probe.clock"
stdbuf -oL nvidia-smi dmon -s pucvt -d 1 -c 30 > "$LOG/step-exit-probe.dmon.txt" &
DMON_PID=$!
stdbuf -oL -eL timeout 40 "$BIN" noise 82 16 > "$LOG/step-exit-probe.txt" 2>&1 &
gpid=$!
ready=0
for _i in $(seq 1 80); do
    if grep -q '\[LOAD\] calibrated' "$LOG/step-exit-probe.txt" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$gpid" 2>/dev/null; then
        break
    fi
    sleep 0.25
done
if [ "$ready" -ne 1 ]; then
    note_fail "step-exit-probe: did not reach the hold"
    kill "$gpid" 2>/dev/null || true
    wait "$gpid" 2>/dev/null || true
    stop_dmon
else
    sleep 0.5
    timeout 20 python3 scripts/present_probe.py "$LOG/present-exit.csv" 12 \
        > "$LOG/present-exit.txt" 2>&1 || note_fail "step-exit-probe: present probe exit $?"
    wait "$gpid" || note_fail "step-exit-probe: gold exit $?"
    stop_dmon
fi
drop_fast_csv
clear_switches
echo "PHASE follow-latency done fails=${FAILS} temp=$(gpu_temp)"

best_f=0
best_r=1
best_c=1
best_t=64
: > "$LOG/peak-search.txt"
peak_one() {
    local r="$1" c="$2" t="$3"
    local name="peak-r${r}-c${c}-t${t}"
    clear_switches
    export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_NO_CSV=1
    export GPUTRONIC_PEAK_REPS="$r" GPUTRONIC_PEAK_CHAINS="$c" GPUTRONIC_THREADS="$t"
    run_gold "$name" 40 25 noise 82 8
    clear_switches
    local rate
    rate=$(awk '/\[TEL\]/{r=$0} END{print r}' "$LOG/${name}.txt" | sed -n 's/.*rate=\([0-9][0-9]*\).*/\1/p')
    if [ -z "$rate" ]; then
        note_fail "${name}: no rate"
        return
    fi
    local flops
    flops=$(python3 -c "print(int(${rate}) * 65536 * ${r} * ${c})")
    echo "PEAK ${name} rate=${rate} flops=${flops}" | tee -a "$LOG/peak-search.txt"
    if [ "$flops" -gt "$best_f" ]; then
        best_f=$flops
        best_r=$r
        best_c=$c
        best_t=$t
    fi
}

for c in 1 4 8 16; do
    peak_one 1 "$c" 64
done
chain=$best_c
prev=$best_f
for r in 2 4 8 16 32; do
    peak_one "$r" "$chain" 64
    python3 -c "import sys; sys.exit(0 if ${best_f} < ${prev} * 1.05 else 1)" && break
    prev=$best_f
done
prev=$best_f
for t in 128 256 512 1024; do
    peak_one "$best_r" "$best_c" "$t"
    python3 -c "import sys; sys.exit(0 if ${best_f} < ${prev} * 1.05 else 1)" && break
    prev=$best_f
done
echo "PEAK best reps=${best_r} chains=${best_c} threads=${best_t} flops=${best_f}" | tee "$LOG/peak-best.txt"
echo "PHASE follow-peak-search done fails=${FAILS} temp=$(gpu_temp)"

clear_switches
export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1
export GPUTRONIC_PEAK_REPS="$best_r" GPUTRONIC_PEAK_CHAINS="$best_c" GPUTRONIC_THREADS="$best_t"
export GPUTRONIC_TRACE_BIN_US=1000
run_gold peak-hold60 90 80 noise 82 60
clear_switches
nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE > "$LOG/peak-hold60-query.txt" || true

hold_max=$(gpu_temp)
if [ -f "$LOG/peak-hold60.dmon.txt" ]; then
    dmax=$(awk 'NF && $1 !~ /#/ {if ($3+0 > m) m=$3+0} END{print m+0}' "$LOG/peak-hold60.dmon.txt")
    dmax=${dmax%.*}
    if [ "${dmax:-0}" -gt "$hold_max" ]; then
        hold_max=$dmax
    fi
fi
echo "peak hold max temp ${hold_max} C"
echo "PHASE follow-hour skipped after the 60 s hold temp=${hold_max}"

nvidia-smi > "$LOG/nvidia-smi-follow-end.txt"
echo "FOLLOW DONE fails=${FAILS} $(date -Is) temp=$(gpu_temp)"
rm -f "$LOG/follow.pid"
exit 0
