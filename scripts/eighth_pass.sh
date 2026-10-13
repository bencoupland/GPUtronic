#!/bin/bash
# Eighth pass. Clamp sweep, real idle, and a peak-throughput hold.
# One GPU job. Stop at 83 C or after three failures.
set -u
cd "$(dirname "$0")/.."
LOG="logs/2026-10-13_eighth-pass"
mkdir -p "$LOG" results
echo $$ > "$LOG/suite.pid"
FAILS=0
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
        echo "BLOCKER: GPU at ${t} C" | tee "$LOG/BLOCKER.txt"
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

echo "EIGHTH START $(date -Is) temp=$(gpu_temp)"
nvidia-smi > "$LOG/nvidia-smi-start.txt"
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks.sm,utilization.gpu,display_active,display_mode,driver_version --format=csv > "$LOG/gpu-start.csv" || true
xrandr --query | awk '/ connected/{print}' > "$LOG/xrandr.txt" || true
if ! make NVCC=/usr/local/cuda-12.8/bin/nvcc ARCH=-arch=sm_120 gold lib transfer episode; then
    echo "BLOCKER: build failed"
    exit 2
fi

run_gold() {
    # name, timeout_s, dmon_samples, then the gold argv
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

# ---- checks, every switch unset ----
clear_switches
echo "=== check gold temp=$(gpu_temp) $(date -Is) ==="
timeout 180 "$BIN" check > "$LOG/check-gold.txt" 2>&1 || note_fail "check-gold: exit $?"
echo "=== check transfer ==="
timeout 180 ./build/gputronic_transfer check > "$LOG/check-transfer.txt" 2>&1 || note_fail "check-transfer: exit $?"
echo "=== check episode ==="
timeout 240 ./build/gputronic_episode check > "$LOG/check-episode.txt" 2>&1 || note_fail "check-episode: exit $?"
drop_fast_csv
echo "PHASE checks done fails=${FAILS} temp=$(gpu_temp)"

# ---- 1. clamp sweep on the cached plant ----
clamp_run() {
    local name="$1" sms="$2"
    clear_switches
    export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_TRACE_BIN_US=1000
    if [ -n "$sms" ]; then
        export GPUTRONIC_ICLAMP_SMS="$sms"
    fi
    run_gold "$name" 100 90 loadstep 60 1.2 20 45
    clear_switches
}

clamp_run t6-scaled ""
clamp_run t6-c6 6
clamp_run t6-c9 9
clamp_run t6-c12 12
echo "PHASE clamp-grid done fails=${FAILS} temp=$(gpu_temp)"

# The switches-off repeat of the smallest clamp that holds the band is started
# by the analyzer once the four traces exist. A missing pick is not a third plant.
python3 scripts/analyze_eighth.py --pick-clamp > "$LOG/clamp-pick.txt" || true
PICK=$(awk '/^PICK /{print $2}' "$LOG/clamp-pick.txt" | tail -1)
echo "clamp pick: ${PICK:-none}"
if [ -n "${PICK:-}" ] && [ "$PICK" != "none" ]; then
    clear_switches
    export GPUTRONIC_TRACE_BIN_US=1000 GPUTRONIC_ICLAMP_SMS="$PICK"
    run_gold "t6-off-${PICK}" 100 90 loadstep 60 1.2 20 45
    clear_switches
fi
echo "PHASE clamp done fails=${FAILS} temp=$(gpu_temp)"

# ---- 2. power sweeps ----
power_run() {
    local mode="$1" n="$2"
    local name="pwr-${mode}-n${n}"
    clear_switches
    export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_NO_CSV=1
    case "$mode" in
        spin) ;;
        exit) export GPUTRONIC_IDLE_EXIT=1 ;;
        relaunch) export GPUTRONIC_RELAUNCH=1 ;;
        *) note_fail "power mode ${mode}"; return ;;
    esac
    run_gold "$name" 70 55 noise "$n" 30
    clear_switches
}

for mode in spin exit relaunch; do
    for n in 4 12 24 36 48 56 72 82; do
        power_run "$mode" "$n"
    done
    echo "PHASE power-${mode} done fails=${FAILS} temp=$(gpu_temp)"
done

# ---- latency and the present probe ----
step_run() {
    local mode="$1" probe="$2"
    local name="step-${mode}"
    if [ "$probe" = "probe" ]; then
        name="step-${mode}-probe"
    fi
    clear_switches
    export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_NO_CSV=1
    export GPUTRONIC_STEP_AT_MS=2000 GPUTRONIC_STEP_SMS=42
    export GPUTRONIC_STEP2_AT_MS=5000 GPUTRONIC_STEP2_SMS=82
    case "$mode" in
        exit) export GPUTRONIC_IDLE_EXIT=1 ;;
        relaunch) export GPUTRONIC_RELAUNCH=1 ;;
    esac
    if [ "$probe" = "tally" ]; then
        export GPUTRONIC_SM_ACCUM=1 GPUTRONIC_TRACE_BIN_US=500
        # The binned trace is the latency record. NO_CSV would drop it.
        unset GPUTRONIC_NO_CSV
    fi
    if [ "$probe" = "probe" ]; then
        wait_cool
        echo "=== ${name} temp=$(gpu_temp) $(date -Is) ==="
        echo "dmon_start $(date +%s.%N)" > "$LOG/${name}.clock"
        stdbuf -oL nvidia-smi dmon -s pucvt -d 1 -c 30 > "$LOG/${name}.dmon.txt" &
        DMON_PID=$!
        stdbuf -oL -eL timeout 40 "$BIN" noise 82 16 > "$LOG/${name}.txt" 2>&1 &
        local gpid=$!
        local ready=0
        for _i in $(seq 1 80); do
            if grep -q '\[LOAD\] calibrated' "$LOG/${name}.txt" 2>/dev/null; then
                ready=1
                break
            fi
            if ! kill -0 "$gpid" 2>/dev/null; then
                break
            fi
            sleep 0.25
        done
        if [ "$ready" -ne 1 ]; then
            note_fail "${name}: did not reach the hold"
            kill "$gpid" 2>/dev/null || true
            wait "$gpid" 2>/dev/null || true
            stop_dmon
        else
            sleep 0.5
            timeout 20 python3 scripts/present_probe.py "$LOG/present-${mode}.csv" 12 \
                > "$LOG/present-${mode}.txt" 2>&1 || note_fail "${name}: present probe exit $?"
            wait "$gpid" || note_fail "${name}: gold exit $?"
            stop_dmon
        fi
        drop_fast_csv
        clear_switches
        wait_cool
        return
    fi
    run_gold "$name" 40 30 noise 82 8
    clear_switches
}

step_run exit tally
step_run exit probe
step_run relaunch tally
step_run relaunch probe
echo "PHASE latency done fails=${FAILS} temp=$(gpu_temp)"

# ---- 3. peak search, then a 60 s hold, then an hour if the die stays cool ----
best_f=0
best_r=1
best_c=1
best_t=64
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
    # Stop once a longer repeat does not add 5%.
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
echo "PHASE peak-search done fails=${FAILS} temp=$(gpu_temp)"

clear_switches
export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1
export GPUTRONIC_PEAK_REPS="$best_r" GPUTRONIC_PEAK_CHAINS="$best_c" GPUTRONIC_THREADS="$best_t"
export GPUTRONIC_TRACE_BIN_US=1000
run_gold peak-hold60 90 80 noise 82 60
clear_switches
nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE > "$LOG/peak-hold60-query.txt" || true

hold_max=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d ' ')
# The hold has already ended; use the dmon file if it recorded a higher sample.
if [ -f "$LOG/peak-hold60.dmon.txt" ]; then
    dmax=$(awk 'NF && $1 !~ /#/ {if ($3+0 > m) m=$3+0} END{print m+0}' "$LOG/peak-hold60.dmon.txt")
    if [ "${dmax:-0}" -gt "$hold_max" ]; then
        hold_max=$dmax
    fi
fi
echo "peak hold max temp ${hold_max} C"
if [ "$hold_max" -ge 70 ]; then
    echo "PHASE hour skipped temp=${hold_max}"
else
    clear_switches
    export GPUTRONIC_PULSE_LOCAL=1 GPUTRONIC_CTRL_CACHE=1 GPUTRONIC_NO_CSV=1
    export GPUTRONIC_PEAK_REPS="$best_r" GPUTRONIC_PEAK_CHAINS="$best_c" GPUTRONIC_THREADS="$best_t"
    # 3600 s hold. The timeout leaves room for calibration. Temp is polled beside it.
    wait_cool
    echo "=== peak-hour temp=$(gpu_temp) $(date -Is) ==="
    echo "dmon_start $(date +%s.%N)" > "$LOG/peak-hour.clock"
    stdbuf -oL nvidia-smi dmon -s pucvt -d 5 -c 800 > "$LOG/peak-hour.dmon.txt" &
    DMON_PID=$!
    stdbuf -oL -eL timeout 3800 "$BIN" noise 82 3600 > "$LOG/peak-hour.txt" 2>&1 &
    gpid=$!
    while kill -0 "$gpid" 2>/dev/null; do
        sleep 60
        t=$(gpu_temp)
        echo "HOUR temp=${t} $(date -Is)"
        if [ "$t" -ge 83 ]; then
            echo "BLOCKER: hour reached ${t} C"
            kill "$gpid" 2>/dev/null || true
            wait "$gpid" 2>/dev/null || true
            stop_dmon
            clear_switches
            exit 2
        fi
    done
    wait "$gpid" || note_fail "peak-hour: exit $?"
    stop_dmon
    clear_switches
    echo "PHASE hour done fails=${FAILS} temp=$(gpu_temp)"
fi

nvidia-smi > "$LOG/nvidia-smi-end.txt"
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks.sm,utilization.gpu,display_active,display_mode --format=csv > "$LOG/gpu-end.csv" || true
echo "SUITE DONE fails=${FAILS} $(date -Is) temp=$(gpu_temp)"
rm -f "$LOG/suite.pid"
exit 0
