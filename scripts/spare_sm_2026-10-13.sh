#!/bin/bash
# How many SMs the 4K60 desktop needs.
# A persistent Gold grid stays resident. A small window and one browser
# window time their frames. Reserve 0, 1, 2, and 4. The default stays 2
# unless this run shows that 2 misses vblank.
set -u
cd "$(dirname "$0")/.."
LOG="logs/2026-10-13_spare-sm"
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
trap stop_dmon EXIT

wait_cool() {
    local t
    t=$(gpu_temp)
    echo "temp ${t} C"
    if [ "$t" -ge 83 ]; then
        echo "BLOCKER: GPU at ${t} C" | tee "$LOG/BLOCKER.txt"
        exit 2
    fi
}

drop_fast_csv() {
    rm -f results/gputronic_gold_fullwidth.csv results/gputronic_gold_load.csv \
          results/gputronic_gold.csv results/gputronic_gold_noise.csv \
          results/gputronic_gold_fullwidth_10ms.csv
}

clear_switches() {
    unset GPUTRONIC_PULSE_LOCAL GPUTRONIC_CTRL_CACHE GPUTRONIC_CTRL_CACHE_US
    unset GPUTRONIC_STEP_AT_MS GPUTRONIC_STEP_SMS GPUTRONIC_STEP2_AT_MS GPUTRONIC_STEP2_SMS
    unset GPUTRONIC_ACTIVE_PASSES GPUTRONIC_ACTIVE_SMS GPUTRONIC_IDLE_QUIET
    unset GPUTRONIC_TENSOR_SENSOR GPUTRONIC_TRACE_BIN_US GPUTRONIC_SM_ACCUM
    unset GPUTRONIC_SM_ACCUM_SM GPUTRONIC_DEADBAND GPUTRONIC_INTEGRAL_CLAMP
    unset GPUTRONIC_OS_RESERVE GPUTRONIC_NO_CSV
}

echo "SPARE SM START $(date -Is) temp=$(gpu_temp)"
nvidia-smi > "$LOG/nvidia-smi-start.txt"
xrandr --query | awk '/ connected/{print}' > "$LOG/xrandr.txt"
cat "$LOG/xrandr.txt"
make NVCC=/usr/local/cuda-12.8/bin/nvcc ARCH=-arch=sm_120 gold

for reserve in 0 1 2 4; do
    n=$((84 - reserve))
    name="r${reserve}"
    wait_cool
    clear_switches
    export GPUTRONIC_OS_RESERVE="$reserve" GPUTRONIC_NO_CSV=1
    echo "=== ${name} launch ${n} temp=$(gpu_temp) $(date -Is) ==="
    stdbuf -oL nvidia-smi dmon -s pucvt -d 1 -c 70 > "$LOG/${name}.dmon.txt" &
    DMON_PID=$!
    stdbuf -oL -eL "$BIN" noise "$n" 50 > "$LOG/gold-${name}.txt" 2>&1 &
    gpid=$!
    ready=0
    for _i in $(seq 1 80); do
        if grep -q '\[LOAD\] calibrated' "$LOG/gold-${name}.txt" 2>/dev/null; then
            ready=1
            break
        fi
        if ! kill -0 "$gpid" 2>/dev/null; then
            break
        fi
        sleep 0.25
    done
    if [ "$ready" -ne 1 ]; then
        echo "FAIL ${name}: kernel did not reach the open-loop hold"
        FAILS=$((FAILS + 1))
        kill "$gpid" 2>/dev/null || true
        wait "$gpid" 2>/dev/null || true
        stop_dmon
        drop_fast_csv
        continue
    fi
    if ! grep -q "launching ${n}, ${reserve} reserved" "$LOG/gold-${name}.txt"; then
        echo "FAIL ${name}: launch line is not ${n} blocks"
        FAILS=$((FAILS + 1))
    fi
    sleep 1
    echo "present probe"
    timeout 25 python3 scripts/present_probe.py "$LOG/present-${name}.csv" 12 \
        > "$LOG/present-${name}.txt" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "FAIL ${name}: present probe exit ${rc}"
        FAILS=$((FAILS + 1))
    fi
    echo "browser window"
    timeout 30 python3 scripts/frame_clock_server.py "$LOG/browser-${name}.csv" 12 8877 \
        > "$LOG/browser-${name}.server.txt" 2>&1 &
    spid=$!
    for _i in $(seq 1 50); do
        if grep -q '^listening' "$LOG/browser-${name}.server.txt" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    firefox --new-window "http://127.0.0.1:8877/" >/dev/null 2>&1 &
    for _i in $(seq 1 60); do
        if wmctrl -l | grep -q "GPUTronic frame clock"; then
            break
        fi
        sleep 0.25
    done
    wmctrl -r "GPUTronic frame clock" -e 0,140,700,1100,720 || true
    wmctrl -a "GPUTronic frame clock" || true
    for _i in $(seq 1 30); do
        if ! kill -0 "$spid" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 "$spid" 2>/dev/null; then
        echo "FAIL ${name}: browser posted no frames"
        FAILS=$((FAILS + 1))
        kill "$spid" 2>/dev/null || true
        wait "$spid" 2>/dev/null || true
    fi
    wmctrl -c "GPUTronic frame clock" || true
    sleep 0.5
    kill "$gpid" 2>/dev/null || true
    wait "$gpid" 2>/dev/null || true
    stop_dmon
    drop_fast_csv
    clear_switches
    echo "temp $(gpu_temp) C"
    if [ "$(gpu_temp)" -ge 83 ]; then
        echo "BLOCKER: ${name} reached $(gpu_temp) C" | tee "$LOG/BLOCKER.txt"
        exit 2
    fi
done

nvidia-smi > "$LOG/nvidia-smi-end.txt"
echo "SUITE DONE fails=${FAILS} $(date -Is) temp=$(gpu_temp)"
rm -f "$LOG/suite.pid"
exit "$FAILS"
