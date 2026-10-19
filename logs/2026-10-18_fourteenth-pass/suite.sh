#!/bin/bash
# Fourteenth pass. One GPU job. Stop a run at 83 C. Two stops end the suite.
# Item 1 retry bound: 4 launches, and stop earlier once a low group and one
# later comparison exist. Item 2 runs only if a low group was seen, and each
# setting is retried once when it is not in the low group.
set -u
ROOT="/tmp/gputronic-fourteenth"
LOG="$ROOT/logs/2026-10-18_fourteenth-pass"
STATUS="$LOG/status.txt"
mkdir -p "$LOG" "$ROOT/results"
cd "$ROOT"
cp -f /tmp/fourteenth-suite.sh "$LOG/suite.sh"
echo $$ > "$LOG/suite.pid"
: > "$STATUS"
: > "$LOG/groups.tsv"
TEMP_STOPS=0

say() { echo "$*" >> "$LOG/suite.log"; }

unset_switches() {
    unset GPUTRONIC_PEAK_ACC GPUTRONIC_PEAK_ROUNDS GPUTRONIC_PEAK_REPS GPUTRONIC_PEAK_CHAINS \
        GPUTRONIC_THREADS GPUTRONIC_PULSE_LOCAL GPUTRONIC_CTRL_CACHE GPUTRONIC_CTRL_CACHE_US \
        GPUTRONIC_RATE_WIN_US GPUTRONIC_TRACE_BIN_US GPUTRONIC_WORK_LOAD \
        GPUTRONIC_IDLE_EXIT GPUTRONIC_RELAUNCH \
        GPUTRONIC_GUEST_BLOCKS GPUTRONIC_GUEST_AT_MS GPUTRONIC_GUEST_STREAM_FIRST \
        GPUTRONIC_GUEST_PRIORITY GPUTRONIC_GOLD_NONBLOCK GPUTRONIC_NO_CSV \
        GPUTRONIC_WORK_COUNTER GPUTRONIC_SM_ACCUM GPUTRONIC_SM_ACCUM_SM \
        GPUTRONIC_IDLE_WORK GPUTRONIC_IDLE_QUIET GPUTRONIC_TARGET_Z GPUTRONIC_ICLAMP_SMS \
        GPUTRONIC_TENSOR_SENSOR GPUTRONIC_OS_RESERVE GPUTRONIC_INTEGRAL_CLAMP \
        GPUTRONIC_STEP_AT_MS GPUTRONIC_STEP_SMS GPUTRONIC_STEP2_AT_MS GPUTRONIC_STEP2_SMS \
        GPUTRONIC_DEADBAND GPUTRONIC_SENSOR_RULE GPUTRONIC_RESERVE_SMS
}

hold_env() {
    unset_switches
    export GPUTRONIC_PEAK_ACC=32
    export GPUTRONIC_PEAK_ROUNDS=512
    export GPUTRONIC_THREADS=512
    export GPUTRONIC_PULSE_LOCAL=1
    export GPUTRONIC_CTRL_CACHE=1
    export GPUTRONIC_TRACE_BIN_US=1000
    export GPUTRONIC_WORK_LOAD=1
}

die_c() {
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

wait_cool() {
    local t
    while true; do
        t=$(die_c)
        if [ "${t:-99}" -lt 70 ]; then
            say "cool ${t}C"
            return 0
        fi
        say "waiting ${t}C"
        sleep 5
    done
}

fail() {
    echo "SUITE_FAIL $*" >> "$STATUS"
    say "SUITE_FAIL $*"
    exit 1
}

classify() {
    local name="$1"
    local rate group
    rate=$(python3 - "$LOG/$name.txt" << 'PY'
import re, sys
t = open(sys.argv[1], errors="replace").read()
m = re.search(r"\[CAL\] rate_ref=([0-9.]+)", t)
print(m.group(1) if m else "NA")
PY
)
    group=na
    if [ "$rate" != "NA" ]; then
        group=$(python3 - "$rate" << 'PY'
import sys
r = float(sys.argv[1])
if r < 48000000:
    print("low")
elif r >= 49000000:
    print("high")
else:
    print("mid")
PY
)
    fi
    echo "$name $group $rate" >> "$LOG/groups.tsv"
    echo "GROUP $name $group $rate" >> "$STATUS"
    say "GROUP $name $group $rate"
    LAST_GROUP=$group
    LAST_RATE=$rate
}

snap_proc() {
    local name="$1" when="$2"
    {
        date -Iseconds
        nvidia-smi
        echo "--- compute apps ---"
        nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
    } > "$LOG/$name-proc-$when.txt" 2>&1 || true
}

clear_results() {
    rm -f "$ROOT/results/gputronic_gold_load.csv" \
          "$ROOT/results/gputronic_gold_loadfile_10ms.csv" \
          "$ROOT/results/gputronic_gold_fullwidth.csv" \
          "$ROOT/results/gputronic_gold_fullwidth_10ms.csv" \
          "$ROOT/results/gputronic_gold_smaccum.csv"
}

move_products() {
    local name="$1" kind="$2"
    if [ "$kind" = "load" ]; then
        if [ -f "$ROOT/results/gputronic_gold_load.csv" ]; then
            mv "$ROOT/results/gputronic_gold_load.csv" "$LOG/$name-tick.csv"
        fi
        if [ -f "$ROOT/results/gputronic_gold_loadfile_10ms.csv" ]; then
            mv "$ROOT/results/gputronic_gold_loadfile_10ms.csv" "$LOG/$name-1ms.csv"
        fi
    else
        if [ -f "$ROOT/results/gputronic_gold_fullwidth.csv" ]; then
            mv "$ROOT/results/gputronic_gold_fullwidth.csv" "$LOG/$name-tick.csv"
        fi
        if [ -f "$ROOT/results/gputronic_gold_fullwidth_10ms.csv" ]; then
            mv "$ROOT/results/gputronic_gold_fullwidth_10ms.csv" "$LOG/$name-1ms.csv"
        fi
    fi
    if [ -f "$ROOT/results/gputronic_gold_smaccum.csv" ]; then
        mv "$ROOT/results/gputronic_gold_smaccum.csv" "$LOG/$name-smaccum.csv"
    fi
}

# kind is load or full. Remaining args are the binary command.
launch_gpu() {
    local name="$1" kind="$2" sec="$3"
    shift 3
    wait_cool
    say "RUN $name kind=$kind sec=$sec"
    echo "RUN $name" >> "$STATUS"
    clear_results
    snap_proc "$name" start
    local dmon_n=$((sec + 80))
    nvidia-smi dmon -s pucm -d 1 -c "$dmon_n" > "$LOG/$name.dmon" 2>&1 &
    local dpid=$!
    nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks_event_reasons.active \
        --format=csv -l 1 > "$LOG/$name.flags" 2>&1 &
    local fpid=$!
    nvidia-smi --query-gpu=timestamp,display_active,display_mode,pstate,utilization.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active \
        --format=csv -lms 1000 > "$LOG/$name.display" 2>&1 &
    local ypid=$!
    {
        echo "ENV $(date -Iseconds)"
        env | grep '^GPUTRONIC_' | sort
        echo "---"
        "$@"
    } > "$LOG/$name.txt" 2>&1 &
    local gpid=$!
    local stopped=0
    while kill -0 "$gpid" 2>/dev/null; do
        local t
        t=$(die_c)
        say "alive $name ${t:-na}C"
        if [ "${t:-0}" -ge 83 ]; then
            echo "TEMP_STOP $name ${t}C" >> "$STATUS"
            say "TEMP_STOP $name ${t}C"
            kill -TERM "$gpid" 2>/dev/null || true
            stopped=1
            wait "$gpid" || true
            TEMP_STOPS=$((TEMP_STOPS + 1))
            break
        fi
        sleep 5
    done
    local rc=0
    if [ "$stopped" -eq 0 ]; then
        wait "$gpid" || rc=$?
    fi
    sleep 1
    kill "$dpid" "$fpid" "$ypid" 2>/dev/null || true
    wait "$dpid" 2>/dev/null || true
    wait "$fpid" 2>/dev/null || true
    wait "$ypid" 2>/dev/null || true
    snap_proc "$name" end
    move_products "$name" "$kind"
    if [ "$rc" -ne 0 ]; then
        echo "RUN_FAIL $name exit $rc" >> "$STATUS"
        fail "$name exit $rc"
    fi
    if [ "$stopped" -eq 0 ]; then
        if [ ! -s "$LOG/$name-1ms.csv" ]; then
            echo "RUN_FAIL $name no trace" >> "$STATUS"
            fail "$name no trace"
        fi
        if [ ! -s "$LOG/$name-tick.csv" ]; then
            echo "RUN_FAIL $name no tick" >> "$STATUS"
            fail "$name no tick"
        fi
        local last need
        last=$(awk -F, 'NR>1{v=$2} END{printf "%.0f", v+0}' "$LOG/$name-1ms.csv")
        need=$((sec * 1000 - 3000))
        if [ "${last:-0}" -lt "$need" ]; then
            echo "RUN_FAIL $name short trace ${last}ms" >> "$STATUS"
            fail "$name short trace ${last}ms"
        fi
        if [ "${GPUTRONIC_SM_ACCUM:-}" = "1" ] && [ ! -s "$LOG/$name-smaccum.csv" ]; then
            echo "RUN_FAIL $name no smaccum" >> "$STATUS"
            fail "$name no smaccum"
        fi
    fi
    classify "$name"
    if [ "$stopped" -eq 0 ]; then
        echo "RUN_OK $name" >> "$STATUS"
        say "RUN_OK $name"
    else
        echo "RUN_TEMP $name" >> "$STATUS"
        say "RUN_TEMP $name stops=$TEMP_STOPS"
        if [ "$TEMP_STOPS" -ge 2 ]; then
            echo "SUITE_DONE temp" >> "$STATUS"
            say "SUITE_DONE temp"
            exit 0
        fi
    fi
}

run_check() {
    local name="$1"
    shift
    say "CHECK $name"
    echo "CHECK $name" >> "$STATUS"
    unset_switches
    snap_proc "$name" start
    nvidia-smi --query-gpu=timestamp,display_active,display_mode,pstate,utilization.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active \
        --format=csv -lms 1000 > "$LOG/$name.display" 2>&1 &
    local ypid=$!
    local rc=0
    "$@" > "$LOG/$name.txt" 2>&1 || rc=$?
    kill "$ypid" 2>/dev/null || true
    wait "$ypid" 2>/dev/null || true
    snap_proc "$name" end
    if [ "$rc" -ne 0 ]; then
        echo "RUN_FAIL $name exit $rc" >> "$STATUS"
        fail "$name exit $rc"
    fi
    echo "RUN_OK $name" >> "$STATUS"
    say "CHECK ok $name"
}

echo "START $(date -Iseconds)" >> "$STATUS"
nvidia-smi > "$LOG/nvidia-smi-start.txt" 2>&1 || true
git -C "$ROOT" rev-parse HEAD > "$LOG/git-rev-start.txt"
{
    echo "ENV $(date -Iseconds)"
    env | grep '^GPUTRONIC_' | sort || true
    echo "---"
    ./build/gputronic_gold help
} > "$LOG/help.txt" 2>&1 || true

run_check unset-home ./build/gputronic_gold run 3
run_check check-gold ./build/gputronic_gold check
run_check check-transfer ./build/gputronic_transfer check
run_check check-episode ./build/gputronic_episode check
echo "CHECKS_DONE" >> "$STATUS"
say "CHECKS_DONE"

cat > "$LOG/steps-item1.txt" << 'EOF'
20000 1.2
65000 1.0
EOF

# Item 1. One pair, 110 s. Up to four launches.
GOT_LOW=0
GOT_HIGH=0
HAD_LOW=0
for tag in a b c d; do
    hold_env
    export GPUTRONIC_TARGET_Z=1.5
    export GPUTRONIC_ICLAMP_SMS=15
    launch_gpu "i1-$tag" load 110 ./build/gputronic_gold loadfile "$LOG/steps-item1.txt" 110
    if [ "$LAST_GROUP" = "low" ]; then GOT_LOW=1; fi
    if [ "$LAST_GROUP" = "high" ]; then GOT_HIGH=1; fi
    if [ "$GOT_LOW" -eq 1 ] && [ "$GOT_HIGH" -eq 1 ]; then
        break
    fi
    if [ "$HAD_LOW" -eq 1 ]; then
        break
    fi
    if [ "$LAST_GROUP" = "low" ]; then HAD_LOW=1; fi
done
echo "ITEM1_DONE low=$GOT_LOW high=$GOT_HIGH" >> "$STATUS"
say "ITEM1_DONE low=$GOT_LOW high=$GOT_HIGH"

# Item 2. Same-group low baseline, then faster cache and a shorter window.
apply_low() {
    hold_env
    export GPUTRONIC_TARGET_Z=1.5
    export GPUTRONIC_ICLAMP_SMS=15
    local kv
    for kv in "$@"; do
        export "$kv"
    done
}

run_low_setting() {
    local name="$1"
    shift
    apply_low "$@"
    launch_gpu "$name" load 110 ./build/gputronic_gold loadfile "$LOG/steps-item1.txt" 110
    if [ "$LAST_GROUP" != "low" ]; then
        apply_low "$@"
        launch_gpu "${name}b" load 110 ./build/gputronic_gold loadfile "$LOG/steps-item1.txt" 110
    fi
}

if [ "$GOT_LOW" -eq 1 ]; then
    run_low_setting i2-base
    if [ "$LAST_GROUP" != "low" ]; then
        echo "ITEM2_SKIP baseline not low" >> "$STATUS"
        say "ITEM2_SKIP baseline not low"
    else
        run_low_setting i2-c250 GPUTRONIC_CTRL_CACHE_US=250
        run_low_setting i2-c100 GPUTRONIC_CTRL_CACHE_US=100
        run_low_setting i2-w4 GPUTRONIC_RATE_WIN_US=4000
        run_low_setting i2-w2 GPUTRONIC_RATE_WIN_US=2000
        run_low_setting i2-both GPUTRONIC_CTRL_CACHE_US=100 GPUTRONIC_RATE_WIN_US=2000
        echo "ITEM2_DONE" >> "$STATUS"
        say "ITEM2_DONE"
    fi
else
    echo "ITEM2_SKIP no low group in item 1" >> "$STATUS"
    say "ITEM2_SKIP no low group in item 1"
fi

# Item 3. Eight fullwidth 15 s, interleaved with three closed-loop pairs.
hold_env
launch_gpu fw-01 full 15 ./build/gputronic_gold fullwidth 15
hold_env
export GPUTRONIC_TARGET_Z=1.5
export GPUTRONIC_ICLAMP_SMS=15
launch_gpu cl-01 load 110 ./build/gputronic_gold loadfile "$LOG/steps-item1.txt" 110
for tag in 02 03; do
    hold_env
    launch_gpu "fw-$tag" full 15 ./build/gputronic_gold fullwidth 15
done
hold_env
export GPUTRONIC_TARGET_Z=1.5
export GPUTRONIC_ICLAMP_SMS=15
launch_gpu cl-02 load 110 ./build/gputronic_gold loadfile "$LOG/steps-item1.txt" 110
for tag in 04 05; do
    hold_env
    launch_gpu "fw-$tag" full 15 ./build/gputronic_gold fullwidth 15
done
hold_env
export GPUTRONIC_TARGET_Z=1.5
export GPUTRONIC_ICLAMP_SMS=15
launch_gpu cl-03 load 110 ./build/gputronic_gold loadfile "$LOG/steps-item1.txt" 110
for tag in 06 07 08; do
    hold_env
    launch_gpu "fw-$tag" full 15 ./build/gputronic_gold fullwidth 15
done
echo "ITEM3_GRID_DONE" >> "$STATUS"
say "ITEM3_GRID_DONE"

# One mapped atomic per pass, two fullwidth repeats.
for tag in a b; do
    hold_env
    unset GPUTRONIC_PULSE_LOCAL
    launch_gpu "fw-localoff-$tag" full 15 ./build/gputronic_gold fullwidth 15
done

session_both() {
    local lows highs
    lows=$(awk '$2=="low"{c++} END{print c+0}' "$LOG/groups.tsv")
    highs=$(awk '$2=="high"{c++} END{print c+0}' "$LOG/groups.tsv")
    if [ "$lows" -ge 1 ] && [ "$highs" -ge 1 ]; then
        return 0
    fi
    return 1
}

run_until_both() {
    local prefix="$1"
    shift
    local cap=2
    if session_both; then cap=4; fi
    local i=1 saw_low=0 saw_high=0
    while [ "$i" -le "$cap" ]; do
        hold_env
        "$@"
        launch_gpu "${prefix}-$(printf '%02d' "$i")" full 15 ./build/gputronic_gold fullwidth 15
        if [ "$LAST_GROUP" = "low" ]; then saw_low=1; fi
        if [ "$LAST_GROUP" = "high" ]; then saw_high=1; fi
        if [ "$saw_low" -eq 1 ] && [ "$saw_high" -eq 1 ]; then
            break
        fi
        if [ "$cap" -eq 2 ] && [ "$i" -ge 2 ]; then
            break
        fi
        i=$((i + 1))
    done
}

run_until_both wc export GPUTRONIC_WORK_COUNTER=1
run_until_both acc export GPUTRONIC_SM_ACCUM=1

echo "SUITE_DONE $(date -Iseconds)" >> "$STATUS"
say "SUITE_DONE"
nvidia-smi > "$LOG/nvidia-smi-end.txt" 2>&1 || true
exit 0
