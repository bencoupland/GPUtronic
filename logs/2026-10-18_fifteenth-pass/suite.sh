#!/bin/bash
# Fifteenth pass. One GPU job. Stop a run at 83 C. Two stops end the suite.
# Open-loop schedule sweeps, then the ceiling hold, then the ramps, then
# one closed-loop confirmation pair (ABAB only if a base bracket hit the
# 49-million per-SM plant).
set -u
ROOT="/tmp/gputronic-fifteenth"
LOG="$ROOT/logs/2026-10-18_fifteenth-pass"
STATUS="$LOG/status.txt"
PICK="/tmp/fifteenth-pick.py"
GEN="/tmp/fifteenth-gen.py"
mkdir -p "$LOG" "$ROOT/results" "$LOG/sched"
cd "$ROOT"
cp -f /tmp/fifteenth-suite.sh "$LOG/suite.sh"
cp -f "$GEN" "$LOG/schedules.py"
cp -f "$PICK" "$LOG/pick.py"
echo $$ > "$LOG/suite.pid"
TEMP_STOPS=0
DISP_ALL=""
cleanup() {
    if [ -n "${DISP_ALL:-}" ]; then
        kill "$DISP_ALL" 2>/dev/null || true
    fi
}
trap cleanup EXIT

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
        GPUTRONIC_DEADBAND GPUTRONIC_SENSOR_RULE GPUTRONIC_RESERVE_SMS \
        GPUTRONIC_OPEN_SCHED GPUTRONIC_ACTIVE_PASSES GPUTRONIC_ACTIVE_SMS
}

die_c() {
    timeout 8 nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
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

done_ok() {
    grep -q "^RUN_OK $1$" "$STATUS" 2>/dev/null
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
          "$ROOT/results/gputronic_gold_noise.csv" \
          "$ROOT/results/gputronic_gold_smaccum.csv"
}

move_products() {
    local name="$1" kind="$2"
    if [ "$kind" = "load" ]; then
        [ -f "$ROOT/results/gputronic_gold_load.csv" ] && mv "$ROOT/results/gputronic_gold_load.csv" "$LOG/$name-tick.csv"
        [ -f "$ROOT/results/gputronic_gold_loadfile_10ms.csv" ] && mv "$ROOT/results/gputronic_gold_loadfile_10ms.csv" "$LOG/$name-1ms.csv"
    elif [ "$kind" = "noise" ]; then
        [ -f "$ROOT/results/gputronic_gold_fullwidth.csv" ] && mv "$ROOT/results/gputronic_gold_fullwidth.csv" "$LOG/$name-tick.csv"
        [ -f "$ROOT/results/gputronic_gold_noise.csv" ] && mv "$ROOT/results/gputronic_gold_noise.csv" "$LOG/$name-1ms.csv"
    else
        [ -f "$ROOT/results/gputronic_gold_fullwidth.csv" ] && mv "$ROOT/results/gputronic_gold_fullwidth.csv" "$LOG/$name-tick.csv"
        [ -f "$ROOT/results/gputronic_gold_fullwidth_10ms.csv" ] && mv "$ROOT/results/gputronic_gold_fullwidth_10ms.csv" "$LOG/$name-1ms.csv"
    fi
    if [ -f "$ROOT/results/gputronic_gold_smaccum.csv" ]; then
        mv "$ROOT/results/gputronic_gold_smaccum.csv" "$LOG/$name-smaccum.csv"
    fi
}

cut_tick() {
    local name="$1"
    [ -s "$LOG/$name-tick.csv" ] || return 0
    python3 "$PICK" cut-tick "$LOG/$name-tick.csv" "$LOG/$name-tick-cut.csv.gz" "$LOG/cut-manifest.txt" "$name" \
        >> "$LOG/suite.log" 2>&1 || fail "$name tick cut"
    rm -f "$LOG/$name-tick.csv"
}

finish_accum() {
    local name="$1"
    [ "${GPUTRONIC_SM_ACCUM:-}" = "1" ] || return 0
    [ -s "$LOG/$name-smaccum.csv" ] || fail "$name no smaccum"
    python3 "$PICK" accum "$LOG/$name-smaccum.csv" "$LOG/$name-smaccum-summary.txt" \
        >> "$LOG/suite.log" 2>&1 || fail "$name accum summary"
    rm -f "$LOG/$name-smaccum.csv"
}

# kind is full, noise, or load. The binary command is the rest.
launch_gpu() {
    local name="$1" kind="$2" sec="$3"
    shift 3
    if done_ok "$name"; then
        say "skip $name"
        return 0
    fi
    wait_cool
    say "RUN $name kind=$kind sec=$sec"
    echo "RUN $name" >> "$STATUS"
    clear_results
    snap_proc "$name" start
    local dmon_n=$((sec + 40))
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
        env | grep '^GPUTRONIC_' | sort || true
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
    if [ "$stopped" -ne 0 ]; then
        echo "RUN_TEMP $name" >> "$STATUS"
        say "RUN_TEMP $name stops=$TEMP_STOPS"
        if [ "$TEMP_STOPS" -ge 2 ]; then
            echo "SUITE_DONE temp" >> "$STATUS"
            say "SUITE_DONE temp"
            exit 0
        fi
        return 2
    fi
    if [ "$rc" -ne 0 ]; then
        echo "RUN_FAIL $name exit $rc" >> "$STATUS"
        fail "$name exit $rc"
    fi
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
    cut_tick "$name"
    finish_accum "$name"
    echo "RUN_OK $name" >> "$STATUS"
    say "RUN_OK $name"
    return 0
}

apply_env() {
    local acc="$1" rounds="$2" threads="$3" extra="$4" sched="$5" kind="$6"
    unset_switches
    export GPUTRONIC_PEAK_ACC="$acc"
    export GPUTRONIC_PEAK_ROUNDS="$rounds"
    export GPUTRONIC_THREADS="$threads"
    export GPUTRONIC_PULSE_LOCAL=1
    export GPUTRONIC_CTRL_CACHE=1
    export GPUTRONIC_TRACE_BIN_US=1000
    export GPUTRONIC_WORK_LOAD=1
    if [ "$kind" = "full" ] && [ "$sched" != "-" ]; then
        export GPUTRONIC_OPEN_SCHED="$sched"
    fi
    if [ "$extra" != "-" ]; then
        local kv
        for kv in $extra; do
            export "$kv"
        done
    fi
}

run_line() {
    local name="$1" kind="$2" sec="$3" acc="$4" rounds="$5" threads="$6" open_sms="$7" extra="$8" sched="$9"
    apply_env "$acc" "$rounds" "$threads" "$extra" "$sched" "$kind"
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        if [ "$kind" = "noise" ]; then
            launch_gpu "$name" noise "$sec" ./build/gputronic_gold noise "$open_sms" "$sec" && return 0
        elif [ "$kind" = "load" ]; then
            launch_gpu "$name" load "$sec" ./build/gputronic_gold loadfile "$sched" "$sec" && return 0
        else
            launch_gpu "$name" full "$sec" ./build/gputronic_gold fullwidth "$sec" && return 0
        fi
        local rc=$?
        if [ "$rc" -eq 2 ] && [ "$attempt" -lt 2 ]; then
            say "retry $name after temp"
            for ext in 1ms.csv tick.csv tick-cut.csv.gz txt dmon flags display; do
                [ -f "$LOG/$name.$ext" ] && mv "$LOG/$name.$ext" "$LOG/$name-hot.$ext"
                [ -f "$LOG/$name-$ext" ] && mv "$LOG/$name-$ext" "$LOG/$name-hot-$ext"
            done
            wait_cool
            continue
        fi
        return "$rc"
    done
}

run_phases() {
    local want=" $1 "
    while IFS=$(printf '\t') read -r name phase kind sec acc rounds threads open_sms extra sched; do
        [ "$name" = "name" ] && continue
        case "$want" in
            *" $phase "*) ;;
            *) continue ;;
        esac
        run_line "$name" "$kind" "$sec" "$acc" "$rounds" "$threads" "$open_sms" "$extra" "$sched" || fail "$name"
    done < "$LOG/launches.tsv"
}

append_point() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$LOG/points.tsv"
}

gpu_idle() {
    local n
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)
    [ "${n:-0}" -eq 0 ]
}

echo "START $(date -Iseconds) TZ=$(date +%Z)" >> "$STATUS"
say "START $(date -Iseconds)"
nvidia-smi > "$LOG/nvidia-smi-start.txt" 2>&1 || true
git -C "$ROOT" rev-parse HEAD > "$LOG/git-rev.txt"
git -C "$ROOT" log -1 --oneline >> "$LOG/git-rev.txt"
if [ ! -f "$LOG/launches.tsv" ]; then
    python3 "$GEN" "$LOG" >> "$LOG/suite.log" 2>&1 || fail "schedule generator"
fi

if ! gpu_idle; then
    fail "GPU was not idle at start"
fi

nvidia-smi --query-gpu=timestamp,display_active,display_mode,pstate,utilization.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active \
    --format=csv -lms 1000 > "$LOG/display-all.csv" 2>&1 &
DISP_ALL=$!

if ! grep -q '^PHASE zero done$' "$STATUS" 2>/dev/null; then
    echo "PHASE zero" >> "$STATUS"
    snap_proc zero start
    nvidia-smi --query-gpu=timestamp,display_active,display_mode,pstate,utilization.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active \
        --format=csv -lms 1000 > "$LOG/zero.display" 2>&1 &
    zpid=$!
    nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks_event_reasons.active \
        --format=csv -l 1 > "$LOG/zero.flags" 2>&1 &
    fpid=$!
    sleep 30
    kill "$zpid" "$fpid" 2>/dev/null || true
    wait "$zpid" 2>/dev/null || true
    wait "$fpid" 2>/dev/null || true
    snap_proc zero end
    if nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -q gputronic; then
        fail "zero window saw a gputronic process"
    fi
    echo "PHASE zero done" >> "$STATUS"
fi

echo "PHASE sweep" >> "$STATUS"
run_phases "smoke"
python3 "$PICK" verify-smoke "$LOG" >> "$LOG/suite.log" 2>&1 || fail "schedule smoke"
echo "SMOKE_OK" >> "$STATUS"
run_phases "floor item2 threads chains sms grid"

echo "PHASE ceiling" >> "$STATUS"
python3 "$PICK" ceiling "$LOG" >> "$LOG/suite.log" 2>&1 || fail "ceiling picker"
# shellcheck disable=SC1091
. "$LOG/ceiling-choice.txt" 2>/dev/null || true
acc=$(grep '^acc=' "$LOG/ceiling-choice.txt" | cut -d= -f2)
rounds=$(grep '^rounds=' "$LOG/ceiling-choice.txt" | cut -d= -f2)
threads=$(grep '^threads=' "$LOG/ceiling-choice.txt" | cut -d= -f2)
say "ceiling acc=$acc rounds=$rounds threads=$threads"
for spec in "ceil-hold -" "ceil-accum GPUTRONIC_SM_ACCUM=1" "ceil-work GPUTRONIC_WORK_COUNTER=1"; do
    cname=${spec%% *}
    cextra=${spec#* }
    if ! grep -q "^${cname}" "$LOG/points.tsv"; then
        append_point "$cname" 0 30000 "$acc" "$rounds" "$threads" 82 1000 ceiling
    fi
    run_line "$cname" full 30 "$acc" "$rounds" "$threads" 82 "$cextra" "-" || fail "$cname"
done

echo "PHASE item4" >> "$STATUS"
run_phases "item4"

echo "PHASE confirm" >> "$STATUS"
python3 "$PICK" confirm "$LOG" >> "$LOG/suite.log" 2>&1 || fail "confirm picker"
mode=$(grep '^mode=' "$LOG/confirm-choice.txt" | cut -d= -f2)
cacc=$(grep '^acc=' "$LOG/confirm-choice.txt" | cut -d= -f2)
crounds=$(grep '^rounds=' "$LOG/confirm-choice.txt" | cut -d= -f2)
cthreads=$(grep '^threads=' "$LOG/confirm-choice.txt" | cut -d= -f2)
say "confirm mode=$mode acc=$cacc rounds=$crounds threads=$cthreads"
csched="$LOG/sched/confirm.sched"
cextra="GPUTRONIC_ICLAMP_SMS=15 GPUTRONIC_TARGET_Z=1.5"
add_confirm_points() {
    local name="$1" a="$2" r="$3" th="$4"
    if grep -q "^${name}" "$LOG/points.tsv"; then
        return 0
    fi
    append_point "$name" 0 15000 "$a" "$r" "$th" 82 1000 confirm-pre
    append_point "$name" 15000 45000 "$a" "$r" "$th" 82 1200 confirm-load
    append_point "$name" 45000 75000 "$a" "$r" "$th" 82 1000 confirm-return
}
if [ "$mode" = "high" ]; then
    names="conf-a1 conf-b1 conf-a2 conf-b2"
else
    names="conf-a conf-b"
fi
for cname in $names; do
    case "$cname" in
        conf-a*)
            add_confirm_points "$cname" "$cacc" "$crounds" "$cthreads"
            run_line "$cname" load 75 "$cacc" "$crounds" "$cthreads" 82 "$cextra" "$csched" || fail "$cname"
            ;;
        conf-b*)
            add_confirm_points "$cname" 32 512 512
            run_line "$cname" load 75 32 512 512 82 "$cextra" "$csched" || fail "$cname"
            rate=$(python3 - "$LOG/$cname.txt" << 'PY'
import re, sys
t = open(sys.argv[1], errors="replace").read()
m = re.search(r"\[CAL\] rate_ref=([0-9.]+)", t)
print(m.group(1) if m else "0")
PY
)
            say "confirm base $cname rate_ref=$rate"
            awk -v r="$rate" 'BEGIN{exit !(r+0 < 46000000)}' && {
                if [ ! -f "$LOG/$cname.retry" ]; then
                    say "retry $cname low rate_ref"
                    touch "$LOG/$cname.retry"
                    # the low launch stays; a second base is conf-b-retry
                    mv "$LOG/$cname.txt" "$LOG/$cname-low.txt" || true
                    mv "$LOG/$cname-1ms.csv" "$LOG/$cname-low-1ms.csv" || true
                    mv "$LOG/$cname-tick-cut.csv.gz" "$LOG/$cname-low-tick-cut.csv.gz" || true
                    # RUN_OK is already recorded. Clear it so run_line repeats.
                    grep -v "^RUN_OK $cname$" "$STATUS" > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"
                    run_line "$cname" load 75 32 512 512 82 "$cextra" "$csched" || fail "$cname retry"
                fi
            }
            ;;
    esac
done

kill "$DISP_ALL" 2>/dev/null || true
wait "$DISP_ALL" 2>/dev/null || true
snap_proc suite end
nvidia-smi > "$LOG/nvidia-smi-end.txt" 2>&1 || true
echo "SUITE_DONE $(date -Iseconds)" >> "$STATUS"
say "SUITE_DONE $(date -Iseconds)"
