#!/usr/bin/env bash
# T8: game co-run with GPUtronic Gold (Linux, Steam + Proton, MangoHud).
# Default game: Warhammer 40,000: Space Marine 2, Steam app id 2183900.
# The game has no built-in benchmark, so each run is a MANUAL fixed route,
# timed by MangoHud (autostart_log + log_duration). GPUtronic only controls
# its own kernel's SM count; it does not throttle the game.
#
# Usage: scripts/t8_game_corun.sh <condition> [run_index]
#   condition: alone | fullwidth | closed
# Env: APPID (default 2183900), DUR (MangoHud log seconds, default 120),
#      WARMUP (seconds before logging starts, default 30), LOGROOT (default logs/t8)
#
# Before the first run: in Steam, set the game's launch options to
#   mangohud %command%
# and fix game settings (preset, resolution, upscaler, frame generation off, vsync off).
set -euo pipefail
cond=${1:?condition: alone|fullwidth|closed}
run=${2:-1}
APPID=${APPID:-2183900}
DUR=${DUR:-120}
WARMUP=${WARMUP:-30}
LOGROOT=${LOGROOT:-logs/t8}
GOLD=${GOLD:-./build/gputronic_gold}
cd "$(dirname "$0")/.."
out="$LOGROOT/${cond}_run${run}"
mkdir -p "$out/mangohud"

echo "wall_start $(date +%s.%N)" > "$out/clock.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,pcie.link.gen.current \
  --format=csv -lms 100 > "$out/smi.csv" &
SMI=$!
nvidia-smi dmon -s t -d 1 > "$out/smi_pcie.txt" &
DMON=$!
GPID=""
cleanup() { [ -n "$GPID" ] && kill "$GPID" 2>/dev/null || true; kill $SMI $DMON 2>/dev/null || true; }
trap cleanup EXIT

gold_secs=$((WARMUP + DUR + 120))
case "$cond" in
  alone) ;;
  fullwidth)
    GPUTRONIC_TENSOR_SENSOR=0 timeout $((gold_secs + 30)) "$GOLD" fullwidth "$gold_secs" > "$out/gold.txt" 2>&1 &
    GPID=$! ;;
  closed)
    GPUTRONIC_TENSOR_SENSOR=0 timeout $((gold_secs + 30)) "$GOLD" run "$gold_secs" 1.5 > "$out/gold.txt" 2>&1 &
    GPID=$! ;;
  *) echo "unknown condition $cond" >&2; exit 1 ;;
esac
if [ -n "$GPID" ]; then
  # Gold calibrates BEFORE the game starts: wait for the calibration lines.
  for _ in $(seq 1 300); do
    grep -q "\[LOAD\] calibrated\|\[CAL\] integral clamp" "$out/gold.txt" 2>/dev/null && break
    kill -0 "$GPID" 2>/dev/null || { echo "Gold exited early; see $out/gold.txt" >&2; exit 1; }
    sleep 0.1
  done
  echo "gold_calibrated $(date +%s.%N)" >> "$out/clock.txt"
fi

# MangoHud options (README): output_folder, autostart_log, log_duration.
export MANGOHUD=1
export MANGOHUD_CONFIG="output_folder=$PWD/$out/mangohud,autostart_log=$WARMUP,log_duration=$DUR"
echo "game_launch $(date +%s.%N)" >> "$out/clock.txt"
steam -applaunch "$APPID"
echo "Play the fixed route now. Logging starts after ${WARMUP}s and lasts ${DUR}s."
sleep $((WARMUP + DUR + 10))
echo "wall_end $(date +%s.%N)" >> "$out/clock.txt"
[ -n "$GPID" ] && cp results/gputronic_gold_run.csv results/gputronic_gold_fullwidth*.csv "$out/" 2>/dev/null || true
echo "Close the game. Then: python3 scripts/analyze_mangohud.py $out/mangohud/*.csv"
