#!/usr/bin/env python3
"""Command latency from a sixth-pass step tally.

The host log prints [STEP] t_ms=... active_sms -> N. The tally CSV has one
row per SM per host sample. Blocks 42 and above are the ones a step to 42
turns off, and a step back to 84 turns on. Latency is the host-sample time
from that print to the new tally level.

Resolution is the spacing of the host samples, plus one cache period.
"""
import csv
import os
import re
import subprocess
import sys

LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "2026-10-13_sixth-pass")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def load(name):
    txt = open(os.path.join(LOG, name + ".txt")).read()
    steps = [(float(m.group(1)), int(m.group(2)))
             for m in re.finditer(r"\[STEP\] t_ms=([0-9.]+) active_sms -> (\d+)", txt)]
    path = os.path.join(LOG, name + ".csv.zst")
    proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, text=True)
    bins = {}
    try:
        for row in csv.DictReader(proc.stdout):
            t = float(row["t_ms"])
            sm = int(row["sm"])
            slot = bins.setdefault(t, {"hi": 0.0, "lo": 0.0, "act": int(row["active_sms"])})
            d = float(row["d_ops_cuda"])
            if sm >= 42:
                slot["hi"] += d
            else:
                slot["lo"] += d
    finally:
        proc.stdout.close()
        proc.wait()
    keys = sorted(bins)
    return steps, keys, bins


def first_stable(keys, bins, field, t0, pred, need=3):
    run = 0
    for t in keys:
        if t < t0:
            continue
        if pred(bins[t][field]):
            run += 1
            if run >= need:
                return t
        else:
            run = 0
    return None


def report(name):
    steps, keys, bins = load(name)
    if len(keys) < 2:
        print(f"{name}: no bins")
        return
    dts = sorted(b - a for a, b in zip(keys, keys[1:]) if 0 < b - a < 5)
    dt = dts[len(dts) // 2] if dts else float("nan")
    print(f"{name}: samples {len(keys)} median spacing {dt:.3f} ms  steps {steps}")
    for t_cmd, n_cmd in steps:
        pre = [bins[t]["hi"] for t in keys if t_cmd - 300 <= t <= t_cmd - 20]
        late = [bins[t]["hi"] for t in keys if t >= keys[-1] - 300]
        pre_m = mean(pre)
        # Host column: when the sampled active_sms first matches the command.
        seen = next((t for t in keys if t >= t_cmd and bins[t]["act"] == n_cmd), None)
        if n_cmd <= 42:
            level = 0.1 * pre_m if pre_m == pre_m else 0.0
            hit = first_stable(keys, bins, "hi", t_cmd, lambda v, level=level: v < level)
            kind = "down"
        else:
            post = [bins[t]["hi"] for t in keys if t_cmd + 50 <= t <= t_cmd + 800]
            post_m = mean(post) if post else mean(late)
            hit = first_stable(keys, bins, "hi", t_cmd, lambda v, post_m=post_m: v > 0.5 * post_m)
            kind = "up"
        host = (seen - t_cmd) if seen is not None else float("nan")
        work = (hit - t_cmd) if hit is not None else float("nan")
        print(f"  {kind} -> {n_cmd} at {t_cmd:.3f} ms  host column {host:.3f} ms  "
              f"tally {work:.3f} ms  pre_hi {pre_m:.0f}")


def main():
    names = sys.argv[1:] or ["e3-250us", "e3-1000us", "e3-4000us"]
    for name in names:
        report(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
