#!/usr/bin/env python3
"""Summarize the seventh-pass logs. Prints measured numbers only."""
import csv
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(ROOT, "logs", "2026-10-13_seventh-pass")


def read_text(name):
    path = os.path.join(LOG, name + ".txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


def load_trace(name):
    path = os.path.join(LOG, name + ".trace.csv.zst")
    if not os.path.exists(path):
        return None
    proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, text=True)
    try:
        return list(csv.DictReader(proc.stdout))
    finally:
        proc.stdout.close()
        proc.wait()


def fmean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def stats(xs):
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def window(rows, t0, t1):
    out = []
    for row in rows:
        t = float(row["t_ms"])
        if t0 <= t < t1:
            out.append(row)
    return out


def fmt_stats(s, digits):
    if s is None:
        return "none"
    return "n=%d mean=%.*f std=%.*f min=%.*f max=%.*f" % (
        s["n"], digits, s["mean"], digits, s["std"], digits, s["min"], digits, s["max"])


def pulse_rate(rows):
    return fmean([float(r["win_pulses"]) for r in rows]) / 0.008


def report_t6(name):
    print(f"\n== {name} ==")
    text = read_text(name)
    if text is None:
        print("MISSING")
        return
    for pat in (r"\[LAW\].*", r"\[CAL\].*", r"\[PULSE\].*", r"\[CTRL\] cached.*", r"\[IDLE\].*", r"\[SNS\].*"):
        found = re.findall(pat, text)
        for line in found[:8]:
            print(line)
    rows = load_trace(name)
    if not rows:
        print("no trace")
        return
    print(f"trace rows {len(rows)} t {float(rows[0]['t_ms']):.1f}..{float(rows[-1]['t_ms']):.1f}")
    bins = [float(rows[i + 1]["t_ms"]) - float(rows[i]["t_ms"])
            for i in range(min(len(rows) - 1, 500))
            if 0 < float(rows[i + 1]["t_ms"]) - float(rows[i]["t_ms"]) < 50]
    if bins:
        bins.sort()
        print(f"median bin {bins[len(bins) // 2]:.3f} ms")
    for label, t0, t1 in (("before 10-19s", 10000, 19000),
                          ("held 25-55s", 25000, 55000),
                          ("first 1s of load 20-21s", 20000, 21000),
                          ("first 2s of load 20-22s", 20000, 22000)):
        part = window(rows, t0, t1)
        z = stats([float(r["z_hat"]) for r in part])
        sms = stats([float(r["active_sms"]) for r in part])
        integ = stats([float(r["integral"]) for r in part])
        rate = pulse_rate(part) if part else float("nan")
        scale = fmean([float(r["scale"]) for r in part]) if part else float("nan")
        print(f"{label}: scale {scale:.3f}")
        print(f"  Z {fmt_stats(z, 4)}")
        print(f"  SMs {fmt_stats(sms, 2)}")
        print(f"  integral {fmt_stats(integ, 4)}")
        print(f"  pulse rate {rate:.0f} /s  (mean win_pulses / 0.008)")
    before = window(rows, 10000, 19000)
    held = window(rows, 25000, 55000)
    first = window(rows, 20000, 21000)
    rb = pulse_rate(before) if before else float("nan")
    rh = pulse_rate(held) if held else float("nan")
    rf = pulse_rate(first) if first else float("nan")
    print(f"held/before {rh / rb:.4f}  first1s/before {rf / rb:.4f}  expected 0.8333")
    after = [(float(r["t_ms"]), float(r["z_hat"]), float(r["active_sms"]))
             for r in rows if float(r["t_ms"]) >= 20000]
    def inside(z):
        return abs(z - 1.5) <= 0.08
    left = any(not inside(z) for t, z, _s in after if t < 25000)
    first_in = next((t for t, z, _s in after if inside(z)), None)
    sustained = None
    for i, (t, z, _s) in enumerate(after):
        if after[-1][0] < t + 200:
            break
        ok = True
        for t2, z2, _s2 in after[i:]:
            if t2 >= t + 200:
                break
            if not inside(z2):
                ok = False
                break
        if ok:
            sustained = t
            break
    print(f"left the 0.08 band in the 5 s after the step: {left}")
    if first_in is None:
        print("first return: never")
    else:
        print(f"first return at {first_in - 20000:.1f} ms after the step (t={first_in:.1f})")
    if sustained is None:
        print("sustained 200 ms inside the band: never")
    else:
        print(f"sustained 200 ms from {sustained - 20000:.1f} ms after the step (t={sustained:.1f})")
    # Where the count sat while the load was held: share of samples at each count.
    counts = {}
    for r in held:
        k = int(float(r["active_sms"]))
        counts[k] = counts.get(k, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
    print("held SM histogram " + " ".join(f"{k}:{v}" for k, v in top))


def dmon_rows(name):
    path = os.path.join(LOG, name + ".dmon.txt")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 16:
                continue
            try:
                rows.append({
                    "pwr": float(p[1]), "temp": float(p[2]), "sm": float(p[4]),
                    "mclk": float(p[10]), "pclk": float(p[11]),
                    "pviol": float(p[12]), "tviol": float(p[13]),
                    "rx": float(p[14]), "tx": float(p[15]),
                })
            except ValueError:
                continue
    return rows


def t0_of(name):
    path = os.path.join(LOG, name + ".t0")
    if not os.path.exists(path):
        return None
    raw = open(path).read().strip()
    return int(raw) if raw else 0


def dmon_slice(name, a, b):
    rows = dmon_rows(name)
    t0 = t0_of(name)
    if t0 is None:
        part = rows[8:]
        note = "t0 missing, skipped first 8 samples"
    else:
        part = rows[t0 + a:t0 + b]
        note = f"t0={t0} samples {t0 + a}..{t0 + b} of {len(rows)}"
    return part, note


def fmt_dmon(part, note):
    if not part:
        return f"no dmon samples ({note})"
    def col(key):
        return [r[key] for r in part]
    pwr, temp, pclk = col("pwr"), col("temp"), col("pclk")
    return ("%s n=%d pwr mean %.1f min %.1f max %.1f  temp %.0f..%.0f  "
            "pclk mean %.0f min %.0f max %.0f  sm%% mean %.0f  "
            "pviol max %.0f tviol max %.0f" % (
                note, len(part), fmean(pwr), min(pwr), max(pwr),
                min(temp), max(temp), fmean(pclk), min(pclk), max(pclk),
                fmean(col("sm")), max(col("pviol")), max(col("tviol"))))


def rate_ref(name):
    text = read_text(name) or ""
    m = re.search(r"\[CAL\] rate_ref=([0-9.]+) pulses/s", text)
    return float(m.group(1)) if m else float("nan")


def report_hold(name):
    print(f"\n== {name} ==")
    text = read_text(name)
    if text is None:
        print("MISSING")
        return
    for line in text.splitlines():
        if line.startswith("[PULSE]") or line.startswith("[CTRL] cached") or line.startswith("[IDLE]") \
                or line.startswith("[CAL] rate_ref") or line.startswith("[CAL] cuda target") \
                or line.startswith("[CAL] integral") or line.startswith("[LAW]"):
            print(line)
    rows = load_trace(name)
    if not rows:
        print("no trace")
    else:
        part = window(rows, 3000, 28000)
        z = stats([float(r["z_hat"]) for r in part])
        sms = stats([float(r["active_sms"]) for r in part])
        rate = pulse_rate(part) if part else float("nan")
        ref = rate_ref(name)
        zwall = ref / rate if rate else float("nan")
        print(f"settled 3-28s Z {fmt_stats(z, 4)}")
        print(f"settled SMs {fmt_stats(sms, 2)}")
        print(f"pulse rate {rate:.0f} /s  Z_wall=rate_ref/rate {zwall:.3f}  rate_ref {ref:.0f}")
    part, note = dmon_slice(name, 3, 28)
    print(fmt_dmon(part, note))


def report_checks():
    print("\n== checks ==")
    for name in ("gold-check", "transfer-check", "episode-check"):
        text = read_text(name)
        if text is None:
            print(f"{name}: MISSING")
            continue
        print(f"-- {name}")
        for line in text.splitlines():
            if line.startswith("[CHECK]") or "GATE:" in line or line.startswith("[SNS]") \
                    or line.startswith("[LAW] target_z"):
                print(line)
        bad = []
        if "[PULSE] local count" in text:
            bad.append("pulse")
        if "including work_scale_milli" in text:
            bad.append("cache")
        if "[IDLE] quiet on" in text:
            bad.append("quiet")
        print("switches printed: " + (", ".join(bad) if bad else "none"))


def mean(xs):
    return fmean(xs)


def report_latency(name):
    print(f"\n== {name} ==")
    text = read_text(name)
    if text is None:
        print("MISSING")
        return
    steps = [(float(m.group(1)), int(m.group(2)))
             for m in re.finditer(r"\[STEP\] t_ms=([0-9.]+) active_sms -> (\d+)", text)]
    path = os.path.join(LOG, name + ".csv.zst")
    if not os.path.exists(path):
        print("no tally")
        return
    proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, text=True)
    bins = {}
    try:
        for row in csv.DictReader(proc.stdout):
            t = float(row["t_ms"])
            sm = int(row["sm"])
            slot = bins.setdefault(t, {"hi": 0.0, "act": int(float(row["active_sms"]))})
            if sm >= 42:
                slot["hi"] += float(row["d_ops_cuda"])
    finally:
        proc.stdout.close()
        proc.wait()
    keys = sorted(bins)
    if len(keys) < 2:
        print("no bins")
        return
    dts = sorted(b - a for a, b in zip(keys, keys[1:]) if 0 < b - a < 5)
    dt = dts[len(dts) // 2] if dts else float("nan")
    print(f"samples {len(keys)} median spacing {dt:.3f} ms  steps {steps}")

    def first_stable(field_pred, t0, need=3):
        run = 0
        for t in keys:
            if t < t0:
                continue
            if field_pred(bins[t]["hi"]):
                run += 1
                if run >= need:
                    return t
            else:
                run = 0
        return None

    for t_cmd, n_cmd in steps:
        pre = [bins[t]["hi"] for t in keys if t_cmd - 300 <= t <= t_cmd - 20]
        pre_m = mean(pre)
        seen = next((t for t in keys if t >= t_cmd and bins[t]["act"] == n_cmd), None)
        if n_cmd <= 42:
            level = 0.1 * pre_m if pre_m == pre_m else 0.0
            hit = first_stable(lambda v, level=level: v < level, t_cmd)
            kind = "down"
        else:
            post = [bins[t]["hi"] for t in keys if t_cmd + 50 <= t <= t_cmd + 800]
            post_m = mean(post) if post else 0.0
            hit = first_stable(lambda v, post_m=post_m: post_m > 0 and v > 0.5 * post_m, t_cmd)
            kind = "up"
        host = (seen - t_cmd) if seen is not None else float("nan")
        work = (hit - t_cmd) if hit is not None else float("nan")
        print(f"  {kind} -> {n_cmd} at {t_cmd:.3f} ms  host column {host:.3f} ms  "
              f"tally {work:.3f} ms  pre_hi {pre_m:.0f}")


def report_smids():
    print("\n== smid ==")
    for name in sorted(os.listdir(LOG)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(LOG, name)
        with open(path) as f:
            for line in f:
                if line.startswith("[SMID] left out:") or line.startswith("[SMID] recorded") or line.startswith("[GPU] "):
                    print(f"{name[:-4]}: {line.rstrip()}")


def main():
    global LOG
    ceiling = 84
    if len(sys.argv) >= 2:
        LOG = os.path.join(ROOT, "logs", sys.argv[1])
    if len(sys.argv) >= 3:
        ceiling = int(sys.argv[2])
    report_checks()
    report_smids()
    report_t6("t6-off")
    report_t6("t6-on")
    for n in (4, 12, 24, 36, 48, 56, 72, ceiling):
        report_hold(f"q-n{n}")
    for n in (4, 36, ceiling):
        report_hold(f"q-n{n}-spin")
    report_latency("e3-quiet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
