#!/usr/bin/env python3
"""Numbers for the sixth-pass note. Reads logs/2026-10-13_sixth-pass.

Experiment 1: pulse rate, GFLOP/s, Z against that plant's 84-SM hold, and
84/n, from the tally analyzer. Temperature, power, and clocks from dmon.
Experiment 2: closed-loop Z and the T6 window, from the 1 ms trace.
Experiment 4: std and lag of the tally rate against the pulse rate, and
PCIe throughput from the 84-SM dmon logs.

Every figure here is read from a measured log. Nothing in this script is a model.
"""
import csv
import math
import os
import re
import subprocess
import sys
from collections import defaultdict

LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "2026-10-13_sixth-pass")
OPS = 65536.0


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def corr(x, y):
    n = min(len(x), len(y))
    if n < 3:
        return float("nan")
    x, y = x[:n], y[:n]
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def dmon(name):
    path = os.path.join(LOG, name + ".dmon.txt")
    rows = []
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 16:
                continue
            try:
                rows.append(tuple(float(p[i]) for i in (1, 2, 4, 10, 11, 12, 13, 14, 15)))
            except ValueError:
                continue
    # pwr, gtemp, sm%, mclk, pclk, pviol, tviol, rx, tx
    use = [r for r in rows if r[0] >= 50 or r[2] >= 40]
    if len(use) < 3:
        use = rows[2:] or rows
    if not use:
        return None
    return {
        "n": len(use),
        "temp_mean": mean([r[1] for r in use]),
        "temp_max": max(r[1] for r in use),
        "pwr_mean": mean([r[0] for r in use]),
        "pwr_max": max(r[0] for r in use),
        "pclk_mean": mean([r[4] for r in use]),
        "pclk_min": min(r[4] for r in use),
        "mclk_mean": mean([r[3] for r in use]),
        "pviol_max": max(r[5] for r in use),
        "tviol_max": max(r[6] for r in use),
        "rx": mean([r[7] for r in use]),
        "tx": mean([r[8] for r in use]),
    }


def analysis(name):
    path = os.path.join(LOG, name + "_analysis.txt")
    if not os.path.isfile(path):
        return None
    text = open(path).read()
    pulses = re.search(r"\[pulses\] (\d+) pulses over ([0-9.]+) s, rate (\d+) /s", text)
    passes = []
    for m in re.finditer(r"^  (\d+) (\d+) (\d+) (\d+) (\d+)\s*$", text, re.M):
        passes.append(tuple(int(m.group(i)) for i in range(1, 6)))
    ratio = re.search(r"\[100 ms\] bins \d+: ratio mean ([0-9.]+), drift ([0-9.]+)%, corr ([0-9.\-nan]+), best lag (\-?\d+)", text)
    lag1 = re.search(r"\[1 host bin\] bins \d+: ratio mean ([0-9.]+), drift ([0-9.]+)%, corr ([0-9.\-nan]+), best lag (\-?\d+)", text)
    lag20 = re.search(r"\[20 ms\] bins \d+: ratio mean ([0-9.]+), drift ([0-9.]+)%, corr ([0-9.\-nan]+), best lag (\-?\d+)", text)
    blocks = re.search(r"\[blocks\] reporting (\d+), with work (\d+), idle (\d+)", text)
    return {
        "rate": float(pulses.group(3)) if pulses else float("nan"),
        "span": float(pulses.group(2)) if pulses else float("nan"),
        "passes": passes,
        "ratio100": ratio.groups() if ratio else None,
        "lag1": lag1.groups() if lag1 else None,
        "lag20": lag20.groups() if lag20 else None,
        "work": int(blocks.group(2)) if blocks else None,
    }


def gflops(rate):
    return rate * OPS / 1e9


def exp1():
    print("== experiment 1 ==")
    rows = []
    for n in (4, 12, 24, 36, 42, 48, 56, 64, 72, 84):
        rows.append((n, "on", analysis(f"e1-n{n}"), dmon(f"e1-n{n}")))
    for n in (84, 48):
        rows.append((n, "off", analysis(f"e1-n{n}-off"), dmon(f"e1-n{n}-off")))
    base = {plant: next(a["rate"] for n, p, a, _d in rows if n == 84 and p == plant and a)
            for plant in ("on", "off")}
    print(f"{'n':>4} {'sw':>3} {'pulse/s':>12} {'GFLOP/s':>8} {'Z':>7} {'84/n':>7} "
          f"{'GPC means':<42} {'CV%':>6} {'T':>5} {'W':>7} {'MHz':>6} {'rx':>6} {'tx':>6}")
    for n, plant, a, d in rows:
        if not a or not a["rate"]:
            print(f"{n:4d} {plant:>3} missing")
            continue
        z = base[plant] / a["rate"] if a["rate"] else float("nan")
        means = [p[2] for p in a["passes"]]
        cv = 100.0 * std(means) / mean(means) if means else float("nan")
        gpc = " ".join(f"{p[0]}:{p[2]/1e6:.2f}M" for p in a["passes"])
        temp = f"{d['temp_mean']:.0f}" if d else "-"
        pwr = f"{d['pwr_mean']:.1f}" if d else "-"
        clk = f"{d['pclk_mean']:.0f}" if d else "-"
        rx = f"{d['rx']:.0f}" if d else "-"
        tx = f"{d['tx']:.0f}" if d else "-"
        print(f"{n:4d} {plant:>3} {a['rate']:12.0f} {gflops(a['rate']):8.2f} {z:7.3f} {84/n:7.3f} "
              f"{gpc:<42} {cv:6.1f} {temp:>5} {pwr:>7} {clk:>6} {rx:>6} {tx:>6}")
        if d:
            print(f"     dmon n={d['n']} temp_max={d['temp_max']:.0f} pwr_max={d['pwr_max']:.1f} "
                  f"pclk_min={d['pclk_min']:.0f} mclk={d['mclk_mean']:.0f} "
                  f"pviol={d['pviol_max']:.0f} tviol={d['tviol_max']:.0f} "
                  f"100ms={a['ratio100']} work={a['work']}")


def open_trace(path):
    if path.endswith(".zst"):
        proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, text=True)
        try:
            yield from csv.DictReader(proc.stdout)
        finally:
            proc.stdout.close()
            proc.wait()
    else:
        with open(path, newline="") as f:
            yield from csv.DictReader(f)


def series_stats(name, skip_ms, end_ms=None):
    """Pulse rate and summed tally rate at the host bin and at 20 ms."""
    path = os.path.join(LOG, name + ".csv.zst")
    if not os.path.isfile(path):
        print(f"{name}: no tally")
        return
    bins = {}
    for row in open_trace(path):
        t = float(row["t_ms"])
        if t < skip_ms:
            continue
        if end_ms is not None and t >= end_ms:
            continue
        slot = bins.setdefault(t, [0.0, None])
        slot[0] += float(row["d_ops_cuda"])
        if slot[1] is None:
            slot[1] = float(row["bin_pulses"])
    keys = sorted(bins)
    if len(keys) < 10:
        print(f"{name}: only {len(keys)} bins")
        return
    dts = [b - a for a, b in zip(keys, keys[1:]) if b > a]
    dt = mean(dts) if dts else 1.0
    pulse = [bins[t][1] / dt for t in keys]
    tally = [bins[t][0] / OPS / dt for t in keys]
    # 20 ms buckets on the host clock.
    bp, bt, width = defaultdict(float), defaultdict(float), defaultdict(float)
    prev = None
    for t in keys:
        b = int(t // 20.0)
        span = (t - prev) if prev is not None and t > prev else dt
        bp[b] += bins[t][1]
        bt[b] += bins[t][0] / OPS
        width[b] += span
        prev = t
    order = sorted(bp)
    p20 = [bp[b] / width[b] for b in order if width[b] > 0]
    t20 = [bt[b] / width[b] for b in order if width[b] > 0]
    best, best_c = 0, corr(tally, pulse)
    for lag in range(-8, 9):
        if lag >= 0:
            c = corr(tally[lag:], pulse[:len(pulse) - lag] if lag else pulse)
        else:
            c = corr(tally[:len(tally) + lag], pulse[-lag:])
        if c > best_c:
            best, best_c = lag, c
    def line(label, xs):
        m = mean(xs)
        s = std(xs)
        print(f"  {label}: n={len(xs)} mean={m:.0f}/s std={s:.0f} cv={100*s/m if m else float('nan'):.2f}%")
    print(f"{name}: bins={len(keys)} median-ish dt={dt:.3f} ms  span={keys[-1]-keys[0]:.0f} ms")
    line("pulse 1 bin", pulse)
    line("tally 1 bin", tally)
    line("pulse 20 ms", p20)
    line("tally 20 ms", t20)
    print(f"  lag of tally vs pulses: {best} bins ({best * dt:.2f} ms) corr {best_c:.3f}")


def exp2():
    print("== experiment 2 ==")
    for name in ("e2-closed", "e2-t6", "e1-n84", "e1-n84-off"):
        path = os.path.join(LOG, name + ".txt")
        if not os.path.isfile(path):
            continue
        print(f"-- {name} calibration --")
        for line in open(path):
            if "[CAL]" in line or "[CTRL]" in line or "[PULSE]" in line or "[LAW]" in line:
                print(line.rstrip())
    path = os.path.join(LOG, "e2-closed.trace.csv.zst")
    if os.path.isfile(path):
        zs, ns, emas, wins = [], [], [], []
        for row in open_trace(path):
            t = float(row["t_ms"])
            if t < 10000 or t > 55000:
                continue
            zs.append(float(row["z_hat"]))
            ns.append(int(row["active_sms"]))
            emas.append(float(row["rate_ema"]))
            wins.append(float(row["win_pulses"]))
        changes = sum(1 for a, b in zip(ns, ns[1:]) if a != b)
        print(f"e2-closed settled 10..55 s: n={len(zs)} z mean {mean(zs):.3f} std {std(zs):.4f} "
              f"min {min(zs):.3f} max {max(zs):.3f}")
        print(f"  active_sms mean {mean(ns):.2f} std {std(ns):.2f} min {min(ns)} max {max(ns)} changes {changes}")
        print(f"  rate_ema mean {mean(emas):.0f} std {std(emas):.0f} cv {100*std(emas)/mean(emas):.2f}%")
        print(f"  win_pulses mean {mean(wins):.0f} std {std(wins):.0f}  (8 ms window)")
        near = sum(1 for z in zs if abs(z - 1.5) <= 0.08)
        print(f"  samples inside the 0.08 deadband of 1.5: {near}/{len(zs)}")
    path = os.path.join(LOG, "e2-t6.trace.csv.zst")
    if os.path.isfile(path):
        before, during = [], []
        for row in open_trace(path):
            t = float(row["t_ms"])
            rec = (float(row["z_hat"]), int(row["active_sms"]), float(row["win_pulses"]),
                   float(row["scale"]), float(row["rate_ema"]))
            if 10000 <= t < 19000:
                before.append(rec)
            elif 25000 <= t < 55000:
                during.append(rec)
        def side(label, rows):
            if not rows:
                print(f"  {label}: empty")
                return
            print(f"  {label}: n={len(rows)} z {mean([r[0] for r in rows]):.3f} "
                  f"std {std([r[0] for r in rows]):.4f} sms {mean([r[1] for r in rows]):.1f} "
                  f"win_pulses {mean([r[2] for r in rows]):.0f} scale {mean([r[3] for r in rows]):.3f} "
                  f"rate_ema {mean([r[4] for r in rows]):.0f}")
        print("e2-t6")
        side("before 10..19 s", before)
        side("load 25..55 s", during)


def exp4():
    print("== experiment 4 ==")
    for name in ("e1-n84", "e1-n48", "e1-n84-off", "e1-n48-off"):
        series_stats(name, 1000)
        a = analysis(name)
        if a:
            print(f"  analyzer 1 bin {a['lag1']}")
            print(f"  analyzer 20 ms {a['lag20']}")
            print(f"  analyzer 100 ms {a['ratio100']}")
        d = dmon(name)
        if d:
            print(f"  pcie rx {d['rx']:.1f} MB/s  tx {d['tx']:.1f} MB/s  "
                  f"pwr {d['pwr_mean']:.1f} W  {d['pclk_mean']:.0f} MHz")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        exp1()
    if which in ("all", "2"):
        exp2()
    if which in ("all", "4"):
        exp4()
    return 0


if __name__ == "__main__":
    sys.exit(main())
