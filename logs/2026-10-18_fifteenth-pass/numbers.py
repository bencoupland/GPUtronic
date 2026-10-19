#!/usr/bin/env python3
"""Fifteenth-pass numbers. No numpy. Reads the log directory, writes the table."""
import csv
import gzip
import math
import os
import re
import statistics
import sys
from datetime import datetime

TARGET = 1.5
RATIO = 601000.0 / 580000.0


def med(xs):
    if not xs:
        return None
    return statistics.median(xs)


def pstdev(xs):
    if len(xs) < 2:
        return 0.0
    return statistics.pstdev(xs)


def rounds_of(base, milli):
    return (int(base) * int(milli) + 500) // 1000


def ops_of(rate, acc, rounds):
    return rate * acc * rounds * 64.0


def goertzel(x, freq, dt):
    w = 2.0 * math.pi * freq * dt
    cw = 2.0 * math.cos(w)
    s1 = 0.0
    s2 = 0.0
    for v in x:
        s0 = v + cw * s1 - s2
        s2 = s1
        s1 = s0
    return s1 * s1 + s2 * s2 - cw * s1 * s2


def ac_at(x, lag):
    if lag < 1 or lag >= len(x):
        return None
    denom = sum(v * v for v in x)
    if denom == 0.0:
        return 0.0
    s = 0.0
    for i in range(len(x) - lag):
        s += x[i] * x[i + lag]
    return s / denom


def line_stats(z, dt):
    """1 ms series. Downsample to 4 ms. Same scan as the fourteenth."""
    n = len(z)
    if n < 256:
        return None
    mu = sum(z) / n
    var = sum((v - mu) * (v - mu) for v in z) / n
    sd = math.sqrt(var)
    if sd < 1e-6:
        return None
    step = max(1, int(round(0.004 / dt)))
    x = [z[i] - mu for i in range(0, n, step)]
    dt2 = dt * step
    if len(x) < 64:
        return None
    band_p = 0.0
    line_p = 0.0
    best_p = -1.0
    best_f = None
    f = 0.5
    while f <= 60.0 + 1e-9:
        pwr = goertzel(x, f, dt2)
        band_p += pwr
        if 7.25 <= f <= 7.45:
            line_p += pwr
        if 2.0 <= f <= 20.0 and pwr > best_p:
            best_p = pwr
            best_f = f
        f += 0.25
    if best_f is not None:
        f = best_f - 0.4
        while f <= best_f + 0.4 + 1e-12:
            pwr = goertzel(x, f, dt2)
            if pwr > best_p:
                best_p = pwr
                best_f = f
            f += 0.02
    denom = sum(v * v for v in x)
    best_ac = -2.0
    best_lag = 0
    lag = int(round(0.050 / dt2))
    lag_hi = min(int(0.400 / dt2), len(x) // 2)
    while lag <= lag_hi:
        s = 0.0
        for i in range(len(x) - lag):
            s += x[i] * x[i + lag]
        a = s / denom if denom else 0.0
        if a > best_ac:
            best_ac = a
            best_lag = lag
        lag += 1
    ac_ms = best_lag * dt2 * 1000.0

    def share_at(period_ms):
        if not band_p or period_ms <= 0:
            return None
        f0 = 1000.0 / period_ms
        acc = 0.0
        f = f0 - 0.1
        while f <= f0 + 0.1 + 1e-12:
            if 0.5 <= f <= 60.0:
                acc += goertzel(x, f, dt2)
            f += 0.02
        return acc / band_p

    moved = abs(ac_ms - 136.0) > 4.0
    return {
        "sd": sd,
        "frac": (line_p / band_p) if band_p else 0.0,
        "frac_lag": share_at(ac_ms) if moved else None,
        "peak_ms": (1000.0 / best_f) if best_f else None,
        "ac_ms": ac_ms,
        "ac": best_ac,
        "ac128": ac_at(x, int(round(0.128 / dt2))),
        "ac136": ac_at(x, int(round(0.136 / dt2))),
        "ac144": ac_at(x, int(round(0.144 / dt2))),
    }


def stretch_sd(z, n=500):
    sds = []
    i = 0
    while i + n <= len(z):
        sds.append(pstdev(z[i:i + n]))
        i += n
    return med(sds)


def in_band(z, target):
    return round(abs(z - target), 5) <= 0.08


_FULL = {}


def load_full(path):
    if path in _FULL:
        return _FULL[path]
    t, scale, zraw, zhat, sms, rate, wall, integral = (
        [], [], [], [], [], [], [], []
    )
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_ms"]))
            scale.append(float(row["scale"]))
            zraw.append(float(row["z_raw"]))
            zhat.append(float(row["z_hat"]))
            sms.append(int(row["active_sms"]))
            rate.append(float(row["rate_ema"]))
            wall.append(float(row["wall_s"]))
            integral.append(float(row["integral"]) if row.get("integral") else 0.0)
    data = {
        "t": t, "scale": scale, "zraw": zraw, "zhat": zhat,
        "sms": sms, "rate": rate, "wall": wall, "integral": integral,
    }
    _FULL[path] = data
    return data


def load_trace(path, t0, t1):
    full = load_full(path)
    i0 = 0
    while i0 < len(full["t"]) and full["t"][i0] < t0:
        i0 += 1
    i1 = i0
    while i1 < len(full["t"]) and full["t"][i1] < t1:
        i1 += 1
    return {k: v[i0:i1] for k, v in full.items()}


def parse_txt(path):
    text = open(path, errors="replace").read() if os.path.isfile(path) else ""
    cal = re.search(r"\[CAL\] rate_ref=([0-9.]+)", text)
    stats = None
    for m in re.finditer(
        r"\[STATS\] loop_iters=(\d+) pi_ticks=(\d+) deadband_held=(\d+) "
        r"integral_updates=(\d+) integral_on_rail=(\d+) sm_moves=(\d+)",
        text,
    ):
        stats = m
    sched = "[SCHED]" in text
    target = "[TARGET]" in text
    rate_line = "[RATE]" in text
    return {
        "rate_ref": float(cal.group(1)) if cal else None,
        "rail": int(stats.group(5)) if stats else None,
        "sm_moves": int(stats.group(6)) if stats else None,
        "integral_updates": int(stats.group(4)) if stats else None,
        "sched": sched,
        "target": target,
        "rate_switch": rate_line,
        "text": text,
    }


def parse_time(s):
    s = s.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass
    return None


def num_unit(s):
    s = s.strip().split()[0].replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def load_series(path):
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts = None
            for k, v in row.items():
                if k and k.startswith("timestamp"):
                    ts = parse_time(v)
                    break
            if ts is None:
                continue
            item = {"t": ts}
            for k, v in row.items():
                if not k:
                    continue
                item[k.split()[0].split("[")[0].strip()] = v.strip()
            rows.append(item)
    return rows


def window_clocks(display, flags, w0, w1):
    dsel = [r for r in display if w0 <= r["t"] <= w1]
    fsel = [r for r in flags if w0 <= r["t"] <= w1]
    clocks = [num_unit(r.get("clocks.current.sm", r.get("clocks.sm", ""))) for r in dsel]
    clocks = [c for c in clocks if c is not None]
    powers = [num_unit(r.get("power.draw", "")) for r in dsel]
    powers = [p for p in powers if p is not None]
    temps = []
    for r in fsel:
        tv = num_unit(r.get("temperature.gpu", ""))
        if tv is not None:
            temps.append(tv)
    reasons = sorted({r.get("clocks_event_reasons.active", "") for r in dsel if r.get("clocks_event_reasons.active")})
    acts = sorted({r.get("display_active", "") for r in dsel if r.get("display_active")})
    mems = [num_unit(r.get("clocks.current.memory", r.get("clocks.mem", ""))) for r in dsel]
    mems = [m for m in mems if m is not None]
    return {
        "sm": med(clocks),
        "power": med(powers),
        "temp": max(temps) if temps else None,
        "reasons": " ".join(reasons),
        "display": " ".join(acts),
        "mem": med(mems),
        "n": len(dsel),
    }


def fmt(x, nd=3):
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def chunk_sd_and_wave(z, dt=0.001):
    st = line_stats(z, dt)
    return st, stretch_sd(z, 500)


def point_rows(log):
    with open(os.path.join(log, "points.tsv")) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def measure_point(log, pt, meta, clocks):
    path = os.path.join(log, pt["name"] + "-1ms.csv")
    if not os.path.isfile(path):
        path = os.path.join(log, pt["name"] + "-1ms.csv.gz")
    if not os.path.isfile(path):
        return []
    tag = pt["tag"]
    # Sweep points drop the first 2 s. Closed-loop holds and ramp bins do not
    # use that drop here; ramp bins are built separately and holds are whole.
    drop = 0 if tag.startswith("confirm") or tag.startswith("ramp") or tag.startswith("cl-") else 2000
    if tag in ("smoke",):
        return []
    t0 = float(pt["t0_ms"]) + drop
    t1 = float(pt["t1_ms"])
    if tag.startswith("ramp"):
        return []
    d = load_trace(path, t0, t1)
    if len(d["rate"]) < 50:
        return []
    acc = int(pt["acc"])
    rounds = int(pt["rounds"])
    threads = int(pt["threads"])
    rmed = med(d["rate"])
    smed = med(d["sms"])
    ops = ops_of(rmed, acc, rounds)
    st, ssd = chunk_sd_and_wave(d["zraw"])
    info = meta.get(pt["name"], {})
    clk = {}
    if d["wall"]:
        clk = window_clocks(clocks["display"].get(pt["name"], []), clocks["flags"].get(pt["name"], []), d["wall"][0], d["wall"][-1])
    warps = threads / 32.0
    pass_s = (smed * warps / rmed) if rmed and smed else None
    row = {
        "launch": pt["name"],
        "tag": tag,
        "t0_ms": f"{t0:.0f}",
        "t1_ms": f"{t1:.0f}",
        "acc": acc,
        "rounds": rounds,
        "threads": threads,
        "sms": smed,
        "plan_sms": int(pt["sms"]),
        "scale_milli": int(pt["scale_milli"]),
        "scale": f"{int(pt['scale_milli']) / 1000.0:.3f}",
        "rate_ref": fmt(info.get("rate_ref"), 1),
        "pulse_rate": fmt(rmed, 1),
        "ps_pulse": fmt(rmed / smed if smed else None, 1),
        "ops_s": fmt(ops, 1),
        "tflops": fmt(ops / 1e12, 4),
        "ops_ps": fmt(ops / smed if smed else None, 1),
        "z_sd": fmt(pstdev(d["zraw"]), 5),
        "z_sd_500": fmt(ssd, 5),
        "ac128": fmt(st["ac128"] if st else None, 3),
        "ac136": fmt(st["ac136"] if st else None, 3),
        "ac144": fmt(st["ac144"] if st else None, 3),
        "ac_lag_ms": fmt(st["ac_ms"] if st else None, 1),
        "ac_best": fmt(st["ac"] if st else None, 3),
        "narrow_share": fmt(st["frac"] if st else None, 4),
        "lag_share": fmt(st["frac_lag"] if st else None, 4),
        "sm_mhz": fmt(clk.get("sm"), 0),
        "power_w": fmt(clk.get("power"), 1),
        "temp_c": fmt(clk.get("temp"), 0),
        "reasons": clk.get("reasons", ""),
        "display_active": clk.get("display", ""),
        "mem_mhz": fmt(clk.get("mem"), 0),
        "pass_s": fmt(pass_s, 9),
        "n": len(d["rate"]),
    }
    return [row]


def ramp_rows(log, pt, meta, clocks):
    path = os.path.join(log, pt["name"] + "-1ms.csv")
    if not os.path.isfile(path):
        return []
    t0 = float(pt["t0_ms"])
    t1 = float(pt["t1_ms"])
    d = load_trace(path, t0, t1)
    if len(d["rate"]) < 50:
        return []
    buckets = {}
    for i, sc in enumerate(d["scale"]):
        key = round(math.floor((sc + 1e-9) / 0.025) * 0.025, 3)
        buckets.setdefault(key, []).append(i)
    base = int(pt["rounds"])
    acc = int(pt["acc"])
    threads = int(pt["threads"])
    info = meta.get(pt["name"], {})
    rows = []
    for key in sorted(buckets):
        idx = buckets[key]
        if len(idx) < 30:
            continue
        rate = [d["rate"][i] for i in idx]
        sms = [d["sms"][i] for i in idx]
        zraw = [d["zraw"][i] for i in idx]
        zhat = [d["zhat"][i] for i in idx]
        milli = int(med([int(d["scale"][i] * 1000.0 + 0.5) for i in idx]))
        rounds = rounds_of(base, milli)
        rmed = med(rate)
        smed = med(sms)
        ops = ops_of(rmed, acc, rounds)
        st, ssd = chunk_sd_and_wave(zraw)
        walls = [d["wall"][i] for i in idx]
        clk = window_clocks(
            clocks["display"].get(pt["name"], []),
            clocks["flags"].get(pt["name"], []),
            walls[0], walls[-1],
        ) if walls else {}
        changes = sum(1 for a, b in zip(sms, sms[1:]) if a != b)
        bin_target = TARGET if "cl" in pt["name"] else 1.0
        ib = sum(1 for z in zhat if in_band(z, bin_target))
        warps = threads / 32.0
        rows.append({
            "launch": pt["name"],
            "tag": pt["tag"] + f"-{key:.3f}",
            "t0_ms": f"{d['t'][idx[0]]:.0f}",
            "t1_ms": f"{d['t'][idx[-1]]:.0f}",
            "acc": acc,
            "rounds": rounds,
            "threads": threads,
            "sms": smed,
            "plan_sms": int(pt["sms"]),
            "scale_milli": milli,
            "scale": f"{key:.3f}",
            "rate_ref": fmt(info.get("rate_ref"), 1),
            "pulse_rate": fmt(rmed, 1),
            "ps_pulse": fmt(rmed / smed if smed else None, 1),
            "ops_s": fmt(ops, 1),
            "tflops": fmt(ops / 1e12, 4),
            "ops_ps": fmt(ops / smed if smed else None, 1),
            "z_sd": fmt(pstdev(zraw), 5),
            "z_sd_500": fmt(ssd, 5),
            "ac128": fmt(st["ac128"] if st else None, 3),
            "ac136": fmt(st["ac136"] if st else None, 3),
            "ac144": fmt(st["ac144"] if st else None, 3),
            "ac_lag_ms": fmt(st["ac_ms"] if st else None, 1),
            "ac_best": fmt(st["ac"] if st else None, 3),
            "narrow_share": fmt(st["frac"] if st else None, 4),
            "lag_share": fmt(st["frac_lag"] if st else None, 4),
            "sm_mhz": fmt(clk.get("sm"), 0),
            "power_w": fmt(clk.get("power"), 1),
            "temp_c": fmt(clk.get("temp"), 0),
            "reasons": clk.get("reasons", ""),
            "display_active": clk.get("display", ""),
            "mem_mhz": fmt(clk.get("mem"), 0),
            "pass_s": fmt((smed * warps / rmed) if rmed and smed else None, 9),
            "n": len(idx),
            "sm_changes": changes,
            "inband": f"{ib}/{len(zhat)}@{bin_target:.1f}",
        })
    return rows


COLS = [
    "launch", "tag", "t0_ms", "t1_ms", "acc", "rounds", "threads", "sms",
    "plan_sms", "scale_milli", "scale", "rate_ref", "pulse_rate", "ps_pulse",
    "ops_s", "tflops", "ops_ps", "z_sd", "z_sd_500", "ac128", "ac136", "ac144",
    "ac_lag_ms", "ac_best", "narrow_share", "lag_share", "sm_mhz", "power_w",
    "temp_c", "reasons", "display_active", "mem_mhz", "pass_s", "n",
]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def svg_scatter(path, series, title, xlabel, ylabel):
    w, h, l, r, t, b = 960, 520, 70, 20, 40, 50
    xs, ys = [], []
    for _name, pts in series:
        for x, y in pts:
            if y is None:
                continue
            xs.append(x)
            ys.append(y)
    if not xs:
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1
    pad_y = (ymax - ymin) * 0.05
    ymin -= pad_y
    ymax += pad_y

    def X(x):
        return l + (x - xmin) / (xmax - xmin) * (w - l - r)

    def Y(y):
        return t + (ymax - y) / (ymax - ymin) * (h - t - b)

    colors = ["#1f4e79", "#b85c38", "#2e7d4f", "#6b4c9a"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{w/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>',
        f'<line x1="{l}" y1="{t}" x2="{l}" y2="{h-b}" stroke="#222"/>',
        f'<line x1="{l}" y1="{h-b}" x2="{w-r}" y2="{h-b}" stroke="#222"/>',
        f'<text x="{(l+w-r)/2}" y="{h-12}" text-anchor="middle" font-family="sans-serif" font-size="12">{xlabel}</text>',
        f'<text x="16" y="{(t+h-b)/2}" text-anchor="middle" font-family="sans-serif" font-size="12" transform="rotate(-90 16 {(t+h-b)/2})">{ylabel}</text>',
    ]
    for i, (name, pts) in enumerate(series):
        col = colors[i % len(colors)]
        pts = [(x, y) for x, y in pts if y is not None]
        if len(pts) >= 2:
            d = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
            parts.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.2" points="{d}"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.4" fill="{col}"/>')
        parts.append(
            f'<text x="{w-r-8}" y="{t+16+i*16}" text-anchor="end" font-family="sans-serif" font-size="12" fill="{col}">{name}</text>'
        )
    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def fnum(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


def stretches(sms, min_len):
    out = []
    i = 0
    n = len(sms)
    while i < n:
        j = i + 1
        while j < n and sms[j] == sms[i]:
            j += 1
        if j - i >= min_len:
            out.append((i, j, sms[i]))
        i = j
    return out


def limit_cycle(sms):
    if len(sms) < 2:
        return False, 0, {}
    from collections import Counter
    changes = 0
    counts = Counter()
    counts[sms[0]] += 1
    for a, b in zip(sms, sms[1:]):
        counts[b] += 1
        if a != b:
            changes += 1
    present = [k for k, v in counts.items() if v >= 5]
    ok = len(present) == 2 and changes >= 6
    return ok, changes, dict(counts)


def enters(zhat, target):
    flags = [in_band(z, target) for z in zhat]
    if not flags:
        return False, None, 0
    last_out = -1
    for i, f in enumerate(flags):
        if not f:
            last_out = i
    stays = last_out < len(flags) - 1
    entry = 0 if last_out < 0 else (last_out + 1 if stays else None)
    ib = sum(1 for f in flags if f)
    return stays, entry, ib


def tick_delays(path, t_lo, t_hi):
    """Move-to-response on the cut tick file. Same rule as the fourteenth."""
    opener = gzip.open if path.endswith(".gz") else open
    closes = []
    moves = []
    prev_sms = None
    prev_rate = None
    with opener(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["time_s"])
            z = float(row["z_raw"])
            sms = int(row["active_sms"])
            rate = row["rate_ema"]
            if prev_sms is None:
                prev_sms = sms
            elif sms != prev_sms:
                moves.append((t, prev_sms, sms, z))
                prev_sms = sms
            if prev_rate is None:
                prev_rate = rate
            elif rate != prev_rate:
                closes.append((t, z, sms))
                prev_rate = rate
    dts = [closes[i][0] - closes[i - 1][0] for i in range(1, min(len(closes), 4000))]
    dts = [d for d in dts if 0.0005 < d < 0.05]
    win = med(dts) if dts else 0.008
    firsts, halves, first_wins, half_wins = [], [], [], []
    held = 0
    ci = 0
    for tm, fro, to, z0 in moves:
        if tm < t_lo or tm >= t_hi:
            continue
        while ci < len(closes) and closes[ci][0] <= tm:
            ci += 1
        if ci >= len(closes):
            break
        if closes[ci][2] != to:
            continue
        held += 1
        fj = None
        for j in range(ci, min(len(closes), ci + 40)):
            if abs(closes[j][1] - z0) >= 5e-6:
                fj = j
                break
        if fj is None:
            continue
        dt1 = closes[fj][0] - tm
        firsts.append(dt1 * 1000.0)
        first_wins.append(dt1 / win)
        t_lim = tm + 0.080
        direction = -1 if to > fro else 1
        span = [c for c in closes[ci:] if c[0] <= t_lim]
        if not span:
            continue
        zs = [c[1] for c in span]
        zend = min(zs) if direction < 0 else max(zs)
        dz = zend - z0
        if abs(dz) < 5e-5 or (dz > 0) != (direction > 0):
            continue
        half_z = z0 + 0.5 * dz
        got = None
        for c in span:
            if (direction < 0 and c[1] <= half_z) or (direction > 0 and c[1] >= half_z):
                got = c[0] - tm
                break
        if got is None:
            continue
        halves.append(got * 1000.0)
        half_wins.append(got / win)
    nmove = sum(1 for tm, *_ in moves if t_lo <= tm < t_hi)
    return {
        "moves": nmove,
        "held": held,
        "win_ms": win * 1000.0,
        "first_ms": med(firsts),
        "half_ms": med(halves),
        "first_win": med(first_wins),
        "half_win": med(half_wins),
        "first_n": len(firsts),
        "half_n": len(halves),
    }


def cal_end_tick(path):
    opener = gzip.open if path.endswith(".gz") else open
    end = 3.0
    prev_rate = None
    with opener(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["time_s"])
            if t >= 4.0:
                break
            rate = row["rate_ema"]
            sms = int(row["active_sms"])
            if prev_rate is not None and rate != prev_rate and sms == 2:
                end = t
            prev_rate = rate
    return end


def hold_story(log, name, pt, lines, target):
    path = os.path.join(log, name + "-1ms.csv")
    if not os.path.isfile(path):
        path = os.path.join(log, name + "-1ms.csv.gz")
    t0 = float(pt["t0_ms"])
    t1 = float(pt["t1_ms"])
    d = load_trace(path, t0, t1)
    if len(d["zhat"]) < 10:
        lines.append(f"  {name} {pt['tag']}: empty")
        return
    n = len(d["zhat"])
    ib = sum(1 for z in d["zhat"] if in_band(z, target))
    last10_i = [i for i, t in enumerate(d["t"]) if t >= t1 - 10000]
    ib10 = sum(1 for i in last10_i if in_band(d["zhat"][i], target))
    sms10 = [d["sms"][i] for i in last10_i]
    ok, changes, counts = limit_cycle(d["sms"])
    ok10, changes10, counts10 = limit_cycle(sms10) if sms10 else (False, 0, {})
    ch10 = sum(1 for a, b in zip(sms10, sms10[1:]) if a != b) if sms10 else 0
    stays, entry, _ib2 = enters(d["zhat"], target)
    majority = max(counts.values()) / n if counts else 0
    chunks = []
    t_a = t0
    while t_a < t1:
        sel = [(r, s) for r, s, t in zip(d["rate"], d["sms"], d["t"]) if t_a <= t < t_a + 10000 and s > 0 and r > 1.0e6]
        if sel:
            chunks.append(med([r for r, _ in sel]) / med([s for _, s in sel]))
        t_a += 10000
    st = line_stats(d["zraw"], 0.001)
    st_sms = line_stats([float(s) for s in d["sms"]], 0.001) if n >= 1000 else None
    sts = sorted(stretches(d["sms"], 500), key=lambda s: s[1] - s[0], reverse=True)[:8]
    fixed = []
    for a, b, nsm in sts:
        fst = line_stats(d["zraw"][a:b], 0.001)
        if fst:
            fixed.append((nsm, b - a, fst))
    tick = os.path.join(log, name + "-tick-cut.csv.gz")
    delay = None
    anchor = None
    if os.path.isfile(tick) and target == TARGET:
        anchor = cal_end_tick(tick)
        delay = tick_delays(tick, anchor + t0 / 1000.0, anchor + t1 / 1000.0)
    end_z = d["zhat"][-1]
    end_i = d["integral"][-1] if d["integral"] else None
    lines.append(
        f"  {name} {pt['tag']} target={target:.2f} t={t0:.0f}..{t1:.0f} n={n} "
        f"inband {ib}/{n} ({100.0 * ib / n:.1f}%) "
        f"last10 {ib10}/{len(last10_i)} ({(100.0 * ib10 / len(last10_i)) if last10_i else 0:.1f}%) "
        f"stays={stays} entry_sample={entry}"
    )
    lines.append(
        f"    sm {min(d['sms'])}..{max(d['sms'])} end={d['sms'][-1]} "
        f"changes={changes} last10_changes={ch10} "
        f"limit={ok} limit10={ok10} majority={majority:.3f} "
        f"counts={dict(counts)} counts10={dict(counts10)} "
        f"end_z={end_z:.5f} end_I={end_i:.5f}" if end_i is not None else
        f"    sm {min(d['sms'])}..{max(d['sms'])} end={d['sms'][-1]} changes={changes}"
    )
    if st:
        extra = ""
        if st["frac_lag"] is not None:
            extra = f" lag_share={st['frac_lag']:.4f}"
        lines.append(
            f"    Z sd={st['sd']:.5f} ac128={st['ac128']:.3f} ac136={st['ac136']:.3f} "
            f"ac144={st['ac144']:.3f} lag={st['ac_ms']:.1f} ms ac={st['ac']:.3f} "
            f"narrow={st['frac']:.4f}{extra}"
        )
    if st_sms:
        lines.append(
            f"    count ac128={st_sms['ac128']:.3f} ac136={st_sms['ac136']:.3f} "
            f"ac144={st_sms['ac144']:.3f} lag={st_sms['ac_ms']:.1f} ms "
            f"narrow={st_sms['frac']:.4f}"
        )
    if fixed:
        bits = []
        for nsm, ln, fst in fixed:
            bits.append(
                f"nsm={nsm} len={ln} sd={fst['sd']:.4f} ac136={fst['ac136']:.3f} "
                f"lag={fst['ac_ms']:.1f} narrow={fst['frac']:.4f}"
            )
        lines.append("    fixed " + " | ".join(bits))
    else:
        lines.append("    fixed none")
    lines.append("    ps_10s " + " ".join(f"{c:.0f}" for c in chunks))
    if delay:
        lines.append(
            f"    delay anchor={anchor:.3f}s moves={delay['moves']} held={delay['held']} "
            f"first={delay['first_ms']} ms ({delay['first_win']} win, n={delay['first_n']}) "
            f"half={delay['half_ms']} ms ({delay['half_win']} win, n={delay['half_n']}) "
            f"win={delay['win_ms']:.2f} ms"
        )


def display_changes(log, lines):
    path = os.path.join(log, "display-all.csv")
    rows = load_series(path)
    acts = []
    prev = None
    for r in rows:
        a = r.get("display_active", "")
        if a != prev:
            acts.append((r["t"], a))
            prev = a
    lines.append(f"display-all samples={len(rows)} changes={max(0, len(acts)-1)}")
    for ts, a in acts:
        lines.append(f"  display {datetime.fromtimestamp(ts).isoformat(timespec='seconds')} {a}")


def main():
    log = sys.argv[1]
    out = open(os.path.join(log, "numbers.txt"), "w")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        print(s, file=out)

    pts = point_rows(log)
    names = sorted({p["name"] for p in pts})
    meta = {}
    clocks = {"display": {}, "flags": {}}
    for name in names:
        meta[name] = parse_txt(os.path.join(log, name + ".txt"))
        clocks["display"][name] = load_series(os.path.join(log, name + ".display"))
        clocks["flags"][name] = load_series(os.path.join(log, name + ".flags"))
        P(
            f"LAUNCH {name} rate_ref={meta[name]['rate_ref']} "
            f"rail={meta[name]['rail']} sm_moves={meta[name]['sm_moves']} "
            f"sched={meta[name]['sched']} target_line={meta[name]['target']}"
        )
    rows = []
    for pt in pts:
        if pt["tag"].startswith("ramp"):
            rows.extend(ramp_rows(log, pt, meta, clocks))
        else:
            rows.extend(measure_point(log, pt, meta, clocks))
    write_csv(os.path.join(log, "sweep-points.csv"), rows)
    P(f"ROWS {len(rows)}")

    def sel(pred):
        return [r for r in rows if pred(r)]

    def item2(direction):
        return [r for r in rows if r["launch"].startswith(f"r2-{direction}-") and r["tag"] in ("sweep", "base")]

    bases = [r for r in rows if r["tag"] == "base"]
    P(f"BASES {len(bases)}")
    ps = [fnum(r, "ps_pulse") for r in bases]
    ps = [p for p in ps if p]
    if ps:
        P(f"base per-SM pulse min={min(ps):.1f} med={med(ps):.1f} max={max(ps):.1f}")
        high = [r for r in bases if fnum(r, "ps_pulse") and fnum(r, "ps_pulse") >= 598000]
        P(f"base brackets at or above 598k per SM: {len(high)}")
    # opening vs closing base of each scheduled launch
    by = {}
    for r in bases:
        by.setdefault(r["launch"], []).append(r)
    stepped = []
    for name, group in by.items():
        group = sorted(group, key=lambda r: float(r["t0_ms"]))
        if len(group) < 2:
            continue
        a = fnum(group[0], "ps_pulse")
        b = fnum(group[-1], "ps_pulse")
        if a and b and b > a * RATIO:
            stepped.append((name, a, b))
    P(f"bases stepped by the 601/580 ratio: {len(stepped)}")
    for name, a, b in stepped:
        P(f"  step {name} {a:.0f} -> {b:.0f}")

    # item 2: any rounds whose ops/s per SM beats the base by the ratio
    ref = med([fnum(r, "ops_ps") for r in bases if fnum(r, "ops_ps")])
    P(f"ref ops/s per SM from bases {ref}")
    above = []
    for r in item2("asc") + item2("desc"):
        v = fnum(r, "ops_ps")
        if ref and v and v >= ref * RATIO and int(r["rounds"]) != 512:
            above.append(r)
    P(f"item2 non-512 points with ops/s per SM >= base * 601/580: {len(above)}")

    # plots
    for key, ylab, fname in (
        ("ps_pulse", "per-SM pulse rate", "plot-ps-rounds.svg"),
        ("ops_ps", "counted ops/s per SM", "plot-ops-rounds.svg"),
        ("ac136", "autocorrelation at 136 ms", "plot-ac136-rounds.svg"),
        ("ac_lag_ms", "strongest lag, ms", "plot-lag-rounds.svg"),
    ):
        series = []
        for direction in ("asc", "desc"):
            pts_xy = []
            for r in sorted(item2(direction), key=lambda r: int(r["rounds"])):
                pts_xy.append((int(r["rounds"]), fnum(r, key)))
            series.append((direction, pts_xy))
        svg_scatter(
            os.path.join(log, fname), series,
            ylab + " against rounds", "rounds", ylab,
        )

    # lags
    lags = {}
    for r in rows:
        if r["tag"] in ("sweep", "base", "floor", "ceiling"):
            lags.setdefault(r["ac_lag_ms"], 0)
            lags[r["ac_lag_ms"]] += 1
    P("lag counts " + " ".join(f"{k}:{v}" for k, v in sorted(lags.items(), key=lambda kv: -kv[1])[:12]))

    P("RAMPS")
    for r in rows:
        if str(r["tag"]).startswith("ramp"):
            P(
                f"  {r['launch']} {r['tag']} rounds={r['rounds']} sms={r['sms']} "
                f"ps={r['ps_pulse']} ops_ps={r['ops_ps']} tflops={r['tflops']} "
                f"zsd={r['z_sd']} ac136={r['ac136']} lag={r['ac_lag_ms']} "
                f"narrow={r['narrow_share']} changes={r.get('sm_changes')} "
                f"inband={r.get('inband')} n={r['n']}"
            )
    P("ITEM5")
    lines = []
    for pt in pts:
        if pt["tag"] in ("confirm-pre", "confirm-load", "confirm-return", "cl-base"):
            hold_story(log, pt["name"], pt, lines, TARGET)
        elif pt["tag"] == "ceiling":
            hold_story(log, pt["name"], pt, lines, 1.0)
    for s in lines:
        P(s)
    dlines = []
    display_changes(log, dlines)
    for s in dlines:
        P(s)
    # hottest flag sample in the suite
    hottest = None
    hot_where = None
    for name, rows_f in clocks["flags"].items():
        for r in rows_f:
            tv = num_unit(r.get("temperature.gpu", ""))
            if tv is None:
                continue
            if hottest is None or tv > hottest:
                hottest = tv
                hot_where = (name, r["t"])
    P(f"hottest {hottest} at {hot_where}")
    out.close()


if __name__ == "__main__":
    main()
