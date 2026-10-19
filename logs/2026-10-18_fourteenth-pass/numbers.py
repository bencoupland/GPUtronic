#!/usr/bin/env python3
"""Numbers for the fourteenth pass. Reads the logs, writes a text summary."""
import csv
import math
import os
import re
import statistics
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
OUTP = "/tmp/fourteenth-numbers.txt"
TARGET = 1.5

out = open(OUTP, "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    print(s, file=out)


def pct(n, d):
    if d <= 0:
        return "0/0"
    return f"{n} / {d} ({100.0 * n / d:.1f}%)"


def in_band(z, target=TARGET):
    return round(abs(z - target), 5) <= 0.08


def med(xs):
    if not xs:
        return None
    return statistics.median(xs)


def fmt(x, nd=3):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


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


def line_stats(z, dt):
    n = len(z)
    if n < 256:
        return None
    mu = sum(z) / n
    var = sum((v - mu) * (v - mu) for v in z) / n
    sd = math.sqrt(var)
    if sd < 1e-6:
        return None
    # 4 ms samples still resolve a 136 ms line and keep the scan small.
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
    while f <= 60.0:
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
        while f <= best_f + 0.4:
            pwr = goertzel(x, f, dt2)
            if pwr > best_p:
                best_p = pwr
                best_f = f
            f += 0.02
    # autocorrelation, 50–400 ms, on the same 4 ms grid
    denom = sum(v * v for v in x)
    best_ac = -2.0
    best_lag = 0
    lag = int(0.050 / dt2)
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
    return {
        "sd": sd,
        "frac": (line_p / band_p) if band_p else 0.0,
        "peak_ms": (1000.0 / best_f) if best_f else None,
        "ac_ms": best_lag * dt2 * 1000.0,
        "ac": best_ac,
    }


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
        return False, 0, []
    changes = 0
    counts = Counter()
    counts[sms[0]] += 1
    for a, b in zip(sms, sms[1:]):
        counts[b] += 1
        if a != b:
            changes += 1
    present = [k for k, v in counts.items() if v >= 5]
    ok = len(present) == 2 and all(counts[k] >= 5 for k in present) and changes >= 6
    return ok, changes, sorted(counts)


def load_1ms(path):
    t, scale, zraw, zhat, sms, integral, rate, rail, pi, moves = (
        [] for _ in range(10)
    )
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row["t_ms"]) / 1000.0)
            scale.append(float(row["scale"]))
            zraw.append(float(row["z_raw"]))
            zhat.append(float(row["z_hat"]))
            sms.append(int(row["active_sms"]))
            integral.append(float(row["integral"]))
            rate.append(float(row["rate_ema"]))
            rail.append(int(row["rail_ticks"]))
            pi.append(int(row["pi_ticks"]))
            moves.append(int(row["sm_moves"]))
    return {
        "t": t,
        "scale": scale,
        "zraw": zraw,
        "zhat": zhat,
        "sms": sms,
        "integral": integral,
        "rate": rate,
        "rail": rail,
        "pi": pi,
        "moves": moves,
    }


def slice_idx(t, i0, i1, sec):
    """Last `sec` seconds of [i0, i1), anchored on the last sample."""
    if i1 <= i0:
        return i0, i0
    t1 = t[i1 - 1]
    t0 = t1 - sec
    a = i0
    while a < i1 and t[a] < t0:
        a += 1
    return a, i1


def hold_report(name, d, i0, i1, label, target=TARGET):
    t, zhat, zraw, sms, rate = d["t"], d["zhat"], d["zraw"], d["sms"], d["rate"]
    n = i1 - i0
    if n <= 0:
        P(f"  {label}: empty")
        return
    ib = sum(1 for i in range(i0, i1) if in_band(zhat[i], target))
    a30, b30 = slice_idx(t, i0, i1, 30.0)
    a10, b10 = slice_idx(t, i0, i1, 10.0)
    ib30 = sum(1 for i in range(a30, b30) if in_band(zhat[i], target))
    ib10 = sum(1 for i in range(a10, b10) if in_band(zhat[i], target))
    ch = sum(1 for i in range(i0 + 1, i1) if sms[i] != sms[i - 1])
    ch30 = sum(1 for i in range(a30 + 1, b30) if sms[i] != sms[i - 1])
    ch10 = sum(1 for i in range(a10 + 1, b10) if sms[i] != sms[i - 1])
    lc, lc_ch, lc_counts = limit_cycle(sms[a30:b30])
    # per-SM rate, ignore the reset zeros
    ps = [rate[i] / sms[i] for i in range(i0, i1) if sms[i] > 0 and rate[i] > 1.0e6]
    # 10 s chunks
    chunks = []
    t_start = t[i0]
    t_end = t[i1 - 1]
    edge = t_start
    while edge < t_end - 1:
        nxt = min(edge + 10.0, t_end + 1e-6)
        xs = [
            rate[i] / sms[i]
            for i in range(i0, i1)
            if edge <= t[i] < nxt and sms[i] > 0 and rate[i] > 1.0e6
        ]
        if xs:
            chunks.append(med(xs))
        edge = nxt
    # fixed-count stretches inside the hold, >= 500 ms
    sub = sms[i0:i1]
    sts = stretches(sub, 500)
    sts = sorted(sts, key=lambda s: s[1] - s[0], reverse=True)[:8]
    sds = []
    fracs = []
    peaks = []
    acs = []
    for a, b, nsm in sts:
        st = line_stats(zraw[i0 + a : i0 + b], 0.001)
        if st:
            sds.append(st["sd"])
            fracs.append(st["frac"])
            peaks.append(st["peak_ms"])
            acs.append(st["ac_ms"])
    # count series over the whole hold
    ct = line_stats([float(s) for s in sms[i0:i1]], 0.001) if n >= 1000 else None
    # first outside / enters and stays
    outside = [i for i in range(i0, i1) if not in_band(zhat[i], target)]
    if not outside:
        stay = "all inside"
    elif outside[-1] == i1 - 1:
        stay = "ends outside"
    else:
        enter = outside[-1] + 1
        tail_ms = (t[i1 - 1] - t[enter]) * 1000.0
        stay = f"enters {t[enter] - t[i0]:.3f}s tail {tail_ms:.0f} ms"
    first_out = "none"
    if outside:
        first_out = f"{(t[outside[0]] - t[i0]) * 1000.0:.1f} ms"
    d_rail = d["rail"][i1 - 1] - d["rail"][i0]
    d_pi = d["pi"][i1 - 1] - d["pi"][i0]
    d_moves = d["moves"][i1 - 1] - d["moves"][i0]
    P(
        f"  {label} t={t[i0]:.3f}..{t[i1 - 1]:.3f}s n={n} "
        f"in={pct(ib, n)} last30={pct(ib30, b30 - a30)} last10={pct(ib10, b10 - a10)}"
    )
    P(
        f"    sm {min(sms[i0:i1])}..{max(sms[i0:i1])} end={sms[i1 - 1]} "
        f"changes={ch} last30={ch30} last10={ch10} "
        f"limit30={lc} counts30={lc_counts} ch30={lc_ch}"
    )
    P(
        f"    z {min(zhat[i0:i1]):.5f}..{max(zhat[i0:i1]):.5f} "
        f"end_z={zhat[i1 - 1]:.5f} end_I={d['integral'][i1 - 1]:.5f} "
        f"first_out={first_out} {stay}"
    )
    P(
        f"    per_sm med={fmt(med(ps), 0)} chunk10={','.join(fmt(c, 0) for c in chunks)} "
        f"rail_d={d_rail} pi_d={d_pi} moves_d={d_moves}"
    )
    if sds:
        P(
            f"    fixed>500ms n={len(sds)} sd_med={statistics.median(sds):.4f} "
            f"line_med={100.0 * statistics.median(fracs):.1f}% "
            f"peak_med={statistics.median(peaks):.1f} ms ac_med={statistics.median(acs):.1f} ms"
        )
    else:
        P("    fixed>500ms none")
    if ct:
        P(
            f"    count_fft peak={ct['peak_ms']:.1f} ms line={100.0 * ct['frac']:.1f}% "
            f"ac={ct['ac_ms']:.1f} ms"
        )


def analyze_1ms(name):
    path = os.path.join(BASE, name + "-1ms.csv")
    sz = os.path.getsize(path)
    d = load_1ms(path)
    n = len(d["t"])
    P(
        f"TRACE {name} bytes={sz} rows={n} t={d['t'][0]:.3f}..{d['t'][-1]:.3f}s "
        f"scale={min(d['scale']):.3f}..{max(d['scale']):.3f}"
    )
    if name.startswith(("fw", "wc", "acc")):
        hold_report(name, d, 0, n, "open", target=1.0)
        return
    # closed loop: find 1.2 then the return to 1.0
    load_i = next((i for i, s in enumerate(d["scale"]) if s >= 1.15), None)
    if load_i is None:
        P("  no load step")
        return
    ret_i = next((i for i in range(load_i, n) if d["scale"][i] <= 1.05), None)
    if ret_i is None:
        ret_i = n
    hold_report(name, d, load_i, ret_i, "load")
    if ret_i < n:
        hold_report(name, d, ret_i, n, "return")


def parse_txt(name):
    path = os.path.join(BASE, name + ".txt")
    if not os.path.exists(path):
        return
    text = open(path, errors="replace").read()
    P(f"LOG {name}")
    for pat in (
        r"\[RATE\] window.*",
        r"\[CTRL\] cached.*",
        r"\[PULSE\] local.*",
        r"\[T9\].*",
        r"\[T11\].*",
        r"\[CAL\] rate_ref=.*",
        r"\[CAL\] cuda table.*",
        r"\[CAL\] cuda target.*",
        r"\[CAL\] integral clamp=.*",
        r"\[STATS\] loop_iters=.*",
        r"\[STATS\] samples=.*",
    ):
        for m in re.finditer(pat, text):
            P(" ", m.group(0))
    works = re.findall(
        r"\[WORK\] t=\d+s pulses=([0-9.]+) batches=([0-9.]+) ratio=([0-9.]+)", text
    )
    if works:
        ratios = [float(w[2]) for w in works]
        pulses = [float(w[0]) for w in works]
        batches = [float(w[1]) for w in works]
        P(
            f"  WORK n={len(works)} pulse_med={statistics.median(pulses):.0f} "
            f"batch_med={statistics.median(batches):.0f} "
            f"ratio {min(ratios):.5f}..{max(ratios):.5f} med={statistics.median(ratios):.5f}"
        )


def tick_file(name):
    path = os.path.join(BASE, name + "-tick.csv")
    # One pass. Keep window closes and move events. Calibration rates.
    phases = {"warm": [], "sus": [], "p61": [], "p41": [], "p20": [], "p2": []}
    closes = []  # t, z, zh, e, integ, sms, rate
    moves = []
    reversals = 0
    windows = 0
    seq = []
    prev_rate = None
    prev_sms = None
    prev_z = None
    n_rows = 0
    t_last = 0.0
    ctrl_hz = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            n_rows += 1
            t = float(row["time_s"])
            t_last = t
            z = float(row["z_raw"])
            sms = int(row["active_sms"])
            rate = row["rate_ema"]
            if n_rows <= 5 or n_rows % 200000 == 0:
                ctrl_hz.append(int(float(row["ctrl_hz"])))
            if prev_sms is None:
                prev_sms = sms
                seq = [sms]
            elif sms != prev_sms:
                moves.append((t, prev_sms, sms, z, len(closes)))
                prev_sms = sms
                if not seq or seq[-1] != sms:
                    seq.append(sms)
            if prev_rate is None:
                prev_rate = rate
            elif rate != prev_rate:
                windows += 1
                if len(seq) >= 3:
                    seen = {}
                    rev = False
                    for i, v in enumerate(seq):
                        if v in seen and i - seen[v] >= 2:
                            rev = True
                            break
                        seen[v] = i
                    if rev:
                        reversals += 1
                closes.append(
                    (
                        t,
                        z,
                        float(row["z_hat"]),
                        float(row["error"]),
                        float(row["integral"]),
                        sms,
                        float(rate) if rate else 0.0,
                    )
                )
                seq = [sms]
                prev_rate = rate
                # calibration buckets on window closes, first 4 s
                if t < 4.0:
                    rv = float(rate)
                    if rv > 1.0:
                        if t < 0.7:
                            phases["warm"].append(rv)
                        elif t < 1.8:
                            phases["sus"].append(rv)
                        elif sms == 61:
                            phases["p61"].append(rv)
                        elif sms == 41:
                            phases["p41"].append(rv)
                        elif sms == 20:
                            phases["p20"].append(rv)
                        elif sms == 2:
                            phases["p2"].append(rv)
            prev_z = z
    # measured window
    dts = [closes[i][0] - closes[i - 1][0] for i in range(1, min(len(closes), 5000))]
    dts = [d for d in dts if 0.0005 < d < 0.05]
    win = statistics.median(dts) if dts else 0.008
    P(
        f"TICK {name} rows={n_rows} t_end={t_last:.3f} closes={len(closes)} "
        f"moves={len(moves)} reversals={reversals} windows={windows} "
        f"win_med={win * 1000:.2f} ms ctrl_hz_samples={ctrl_hz[:6]}"
    )
    for k in ("warm", "sus", "p61", "p41", "p20", "p2"):
        xs = phases[k]
        P(
            f"  cal {k} n={len(xs)} med={fmt(med(xs), 0)} "
            f"first={fmt(xs[0] if xs else None, 0)} last={fmt(xs[-1] if xs else None, 0)}"
        )
    # delays on moves that are still in force at the next close
    firsts = []
    halves = []
    first_wins = []
    half_wins = []
    held = 0
    no_half = 0
    # index closes by time via a pointer
    ci = 0
    nc = len(closes)
    for mi, (tm, fro, to, z0, _) in enumerate(moves):
        if tm < 3.0:
            continue
        while ci < nc and closes[ci][0] <= tm:
            ci += 1
        if ci >= nc:
            break
        # next close
        t_next, z_next, _, _, _, sms_next, _ = closes[ci]
        if sms_next != to:
            continue
        held += 1
        # first z change after the move
        fj = None
        for j in range(ci, min(nc, ci + 40)):
            if abs(closes[j][1] - z0) >= 5e-6:
                fj = j
                break
        if fj is None:
            continue
        dt1 = closes[fj][0] - tm
        firsts.append(dt1 * 1000.0)
        first_wins.append(dt1 / win)
        # half toward the extreme in the next 80 ms, expected direction
        t_lim = tm + 0.080
        direction = -1 if to > fro else 1
        span = []
        for j in range(ci, nc):
            if closes[j][0] > t_lim:
                break
            span.append(closes[j])
        if not span:
            no_half += 1
            continue
        zs = [c[1] for c in span]
        zend = min(zs) if direction < 0 else max(zs)
        dz = zend - z0
        if abs(dz) < 5e-5 or (dz > 0) != (direction > 0):
            no_half += 1
            continue
        half_z = z0 + 0.5 * dz
        got = None
        for c in span:
            if (direction < 0 and c[1] <= half_z) or (direction > 0 and c[1] >= half_z):
                got = c[0] - tm
                break
        if got is None:
            no_half += 1
            continue
        halves.append(got * 1000.0)
        half_wins.append(got / win)
    P(
        f"  delay held={held} first_n={len(firsts)} half_n={len(halves)} no_half={no_half} "
        f"first_med={fmt(med(firsts), 2)} ms ({fmt(med(first_wins), 2)} win) "
        f"half_med={fmt(med(halves), 2)} ms ({fmt(med(half_wins), 2)} win)"
    )
    if firsts:
        within1 = sum(1 for w in first_wins if w <= 1.5)
        P(f"  first_change within 1.5 windows: {pct(within1, len(first_wins))}")
    # moves per post-cal region, using cal end = last close before t=4 with sms==2, else 3.0
    cal_end = 3.0
    for c in closes:
        if c[0] < 4.0 and c[5] == 2:
            cal_end = c[0]
    P(f"  cal_end~{cal_end:.3f}")
    regions = {
        "pre": (cal_end, cal_end + 20),
        "load": (cal_end + 20, cal_end + 65),
        "return": (cal_end + 65, cal_end + 110),
    }
    for label, (a, b) in regions.items():
        mv = [m for m in moves if a <= m[0] < b]
        # last 30 and last 10 anchored on b
        m30 = [m for m in mv if m[0] >= b - 30]
        m10 = [m for m in mv if m[0] >= b - 10]
        # reversals approximated: moves whose next move returns, inside a short gap
        P(f"  tick_moves {label} {len(mv)} last30={len(m30)} last10={len(m10)}")
    # excerpt: densest 1.6 s of the load region
    a, b = regions["load"]
    if closes:
        best = None
        # step 0.2 s
        t0 = a
        load_moves = [m[0] for m in moves if a <= m[0] < b]
        while t0 < b - 1.6:
            c = sum(1 for tm in load_moves if t0 <= tm < t0 + 1.6)
            if best is None or c > best[0]:
                best = (c, t0)
            t0 += 0.2
        if best and best[0] > 0:
            e0, e1 = best[1], best[1] + 1.6
            P(f"  excerpt {e0:.3f}..{e1:.3f} moves={best[0]}")
            excerpt = os.path.join(BASE, name + "-excerpt.csv")
            # only write excerpt for the primary hunting run
            rows_out = []
            for c in closes:
                if e0 <= c[0] <= e1:
                    rows_out.append(c)
            if name == "i1-a":
                with open(excerpt, "w") as ef:
                    ef.write("time_s,z_raw,z_hat,error,integral,active_sms,win_close\n")
                    for c in rows_out:
                        ef.write(
                            f"{c[0]:.6f},{c[1]:.5f},{c[2]:.5f},{c[3]:.5f},{c[4]:.5f},{c[5]},1\n"
                        )
                # print two periods worth: first 0.40 s of window closes
                shown = [c for c in rows_out if c[0] <= e0 + 0.40]
                P(f"  excerpt_rows {len(rows_out)} shown {len(shown)}")
                for c in shown:
                    P(
                        f"    {c[0]:.6f} z={c[1]:.5f} hat={c[2]:.5f} e={c[3]:.5f} "
                        f"I={c[4]:.5f} sms={c[5]}"
                    )
    return


def clocks():
    temps = []
    reasons = Counter()
    disp = Counter()
    disp_changes = []
    busy_sm = []
    busy_mem = Counter()
    busy_pwr = []
    for fn in sorted(os.listdir(BASE)):
        path = os.path.join(BASE, fn)
        if fn.endswith(".flags"):
            with open(path, errors="replace") as f:
                for line in f:
                    if not line[:1].isdigit():
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 6:
                        continue
                    try:
                        temps.append(int(parts[1]))
                    except ValueError:
                        pass
                    reasons[parts[5]] += 1
        elif fn.endswith(".display"):
            prev = None
            with open(path, errors="replace") as f:
                for line in f:
                    if not line[:1].isdigit():
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 8:
                        continue
                    disp[parts[1]] += 1
                    if prev is not None and parts[1] != prev:
                        disp_changes.append((fn, parts[0], prev, parts[1]))
                    prev = parts[1]
                    util = parts[4].replace("%", "").strip()
                    try:
                        u = float(util)
                    except ValueError:
                        u = 0.0
                    if u >= 90.0:
                        sm = parts[5].replace("MHz", "").strip()
                        mem = parts[6].replace("MHz", "").strip()
                        pwr = parts[7].replace("W", "").strip()
                        try:
                            busy_sm.append(int(float(sm)))
                            busy_mem[mem] += 1
                            busy_pwr.append(float(pwr))
                        except ValueError:
                            pass
    P(
        f"CLOCKS temp {min(temps) if temps else '—'}..{max(temps) if temps else '—'} C "
        f"n={len(temps)} reasons={dict(reasons)}"
    )
    P(f"DISPLAY {dict(disp)} changes={len(disp_changes)}")
    for c in disp_changes[:12]:
        P("  change", c)
    if busy_sm:
        P(
            f"BUSY sm {min(busy_sm)}..{max(busy_sm)} MHz mem={dict(busy_mem)} "
            f"pwr {min(busy_pwr):.0f}..{max(busy_pwr):.0f} W n={len(busy_sm)}"
        )


def smaccum(name):
    path = os.path.join(BASE, name + "-smaccum.csv")
    ops = Counter()
    bins = Counter()
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            phys = int(row["phys_sm"])
            ops[phys] += int(row["d_ops_cuda"])
            bins[phys] += 1
    if not ops:
        P(f"ACCUM {name} empty")
        return
    rates = []
    for phys, total in ops.items():
        # bins are 1 ms, so ops per second is total / (bins/1000) = total * 1000 / bins
        rates.append((total * 1000.0 / bins[phys], phys, bins[phys]))
    rates.sort()
    vals = [v for v, _, _ in rates]
    P(
        f"ACCUM {name} sms={len(vals)} min={vals[0]:.3e} med={statistics.median(vals):.3e} "
        f"max={vals[-1]:.3e} max/min={vals[-1] / vals[0]:.3f}"
    )
    P("  low", ", ".join(f"{p}:{v:.3e}" for v, p, _ in rates[:5]))
    P("  high", ", ".join(f"{p}:{v:.3e}" for v, p, _ in rates[-5:]))


def main():
    runs = []
    for fn in sorted(os.listdir(BASE)):
        if fn.endswith(".txt") and not fn.startswith("nvidia") and fn not in (
            "help.txt",
            "status.txt",
        ):
            # skip proc files; those are name-proc-start.txt
            if "-proc-" in fn:
                continue
            runs.append(fn[:-4])
    P("RUNS", " ".join(runs))
    for name in runs:
        parse_txt(name)
    P("\n==== 1 ms ====")
    for name in runs:
        if os.path.exists(os.path.join(BASE, name + "-1ms.csv")):
            analyze_1ms(name)
    P("\n==== clocks ====")
    clocks()
    P("\n==== accum ====")
    for name in ("acc-01", "acc-02"):
        smaccum(name)
    P("\n==== ticks ====")
    for name in (
        "i1-a",
        "i1-b",
        "i2-base",
        "i2-c250",
        "i2-c100",
        "i2-w4",
        "i2-w2",
        "i2-both",
        "fw-01",
        "fw-localoff-a",
        "cl-01",
    ):
        tick_file(name)
    P("DONE")
    out.close()


if __name__ == "__main__":
    main()
