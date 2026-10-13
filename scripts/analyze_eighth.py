#!/usr/bin/env python3
"""Reduce the eighth-pass logs.

--pick-clamp reads the four cached T6 traces and prints one line
`PICK <sms|none>`. The smallest of 6, 9, and 12 that holds the 0.08 band
during the loaded window wins. If today's scaled clamp already holds, or
none of the wider clamps do, the line is `PICK none`.

The default run prints the measured tables for the write-up.
Search holds are labeled as 8 s. The confirmation hold is the 60 s run.
"""
import csv
import os
import re
import statistics
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.environ.get("EIGHTH_LOG", os.path.join(ROOT, "logs", "2026-10-13_eighth-pass"))

TARGET = 1.5
BAND = 0.08
# Held is the loaded window. The load comes off at 45 s, so 25–44 s
# does not mix the release into the clamp decision.
HELD = (25000.0, 44000.0)
BEFORE = (10000.0, 19000.0)
RELEASE = (50000.0, 58000.0)
OPS_PER_PULSE = 65536  # GOLD_OPS_CUDA_WARP_PASS at reps=chains=1
CUDA_CORES_PER_SM = 128
FLOPS_PER_FMA = 2


def read_text(name):
    path = os.path.join(LOG, name + ".txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


def load_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def load_trace(name):
    path = os.path.join(LOG, name + ".csv")
    return load_csv(path)


def window(rows, t0, t1):
    out = []
    for row in rows:
        t = float(row["t_ms"])
        if t0 <= t < t1:
            out.append(row)
    return out


def col(rows, key, cast=float):
    return [cast(r[key]) for r in rows]


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


def fmt(s, digits):
    if s is None:
        return "none"
    return "n=%d mean=%.*f std=%.*f min=%.*f max=%.*f" % (
        s["n"], digits, s["mean"], digits, s["std"], digits, s["min"], digits, s["max"])


def in_band(z):
    return abs(z - TARGET) < BAND


def pulse_rate(rows):
    """Mean of the binned pulse rate. win_pulses is one rate window, not one bin."""
    if not rows or "win_pulses" not in rows[0]:
        return float("nan")
    return statistics.fmean(float(r["win_pulses"]) for r in rows) / 0.008


def rail_fraction(rows):
    """Share of samples with |integral| on the clamp. Matches the host rail test."""
    if not rows:
        return float("nan"), float("nan")
    hit = 0
    near = 0
    for r in rows:
        integ = abs(float(r["integral"]))
        clamp = abs(float(r["clamp"]))
        if clamp <= 0:
            continue
        if integ >= 0.999 * clamp:
            hit += 1
        if abs(integ - clamp) <= 0.02 or integ >= 0.999 * clamp:
            near += 1
    return hit / len(rows), near / len(rows)


def band_fraction(rows):
    if not rows:
        return float("nan")
    zs = col(rows, "z_hat")
    return sum(1 for z in zs if in_band(z)) / len(zs)


def holds(rows):
    part = window(rows, *HELD)
    if len(part) < 50:
        return False, part
    zs = col(part, "z_hat")
    mean = statistics.fmean(zs)
    frac = sum(1 for z in zs if in_band(z)) / len(zs)
    ok = 1.42 <= mean <= 1.58 and frac >= 0.80
    return ok, part


def leave_and_return(rows):
    """Milliseconds after the 20 s step until Z leaves, and until it stays 100 ms."""
    after = [(float(r["t_ms"]), float(r["z_hat"])) for r in rows if float(r["t_ms"]) >= 20000.0]
    left = None
    for t, z in after:
        if not in_band(z):
            left = t - 20000.0
            break
    back = None
    if left is not None:
        for i, (t, z) in enumerate(after):
            if t < 20000.0 + left:
                continue
            if not in_band(z):
                continue
            end = t + 100.0
            if after[-1][0] < end:
                break
            ok = True
            for t2, z2 in after[i:]:
                if t2 >= end:
                    break
                if not in_band(z2):
                    ok = False
                    break
            if ok:
                back = t - 20000.0
                break
    return left, back


def trace_brief(name):
    rows = load_trace(name)
    if not rows:
        return {"name": name, "missing": True}
    ok, held = holds(rows)
    before = window(rows, *BEFORE)
    release = window(rows, *RELEASE)
    left, back = leave_and_return(rows)
    rail, near = rail_fraction(held)
    return {
        "name": name,
        "missing": False,
        "holds": ok,
        "rows": len(rows),
        "before_z": stats(col(before, "z_hat")) if before else None,
        "before_sm": stats(col(before, "active_sms")) if before else None,
        "before_rate": pulse_rate(before),
        "held_z": stats(col(held, "z_hat")) if held else None,
        "held_sm": stats(col(held, "active_sms")) if held else None,
        "held_rate": pulse_rate(held),
        "held_in_band": band_fraction(held),
        "held_rail": rail,
        "held_rail_near": near,
        "release_z": stats(col(release, "z_hat")) if release else None,
        "release_sm": stats(col(release, "active_sms")) if release else None,
        "release_rate": pulse_rate(release),
        "left_ms": left,
        "back_ms": back,
        "clamp_line": _clamp_line(name),
    }


def _clamp_line(name):
    text = read_text(name)
    if not text:
        return ""
    found = re.findall(r"\[CAL\] integral clamp=.*", text)
    return found[-1] if found else ""


def pick_lines():
    order = [("t6-scaled", None), ("t6-c6", 6), ("t6-c9", 9), ("t6-c12", 12)]
    lines = []
    choice = "none"
    scaled_holds = False
    for name, sms in order:
        brief = trace_brief(name)
        if brief["missing"]:
            lines.append(f"{name}: MISSING")
            continue
        z = brief["held_z"]
        lines.append(
            "%s holds=%s in_band=%.3f rail=%.3f Z %s SMs %s left_ms=%s back_ms=%s"
            % (name, int(brief["holds"]), brief["held_in_band"], brief["held_rail"],
               fmt(z, 4), fmt(brief["held_sm"], 2),
               "none" if brief["left_ms"] is None else "%.1f" % brief["left_ms"],
               "none" if brief["back_ms"] is None else "%.1f" % brief["back_ms"]))
        if brief["clamp_line"]:
            lines.append("  " + brief["clamp_line"])
        if name == "t6-scaled":
            scaled_holds = brief["holds"]
        elif brief["holds"] and choice == "none" and not scaled_holds:
            choice = str(sms)
    if scaled_holds:
        lines.append("scaled holds under the 1.2 load; do not widen")
        choice = "none"
    elif choice == "none":
        lines.append("none of 6, 9, 12 held the band")
    return choice, lines


def write_fake_trace(path, held_z, frac_out=0.0):
    """One-millisecond rows. frac_out of the held window sits at 1.70."""
    header = ("wall_s,t_ms,scale,z_raw,z_hat,active_sms,integral,clamp,rate_ema,sensor,"
              "win_pulses,p_term,loop_iters,loop_hz,pi_ticks,deadband_held,integral_updates,"
              "rail_ticks,sm_moves,p_only_moves\n")
    with open(path, "w") as f:
        f.write(header)
        for t in range(0, 60000, 10):
            z = 1.50
            sm = 55
            integ = -0.2
            scale = 1.0
            if 20000 <= t < 45000:
                scale = 1.2
                sm = 61
                integ = -1.4
                z = held_z
                if frac_out > 0 and (t // 10) % 100 < int(frac_out * 100):
                    z = 1.70
            if t >= 45000:
                z = 1.35
                sm = 58
            rate_w = 2000.0
            f.write("0,%d,%.3f,%.5f,%.5f,%d,%.5f,1.400,1.0,cuda,%.0f,0,1,1,1,0,0,0,0,0\n" % (
                t, scale, z, z, sm, integ, rate_w))


def self_test():
    cases = [
        ("c6 holds", {"t6-scaled": (1.64, 0.0), "t6-c6": (1.50, 0.05),
                      "t6-c9": (1.50, 0.0), "t6-c12": (1.50, 0.0)}, "6"),
        ("only 12", {"t6-scaled": (1.64, 0.0), "t6-c6": (1.64, 0.0),
                     "t6-c9": (1.70, 0.5), "t6-c12": (1.49, 0.10)}, "12"),
        ("none", {"t6-scaled": (1.64, 0.0), "t6-c6": (1.62, 0.4),
                  "t6-c9": (1.60, 0.4), "t6-c12": (1.40, 0.5)}, "none"),
        ("scaled holds", {"t6-scaled": (1.50, 0.05), "t6-c6": (1.50, 0.0),
                          "t6-c9": (1.50, 0.0), "t6-c12": (1.50, 0.0)}, "none"),
    ]
    failed = 0
    for label, spec, expect in cases:
        with tempfile.TemporaryDirectory() as tmp:
            global LOG
            old = LOG
            LOG = tmp
            try:
                for name, (z, frac) in spec.items():
                    write_fake_trace(os.path.join(tmp, name + ".csv"), z, frac)
                choice, lines = pick_lines()
            finally:
                LOG = old
        if choice != expect:
            failed += 1
            print(f"SELF-TEST FAIL {label}: got {choice} expected {expect}")
            for line in lines:
                print("  " + line)
        else:
            print(f"SELF-TEST OK {label} -> {choice}")
    if failed:
        sys.exit(1)
    print("SELF-TEST PASS")


def grab(text, pat):
    if not text:
        return []
    return re.findall(pat, text)


def dmon_rows(name):
    path = os.path.join(LOG, name + ".dmon.txt")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            p = line.split()
            if len(p) < 12:
                continue
            try:
                rows.append({
                    "pwr": float(p[1]), "temp": float(p[2]), "sm": float(p[4]),
                    "mclk": float(p[10]), "pclk": float(p[11]),
                })
            except ValueError:
                continue
    return rows


def clock_epoch(name):
    path = os.path.join(LOG, name + ".clock")
    if not os.path.exists(path):
        return None
    m = re.search(r"dmon_start ([0-9.]+)", open(path).read())
    return float(m.group(1)) if m else None


def load_wall(name):
    text = read_text(name)
    if not text:
        return None
    m = re.search(r"schedule t=0 at wall ([0-9]+\.[0-9]+)", text)
    return float(m.group(1)) if m else None


def dmon_slice(name, t0_s, t1_s):
    """dmon samples whose 1 s slot falls inside schedule seconds [t0, t1)."""
    rows = dmon_rows(name)
    start = clock_epoch(name)
    wall = load_wall(name)
    if start is None or wall is None:
        return rows, "clock or load wall missing; all samples"
    part = []
    for i, row in enumerate(rows):
        sched = (start + i) - wall
        if t0_s <= sched < t1_s:
            part.append(row)
    return part, "samples %d of %d in schedule %.0f..%.0fs" % (len(part), len(rows), t0_s, t1_s)


def fmt_power(part, note):
    if not part:
        return "no dmon (%s)" % note
    pwr = stats([r["pwr"] for r in part])
    temp = stats([r["temp"] for r in part])
    sm = stats([r["sm"] for r in part])
    clk = stats([r["pclk"] for r in part])
    return "%s | power %s | temp %s | sm%% %s | pclk %s" % (
        note, fmt(pwr, 1), fmt(temp, 1), fmt(sm, 1), fmt(clk, 0))


def tel_after_load(name):
    text = read_text(name)
    if not text:
        return []
    if "[LOAD] calibrated" in text:
        text = text.split("[LOAD] calibrated", 1)[1]
    out = []
    for line in text.splitlines():
        m = re.search(r"\[TEL\].*Z_hat=([0-9.]+).*rate=([0-9.]+).*\| n=\s*([0-9]+).*pulses=([0-9]+)", line)
        if m:
            out.append({
                "z": float(m.group(1)),
                "rate": float(m.group(2)),
                "n": int(m.group(3)),
                "pulses": int(m.group(4)),
            })
    return out


def rate_ref_of(name):
    text = read_text(name)
    if not text:
        return None
    found = re.findall(r"\[CAL\] rate_ref=([0-9.]+) pulses/s", text)
    return float(found[-1]) if found else None


def years_to_wrap(per_second):
    if per_second <= 0:
        return float("inf")
    return (2 ** 64) / per_second / (365.25 * 24 * 3600)


def fp32_peak(sms, clock_mhz):
    return sms * CUDA_CORES_PER_SM * FLOPS_PER_FMA * clock_mhz * 1e6


def present_summary(path):
    rows = load_csv(path)
    if not rows:
        return "MISSING"
    dts = []
    for row in rows:
        t = float(row["t_ms"])
        dt = float(row["dt_ms"])
        if t < 500.0 or dt <= 0:
            continue
        dts.append((t, dt))
    if not dts:
        return "no frames after 500 ms"

    def pack(part):
        if not part:
            return "none"
        xs = sorted(d for _t, d in part)
        def pct(p):
            return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
        over = sum(1 for d in xs if d > 20.0)
        return "n=%d median=%.2f p99=%.2f max=%.2f over20ms=%d" % (
            len(xs), pct(0.5), pct(0.99), xs[-1], over)

    # The probe starts about 0.5 s after the schedule. Steps are at 2 s and 5 s.
    bands = [("all after 0.5s", 500, 1e9),
             ("before first step", 500, 1400),
             ("across 82to42", 1400, 2200),
             ("between", 2200, 4300),
             ("across 42to82", 4300, 5200),
             ("after", 5200, 1e9)]
    lines = []
    for label, a, b in bands:
        lines.append(f"  {label}: {pack([(t, d) for t, d in dts if a <= t < b])}")
    return "\n".join(lines)


def smid_report(name):
    text = read_text(name)
    if not text:
        return ["MISSING"]
    lines = []
    for line in text.splitlines():
        if line.startswith("[SMID] left out") or line.startswith("[SMID] launched") \
                or line.startswith("[SMID] recorded") or line.startswith("[SMID] live_count") \
                or line.startswith("[IDLE] ") or line.startswith("[RELAUNCH] ") \
                or line.startswith("[SPAWN] ") or line.startswith("[SMID] launch"):
            lines.append(line)
    return lines


def step_latency(name):
    text = read_text(name) or ""
    steps = []
    for line in text.splitlines():
        m = re.search(r"\[STEP\] t_ms=([0-9.]+) active_sms -> ([0-9]+)", line)
        if m:
            steps.append((float(m.group(1)), int(m.group(2)), []))
            continue
        if line.startswith("[RELAUNCH] ") or line.startswith("[SPAWN] "):
            if steps:
                steps[-1][2].append(line.strip())
    out = []
    for t, n, extra in steps:
        out.append(f"step t={t:.1f} -> {n}")
        for line in extra[:4]:
            out.append("  " + line)
    return out


def tally_latency(name, down_to, up_to):
    """Time from each step until logical SMs at or above the cut stop or start."""
    path = os.path.join(LOG, name + ".smaccum.csv.zst")
    plain = os.path.join(LOG, name + ".smaccum.csv")
    if os.path.exists(path):
        proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, text=True)
        src = proc.stdout
    elif os.path.exists(plain):
        proc = None
        src = open(plain)
    else:
        return ["no tally"]
    last_high = None
    first_high_after = None
    try:
        reader = csv.DictReader(src)
        for row in reader:
            t = float(row["t_ms"])
            sm = int(float(row["sm"]))
            dops = float(row["d_ops_cuda"])
            if sm < down_to or dops <= 0:
                continue
            if 2000.0 <= t < 5000.0:
                last_high = t
            if t >= 5000.0 and first_high_after is None:
                first_high_after = t
    finally:
        src.close()
        if proc:
            proc.wait()
    lines = []
    if last_high is None:
        lines.append("down: no ops on logical SM >= %d after 2000 ms (inside one bin, or never started)" % down_to)
    else:
        lines.append("down: last ops on logical SM >= %d at %.2f ms after the step" % (down_to, last_high - 2000.0))
    if first_high_after is None:
        lines.append("up: logical SM >= %d did not resume" % up_to)
    else:
        lines.append("up: first ops on logical SM >= %d at %.2f ms after the step" % (up_to, first_high_after - 5000.0))
    return lines


def report_checks():
    print("\n== checks (switches unset) ==")
    for name in ("check-gold", "check-transfer", "check-episode"):
        text = read_text(name)
        if not text:
            print(f"{name}: MISSING")
            continue
        result = "PASS" if "RESULT: PASS" in text else "not PASS"
        means = re.findall(r"\[CHECK\] pulses=.*z_mean=([0-9.]+)", text)
        free = re.findall(r"\[CHECK\] free-run ([0-9.]+) pulse/s.*closed ([0-9.]+) pulse/s", text)
        print(f"{name}: {result} z_mean={means[-1] if means else 'none'} "
              f"free/closed={free[-1] if free else 'n/a'}")


def report_clamp():
    print("\n== clamp ==")
    for name in ("t6-scaled", "t6-c6", "t6-c9", "t6-c12"):
        b = trace_brief(name)
        if b["missing"]:
            print(f"{name}: MISSING")
            continue
        print(f"\n{name} holds={int(b['holds'])} rows={b['rows']}")
        if b["clamp_line"]:
            print(b["clamp_line"])
        print(f"  before Z {fmt(b['before_z'], 4)} SMs {fmt(b['before_sm'], 2)} rate {b['before_rate']:.0f}/s")
        print(f"  held   Z {fmt(b['held_z'], 4)} SMs {fmt(b['held_sm'], 2)} rate {b['held_rate']:.0f}/s "
              f"in_band={b['held_in_band']:.3f} rail={b['held_rail']:.3f}")
        print(f"  release Z {fmt(b['release_z'], 4)} SMs {fmt(b['release_sm'], 2)} rate {b['release_rate']:.0f}/s")
        print(f"  left band {b['left_ms']} ms after the step; back for 100 ms at {b['back_ms']}")
        if b["before_rate"] and b["held_rate"] and b["before_rate"] > 0:
            print(f"  held/before rate {b['held_rate'] / b['before_rate']:.3f}")
    off = [n for n in os.listdir(LOG) if n.startswith("t6-off-") and n.endswith(".txt")] if os.path.isdir(LOG) else []
    for fn in sorted(off):
        name = fn[:-4]
        b = trace_brief(name)
        if b["missing"]:
            continue
        print(f"\n{name} (switches off) holds={int(b['holds'])}")
        if b["clamp_line"]:
            print(b["clamp_line"])
        print(f"  before Z {fmt(b['before_z'], 4)} SMs {fmt(b['before_sm'], 2)} rate {b['before_rate']:.0f}/s")
        print(f"  held   Z {fmt(b['held_z'], 4)} SMs {fmt(b['held_sm'], 2)} rate {b['held_rate']:.0f}/s "
              f"in_band={b['held_in_band']:.3f} rail={b['held_rail']:.3f}")
        print(f"  release Z {fmt(b['release_z'], 4)} SMs {fmt(b['release_sm'], 2)}")
        print(f"  left band {b['left_ms']} ms; back {b['back_ms']}")


def report_power():
    print("\n== power (cached plant, 30 s open loop, dmon 5–28 s of the hold) ==")
    print("mode n power_W temp_C sm% pclk_MHz rate_Mpulse z n_tel")
    for mode in ("spin", "exit", "relaunch"):
        for n in (4, 12, 24, 36, 48, 56, 72, 82):
            name = f"pwr-{mode}-n{n}"
            part, note = dmon_slice(name, 5.0, 28.0)
            tels = tel_after_load(name)
            mid = tels[5:-2] if len(tels) > 8 else tels
            rate = statistics.fmean(t["rate"] for t in mid) if mid else float("nan")
            z = statistics.fmean(t["z"] for t in mid) if mid else float("nan")
            pwr = statistics.fmean(r["pwr"] for r in part) if part else float("nan")
            temp = statistics.fmean(r["temp"] for r in part) if part else float("nan")
            sm = statistics.fmean(r["sm"] for r in part) if part else float("nan")
            clk = statistics.fmean(r["pclk"] for r in part) if part else float("nan")
            print(f"{mode} {n:3d} {pwr:7.1f} {temp:5.1f} {sm:6.1f} {clk:7.0f} {rate/1e6:8.2f} {z:6.3f} ntel={len(tels)} {note}")


def report_idle_maps():
    print("\n== smid and command lines ==")
    for name in ("pwr-exit-n4", "pwr-exit-n82", "pwr-relaunch-n4", "pwr-relaunch-n82",
                 "pwr-spin-n4", "pwr-spin-n82", "step-exit", "step-relaunch"):
        print(f"\n-- {name}")
        lines = smid_report(name)
        # The start map is long. Keep ends and every relaunch or spawn.
        for line in lines:
            if line.startswith("[SMID] launch") or line.startswith("[SMID] left") \
                    or line.startswith("[SMID] live") or line.startswith("[SMID] recorded") \
                    or line.startswith("[SMID] launched") or line.startswith("[IDLE] ") \
                    or line.startswith("[RELAUNCH] ") or line.startswith("[SPAWN] "):
                if len(line) > 240:
                    print(line[:180] + " ... " + line[-50:])
                else:
                    print(line)


def report_latency():
    print("\n== latency ==")
    for mode in ("exit", "relaunch"):
        print(f"\nstep-{mode}")
        for line in step_latency(f"step-{mode}"):
            print(line)
        for line in tally_latency(f"step-{mode}", 42, 42):
            print(line)
        print(present_summary(os.path.join(LOG, f"present-{mode}.csv")))


def report_peak():
    print("\n== peak ==")
    path = os.path.join(LOG, "peak-search.txt")
    if os.path.exists(path):
        print(open(path).read().rstrip())
    else:
        print("no peak-search.txt")
    best = os.path.join(LOG, "peak-best.txt")
    if os.path.exists(best):
        print(open(best).read().rstrip())
    print("\n-- peak-hold60 (60 s confirmation)")
    tels = tel_after_load("peak-hold60")
    mid = tels[5:-2] if len(tels) > 8 else tels
    part, note = dmon_slice("peak-hold60", 10.0, 55.0)
    text = read_text("peak-hold60") or ""
    reps = re.findall(r"\[PEAK\] reps=(\d+) chains=(\d+) threads=(\d+)", text)
    r = c = t = 1
    if reps:
        r, c, t = (int(x) for x in reps[-1])
    rate = statistics.fmean(x["rate"] for x in mid) if mid else float("nan")
    flops = rate * OPS_PER_PULSE * r * c if rate == rate else float("nan")
    clk = statistics.fmean(x["pclk"] for x in part) if part else float("nan")
    pwr = stats([x["pwr"] for x in part]) if part else None
    temp = stats([x["temp"] for x in part]) if part else None
    chip = fp32_peak(84, clk) if clk == clk else float("nan")
    launched = fp32_peak(82, clk) if clk == clk else float("nan")
    print(f"reps={r} chains={c} threads={t}")
    print(f"TEL samples in the middle {len(mid)} of {len(tels)} rate={rate:.0f}/s")
    print(f"script product rate*65536*reps*chains={flops:.3e}")
    print("That product is not issued FLOP/s. ptxas shortens the nest; count FFMA in SASS.")
    print(f"dmon {note}")
    print(f"power {fmt(pwr, 1)} temp {fmt(temp, 1)} pclk mean {clk:.0f} MHz")
    print(f"FP32 peak at that clock: 84 SM {chip:.3e}  82 SM {launched:.3e}")
    if len(tels) >= 2:
        d_pulses = tels[-1]["pulses"] - tels[0]["pulses"]
        monotonic = tels[-1]["pulses"] >= tels[0]["pulses"]
        print(f"pulse counter {tels[0]['pulses']} -> {tels[-1]['pulses']} delta={d_pulses} monotonic={monotonic}")
        print(f"64-bit pulse wrap at this rate: {years_to_wrap(rate):.1f} years")
    print("counter width: unsigned long long, 64 bits, wraps, does not saturate")
    hour = read_text("peak-hour")
    if hour:
        ht = tel_after_load("peak-hour")
        print(f"\n-- peak-hour TEL samples {len(ht)}")
        if len(ht) >= 2:
            print(f"pulses {ht[0]['pulses']} -> {ht[-1]['pulses']} monotonic={ht[-1]['pulses'] >= ht[0]['pulses']}")
    else:
        print("\n-- peak-hour not run")
    q = os.path.join(LOG, "peak-hold60-query.txt")
    if os.path.exists(q):
        interesting = []
        for line in open(q):
            if any(k in line for k in ("HW Slowdown", "SW Power", "HW Thermal", "Clock Policy",
                                       "Idle", "SW Thermal", "HW Power")):
                interesting.append(line.rstrip())
        if interesting:
            print("throttle query:")
            for line in interesting[:20]:
                print(" ", line)


def report_machine():
    print("\n== machine ==")
    for name in ("nvidia-smi-start.txt", "nvidia-smi-end.txt", "gpu-start.csv", "gpu-end.csv", "xrandr.txt"):
        path = os.path.join(LOG, name)
        print(f"\n-- {name}")
        if not os.path.exists(path):
            print("MISSING")
            continue
        text = open(path).read()
        if name.startswith("nvidia-smi"):
            keep = []
            grab_next = False
            for line in text.splitlines():
                if line.startswith("GPU ") or "Driver Version" in line or "CUDA Version" in line \
                        or "GPU UUID" in line or "Product Name" in line:
                    keep.append(line)
                if "Processes" in line:
                    grab_next = True
                if grab_next:
                    keep.append(line)
            print("\n".join(keep[:40]))
        else:
            print(text[:800])


def full_report():
    print(f"log {LOG}")
    report_machine()
    report_checks()
    report_clamp()
    report_power()
    report_idle_maps()
    report_latency()
    report_peak()
    # Modelled overflow at the brochure clock, so the hour is not required for the bound.
    brochure_hz = 2617e6
    peak84 = 84 * CUDA_CORES_PER_SM * FLOPS_PER_FMA * brochure_hz
    print("\n== overflow arithmetic (modelled) ==")
    print(f"brochure FP32 peak 84 * 128 * 2 * 2617 MHz = {peak84:.3e} FLOP/s")
    print(f"per-SM at that peak {peak84 / 84:.3e} FLOP/s, 64-bit wrap {years_to_wrap(peak84 / 84):.1f} years")
    print("A 60 s or 1 hour hold cannot wrap either counter at any rate this card can issue.")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    if "--pick-clamp" in sys.argv:
        choice, lines = pick_lines()
        for line in lines:
            print(line)
        print(f"PICK {choice}")
        return
    full_report()


if __name__ == "__main__":
    main()
