#!/usr/bin/env python3
"""Analyze a GPUTronic Gold load test (T6 loadstep, T7 bursts, loadfile).

Input: the 10 ms bin CSV written by `gputronic_gold loadstep|bursts|loadfile`
(columns wall_s,t_ms,scale,z_raw,z_hat,active_sms,integral,clamp,rate_ema,sensor).

Usage: python3 scripts/analyze_loadstep.py results/gputronic_gold_loadstep_10ms.csv [--target 1.5] [--tol 0.02] [--recover-s 2]
Pure Python, no third-party modules.
"""
import csv
import sys


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "t": float(r["t_ms"]), "scale": float(r["scale"]),
                    "z": float(r["z_hat"]), "zr": float(r["z_raw"]),
                    "n": int(float(r["active_sms"])), "i": float(r["integral"]),
                    "c": float(r["clamp"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def segments(rows):
    """List of (start_ms, end_ms, scale) for every run of constant scale != 1."""
    segs, cur = [], None
    for r in rows:
        on = abs(r["scale"] - 1.0) > 1e-6
        if on and cur is None:
            cur = [r["t"], r["t"], r["scale"]]
        elif on:
            cur[1] = r["t"]
        elif cur is not None:
            segs.append((cur[0], r["t"], cur[2]))
            cur = None
    if cur is not None:
        segs.append((cur[0], rows[-1]["t"] + 10.0, cur[2]))
    return segs


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def recovery(rows, t_from, target, tol):
    """ms after t_from until z stays within tol of target for the rest of rows, else None."""
    after = [r for r in rows if r["t"] >= t_from]
    for k in range(len(after)):
        if all(abs(r["z"] - target) <= tol for r in after[k:]):
            return after[k]["t"] - t_from
    return None


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 1
    path = a[0]
    target = float(a[a.index("--target") + 1]) if "--target" in a else 1.5
    tol = float(a[a.index("--tol") + 1]) if "--tol" in a else 0.02
    rec_s = float(a[a.index("--recover-s") + 1]) if "--recover-s" in a else 2.0
    rows = load(path)
    if not rows:
        print("no rows")
        return 1
    segs = segments(rows)
    at_clamp = mean([1.0 if r["c"] > 0 and abs(r["i"]) >= 0.999 * r["c"] else 0.0 for r in rows])
    hist = {}
    for r in rows:
        hist[r["n"]] = hist.get(r["n"], 0) + 1
    print(f"file {path}: {len(rows)} bins, target {target}, tol {tol}")
    print(f"integral at clamp: {100 * at_clamp:.1f}% of bins (clamp {rows[-1]['c']:.3f})")
    top = sorted(hist.items(), key=lambda kv: -kv[1])[:8]
    print("SM-count distribution (top): " + ", ".join(f"{n}:{100 * c / len(rows):.1f}%" for n, c in top))
    long_segs = [s for s in segs if s[1] - s[0] >= 1000.0]
    ok = True
    if long_segs:  # T6: sustained step(s)
        for s0, s1, sc in long_segs:
            pre = [r["z"] for r in rows if s0 - 5000 <= r["t"] < s0]
            on = [r for r in rows if s0 <= r["t"] < s1]
            rec = recovery(on, s0, target, tol)
            clamp_on = mean([1.0 if r["c"] > 0 and abs(r["i"]) >= 0.999 * r["c"] else 0.0 for r in on])
            tail = mean([r["z"] for r in on[-200:]])
            res = "never" if rec is None else f"{rec / 1000:.2f} s"
            passed = rec is not None and rec <= rec_s * 1000
            ok &= passed
            print(f"STEP x{sc:.2f} at {s0 / 1000:.1f}s: Z before {mean(pre):.3f}, peak {max(r['z'] for r in on):.3f}, "
                  f"recovered {res}, Z at end {tail:.3f}, at clamp {100 * clamp_on:.0f}% -> "
                  f"{'PASS' if passed else ('FAIL (parked at clamp: reproduces the bug)' if clamp_on > 0.5 else 'FAIL')}")
    short = [s for s in segs if s[1] - s[0] < 1000.0]
    if short:  # T7: bursts
        print("burst_ms,scale,peak_z,ms_above_target,sms_moved")
        by_len = {}
        for s0, s1, sc in short:
            win = [r for r in rows if s0 - 50 <= r["t"] <= s1 + 300]
            base = [r["n"] for r in rows if s0 - 50 <= r["t"] < s0] or [win[0]["n"]]
            moved = any(r["n"] != base[-1] for r in win)
            above = 10.0 * sum(1 for r in win if r["z"] > target)
            d = round(s1 - s0)
            by_len.setdefault(d, []).append((max(r["z"] for r in win), above, moved))
            print(f"{d},{sc:.2f},{max(r['z'] for r in win):.3f},{above:.0f},{int(moved)}")
        for d in sorted(by_len):
            v = by_len[d]
            print(f"summary {d} ms bursts: n={len(v)} mean peak Z {mean([x[0] for x in v]):.3f}, "
                  f"controller moved SMs in {sum(x[2] for x in v)}/{len(v)}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
