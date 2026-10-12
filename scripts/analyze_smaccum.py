#!/usr/bin/env python3
"""T11: compare the per-SM ops accumulator with the pulse counter, bin by bin.

Input: results/gputronic_gold_smaccum.csv (written when GPUTRONIC_SM_ACCUM=1 or
GPUTRONIC_SM_ACCUM_SM=<id> and a trace mode runs: noise, loadfile, loadstep, ...).
Per bin: accumulator CUDA ops (summed over reporting SMs; a single SM is scaled by
active_sms) vs pulses x 65,536 ops per CUDA warp pass. Reports correlation of the
two per-bin rates, the ratio's mean and drift (max |rolling mean - overall| in %),
and the lag (bins) that maximises correlation. Also at 10-bin aggregation.
Pass: ratio drift < 1% and correlation >= 0.9 (at the 10-bin level).
Usage: python3 scripts/analyze_smaccum.py results/gputronic_gold_smaccum.csv [--ops-per-pulse 65536] [--skip-bins 200]
"""
import csv
import sys


def corr(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sxy / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def agg(xs, k):
    return [sum(xs[i:i + k]) for i in range(0, len(xs) - k + 1, k)]


def report(label, acc, pul):
    ratios = [a / p for a, p in zip(acc, pul) if p > 0]
    if not ratios:
        print(f"[{label}] no bins with pulses")
        return False, float("nan"), float("nan")
    m = sum(ratios) / len(ratios)
    w = max(1, len(ratios) // 10)
    roll = [sum(ratios[i:i + w]) / w for i in range(0, len(ratios) - w + 1, w)]
    drift = max(abs(r - m) / m * 100 for r in roll) if m else float("nan")
    best = max(range(-5, 6), key=lambda L: corr(acc[max(0, L):len(acc) + min(0, L)],
                                                 pul[max(0, -L):len(pul) + min(0, -L)]) if len(acc) > 12 else 0)
    c0 = corr(acc, pul)
    print(f"[{label}] bins {len(acc)}: ratio acc/pulse-ops mean {m:.4f}, drift {drift:.2f}%, corr {c0:.3f}, best lag {best} bins")
    return True, drift, c0


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 1
    opp = float(a[a.index("--ops-per-pulse") + 1]) if "--ops-per-pulse" in a else 65536.0
    skip = int(a[a.index("--skip-bins") + 1]) if "--skip-bins" in a else 200
    bins = {}
    with open(a[0], newline="") as f:
        for r in csv.DictReader(f):
            t = r["t_ms"]
            b = bins.setdefault(t, {"acc": 0.0, "nsm": 0, "act": int(r["active_sms"]), "pul": float(r["bin_pulses"])})
            b["acc"] += float(r["d_ops_cuda"])
            b["nsm"] += 1
    keys = sorted(bins, key=float)[skip + 1:]   # drop calibration-adjacent and first-delta bins
    if not keys:
        print("no bins")
        return 1
    single = all(bins[k]["nsm"] == 1 for k in keys)
    acc = [bins[k]["acc"] * (bins[k]["act"] if single else 1) for k in keys]
    pul = [bins[k]["pul"] * opp for k in keys]
    print(f"{'single SM scaled by active_sms' if single else 'sum over reporting SMs'}; ops per pulse {opp:.0f}")
    report("1 bin", acc, pul)
    ok, drift, c = report("10 bins", agg(acc, 10), agg(pul, 10))
    passed = ok and drift < 1.0 and c >= 0.9
    print(f"T11 {'PASS' if passed else 'FAIL'} (drift < 1% and corr >= 0.9 at 10 bins)")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
