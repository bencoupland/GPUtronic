#!/usr/bin/env python3
"""T11: compare the per-SM ops accumulator with the pulse counter.

Input: results/gputronic_gold_smaccum.csv, written when GPUTRONIC_SM_ACCUM=1
or GPUTRONIC_SM_ACCUM_SM=<id> during a trace mode (noise, loadstep, ...).

Per host bin the file has one row per reporting SM. CUDA ops are summed.
A single reporting SM is scaled by active_sms, which is right only if the
SMs are doing the same work. Pulses are bin_pulses x 65,536 ops per CUDA
warp pass. The first second is dropped so the open-loop command can settle.
The host already omits the pre-log cumulative sample.

The device publishes on the pass that crosses 1 ms, so host bins of 1 ms
are often empty. The comparison that counts is 100 ms. Pass there: drift
under 1%, correlation at least 0.9 within 2 bins. When every SM reports,
the ratio mean must also sit within 5% of 1. One SM, scaled by the SM
count, is that SM's share of the chip and may sit away from 1.
Usage:
  python3 scripts/analyze_smaccum.py results/gputronic_gold_smaccum.csv
      [--ops-per-pulse 65536] [--skip-ms 1000]
"""
import csv
import sys
from collections import defaultdict


def corr(x, y):
    n = min(len(x), len(y))
    if n < 3:
        return float("nan")
    x, y = x[:n], y[:n]
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sxy / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def aligned(acc, pul, lag):
    if lag >= 0:
        return acc[lag:], pul[:len(pul) - lag] if lag else pul
    return acc[:len(acc) + lag], pul[-lag:]


def buckets(keys, acc, pul, ms):
    ba, bp = {}, {}
    for k, a, p in zip(keys, acc, pul):
        b = int(float(k) // ms)
        ba[b] = ba.get(b, 0.0) + a
        bp[b] = bp.get(b, 0.0) + p
    order = sorted(ba)
    return [ba[b] for b in order], [bp[b] for b in order]


def report(label, acc, pul):
    ratios = [a / p for a, p in zip(acc, pul) if p > 0]
    if not ratios:
        print(f"[{label}] no bins with pulses")
        return False
    mean = sum(ratios) / len(ratios)
    w = max(1, len(ratios) // 10)
    roll = [sum(ratios[i:i + w]) / len(ratios[i:i + w]) for i in range(0, len(ratios) - w + 1, w)]
    drift = max(abs(r - mean) / mean * 100 for r in roll) if mean else float("nan")
    best_lag, best_c = 0, corr(acc, pul)
    for lag in range(-5, 6):
        c = corr(*aligned(acc, pul, lag))
        if c > best_c:
            best_lag, best_c = lag, c
    print(f"[{label}] bins {len(acc)}: ratio mean {mean:.4f}, drift {drift:.2f}%, "
          f"corr {corr(acc, pul):.3f}, best lag {best_lag} corr {best_c:.3f}")
    return mean, drift, best_c, best_lag


def spread(label, means):
    if not means:
        print(f"[{label}] no rows")
        return
    mu = sum(means) / len(means)
    var = sum((m - mu) ** 2 for m in means) / len(means)
    cv = (var ** 0.5) / mu * 100 if mu else float("nan")
    lo, hi = min(means), max(means)
    print(f"[{label}] n {len(means)}, mean ops/bin {mu:.0f}, cv {cv:.1f}%, "
          f"range {lo:.0f}..{hi:.0f}, max/min {hi / lo if lo else float('nan'):.3f}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    opp = float(args[args.index("--ops-per-pulse") + 1]) if "--ops-per-pulse" in args else 65536.0
    skip_ms = float(args[args.index("--skip-ms") + 1]) if "--skip-ms" in args else 1000.0
    bins = {}
    per_block = defaultdict(lambda: [0.0, 0])
    phys_hist = defaultdict(lambda: defaultdict(int))
    phys_sum = defaultdict(float)
    phys_bins = defaultdict(set)
    with open(args[0], newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["t_ms"])
            if t < skip_ms:
                continue
            slot = bins.setdefault(t, {"acc": 0.0, "nsm": 0, "act": int(row["active_sms"]),
                                       "pul": float(row["bin_pulses"])})
            dops = float(row["d_ops_cuda"])
            slot["acc"] += dops
            slot["nsm"] += 1
            sm = int(row["sm"])
            per_block[sm][0] += dops
            per_block[sm][1] += 1
            if "phys_sm" in row and row["phys_sm"] != "":
                phys = int(row["phys_sm"])
                phys_hist[sm][phys] += 1
                phys_sum[phys] += dops
                phys_bins[phys].add(t)
    keys = sorted(bins)
    if len(keys) < 30:
        print(f"only {len(keys)} bins after {skip_ms:.0f} ms")
        return 1
    # Host bins are about 1 ms. Grouping by 10 uses ten successive rows.
    single = all(bins[k]["nsm"] == 1 for k in keys)
    acc = [bins[k]["acc"] * (bins[k]["act"] if single else 1) for k in keys]
    pul = [bins[k]["pul"] * opp for k in keys]
    print(f"{'single SM scaled by active_sms' if single else 'sum over reporting SMs'}; "
          f"ops per pulse {opp:.0f}; skipped t < {skip_ms:.0f} ms; {len(keys)} bins")
    if not single and per_block:
        active = [(sm, tot / n) for sm, (tot, n) in per_block.items() if n and tot / n > 0]
        idle = sum(1 for tot, n in per_block.values() if n and tot == 0)
        print(f"[blocks] reporting {len(per_block)}, with work {len(active)}, idle {idle}")
        if active:
            spread("blocks", [m for _, m in active])
            ordered = sorted(active)
            biggest = 1.0
            where = None
            for (b0, m0), (b1, m1) in zip(ordered, ordered[1:]):
                ratio = (m1 / m0) if m0 else float("nan")
                flip = (m0 / m1) if m1 else float("nan")
                if ratio > biggest:
                    biggest, where = ratio, (b0, b1, m0, m1)
                if flip > biggest:
                    biggest, where = flip, (b1, b0, m1, m0)
            if where and biggest >= 1.2:
                print(f"[jump] largest step in block order is {biggest:.3f}x "
                      f"between block {where[0]} ({where[2]:.0f}) and block {where[1]} ({where[3]:.0f})")
            else:
                print("[jump] no neighboring blocks differ by 1.2x or more")
            print("[block means] block phys_sm ops_per_bin")
            for sm, mean in ordered:
                hist = phys_hist.get(sm)
                if not hist:
                    phys_s = "-"
                else:
                    mode = max(hist, key=hist.get)
                    extra = "" if len(hist) == 1 else " conflict"
                    phys_s = f"{mode}{extra}"
                print(f"  {sm} {phys_s} {mean:.0f}")
        if phys_hist:
            print("[map] block:phys_sm")
            parts = []
            for sm in sorted(per_block):
                hist = phys_hist.get(sm)
                if not hist:
                    continue
                mode = max(hist, key=hist.get)
                mark = "" if len(hist) == 1 else "*"
                parts.append(f"{sm}:{mode}{mark}")
            print("  " + " ".join(parts))
            occupancy = defaultdict(int)
            for sm, hist in phys_hist.items():
                mode = max(hist, key=hist.get)
                occupancy[mode] += 1
            share = defaultdict(int)
            for n in occupancy.values():
                share[n] += 1
            occ_s = ", ".join(f"{n} blocks on {share[n]} SMs" for n in sorted(share))
            print(f"[placement] {len(occupancy)} physical SMs hold a block ({occ_s})")
            phys_means = [phys_sum[p] / len(phys_bins[p]) for p in phys_bins if phys_sum[p] > 0 and phys_bins[p]]
            spread("phys", phys_means)
    report("1 host bin", acc, pul)
    # The device publishes on the pass that crosses 1 ms, so a publish lands about
    # every 1.2 ms and many 1 ms host bins are empty. 20 ms is wide enough to
    # line the two counters up. 100 ms is the steady comparison.
    for ms in (20, 100):
        grouped = report(f"{ms} ms", *buckets(keys, acc, pul, ms))
    if grouped is False:
        print("T11 FAIL")
        return 2
    mean, drift, best_c, best_lag = grouped
    # One reporting SM, scaled by active_sms, measures that SM's share of the
    # chip. The share can sit away from 1. The sum over every SM is the check
    # that the tally and the pulse counter count the same passes.
    ratio_ok = (0.95 <= mean <= 1.05) or single
    passed = ratio_ok and drift < 1.0 and best_c >= 0.9 and abs(best_lag) <= 2
    if single:
        print(f"[share] this SM, scaled by active_sms, sits at {mean:.3f} of the pulse counter")
    print(f"T11 {'PASS' if passed else 'FAIL'} "
          f"(100 ms drift < 1%, corr >= 0.9 within 2 bins"
          f"{'' if single else ', ratio mean 0.95..1.05'})")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
