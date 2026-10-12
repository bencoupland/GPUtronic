#!/usr/bin/env python3
"""T9 pulse-to-work ratio check.

Input: results/gputronic_gold_ratio.csv from `gputronic_gold ratio`.
Pass if every SM count's ratio is within --tol-pct (default 1.0) of the
first (full-width) row.
Usage: python3 scripts/analyze_ratio.py results/gputronic_gold_ratio.csv [--tol-pct 1.0]
"""
import csv
import sys


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 1
    tol = float(a[a.index("--tol-pct") + 1]) if "--tol-pct" in a else 1.0
    rows = list(csv.DictReader(open(a[0], newline="")))
    if not rows:
        print("no rows")
        return 1
    r0 = float(rows[0]["ratio"])
    worst = 0.0
    print("active_sms,pulses,work_batches,ratio,drift_pct")
    for r in rows:
        ratio = float(r["ratio"])
        drift = 100.0 * (ratio - r0) / r0 if r0 else float("nan")
        worst = max(worst, abs(drift))
        print(f"{r['active_sms']},{r['pulses']},{r['work_batches']},{ratio:.5f},{drift:+.3f}")
    ok = worst <= tol
    print(f"max |drift| {worst:.3f}% vs tolerance {tol}% -> {'PASS' if ok else 'FAIL: pulses do not track work; revisit rate_ref / Z'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
