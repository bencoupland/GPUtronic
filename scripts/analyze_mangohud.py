#!/usr/bin/env python3
"""Frame-time summary from a MangoHud CSV log (T8).

MangoHud logs start with system-info lines, then a header row containing
`frametime` (ms per frame). This script finds that header and reads frametime.
Lows are computed from the slowest frames: 1% low = 1000 / mean of the slowest
1% of frame times (one common definition; state it when reporting).
Usage: python3 scripts/analyze_mangohud.py LOG.csv [LOG2.csv ...] [--skip-s 0]
"""
import csv
import math
import sys


def frametimes(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    for k, r in enumerate(rows):
        cols = [c.strip() for c in r]
        if "frametime" in cols:
            j = cols.index("frametime")
            out = []
            for rr in rows[k + 1:]:
                try:
                    v = float(rr[j])
                    if v > 0:
                        out.append(v)
                except (ValueError, IndexError):
                    pass
            return out
    raise ValueError(f"{path}: no 'frametime' column found")


def low(ft, pct):
    n = max(1, int(math.ceil(len(ft) * pct / 100.0)))
    worst = sorted(ft, reverse=True)[:n]
    return 1000.0 / (sum(worst) / n)


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 1
    skip = float(a[a.index("--skip-s") + 1]) if "--skip-s" in a else 0.0
    files = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--skip-s")]
    print("file,frames,avg_fps,low_1pct_fps,low_0.1pct_fps,frametime_mean_ms,frametime_stdev_ms,p99_ms")
    for p in files:
        ft = frametimes(p)
        t, k = 0.0, 0
        while k < len(ft) and t < skip * 1000.0:
            t += ft[k]
            k += 1
        ft = ft[k:]
        m = sum(ft) / len(ft)
        sd = math.sqrt(sum((x - m) ** 2 for x in ft) / len(ft))
        p99 = sorted(ft)[min(len(ft) - 1, int(0.99 * len(ft)))]
        print(f"{p},{len(ft)},{1000.0 / m:.2f},{low(ft, 1):.2f},{low(ft, 0.1):.2f},{m:.3f},{sd:.3f},{p99:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
