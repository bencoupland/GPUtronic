#!/usr/bin/env python3
"""Present-interval summary for the spare-SM run. Drops the opening half-second."""
import csv
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(ROOT, "logs", "2026-10-13_spare-sm")


def load(path, dt_col, t_col=None):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    dts = []
    t = 0.0
    for row in rows:
        dt = float(row[dt_col])
        if t_col and row.get(t_col):
            t = float(row[t_col])
            if t < 500.0:
                continue
        else:
            t += dt
            if t < 500.0:
                continue
        if dt <= 0:
            continue
        dts.append(dt)
    return dts


def summarize(dts):
    if not dts:
        return "no frames"
    dts_s = sorted(dts)
    def pct(p):
        return dts_s[min(len(dts_s) - 1, int(round(p * (len(dts_s) - 1))))]
    over20 = sum(1 for d in dts if d > 20.0)
    over33 = sum(1 for d in dts if d > 33.4)
    return ("n=%d median=%.2f p99=%.2f max=%.2f min=%.2f over20ms=%d over33ms=%d"
            % (len(dts), pct(0.5), pct(0.99), dts_s[-1], dts_s[0], over20, over33))


def main():
    for reserve in (0, 1, 2, 4):
        name = f"r{reserve}"
        print(f"\n== reserve {reserve} launch {84 - reserve} ==")
        gold = os.path.join(LOG, f"gold-{name}.txt")
        if os.path.exists(gold):
            with open(gold) as f:
                for line in f:
                    if (line.startswith("[GPU] ") or line.startswith("[SMID] left")
                            or line.startswith("[SMID] recorded") or line.startswith("[LOOP]")):
                        print(line.rstrip())
        present = load(os.path.join(LOG, f"present-{name}.csv"), "dt_ms", "t_ms")
        browser = load(os.path.join(LOG, f"browser-{name}.csv"), "dt_ms")
        print("window ", summarize(present) if present is not None else "MISSING")
        print("browser", summarize(browser) if browser is not None else "MISSING")
        bpath = os.path.join(LOG, f"browser-{name}.csv")
        if os.path.exists(bpath):
            with open(bpath) as f:
                row = next(csv.DictReader(f), None)
            if row:
                print(f"browser window {row.get('w')}x{row.get('h')} "
                      f"screen {row.get('sw')}x{row.get('sh')} hidden={row.get('hidden')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
