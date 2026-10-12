#!/usr/bin/env python3
"""Bin the 10 µs Gold CSVs to 10 ms so the dispatch seat can replot them.

Each output row is one 10 ms bin:
  t_s, n, z_mean, z_min, z_max, z_gt_1_6, rate, rate_t, z_cuda, z_tensor,
  integral, sensor, samples

n and sensor are the last sample in the bin. z_gt_1_6 is how many 10 µs
samples in the bin had z_hat > 1.6. The raw files stay on the 5080.
"""
import csv
import os
from collections import defaultdict

SRC = "/tmp/gputronic-cf/logs/2026-10-12_5080"
DST = "/tmp/gputronic-cf/logs/2026-10-12_5080/traces_10ms"
os.makedirs(DST, exist_ok=True)

FILES = [
    "T1_off_dyno.csv",
    "T1_on_dyno.csv",
    "T2_run.csv",
    "T2_closedloop.csv",
    "T2_step.csv",
    "T2_off.csv",
    "T2_release.csv",
    "T3_transfer_closedloop.csv",
    "T3_episode_closedloop.csv",
    "T5_k0.csv",
    "T5_k1.csv",
    "T5_k2.csv",
    "T5_k4.csv",
]

# Rerun closed loops, if the results/ copies are still the post-fix runs.
RERUN = {
    "/tmp/gputronic-cf/results/gputronic_transfer_closedloop.csv": "T3_transfer_rerun_closedloop.csv",
    "/tmp/gputronic-cf/results/gputronic_episode_closedloop.csv": "T3_episode_rerun_closedloop.csv",
}

HEADER = (
    "t_s,n,z_mean,z_min,z_max,z_gt_1_6,rate,rate_t,"
    "z_cuda,z_tensor,integral,sensor,samples\n"
)


def bin_file(src, dst, dt=0.01):
    bins = []
    cur = None

    def flush(b):
        if not b or b["n"] == 0:
            return
        n = b["n"]
        bins.append((
            b["t0"] + dt * 0.5,
            b["n_last"],
            b["z"] / n,
            b["zmin"],
            b["zmax"],
            b["over"],
            b["rate"] / n,
            b["rate_t"] / n,
            b["zc"] / n,
            b["zt"] / n,
            b["integ"] / n,
            b["sensor"],
            n,
        ))

    with open(src, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t = float(row["time_s"])
            k = int(t / dt)
            if cur is None or k != cur["k"]:
                flush(cur)
                cur = {
                    "k": k,
                    "t0": k * dt,
                    "n": 0,
                    "z": 0.0,
                    "zmin": 1e9,
                    "zmax": -1e9,
                    "over": 0,
                    "rate": 0.0,
                    "rate_t": 0.0,
                    "zc": 0.0,
                    "zt": 0.0,
                    "integ": 0.0,
                    "n_last": 0,
                    "sensor": 0,
                }
            z = float(row["z_hat"])
            cur["n"] += 1
            cur["z"] += z
            if z < cur["zmin"]:
                cur["zmin"] = z
            if z > cur["zmax"]:
                cur["zmax"] = z
            if z > 1.6:
                cur["over"] += 1
            cur["rate"] += float(row["rate_ema"])
            cur["rate_t"] += float(row["rate_tensor_ema"])
            cur["zc"] += float(row["z_cuda"])
            cur["zt"] += float(row["z_tensor"])
            cur["integ"] += float(row["integral"])
            cur["n_last"] = int(float(row["active_sms"]))
            cur["sensor"] = int(float(row["ctrl_sensor"]))
        flush(cur)

    with open(dst, "w") as out:
        out.write(HEADER)
        for row in bins:
            out.write(
                "%.3f,%d,%.5f,%.5f,%.5f,%d,%.1f,%.1f,%.5f,%.5f,%.5f,%d,%d\n"
                % row
            )
    return len(bins)


def main():
    made = []
    for name in FILES:
        src = os.path.join(SRC, name)
        if not os.path.isfile(src):
            print("missing", name)
            continue
        dst = os.path.join(DST, name)
        n = bin_file(src, dst)
        made.append((name, n, os.path.getsize(dst)))
    for src, name in RERUN.items():
        if not os.path.isfile(src):
            print("missing rerun", src)
            continue
        dst = os.path.join(DST, name)
        n = bin_file(src, dst)
        made.append((name, n, os.path.getsize(dst)))
    total = 0
    for name, n, sz in made:
        total += sz
        print(f"{name:40s} bins={n:6d}  {sz/1024:7.1f} KiB")
    print(f"total {total/1024:.1f} KiB  files {len(made)}")


if __name__ == "__main__":
    main()
