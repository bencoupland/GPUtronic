#!/usr/bin/env python3
"""Cheap readers for the fifteenth-pass suite. No wave analysis, no numpy."""
import csv
import gzip
import os
import statistics
import sys

RATIO = 601000.0 / 580000.0
HIGH_PS = 598000.0


def med(xs):
    if not xs:
        return None
    return statistics.median(xs)


def rounds_of(base, milli):
    return (base * milli + 500) // 1000


def ops_of(rate, acc, rounds):
    return rate * acc * rounds * 64.0


def load_rows(path, t0, t1):
    rate, sms, scale, zraw = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["t_ms"])
            if t < t0 or t >= t1:
                continue
            rate.append(float(row["rate_ema"]))
            sms.append(int(row["active_sms"]))
            scale.append(float(row["scale"]))
            zraw.append(float(row["z_raw"]))
    return rate, sms, scale, zraw


def launches(log):
    out = {}
    with open(os.path.join(log, "launches.tsv")) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["name"]] = row
    return out


def points(log):
    with open(os.path.join(log, "points.tsv")) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def measure(log, pt, drop_ms):
    path = os.path.join(log, pt["name"] + "-1ms.csv")
    if not os.path.isfile(path):
        return None
    t0 = float(pt["t0_ms"]) + drop_ms
    t1 = float(pt["t1_ms"])
    if t1 <= t0:
        return None
    rate, sms, scale, zraw = load_rows(path, t0, t1)
    if len(rate) < 50:
        return None
    acc = int(pt["acc"])
    rounds = int(pt["rounds"])
    rmed = med(rate)
    smed = med(sms)
    if not smed:
        return None
    ops = ops_of(rmed, acc, rounds)
    return {
        "name": pt["name"],
        "tag": pt["tag"],
        "acc": acc,
        "rounds": rounds,
        "threads": int(pt["threads"]),
        "sms": int(pt["sms"]),
        "milli": int(pt["scale_milli"]),
        "rate": rmed,
        "active": smed,
        "ps": rmed / smed,
        "ops": ops,
        "ops_ps": ops / smed,
        "n": len(rate),
        "zsd": statistics.pstdev(zraw) if len(zraw) > 1 else 0.0,
    }


def phase_of(log):
    return {k: v["phase"] for k, v in launches(log).items()}


def collect(log, phases, tags, drop_ms=2000):
    ph = phase_of(log)
    rows = []
    for pt in points(log):
        if pt["tag"] not in tags:
            continue
        if ph.get(pt["name"]) not in phases:
            continue
        m = measure(log, pt, drop_ms)
        if m:
            m["phase"] = ph[pt["name"]]
            rows.append(m)
    return rows


def ramp_bins(log):
    """0.025-wide scale bins on the open-loop and closed-loop ramps."""
    ph = phase_of(log)
    rows = []
    for pt in points(log):
        if pt["tag"] not in ("ramp-up", "ramp-down"):
            continue
        if ph.get(pt["name"]) != "item4":
            continue
        path = os.path.join(log, pt["name"] + "-1ms.csv")
        if not os.path.isfile(path):
            continue
        t0 = float(pt["t0_ms"])
        t1 = float(pt["t1_ms"])
        rate, sms, scale, zraw = load_rows(path, t0, t1)
        buckets = {}
        base = int(pt["rounds"])
        acc = int(pt["acc"])
        for r, s, sc, z in zip(rate, sms, scale, zraw):
            milli = int(sc * 1000.0 + 0.5)
            b = int(sc / 0.025) * 0.025
            # keep the key stable
            key = round(b + 1e-9, 3)
            buckets.setdefault(key, []).append((r, s, milli, z))
        for key, items in buckets.items():
            if len(items) < 20:
                continue
            rs = [it[0] for it in items]
            ss = [it[1] for it in items]
            ms = [it[2] for it in items]
            rmed = med(rs)
            smed = med(ss)
            milli = int(med(ms))
            rounds = rounds_of(base, milli)
            ops = ops_of(rmed, acc, rounds)
            rows.append({
                "name": pt["name"],
                "tag": pt["tag"],
                "phase": "item4",
                "acc": acc,
                "rounds": rounds,
                "threads": int(pt["threads"]),
                "sms": int(med(ss)),
                "milli": milli,
                "rate": rmed,
                "active": smed,
                "ps": rmed / smed if smed else 0.0,
                "ops": ops,
                "ops_ps": ops / smed if smed else 0.0,
                "n": len(items),
                "bin": key,
            })
    return rows


def write_choice(path, fields):
    with open(path, "w") as f:
        for k, v in fields.items():
            f.write(f"{k}={v}\n")


def cmd_ceiling(log):
    phases = {"item2", "threads", "chains", "sms", "grid"}
    rows = collect(log, phases, {"sweep", "base"})
    if not rows:
        sys.exit("ceiling: no measured points")
    best = max(rows, key=lambda r: r["ops"])
    write_choice(os.path.join(log, "ceiling-choice.txt"), {
        "acc": best["acc"],
        "rounds": best["rounds"],
        "threads": best["threads"],
        "sms": 82,
        "source_sms": best["sms"],
        "ops": f"{best['ops']:.6e}",
        "tflops": f"{best['ops'] / 1e12:.6f}",
        "ps": f"{best['ps']:.1f}",
        "source": best["name"],
        "source_tag": best["tag"],
        "source_rounds": best["rounds"],
        "n_points": len(rows),
    })
    print(
        f"ceiling {best['name']} acc={best['acc']} rounds={best['rounds']} "
        f"threads={best['threads']} sms={best['sms']} ops={best['ops']:.6e} "
        f"tflops={best['ops']/1e12:.3f}"
    )


def cmd_confirm(log):
    early = {"item2", "threads", "chains", "sms", "grid"}
    bases = [r for r in collect(log, early, {"base"}) if r["ps"] > 0]
    if not bases:
        sys.exit("confirm: no base brackets")
    ref_ps = med([r["ps"] for r in bases])
    ref_ops = med([r["ops_ps"] for r in bases])
    # The 49-million group is the per-SM pulse plant at the base workload
    # (about 601k), including a base bracket that steps up after a ramp.
    # A higher ops/s from more FMAs per pulse is item 2's question, not this
    # group. A shorter pass also raises the pulse rate without being that plant.
    plant = []
    for r in collect(log, early | {"item4"}, {"base"}):
        base_like = (
            r["acc"] == 32 and r["rounds"] == 512 and r["threads"] == 512
            and r["sms"] == 82 and r["milli"] == 1000
        )
        if base_like and r["ps"] >= HIGH_PS:
            r["why"] = "base-bracket"
            plant.append(r)
    for r in ramp_bins(log):
        if r["rounds"] == 512 and r["ps"] >= HIGH_PS and abs(r["milli"] - 1000) <= 12:
            r["why"] = "ramp-base"
            plant.append(r)
    ops_hits = [
        r for r in collect(log, early | {"item4"}, {"base", "sweep"})
        if r["ops_ps"] >= ref_ops * RATIO
    ]
    best_base = max(bases, key=lambda r: r["ps"])
    fields = {
        "ref_ps": f"{ref_ps:.1f}",
        "ref_ops_ps": f"{ref_ops:.6e}",
        "n_bases": len(bases),
        "n_plant": len(plant),
        "n_ops_ratio": len(ops_hits),
        "best_base_ps": f"{best_base['ps']:.1f}",
        "best_base_source": best_base["name"],
    }
    if plant:
        # Prefer a real base-bracket jump. Otherwise the highest per-SM pulse.
        bracket = [r for r in plant if r["why"] == "base-bracket"]
        pick = max(bracket or plant, key=lambda r: r["ps"])
        fields.update({
            "mode": "high",
            "acc": pick["acc"],
            "rounds": pick["rounds"],
            "threads": pick["threads"],
            "why": pick["why"],
            "source": pick["name"],
            "ps": f"{pick['ps']:.1f}",
            "reason": "a point cleared the 49-million per-SM pulse test",
        })
    else:
        fields.update({
            "mode": "labelled",
            "acc": 32,
            "rounds": 512,
            "threads": 512,
            "why": "no-plant-jump",
            "source": best_base["name"],
            "ps": f"{best_base['ps']:.1f}",
            "reason": "no point reached the 49-million per-SM plant; labelled pair at the base",
        })
    write_choice(os.path.join(log, "confirm-choice.txt"), fields)
    print(
        f"confirm mode={fields['mode']} why={fields['why']} source={fields['source']} "
        f"ref_ps={fields['ref_ps']} plant={fields['n_plant']} ops_ratio={fields['n_ops_ratio']}"
    )


def cmd_verify_smoke(log):
    path = os.path.join(log, "sched-smoke-1ms.csv")
    txt = open(os.path.join(log, "sched-smoke.txt"), errors="replace").read()
    if "[SCHED]" not in txt:
        sys.exit("smoke: no [SCHED] line")
    if "3 points" not in txt:
        sys.exit("smoke: point count was not 3")
    checks = [
        (500, 2500, 82, 1.0),
        (3500, 5500, 40, 1.2),
        (6500, 8500, 82, 1.0),
    ]
    for t0, t1, want_sms, want_scale in checks:
        rate, sms, scale, _ = load_rows(path, t0, t1)
        if len(sms) < 20:
            sys.exit(f"smoke: short window {t0}")
        if med(sms) != want_sms:
            sys.exit(f"smoke: sms {med(sms)} != {want_sms} in {t0}")
        if abs(med(scale) - want_scale) > 0.002:
            sys.exit(f"smoke: scale {med(scale)} != {want_scale} in {t0}")
    print("smoke ok")


def cmd_cut_tick(src, dst, manifest, name):
    raw = 0
    kept = 0
    prev = None
    with open(src, newline="") as f, gzip.open(dst, "wt", newline="") as g:
        header = f.readline()
        if not header:
            sys.exit(f"cut: empty {src}")
        cols = header.strip().split(",")
        try:
            iz = cols.index("z_raw")
            ir = cols.index("rate_ema")
            ia = cols.index("active_sms")
            it = cols.index("time_s")
        except ValueError:
            sys.exit(f"cut: header {header.strip()}")
        g.write(header)
        for line in f:
            raw += 1
            parts = line.rstrip("\n").split(",")
            if len(parts) <= max(iz, ir, ia, it):
                continue
            t = float(parts[it])
            key = (parts[iz], parts[ir], parts[ia])
            if t <= 3.0 or key != prev:
                g.write(line if line.endswith("\n") else line + "\n")
                kept += 1
            prev = key
    # read the gzip back before the caller deletes the raw file
    n = 0
    with gzip.open(dst, "rt") as g:
        for _ in g:
            n += 1
    if n != kept + 1:
        sys.exit(f"cut: readback {n} != {kept + 1}")
    nbytes = os.path.getsize(dst)
    with open(manifest, "a") as mf:
        mf.write(f"{name}\traw={raw}\tkept={kept}\tgzip={nbytes}\n")
    print(f"cut {name} raw={raw} kept={kept} gzip={nbytes}")


def cmd_accum(src, dst):
    first = {}
    last = {}
    t_first = None
    t_last = None
    with open(src, newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["t_ms"])
            if t < 2000:
                continue
            sm = int(row["sm"])
            ops = int(float(row["ops_cuda"]))
            if sm not in first:
                first[sm] = ops
                if t_first is None:
                    t_first = t
            last[sm] = ops
            t_last = t
    if not last or t_first is None or t_last is None or t_last <= t_first:
        sys.exit("accum: no rows")
    dt = (t_last - t_first) / 1000.0
    rates = []
    finals = []
    for sm, ops in sorted(last.items()):
        finals.append(ops)
        rates.append((ops - first[sm]) / dt)
    finals.sort()
    rates.sort()
    def q(xs):
        return xs[0], statistics.median(xs), xs[-1]
    fmin, fmed, fmax = q(finals)
    rmin, rmed, rmax = q(rates)
    with open(dst, "w") as out:
        out.write(
            f"sms={len(finals)} dt_s={dt:.3f} "
            f"final_min={fmin:.6e} final_med={fmed:.6e} final_max={fmax:.6e} "
            f"final_max_over_min={fmax / fmin if fmin else 0:.6f}\n"
        )
        out.write(
            f"rate_min={rmin:.6e} rate_med={rmed:.6e} rate_max={rmax:.6e} "
            f"rate_max_over_min={rmax / rmin if rmin else 0:.6f}\n"
        )
    print(open(dst).read().strip())


def main():
    cmd = sys.argv[1]
    if cmd == "ceiling":
        cmd_ceiling(sys.argv[2])
    elif cmd == "confirm":
        cmd_confirm(sys.argv[2])
    elif cmd == "verify-smoke":
        cmd_verify_smoke(sys.argv[2])
    elif cmd == "cut-tick":
        cmd_cut_tick(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "accum":
        cmd_accum(sys.argv[2], sys.argv[3])
    else:
        sys.exit(f"unknown {cmd}")


if __name__ == "__main__":
    main()
