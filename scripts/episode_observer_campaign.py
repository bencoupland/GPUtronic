#!/usr/bin/env python3
"""Overnight episode observer / throw campaign. Gold 1.0 is not edited.

Question: can a launch-complete plant self-describe its timescale well enough
that throw (chunked sleep or host wait) reaches Z=1.50 on stock Gold, and does
the 8 ms window still see Z when an episode is as long as the window?

Writes incrementally under results/episode_campaign/<stamp>/ so a kill still
leaves a report.
"""
from __future__ import annotations

import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/media/dad/1TB HDD/GPUTronic Agent")
BIN = ROOT / "build" / "gputronic_episode"
JSON_RE = re.compile(r"^JSON:(\{.*\})\s*$", re.M)

# Plant periods vs 8 ms observer. 1 and 64 are the hard ends.
PULSES = [1, 2, 4, 8, 16, 32, 64]
HONOURS = ["once", "chunk2", "chunk4", "host"]

# Fine sleep: map the 500 µs–2 ms cliff (the 1 ms == 2 ms plateau).
FINE_SLEEPS = [
    0, 50000, 100000, 200000, 300000, 400000, 500000,
    600000, 700000, 800000, 900000, 1000000,
    1200000, 1500000, 2000000,
]
DUR_SLEEPS = [0, 200000, 500000, 1000000, 2000000]


def stamp_dir() -> Path:
    name = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = ROOT / "results" / "episode_campaign" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    return d


def gpu_temp() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def cooldown(log, limit=82.0, resume=74.0):
    t = gpu_temp()
    if t is None or t < limit:
        return t
    log(f"thermal {t:.0f}C >= {limit:.0f}C — waiting")
    while True:
        time.sleep(20)
        t = gpu_temp()
        if t is None or t <= resume:
            log(f"thermal resume {t}")
            return t


def parse_json_lines(text: str) -> list[dict]:
    found = []
    for m in JSON_RE.finditer(text):
        try:
            found.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            continue
    return found


def run_episode(outdir: Path, tag: str, mode: str, extra_args: list[str],
                env_extra: dict, log, timeout=600) -> dict:
    cooldown(log)
    prefix = str(outdir / tag)
    env = os.environ.copy()
    env.update(env_extra)
    env["GPUTRONIC_EPISODE_PREFIX"] = prefix
    cmd = [str(BIN), mode, *extra_args]
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd, cwd=str(ROOT), env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
        )
        rc = p.returncode
        text = p.stdout or ""
    except subprocess.TimeoutExpired as e:
        rc = 124
        raw = e.stdout or b""
        text = (raw.decode() if isinstance(raw, bytes) else raw) + "\nTIMEOUT\n"
    dt = time.time() - t0
    (outdir / "logs" / f"{tag}.log").write_text(text)
    recs = parse_json_lines(text)
    rec = recs[-1] if recs else {}
    rec["_tag"] = tag
    rec["_rc"] = rc
    rec["_dt_s"] = round(dt, 2)
    rec["_temp_c"] = gpu_temp()
    rec["_mode_arg"] = mode
    return rec


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.fmean(xs), statistics.pstdev(xs)


def proposed_window(episode_us: float) -> float:
    """Self-tune candidate: rate window covers ~4 plant periods, clamp [8, 50] ms."""
    if episode_us <= 0:
        return 0.008
    return min(0.050, max(0.008, 4.0 * episode_us * 1e-6))


def write_report(outdir: Path, fine: list[dict], durs: list[dict],
                 loops: list[dict], heroes: list[dict], gates: dict) -> None:
    lines = []
    lines.append("# Episode observer campaign")
    lines.append("")
    lines.append(f"Stamp: {outdir.name}")
    lines.append("Gold 1.0 source was not edited. max_sleep on closed-loop stayed 500 µs.")
    lines.append("")
    lines.append("## Fine sleep (pulses=4, honour=once)")
    lines.append("")
    lines.append("| sleep_ns | rate | GFLOP/s | Z |")
    lines.append("|--------:|-----:|--------:|--:|")
    # fine dyno JSON is one rec; points live in the summary csv
    sum_csv = outdir / "fine_once_dyno_summary.csv"
    if sum_csv.exists():
        with sum_csv.open() as f:
            for row in csv.DictReader(f):
                lines.append(
                    f"| {row['sleep_ns']} | {float(row['tile_rate']):.1f} | "
                    f"{float(row['gflops']):.4f} | {float(row['z_hat']):.3f} |"
                )
    if fine:
        r = fine[0]
        lines.append("")
        lines.append(
            f"mean_episode={r.get('mean_episode_us')} us  "
            f"F-09 period={r.get('f09_period_ns')} ns  "
            f"gold_act={r.get('gold_act')} plant_ok={r.get('plant_ok')}"
        )
        if r.get("mean_episode_us"):
            lines.append(
                f"Proposed rate window = {1e3*proposed_window(float(r['mean_episode_us'])):.1f} ms "
                f"(4× plant period, clamped)."
            )
    lines.append("")
    lines.append("## Duration × open-loop (honour=once)")
    lines.append("")
    lines.append("| pulses | mean_ep_us | F09_ns | rate0 | z0 | z500 | z1ms | window | pulses/8ms |")
    lines.append("|------:|----------:|-------:|------:|----:|-----:|-----:|:------:|----------:|")
    for r in durs:
        p8 = (float(r.get("rate0") or 0) * 0.008)
        lines.append(
            f"| {r.get('pulses')} | {r.get('mean_episode_us')} | {r.get('f09_period_ns')} | "
            f"{r.get('rate0')} | {r.get('z0')} | {r.get('z500')} | {r.get('z2ms')} | "
            f"{r.get('window_ok')} | {p8:.1f} |"
        )
    lines.append("")
    lines.append("F-09 period is `1e9 * num_sm / rate` — pulse interval, not launch period.")
    lines.append("If mean_episode_us >> F09_ns/1000, auto sleep_scale is looking at the wrong clock.")
    lines.append("")
    lines.append("## Closed-loop throw (stock max_sleep=500 µs)")
    lines.append("")
    lines.append("| pulses | honour | n | z_mean± | drop | tight | pass | p/8ms |")
    lines.append("|------:|:-------|--:|--------:|-----:|:-----:|:----:|------:|")
    groups: dict[tuple, list[dict]] = {}
    for r in loops:
        groups.setdefault((r.get("pulses"), r.get("honour")), []).append(r)
    for key in sorted(groups, key=lambda k: (int(k[0] or 0), str(k[1]))):
        rs = groups[key]
        zs = [float(x["z_mean"]) for x in rs if "z_mean" in x]
        drops = []
        for x in rs:
            fr, cl = x.get("fr_gflops"), x.get("cl_gflops")
            if fr and cl and float(fr) > 0:
                drops.append(1.0 - float(cl) / float(fr))
        zm, zsdev = mean_std(zs)
        dm, _ = mean_std(drops)
        tight = sum(1 for x in rs if x.get("track_tight"))
        passed = sum(1 for x in rs if x.get("pass"))
        p8s = [float(x["pulses_per_8ms"]) for x in rs if "pulses_per_8ms" in x]
        p8, _ = mean_std(p8s)
        lines.append(
            f"| {key[0]} | {key[1]} | {len(rs)} | {zm:.3f}±{zsdev:.3f} | "
            f"{100*dm:.1f}% | {tight}/{len(rs)} | {passed}/{len(rs)} | {p8:.1f} |"
        )
    lines.append("")
    lines.append("tight = |z_mean - 1.50| < 0.20. That is the self-tune target, not the soft gate.")
    lines.append("")
    if heroes:
        lines.append("## Long closed-loop hero cells (60 s)")
        lines.append("")
        for r in heroes:
            lines.append(
                f"- pulses={r.get('pulses')} honour={r.get('honour')} "
                f"z_mean={r.get('z_mean')} drop={r.get('cl_gflops')}/{r.get('fr_gflops')} "
                f"sleep={r.get('sleep_ns')} p/8ms={r.get('pulses_per_8ms')}"
            )
        lines.append("")
    lines.append("## Proposed self-tune law (not in Gold 1.0)")
    lines.append("")
    lines.append("1. Measure launch period T from pulse Δ / (sources × pulses_per_launch), or from wall launches.")
    lines.append("2. Rate window W = clamp(4T, 8 ms, 50 ms). Do not form Z on a window shorter than the plant.")
    lines.append("3. Throw: honour `throttle_sleep_ns` N times such that N × cmd reaches ~T, or wait on the host.")
    lines.append("   Stock 500 µs × N is still Gold 1.0 — the plant supplies N. Kp stays.")
    lines.append("4. F-09 scale should use T, not the per-SM pulse interval.")
    lines.append("")
    lines.append("## End-of-night gates")
    lines.append("")
    for name, ok in gates.items():
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
    lines.append("")
    (outdir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    os.chdir(ROOT)
    outdir = stamp_dir()
    status = outdir / "STATUS.txt"

    def log(msg: str) -> None:
        line = f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with status.open("a") as f:
            f.write(line + "\n")

    log(f"campaign dir {outdir}")
    log(f"gpu temp {gpu_temp()}")

    if not BIN.exists():
        log("building episode")
        r = subprocess.run(["make", "episode"], cwd=str(ROOT))
        if r.returncode != 0:
            log("BUILD FAIL")
            return 1

    manifest = {
        "pulses": PULSES,
        "honours": HONOURS,
        "fine_sleeps": FINE_SLEEPS,
        "started": datetime.now(timezone.utc).isoformat(),
        "note": "Gold 1.0 frozen; episode knobs only",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    jsonl = outdir / "records.jsonl"
    fine: list[dict] = []
    durs: list[dict] = []
    loops: list[dict] = []
    heroes: list[dict] = []

    # --- Phase A: cliff map ---
    log("phase A fine sleep sweep")
    rec = run_episode(
        outdir, "fine_once", "dyno", [],
        {
            "GPUTRONIC_EPISODE_PULSES": "4",
            "GPUTRONIC_EPISODE_HONOUR": "once",
            "GPUTRONIC_EPISODE_SLEEPS": ",".join(str(s) for s in FINE_SLEEPS),
        },
        log, timeout=900,
    )
    fine.append(rec)
    append_jsonl(jsonl, rec)
    # copy summary next to report if produced
    src = outdir / "fine_once_dyno_summary.csv"
    log(f"phase A done gold_act={rec.get('gold_act')} z500={rec.get('z500')} z2ms={rec.get('z2ms')}")

    # --- Phase B: duration vs window ---
    log("phase B duration sweep")
    for n in PULSES:
        rec = run_episode(
            outdir, f"dur_p{n}", "dyno", [],
            {
                "GPUTRONIC_EPISODE_PULSES": str(n),
                "GPUTRONIC_EPISODE_HONOUR": "once",
                "GPUTRONIC_EPISODE_SLEEPS": ",".join(str(s) for s in DUR_SLEEPS),
            },
            log, timeout=700,
        )
        durs.append(rec)
        append_jsonl(jsonl, rec)
        log(f"  p={n} ep={rec.get('mean_episode_us')}us z0={rec.get('z0')} z500={rec.get('z500')} win={rec.get('window_ok')}")
        write_report(outdir, fine, durs, loops, heroes, {})

    # --- Phase C: throw × duration, 3 repeats ---
    log("phase C closed-loop throw repeats")
    for rep in range(3):
        for n in [4, 16, 64]:
            for hon in HONOURS:
                tag = f"cl_p{n}_{hon}_r{rep}"
                rec = run_episode(
                    outdir, tag, "closedloop", ["20"],
                    {
                        "GPUTRONIC_EPISODE_PULSES": str(n),
                        "GPUTRONIC_EPISODE_HONOUR": hon,
                        "GPUTRONIC_EPISODE_CL_SEC": "20",
                    },
                    log, timeout=180,
                )
                loops.append(rec)
                append_jsonl(jsonl, rec)
                log(f"  {tag} z_mean={rec.get('z_mean')} pass={rec.get('pass')} tight={rec.get('track_tight')}")
                write_report(outdir, fine, durs, loops, heroes, {})

    # --- Phase D: 60 s hero cells ---
    log("phase D 60s heroes")
    for n, hon in [(4, "once"), (4, "chunk4"), (4, "host"), (16, "chunk4"), (64, "chunk4"), (64, "host")]:
        rec = run_episode(
            outdir, f"hero_p{n}_{hon}", "closedloop", ["60"],
            {
                "GPUTRONIC_EPISODE_PULSES": str(n),
                "GPUTRONIC_EPISODE_HONOUR": hon,
                "GPUTRONIC_EPISODE_CL_SEC": "60",
            },
            log, timeout=240,
        )
        heroes.append(rec)
        append_jsonl(jsonl, rec)
        log(f"  hero p={n} {hon} z_mean={rec.get('z_mean')} pass={rec.get('pass')}")
        write_report(outdir, fine, durs, loops, heroes, {})

    # --- Phase E: extra duration×honour dyno at the long end ---
    log("phase E long-episode dyno with throw")
    for n in [16, 64]:
        for hon in ["once", "chunk4", "host"]:
            rec = run_episode(
                outdir, f"dur_p{n}_{hon}", "dyno", [],
                {
                    "GPUTRONIC_EPISODE_PULSES": str(n),
                    "GPUTRONIC_EPISODE_HONOUR": hon,
                    "GPUTRONIC_EPISODE_SLEEPS": ",".join(str(s) for s in DUR_SLEEPS),
                },
                log, timeout=700,
            )
            append_jsonl(jsonl, rec)
            durs.append(rec)
            write_report(outdir, fine, durs, loops, heroes, {})

    # --- Phase F: regression gates ---
    log("phase F gold/transfer/episode default gates")
    gates = {}
    for name, cmd, tout in [
        ("gold", [str(ROOT / "build" / "gputronic_gold"), "check"], 180),
        ("transfer", [str(ROOT / "build" / "gputronic_transfer"), "check"], 180),
        ("episode_default", [str(BIN), "check"], 180),
    ]:
        cooldown(log)
        try:
            p = subprocess.run(cmd, cwd=str(ROOT), text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=tout)
            text = p.stdout or ""
            ok = p.returncode == 0
        except Exception as e:
            text = str(e)
            ok = False
        (outdir / "logs" / f"gate_{name}.log").write_text(text)
        gates[name] = ok
        log(f"  gate {name}={'PASS' if ok else 'FAIL'}")

    write_report(outdir, fine, durs, loops, heroes, gates)
    fields = ["_tag", "mode", "pulses", "honour", "z_mean", "z0", "z500", "z2ms",
              "fr_gflops", "cl_gflops", "mean_episode_us", "pulses_per_8ms",
              "pass", "track_tight", "gold_act", "window_ok", "_temp_c", "_rc"]
    write_csv(outdir / "all_records.csv",
              [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()],
              fields)

    log(f"DONE report={outdir / 'REPORT.md'}")
    print(outdir)
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
