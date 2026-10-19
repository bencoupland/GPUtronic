#!/usr/bin/env python3
"""Build open-loop schedules for the fifteenth pass.

Scale strings are checked against the same milli conversion the reader uses:
(unsigned)(atof(scale) * 1000 + 0.5). Rounds are the kernel formula
(base * milli + 500) / 1000.
"""
import os
import subprocess
import sys
import tempfile

BASE_ACC = 32
BASE_ROUNDS = 512
BASE_THREADS = 512
BASE_SMS = 82

ITEM2 = [
    1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384,
    410, 422, 435, 448, 461, 474, 486, 499, 512, 525, 538,
    550, 563, 576, 589, 602, 614, 640, 768, 896, 1024, 1280,
    1536, 2048, 2560, 3072, 4096,
]
STAIR = [
    410, 422, 435, 448, 461, 474, 486, 499, 512, 525, 538,
    550, 563, 576, 589, 602, 614,
]
SMS_ASC = (
    [2, 4, 6, 8]
    + list(range(12, 81, 4))
    + [57, 82]
)


def rounds_of(base, milli):
    return (base * milli + 500) // 1000


def milli_for(rounds, base=BASE_ROUNDS):
    approx = int(round(rounds * 1000 / base))
    found = []
    for m in range(max(1, approx - 8), approx + 9):
        if rounds_of(base, m) == rounds:
            found.append(m)
    if not found:
        raise SystemExit(f"no scale_milli for rounds {rounds} at base {base}")
    target = rounds * 1000 / base
    return min(found, key=lambda m: (abs(m - target), m))


def scale_str(milli):
    return f"{milli / 1000.0:.6f}"


def verify_scales(pairs):
    """pairs: (milli, scale_str). Uses C double, the reader's conversion."""
    src = r"""
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    char buf[128];
    unsigned int m, bad = 0;
    while (scanf("%u %127s", &m, buf) == 2) {
        double sc = atof(buf);
        unsigned int got = (unsigned int)(sc * 1000.0 + 0.5);
        if (got != m) {
            printf("MISS %u %s -> %u\n", m, buf, got);
            bad++;
        }
    }
    return bad ? 1 : 0;
}
"""
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "rt.c")
        bpath = os.path.join(d, "rt")
        with open(cpath, "w") as f:
            f.write(src)
        subprocess.check_call(["cc", "-O2", "-o", bpath, cpath], stdout=subprocess.DEVNULL)
        text = "".join(f"{m} {s}\n" for m, s in pairs)
        r = subprocess.run([bpath], input=text, text=True, capture_output=True)
        if r.returncode != 0:
            sys.stderr.write(r.stdout)
            sys.stderr.write(r.stderr)
            raise SystemExit("scale round-trip failed")


class Plan:
    def __init__(self, out):
        self.out = out
        self.sched = os.path.join(out, "sched")
        os.makedirs(self.sched, exist_ok=True)
        self.launches = []
        self.points = []
        self.pairs = []

    def write_sched(self, name, rows, two_col=False):
        path = os.path.join(self.sched, name + ".sched")
        with open(path, "w") as f:
            f.write("# t_ms scale active_sms\n" if not two_col else "# t_ms scale\n")
            prev = -1.0
            for t_ms, milli, sms in rows:
                if t_ms <= prev + 1e-9:
                    raise SystemExit(f"{name} times not increasing at {t_ms}")
                prev = t_ms
                s = scale_str(milli)
                self.pairs.append((milli, s))
                if two_col:
                    f.write(f"{t_ms:.3f} {s}\n")
                else:
                    f.write(f"{t_ms:.3f} {s} {sms}\n")
        return path

    def launch(self, name, phase, kind, sec, acc, rounds, threads, open_sms, extra, sched):
        self.launches.append(
            (name, phase, kind, sec, acc, rounds, threads, open_sms, extra, sched)
        )

    def point(self, name, t0, t1, acc, rounds, threads, sms, milli, tag):
        self.points.append(
            (name, t0, t1, acc, rounds, threads, sms, milli, tag)
        )

    def save(self):
        with open(os.path.join(self.out, "launches.tsv"), "w") as f:
            f.write("name\tphase\tkind\tsec\tacc\trounds\tthreads\topen_sms\textra\tsched\n")
            for row in self.launches:
                f.write("\t".join(str(x) for x in row) + "\n")
        with open(os.path.join(self.out, "points.tsv"), "w") as f:
            f.write("name\tt0_ms\tt1_ms\tacc\trounds\tthreads\tsms\tscale_milli\ttag\n")
            for row in self.points:
                f.write("\t".join(str(x) for x in row) + "\n")


def tag_for(acc, rounds, threads, sms, milli):
    if (
        acc == BASE_ACC
        and rounds == BASE_ROUNDS
        and threads == BASE_THREADS
        and sms == BASE_SMS
        and milli == 1000
    ):
        return "base"
    return "sweep"


def add_plain(plan, name, phase, kind, sec, acc, rounds, threads, sms, tag=None):
    plan.launch(name, phase, kind, sec, acc, rounds, threads, sms, "-", "-")
    plan.point(
        name, 0, sec * 1000, acc, rounds, threads, sms, 1000,
        tag or tag_for(acc, rounds, threads, sms, 1000),
    )


def add_chunk(plan, name, phase, spec, point_s, base_s, acc, threads, peak_rounds):
    """spec: list of (rounds, sms). Opens and closes on the base point."""
    rows = []
    t = 0
    rows.append((t, 1000, BASE_SMS))
    plan.point(name, t, t + base_s * 1000, acc, peak_rounds, threads, BASE_SMS, 1000,
               tag_for(acc, peak_rounds, threads, BASE_SMS, 1000))
    t += base_s * 1000
    for rounds, sms in spec:
        milli = 1000 if rounds == peak_rounds else milli_for(rounds, peak_rounds)
        if rounds_of(peak_rounds, milli) != rounds:
            raise SystemExit(f"{name} rounds {rounds} milli {milli}")
        rows.append((t, milli, sms))
        plan.point(
            name, t, t + point_s * 1000, acc, rounds, threads, sms, milli,
            tag_for(acc, rounds, threads, sms, milli),
        )
        t += point_s * 1000
    rows.append((t, 1000, BASE_SMS))
    plan.point(name, t, t + base_s * 1000, acc, peak_rounds, threads, BASE_SMS, 1000,
               tag_for(acc, peak_rounds, threads, BASE_SMS, 1000))
    t += base_s * 1000
    if t % 1000 != 0:
        raise SystemExit(f"{name} duration not whole seconds")
    path = plan.write_sched(name, rows)
    plan.launch(name, phase, "full", t // 1000, acc, peak_rounds, threads, BASE_SMS, "-", path)


def add_ramp(plan, name, phase, leg_s, kind, extra, two_col):
    """10 s base, 400 milli-steps up, 400 milli-steps down, 10 s base.

    Milli 800 is the first up point and is held for one step. Milli 1200 is
    the last up point, at the end of the up leg, and is held for one step
    counted with the up leg. The way down runs 1199 to 801. The closing base
    takes the final timestamp, so milli 800 is not held again on the way down.
    """
    step = (leg_s * 1000) / 400.0
    up0 = 10000.0
    up1 = up0 + leg_s * 1000
    down1 = up1 + leg_s * 1000
    end = down1 + 10000.0
    rows = [(0.0, 1000, BASE_SMS)]
    for i in range(401):
        rows.append((up0 + i * step, 800 + i, BASE_SMS))
    for i in range(1, 400):
        rows.append((up1 + i * step, 1200 - i, BASE_SMS))
    rows.append((down1, 1000, BASE_SMS))
    base_tag = "cl-base" if kind == "load" else "base"
    plan.point(name, 0, 10000, BASE_ACC, BASE_ROUNDS, BASE_THREADS, BASE_SMS, 1000, base_tag)
    # 1200 is held for one step after up1; count that step with the up leg.
    plan.point(name, up0, up1 + step, BASE_ACC, BASE_ROUNDS, BASE_THREADS, BASE_SMS, 1000, "ramp-up")
    plan.point(name, up1 + step, down1, BASE_ACC, BASE_ROUNDS, BASE_THREADS, BASE_SMS, 1000, "ramp-down")
    plan.point(name, down1, end, BASE_ACC, BASE_ROUNDS, BASE_THREADS, BASE_SMS, 1000, base_tag)
    path = plan.write_sched(name, rows, two_col=two_col)
    plan.launch(
        name, phase, kind, int(end // 1000), BASE_ACC, BASE_ROUNDS, BASE_THREADS, BASE_SMS, extra, path
    )


def build(out):
    if len(ITEM2) != 40:
        raise SystemExit(f"item 2 has {len(ITEM2)} points")
    if len(STAIR) != 17:
        raise SystemExit(f"stair has {len(STAIR)} levels")
    if len(SMS_ASC) != 24:
        raise SystemExit(f"sms list has {len(SMS_ASC)}")
    if rounds_of(512, 800) != 410 or rounds_of(512, 1200) != 614:
        raise SystemExit("ramp endpoints do not land on 410 and 614")
    if rounds_of(512, 1000) != 512:
        raise SystemExit("base milli is not 512 rounds")
    for r in ITEM2:
        m = 1000 if r == 512 else milli_for(r)
        if rounds_of(512, m) != r:
            raise SystemExit(f"formula miss {r}")
    for r, expect_m_near in zip(STAIR, [round(x * 1000) for x in [i * 0.025 + 0.8 for i in range(17)]]):
        m = 1000 if r == 512 else milli_for(r)
        if abs(m - expect_m_near) > 2 and r != 512:
            # milli_for picks the integer that hits the formula, which may
            # differ by 1 from the naive tenth. The rounds check above holds.
            pass
    plan = Plan(out)

    # Switch smoke. Not a sweep point.
    rows = [(0, 1000, 82), (3000, 1200, 40), (6000, 1000, 82)]
    path = plan.write_sched("sched-smoke", rows)
    plan.launch("sched-smoke", "smoke", "full", 9, BASE_ACC, BASE_ROUNDS, BASE_THREADS, BASE_SMS, "-", path)
    plan.point("sched-smoke", 0, 3000, BASE_ACC, 512, BASE_THREADS, 82, 1000, "smoke")
    plan.point("sched-smoke", 3000, 6000, BASE_ACC, 614, BASE_THREADS, 40, 1200, "smoke")
    plan.point("sched-smoke", 6000, 9000, BASE_ACC, 512, BASE_THREADS, 82, 1000, "smoke")

    add_plain(plan, "floor-fw", "floor", "full", 15, 4, 1, 32, 82, "floor")
    add_plain(plan, "floor-n2", "floor", "noise", 15, 4, 1, 32, 2, "floor")

    def chunks(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    for direction, seq in (("asc", ITEM2), ("desc", list(reversed(ITEM2)))):
        for i, group in enumerate(chunks(seq, 8)):
            add_chunk(
                plan, f"r2-{direction}-{i}", "item2",
                [(r, BASE_SMS) for r in group], 10, 10, BASE_ACC, BASE_THREADS, BASE_ROUNDS,
            )

    threads = list(range(32, 1025, 32))
    if len(threads) != 32:
        raise SystemExit(f"thread list {len(threads)}")
    b = 0
    add_plain(plan, f"base-th-{b}", "threads", "full", 12, BASE_ACC, BASE_ROUNDS, BASE_THREADS, BASE_SMS)
    for i, th in enumerate(threads):
        add_plain(plan, f"th-{th:04d}", "threads", "full", 12, BASE_ACC, BASE_ROUNDS, th, BASE_SMS)
        if (i + 1) % 8 == 0:
            b += 1
            add_plain(plan, f"base-th-{b}", "threads", "full", 12, BASE_ACC, BASE_ROUNDS, BASE_THREADS, BASE_SMS)

    chains = [(4, 512), (8, 512), (16, 512), (32, 512), (4, 4096), (8, 2048), (16, 1024), (32, 512)]
    add_plain(plan, "base-ch-0", "chains", "full", 12, BASE_ACC, BASE_ROUNDS, BASE_THREADS, BASE_SMS)
    for i, (acc, rnd) in enumerate(chains):
        add_plain(plan, f"ch-a-{i}-{acc}-{rnd}", "chains", "full", 12, acc, rnd, BASE_THREADS, BASE_SMS)
    add_plain(plan, "base-ch-1", "chains", "full", 12, BASE_ACC, BASE_ROUNDS, BASE_THREADS, BASE_SMS)
    for i, (acc, rnd) in enumerate(reversed(chains)):
        add_plain(plan, f"ch-b-{i}-{acc}-{rnd}", "chains", "full", 12, acc, rnd, BASE_THREADS, BASE_SMS)
    add_plain(plan, "base-ch-2", "chains", "full", 12, BASE_ACC, BASE_ROUNDS, BASE_THREADS, BASE_SMS)

    desc_sms = list(reversed(SMS_ASC))
    for direction, seq in (("asc", SMS_ASC), ("desc", desc_sms)):
        for i, group in enumerate(chunks(seq, 8)):
            add_chunk(
                plan, f"sms-{direction}-{i}", "sms",
                [(BASE_ROUNDS, n) for n in group], 10, 10, BASE_ACC, BASE_THREADS, BASE_ROUNDS,
            )

    add_plain(plan, "base-grid-0", "grid", "full", 12, BASE_ACC, BASE_ROUNDS, BASE_THREADS, BASE_SMS)
    for th in (64, 128, 256, 512, 1024):
        seq = [(512, BASE_SMS), (32, BASE_SMS), (128, BASE_SMS), (2048, BASE_SMS), (512, BASE_SMS)]
        # 10 s levels, and the ends are rounds 512. A separate base fullwidth
        # brackets the grid because a non-512-thread launch cannot be the base.
        add_chunk(plan, f"grid-t{th}", "grid", seq[1:-1], 10, 10, BASE_ACC, th, BASE_ROUNDS)
        # add_chunk already wraps base rounds at this thread count. Replace the
        # interior with 32, 128, 2048. The wrapper is rounds 512. Good.
        # seq[1:-1] is 32, 128, 2048. Wrapper adds 512 at both ends. Yes.
    add_plain(plan, "base-grid-1", "grid", "full", 12, BASE_ACC, BASE_ROUNDS, BASE_THREADS, BASE_SMS)

    add_ramp(plan, "ramp30", "item4", 30, "full", "-", False)
    add_ramp(plan, "ramp60", "item4", 60, "full", "-", False)
    for direction, seq in (("up", STAIR), ("down", list(reversed(STAIR)))):
        add_chunk(
            plan, f"stair-{direction}", "item4",
            [(r, BASE_SMS) for r in seq], 10, 10, BASE_ACC, BASE_THREADS, BASE_ROUNDS,
        )
    add_ramp(plan, "ramp60-cl", "item4", 60, "load", "GPUTRONIC_ICLAMP_SMS=15 GPUTRONIC_TARGET_Z=1.5", True)

    # Confirm schedule. The suite appends the launches after it knows the workload.
    rows = [(0, 1000, BASE_SMS), (15000, 1200, BASE_SMS), (45000, 1000, BASE_SMS)]
    plan.write_sched("confirm", rows, two_col=True)

    verify_scales(plan.pairs)
    plan.save()
    n_item2 = sum(1 for L in plan.launches if L[1] == "item2")
    longest = max(plan.launches, key=lambda L: L[3])
    print(f"launches {len(plan.launches)} points {len(plan.points)} item2 {n_item2}")
    print(f"longest {longest[0]} {longest[3]}s")
    over = [L[0] for L in plan.launches if L[3] > 300]
    if over:
        raise SystemExit(f"launch over 5 min: {over}")
    if n_item2 != 10:
        raise SystemExit(f"expected 10 item2 launches, got {n_item2}")


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fifteenth-plan"
    build(dest)
