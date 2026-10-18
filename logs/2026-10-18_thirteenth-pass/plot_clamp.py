#!/usr/bin/env python3
"""Draw z_hat and active_sms from a 1 ms loadfile trace. No extra packages.

One point about every 100 ms. The band is target ± 0.08.
Usage:
  plot_clamp.py TITLE SRC.csv DST.svg [z_lo z_hi n_lo n_hi target [steps]]
steps is a comma-separated list of seconds. Default is the twelfth-pass
60,120,180,240,300,360 schedule. Default axes are z 1.2–2.0, SM 40–82, target 1.5.
"""
import csv
import sys
from pathlib import Path

title = sys.argv[1]
src = Path(sys.argv[2])
dst = Path(sys.argv[3])
z_lo = float(sys.argv[4]) if len(sys.argv) > 4 else 1.2
z_hi = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
n_lo = float(sys.argv[6]) if len(sys.argv) > 6 else 40
n_hi = float(sys.argv[7]) if len(sys.argv) > 7 else 82
target = float(sys.argv[8]) if len(sys.argv) > 8 else 1.5
if len(sys.argv) > 9 and sys.argv[9]:
    steps = [float(s) for s in sys.argv[9].split(",") if s]
else:
    steps = [60, 120, 180, 240, 300, 360]

t, z, n = [], [], []
with src.open() as f:
    for row in csv.DictReader(f):
        t.append(float(row["t_ms"]) / 1000.0)
        z.append(float(row["z_hat"]))
        n.append(int(float(row["active_sms"])))

step = 100
ts, zs, ns = t[::step], z[::step], n[::step]
if ts[-1] != t[-1]:
    ts.append(t[-1])
    zs.append(z[-1])
    ns.append(n[-1])

W, H = 960, 520
L, R, T = 58, 16, 28
gap = 36
plot_w = W - L - R
panel_h = 200
y0 = T
y1 = T + panel_h + gap
t0, t1 = t[0], t[-1]
if t1 <= t0:
    t1 = t0 + 1.0

def sx(sec):
    return L + (sec - t0) / (t1 - t0) * plot_w

def sy(val, lo, hi, top):
    if val < lo:
        val = lo
    if val > hi:
        val = hi
    return top + (hi - val) / (hi - lo) * panel_h

def poly(xs, ys):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

z_pts = poly((sx(s) for s in ts), (sy(v, z_lo, z_hi, y0) for v in zs))
n_pts = poly((sx(s) for s in ts), (sy(v, n_lo, n_hi, y1) for v in ns))

def hline(val, lo, hi, top, color, dash=""):
    y = sy(val, lo, hi, top)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{color}" stroke-width="1"{extra}/>'

def band():
    y_a = sy(target + 0.08, z_lo, z_hi, y0)
    y_b = sy(target - 0.08, z_lo, z_hi, y0)
    top, bot = (y_a, y_b) if y_b > y_a else (y_b, y_a)
    return (f'<rect x="{L}" y="{top:.1f}" width="{plot_w}" height="{bot-top:.1f}" '
            f'fill="#d9ead3"/>')

vlines = []
for s in steps:
    if s < t0 or s > t1:
        continue
    x = sx(s)
    vlines.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+panel_h}" stroke="#bbbbbb" stroke-width="1"/>')
    vlines.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y1+panel_h}" stroke="#bbbbbb" stroke-width="1"/>')

def ticks(lo, hi, top, labels):
    out = []
    for val, text in labels:
        y = sy(val, lo, hi, top)
        out.append(f'<line x1="{L-4}" y1="{y:.1f}" x2="{L}" y2="{y:.1f}" stroke="#333"/>')
        out.append(f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#333">{text}</text>')
    return "\n".join(out)

span = t1 - t0
tick_step = 60 if span > 200 else (30 if span > 80 else 15)
xlabels = []
sec = 0.0
while sec <= t1 + 1e-6:
    x = sx(sec)
    label = str(int(sec)) if abs(sec - int(sec)) < 1e-6 else f"{sec:.0f}"
    xlabels.append(f'<line x1="{x:.1f}" y1="{y1+panel_h}" x2="{x:.1f}" y2="{y1+panel_h+4}" stroke="#333"/>')
    xlabels.append(f'<text x="{x:.1f}" y="{y1+panel_h+18}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#333">{label}</text>')
    sec += tick_step

def zlabels():
    vals = [z_lo, target - 0.08, target, target + 0.08, z_hi]
    out = []
    seen = set()
    for v in vals:
        text = f"{v:.2f}".rstrip("0").rstrip(".")
        if text in seen:
            continue
        seen.add(text)
        out.append((v, text))
    return out

def nlabels():
    span_n = n_hi - n_lo
    step_n = 10 if span_n > 20 else 4
    vals = []
    v = int(n_lo)
    while v < n_hi:
        vals.append(v)
        v += step_n
    if int(n_hi) not in vals:
        vals.append(int(n_hi))
    return [(float(v), str(v)) for v in vals]

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="{W}" height="{H}" fill="#ffffff"/>
<text x="{L}" y="18" font-size="14" font-family="sans-serif" fill="#222">{title}</text>
{band()}
{''.join(vlines)}
{hline(target, z_lo, z_hi, y0, "#6aa84f", "4 3")}
<polyline fill="none" stroke="#1a73e8" stroke-width="1.2" points="{z_pts}"/>
<rect x="{L}" y="{y1}" width="{plot_w}" height="{panel_h}" fill="none" stroke="#cccccc"/>
<rect x="{L}" y="{y0}" width="{plot_w}" height="{panel_h}" fill="none" stroke="#cccccc"/>
<polyline fill="none" stroke="#e37400" stroke-width="1.2" points="{n_pts}"/>
{ticks(z_lo, z_hi, y0, zlabels())}
{ticks(n_lo, n_hi, y1, nlabels())}
{''.join(xlabels)}
<text x="16" y="{y0+panel_h/2}" font-size="12" font-family="sans-serif" fill="#333" transform="rotate(-90 16 {y0+panel_h/2})">z_hat</text>
<text x="16" y="{y1+panel_h/2}" font-size="12" font-family="sans-serif" fill="#333" transform="rotate(-90 16 {y1+panel_h/2})">active SMs</text>
<text x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#333">seconds after calibration</text>
</svg>
'''
dst.write_text(svg)
print(f"wrote {dst} points {len(ts)} bytes {dst.stat().st_size}")
