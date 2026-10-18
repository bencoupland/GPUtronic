#!/usr/bin/env python3
"""Draw z_hat and active_sms from a 1 ms loadfile trace. No extra packages.

One point about every 100 ms. The band and the six step times are exact.
Usage: tenth-cont-plot.py SRC.svg-title SRC.csv DST.svg
"""
import csv
import sys
from pathlib import Path

src = Path(sys.argv[2])
dst = Path(sys.argv[3])
title = sys.argv[1]

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

def sx(sec):
    return L + (sec - t[0]) / (t[-1] - t[0]) * plot_w

def sy(val, lo, hi, top):
    return top + (hi - val) / (hi - lo) * panel_h

z_lo, z_hi = 1.2, 2.0
n_lo, n_hi = 40, 82

def poly(xs, ys):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

z_pts = poly((sx(s) for s in ts), (sy(v, z_lo, z_hi, y0) for v in zs))
n_pts = poly((sx(s) for s in ts), (sy(v, n_lo, n_hi, y1) for v in ns))

def hline(val, lo, hi, top, color, dash=""):
    y = sy(val, lo, hi, top)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{color}" stroke-width="1"{extra}/>'

def band():
    y_a = sy(1.58, z_lo, z_hi, y0)
    y_b = sy(1.42, z_lo, z_hi, y0)
    return (f'<rect x="{L}" y="{y_a:.1f}" width="{plot_w}" height="{y_b-y_a:.1f}" '
            f'fill="#d9ead3"/>')

steps = [60, 120, 180, 240, 300, 360]
vlines = []
for s in steps:
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

xlabels = []
for sec in range(0, 421, 60):
    x = sx(sec)
    xlabels.append(f'<line x1="{x:.1f}" y1="{y1+panel_h}" x2="{x:.1f}" y2="{y1+panel_h+4}" stroke="#333"/>')
    xlabels.append(f'<text x="{x:.1f}" y="{y1+panel_h+18}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#333">{sec}</text>')

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="{W}" height="{H}" fill="#ffffff"/>
<text x="{L}" y="18" font-size="14" font-family="sans-serif" fill="#222">{title}</text>
{band()}
{''.join(vlines)}
{hline(1.50, z_lo, z_hi, y0, "#6aa84f", "4 3")}
<polyline fill="none" stroke="#1a73e8" stroke-width="1.2" points="{z_pts}"/>
<rect x="{L}" y="{y1}" width="{plot_w}" height="{panel_h}" fill="none" stroke="#cccccc"/>
<rect x="{L}" y="{y0}" width="{plot_w}" height="{panel_h}" fill="none" stroke="#cccccc"/>
<polyline fill="none" stroke="#e37400" stroke-width="1.2" points="{n_pts}"/>
{ticks(z_lo, z_hi, y0, [(1.2,"1.2"),(1.5,"1.5"),(1.8,"1.8"),(2.0,"2.0")])}
{ticks(n_lo, n_hi, y1, [(40,"40"),(50,"50"),(60,"60"),(70,"70"),(82,"82")])}
{''.join(xlabels)}
<text x="16" y="{y0+panel_h/2}" font-size="12" font-family="sans-serif" fill="#333" transform="rotate(-90 16 {y0+panel_h/2})">z_hat</text>
<text x="16" y="{y1+panel_h/2}" font-size="12" font-family="sans-serif" fill="#333" transform="rotate(-90 16 {y1+panel_h/2})">active SMs</text>
<text x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#333">seconds after calibration</text>
</svg>
'''
dst.write_text(svg)
print(f"wrote {dst} points {len(ts)} bytes {dst.stat().st_size}")
