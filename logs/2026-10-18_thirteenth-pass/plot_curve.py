#!/usr/bin/env python3
"""Trip rate and last-10 s in-band share against step size. No extra packages.

Input rows, whitespace separated:
  scale trip_rate inband_share repeat
trip_rate and inband_share are fractions from 0 to 1.
One point per repeat. The line is the mean of the repeats at that scale.
"""
import sys
from collections import defaultdict
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
rows = []
for line in src.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    scale, trip, band, rep = line.split()[:4]
    rows.append((float(scale), float(trip), float(band), rep))

scales = sorted({r[0] for r in rows})
by = defaultdict(list)
for r in rows:
    by[r[0]].append(r)

W, H = 960, 520
L, R, Tm = 58, 24, 28
gap = 48
plot_w = W - L - R
panel_h = 180
y0 = Tm
y1 = Tm + panel_h + gap
x_lo, x_hi = 1.04, 1.21

def sx(scale):
    return L + (scale - x_lo) / (x_hi - x_lo) * plot_w

def sy(val, top):
    if val < 0:
        val = 0
    if val > 1:
        val = 1
    return top + (1 - val) * panel_h

colors = {"a": "#1a73e8", "b": "#e37400"}

def points(which):
    out = []
    means = []
    for i, scale in enumerate(scales):
        vals = []
        for rec in by[scale]:
            val = rec[1] if which == "trip" else rec[2]
            vals.append(val)
            col = colors.get(rec[3], "#333")
            jitter = -4 if rec[3] == "a" else 4
            x = sx(scale) + jitter
            y = sy(val, y0 if which == "trip" else y1)
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{col}"/>')
        if vals:
            means.append((scale, sum(vals) / len(vals)))
    if len(means) >= 2:
        pts = " ".join(f"{sx(s):.1f},{sy(v, y0 if which == 'trip' else y1):.1f}" for s, v in means)
        out.insert(0, f'<polyline fill="none" stroke="#555" stroke-width="1.2" points="{pts}"/>')
    return "\n".join(out)

def yticks(top):
    out = []
    for v in (0, 0.25, 0.5, 0.75, 1):
        y = sy(v, top)
        out.append(f'<line x1="{L-4}" y1="{y:.1f}" x2="{L}" y2="{y:.1f}" stroke="#333"/>')
        out.append(f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#333">{v:.0%}</text>')
    return "\n".join(out)

def frame(top):
    return f'<rect x="{L}" y="{top}" width="{plot_w}" height="{panel_h}" fill="none" stroke="#cccccc"/>'

xticks = []
for scale in scales:
    x = sx(scale)
    text = f"{scale:.3f}".rstrip("0").rstrip(".")
    xticks.append(f'<line x1="{x:.1f}" y1="{y1+panel_h}" x2="{x:.1f}" y2="{y1+panel_h+4}" stroke="#333"/>')
    xticks.append(f'<text x="{x:.1f}" y="{y1+panel_h+18}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#333">{text}</text>')

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="{W}" height="{H}" fill="#ffffff"/>
<text x="{L}" y="18" font-size="14" font-family="sans-serif" fill="#222">Compiled clamp: trip rate and last 10 s in band</text>
{frame(y0)}
{points("trip")}
{yticks(y0)}
<text x="16" y="{y0+panel_h/2}" font-size="12" font-family="sans-serif" fill="#333" transform="rotate(-90 16 {y0+panel_h/2})">loaded-hold trip rate</text>
{frame(y1)}
{points("band")}
{yticks(y1)}
{''.join(xticks)}
<text x="16" y="{y1+panel_h/2}" font-size="12" font-family="sans-serif" fill="#333" transform="rotate(-90 16 {y1+panel_h/2})">last 10 s in band</text>
<text x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#333">step size (schedule scale)</text>
<circle cx="{W-150}" cy="16" r="4" fill="#1a73e8"/>
<text x="{W-142}" y="20" font-size="12" font-family="sans-serif" fill="#333">repeat a</text>
<circle cx="{W-70}" cy="16" r="4" fill="#e37400"/>
<text x="{W-62}" y="20" font-size="12" font-family="sans-serif" fill="#333">repeat b</text>
</svg>
'''
dst.write_text(svg)
print(f"wrote {dst} bytes {dst.stat().st_size}")
