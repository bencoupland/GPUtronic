#!/usr/bin/env python3
"""A small window on the compositor frame clock.

Writes one row per frame. The clock is the one Cinnamon drives for a
normal window. The first rows include the map. The reducer drops them.
"""
import sys

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk

out_path = sys.argv[1]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0

rows = []
start = [None]
phase = [0.0]

win = Gtk.Window(title="GPUtronic present probe")
win.set_default_size(960, 540)
win.move(120, 120)
area = Gtk.DrawingArea()
win.add(area)


def on_draw(widget, cr):
    cr.set_source_rgb(0.08, 0.10, 0.12)
    cr.paint()
    cr.set_source_rgb(0.20, 0.75, 0.90)
    x = 40.0 + (phase[0] * 0.12) % 700.0
    cr.rectangle(x, 180.0, 120.0, 120.0)
    cr.fill()
    return False


def tick(widget, clock):
    t_us = clock.get_frame_time()
    if start[0] is None:
        start[0] = t_us
    t_ms = (t_us - start[0]) / 1000.0
    phase[0] = t_ms
    dt_ms = (t_ms - rows[-1][0]) if rows else 0.0
    interval = ""
    presented = ""
    try:
        info = clock.get_refresh_info(t_us)
        if isinstance(info, tuple) and len(info) >= 2:
            interval, presented = info[-2], info[-1]
    except Exception:
        pass
    rows.append((t_ms, dt_ms, interval, presented))
    widget.queue_draw()
    if t_ms >= seconds * 1000.0 and len(rows) > 30:
        win.close()
        return False
    return True


area.connect("draw", on_draw)
armed = [False]

def on_map(*_a):
    if armed[0]:
        return
    armed[0] = True
    area.add_tick_callback(tick)

win.connect("map-event", on_map)
win.connect("destroy", Gtk.main_quit)
GLib.timeout_add_seconds(int(seconds) + 8, lambda: (win.close(), False)[1])
win.show_all()
Gtk.main()

with open(out_path, "w") as f:
    f.write("t_ms,dt_ms,refresh_interval_us,presentation_us\n")
    for t_ms, dt_ms, interval, presented in rows:
        f.write(f"{t_ms:.3f},{dt_ms:.3f},{interval},{presented}\n")
print(f"frames={len(rows)} wrote={out_path}")
