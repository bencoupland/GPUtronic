# GPUTronic

Closed-loop control of instruction-flow throughput on a GPU.

Reference: **Gold 1.0.0**

![GPUTRONIC](docs/assets/gputronic-mark.jpg)

Most GPU software still treats the device as a firehose with a thermal fuse: launch work, measure heat, hope. Gold treats useful work rate as a feedback variable. Observe it, estimate impedance from a calibrated tachometer, actuate on a short cable. Thermal and power stay with the stock driver. Once the plant interface is honest, this is ordinary control engineering — the same class of loop that is already standard on every other plant we bother to regulate.

The specification of record is `docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md`.

## Quick start (RTX 5080 / CUDA 12.8)

```bash
make gold
./build/gputronic_gold check     # dyno + closed-loop gate
./build/gputronic_gold run 30
make transfer && ./build/gputronic_transfer check
```

## What it is

- **Q** — useful work rate (aggregate atomic pulses)
- **Z** — impedance `Z = rate_ref / rate` after a sustained free-run window
- **D** — stock NVIDIA driver (not in the fast loop)
- **Observer** — 2-state Kalman on an ~8 ms rate window
- **Controller** — one-sided tracking PI; `sleep_scale` from measured loop period
- **Actuator** — `__nanosleep` via zero-copy mapped control block

Cooperative workloads only: publish pulses, honour `throttle_sleep_ns`.

## Layout

```
include/gputronic.h         public C API
src/gputronic_gold.cu       Gold core + CLI
src/gputronic_transfer.cu   second plant (GEMM tiles)
examples/gold_demo.c
docs/                       function frame, launch drafts
archive/v0.1-github/        first public PoC (GTX 1080)
GOLD.md STATUS.md TRANSFER.md
```

## History

v0.1 on this repo was a GTX 1080 proof of concept (`main.cu`). That commit stays in the log. The files live under `archive/v0.1-github/`. Gold 1.0 is the current root.

## License

MIT
