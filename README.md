# GPUTronic

Closed-loop control for a GPU execution path.

Reference: **Gold 1.0.0**

![GPUTRONIC on an Audi B5 cluster — G7 ECU block/target/actual/throttle/temp](docs/assets/gputronic-mark.jpg)

The first language for this project was not FOC. It was closed-loop torque demand: Bosch Motronic on an Audi B5 S4 2.7T, then the same idea on silicon. Q is useful work rate. Z is impedance from a calibrated tachometer. D stays with the stock driver. Once the plant interface is honest, the rest is ordinary control engineering.

## Quick start (RTX 5080 / CUDA 12.8)

```bash
make gold
./build/gputronic_gold check     # dyno + closed-loop gate
./build/gputronic_gold run 30
make transfer && ./build/gputronic_transfer check
```

## What it is

- **Q-axis** — useful work rate (aggregate atomic pulses)
- **Z-axis** — impedance `Z = rate_ref / rate` after a sustained free-run window
- **D-axis** — stock NVIDIA driver
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
docs/                       essay, launch drafts, mark
archive/v0.1-github/        first public PoC (GTX 1080)
GOLD.md STATUS.md TRANSFER.md
```

## History

v0.1 on this repo was a GTX 1080 “instruction pressure engine” PoC (`main.cu`, 10 Hz ECU loop, RPM analogy). That commit stays in the log. The files live under `archive/v0.1-github/`. Gold 1.0 is the current root.

## License

MIT
