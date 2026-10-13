# GPUtronic Episode Plant

Third counted kernel on the Gold interface. The kernel returns. The host launches it again.

```bash
make episode
./build/gputronic_episode check
# === EPISODE GATE: dyno=PASS closedloop=PASS ===
```

Uses stock Gold (`launch_selftest_kernel = 0`). Blocks at or above `active_sms` return without adding a pulse. The host does launch, then `cudaDeviceSynchronize`.

## Why this kernel exists

Transfer stays in a loop. A lot of real code launches a kernel and gets it back. This kernel asks whether the 8 ms window still sees Z in that case.

## Plant

- Finite kernel, 84×64, same 8×8×8 GEMM tiles as transfer
- 32 tiles per pulse × 4 pulses per launch, then return
- `atomicAdd` on Gold `total_work_pulses`
- 2026-09-23 closed loop: 20.3 → 15.0 GFLOP/s, Z mean ≈ 1.36

The sections below about one sleep per launch are the measurements from before the control became an SM count. They explain why sleep duration was a poor control on a kernel that returns.

## Live numbers (RTX 5080, 2026-08-15)

### Open-loop dyno

| sleep | pulse/s | GFLOP/s | Z_hat |
|------:|--------:|--------:|------:|
| 0 | 718695 | 23.55 | 0.983 |
| 200 µs | 628228 | 20.59 | 1.124 |
| 500 µs | 623911 | 20.44 | 1.125 |
| 1 ms | 318739 | 10.44 | 2.235 |
| 2 ms | 318812 | 10.45 | 2.233 |

- Free-run Z ≈ 0.98. The ~8 ms window sees Z on a bursty plant. That is the main result: Z is still measurable on a bursty kernel.
- Tile rate and GFLOP/s stay coupled.
- R² ≈ 0.85.
- Stock Gold 500 µs is a **13%** cut (Z ≈ 1.13). It does **not** reach target 1.50.
- Authority appears at the 1 ms step (Z ≈ 2.23, −56%). 1 ms and 2 ms are the same rate: `__nanosleep` is not a linear wall clock here, same as the persistent plants.

### Closed-loop (stock max_sleep = 500 µs)

| | pulse/s | GFLOP/s | sleep | Z_mean |
|--|--------:|--------:|------:|-------:|
| sustained free-run | 728741 | 23.88 | 0 | 1.00 |
| closed-loop window | 633701 | 20.77 | ~240 µs | 1.15 |

Useful work dropped 13% when the loop engaged. Direction matches the model. Tracking to 1.50 is **not** claimed — the dyno already said 500 µs parks near Z 1.13. The PI is engaged; the actuator does not have the throw.

Gate PASS is: window alive, coupled FLOPs, useful drop ≥10%, cal not swallowed. It is not a claim that Z=1.50 transferred.

## What this is — and is not

**Is:** proof that an episodic cooperative plant can publish Q, that the 8 ms observer still forms Z, and that counted FLOPs move when sleep has authority.

**Is not:** llama.cpp, a game hook, or a reason to retune Gold 1.0 Kp. Contending mode is still a different plant.

## Overnight campaign (2026-08-16)

`results/episode_campaign/20260816_021956/REPORT.md`

- 8 ms window still sees Z at 1…64 pulses/launch (episode 0.5–3.8 ms). Observer starvation was not the failure.
- `__nanosleep` cliff on this 5080 is **500 → 600 µs** (Z 1.22 → 2.39), then flat to 2 ms.
- F-09 period (~55–290 µs) is the pulse interval. Launch period is 5–70× larger. Wrong clock.
- Throw on stock 500 µs, no Kp change:

| pulses | T | once | chunk2 | chunk4 | host |
|------:|----:|-----:|-------:|-------:|-----:|
| 4 | 0.6 ms | 1.22 | **1.37** | **1.38** | **1.38** |
| 16 | 1.3 ms | 1.10 | 1.19 | **1.35** | 1.18 |
| 64 | 3.8 ms | 1.04 | 1.08 | 1.13 | 1.05 |

tight = |Z−1.50|<0.20. N must scale with T. Gold + transfer + default episode still PASS.

## What a normal launch looked like with sleep

These numbers are from when the control was sleep duration. A kernel that sleeps once, for at most 500 µs, at the end of a long launch does not reach Z = 1.5. The rate is visible. The target is not.

Ways that did move a long launch, in those tests:

1. Sleep more than once before the kernel returns.
2. Wait on the CPU between launches, for a time that is a real fraction of the launch.
3. Stop using sleep duration. The current code does this. It sets how many SMs run.

Do not paper over a weak control by raising Kp.

## Files

- `src/gputronic_episode.cu`
- `build/gputronic_episode`
- `results/gputronic_episode_*_report.txt`
- CSV evidence under `results/`

## Proof demo (2026-09-21, RTX 5080, stock driver)

Gold source, Kp, Ki, and the 500 µs clamp were not touched. `./build/gputronic_episode proof`.

The knob in this test was tiles per pulse. Same 8×8×8 math. The reference kernel ("twin" in the table) does not use the control block. The instrumented kernel does one mapped `atomicAdd` per pulse and a fence per launch.

| tiles/pulse | reference GFLOP/s | with counter GFLOP/s | ratio | launch |
|------------:|-------------:|---------------:|------------:|-------:|
| 32 | 739 | 50 | 0.07 | 15 µs |
| 2048 | 1321 | 1195 | 0.90 | 0.53 ms |
| 8192 | 1355 | 1324 | 0.98 | 2.1 ms |
| 32768 | 1361 | 1358 | 1.00 | 8.3 ms |

The atomic is the plant at 32 tiles. By 8192 it is noise. Past that, more tiles do not buy rate. The useful batch is the one whose free-run, with Gold's own 10 µs path running, stays within about a tenth of the twin and still feeds the 8 ms window.

Chosen cell: **32768 tiles, 4 pulses/launch**. Paired measurement, Gold in the process:

- Reference kernel without the counter: 1367 GFLOP/s
- Counter on, sleep 0, Z_hat 1.02: **1247 GFLOP/s, 0.913 of the reference (counting costs about 9%)**
- Launch T = 9.03 ms. About 320 pulses in each 8 ms window.

Open-loop host wait after `cudaDeviceSynchronize`, device sleep held at 0. Wall \(Z = r_\mathrm{free}/r\):

| wait / T | actual wait | Z | GFLOP/s |
|----------:|------------:|--:|--------:|
| 0 | 0 | 1.00 | 1253 |
| 0.25 | 2.34 ms | 1.26 | 990 |
| 0.50 | 4.60 ms | **1.50** | 829 |
| 0.75 | 6.86 ms | 1.76 | 708 |
| 1.00 | 9.12 ms | 2.01 | 620 |
| 1.25 | 11.4 ms | 2.25 | 553 |

That is the duty cycle. Half a launch of real wall time is the setpoint, and counted FLOPs follow it. `__nanosleep` of the stock clamp, repeated, does not land on that slope: N=8 gives Z 1.42, N=16 gives Z 1.89. One longer nanosleep is the old plateau. The throw that grades through 1.50 is a host wait.

Closed loop, stock PI, stock clamp. The plant's only added law: a command of 500 µs waits the 4.51 ms that open-loop held at Z 1.50, and a smaller command waits the same fraction. Shorter waits are linear, so the PI has a slope. It does not sit on 1.50. It bangs between base and the clamp, because F-09's scale is against its own 8e6 cap here (it would like to command ~1.3 ms) and any real error saturates. Wall Z over the engaged window was **1.34**, 932 GFLOP/s. Same character as Gold's own closed loop on the persistent plant.

Release, target set back to 1.0: sleep back to 5 ns, **1240 GFLOP/s** against a free-run of 1247, Z_hat 1.03.

A mean-command rescale (treat the ~270 µs average as full scale) was run once and dropped. The average was a mixture of 5 ns and 500 µs, so enlarging every kick moved the mean to ~1.48 and the trace hunted from about 1.2 to 1.9.

`make episode-check` remains the original once-per-launch gate. Proof is a separate mode.
