# Tenth pass, continued, 17 October 2026

This continues the tenth pass. The parent is `test-results/2026-10-17_tenth-pass` at `4224435`. The guest switch was already in that tree. Logs are `logs/2026-10-17_tenth-pass-continued/`. Kp 0.55, Ki 0.08, deadband 0.08, the 82-block launch, and the floor of 2 were not changed. The compiled integral clamp was not changed.

The clamp runs on the 1.2 load are not in this note yet. They start only after this result is on the remote.

## 1. Ninth-pass results on main

`e818644` was not an ancestor of `origin/main` (`b611a2b`). Pull request 6 merged `test-results/2026-10-13_ninth-pass` into `main` with a merge commit. The merge was clean. `origin/main` is `41a4598`, parents `b611a2b` and `e818644`. That commit lands the README landing page, the Results section, the renamed `docs/GPUtronic-GOLD-1.0-FUNCTION-FRAME.md`, and `logs/2026-10-13_ninth-pass/`.

`test-results/2026-10-17_tenth-pass` is still `4224435`. The ninth-pass branch is still `e818644`.

## 2. Where the guest's work went

The guest kernel now records, per block, the physical SM id, the GPU time the block entered, and the GPU time of its first published pulse. The host prints those at the stop, next to `[GUEST] launched at` and `[GUEST] gold stop at`. It also prints, once a second, how many of those words it can already see. Unset, no guest is launched.

Phase C was repeated from the tenth pass: `run 40`, 28 guest blocks at 8000 ms, idle-exit on, `GPUTRONIC_NO_CSV=1`, and the full-pipe switches (`GPUTRONIC_PEAK_ACC=32`, `GPUTRONIC_PEAK_ROUNDS=512`, `GPUTRONIC_THREADS=512`, `GPUTRONIC_PULSE_LOCAL=1`, `GPUTRONIC_CTRL_CACHE=1`). The window below is the 32 printed samples from `t=9` through `t=40`. Inside the band means `|Z-1.5| <= 0.08`.

The timestamps are the same shape on every run. All 28 blocks enter, and the host first sees the entry word and the first pulse on the stop sample. On the GPU timer the first pulse is 43 µs to 47 µs after entry, and `last_ns` equals `first_ns`. The stop line is `pulses=448 live=28` (28 blocks × 16 warps, one pass). During the hold the one-second guest rate is 0, which is 0 of the phase B solo rate 9,040,655 pulses/s. The blocks waited until Gold was told to stop, then each warp finished one pass. They did not enter early, and the mapped word is not hiding an earlier pulse: once a block runs, the host sees it on that sample.

The SM ids are the low ones (0–27, or 1–28 on two of the runs). Those ids are also in the Gold resident list printed at launch. The blocks arrived on them only as Gold stopped, so this is not a hold spent sharing an SM.

| Run | What was set | Z_raw mean, min, max, in band | Z_hat mean, min, max, in band | SM counts | Guest rate | Stop |
| --- | --- | --- | --- | --- | --- | --- |
| `h1-connections` | `CUDA_DEVICE_MAX_CONNECTIONS=32` | 1.4947, 1.422, 1.597, 30 of 32 | 1.4937, 1.422, 1.597, 30 of 32 | 54 on 12, 57 on 20 | 0 | pulses=448, stop 40005 ms, seen 40006 ms |
| `h2-stream-first` | `GPUTRONIC_GUEST_STREAM_FIRST=1` | 1.4968, 1.448, 1.575, 32 of 32 | 1.4973, 1.447, 1.575, 32 of 32 | 54 on 16, 57 on 16 | 0 | pulses=448, stop 40006 ms, seen 40007 ms |
| `h2-priority` | `GPUTRONIC_GUEST_PRIORITY=1` | 1.5035, 1.420, 1.562, 32 of 32 | 1.5046, 1.420, 1.562, 32 of 32 | 54 on 6, 57 on 26 | 0 | pulses=448, stop 40005 ms, seen 40006 ms |
| `h3-nonblock` | `GPUTRONIC_GOLD_NONBLOCK=1` | 1.5117, 1.429, 1.616, 30 of 32 | 1.5103, 1.429, 1.616, 30 of 32 | 54 on 3, 55 on 5, 57 on 9, 58 on 15 | 0 | pulses=448, stop 40006 ms, seen 40006 ms |

The two samples outside the band on `h1-connections` are `t=13` (1.597) and `t=27` (1.581). On `h3-nonblock` they are `t=14` (1.616) and `t=16` (1.584 on Z_raw, 1.583 on Z_hat).

`h2-priority` printed `stream priority=-5 (highest; range -5..0)`. `h2-stream-first` printed `stream created before the Gold launch`. `h3-nonblock` printed `persistent kernel on a non-blocking stream`. None of the four changed the guest rate.

The 54 to 57 step is a refill, not the guest. In `h2-stream-first.txt`, between `t=8` (`n=54`) and `t=9` (`n=57`), the log has `[SPAWN] base=54 n=3`. That refill had not published an SM id 2 ms later (`ready=0`). The last spawn in that file is between `t=8` and `t=9` (`count=15`). From `t=10` through `t=40` there is no `[SPAWN]` line, and the guest stays at 0. A pending refill is not what held the guest for the rest of the run.

`dmon -s pucm -d 1` and `clocks_event_reasons.active` (every sample `0x0000000000000000`):

| Run | dmon samples | Power | Die |
| --- | --- | --- | --- |
| `h1-connections` | 42 | 61–211 W | 52–66 °C |
| `h2-stream-first` | 41 | 37–205 W | 43–57 °C |
| `h2-priority` | 41 | 60–210 W | 48–61 °C |
| `h3-nonblock` | 41 | 40–206 W | 43–57 °C |

No run was stopped for temperature. Plain `nvidia-smi` before the clean checks was 43 °C and 38 W (`nvidia-smi-start.txt`). After the checks it was 50 °C and 42 W (`nvidia-smi-end.txt`).

The listed tries are ruled out. A shared hardware queue at the default connection count is ruled out by `CUDA_DEVICE_MAX_CONNECTIONS=32`. Stream order and the highest stream priority are ruled out by the two `h2` runs. The refill spawns are ruled out by the quiet stretch in `h2-stream-first`. A hidden mapped counter is ruled out because the entry stamp and the first pulse are one pass apart and the host sees both at the stop. Launching Gold itself on a non-blocking stream, which was not one of the listed tries, did not change it either.

The controller's view is still incomplete while a second kernel is resident. The guest's 448 pulses show up only as Gold stops.

`GPUTRONIC_GUEST_STREAM_FIRST`, `GPUTRONIC_GUEST_PRIORITY`, and `GPUTRONIC_GOLD_NONBLOCK` are test switches. Unset, the guest stream is still created after Gold's launch, at the default priority, and Gold's kernel stays on the default stream.

## Checks

Switches unset, on this source. The `[CHECK] z_mean` lines are:

- Gold `z_mean=1.468` (`check-gold.txt`). `=== GOLD GATE: dyno=PASS closedloop=PASS ===`
- Transfer `z_mean=1.497` (`check-transfer.txt`). `=== TRANSFER GATE: dyno=PASS closedloop=PASS ===`
- Episode `z_mean=1.443` (`check-episode.txt`). `=== EPISODE GATE: dyno=PASS closedloop=PASS ===`

The unset Gold lines are `[GPU] ... launching 82 ... thr/block=64` and `[CTRL] Gold | target=1.50 Kp=0.550 Ki=0.0800 Kd=0.0000 deadband=0.080`.
