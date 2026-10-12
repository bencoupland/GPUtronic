# Next changes, and the paths after them

5080 seat, 2026-10-12. This is the order. The suite score is
`5080-2026-10-12.md` on `test-results/2026-10-12_5080`. The run notes under it are
`5080-2026-10-12-dispatch.md` on the same branch (`42b88c9`). Hardware was an RTX
5080, driver 595.99.02, kernel 7.0.0-38, CUDA 12.8. Kp stays 0.55. Ki stays 0.08.
Nothing here merges to `main`.

Do section A before any of the paths. Re-run the three checks on the 5080 after A.
A pass means the counter is alive and the rate moved the right way. It does not mean
the trace sat on 1.50.

## A. Changes on this PR

### A1. Port the free-run wait

**Done** (`4152fb9`, cherry-pick of `ff4e97c`). Not yet re-run.

`mode_closedloop` in Transfer and Episode slept 2.2 s and then called that window
the full-width rate. The four-point table runs until about 3 s, so the window landed
on a probe step. Both gates failed with Z already near the target. Commit `ff4e97c`
on `test-results/2026-10-12_5080` waits 4.5 s, forces full width, then takes the same
one-second sample. Bring that wait across. Do not retune anything to make the gate
green.

Done when, on this 5080: transfer and episode `check` both print PASS, the free-run
line is near the calibration `rate_ref` (transfer about 6.5 GFLOP/s, episode about
18 GFLOP/s), and closed-loop mean Z is in 1.40–1.55.

### A2. Tensor sensor off by default

Measured, Gold dyno. Tensor off, 84 SMs: 555k pulses/s. Tensor on: 337k, a 39% cut.
The tensor counter then runs at half the CUDA-core rate at every width, so the two Z
values are the same number. The selector has nothing to choose. With the warp off,
the full-width rate matches the old law's ceiling on this driver.

Default `enable_tensor_sensor` to 0. Keep `GPUTRONIC_TENSOR_SENSOR=1` as the override.
Transfer and Episode are already off.

Done when a default Gold `check` prints `tensor sensor off` and the full-width rate
is in the 550k–590k band, not the 340k band.

### A3. Deadband back near 0.08

**Done** (`8a2011f`): default 0.08; `GPUTRONIC_DEADBAND` override ported (`52f4b9e`). Not yet re-run.

The 0.015 band does not hold the count. Z moves by about 0.05 to 0.07, so the count
walks. Gold with the new law and the tensor warp on sat at mean Z 1.487 with the
count from 44 to 52.

One later run, tensor off, `GPUTRONIC_DEADBAND=0.08` (the override is on the results
branch only, commit `42b88c9`; the default in the header is still 0.015). The table
landed at `n_ff` 48.3. Last 20 s: mean Z 1.516, standard deviation 0.038, 94% of
samples at 46 SMs, integral frozen at 0. Same pulse rate as the tight-band run,
389k. The tenth came from the table and the two-sided law. The tight band is what
keeps the count moving.

Set `GPUTRONIC_GOLD_DEADBAND` to 0.08. Leave Kp and Ki. This was one probe. It
landed inside the band, the law cut two SMs, and then it froze. A probe that lands
at Z 1.35 is still outside 0.08 and will still trim. The wide band freezes a start
that is already inside about 1.42 to 1.58. It does not freeze a bad start.

Done when a default Gold `run 30 1.5` with the tensor sensor off holds mean Z in
1.45–1.55 and one SM count for most of the last 20 s. If a second run freezes
outside that Z band, write down the `n_ff` and the Z. Do not move Kp or Ki.

### A4. Take `__nanosleep` out of the device kernels

**Done** (`06fdee5`): all device calls in Gold, Transfer, Episode and Show removed; no `clock64`.
sm_120 builds; `nvcc -arch=sm_61 -c` of all four kernels compiles. **Risk to check on the 5080:** idle
and paused blocks now re-read `control_flags`/`active_sms` from mapped host memory with no pause, so
they may add PCIe read traffic that slows the working blocks. Compare T2/T10 free-run rate with
the previous build. The `follow` translator in Transfer is still there.

Sleep duration is not the actuator. On this 5080 a sleep command did nothing until
about 500 µs, then Z stepped off a cliff. That is one coarse notch. The SM count
grades the rate from 84 down to 4.

The calls that remain are not commands.

- Gold: a 5 µs poll on the pause flag, a 5 µs poll on blocks past `active_sms`, and
  a 1 µs nap after reset clears the per-SM counter. The block past the count does
  not pulse. The 5 µs is only how long it waits before reading the count again.
- The same 5 µs polls are in Transfer, Episode's pause path, and Show.
- Transfer still has the old stall, `__nanosleep(sleep_ns)` at the end of the work,
  and a path that turns `throttle_sleep_ns` into an SM drop (`follow`). The stock
  loop does not take the stall: the sleep value sits at a few nanoseconds, and the
  stall only runs above 1 µs. The cable path writes `active_sms` directly.
- Episode's epilogue does the same commanded stall when `sleep_ns > 1000`. A block
  past `active_sms` already returns with no sleep, and the closed loop still tracks.
  That is the proof the sleep is not the law.

Delete those device `__nanosleep` calls. An idle persistent block reads `active_sms`
and goes around. Do not add a `clock64` spin in their place. A spin burns the SM and
does not idle it, and it is still not an actuator. An idle block that spins and an
idle block that sleeps both keep the block resident. Neither one hands that SM to
another kernel. Launching fewer blocks does.

Leave the host `clock_nanosleep` in the episode pump. That is a CPU wait between
launches, not an SM command.

The `follow` translator, sleep duration in and SM drop out, is the old command
wearing the new actuator. Do not keep it as a second law. Stock control writes
`active_sms`.

Done when `nvcc -arch=sm_120` and `nvcc -arch=sm_61` both get past the device
compile of Gold, and a 5080 Gold `check` still passes with the tensor sensor off.
sm_61 will not be run here. Compiling it is the check that no `__nanosleep` remains.

## B. Do not do these on the back of the model

- **Integral clamp now scaled (user decision, 2026-10-12, overrides the earlier "do not scale" advice).** Default `integral_clamp` 0 means: after calibration, clamp = 0.04 × num_sm / (Ki × sms_per_u), so the integral can move at most 4% of the SMs (84 SMs: ~3.4 SMs, close to the old ~4; 20 SMs: 0.8 SM). A config value > 0 still fixes it. Printed in the `[CAL] integral clamp=` line. Unmeasured on hardware; the C1 load step should validate it.
- (Superseded by the line above.) Do not scale `GPUTRONIC_GOLD_INTEGRAL_CLAMP`. Ki times 1.0 times `sms_per_u` is
  about four SMs. That limit is real, and it is not yet the thing that misses. One
  episode run parked on the clamp at Z 1.43. The re-run did not: from 8 s on, mean
  Z 1.467, count about 24, integral mean 0.55 and maximum 0.83. The reserve-4 run
  hit the clamp because the integral was undoing the extra SMs, which is what that
  knob asks it to do. The line "Z hangs at 1.62–1.74 after a 20% load" is the
  simulator. No load step has been measured. A bigger clamp, with a deadband that
  lets the integral wind on every noisy sample, is how the count starts walking
  farther. Measure path C1 first. Then decide.
- Do not turn Kd on. The derivative still spikes. One sample read −1.16 per second
  while Z was 1.42.
- Do not add `clock64` as a fallback actuator. See A4.
- Do not start the counter that keeps going after a kernel returns. That is path C5.
  It is not part of cleaning up this PR.

## Telemetry and noise (new, 2026-10-12)

- `GPUTRONIC_TRACE_BIN_US=1000` for 1 ms traces; PI counters in the trace and `[STATS]`; T10 `noise`
  mode for the open-loop noise floor; analyzer prints SM walk / noise / counters. See `TEST_SUITE.md`.
- `work_scale_milli` is now read only while `GPUTRONIC_FLAG_LOAD` is set (`52f4b9e`), so normal runs do
  not pay the extra mapped read blamed for 435–450k vs 533k free-run.

## T9. Pulse-to-work ratio (new, run early)

`timeout 120 ./build/gputronic_gold ratio 5` then `python3 scripts/analyze_ratio.py results/gputronic_gold_ratio.csv`.
Open loop at 84, 63, 42, 21 SMs (N, 3N/4, N/2, N/4 on other cards). Compares mapped-counter pulses with a
device-memory count of completed batches. Pass: drift within 1% of the 84-SM ratio. If it drifts, the Z
math is wrong and `rate_ref` needs revisiting. Details in `TEST_SUITE.md`.

## C. Paths, after A is on the 5080

### C1. Plant-gain step, before any clamp change

**Implemented as T6** (`gputronic_gold loadstep`, field `work_scale_milli`, see `TEST_SUITE.md`); 1.2× only, no 0.8×.
Baseline with `GPUTRONIC_INTEGRAL_CLAMP=1.0`, then the scaled clamp. Not yet run.

This is the load test that matches the model. Change the work per pulse in the
kernel we own, by a field in the control block, something like 0.8× and 1.2× the
FMA count. Tensor sensor off. Deadband at the A3 default. After calibration plus
10 s, step the work, hold 20 s, step back.

Record mean Z, the SM count, the integral, and how long the integral sits on 1.0.
Do this once before anyone edits the clamp, and keep that log.

A second CUDA process is a different question. It is coexistence, path C3. The
5 October share run already showed the shape: a short launch does not slow down
when you launch fewer blocks, because the blocks that remain fill the chip. A
resident grid whose idle blocks sleep left much less for the other kernel. That
will not be a clean 20% change in plant gain, and it will not prove the clamp.

### C2. The shoulder sample in the Gold table

`n_ff` at target 1.5 repeated at 47 to 50 on every Gold run except one. That one
sampled 63 SMs at Z 1.32 and got `n_ff` 55.4. Every other 63-SM sample was Z 1.18
to 1.23. Episode repeated at 25.8 to 27.3.

A longer dwell at 63 SMs, or a table point at 56 SMs where the shoulder actually
is, is the fix if that miss shows up again. Do not lengthen every probe point on
a hunch. The PI trimmed the 55-SM start back into the Z band. The count it settled
on was not the same count as the other runs.

### C3. Coexistence, separate from C1

Question: does holding one kernel at Z 1.5 leave a real rate for a second kernel.
The comparison that matters is launching only `active_sms` blocks, against a
resident grid of 84 whose idle blocks skip. Sleeping or spinning inside the idle
block is not "SM handed over." Launch width is.

This is the share run, not a new plant. Do not build it until A is in and someone
asks for the coexistence number again.

### C4. Bursts, after C1 exists

Bursts of 2, 4, 8, 16, 50 and 200 ms. The 8 ms rate window is why the short ones
should be invisible and the 16 ms ones should move the count inside one or two
windows. Record peak Z and the time above 1.5. Do not treat a miss as a reason to
raise Kp.

### C5. One counter that outlives the launch

The loop is closer to 1.5 on kernels this tree compiles and counts. The open
measurement is still one counter, in code we compile ourselves, that keeps going
after a kernel returns, as the normal meter rather than a special mode. A driver
change is how that could apply to a program we do not modify. Do not start C5
until Ben asks. Integrating into another program is not this measurement.

### C6. GTX 1080, after A4

20 SMs, sm_61. No `__nanosleep` left, so no `clock64` stand-in. Build
`ARCH=-arch=sm_61`. Then, on that machine: zero-copy round trip, dyno rate falling
from 20 SMs to 4, closed loop with the same gains. One SM is a large piece of Z
there, so the count should sit on one value or two, not walk. The 0.08 deadband
matters more on 20 SMs than on 84. Do not retune Kp or Ki to make a 20-SM card
look like the 5080.

### C7. A game beside Gold

A co-run shows whether a Gold kernel held at Z 1.5 leaves the game's frame times
alone better than the same kernel at full width. GPUTronic does not see the game
and does not throttle it. The Cyberpunk write-up's conditions A through D are the
right shape. Use the tensor sensor off, and finish calibration before the game
starts, or `rate_ref` is not the kernel's own ceiling.

Condition B in that write-up is `run` at Z 1.0, called the full-width reference.
With the 0.015 band that request sat at a mean of 83 SMs, and only 57% of the
samples were at 84. Full width is an open-loop count of 84 (`active_sms` held at
the device count), not a target of 1.0. There is no `run` argument that holds the
count. `gputronic_set_open_loop_sms(h, 0)` means every SM. `run` at Z 1.0 is still
the closed loop. Condition D in that write-up is Gold alone at the same target, not
the free-run ceiling. The ceiling is the calibration `rate_ref`.

This box's measured game was Space Marine 2 at 4K, Ultra, DLSS Quality, frame
generation on. Do not change those settings to match a Cyberpunk preset. A game
co-run is a demo. It is not C5.
