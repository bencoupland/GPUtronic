# X thread — GPUtronic Gold 1.0.0
# @benracoupland
# Draft from the first public post, when the control was sleep duration.
# The repo now uses an SM count. See the README.

> **Superseded.** This describes the original sleep-duration control. Commit `600ed66` (2026-09-23) replaced sleep actuation with SM-count actuation (`active_sms`), and PR #1 removed the remaining sleep API (`gputronic_set_open_loop_sleep`, `gputronic_get_sleep_ns`, the sleep scale settings) and made the controller two-sided. See the README and `docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md` for the current design.

# Posts were checked to stay within 280 characters.

---

1/
GPUtronic Gold 1.0.0

Reference closed-loop control on a GPU execution path.

RTX 5080 / CUDA 12.8
make gold && ./build/gputronic_gold check → PASS
make transfer && ./build/gputronic_transfer check → PASS

---

2/
Model

GPU as a controllable plant, not a batch black box.

Q — useful work rate (pulse tach)
Z — impedance from rate (sustained free-run cal ≈ 1)
D — thermal/power left to stock driver

Observe, estimate, actuate.

---

3/
Implementation

• zero-copy control block
• aggregate atomicAdd on total_work_pulses
• Z from a sustained rate window
• rate over ~8 ms, not a 10 µs tick
• sleep_scale from loop period
• PI → nanosleep; one-sided when Z < target

---

4/
Why those last three

Peak-EMA cal parked free-run Z above the setpoint, so the loop never engaged.

A 10 µs rate EMA cannot see a ~300 µs plant.

Stock scale commanded ~8 µs of sleep. The actuator only bites near 200–500 µs.

---

5/
Scope

In: cooperative workloads that publish pulses and read throttle_sleep_ns.

Out: programs we did not modify. They do not publish pulses or read the control value.

That is a different problem.

---

6/
Measured (this 5080)

Gold plant: 445 → 324 kpulse/s at target Z=1.5 (−27%). Z_mean≈1.37.

Transfer plant (8×8 GEMM tiles): 8.77 → 6.55 GFLOP/s (−25%), same controller.

CSV every run.

---

7/
Claim

A feedback loop on GPU work rate is practical once the counter and the control input are real.

Gold 1.0.0 is that interface, plus one second plant that moves counted FLOPs.

---

# Single post

GPUtronic Gold 1.0.0 — closed-loop GPU governor (RTX 5080).

Q tach · sustained Z cal · period-scaled PI → nanosleep · zero-copy.

Gold check PASS. Transfer GEMM plant: 8.77→6.55 GFLOP/s at Z=1.5.

Cooperative only.
