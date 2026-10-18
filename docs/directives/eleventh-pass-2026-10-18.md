# Eleventh pass: Z = 1 as home, the clamp as a guardrail, and a complete view of every resident kernel

This follows the tenth pass and its continuation, and builds on what tonight's discussion settled: Z = 1 as the home state, the integral clamp as a guardrail, and closed-loop feedback as the differentiator.

This follows `docs/directives/tenth-pass-continuation-2026-10-17.md` and its results on `test-results/2026-10-17_tenth-pass-continued` (`740efea`, report `docs/analysis/tenth-pass-continued-2026-10-17.md`). Start from that results branch at `740efea`. Post everything to a new branch, `test-results/2026-10-18_eleventh-pass`, with logs under `logs/2026-10-18_eleventh-pass/`. Do not force-push, and don't merge anything.

## Start here

The full budget is available. Complete items 1 to 5, then the report and checks in item 6.

**You may argue.** If you disagree with any directive here, say so in the report, with your reasoning and whatever evidence you have, before or instead of carrying it out. A reasoned objection counts as a result. Don't quietly work around a directive. Argue it in writing.

## How GPUtronic is meant to work

These principles shape every item below. Read them first.

- **The fuel-injection model: universal, not tuned per board.** GPUtronic should run on any card with the same behaviour, with no per-board tuning service. Calibration it does for itself is fine. Hand tuning per board is not the product.
- **Z = 1 is the home state.** By default the system targets Z = 1 and runs as fast as free run. Holding a rate, not slowing the work down, is what it does by default.
- **Deterministic mode is the opt-in exception.** A commanded Z (holding a slower rate in exchange for bounded latency) is for hard real-time cases such as self-driving vision stacks. It isn't the floor, and it isn't the default.
- **Commanded Z limits throughput. The clamp doesn't.** If a workload needs its throughput limited, a commanded Z does it. The integral clamp is a guardrail that should almost never trip. If it trips on real workloads, the fix is upstream (calibration, feedforward, or the plant model), not a tighter or looser clamp.
- **The stock driver owns physical limits. GPUtronic owns scheduling behaviour.** Thermal, power and clocks belong to the driver. GPUtronic doesn't read or fight them, and the two stay orthogonal. Stopping a test run at 83 °C is a test safety rule, not controller logic.
- **Standalone, loosely coupled.** Control what matters and delegate the rest. Fewer variables to account for means more flexibility. Don't add a knob, a coupling or a dependency unless a result needs it.
- **Closed loop is the differentiator.** The published approaches to bounded GPU latency are static and open loop. NVIDIA DRIVE's System Task Manager is a static, non-preemptive scheduler, and the UNC partitioning work (Bakita and Anderson) sets partitions ahead of time. GPUtronic holds a rate with feedback. Keep the design pointed at that.

## Where things stand

- **The throttle works.** It holds Z in the band at whatever rate is asked for, up to free run. The ninth pass, tenth-pass phases A and C, and all four guest runs in the continuation held it. The continuation's three checks all passed with switches unset (Gold 1.468, Transfer 1.497, Episode 1.443), with no regressions.
- **The wider clamp unstuck the integral.** On the 1.2 load step, the compiled 3.28-SM clamp sat on its rail for 100% of the last 30 s of every loaded hold, and that window was 55.0–57.3% in band. At `GPUTRONIC_ICLAMP_SMS=9` and `12`, the integral never reached the rail (`integral_on_rail=0`), and the loaded holds were 81.0–83.0% in band. At 6 SMs it still ended on the clamp. The cost: the returns to 1.0 fell from 96–98% in band to 85–87% at 9 and 84–86% at 12, and the undershoot went from about 1.36 to about 1.31. At 12, the highest `z_hat` of the run was 1.97583. 15 SMs wasn't run.
- **The guest problem is open.** In every guest run the 28 guest blocks entered only when Gold was told to stop. Raising `CUDA_DEVICE_MAX_CONNECTIONS` to 32, creating the guest stream first, giving it the highest stream priority, and putting Gold on a non-blocking stream all left the guest at 0. Refill spawns and counter visibility were ruled out too. Your conclusion was "one grid, one total": while a second kernel is resident, the controller's view is incomplete.

## What the UNC work says about NVIDIA scheduling

From J. Bakita and J. H. Anderson (UNC Chapel Hill): "Hardware Compute Partitioning on NVIDIA GPUs" (RTAS 2023) and "Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management" (RTAS 2024). These are their findings, reported on their hardware. Treat them as hypotheses for this card until a log here confirms them.

- **Channels.** Every kernel launch goes through a hardware channel, and a task's channel count limits its parallelism. CUDA defaults to eight. Using more streams than channels causes false dependencies: a ready kernel waits behind unrelated work. A channel isn't freed until the last kernel in its stream has fully dispatched. Streams waiting for a channel don't get one in first-come order, so later kernels can cut ahead.
- **Runlists.** A runlist runs one task at a time, and tasks on it take turns in time slices. Copy and compute engines can share a runlist, so they aren't independent even when CUDA reports separate copy engines.
- **The dispatcher is greedy.** The hardware dispatcher hands out blocks from every runnable kernel equally. A flood of kernels takes SMs from everyone.
- **Stream priorities can evict.** Kernel slots fill in priority order, and higher-priority kernels can evict lower-priority ones from them. That lets one partition block another. Their advice: keep pending kernels under the slot count, or don't use stream priorities.
- **SM count is a smooth speed actuator.** In their YOLOv2 study, execution time fell smoothly as TPCs were added. They describe TPC count as a proxy for priority, with diminishing returns past a point. That supports GPUtronic's choice of SM count as the actuator.
- **The SM disable mask works on TPCs.** A field in NVIDIA's kernel descriptor, which they call the SM mask, actually disables TPCs (SM pairs). Their user-space library `libsmctrl` sets it per kernel, per stream or globally. That makes a hardware fence, not a time share. In their study, a malfunctioning competitor pushed a YOLOv2 instance from 24 ms per frame alone to 82 ms, and a 4/4 TPC split brought it back to about 45 ms. The limits: 64 TPCs at most, no coverage for kernels launched from the GPU itself, and it works by hooking undocumented CUDA internals.

## Standing rules

- Kp 0.55, Ki 0.08, Kd 0, deadband 0.08, 82-block launch, floor 2. Physical SMs 82 and 83 stay unlaunched. None of these change.
- **The controller's view has to be complete.** The guest work is a correctness question, not a performance one. No uncounted work runs alongside the throttle. Idle-exit exists so the controller sees everything on the card, not to hand capacity to another kernel. Every kernel sharing the card in these tests counts its own work.
- The guest has to be schedulable and counted at any load, from the floor to the ceiling.
- **Accepted limit:** on the stock driver, idle SMs spin at the GPU clock instead of sleeping. That cost is accepted.
- **Out of scope:** custom drivers, per-SM sleep, per-SM clock or power gating, thermal or power logic in the controller, and anything below the CUDA runtime and driver API. Stock, documented CUDA features are in scope. Whether `libsmctrl` falls inside this line is an open question for item 4. Don't assume it does.
- Idle-exit stays a test switch, off by default. Any new switch is test only and off by default. Unset, the binary behaves exactly like `740efea`, except for anything Ben approves under item 1.
- One GPU job at a time. Stop any run at 83 °C; the log up to the stop is the result.
- No number that isn't measured in this pass or already in the repo, quoted from its file. An early stop or a ruled-out idea is a result, not a failed run.
- The hold region is a deadband, not hysteresis. The Kalman filter is described only as the code uses it, which is logged, not used for control. No motor-control terms for the loop.
- Leave every earlier results branch, the notes branch, `archive/` and the tenth-pass directives as they are.

## 1. Make Z = 1 the home state, with commanded Z as the opt-in

- Report what the binary targets with every switch unset, and where in the code that's set. Report how a commanded Z is given today, and which tests command it. The three `CHECK` targets (about 1.47–1.50) are commanded.
- If unset doesn't already mean "target Z = 1, run at free-run speed", propose the smallest change that makes it so, with commanded Z as an explicit opt-in. **[Ben to approve before it changes the unset behaviour.]** Until he does, put it behind a test switch.
- Run one Z = 1 hold on the full-pipe plant and on the 1.2 load step. Report Z, percent in band, and the count. Does Z = 1 sit at free run without the integral winding toward the ceiling?
- Rerun `make gold`, `make transfer` and `make episode` and record the three `CHECK` lines next to 1.468, 1.497 and 1.443.

## 2. The integral clamp as a guardrail

The clamp's job is to almost never trip. Measure it that way.

- Add a trip counter if one doesn't exist: the fraction of ticks with the integral on the rail, per hold, printed on the summary line. Reporting only, off the control path.
- Run the 1.2 load step (`logs/2026-10-17_tenth-pass-continued/steps.txt`, `loadfile ... 420`, `GPUTRONIC_TRACE_BIN_US=1000`, idle-exit and guest unset) at `GPUTRONIC_ICLAMP_SMS=15`, to finish the 6/9/12/15 sweep. Report it with the same step table.
- From the four runs, report the smallest clamp where the trip rate is zero on every hold, and what each wider setting costs on the down steps and the overshoot. Don't pick for Ben.
- **The down steps got worse at 9 and 12.** Per the guardrail principle, a wider clamp alone isn't the fix. Look upstream for why the integral needed that range at all: the calibration (`sms_per_u`), the feedforward count, or the plant's response to the 1.2 load. Name the cause you find, or say it's still open.
- **[Ben to confirm the compiled default: 9, 12, or keep it a switch for one more pass.]** If he keeps it a switch, use 9 for every closed-loop run in this pass.

## 3. The driver owns physical limits

A test, not a feature. During one Z = 1 hold and one commanded-Z hold, log the SM clock (via NVML or `nvidia-smi`, read only) beside Z. If the driver drops clocks for thermal or power reasons, does the controller fight it by adding SMs and winding the integral? Report what you see. Add no thermal or power logic. If the loop does fight the driver, describe how, and propose the smallest decoupling for Ben to review.

## 4. The guest: a complete view of every resident kernel

**Goal.** Correctness. When a second kernel is resident, its work is counted as it happens and is visible to the controller. The guest's rate shows on every one-second line, at any Gold count.

**First, confirm the shape and test the UNC explanation.** Rerun phase C (`run 40`, guest 28 blocks at 8000 ms, idle-exit on, `GPUTRONIC_NO_CSV=1`, the full-pipe switches, clamp at the item 2 setting) with per-block entry timestamps on. The hypothesis is that Gold's 82 resident blocks keep their SMs occupied even when idle and spinning, so the greedy dispatcher has nowhere to place guest blocks until Gold exits. Check it against the logs. Also check the UNC rules on this run: streams within the channel count, and no stream priorities (the earlier priority test may have caused eviction rather than helped).

**Angles, in this order. Each is a hypothesis.**

1. *One grid, two totals.* Bring the second workload inside the Gold launch. Blocks above `active_sms` run a guest body and add to their own counter, instead of idling. The controller reads two totals from one grid: Gold's, which it governs, and the guest's, which it sees. Test switch, off by default. Report whether Z holds in band, and the guest rate against its phase B solo rate (9,040,655 pulses/s).
2. *A per-kernel view.* One counter per launched grid, read on the same tick as `total_work_pulses`, printed beside Z. Reporting, not control.
3. *Hardware partition with a stock feature: green contexts.* Check whether the CUDA driver API's green contexts (SM resource partitioning) work on this card and driver. If they do, give Gold one partition and the guest the SMs the throttle has idled. The controller still counts the guest. Report whether the guest runs while Gold is resident, and whether Z holds. If they don't work, record the error. That rules it out.
4. *Hardware partition via the TPC mask (`libsmctrl`): evaluation only.* Don't integrate or run it yet. Write a short case: what it touches (undocumented CUDA internals, kernel descriptors), whether that falls under "custom drivers" or "below the driver API", whether it gives anything green contexts don't, and the 64-TPC limit against this card. Argue for or against it. **[Ben to decide whether it's allowed.]**
5. *Launch order.* Launch the guest before Gold, then start Gold with idle-exit. Does the guest keep its SMs, does Gold's count account for them, and does Z still hold?
6. *The clamp and calibration with more than one resident grid.* Only if an angle above gets a second grid running alongside Gold. Report `sms_per_u`, feedforward and the clamp with and without the guest resident. Leave the gains alone.

Fix nothing in Gold's controller to make room. If every angle is ruled out, say plainly that the view is still incomplete with a second kernel resident, and how each angle was ruled out.

## 5. What success looks like

These are criteria, not predictions.

- **Home state.** The unset target is reported. Either Z = 1 is the default or Ben has a proposal to approve. A Z = 1 hold runs at free run without the integral winding.
- **Guardrail.** The 15-SM run is in, the trip rate is reported for 6/9/12/15, and the down-step regression has a named upstream cause or is plainly marked open.
- **Driver orthogonality.** The clock-versus-Z log shows whether the loop fights the driver.
- **The guest is counted or understood.** Either one angle gives a phase C run where the guest's work is counted on every one-second line while Gold holds Z in band, at 28 blocks and at least one other count, or every angle has a logged reason it was ruled out. The `libsmctrl` case is written either way.
- **No regressions.** Unset, the `[CTRL]` line shows Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, launch 82. Every new switch is off by default. All three checks pass.

## 6. Report and checks

Report at `docs/analysis/eleventh-pass-2026-10-18.md`, in the style of the tenth-pass reports. Include:

- each item's result, with the three `CHECK` lines;
- the 15-SM step table beside the 6, 9 and 12 results, with trip rates;
- the clock-versus-Z findings;
- each guest angle, with timestamps and rates;
- the `libsmctrl` case;
- any switch added, with confirmation it's off by default;
- any directive you argued with, and why;
- the item 5 criteria, with a plain yes or no for each.

Checks:

- `make gold`, `make transfer` and `make episode` build and pass `CHECK` with every switch unset.
- Every number in the report matches a committed log in `logs/2026-10-18_eleventh-pass/`.
- `main` and every earlier results branch are unchanged (`git ls-remote origin`).
