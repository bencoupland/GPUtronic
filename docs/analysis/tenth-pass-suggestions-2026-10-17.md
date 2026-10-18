# Tenth pass, suggested 17 October 2026

This is a suggestion from the seat on the RTX 5080. It is not a directive. No kernel was launched to write it. The lab checkout at `/media/dad/1TB HDD/GPUtronic` was left where it was, on `b8b66e7`. GitHub `main` is `b611a2b` (pull request #5). The next GPU run starts when Ben, or a directive on `controller-fixes`, names the question. Until then the gains, the clamp, the switches, and the idle behaviour stay as they are.

The question proposed below is the one already named on the ninth-pass landing page: hold Z while another kernel uses the SMs this loop has turned off.

## Where the lines are

| Line | Commit | What it holds |
|---|---|---|
| GitHub `main` | `b611a2b` | Pull request #5. The ninth-pass directive is in the tree. The ninth-pass results are not. |
| `test-results/2026-10-13_ninth-pass` | `e818644` | The landing-page README, the rename of the function frame, and the closed-loop log. Not an ancestor of `b611a2b`. |
| Lab folder | `b8b66e7` | 18 commits behind GitHub `main`. Clean at the time of this note. |
| `origin/controller-fixes` | `e1556f8` | The ninth-pass directive, already merged by pull request #5. |

A measurement for this suggestion belongs on a new `test-results/YYYY-MM-DD_tenth-pass` branch, started from GitHub `main` at `b611a2b`. The lab folder is the wrong tree to build it in until that folder is updated. Merging `e818644` is a separate decision, and it is Ben's.

## What the ninth pass already measured

The run is `logs/2026-10-13_ninth-pass/closed.txt` on `e818644`. The command was `./build/gputronic_gold run 25` with `GPUTRONIC_PEAK_ACC=32`, `GPUTRONIC_PEAK_ROUNDS=512`, `GPUTRONIC_THREADS=512`, `GPUTRONIC_PULSE_LOCAL=1`, `GPUTRONIC_CTRL_CACHE=1`, and `GPUTRONIC_NO_CSV=1`. Gains stayed at Kp 0.55 and Ki 0.08. Deadband 0.08. Launch 82. Floor 2. Idle blocks kept spinning. Nothing else was on the card except Xorg and Cinnamon. Driver 595.99.02, CUDA 12.8, `sm_120`.

From 4 s through 25 s there are 22 one-second samples. Their mean Z is 1.502. The smallest is 1.439 and the largest is 1.561, so the largest distance from 1.5 is 0.061. All 22 are inside the 0.08 band. Eighteen samples are at 54 SMs, three at 55, and one at 57. Mean count 54.3.

The `[STATS]` line on the same file is a different window. It covers the whole run, including calibration: mean Z 1.4964, standard deviation 0.0363, count 47..64, `integral_on_rail=0`, and 366683 of 373822 PI ticks held by the deadband. Quote 1.502 for the settled window. Quote 1.4964 only as that whole-run line.

`logs/2026-10-13_ninth-pass/closed.dmon` peaks at 208 W on one sample while the clock is still climbing. From the fourth sample on, the board sits at 174 W, then 160 W, through the calibration steps, and the closed-loop stretch is 181–182 W at 59–60 °C, SM busy 100%, clock 2857–2887 MHz. Power-violation and thermal-violation stay 0. The held power matches the full-pipe closed loop, which sat near 183 W.

That full-pipe loop, the night before, is the same kernel and the same gains with no load either: mean Z 1.497, count 54 or 55, 22 of 22 inside the band (`docs/analysis/full-pipe-2026-10-13.md`). Two undisturbed runs agree. A third one, with the same switches, would reprint them.

The reference on the ninth run was 45.24 million pulses/s. Feedforward count 55.9. `sms_per_u` 31.0. The integral clamp printed as 3.28 SMs of 82. The integral never sat on that rail. The loop stepped onto the feedforward count and stayed there.

## The question

On the 32×512 kernel, with the controller on, can a second kernel in the same process take the SMs whose blocks have returned, and does Z of the first kernel stay inside the 0.08 band while that happens?

The ninth-pass README, on `e818644`, already states the destination in one line: the next step is holding Z under a real competing GPU load. The two closed loops above had no second kernel. The open-loop sweep had no second kernel. This is the measurement those runs left open.

## Why the idle block has to return

On the current default, a block past `active_sms` stays resident, skips the FMAs, and reads the control words again. It still owns the SM. The full-pipe sweep left that path on, and 4 SMs still drew 121 W. The eighth pass measured two other paths, on the empty kernel, and left both as test switches (`docs/analysis/eighth-pass-2026-10-13.md`, section 2):

- `GPUTRONIC_IDLE_EXIT=1`. An idle block returns. The host launches a replacement on its own stream when the count rises. At 4 SMs the empty kernel drew 71 W, against 84 W while spinning. The step 82 → 42 → 82 showed in the tally at 1.14 ms down and 0.71 ms back.
- `GPUTRONIC_RELAUNCH=1`. The whole grid retires and the host launches the commanded count, after a hold so one tick does not relaunch. At 4 SMs the empty kernel drew 66 W. The sync was 832 µs down and 609 µs up.

Setting both switches turns both off. The compiled default is still the spinning block.

A second kernel only gets an SM if the first kernel's block has left it. The 5 October share run, on the archived tree, is the earlier version of that fact: a resident grid whose idle blocks stayed put left the guest near 0.10 of its solo rate, and launching only the working blocks left the guest a real rate. That run is `archive/lab-2026-10-05`. It is a different controller. It is not a result for this plant. The device property `concurrentKernels` was 1 there, which is why two kernels in one process could occupy the chip together.

A second process is a different experiment. The 5 October run beside Space Marine 2 finished guest work at about a third of that guest's solo rate, and the rate was the same at 21 blocks and at 84. The SM count was not what moved. A game, or any other process, timeslices with this one. It does not sit in the holes `GPUTRONIC_IDLE_EXIT` opens. Leave that for later.

`GPUTRONIC_RELAUNCH` also opens holes, because the grid is only as wide as the command. It does it by retiring every block, including the ones that are still working. The ninth run's `[STATS]` line reports 74 SM-count moves. Many of those are the calibration steps. A relaunch on each of them stops the pulse stream for the sync. Exit leaves the working blocks resident and only fills a hole when the count rises. That is the switch for this test. Relaunch stays off. Adopting either switch as the compiled default is a later decision, after this plant has been measured with exit on.

The eighth-pass watt figures are the empty kernel. Exit on the 32×512 kernel has not been measured. Phase A below is that measurement. The 121 W floor at 4 SMs was the spinning full-pipe plant, and it is not a prediction for the exit plant.

## The run, once the question is accepted

One GPU job. Stop the run if the die reaches 83 °C. Keep each phase inside a minute. The stock driver still owns clocks, power, and temperature. Kp 0.55, Ki 0.08, deadband 0.08, floor 2, launch 82, scaled clamp. Tensor sensor off. `GPUTRONIC_ICLAMP_SMS` unset. `GPUTRONIC_RELAUNCH` unset. `GPUTRONIC_IDLE_QUIET` unset.

Build from GitHub `main` at `b611a2b`, with CUDA 12.8 and `-arch=sm_120`. Before the new switch exists, `make gold`, `make transfer`, and `make episode` with every test switch unset, and record the three CHECK mean-Z lines. Those checks are the build this pass has to keep passing. The tenth-pass result is the three phases below.

The guest does not exist in this tree. It is a test switch, off unless the run sets it. A small addition, in the Gold process:

- After the closed loop has settled, launch a second kernel on its own non-blocking stream.
- The grid size is fixed. It does not read `active_sms`. It does not add to `total_work_pulses`. It adds to its own mapped word, with `atomicAdd`, so its rate cannot move Z.
- The body is the same 32-chain, 512-round independent FMA, so a finished guest pulse is the same work as a finished Gold pulse. One pulse is still one finished block.
- The one-second line prints the guest rate next to Z and the SM count. A `[SPAWN]` line already prints when exit refills a hole. Leave that print on.
- Unset, the binary does not launch the guest, and the checks stay on the kernel they already measure.

Size the guest at 28 blocks for the first cut. The settled count on this plant is about 54 of 82, which leaves 28 launched blocks idle, and physical SMs 82 and 83 stay empty for the desktop. Twenty-eight is a hypothesis taken from those holds. If phase A settles somewhere else, use the idle count from phase A and write the number down. Do not put the guest on the two desktop SMs.

Three phases, in this order. Each one can end the pass.

1. Phase A, no guest. `run 25` with the full-pipe switches plus `GPUTRONIC_IDLE_EXIT=1`. Same window as the ninth run: mean Z, min, max, how many of the 22 samples sit inside the 0.08 band, and the SM counts, from 4 s through 25 s. Power and temperature from `nvidia-smi dmon`. If Z leaves the band, stop. Exit changed the plant, and adding a guest on top of that would mix two causes. Write the log and the note.
2. Phase B, guest alone. Gold is not resident. Ten seconds at the block count chosen above. Record the guest's own pulse rate, power, and temperature. That rate is the solo figure phase C is compared with.
3. Phase C, both. Gold as in phase A. The guest starts after 8 s, so calibration and the first settle are still Gold alone. Hold 40 s, or until 83 °C. Record mean Z and band compliance on the one-second samples after the guest starts, the SM count, the guest rate as a fraction of phase B, the `[SPAWN]` lines, power, temperature, and the violation flags.

What the numbers would mean, without changing anything:

- Z stays inside the band and the guest holds a large fraction of its solo rate. The loop kept the requested rate while the freed SMs did other work. That is the result the landing page asked for.
- Z stays inside the band and the guest is near zero. The blocks did not actually give the SMs up, or the guest could not start. The exit switch did not yield under a second launch. That is also a finished result.
- Z rises out of the band and the count climbs. The guest took time, power, or an SM the first kernel still needed. The clamp is 3.28 SMs on this plant. A climb that stops on that rail is the same authority limit the seventh pass already measured on the lighter cached plant, now on a kernel whose rate really follows the count. Record where it sits. Leave Kp, Ki, and the clamp where they are.
- The die reaches 83 °C. Stop. The log up to the stop is the result.

`work_scale_milli` stays out of this pass. On the 32×512 kernel it skips pulses and leaves the FMAs running, so the counter reports a slower job while the SMs do the same work. The full-pipe note already says that.

## What stays as it is

- Kp 0.55, Ki 0.08, Kd 0, deadband 0.08, scaled integral clamp. The eighth pass tried clamps of 6, 9, and 12 SMs on a 25 s load of scale 1.2 and did not widen the default. The ninth run never touched the rail. A wider clamp needs its own written plan.
- The 32×512 kernel, the pulse batch, the control-word cache, and idle-exit stay off in the binary the checks measure. A run sets them. Promoting any of them is a separate commit after a measurement, and only if Ben asks.
- The command floor stays 2. The launch stays 82 blocks. Physical SMs 82 and 83 stay unlaunched.
- No hour hold. The thermal shape asked for in the eighth pass was already flat inside a minute.
- The GTX 1080 stays on the shelf. `docs/analysis/gtx1080-bringup-plan.md` is still the note. Driver 595 does not run that card.
- The counter that keeps counting after a launch returns stays unbuilt. It is the next meter when Ben asks for it. It answers a different question: what watches the pulse between kernels. This pass is two kernels at the same time.
- A game co-run stays off this pass. GPUTronic only trims a kernel that counts its own work and reads `active_sms`. The beside run of 5 October already showed a second process ignoring the SM count.
- Another undisturbed `run 25` of the spinning 32×512 plant. The 1.497 and the 1.502 are enough.
- Ben's X profile. Read on 17 October 2026, `@benracoupland` already says he is the creator of GPUtronic and links the project. The sentence drafted in the ninth-pass directive was not applied, and it does not need to be. The profile is his.

## The other open items, after this one

These are real, and they wait until the question above has an answer or Ben picks a different one.

1. Land `e818644` on `main`, or leave it as the results branch it is. GitHub `main` still opens the README with "Closed-loop control of instruction-flow throughput on a GPU." The cruise-control line, the thread link, the Pascal governor line, the Results section, and the renamed function frame `docs/GPUtronic-GOLD-1.0-FUNCTION-FRAME.md` are on `e818644` only. The directive that asked for them is already on `main`. Merging the results branch is the landing. This note does not do it.
2. A disturbance that skips the FMA block, and does not pulse for the skipped work. That changes how much work one SM finishes. It has not been run. The scaled clamp can add about 3.28 SMs. On this plant `sms_per_u` is about 31, and the open-loop rate is close to proportional to the count, so that authority is about 6% of the settled rate. A loss much larger than that will park Z high for the same reason the seventh-pass load did, on the lighter plant, at Z 1.639. Size a later run inside that authority if the point is to see the loop reject it. Leave the clamp alone either way.
3. Which idle path becomes the compiled behaviour. Exit and relaunch both cut power on the empty kernel, and both moved the command in under 2 ms, with the desktop still on the 16.7 ms refresh. Phase A is the first time exit is under the PI on the 32×512 kernel. The choice waits for that log.
4. More rounds of this FMA block. The eighth pass already saw 2048 rounds sit at 46.6×10^12 FLOP/s and 209 W, next to 47.0×10^12 and 210 W at 512 rounds. The 360 W board limit was not the stop. A tensor-core body, or a kernel that moves a lot of memory, would be a new plant.

## How to read a later log

Quote the CHECK line, not the STATS line, for the three gates. For a closed loop, quote the one-second samples from the settled window and say which seconds. Printed Z clips at 10, so a low-count row needs the pulse rate as well. Issued FLOP/s on this kernel is the measured pulse rate times 32 lanes times 32 chains times 512 rounds times 2. The eighth-pass empty-kernel hour is a counter result. The script product from that search is not issued work.

The lab folder does not contain this note until the branch is merged or the folder is updated. The branch is `notes/tenth-pass-suggestions-2026-10-17`.
