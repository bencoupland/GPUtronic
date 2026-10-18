# Tenth pass, continued: get the freed SMs to a second kernel, and unstick the integral under load

This continues `docs/directives/tenth-pass-2026-10-17.md`. It is not an eleventh pass. It is based on the results on `test-results/2026-10-17_tenth-pass` (`4224435`, report `docs/analysis/tenth-pass-2026-10-17.md`) and on Ben's conclusions from reading them. Start from that results branch at `4224435`, so the guest switch is already in the tree. Post everything to a new branch, `test-results/2026-10-17_tenth-pass-continued`, with logs under `logs/2026-10-17_tenth-pass-continued/`. Do not force-push. The only merge this directive asks for is item 1.

## Start here

Ben's usage reset tonight, so the full budget is available. Complete every item in this directive: items 1 to 3, then the report and checks in item 5.

## Where things stand

- **The throttle works.** The primary goal of GPUtronic is the throttle: hold Z in the band, at whatever rate is asked for, anywhere up to free-run, so the workload's pressure on the card is set by the loop instead of by the workload. The ninth pass (mean Z 1.502, 22 of 22 in band), the tenth pass phase A with idle-exit (1.4825, 22 of 22, all at 54 SMs) and phase C (1.4963, 32 of 32) all held it. That part is done. Nothing in this directive should change it.
- **The guest got nothing.** In phase C the guest's `[TEL]` field was 0 on all 32 samples. The stop line was `[GUEST] stopped pulses=448 live=28`, which is 28 blocks × 16 warps, one pass each. So the 28 guest blocks did start, but each warp finished exactly one pass, and only after Gold was told to stop. On its own (phase B) the same guest held 9,040,655 pulses/s. The SMs idle-exit freed were not handed over to the second kernel while Gold was running. During phase C Gold's count also went from 54 to 57 (`[SPAWN] base=54 n=3`), and the report gives no cause for that.
- **The integral sticks under load.** In the load-step run at scale 1.2, the integral was on the clamp for 100% of the control ticks in the last 30 s of every loaded hold (three of three). The count reached 61, only 55.0–57.3% of those bins were in band, and the tail that stayed in band was 26, 19 and 3 ms. The returns to 1.0 were 96.1–97.5% in band with 0% on the clamp. The printed clamp authority was 3.28 SMs of 82.

## Standing rules

- Kp 0.55, Ki 0.08, Kd 0, deadband 0.08, 82-block launch, floor 2. Physical SMs 82 and 83 stay unlaunched. None of these change in this pass.
- **No uncounted concurrent work.** Every kernel the loop shares the card with in these tests counts its own finished work, the way the guest does. Nothing runs on the card that the report can't account for.
- **The guest must be schedulable under any load.** Whatever count Gold is at, from the floor to the ceiling, a second kernel in the process must be able to start on an SM Gold has left and make progress there. A fix that only works at one count, or only before Gold launches, isn't the fix.
- **Accepted limit.** On the stock driver, an SM nobody is using doesn't sleep. It sits at the GPU clock. That cost is accepted for now. Don't try to measure it away or work around it in this pass.
- **Out of scope.** Do not propose, prototype or write about a custom driver, per-SM sleep, per-SM clock or power gating, or anything below the CUDA runtime. That is a separate future project, and Ben will open it himself.
- Idle-exit (`GPUTRONIC_IDLE_EXIT=1`) stays a test switch, an experiment, off by default. Phase A shows it holds the band. Don't promote it.
- Any new switch is test only and off by default. Unset, the binary behaves exactly like `4224435` with the guest unset, and like `main` for the `[CTRL]` line.
- One GPU job at a time. Stop any run at 83 °C; the log up to the stop is the result.
- No number that isn't measured in this pass or already written in the repo. Quote each one from its source file. The causes listed in item 2 are hypotheses to test, not findings.
- The hold region is a deadband, not hysteresis. The Kalman filter is described only as the code does it.
- Leave `test-results/2026-10-17_tenth-pass`, `test-results/2026-10-13_ninth-pass` (apart from item 1's merge), the notes branch and `archive/` as they are.

## 1. Merge the ninth-pass results to main (if not done)

Check first: `git merge-base --is-ancestor e818644 origin/main`. If it already is an ancestor, skip this item and say so.

If not, open a pull request from `test-results/2026-10-13_ninth-pass` into `main` and merge it with a normal merge commit. That lands the README landing page, the Results section, the renamed `docs/GPUtronic-GOLD-1.0-FUNCTION-FRAME.md` and the ninth-pass logs. Ben asked for this merge. If it conflicts, don't resolve it by hand and don't force anything. Stop the item, note the conflicting files in the report, and go on to item 2.

Note: the tenth-pass commit added the guest row to `docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md`, the old spelling. Once the merge has landed, the continued branch carries that row in the renamed file, and the old-spelling file doesn't come back.

## 2. Get the freed SMs to the guest

**Goal.** In a phase C run, with Gold holding Z in band, the guest makes real progress on the SMs Gold isn't using, and its rate shows on every one-second line.

**First, find out where the guest waited.** Add to the guest kernel, behind the existing guest switch, one mapped word per block: the `globaltimer` and `%smid` when the block first enters, and the time of its first published pulse. Print them at the end of the run beside the `[GUEST] launched at` time and the time Gold was told to stop. That separates three cases:

- the blocks entered only after Gold stopped (the guest waited in a queue the whole time);
- the blocks entered early but couldn't publish (a visibility problem with the mapped pulse word);
- the blocks entered early, on SMs Gold also held (sharing an SM, not getting a free one).

**Hypotheses to check, in this order.** These come from reading the code. None of them is measured yet.

1. *A shared hardware queue.* Gold already creates many streams (the wave streams, `kstream`, `side_stream`). The CUDA runtime maps streams onto a limited number of hardware work queues, `CUDA_DEVICE_MAX_CONNECTIONS`, which defaults to 8. If the guest stream lands on the same queue as a persistent Gold launch, the guest waits behind it until Gold returns. That would look exactly like 448 pulses at stop. Test it: rerun phase C with `CUDA_DEVICE_MAX_CONNECTIONS=32` set in the environment, and nothing else changed.
2. *Stream order and priority.* Create the guest stream before Gold's streams, or with `cudaStreamCreateWithPriority` at the highest priority. One run each. Report which, if either, changes the result.
3. *The refill spawns.* Idle-exit's `[SPAWN]` launches refill holes when the count rises. Check whether a pending spawn, or a refill at a base the guest could use, takes the SM before the guest's blocks are dispatched. The 54 to 57 move in phase C is the place to look.
4. *Visibility.* If case two above is what the timestamps show, check the guest's mapped counter path (`atomicAdd` on mapped host memory, `__threadfence_system`) against the one Gold uses, which does show up every second.

Fix the first cause the timestamps point to, as a test switch or as an environment setting the run documents. Fix only that. Don't change Gold's controller to make room.

**The run.** Phase C again, same as the tenth pass (`run 40`, guest 28 blocks at 8000 ms, idle-exit on, `GPUTRONIC_NO_CSV=1`), with the fix. Then the same with the guest at 54 blocks, and with a fixed guest at 40 blocks while Gold holds a target that settles near the floor, if `run [sec] [Z]` gets there. That covers the "under any load" rule. For each run report: mean Z, min, max and band share after the guest starts; the SM counts; the guest rate as a fraction of phase B's 9,040,655 pulses/s (or a fresh phase B at the new block count); the block entry and first-pulse times; the `[SPAWN]` lines; power and temperature.

If no fix works, that is the result. Report which hypotheses were ruled out and how.

## 3. Unstick the integral under the 1.2 load

Only after item 2 has a pushed result. Idle-exit unset and guest unset, so this is the same default plant as the tenth-pass load step.

The integral sat on the 3.28-SM clamp on every tick of every loaded hold. The eighth pass tried `GPUTRONIC_ICLAMP_SMS` at 6, 9 and 12 on the lighter cached plant, and none of them brought Z back into the band. That has never been run on the 32×512 full-pipe plant. Rerun the tenth-pass load step (`logs/2026-10-17_tenth-pass/steps.txt`, `loadfile ... 420`, `GPUTRONIC_TRACE_BIN_US=1000`) with `GPUTRONIC_ICLAMP_SMS` at 6, then 9, then 12. These are test settings. The compiled default clamp doesn't change, and adopting a wider one is Ben's decision after he sees the numbers. If none of the three runs takes the integral off the rail, try 15 only if Ben has already said yes to going wider. Otherwise stop and report.

For each run, use the tenth-pass step table: first bin outside, enters and stays, tail that stays, highest and lowest `z_hat`, SM before and at end, the share of the last 30 s in band, the share on the clamp, SM changes and counts in the last 30 s. Watch the returns to 1.0 as well. A wider clamp could make the down steps worse, and that has to show up in the table.

`work_scale_milli` is still a synthetic load: the counter reports a slower job while the FMAs keep running. Say so in the report.

## 4. Carried forward, not run here

- Phase A idle-exit stays an experiment. Its band result (22 of 22 at 54 SMs) stands. Which idle path becomes the compiled default waits until item 2 has been fixed and measured.
- A disturbance that skips the FMA block itself, more FMA rounds, a tensor-core or memory-heavy body, the GTX 1080, and a game beside the loop all stay open.
- The idle-SM clock cost on the stock driver (see Standing rules) stays accepted.

## 5. Report and checks

Report at `docs/analysis/tenth-pass-continued-2026-10-17.md`, short, in the same style as the tenth-pass report: item 1's outcome, the item 2 timestamps and which case they showed, each hypothesis tested and its result, the phase C tables with the fix, the item 3 step tables, any switch or environment setting added (and confirmation that it's off by default).

Checks:

- `make gold`, `make transfer` and `make episode` build and pass `CHECK` with every switch unset. Record the three `[CHECK] z_mean` lines.
- The unset `[CTRL]` line shows Kp 0.550, Ki 0.0800, Kd 0.0000, deadband 0.080, launch 82.
- Every number in the report matches a committed log in `logs/2026-10-17_tenth-pass-continued/`.
- `test-results/2026-10-17_tenth-pass` is unchanged (`git ls-remote origin`). `main` has changed only by item 1's merge, if that ran.
