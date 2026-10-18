# Ninth pass, 13 October 2026

Text first, then one closed-loop run, then the cleanup check. Nothing was merged. The branch is `test-results/2026-10-13_ninth-pass`, started from `main` at `b08a791`.

## The top of the README

```
# GPUtronic

Cruise control for a parallel processor. The GPU counts its own finished work, and a PI loop decides how many SMs run, above the driver and below the workload.

Reference: **GPUtronic 1.0 Gold**

![GPUtronic](docs/assets/gputronic-mark.jpg)

On an RTX 5080, the pulse rate tracked the SM count, and the closed loop held mean Z at 1.497 inside the 0.08 band. That run is docs/analysis/full-pipe-2026-10-13.md.

The next step is holding Z under a real competing GPU load.

GPUtronic started with the Pascal governor, a simple GPU governor released in January 2026.

The thread that points here starts at https://x.com/benracoupland/status/2109921630009053318.
```

The 1.497 figure is the earlier full-pipe write-up, left where it was further down the README as well. The new run is the Results section below.

## Rename

`docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md` is now `docs/GPUtronic-GOLD-1.0-FUNCTION-FRAME.md`. The links updated with it are `README.md` (three places), `AGENTS.md`, `GOLD.md`, `docs/README.md`, `docs/GOLD-1.0.0-release-post.md`, `docs/GOLD-1.0.0-x-thread.md`, and comments in `include/gputronic.h`, `src/gputronic_gold.cu`, and `src/gputronic_transfer.cu`. The comments are the only edits in those three source files. The copy of the function frame at the end of `README.md` still matches the standalone file.

`git grep GPUTRONIC-GOLD` outside `archive/` and `logs/` is not empty. The three remaining lines are in `docs/directives/ninth-pass-2026-10-13.md`, which is the note that asked for the rename and still quotes the old path. That file was left as the record of the ask.

## Results

The section in the README, measured on this RTX 5080 (driver 595.99.02, CUDA 12.8, `sm_120`):

From 4 s through 25 s there are 22 one-second samples in `logs/2026-10-13_ninth-pass/closed.txt`. Mean Z is 1.502 on both the raw and the filtered column. All 22 are inside the 0.08 band. The largest distance from 1.5 on those samples is 0.061. Of the 22 samples, 18 were at 54 SMs, 3 at 55, and 1 at 57.

The open-loop baseline cited next to that is `docs/analysis/full-pipe-2026-10-13.md` and `logs/2026-10-13_full-pipe/`: Z 1.013 at 82 SMs (`n82.txt`) and Z 1.459 at 56 SMs (`n56.txt`).

The run printed target 1.50, Kp 0.550, Ki 0.0800, deadband 0.080, launch 82, floor 2. `nvidia-smi dmon` in `logs/2026-10-13_ninth-pass/closed.dmon` peaked at 208 W and 60 °C. The power-violation and thermal-violation columns stayed 0. The program exited at 25 s.

## The damaged clone

`/workspace` is not on this machine, so `/workspace/GPUtronic` was not there to move and `/workspace/GPUtronic-fresh` was not there either. Nothing was quarantined. The checkout at `/media/dad/1TB HDD/GPUtronic` still has its git directory and was left as it was.

## Left for Ben

Item 9 was not applied. Nothing was changed on @benracoupland. The blurb, for him to paste when he is at the account:

GPUtronic is cruise control for a parallel processor. The repo, and the dyno result in the README, are at https://github.com/bencoupland/GPUtronic.

## Checks

- `make gold`, `make transfer`, and `make episode` built with the test switches unset. The library build still warns that `mode_run` and the other CLI functions are unused under `-DGPUTRONIC_NO_MAIN`. That warning was already there.
- Every markdown link in `README.md` resolves.
- The Results numbers 1.502, 22 of 22, and the 18/3/1 SM counts are the one-second lines from 4 s through 25 s in `logs/2026-10-13_ninth-pass/closed.txt`. The baseline figures 1.013 and 1.459 are quoted from `docs/analysis/full-pipe-2026-10-13.md`.
- Prose added in this pass spells the project GPUtronic. File names, function names, and the `GPUTRONIC_*` macros are unchanged. The one `GPUTronic` string outside `archive/`, `logs/`, and `results/` is the archive path named from `docs/GPUtronic-essay.md`.
- The Pascal governor line is in the README.
- The work went in the Start here order: the README text and the rename, then the dyno, then the cleanup check.
