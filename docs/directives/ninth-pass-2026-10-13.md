# Ninth pass: the README as a landing page

This pass is text only. There are no GPU runs and no code changes. Start from `main` at `b8b66e7`. Post the change to a new branch, `test-results/2026-10-13_ninth-pass`. Do not merge anything, and do not force-push.

## Where things stand

- Ben posted a two-post thread on X about GPUtronic on 13 October. [Post 1](https://x.com/benracoupland/status/2109921630009053318) opens with "What if a parallel processor could have cruise control?" [Post 2](https://x.com/benracoupland/status/2109922402457296948) replies to it and links to the repo.
- The README on `main` is honest and thorough. But someone arriving from the thread won't find what it promised. The words "cruise control" aren't there, the 5080 closed-loop result sits about 100 lines down, and the first table a reader reaches is the old sleep-duration data.
- The README has no link back to the thread.
- Two old spellings are left: the image alt text says `GPUTRONIC`, and the function-frame file is still `docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md`.

## Standing rules

- Text only. Do not touch `src/`, `include/` logic, gains, switches or results. Kp 0.55, Ki 0.08 and the 0.08 deadband stay as they are.
- Do not add any number that isn't already measured and written down in the repo. Quote each one from its source file.
- Keep the plain-English style the README already uses.
- Leave `archive/`, `logs/` and `results/` alone, including their old spellings.

## 1. Replace the line under the title

Replace "Closed-loop control of instruction-flow throughput on a GPU." with exactly:

> Cruise control for a parallel processor. The GPU counts its own finished work, and a PI loop decides how many SMs run, above the driver and below the workload.

## 2. Bring the 5080 result to the top

Right after the image, add a short block with the closed-loop result on the RTX 5080. Use the full-pipe run in `docs/analysis/full-pipe-2026-10-13.md`: the pulse rate tracked the SM count, and the closed loop held mean Z at 1.497 inside the 0.08 band. Name the source file. Leave the full section further down where it is.

## 3. Add a one-line status

In the same block, add one line saying the next step is holding Z under a real competing GPU load. Don't give a target number.

## 4. Fix the leftover spelling

- Change the image alt text from `GPUTRONIC` to `GPUtronic`.
- Rename `docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md` to `docs/GPUtronic-GOLD-1.0-FUNCTION-FRAME.md` with `git mv`. Update every link to it outside `archive/` and `logs/`. Right now that means `README.md` (three places), `AGENTS.md`, `GOLD.md`, `docs/README.md`, `docs/GOLD-1.0.0-release-post.md`, `docs/GOLD-1.0.0-x-thread.md`, and the comments in `include/gputronic.h`, `src/gputronic_gold.cu` and `src/gputronic_transfer.cu`. Comments only in those three source files.
- After the rename, run `git grep GPUTRONIC-GOLD` outside `archive/` and `logs/` and confirm it's empty.

## 5. Link back to the X thread

Add one line near the top block linking to post 1 of the thread: `https://x.com/benracoupland/status/2109921630009053318`.

## Checks

- Build the three standard targets with every switch unset, and confirm they still compile. No runs are needed.
- Check that every link in `README.md` still resolves after the rename.

## The report

Put it at `docs/analysis/ninth-pass-2026-10-13.md`. It should be short: the new top of the README as it now reads, the list of files touched by the rename, and the result of the grep and build checks.
