# Ninth pass: the README as a landing page, plus a controller-engaged dyno run

Items 1 to 5 and 8 are text only. Item 6 is the one GPU run in this pass, item 7 is cleanup on the box, and item 9 is a text change on Ben's X profile. There are no code changes. Start from `main` at `b8b66e7`. Post the change to a new branch, `test-results/2026-10-13_ninth-pass`. Do not merge anything, and do not force-push.

## Where things stand

- Ben posted a two-post thread on X about GPUtronic on 13 October. [Post 1](https://x.com/benracoupland/status/2109921630009053318) opens with "What if a parallel processor could have cruise control?" [Post 2](https://x.com/benracoupland/status/2109922402457296948) replies to it and links to the repo.
- The README on `main` is honest and thorough. But someone arriving from the thread won't find what it promised. The words "cruise control" aren't there, the 5080 closed-loop result sits about 100 lines down, and the first table a reader reaches is the old sleep-duration data.
- The README has no link back to the thread.
- Two old spellings are left: the image alt text says `GPUTRONIC`, and the function-frame file is still `docs/GPUTRONIC-GOLD-1.0-FUNCTION-FRAME.md`.
- The README shows the full-pipe result, but there is no fresh run with the controller engaged to sit next to the open-loop baseline. Item 6 adds one, and its numbers go in a short Results section.
- The old clone at `/workspace/GPUtronic` on the box lost its Git metadata. The working copy is `/workspace/GPUtronic-fresh`.
- The README doesn't say where the project came from. It started with the Pascal governor, released in January 2026.
- Ben's X profile ([@benracoupland](https://x.com/benracoupland)) says nothing about GPUtronic. Someone who clicks through from the thread doesn't land on the same story as the thread and the README.

## Standing rules

- Items 1 to 5 and 8 are text only. Do not touch `src/`, `include/` logic, gains, switches or results for them. Kp 0.55, Ki 0.08 and the 0.08 deadband stay as they are.
- Item 6 is the only item that runs hardware, and the only one that adds results or logs. Those go on the results branch, `test-results/2026-10-13_ninth-pass`.
- Item 7 touches only the box filesystem, outside the repo. Nothing is committed for it.
- Item 9 touches only Ben's X profile, outside the repo. Nothing is committed for it. It is text only, with no numbers.
- Do not force-push, and never merge to `main`. Do not turn any test switch on by default.
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

## 6. Dyno run with the controller engaged

Run the full-pipe workload on the RTX 5080 (sm_120) with the GPUtronic controller active. Keep the PI gains as they are: Kp 0.55, Ki 0.08, deadband 0.08. Keep the 82-SM ceiling and the floor of 2. Capture the Z trace.

Commit the raw logs to `test-results/2026-10-13_ninth-pass`, under `logs/2026-10-13_ninth-pass/`. Then add a short "Results" section to the README with three things from this run: the measured mean Z, band compliance (the share of samples inside the 0.08 band), and a comparison to the open-loop baseline. For the baseline, cite `docs/analysis/full-pipe-2026-10-13.md` and the fixed-SM logs in `logs/2026-10-13_full-pipe/` by path.

Only use numbers measured in this run or already written in those files. Don't invent a target. If the run fails, say so in the Results section, with what went wrong, rather than filling in numbers.

## 7. Clean up the damaged clone

On the box, move `/workspace/GPUtronic` to `/workspace/_quarantine/GPUtronic-damaged-2026-10-13` (or remove it). It lost its Git metadata, and `/workspace/GPUtronic-fresh` is the working copy. This is outside the repo, so nothing gets committed. Just report what was done.

## 8. Add the Pascal governor lineage line

Add one line to the README naming the Pascal governor, the simple GPU governor released in January 2026, as the project that started GPUtronic. It gives the current closed-loop work a story arc. Keep it to one line.

## 9. Add an X profile blurb

Add a short "About this project" blurb to Ben's X profile ([@benracoupland](https://x.com/benracoupland)), one or two sentences pointing at the GPUtronic repo (`https://github.com/bencoupland/GPUtronic`) and the dyno result in the README. Anyone who clicks the profile after reading the thread should land on the same story. Text only, no numbers, and keep the GPUtronic spelling. This is an account-side change, like item 7: nothing gets committed. Put the exact blurb text in the report, and say whether it was applied to the profile.

## Checks

- Build the three standard targets with every switch unset, and confirm they still compile.
- Check that every link in `README.md` still resolves after the rename.
- Check that every number in the Results section matches the committed log in `logs/2026-10-13_ninth-pass/`.
- Run `git grep -i gputronic` outside `archive/`, `logs/` and `results/`, and confirm every hit uses the GPUtronic spelling (or the unchanged `GPUTRONIC_*` environment variables).
- Confirm `/workspace/GPUtronic` is gone from `/workspace`.
- Confirm the Pascal governor lineage line is in the README.
- Confirm the X profile blurb is live on @benracoupland, links to the repo, and has no numbers.

## The report

Put it at `docs/analysis/ninth-pass-2026-10-13.md`. It should be short: the new top of the README as it now reads, the list of files touched by the rename, the Results section as written with the path to its logs, where the damaged clone went, the X profile blurb as written and whether it is live, and the result of the checks.
