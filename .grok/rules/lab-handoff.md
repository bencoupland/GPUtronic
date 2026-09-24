# Lab handoff (Grok Build auto-load)

You are a second seat on GPUTronic, not a new project. Model is Grok 4.7 even if x.ai/cli still says 4.6.

Before any plant, edit, or commit, read these with a file tool. They are not all instruction files:

- `GROK-NOTES.md` (gitignored; still read it). The **2026-09-23 pickup** is the current shift. The 2026-09-21 pickup under it is history.
- `STATUS.md`
- `EPISODE.md`

Then follow this:

- The control is `active_sms`. SMs past that count skip the work and do not increment the pulse counter. At least 4 stay on. Do not switch the control back to sleep duration. The old source is in `archive/gold-1.0.0-nanosleep/`.
- Do not change Kp or Ki to close the gap under Z=1.5. Do not let the last SM overwrite `total_work_pulses`.
- Do not clone llama.cpp or add another test kernel. The open job is the same program tracking a requested rate on the kernels already here when the workload changes, without a new manual fit for each one.
- The three checks passed on 2026-09-23 after the switch to SM count (driver 595.91.07, CUDA 12.8). Re-run gold, then transfer, then episode, if the source or the driver changed or the date has moved. A pass means the counter is alive and the rate moved the right way. It does not mean Z sat on 1.50.
- Do not commit unless Ben asks. The tree is dirty on purpose.
