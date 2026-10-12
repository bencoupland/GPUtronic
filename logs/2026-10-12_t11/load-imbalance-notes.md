# Load imbalance across SMs (T11 follow-up)

Notes only. No code changed and no tests run for this note.

## What T11 showed
With all SMs reporting, open loop at 46 working SMs, the per-SM tallies summed to the pulse counter (ratio 1.000, correlation 0.995 at 100 ms). Across the working SMs, though, the work per bin was uneven: coefficient of variation 29%, range about 444k to 833k ops per bin (`sm_all_analysis.txt`). SM 0 sat at 0.75 of the average.

## Current thinking
The 29% spread is load imbalance, not hardware differences. Every SM on the 5080 is the same functional unit with the same maximum clock, and the SMs are not clocked individually. A healthy chip should give roughly equal work per SM for identical blocks. So something in how blocks or pulses are spread across SMs is skewing the work.

## Candidates to check
- **Block-to-SM placement.** The accumulator keys its slot by `blockIdx.x` (`src/gputronic_gold.cu`, `const int sm = blockIdx.x`), not by the physical SM. CUDA does not promise that 84 blocks land one per SM. If two blocks share an SM while another SM sits empty, the result looks exactly like this spread. Reading `%smid` per block settles it.
- **Contention on the mapped atomic.** Every pass ends in a mapped PCIe `atomicAdd` on `total_work_pulses`. If some SMs win that round trip more often, by distance to the PCIe interface or position in the GPC/L2 layout, they finish more passes.
- **Which blocks are working.** Blocks at or above `active_sms` idle, so the working set is always blocks 0..n-1. If those map unevenly onto GPCs, some GPCs are crowded and others are empty.
- **Idle polling.** Idle blocks now poll mapped memory with no pause. If an idle block shares an SM or GPC with a working one, it can steal issue slots or memory bandwidth.

## Next step
Find which SMs are starving at any instant:
1. Record `%smid` (and `%nsmid`) per block at launch, and write the block-to-SM map to the log.
2. Key the accumulator by physical SM as well as by block, and log per-SM work per 20 ms bin. At 1 ms, the per-bin counts are too coarse: correlation was 0.66.
3. Look for persistent losers (the same SMs low every bin) versus rotating losers (random each bin). Persistent points at placement or topology. Rotating points at contention.
4. Repeat at full width (84) and at 46, and with idle polling paused, to separate the candidates above.
