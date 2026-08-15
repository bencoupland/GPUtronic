# AGENTS.md – GPUTronic Project

You are working on **GPUTronic**, FOC-style closed-loop control for GPU execution.

## Core Identity
- Treat the GPU as a controllable dynamical system
- Q-axis = useful work rate (aggregate atomic pulses)
- Z-axis = impedance proxy `Z = rate_ref / rate` after free-run calibration
- D-axis = stock driver (do not put thermal in the 100 kHz loop)
- Persistent kernels + zero-copy + high-frequency observer

## Current Reference: Gold 1.0.0
- Binary: `build/gputronic_gold` (`make gold`)
- Source: `src/gputronic_gold.cu` + `include/gputronic.h`
- Gate: `./build/gputronic_gold check` must PASS
- Target Z = 1.5, one-sided tracking PI, 2-state Kalman
- Do not reintroduce last-SM overwrite of `total_work_pulses`

## Key Principles
- Never sacrifice the ability to measure and control Z
- Prefer fundamental control over workload-specific hacks
- Keep the control loop clean and model-based
- Document the why behind every tuning decision
- Archive experiments; do not fork Gold in-place without a new version tag

## Next Major Milestone
Cooperative integration into real codebases (llama.cpp / custom CUDA), then an honestly named contending mode for uninstrumented apps. Lab Gold is done; product path is embed + workload registration.

## Communication Style
- Think in control theory terms (setpoints, observers, plant gain)
- Be precise about timescales (GPU cycles vs control loop period)
- Celebrate when Z tracking and performance improve together
