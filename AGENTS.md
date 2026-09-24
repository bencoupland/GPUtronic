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
- Actuator: `active_sms`. Idle SMs skip the work and do not pulse. Floor is 4. Nanosleep duration is not the command. Previous sources: `archive/gold-1.0.0-nanosleep/`
- Target Z = 1.5, one-sided tracking PI, 2-state Kalman. Kp=0.55, Ki=0.08. Do not retune them to shave the tenth.
- Observer: rate over about 1–8 ms. Path is 10 µs. Do not bring back a 10 µs rate EMA.
- Do not reintroduce last-SM overwrite of `total_work_pulses`

## Key Principles
- Never sacrifice the ability to measure and control Z
- Prefer fundamental control over workload-specific hacks
- Keep the control loop clean and model-based
- Document the why behind every tuning decision
- Archive experiments; do not fork Gold in-place without a new version tag

## Next Major Milestone
Same program, same gains, on the three test kernels we already have, including when the workload changes and the requested rate does not. Then one counter that keeps going after a kernel returns, in code we compile ourselves. A driver change is how that could apply to a program we do not modify. Integrating into llama.cpp is not the next measurement. A mode for unmodified programs stays unnamed until we can count their work without their help.

## Communication Style
- Think in control theory terms (setpoints, observers, plant gain)
- Be precise about timescales (GPU cycles vs control loop period)
- Celebrate when Z tracking and performance improve together
