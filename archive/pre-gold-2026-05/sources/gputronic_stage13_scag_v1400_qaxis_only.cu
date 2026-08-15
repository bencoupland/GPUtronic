// =============================================================================
// GPUTronic Stage 13 SCAG v14.0.1 — Q-AXIS-ONLY MODE (D-AXIS DISABLED)
// Target: Blackwell (RTX 5080) — Pure work counter feedback control
// Author: GPUTronic Architect for Ben Coupland
// Date: 2026-05-16
// =============================================================================

// D-AXIS REMOVED ENTIRELY:
// - No NVML telemetry (too slow anyway, ~20ms vs 10µs control loop)
// - No thermal weakening (driver handles all of this automatically)
// - No power limit enforcement (driver enforces PLimit)
// - No impedance coupling to D-axis state

// Q-AXIS IS THE REVOLUTIONARY CORE:
// - Work counter feedback at 100kHz+ via atomic counters
// - Sensorless Kalman observer using only work throughput measurements
// - Full throttle range (q=0.05→1.0) for testing stability
// - Predictive stall avoidance via dZ/dt estimation
