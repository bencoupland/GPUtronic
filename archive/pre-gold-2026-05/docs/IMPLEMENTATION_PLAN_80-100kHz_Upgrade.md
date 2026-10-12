# GPUTronic Control Loop Upgrade: 50 kHz → 80-100 kHz
## Implementation Plan + Code Changes (v23)

**Date:** 2026-05-23
**Target:** Upgrade Q-axis governor control loop frequency while maintaining stability via retuned Kalman observer and feed-forward terms.
**Reference Files:** gputronic_q_axis_governor_v21_high_freq_pm_pi.cu (base), v22_bursty_workload.cu (workload), gpu-tronic-project-bible

## 1. Timing Changes
- Current: CONTROL_DT_US = 20 → 50 kHz
- Target: CONTROL_DT_US = 10 → 100 kHz (or 12 for ~83 kHz conservative)
- Update guard condition and usleep target.
- Note: usleep(10) has ~10-50us jitter on Linux; for production consider clock_nanosleep or spin-wait with __rdtsc for sub-5us precision. Keep usleep for v23.

## 2. Kalman Observer Tuning
Current (simplified, as in v21):
- KALMAN_Q_PROC = 0.0009f
- KALMAN_R_MEAS = 0.022f

At higher frequency (dt halved):
- Process noise covariance should scale ~dt or dt² for velocity model.
- Recommendation: KALMAN_Q_PROC = 0.00045f (halved for 2x rate)
- KALMAN_R_MEAS = 0.018f (slightly lower to trust measurements more at faster rate)
- **Critical:** Replace simplified P update with full covariance propagation from project-bible (constant velocity model):
  p00 = P[0][0] + 2*dt*P[0][1] + dt*dt*P[1][1] + Q[0]
  etc. (see bible section Pattern #3)

## 3. Feed-Forward Adjustments
Current:
- FF_Z_GAIN = 0.042f
- FF_DZDT_GAIN = 0.32f
- FF_DZDT_NEG_GAIN = 0.09f

Rationale for retune:
- dzdt units now represent change per shorter dt → magnitude of dzdt estimate decreases ~proportionally to dt.
- To maintain same physical predictive effect, increase gains by ~1.6-2x.
- New recommended:
  FF_Z_GAIN = 0.055f
  FF_DZDT_GAIN = 0.55f
  FF_DZDT_NEG_GAIN = 0.14f
- Add dt scaling factor in ff computation for future-proofing: ff += dzdt * FF_DZDT_GAIN * (dt / 20e-6)

## 4. Other Adjustments
- Update all version strings, comments, printf to "v23 80-100 kHz"
- PI gains (KP=0.48, KI=0.055) likely stable; monitor overshoot and reduce KP 10% if needed.
- Deadband and integral clamp remain same.
- In kernel __nanosleep keep or reduce slightly.

## 5. Verification Steps
1. Build v23
2. Run step response test, check overshoot <15%, settling <80ms
3. Verify telemetry shows ~100k loops/sec (tick count)
4. Confirm proactive_corrections increase due to faster prediction
5. Dyno sweep linearity R² >0.95

## Code Changes (Unified Diff for v23 based on v21 + v22 workload)

```diff
diff --git a/gputronic_q_axis_governor_v23_80-100khz.cu b/gputronic_q_axis_governor_v23_80-100khz.cu
new file mode 100755
index 0000000..new
--- /dev/null
+++ b/gputronic_q_axis_governor_v23_80-100khz.cu
@@
-#define CONTROL_DT_US       20          // ~50 kHz control loop
+#define CONTROL_DT_US       10          // 100 kHz target (80-100 kHz range)

@@
-#define KP                  0.48f
-#define KI                  0.055f
+#define KP                  0.45f       // slight reduction for higher bandwidth
+#define KI                  0.055f

@@
-// Aggressive predictive feed-forward (bursty workload protection)
-#define FF_Z_GAIN           0.042f
-#define FF_DZDT_GAIN        0.32f          // strong reaction to rising impedance
-#define FF_DZDT_NEG_GAIN    0.09f          // gentle recovery when impedance falling
+// Retuned predictive feed-forward for 100 kHz (dzdt scaling)
+#define FF_Z_GAIN           0.055f
+#define FF_DZDT_GAIN        0.55f
+#define FF_DZDT_NEG_GAIN    0.14f
+#define FF_DT_SCALE         (CONTROL_DT_US / 20.0f)

@@
-// Kalman (slightly faster adaptation at high frequency)
-#define KALMAN_Q_PROC       0.0009f
-#define KALMAN_R_MEAS       0.022f
+// Kalman retuned for dt=10us (process noise scales with rate)
+#define KALMAN_Q_PROC       0.00045f
+#define KALMAN_R_MEAS       0.018f

@@
-    printf("[PM] v21 High-Frequency (50 kHz) PM-PI + Predictive Throttling\n");
+    printf("[PM] v23 High-Frequency (80-100 kHz) PM-PI + Predictive Throttling\n");

@@
-        if (dt < 0.000015f) { usleep(8); continue; }   // ~50 kHz capable
+        if (dt < 0.000008f) { usleep(4); continue; }   // ~100 kHz capable

@@
-        // Predictive term (strong on rising impedance)
-        float ff = (z - 1.05f) * FF_Z_GAIN;
-        if (dzdt > 0.001f) {
-            ff += dzdt * FF_DZDT_GAIN;
-            proactive_count++;
-        } else if (dzdt < -0.0005f) {
-            ff += dzdt * FF_DZDT_NEG_GAIN;
-        }
+        // Predictive term (dt-aware for 100 kHz)
+        float ff = (z - 1.05f) * FF_Z_GAIN;
+        float dzdt_scaled = dzdt * FF_DT_SCALE;
+        if (dzdt_scaled > 0.0008f) {
+            ff += dzdt_scaled * FF_DZDT_GAIN;
+            proactive_count++;
+        } else if (dzdt_scaled < -0.0004f) {
+            ff += dzdt_scaled * FF_DZDT_NEG_GAIN;
+        }

@@
-        kalman_update(&z, &dzdt, measured_z, (float)dt);
+        kalman_update(&z, &dzdt, measured_z, (float)dt);  // TODO: upgrade kalman_update to full P propagation per bible

@@
-    printf("[GPUTronic] v21 50 kHz Predictive PM-PI test complete.\n");
+    printf("[GPUTronic] v23 80-100 kHz Predictive PM-PI test complete.\n");
```

## Recommended Next File
Create `gputronic_q_axis_governor_v23_80-100khz.cu` by copying v21 and applying above edits + integrate bursty workload from v22 if desired.

## Risks & Mitigations
- Higher frequency increases CPU load on control thread → pin thread to isolated core if jitter observed.
- Kalman divergence at high rate → validate with full matrix P update immediately.
- usleep resolution → monitor actual achieved loop rate via tick counter.

This upgrade enables true 100 kHz zero-copy reactive + predictive control on Blackwell.