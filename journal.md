
---
## 2026-05-21 11:06:04 -- success

**Purpose**: Phase 1 of async refactor — add chunk-boundary telemetry (no behavior change). Verify in sync mode that `state_vs_a0(arm)` is small at chunk boundaries, which is the assumption Phase 3 (time-aligned async) will rely on.

**Notes**: Task completed successfully (right cube to box, return, left cube to box). User Ctrl-C'd at task completion rather than wait for the 3-minute task wall clock. No motor errors. Camera-before-arms reorder still holding. Phase 1 boundary log is producing the expected `[boundary] #N  state_vs_a0(arm)=X.XXX rad  tail_vs_a0(arm)=X.XXX rad ...` lines, 57 of them across the run.

**Phase 1 measured metrics** (57 boundary samples):
- `state_vs_a0(arm)`: mean ~0.07 rad, median ~0.05 rad, peak 0.372 rad (1 outlier).
  - 54% of boundaries under 0.05 rad (Phase 1 pass criterion was ≥90% — NOT strictly met).
  - 95% under 0.15 rad (within `--max-step-rad` cap).
- `tail_vs_a0(arm)`: mean ~0.07 rad, peak 0.283 rad. Model's plan-to-plan continuity is good.
- `state_vs_a0(grip)`: mostly 0.00–0.10, with two transitions where the right gripper was actively opening/closing (0.21, 0.41 — expected, model commanding a gripper change).
- Clip rate: 5–10% per chunk, occasional spikes to 11–14% during fast motion.
- RTT: 310–340 ms (mean ~320), same as last run.
- Replan rate: ~1.5 Hz (57 boundaries / ~37 s of inference).

**Interpretation**: the model is NOT producing a[0] ≈ current_state. It's producing meaningful first actions that step forward by 0.05–0.15 rad on average and occasionally 0.2+ rad. This is the model intentionally producing fast motion at chunk start, not a measurement issue (same magnitudes match the `a0_d` from per-query log). Implication for Phase 2 (naive async): the "first action jump" will be larger than my earlier theoretical estimate — could be 0.2–0.4 rad on bad chunks. The per-step clip at 0.15 will catch most of it; Phase 3 (time-aligned) should fix it cleanly.

**Phase 1 verdict**: telemetry works, criterion partially met. Proceeding to Phase 2 is still informative — the predicted naive-async failure mode is real and should be measurable.

**Command**:
```
./scripts/run_client.sh \
    --left-can can0 --right-can can1 \
    --left-gripper linear_4310 --right-gripper linear_4310 \
    --top-cam-v4l2 /dev/video12 \
    --left-cam-serial 427622271914 \
    --right-cam-serial 352122272708 \
    --cam-width 640 --cam-height 360 \
    --horizon-stride 10 \
    --move-to-ready \
    --rerun --rerun-save /tmp/yam_phase1.rrd \
    --server-url http://127.0.0.1:8202/act \
    --instruction "First, pick up the right orange cube with your right arm and put it in the grey box. Then, move your right arm back to the starting position. Next, pick up the left orange cube with your left arm and put it in the grey box"
```

**Configuration**:
- `cam_width`: 640, `cam_height`: 360, `cam_fps`: 30
- `horizon_stride`: 10
- `max_step_rad`: 0.15, `gripper_step`: 0.15
- `move_to_ready`: True
- `rerun`: True, `rerun_save`: /tmp/yam_phase1.rrd
- `server_url`: http://127.0.0.1:8202/act
- `log file`: /tmp/yam_phase1.log
- `rrd file`: /tmp/yam_phase1.rrd

**Duration**: ~55 s (process start to Ctrl-C), ~37 s of inference loop.

**Status**: success (manually logged because Ctrl-C path didn't reach the journal prompt — fixing next).
