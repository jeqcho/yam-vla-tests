
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

---
## 2026-05-21 14:20:38 -- success  (repl attempt #1)

**Instruction**: 'pick up the orange cube and p ut it in the box'

**Duration**: 0m 23s

**Attempt stats**:
- chunks: 38
- rtt_ms: mean 430, p95 423, max 4059
- horizon_arm_span (rad): mean 0.443, max 0.883
- state_vs_a0 at boundaries (rad): mean 0.055, max 0.141
- clip rate: 29/3108 dim-steps (0.9%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:23:41 -- failure  (repl attempt #1)

**Instruction**: 'pick up the orange cube from the box using your left arm and put it down outside the box'

**Duration**: 0m 48s

**Attempt stats**:
- chunks: 91
- rtt_ms: mean 330, p95 353, max 848
- horizon_arm_span (rad): mean 0.186, max 0.879
- state_vs_a0 at boundaries (rad): mean 0.014, max 0.024
- clip rate: 0/7560 dim-steps (0.0%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:24:43 -- success  (repl attempt #2)

**Instruction**: 'pick up the orange cube with your left arm and put it into the box'

**Duration**: 0m 30s

**Attempt stats**:
- chunks: 56
- rtt_ms: mean 339, p95 364, max 1075
- horizon_arm_span (rad): mean 0.315, max 0.664
- state_vs_a0 at boundaries (rad): mean 0.036, max 0.113
- clip rate: 152/4620 dim-steps (3.3%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:26:24 -- success  (repl attempt #3)

**Instruction**: 'pick up the orange cube with your left arm and put it into the box'

**Notes**: a bit slow at the start

**Duration**: 0m 58s

**Attempt stats**:
- chunks: 112
- rtt_ms: mean 324, p95 353, max 392
- horizon_arm_span (rad): mean 0.337, max 1.221
- state_vs_a0 at boundaries (rad): mean 0.034, max 0.135
- clip rate: 276/9338 dim-steps (3.0%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:28:36 -- failure  (repl attempt #4)

**Instruction**: 'pick up the orange cube with your right arm and put it outside the box'

**Duration**: 1m 38s

**Attempt stats**:
- chunks: 189
- rtt_ms: mean 320, p95 353, max 435
- horizon_arm_span (rad): mean 0.138, max 0.930
- state_vs_a0 at boundaries (rad): mean 0.022, max 0.123
- clip rate: 4/15792 dim-steps (0.0%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:30:24 -- success  (repl attempt #5)

**Instruction**: 'pick up the orange cube with your right arm and put it on top of the grey box.'

**Notes**: took some time to let go

**Duration**: 1m 10s

**Attempt stats**:
- chunks: 132
- rtt_ms: mean 330, p95 353, max 963
- horizon_arm_span (rad): mean 0.259, max 1.055
- state_vs_a0 at boundaries (rad): mean 0.041, max 0.186
- clip rate: 51/11088 dim-steps (0.5%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:32:18 -- failure  (repl attempt #6)

**Instruction**: 'pick up the left cube and put it in the middle'

**Notes**: picks up but no put on middle

**Duration**: 0m 50s

**Attempt stats**:
- chunks: 95
- rtt_ms: mean 330, p95 362, max 969
- horizon_arm_span (rad): mean 0.256, max 0.711
- state_vs_a0 at boundaries (rad): mean 0.044, max 0.160
- clip rate: 182/7952 dim-steps (2.3%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:33:50 -- failure  (repl attempt #7)

**Instruction**: 'pick up the cube and put it on the blue spot'

**Notes**: arms died halfway

**Duration**: 1m 02s

**Attempt stats**:
- chunks: 113
- rtt_ms: mean 350, p95 358, max 3471
- horizon_arm_span (rad): mean 0.208, max 0.617
- state_vs_a0 at boundaries (rad): mean 0.033, max 0.457
- clip rate: 19/9408 dim-steps (0.2%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:35:04 -- failure  (repl attempt #8)

**Instruction**: 'push the orange cube to the blue spot'

**Duration**: 0m 46s

**Attempt stats**:
- chunks: 88
- rtt_ms: mean 328, p95 360, max 825
- horizon_arm_span (rad): mean 0.211, max 0.516
- state_vs_a0 at boundaries (rad): mean 0.026, max 0.139
- clip rate: 8/7364 dim-steps (0.1%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:35:38 -- failure  (repl attempt #9)

**Instruction**: 'push the orange cube to the blue spot'

**Notes**: more reasonable with right arm, tho failed at place

**Duration**: 0m 23s

**Attempt stats**:
- chunks: 44
- rtt_ms: mean 325, p95 354, max 407
- horizon_arm_span (rad): mean 0.201, max 0.578
- state_vs_a0 at boundaries (rad): mean 0.048, max 0.144
- clip rate: 7/3612 dim-steps (0.2%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:37:25 -- failure  (repl attempt #10)

**Instruction**: 'pick up the orange cube and place it exactly on the blue spot'

**Duration**: 1m 04s

**Attempt stats**:
- chunks: 121
- rtt_ms: mean 329, p95 363, max 862
- horizon_arm_span (rad): mean 0.274, max 0.812
- state_vs_a0 at boundaries (rad): mean 0.037, max 0.159
- clip rate: 35/10080 dim-steps (0.3%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:38:31 -- success  (repl attempt #11)

**Instruction**: 'pick up the orange cube and put it in the box'

**Duration**: 0m 15s

**Attempt stats**:
- chunks: 26
- rtt_ms: mean 392, p95 376, max 2070
- horizon_arm_span (rad): mean 0.273, max 0.648
- state_vs_a0 at boundaries (rad): mean 0.048, max 0.174
- clip rate: 82/2100 dim-steps (3.9%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:39:42 -- success  (repl attempt #12)

**Instruction**: 'move the orange cube to the blue spot'

**Notes**: cube at right side ok

**Duration**: 0m 45s

**Attempt stats**:
- chunks: 86
- rtt_ms: mean 330, p95 352, max 853
- horizon_arm_span (rad): mean 0.306, max 0.771
- state_vs_a0 at boundaries (rad): mean 0.064, max 0.230
- clip rate: 90/7210 dim-steps (1.2%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:41:30 -- failure  (repl attempt #13)

**Instruction**: 'move the orange cube to the blue spot'

**Notes**: cube at left

**Duration**: 1m 24s

**Attempt stats**:
- chunks: 162
- rtt_ms: mean 320, p95 348, max 452
- horizon_arm_span (rad): mean 0.229, max 0.541
- state_vs_a0 at boundaries (rad): mean 0.054, max 0.272
- clip rate: 31/13552 dim-steps (0.2%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:43:43 -- success  (repl attempt #14)

**Instruction**: 'move the orange cube to the blue spot'

**Duration**: 1m 50s

**Attempt stats**:
- chunks: 209
- rtt_ms: mean 326, p95 369, max 423
- horizon_arm_span (rad): mean 0.307, max 1.123
- state_vs_a0 at boundaries (rad): mean 0.052, max 0.309
- clip rate: 97/17486 dim-steps (0.6%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:46:13 -- failure  (repl attempt #15)

**Instruction**: 'move the cube to the middle of the blue cross'

**Duration**: 0m 59s

**Attempt stats**:
- chunks: 112
- rtt_ms: mean 333, p95 349, max 1521
- horizon_arm_span (rad): mean 0.266, max 0.688
- state_vs_a0 at boundaries (rad): mean 0.058, max 0.380
- clip rate: 135/9324 dim-steps (1.4%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:48:57 -- failure  (repl attempt #18)

**Instruction**: 'move the cubes to the center of the blue cross'

**Duration**: 0m 44s

**Attempt stats**:
- chunks: 85
- rtt_ms: mean 326, p95 352, max 576
- horizon_arm_span (rad): mean 0.496, max 1.270
- state_vs_a0 at boundaries (rad): mean 0.051, max 0.258
- clip rate: 309/7056 dim-steps (4.4%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:52:01 -- failure  (repl attempt #19)

**Instruction**: 'stack the cubes vertically'

**Duration**: 2m 12s

**Attempt stats**:
- chunks: 247
- rtt_ms: mean 335, p95 368, max 2310
- horizon_arm_span (rad): mean 0.315, max 0.821
- state_vs_a0 at boundaries (rad): mean 0.039, max 0.203
- clip rate: 549/20664 dim-steps (2.7%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:53:27 -- failure  (repl attempt #20)

**Instruction**: 'put the right cube on top of the middle cube'

**Duration**: 1m 06s

**Attempt stats**:
- chunks: 123
- rtt_ms: mean 336, p95 360, max 1476
- horizon_arm_span (rad): mean 0.177, max 0.550
- state_vs_a0 at boundaries (rad): mean 0.034, max 0.133
- clip rate: 5/10248 dim-steps (0.0%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:54:32 -- failure  (repl attempt #21)

**Instruction**: 'pick up the cube and throw it'

**Duration**: 0m 41s

**Attempt stats**:
- chunks: 72
- rtt_ms: mean 366, p95 372, max 2167
- horizon_arm_span (rad): mean 0.275, max 0.786
- state_vs_a0 at boundaries (rad): mean 0.044, max 0.186
- clip rate: 6/5964 dim-steps (0.1%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:55:51 -- failure  (repl attempt #22)

**Instruction**: 'turn the box upside down'

**Duration**: 0m 41s

**Attempt stats**:
- chunks: 79
- rtt_ms: mean 330, p95 355, max 851
- horizon_arm_span (rad): mean 0.388, max 0.992
- state_vs_a0 at boundaries (rad): mean 0.026, max 0.165
- clip rate: 10/6552 dim-steps (0.2%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:56:49 -- failure  (repl attempt #23)

**Instruction**: 'pick up the grey box with your right hand'

**Duration**: 0m 41s

**Attempt stats**:
- chunks: 78
- rtt_ms: mean 332, p95 358, max 835
- horizon_arm_span (rad): mean 0.353, max 1.234
- state_vs_a0 at boundaries (rad): mean 0.033, max 0.224
- clip rate: 13/6552 dim-steps (0.2%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:57:59 -- failure  (repl attempt #24)

**Instruction**: 'pick up the tape roll with your right hand'

**Duration**: 0m 49s

**Attempt stats**:
- chunks: 95
- rtt_ms: mean 320, p95 340, max 418
- horizon_arm_span (rad): mean 0.246, max 1.035
- state_vs_a0 at boundaries (rad): mean 0.030, max 0.118
- clip rate: 1/7896 dim-steps (0.0%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 14:58:46 -- failure  (repl attempt #25)

**Instruction**: 'pick up the tape roll with your right gripper'

**Notes**: left gripper getting close

**Duration**: 0m 30s

**Attempt stats**:
- chunks: 56
- rtt_ms: mean 340, p95 361, max 1212
- horizon_arm_span (rad): mean 0.207, max 0.578
- state_vs_a0 at boundaries (rad): mean 0.047, max 0.230
- clip rate: 19/4648 dim-steps (0.4%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 15:22:49 -- failure  (repl attempt #1)

**Instruction**: 'pick up the smaller cube and put it in the grey box'

**Duration**: 1m 09s

**Attempt stats**:
- chunks: 179
- rtt_ms: mean 187, p95 295, max 734
- horizon_arm_span (rad): mean 0.271, max 0.758
- state_vs_a0 at boundaries (rad): mean 0.037, max 0.288
- clip rate: 45/14952 dim-steps (0.3%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 15:56:29 -- success  (repl attempt #1)

**Instruction**: 'put the cube in the box'

**Duration**: 0m 24s

**Attempt stats**:
- chunks: 63
- rtt_ms: mean 193, p95 296, max 649
- horizon_arm_span (rad): mean 0.479, max 1.684
- state_vs_a0 at boundaries (rad): mean 0.045, max 0.166
- clip rate: 48/5208 dim-steps (0.9%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 15:58:01 -- success  (repl attempt #2)

**Instruction**: 'put the red marker pen into the box'

**Duration**: 0m 25s

**Attempt stats**:
- chunks: 67
- rtt_ms: mean 178, p95 295, max 753
- horizon_arm_span (rad): mean 0.357, max 1.434
- state_vs_a0 at boundaries (rad): mean 0.053, max 0.187
- clip rate: 10/5544 dim-steps (0.2%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:00:19 -- failure  (repl attempt #3)

**Instruction**: 'put the cube into the tape roll'

**Duration**: 0m 20s

**Attempt stats**:
- chunks: 55
- rtt_ms: mean 170, p95 249, max 693
- horizon_arm_span (rad): mean 0.251, max 0.703
- state_vs_a0 at boundaries (rad): mean 0.043, max 0.124
- clip rate: 0/4536 dim-steps (0.0%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:00:39 -- failure  (repl attempt #4)

**Instruction**: 'put the cube into the tape roll'

**Duration**: 0m 10s

**Attempt stats**:
- chunks: 29
- rtt_ms: mean 166, p95 260, max 288
- horizon_arm_span (rad): mean 0.336, max 0.900
- state_vs_a0 at boundaries (rad): mean 0.045, max 0.104
- clip rate: 0/2394 dim-steps (0.0%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:01:14 -- failure  (repl attempt #5)

**Instruction**: 'put the cube into the tape roll'

**Duration**: 0m 26s

**Attempt stats**:
- chunks: 68
- rtt_ms: mean 180, p95 293, max 299
- horizon_arm_span (rad): mean 0.277, max 0.719
- state_vs_a0 at boundaries (rad): mean 0.061, max 0.223
- clip rate: 80/5712 dim-steps (1.4%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:02:19 -- failure  (repl attempt #6)

**Instruction**: 'put the cube into the tape roll'

**Duration**: 0m 48s

**Attempt stats**:
- chunks: 129
- rtt_ms: mean 176, p95 293, max 302
- horizon_arm_span (rad): mean 0.229, max 0.675
- state_vs_a0 at boundaries (rad): mean 0.039, max 0.159
- clip rate: 28/10752 dim-steps (0.3%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:05:21 -- failure  (repl attempt #7)

**Instruction**: 'put the cube into the tape roll'

**Duration**: 2m 33s

**Attempt stats**:
- chunks: 399
- rtt_ms: mean 183, p95 294, max 366
- horizon_arm_span (rad): mean 0.086, max 0.839
- state_vs_a0 at boundaries (rad): mean 0.026, max 0.156
- clip rate: 7/33474 dim-steps (0.0%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:06:51 -- failure  (repl attempt #8)

**Instruction**: 'remove the cube from the box'

**Duration**: 0m 50s

**Attempt stats**:
- chunks: 112
- rtt_ms: mean 245, p95 389, max 1718
- horizon_arm_span (rad): mean 0.494, max 1.020
- state_vs_a0 at boundaries (rad): mean 0.031, max 0.175
- clip rate: 4/9324 dim-steps (0.0%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:09:00 -- success  (repl attempt #9)

**Instruction**: 'place the cube on top of the book'

**Duration**: 1m 41s

**Attempt stats**:
- chunks: 255
- rtt_ms: mean 195, p95 296, max 1202
- horizon_arm_span (rad): mean 0.256, max 0.707
- state_vs_a0 at boundaries (rad): mean 0.027, max 0.109
- clip rate: 286/21420 dim-steps (1.3%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:12:22 -- success  (repl attempt #10)

**Instruction**: 'put the fork on the plate'

**Duration**: 1m 25s

**Attempt stats**:
- chunks: 218
- rtt_ms: mean 191, p95 296, max 752
- horizon_arm_span (rad): mean 0.229, max 1.248
- state_vs_a0 at boundaries (rad): mean 0.029, max 0.154
- clip rate: 8/18228 dim-steps (0.0%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:13:20 -- success  (repl attempt #11)

**Instruction**: 'put the fork on the plate'

**Duration**: 0m 43s

**Attempt stats**:
- chunks: 111
- rtt_ms: mean 191, p95 295, max 904
- horizon_arm_span (rad): mean 0.293, max 0.603
- state_vs_a0 at boundaries (rad): mean 0.032, max 0.233
- clip rate: 6/9240 dim-steps (0.1%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:21:20 -- failure  (repl attempt #12)

**Instruction**: 'put the fork on the plate'

**Notes**: left

**Duration**: 5m 56s

**Attempt stats**:
- chunks: 909
- rtt_ms: mean 190, p95 297, max 360
- horizon_arm_span (rad): mean 0.181, max 0.762
- state_vs_a0 at boundaries (rad): mean 0.024, max 0.153
- clip rate: 44/76272 dim-steps (0.1%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:22:16 -- success  (repl attempt #13)

**Instruction**: 'put the fork on the plate'

**Duration**: 0m 34s

**Attempt stats**:
- chunks: 91
- rtt_ms: mean 183, p95 285, max 301
- horizon_arm_span (rad): mean 0.241, max 0.573
- state_vs_a0 at boundaries (rad): mean 0.047, max 0.204
- clip rate: 16/7574 dim-steps (0.2%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:23:40 -- failure  (repl attempt #14)

**Instruction**: 'put the spoon on the plate'

**Duration**: 0m 40s

**Attempt stats**:
- chunks: 104
- rtt_ms: mean 185, p95 293, max 356
- horizon_arm_span (rad): mean 0.187, max 0.656
- state_vs_a0 at boundaries (rad): mean 0.023, max 0.100
- clip rate: 0/8666 dim-steps (0.0%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:24:52 -- success  (repl attempt #15)

**Instruction**: 'put the spoon on the plate'

**Duration**: 0m 59s

**Attempt stats**:
- chunks: 152
- rtt_ms: mean 186, p95 296, max 361
- horizon_arm_span (rad): mean 0.459, max 1.068
- state_vs_a0 at boundaries (rad): mean 0.036, max 0.153
- clip rate: 7/12726 dim-steps (0.1%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:25:55 -- success  (repl attempt #16)

**Instruction**: 'put the spoon on the plate'

**Duration**: 0m 37s

**Attempt stats**:
- chunks: 95
- rtt_ms: mean 196, p95 301, max 326
- horizon_arm_span (rad): mean 0.317, max 0.804
- state_vs_a0 at boundaries (rad): mean 0.041, max 0.185
- clip rate: 5/7910 dim-steps (0.1%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:27:40 -- failure  (repl attempt #17)

**Instruction**: 'remove the spoon from the plate'

**Duration**: 1m 22s

**Attempt stats**:
- chunks: 206
- rtt_ms: mean 198, p95 304, max 366
- horizon_arm_span (rad): mean 0.198, max 0.709
- state_vs_a0 at boundaries (rad): mean 0.025, max 0.144
- clip rate: 2/17220 dim-steps (0.0%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:28:32 -- success  (repl attempt #18)

**Instruction**: 'put the knife on the plate'

**Duration**: 0m 37s

**Attempt stats**:
- chunks: 99
- rtt_ms: mean 179, p95 293, max 303
- horizon_arm_span (rad): mean 0.300, max 0.795
- state_vs_a0 at boundaries (rad): mean 0.043, max 0.272
- clip rate: 25/8232 dim-steps (0.3%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:31:24 -- failure  (repl attempt #19)

**Instruction**: 'put the spoon and fork on the plate'

**Notes**: spoon ok, then keep touching spoon

**Duration**: 1m 44s

**Attempt stats**:
- chunks: 267
- rtt_ms: mean 187, p95 293, max 4304
- horizon_arm_span (rad): mean 0.269, max 0.695
- state_vs_a0 at boundaries (rad): mean 0.037, max 0.268
- clip rate: 79/22428 dim-steps (0.4%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:32:10 -- failure  (repl attempt #20)

**Instruction**: 'put the fork on the plate'

**Duration**: 0m 22s

**Attempt stats**:
- chunks: 61
- rtt_ms: mean 174, p95 290, max 699
- horizon_arm_span (rad): mean 0.231, max 0.570
- state_vs_a0 at boundaries (rad): mean 0.028, max 0.131
- clip rate: 4/5110 dim-steps (0.1%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:33:29 -- failure  (repl attempt #21)

**Instruction**: 'There is a spoon on the plate and a fork beside the plate. Put the fork on the plate.'

**Duration**: 0m 25s

**Attempt stats**:
- chunks: 65
- rtt_ms: mean 193, p95 293, max 872
- horizon_arm_span (rad): mean 0.247, max 0.755
- state_vs_a0 at boundaries (rad): mean 0.039, max 0.227
- clip rate: 17/5460 dim-steps (0.3%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:36:30 -- success  (repl attempt #22)

**Instruction**: 'There is a spoon on the plate and a fork beside the plate. Put the fork on the plate.'

**Duration**: 2m 48s

**Attempt stats**:
- chunks: 436
- rtt_ms: mean 183, p95 296, max 327
- horizon_arm_span (rad): mean 0.325, max 1.246
- state_vs_a0 at boundaries (rad): mean 0.041, max 0.246
- clip rate: 119/36540 dim-steps (0.3%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:37:29 -- success  (repl attempt #23)

**Instruction**: 'put the fork on the plate'

**Duration**: 0m 36s

**Attempt stats**:
- chunks: 95
- rtt_ms: mean 185, p95 294, max 635
- horizon_arm_span (rad): mean 0.325, max 0.861
- state_vs_a0 at boundaries (rad): mean 0.040, max 0.222
- clip rate: 8/7924 dim-steps (0.1%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:39:28 -- success  (repl attempt #24)

**Instruction**: 'put the fork on the plate'

**Duration**: 1m 43s

**Attempt stats**:
- chunks: 270
- rtt_ms: mean 179, p95 294, max 364
- horizon_arm_span (rad): mean 0.268, max 0.984
- state_vs_a0 at boundaries (rad): mean 0.030, max 0.150
- clip rate: 8/22596 dim-steps (0.0%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:39:57 -- success  (repl attempt #25)

**Instruction**: 'remove the fork from the plate'

**Duration**: 0m 17s

**Attempt stats**:
- chunks: 45
- rtt_ms: mean 184, p95 295, max 313
- horizon_arm_span (rad): mean 0.419, max 1.072
- state_vs_a0 at boundaries (rad): mean 0.058, max 0.240
- clip rate: 20/3696 dim-steps (0.5%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:41:36 -- failure  (repl attempt #27)

**Instruction**: 'put the fork on the plate'

**Duration**: 0m 57s

**Attempt stats**:
- chunks: 154
- rtt_ms: mean 173, p95 292, max 358
- horizon_arm_span (rad): mean 0.356, max 1.562
- state_vs_a0 at boundaries (rad): mean 0.037, max 0.220
- clip rate: 26/12852 dim-steps (0.2%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:42:07 -- failure  (repl attempt #28)

**Instruction**: 'put the fork on the plate'

**Notes**: goes for the spoon

**Duration**: 0m 17s

**Attempt stats**:
- chunks: 48
- rtt_ms: mean 173, p95 292, max 297
- horizon_arm_span (rad): mean 0.377, max 0.876
- state_vs_a0 at boundaries (rad): mean 0.049, max 0.148
- clip rate: 10/3948 dim-steps (0.3%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:44:06 -- failure  (repl attempt #29)

**Instruction**: 'put the fork on the plate'

**Notes**: picked up fork but otherwise fail

**Duration**: 1m 45s

**Attempt stats**:
- chunks: 276
- rtt_ms: mean 181, p95 295, max 318
- horizon_arm_span (rad): mean 0.377, max 1.434
- state_vs_a0 at boundaries (rad): mean 0.051, max 0.374
- clip rate: 65/23100 dim-steps (0.3%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 16:53:22 -- unclear

**Notes**: tested the new rtc inference, it was smooth but i didn't set up the task properly

**Duration**: 0m 24s

**Command**:
```
./experimental/rtc/run_client_rtc.sh --top-cam-serial 349622072241 --left-cam-serial 427622271914 --right-cam-serial 352122272708 --move-to-ready --execution-horizon 8 --max-step-rad 0.05
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `execution_horizon`: 8
- `gripper_step`: 0.15
- `instruction`: first pick up the left orange cube and put it in the box, then pick up the right orange cube and put it in the box
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.05
- `move_to_ready`: True
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `rtc_min_inference_delay`: 1
- `rtc_rtt_buffer_size`: 8
- `server_url`: http://127.0.0.1:8203/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 17:01:22 -- failure

**Purpose**: to see if rtc worked on these arms

**Notes**: inference is smooth but wrong

**Duration**: 2m 14s

**Command**:
```
./experimental/rtc/run_client_rtc.sh --top-cam-serial 349622072241 --left-cam-serial 427622271914 --right-cam-serial 352122272708 --move-to-ready --execution-horizon 8 --max-step-rad 0.05 --instruction First\,\ pick\ up\ the\ right\ orange\ cube\ with\ your\ right\ arm\ and\ put\ it\ in\ the\ grey\ box.\ Then\,\ move\ your\ right\ arm\ back\ to\ the\ starting\ position.\ Next\,\ pick\ up\ the\ left\ orange\ cube\ with\ your\ left\ arm\ and\ put\ it\ in\ the\ grey\ box
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `execution_horizon`: 8
- `gripper_step`: 0.15
- `instruction`: 'First, pick up the right orange cube with your right arm and put it in the grey box. Then, move your right arm back to the starting position. Next, pick up the left orange cube with your left arm and put it in the grey box'
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.05
- `move_to_ready`: True
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `rtc_min_inference_delay`: 1
- `rtc_rtt_buffer_size`: 8
- `server_url`: http://127.0.0.1:8203/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 17:12:51 -- failure

**Purpose**: to see if horizon 16 is better than horizon 8 for rtc

**Notes**: arms are jerky

**Duration**: 0m 37s

**Command**:
```
./experimental/rtc/run_client_rtc.sh --top-cam-serial 349622072241 --left-cam-serial 427622271914 --right-cam-serial 352122272708 --move-to-ready --execution-horizon 16 --max-step-rad 0.05 --instruction First\,\ pick\ up\ the\ right\ orange\ cube\ with\ your\ right\ arm\ and\ put\ it\ in\ the\ grey\ box.\ Then\,\ move\ your\ right\ arm\ back\ to\ the\ starting\ position.\ Next\,\ pick\ up\ the\ left\ orange\ cube\ with\ your\ left\ arm\ and\ put\ it\ in\ the\ grey\ box
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `execution_horizon`: 16
- `gripper_step`: 0.15
- `instruction`: 'First, pick up the right orange cube with your right arm and put it in the grey box. Then, move your right arm back to the starting position. Next, pick up the left orange cube with your left arm and put it in the grey box'
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.05
- `move_to_ready`: True
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `rtc_min_inference_delay`: 1
- `rtc_rtt_buffer_size`: 8
- `server_url`: http://127.0.0.1:8203/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-21 17:17:59 -- failure

**Purpose**: wanted to see if num-steps 5 could fix things

**Notes**: arm movement was just wrong

**Duration**: 0m 50s

**Command**:
```
./experimental/rtc/run_client_rtc.sh --top-cam-serial 349622072241 --left-cam-serial 427622271914 --right-cam-serial 352122272708 --move-to-ready --execution-horizon 8 --num-steps 5 --max-step-rad 0.15 --instruction $'First, pick up the right orange cube with your right arm and put it in the grey box. \n  Then, move your right arm back to the starting position. Next, pick up the left orange cube with your \n  left arm and put it in the grey box'
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `execution_horizon`: 8
- `gripper_step`: 0.15
- `instruction`: 'First, pick up the right orange cube with your right arm and put it in the grey box. \n  Then, move your right arm back to the starting position. Next, pick up the left orange cube with your \n  left arm and put it in the grey box'
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can0
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `move_to_ready`: True
- `num_steps`: 5
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can1
- `right_gripper`: linear_4310
- `rtc_min_inference_delay`: 1
- `rtc_rtt_buffer_size`: 8
- `server_url`: http://127.0.0.1:8203/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-22 16:00:50 -- failure  (repl attempt #1)

**Instruction**: 'put the orange cube input the orange cube into the box'

**Duration**: 0m 31s

**Attempt stats**:
- chunks: 85
- rtt_ms: mean 170, p95 291, max 732
- horizon_arm_span (rad): mean 0.369, max 0.766
- state_vs_a0 at boundaries (rad): mean 0.035, max 0.124
- clip rate: 11/7056 dim-steps (0.2%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-22 16:01:32 -- success  (repl attempt #2)

**Instruction**: 'put the orange cube into the box'

**Duration**: 0m 20s

**Attempt stats**:
- chunks: 52
- rtt_ms: mean 189, p95 293, max 680
- horizon_arm_span (rad): mean 0.379, max 0.783
- state_vs_a0 at boundaries (rad): mean 0.067, max 0.225
- clip rate: 78/4284 dim-steps (1.8%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-22 16:13:31 -- failure  (repl attempt #1)

**Instruction**: 'Stack two orange blocks vertically'

**Notes**: [eval task 1/10] 

**Duration**: 2m 16s

**Attempt stats**:
- chunks: 337
- rtt_ms: mean 204, p95 301, max 598
- horizon_arm_span (rad): mean 0.307, max 1.117
- state_vs_a0 at boundaries (rad): mean 0.045, max 0.235
- clip rate: 611/28252 dim-steps (2.2%)

**Command**:
```
./eval-10-tasks/run_eval.sh ''
```

**Configuration**:
- `attempts`: 3
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `results_dir`: /home/andon/yam-tests/molmoact2-setup/eval-10-tasks/results
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-22 16:38:49 -- success  (repl attempt #9)

**Instruction**: 'Put the knife in the box'

**Notes**: [eval task 3/10] 

**Duration**: 0m 45s

**Attempt stats**:
- chunks: 121
- rtt_ms: mean 171, p95 291, max 303
- horizon_arm_span (rad): mean 0.283, max 0.730
- state_vs_a0 at boundaries (rad): mean 0.036, max 0.163
- clip rate: 6/10150 dim-steps (0.1%)

**Command**:
```
./eval-10-tasks/run_eval.sh ''
```

**Configuration**:
- `attempt_timeout_s`: 60.0
- `attempts`: 3
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `results_dir`: /home/andon/yam-tests/molmoact2-setup/eval-10-tasks/results
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-22 17:20:29 -- failure  (repl attempt #1)

**Instruction**: 'lift up the picture frame and make it stand upright'

**Duration**: 2m 09s

**Attempt stats**:
- chunks: 344
- rtt_ms: mean 175, p95 293, max 735
- horizon_arm_span (rad): mean 0.407, max 1.160
- state_vs_a0 at boundaries (rad): mean 0.034, max 0.154
- clip rate: 52/28840 dim-steps (0.2%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-22 17:22:26 -- success  (repl attempt #2)

**Instruction**: 'fold the flap behind the picture frame and place the picture frame flat on the table.'

**Duration**: 0m 58s

**Attempt stats**:
- chunks: 155
- rtt_ms: mean 176, p95 294, max 729
- horizon_arm_span (rad): mean 0.341, max 1.071
- state_vs_a0 at boundaries (rad): mean 0.040, max 0.192
- clip rate: 18/12950 dim-steps (0.1%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-22 17:24:00 -- success  (repl attempt #3)

**Instruction**: 'fold the flap behind the picture frame and place the picture frame flat on the table.'

**Duration**: 0m 07s

**Attempt stats**:
- chunks: 19
- rtt_ms: mean 198, p95 293, max 305
- horizon_arm_span (rad): mean 0.324, max 0.759
- state_vs_a0 at boundaries (rad): mean 0.056, max 0.144
- clip rate: 4/1568 dim-steps (0.3%)

**Command**:
```
./scripts/run_repl.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 11:41:15 -- failure  (repl attempt #1)

**Instruction**: 'put the orange cube into the box'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 35s

**Attempt stats**:
- chunks: 94
- rtt_ms: mean 173, p95 293, max 611
- horizon_arm_span (rad): mean 0.370, max 0.855
- state_vs_a0 at boundaries (rad): mean 0.041, max 0.187
- clip rate: 22/7812 dim-steps (0.3%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 11:41:33 -- failure  (repl attempt #2)

**Instruction**: 'put the orange cube into the box'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 08s

**Attempt stats**:
- chunks: 20
- rtt_ms: mean 226, p95 297, max 332
- horizon_arm_span (rad): mean 0.359, max 0.689
- state_vs_a0 at boundaries (rad): mean 0.079, max 0.182
- clip rate: 16/1596 dim-steps (1.0%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 11:42:03 -- success  (repl attempt #3)

**Instruction**: 'put the orange cube into the box'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 17s

**Attempt stats**:
- chunks: 46
- rtt_ms: mean 172, p95 286, max 296
- horizon_arm_span (rad): mean 0.346, max 0.680
- state_vs_a0 at boundaries (rad): mean 0.063, max 0.157
- clip rate: 40/3794 dim-steps (1.1%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 11:42:53 -- failure  (repl attempt #4)

**Instruction**: 'put the orange cube into the box'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 22s

**Attempt stats**:
- chunks: 59
- rtt_ms: mean 181, p95 296, max 341
- horizon_arm_span (rad): mean 0.343, max 0.652
- state_vs_a0 at boundaries (rad): mean 0.050, max 0.148
- clip rate: 4/4872 dim-steps (0.1%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 11:43:53 -- failure  (repl attempt #5)

**Instruction**: 'pick up the orange cube and put it in the box'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 18s

**Attempt stats**:
- chunks: 49
- rtt_ms: mean 187, p95 293, max 598
- horizon_arm_span (rad): mean 0.370, max 0.831
- state_vs_a0 at boundaries (rad): mean 0.061, max 0.146
- clip rate: 26/4032 dim-steps (0.6%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 11:46:24 -- failure  (repl attempt #6)

**Instruction**: 'place the orange cube in the gray box'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 14s

**Attempt stats**:
- chunks: 37
- rtt_ms: mean 184, p95 295, max 590
- horizon_arm_span (rad): mean 0.349, max 0.674
- state_vs_a0 at boundaries (rad): mean 0.052, max 0.181
- clip rate: 9/3080 dim-steps (0.3%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh ''
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 11:51:23 -- failure  (repl attempt #1)

**Instruction**: 'place the orange cube in the gray box'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 30s

**Attempt stats**:
- chunks: 83
- rtt_ms: mean 169, p95 238, max 715
- horizon_arm_span (rad): mean 0.343, max 0.814
- state_vs_a0 at boundaries (rad): mean 0.054, max 0.185
- clip rate: 19/6888 dim-steps (0.3%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun-connect 127.0.0.1:9876
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun_connect`: 127.0.0.1:9876
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 12:06:18 -- failure  (repl attempt #1)

**Instruction**: 'place the orange cube in the gray box'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 33s

**Attempt stats**:
- chunks: 88
- rtt_ms: mean 174, p95 294, max 579
- horizon_arm_span (rad): mean 0.306, max 0.828
- state_vs_a0 at boundaries (rad): mean 0.039, max 0.163
- clip rate: 6/7364 dim-steps (0.1%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun-connect 127.0.0.1:62728
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun_connect`: 127.0.0.1:62728
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 12:10:01 -- failure  (repl attempt #1)

**Instruction**: 'place the orange cube in the gray box'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 42s

**Attempt stats**:
- chunks: 88
- rtt_ms: mean 277, p95 318, max 646
- horizon_arm_span (rad): mean 0.370, max 0.805
- state_vs_a0 at boundaries (rad): mean 0.050, max 0.195
- clip rate: 15/7336 dim-steps (0.2%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 12:10:58 -- failure  (repl attempt #2)

**Instruction**: 'pick up the orange cube'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 24s

**Attempt stats**:
- chunks: 51
- rtt_ms: mean 281, p95 318, max 788
- horizon_arm_span (rad): mean 0.350, max 0.719
- state_vs_a0 at boundaries (rad): mean 0.070, max 0.186
- clip rate: 13/4214 dim-steps (0.3%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 12:12:03 -- failure  (repl attempt #3)

**Instruction**: 'pick up the orange cube'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 40s

**Attempt stats**:
- chunks: 84
- rtt_ms: mean 277, p95 320, max 346
- horizon_arm_span (rad): mean 0.362, max 1.012
- state_vs_a0 at boundaries (rad): mean 0.059, max 0.178
- clip rate: 35/6972 dim-steps (0.5%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 12:19:42 -- success  (repl attempt #1)

**Instruction**: 'place the orange cube in the gray box'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 36s

**Attempt stats**:
- chunks: 74
- rtt_ms: mean 293, p95 320, max 749
- horizon_arm_span (rad): mean 0.215, max 0.797
- state_vs_a0 at boundaries (rad): mean 0.048, max 0.189
- clip rate: 168/6132 dim-steps (2.7%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 12:20:33 -- success  (repl attempt #3)

**Instruction**: 'place the orange cube in the gray basket'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 26s

**Attempt stats**:
- chunks: 56
- rtt_ms: mean 265, p95 317, max 325
- horizon_arm_span (rad): mean 0.402, max 1.027
- state_vs_a0 at boundaries (rad): mean 0.053, max 0.203
- clip rate: 64/4634 dim-steps (1.4%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 12:21:47 -- success  (repl attempt #5)

**Instruction**: 'place the orange cube in the gray basket'

**Notes**: [policy=molmoact2] official try 1

**Duration**: 0m 23s

**Attempt stats**:
- chunks: 47
- rtt_ms: mean 288, p95 316, max 757
- horizon_arm_span (rad): mean 0.340, max 0.803
- state_vs_a0 at boundaries (rad): mean 0.060, max 0.208
- clip rate: 132/3934 dim-steps (3.4%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 12:22:36 -- success  (repl attempt #6)

**Instruction**: 'place the orange cube in the gray basket'

**Notes**: [policy=molmoact2] official try 2

**Duration**: 0m 30s

**Attempt stats**:
- chunks: 64
- rtt_ms: mean 281, p95 321, max 343
- horizon_arm_span (rad): mean 0.308, max 0.846
- state_vs_a0 at boundaries (rad): mean 0.052, max 0.302
- clip rate: 167/5292 dim-steps (3.2%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 12:23:08 -- success  (repl attempt #7)

**Instruction**: 'place the orange cube in the gray basket'

**Notes**: [policy=molmoact2] official try 3

**Duration**: 0m 19s

**Attempt stats**:
- chunks: 40
- rtt_ms: mean 292, p95 321, max 398
- horizon_arm_span (rad): mean 0.303, max 1.363
- state_vs_a0 at boundaries (rad): mean 0.065, max 0.376
- clip rate: 130/3276 dim-steps (4.0%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_host`: 127.0.0.1
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 12:56:26 -- failure  (repl attempt #1)

**Instruction**: 'place the orange cube in the gray basket'

**Notes**: [policy=gr00t-n17] 

**Duration**: 0m 24s

**Attempt stats**:
- chunks: 79
- rtt_ms: mean 177, p95 213, max 226
- horizon_arm_span (rad): mean 0.114, max 0.363
- state_vs_a0 at boundaries (rad): mean 0.043, max 0.139
- clip rate: 0/4382 dim-steps (0.0%)

**Command**:
```
./eval-yam/scripts/run_repl_gr00t-n17.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 4
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: gr00t-n17
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: unknown
- `server_host`: 127.0.0.1
- `server_meta`: {'status': 'ok', 'message': 'Server is running', 'backend': 'gr00t-n17', 'transport': 'zmq tcp://127.0.0.1:5556'}
- `server_norm_tag`: unknown
- `server_port`: 5556
- `server_repo_id`: unknown
- `server_url`: tcp://127.0.0.1:5556
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 12:57:00 -- failure  (repl attempt #2)

**Instruction**: 'place the orange cube in the gray basket'

**Notes**: [policy=gr00t-n17] 

**Duration**: 0m 16s

**Attempt stats**:
- chunks: 51
- rtt_ms: mean 182, p95 209, max 215
- horizon_arm_span (rad): mean 0.131, max 0.279
- state_vs_a0 at boundaries (rad): mean 0.042, max 0.101
- clip rate: 0/2800 dim-steps (0.0%)

**Command**:
```
./eval-yam/scripts/run_repl_gr00t-n17.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 4
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: gr00t-n17
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: unknown
- `server_host`: 127.0.0.1
- `server_meta`: {'status': 'ok', 'message': 'Server is running', 'backend': 'gr00t-n17', 'transport': 'zmq tcp://127.0.0.1:5556'}
- `server_norm_tag`: unknown
- `server_port`: 5556
- `server_repo_id`: unknown
- `server_url`: tcp://127.0.0.1:5556
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 12:57:44 -- failure  (repl attempt #4)

**Instruction**: 'place the orange cube in the gray basket'

**Notes**: [policy=gr00t-n17] 

**Duration**: 0m 15s

**Attempt stats**:
- chunks: 51
- rtt_ms: mean 161, p95 206, max 214
- horizon_arm_span (rad): mean 0.114, max 0.227
- state_vs_a0 at boundaries (rad): mean 0.053, max 0.122
- clip rate: 1/2856 dim-steps (0.0%)

**Command**:
```
./eval-yam/scripts/run_repl_gr00t-n17.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 4
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: gr00t-n17
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: unknown
- `server_host`: 127.0.0.1
- `server_meta`: {'status': 'ok', 'message': 'Server is running', 'backend': 'gr00t-n17', 'transport': 'zmq tcp://127.0.0.1:5556'}
- `server_norm_tag`: unknown
- `server_port`: 5556
- `server_repo_id`: unknown
- `server_url`: tcp://127.0.0.1:5556
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 13:10:30 -- failure  (repl attempt #1)

**Instruction**: 'place the orange cube in the gray basket'

**Notes**: [policy=pi05] 

**Duration**: 1m 05s

**Attempt stats**:
- chunks: 184
- rtt_ms: mean 88, p95 93, max 99
- horizon_arm_span (rad): mean 0.111, max 0.439
- state_vs_a0 at boundaries (rad): mean 0.051, max 0.137
- clip rate: 11/20608 dim-steps (0.1%)

**Command**:
```
./eval-yam/scripts/run_repl_pi05.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 8
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: pi05
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: unknown
- `server_host`: 127.0.0.1
- `server_meta`: {'backend': 'pi05', 'transport': 'ws://127.0.0.1:8000'}
- `server_norm_tag`: unknown
- `server_port`: 8000
- `server_repo_id`: unknown
- `server_url`: ws://127.0.0.1:8000
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 13:11:45 -- failure  (repl attempt #2)

**Instruction**: 'pick up the orange cube'

**Notes**: [policy=pi05] 

**Duration**: 1m 00s

**Attempt stats**:
- chunks: 166
- rtt_ms: mean 92, p95 96, max 98
- horizon_arm_span (rad): mean 0.209, max 0.895
- state_vs_a0 at boundaries (rad): mean 0.064, max 0.206
- clip rate: 12/18522 dim-steps (0.1%)

**Command**:
```
./eval-yam/scripts/run_repl_pi05.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 8
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: pi05
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: unknown
- `server_host`: 127.0.0.1
- `server_meta`: {'backend': 'pi05', 'transport': 'ws://127.0.0.1:8000'}
- `server_norm_tag`: unknown
- `server_port`: 8000
- `server_repo_id`: unknown
- `server_url`: ws://127.0.0.1:8000
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 13:22:44 -- failure  (repl attempt #1)

**Instruction**: 'place the orange cube in the gray basket'

**Notes**: [policy=pi05] 

**Duration**: 0m 36s

**Attempt stats**:
- chunks: 103
- rtt_ms: mean 84, p95 87, max 87
- horizon_arm_span (rad): mean 0.499, max 0.811
- state_vs_a0 at boundaries (rad): mean 0.074, max 0.166
- clip rate: 3/11522 dim-steps (0.0%)

**Command**:
```
./eval-yam/scripts/run_repl_pi05.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 8
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: pi05
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: unknown
- `server_host`: 127.0.0.1
- `server_meta`: {'backend': 'pi05', 'transport': 'ws://127.0.0.1:8000'}
- `server_norm_tag`: unknown
- `server_port`: 8000
- `server_repo_id`: unknown
- `server_url`: ws://127.0.0.1:8000
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 13:26:45 -- failure  (repl attempt #1)

**Instruction**: 'place the orange cube in the gray basket'

**Notes**: [policy=pi05] back to resting immediately

**Duration**: 0m 00s

**Attempt stats**:
- chunks: 3
- rtt_ms: mean 84, p95 86, max 87
- horizon_arm_span (rad): mean 0.180, max 0.227
- state_vs_a0 at boundaries (rad): mean 1.824, max 2.144
- clip rate: 92/266 dim-steps (34.6%)

**Command**:
```
./eval-yam/scripts/run_repl_pi05.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 8
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: pi05
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: unknown
- `server_host`: 127.0.0.1
- `server_meta`: {'backend': 'pi05', 'transport': 'ws://127.0.0.1:8000'}
- `server_norm_tag`: unknown
- `server_port`: 8000
- `server_repo_id`: unknown
- `server_url`: ws://127.0.0.1:8000
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 13:27:24 -- failure  (repl attempt #2)

**Instruction**: 'place the orange cube in the gray basket'

**Notes**: [policy=pi05] 

**Duration**: 0m 15s

**Attempt stats**:
- chunks: 43
- rtt_ms: mean 85, p95 91, max 97
- horizon_arm_span (rad): mean 0.193, max 0.700
- state_vs_a0 at boundaries (rad): mean 0.549, max 2.132
- clip rate: 1413/4802 dim-steps (29.4%)

**Command**:
```
./eval-yam/scripts/run_repl_pi05.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 8
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: pi05
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: unknown
- `server_host`: 127.0.0.1
- `server_meta`: {'backend': 'pi05', 'transport': 'ws://127.0.0.1:8000'}
- `server_norm_tag`: unknown
- `server_port`: 8000
- `server_repo_id`: unknown
- `server_url`: ws://127.0.0.1:8000
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 13:38:05 -- failure  (repl attempt #1)

**Instruction**: 'pick up the orange cube'

**Notes**: [policy=pi05] 

**Duration**: 0m 37s

**Attempt stats**:
- chunks: 107
- rtt_ms: mean 85, p95 89, max 90
- horizon_arm_span (rad): mean 0.737, max 1.080
- state_vs_a0 at boundaries (rad): mean 0.131, max 0.500
- clip rate: 92/11872 dim-steps (0.8%)

**Command**:
```
./eval-yam/scripts/run_repl_pi05.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 8
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: pi05
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: unknown
- `server_host`: 127.0.0.1
- `server_meta`: {'backend': 'pi05', 'transport': 'ws://127.0.0.1:8000'}
- `server_norm_tag`: unknown
- `server_port`: 8000
- `server_repo_id`: unknown
- `server_url`: ws://127.0.0.1:8000
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 13:40:26 -- failure  (repl attempt #2)

**Instruction**: 'pick up the orange cube'

**Notes**: [policy=pi05] 

**Duration**: 0m 33s

**Attempt stats**:
- chunks: 45
- rtt_ms: mean 84, p95 89, max 99
- horizon_arm_span (rad): mean 0.501, max 1.186
- state_vs_a0 at boundaries (rad): mean 0.169, max 0.287
- clip rate: 74/12586 dim-steps (0.6%)

**Command**:
```
./eval-yam/scripts/run_repl_pi05.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 20
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: pi05
- `ramp_duration_s`: 5.0
- `rerun`: True
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: unknown
- `server_host`: 127.0.0.1
- `server_meta`: {'backend': 'pi05', 'transport': 'ws://127.0.0.1:8000'}
- `server_norm_tag`: unknown
- `server_port`: 8000
- `server_repo_id`: unknown
- `server_url`: ws://127.0.0.1:8000
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 13:51:30 -- failure  (repl attempt #1)

**Instruction**: 'pick up the orange cube'

**Notes**: [policy=pi05] 

**Duration**: 0m 14s

**Attempt stats**:
- chunks: 43
- rtt_ms: mean 84, p95 87, max 88
- horizon_arm_span (rad): mean 0.547, max 0.828
- state_vs_a0 at boundaries (rad): mean 0.083, max 0.129
- clip rate: 4/4704 dim-steps (0.1%)

**Command**:
```
./eval-yam/scripts/run_repl_pi05.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 8
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: pi05
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_135100_pi05.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: unknown
- `server_host`: 127.0.0.1
- `server_meta`: {'backend': 'pi05', 'transport': 'ws://127.0.0.1:8000'}
- `server_norm_tag`: unknown
- `server_port`: 8000
- `server_repo_id`: unknown
- `server_url`: ws://127.0.0.1:8000
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 13:52:56 -- failure  (repl attempt #1)

**Instruction**: 'pick up the orange cube'

**Notes**: [policy=pi05] 

**Duration**: 0m 05s

**Attempt stats**:
- chunks: 15
- rtt_ms: mean 85, p95 87, max 88
- horizon_arm_span (rad): mean 0.464, max 0.797
- state_vs_a0 at boundaries (rad): mean 0.097, max 0.120
- clip rate: 3/1568 dim-steps (0.2%)

**Command**:
```
./eval-yam/scripts/run_repl_pi05.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 8
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: pi05
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_135235_pi05.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: unknown
- `server_host`: 127.0.0.1
- `server_meta`: {'backend': 'pi05', 'transport': 'ws://127.0.0.1:8000'}
- `server_norm_tag`: unknown
- `server_port`: 8000
- `server_repo_id`: unknown
- `server_url`: ws://127.0.0.1:8000
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:05:08 -- failure  (repl attempt #1)

**Instruction**: 'pick up the orange cube'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 29s

**Attempt stats**:
- chunks: 63
- rtt_ms: mean 271, p95 313, max 665
- horizon_arm_span (rad): mean 0.160, max 0.828
- state_vs_a0 at boundaries (rad): mean 0.027, max 0.117
- clip rate: 13/5208 dim-steps (0.2%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_140419_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:06:46 -- failure  (repl attempt #2)

**Instruction**: 'pick up the orange cube'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 44s

**Attempt stats**:
- chunks: 96
- rtt_ms: mean 260, p95 313, max 351
- horizon_arm_span (rad): mean 0.271, max 0.812
- state_vs_a0 at boundaries (rad): mean 0.047, max 0.216
- clip rate: 33/8008 dim-steps (0.4%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_140419_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:07:21 -- success  (repl attempt #3)

**Instruction**: 'place the orange cube in the gray box'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 14s

**Attempt stats**:
- chunks: 30
- rtt_ms: mean 280, p95 312, max 779
- horizon_arm_span (rad): mean 0.250, max 0.713
- state_vs_a0 at boundaries (rad): mean 0.055, max 0.160
- clip rate: 60/2436 dim-steps (2.5%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_140419_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:11:30 -- success  (repl attempt #1)

**Instruction**: 'place the blue toy in the box'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 25s

**Attempt stats**:
- chunks: 54
- rtt_ms: mean 277, p95 319, max 766
- horizon_arm_span (rad): mean 0.350, max 0.924
- state_vs_a0 at boundaries (rad): mean 0.054, max 0.246
- clip rate: 90/4452 dim-steps (2.0%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_140805_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:12:28 -- failure  (repl attempt #2)

**Instruction**: 'place the brown track into the gray box'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 19s

**Attempt stats**:
- chunks: 41
- rtt_ms: mean 263, p95 316, max 620
- horizon_arm_span (rad): mean 0.341, max 0.871
- state_vs_a0 at boundaries (rad): mean 0.052, max 0.115
- clip rate: 19/3388 dim-steps (0.6%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_140805_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:12:58 -- failure  (repl attempt #3)

**Instruction**: 'place the brown track in the gray basket'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 11s

**Attempt stats**:
- chunks: 24
- rtt_ms: mean 273, p95 308, max 311
- horizon_arm_span (rad): mean 0.388, max 0.781
- state_vs_a0 at boundaries (rad): mean 0.073, max 0.256
- clip rate: 33/1932 dim-steps (1.7%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_140805_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:13:21 -- failure  (repl attempt #4)

**Instruction**: 'place the object in the gray basket'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 08s

**Attempt stats**:
- chunks: 17
- rtt_ms: mean 296, p95 408, max 763
- horizon_arm_span (rad): mean 0.376, max 0.604
- state_vs_a0 at boundaries (rad): mean 0.087, max 0.211
- clip rate: 19/1386 dim-steps (1.4%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_140805_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:13:39 -- failure  (repl attempt #5)

**Instruction**: 'pick up the object on the right'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 07s

**Attempt stats**:
- chunks: 15
- rtt_ms: mean 276, p95 311, max 317
- horizon_arm_span (rad): mean 0.637, max 1.018
- state_vs_a0 at boundaries (rad): mean 0.115, max 0.208
- clip rate: 34/1176 dim-steps (2.9%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_140805_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:20:55 -- failure  (repl attempt #6)

**Instruction**: 'pick up the object on the right'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 14s

**Attempt stats**:
- chunks: 30
- rtt_ms: mean 268, p95 316, max 320
- horizon_arm_span (rad): mean 0.403, max 0.773
- state_vs_a0 at boundaries (rad): mean 0.086, max 0.202
- clip rate: 38/2492 dim-steps (1.5%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_140805_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:21:13 -- failure  (repl attempt #7)

**Instruction**: 'pick up the object on the left'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 05s

**Attempt stats**:
- chunks: 11
- rtt_ms: mean 289, p95 314, max 316
- horizon_arm_span (rad): mean 0.414, max 0.555
- state_vs_a0 at boundaries (rad): mean 0.091, max 0.144
- clip rate: 26/840 dim-steps (3.1%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_140805_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:22:03 -- failure  (repl attempt #8)

**Instruction**: 'pick up the wood'

**Notes**: [policy=molmoact2] got basket

**Duration**: 0m 05s

**Attempt stats**:
- chunks: 12
- rtt_ms: mean 288, p95 518, max 781
- horizon_arm_span (rad): mean 0.406, max 0.523
- state_vs_a0 at boundaries (rad): mean 0.076, max 0.119
- clip rate: 12/980 dim-steps (1.2%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_140805_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:23:22 -- failure  (repl attempt #9)

**Instruction**: 'pick up the wood'

**Notes**: [policy=molmoact2] 

**Duration**: 1m 05s

**Attempt stats**:
- chunks: 136
- rtt_ms: mean 275, p95 320, max 377
- horizon_arm_span (rad): mean 0.268, max 1.277
- state_vs_a0 at boundaries (rad): mean 0.034, max 0.159
- clip rate: 30/11340 dim-steps (0.3%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_140805_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:23:55 -- failure  (repl attempt #10)

**Instruction**: 'pick up the wood'

**Notes**: [policy=molmoact2] move the wood from right to left

**Duration**: 0m 23s

**Attempt stats**:
- chunks: 50
- rtt_ms: mean 274, p95 316, max 332
- horizon_arm_span (rad): mean 0.317, max 0.920
- state_vs_a0 at boundaries (rad): mean 0.031, max 0.104
- clip rate: 2/4158 dim-steps (0.0%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_140805_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:25:51 -- failure  (repl attempt #1)

**Instruction**: 'move the wood from right to left'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 43s

**Attempt stats**:
- chunks: 93
- rtt_ms: mean 269, p95 316, max 767
- horizon_arm_span (rad): mean 0.266, max 0.800
- state_vs_a0 at boundaries (rad): mean 0.047, max 0.149
- clip rate: 16/7770 dim-steps (0.2%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:27:03 -- failure  (repl attempt #2)

**Instruction**: 'move the wood from right to left'

**Notes**: [policy=molmoact2] does pick it up tho

**Duration**: 1m 01s

**Attempt stats**:
- chunks: 134
- rtt_ms: mean 259, p95 315, max 341
- horizon_arm_span (rad): mean 0.351, max 1.133
- state_vs_a0 at boundaries (rad): mean 0.037, max 0.356
- clip rate: 42/11172 dim-steps (0.4%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:28:02 -- failure  (repl attempt #3)

**Instruction**: 'push the wood from right to left'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 36s

**Attempt stats**:
- chunks: 77
- rtt_ms: mean 274, p95 322, max 335
- horizon_arm_span (rad): mean 0.281, max 0.693
- state_vs_a0 at boundaries (rad): mean 0.036, max 0.136
- clip rate: 2/6384 dim-steps (0.0%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:29:56 -- failure  (repl attempt #4)

**Instruction**: 'push the wood from right to left'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 42s

**Attempt stats**:
- chunks: 90
- rtt_ms: mean 266, p95 318, max 376
- horizon_arm_span (rad): mean 0.269, max 0.859
- state_vs_a0 at boundaries (rad): mean 0.044, max 0.133
- clip rate: 12/7518 dim-steps (0.2%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:34:07 -- failure  (repl attempt #5)

**Instruction**: 'put the blue toy to the left'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 28s

**Attempt stats**:
- chunks: 60
- rtt_ms: mean 271, p95 319, max 368
- horizon_arm_span (rad): mean 0.257, max 0.773
- state_vs_a0 at boundaries (rad): mean 0.024, max 0.068
- clip rate: 0/4970 dim-steps (0.0%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:36:45 -- failure  (repl attempt #6)

**Instruction**: 'place the blue toy to the left'

**Notes**: [policy=molmoact2] 

**Duration**: 2m 21s

**Attempt stats**:
- chunks: 299
- rtt_ms: mean 268, p95 321, max 397
- horizon_arm_span (rad): mean 0.289, max 1.250
- state_vs_a0 at boundaries (rad): mean 0.039, max 0.201
- clip rate: 55/25116 dim-steps (0.2%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:38:50 -- failure  (repl attempt #7)

**Instruction**: 'place the knife on the plate'

**Notes**: [policy=molmoact2] stack the cups together

**Duration**: 1m 33s

**Attempt stats**:
- chunks: 200
- rtt_ms: mean 261, p95 315, max 684
- horizon_arm_span (rad): mean 0.258, max 0.789
- state_vs_a0 at boundaries (rad): mean 0.037, max 0.150
- clip rate: 13/16772 dim-steps (0.1%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:40:24 -- failure  (repl attempt #8)

**Instruction**: 'stack the cups'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 46s

**Attempt stats**:
- chunks: 99
- rtt_ms: mean 267, p95 332, max 790
- horizon_arm_span (rad): mean 0.264, max 0.946
- state_vs_a0 at boundaries (rad): mean 0.045, max 0.210
- clip rate: 30/8232 dim-steps (0.4%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:41:43 -- failure  (repl attempt #10)

**Instruction**: 'stack the cups'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 28s

**Attempt stats**:
- chunks: 62
- rtt_ms: mean 263, p95 321, max 337
- horizon_arm_span (rad): mean 0.373, max 0.680
- state_vs_a0 at boundaries (rad): mean 0.057, max 0.185
- clip rate: 19/5124 dim-steps (0.4%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:43:23 -- failure  (repl attempt #11)

**Instruction**: 'stack the cups'

**Notes**: [policy=molmoact2] 

**Duration**: 1m 09s

**Attempt stats**:
- chunks: 146
- rtt_ms: mean 269, p95 320, max 387
- horizon_arm_span (rad): mean 0.274, max 0.707
- state_vs_a0 at boundaries (rad): mean 0.054, max 0.187
- clip rate: 28/12250 dim-steps (0.2%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:45:23 -- failure  (repl attempt #13)

**Instruction**: 'put the blue toy on the wooden tracks'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 22s

**Attempt stats**:
- chunks: 48
- rtt_ms: mean 269, p95 332, max 346
- horizon_arm_span (rad): mean 0.288, max 0.757
- state_vs_a0 at boundaries (rad): mean 0.053, max 0.195
- clip rate: 29/3948 dim-steps (0.7%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:46:07 -- failure  (repl attempt #14)

**Instruction**: 'pick up the blue toy'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 22s

**Attempt stats**:
- chunks: 49
- rtt_ms: mean 267, p95 314, max 787
- horizon_arm_span (rad): mean 0.283, max 0.630
- state_vs_a0 at boundaries (rad): mean 0.034, max 0.149
- clip rate: 2/4032 dim-steps (0.0%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:47:57 -- failure  (repl attempt #15)

**Instruction**: 'pick up the blue toy'

**Notes**: [policy=molmoact2] 

**Duration**: 1m 35s

**Attempt stats**:
- chunks: 204
- rtt_ms: mean 265, p95 323, max 388
- horizon_arm_span (rad): mean 0.325, max 0.975
- state_vs_a0 at boundaries (rad): mean 0.046, max 0.355
- clip rate: 59/17052 dim-steps (0.3%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:48:46 -- failure  (repl attempt #16)

**Instruction**: 'pick up the blue toy'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 36s

**Attempt stats**:
- chunks: 78
- rtt_ms: mean 268, p95 322, max 337
- horizon_arm_span (rad): mean 0.095, max 0.934
- state_vs_a0 at boundaries (rad): mean 0.020, max 0.026
- clip rate: 0/6468 dim-steps (0.0%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:49:47 -- failure  (repl attempt #17)

**Instruction**: 'pick up the blue toy'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 48s

**Attempt stats**:
- chunks: 101
- rtt_ms: mean 275, p95 314, max 344
- horizon_arm_span (rad): mean 0.068, max 0.566
- state_vs_a0 at boundaries (rad): mean 0.019, max 0.025
- clip rate: 0/8442 dim-steps (0.0%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:51:07 -- failure  (repl attempt #18)

**Instruction**: 'pick up the blue toy'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 12s

**Attempt stats**:
- chunks: 26
- rtt_ms: mean 265, p95 320, max 337
- horizon_arm_span (rad): mean 0.421, max 1.063
- state_vs_a0 at boundaries (rad): mean 0.095, max 0.198
- clip rate: 23/2156 dim-steps (1.1%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_142442_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:53:51 -- failure  (repl attempt #1)

**Instruction**: 'push the blue toy along the wooden tracks'

**Notes**: [policy=molmoact2] 

**Duration**: 1m 14s

**Attempt stats**:
- chunks: 159
- rtt_ms: mean 262, p95 319, max 619
- horizon_arm_span (rad): mean 0.224, max 0.936
- state_vs_a0 at boundaries (rad): mean 0.029, max 0.173
- clip rate: 13/13342 dim-steps (0.1%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_145209_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:54:53 -- failure  (repl attempt #2)

**Instruction**: 'put the blue toy in the circle'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 11s

**Attempt stats**:
- chunks: 23
- rtt_ms: mean 295, p95 340, max 771
- horizon_arm_span (rad): mean 0.281, max 0.531
- state_vs_a0 at boundaries (rad): mean 0.054, max 0.225
- clip rate: 6/1848 dim-steps (0.3%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_145209_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 14:57:18 -- success  (repl attempt #3)

**Instruction**: 'put the blue toy in the circle'

**Notes**: [policy=molmoact2] 

**Duration**: 2m 15s

**Attempt stats**:
- chunks: 285
- rtt_ms: mean 271, p95 323, max 384
- horizon_arm_span (rad): mean 0.323, max 0.942
- state_vs_a0 at boundaries (rad): mean 0.048, max 0.356
- clip rate: 605/23856 dim-steps (2.5%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_145209_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 15:00:21 -- failure  (repl attempt #4)

**Instruction**: 'place the blue toy on the edge of the circle'

**Notes**: [policy=molmoact2] 

**Duration**: 2m 46s

**Attempt stats**:
- chunks: 352
- rtt_ms: mean 268, p95 319, max 788
- horizon_arm_span (rad): mean 0.374, max 1.088
- state_vs_a0 at boundaries (rad): mean 0.033, max 0.230
- clip rate: 221/29484 dim-steps (0.7%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_145209_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 15:33:45 -- failure  (repl attempt #1)

**Instruction**: 'fold the right flap upwards like the left'

**Notes**: [policy=molmoact2] touched the right flap

**Duration**: 0m 36s

**Attempt stats**:
- chunks: 79
- rtt_ms: mean 262, p95 321, max 760
- horizon_arm_span (rad): mean 0.317, max 1.057
- state_vs_a0 at boundaries (rad): mean 0.062, max 0.301
- clip rate: 18/6552 dim-steps (0.3%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_153234_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 15:34:52 -- failure  (repl attempt #2)

**Instruction**: 'grasp the left flap'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 42s

**Attempt stats**:
- chunks: 90
- rtt_ms: mean 274, p95 313, max 784
- horizon_arm_span (rad): mean 0.314, max 1.090
- state_vs_a0 at boundaries (rad): mean 0.054, max 0.248
- clip rate: 29/7476 dim-steps (0.4%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_153234_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 15:36:39 -- failure  (repl attempt #3)

**Instruction**: 'fold the right flap upwards'

**Notes**: [policy=molmoact2] 

**Duration**: 1m 26s

**Attempt stats**:
- chunks: 182
- rtt_ms: mean 272, p95 318, max 788
- horizon_arm_span (rad): mean 0.353, max 1.161
- state_vs_a0 at boundaries (rad): mean 0.053, max 0.169
- clip rate: 52/15232 dim-steps (0.3%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_153234_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 15:40:04 -- failure  (repl attempt #4)

**Instruction**: 'fold the flap upwards'

**Notes**: [policy=molmoact2] did try to fold it

**Duration**: 3m 06s

**Attempt stats**:
- chunks: 393
- rtt_ms: mean 269, p95 317, max 787
- horizon_arm_span (rad): mean 0.390, max 1.006
- state_vs_a0 at boundaries (rad): mean 0.058, max 0.222
- clip rate: 201/32928 dim-steps (0.6%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_153234_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 15:41:24 -- failure  (repl attempt #5)

**Instruction**: 'flatten the green sheet'

**Notes**: [policy=molmoact2] f

**Duration**: 0m 54s

**Attempt stats**:
- chunks: 115
- rtt_ms: mean 271, p95 322, max 385
- horizon_arm_span (rad): mean 0.382, max 1.021
- state_vs_a0 at boundaries (rad): mean 0.059, max 0.227
- clip rate: 59/9590 dim-steps (0.6%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_153234_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 15:51:17 -- failure  (repl attempt #6)

**Instruction**: 'lift the flap'

**Notes**: [policy=molmoact2] lift up the picture frame

**Duration**: 0m 33s

**Attempt stats**:
- chunks: 71
- rtt_ms: mean 273, p95 314, max 775
- horizon_arm_span (rad): mean 0.454, max 1.088
- state_vs_a0 at boundaries (rad): mean 0.076, max 0.211
- clip rate: 109/5880 dim-steps (1.9%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_153234_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 15:52:20 -- failure  (repl attempt #7)

**Instruction**: 'lift the flap'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 13s

**Attempt stats**:
- chunks: 30
- rtt_ms: mean 263, p95 317, max 341
- horizon_arm_span (rad): mean 0.317, max 0.723
- state_vs_a0 at boundaries (rad): mean 0.027, max 0.088
- clip rate: 0/2464 dim-steps (0.0%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_153234_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 15:53:27 -- failure  (repl attempt #8)

**Instruction**: 'lift up the black picture frame'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 52s

**Attempt stats**:
- chunks: 112
- rtt_ms: mean 267, p95 320, max 695
- horizon_arm_span (rad): mean 0.453, max 1.205
- state_vs_a0 at boundaries (rad): mean 0.028, max 0.194
- clip rate: 8/9324 dim-steps (0.1%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_153234_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 15:54:01 -- failure  (repl attempt #9)

**Instruction**: 'put up the picture frame'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 21s

**Attempt stats**:
- chunks: 46
- rtt_ms: mean 259, p95 316, max 768
- horizon_arm_span (rad): mean 0.471, max 1.303
- state_vs_a0 at boundaries (rad): mean 0.084, max 0.260
- clip rate: 63/3850 dim-steps (1.6%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_153234_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 15:55:07 -- failure  (repl attempt #10)

**Instruction**: 'pick up the picture frame'

**Notes**: [policy=molmoact2] 

**Duration**: 0m 26s

**Attempt stats**:
- chunks: 57
- rtt_ms: mean 259, p95 312, max 364
- horizon_arm_span (rad): mean 0.393, max 0.971
- state_vs_a0 at boundaries (rad): mean 0.031, max 0.167
- clip rate: 4/4704 dim-steps (0.1%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_153234_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 16:03:06 -- failure  (repl attempt #1)

**Instruction**: 'Lift up the big flap, then lift up the small flap, then place the big flap on the small flap.'

**Notes**: [ikea-10 task 1/10: FISKBO / 8x10 picture frame] [prompt=atomic_1] [policy=molmoact2] 

**Duration**: 1m 56s

**Attempt stats**:
- chunks: 241
- rtt_ms: mean 281, p95 316, max 791
- horizon_arm_span (rad): mean 0.295, max 1.326
- state_vs_a0 at boundaries (rad): mean 0.032, max 0.174
- clip rate: 7/20202 dim-steps (0.0%)

**Command**:
```
./ikea-10/eval/run_eval_molmoact2.sh --rerun
```

**Configuration**:
- `attempt_timeout_s`: 120.0
- `attempts`: 3
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `results_dir`: /home/andon/yam-tests/ikea-10/eval/results/molmoact2
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 16:06:29 -- failure  (repl attempt #1)

**Instruction**: 'Put up the picture frame.'

**Notes**: [ikea-10 task 1/10: FISKBO / 8x10 picture frame] [prompt=atomic_2] [policy=molmoact2] 

**Duration**: 2m 00s (timed out)

**Attempt stats**:
- chunks: 249
- rtt_ms: mean 278, p95 315, max 798
- horizon_arm_span (rad): mean 0.302, max 0.865
- state_vs_a0 at boundaries (rad): mean 0.057, max 0.256
- clip rate: 85/20916 dim-steps (0.4%)

**Command**:
```
./ikea-10/eval/run_eval_molmoact2.sh --rerun
```

**Configuration**:
- `attempt_timeout_s`: 120.0
- `attempts`: 3
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `results_dir`: /home/andon/yam-tests/ikea-10/eval/results/molmoact2
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 16:08:43 -- failure  (repl attempt #2)

**Instruction**: 'Put up the picture frame.'

**Notes**: [ikea-10 task 1/10: FISKBO / 8x10 picture frame] [prompt=atomic_2] [policy=molmoact2] 

**Duration**: 1m 38s

**Attempt stats**:
- chunks: 206
- rtt_ms: mean 276, p95 318, max 387
- horizon_arm_span (rad): mean 0.254, max 1.121
- state_vs_a0 at boundaries (rad): mean 0.040, max 0.193
- clip rate: 32/17220 dim-steps (0.2%)

**Command**:
```
./ikea-10/eval/run_eval_molmoact2.sh --rerun
```

**Configuration**:
- `attempt_timeout_s`: 120.0
- `attempts`: 3
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `results_dir`: /home/andon/yam-tests/ikea-10/eval/results/molmoact2
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 16:14:36 -- failure  (repl attempt #1)

**Instruction**: 'Lift up the right metal end.'

**Notes**: [ikea-10 task 2/10: GREJIG / BAGGMUCK / shoe rack + drip tray (set of 3)] [prompt=atomic_2] [policy=molmoact2] 

**Duration**: 1m 17s

**Attempt stats**:
- chunks: 167
- rtt_ms: mean 261, p95 313, max 764
- horizon_arm_span (rad): mean 0.292, max 1.038
- state_vs_a0 at boundaries (rad): mean 0.023, max 0.165
- clip rate: 8/13944 dim-steps (0.1%)

**Command**:
```
./ikea-10/eval/run_eval_molmoact2.sh --rerun
```

**Configuration**:
- `attempt_timeout_s`: 120.0
- `attempts`: 3
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `results_dir`: /home/andon/yam-tests/ikea-10/eval/results/molmoact2
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 16:15:49 -- failure  (repl attempt #2)

**Instruction**: 'Lift up the right metal end.'

**Notes**: [ikea-10 task 2/10: GREJIG / BAGGMUCK / shoe rack + drip tray (set of 3)] [prompt=atomic_2] [policy=molmoact2] 

**Duration**: 0m 42s

**Attempt stats**:
- chunks: 96
- rtt_ms: mean 236, p95 302, max 371
- horizon_arm_span (rad): mean 0.240, max 0.874
- state_vs_a0 at boundaries (rad): mean 0.018, max 0.036
- clip rate: 0/7994 dim-steps (0.0%)

**Command**:
```
./ikea-10/eval/run_eval_molmoact2.sh --rerun
```

**Configuration**:
- `attempt_timeout_s`: 120.0
- `attempts`: 3
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `results_dir`: /home/andon/yam-tests/ikea-10/eval/results/molmoact2
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 16:26:17 -- failure  (repl attempt #1)

**Instruction**: 'Lift up the right white leg.'

**Notes**: [ikea-10 task 3/10: KLIPSK / bed tray] [prompt=atomic_3] [policy=molmoact2] 

**Duration**: 1m 26s

**Attempt stats**:
- chunks: 177
- rtt_ms: mean 285, p95 317, max 784
- horizon_arm_span (rad): mean 0.260, max 1.119
- state_vs_a0 at boundaries (rad): mean 0.021, max 0.083
- clip rate: 0/14798 dim-steps (0.0%)

**Command**:
```
./ikea-10/eval/run_eval_molmoact2.sh --rerun
```

**Configuration**:
- `attempt_timeout_s`: 120.0
- `attempts`: 3
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `results_dir`: /home/andon/yam-tests/ikea-10/eval/results/molmoact2
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 16:28:49 -- failure  (repl attempt #2)

**Instruction**: 'Flip the table upright.'

**Notes**: [ikea-10 task 3/10: KLIPSK / bed tray] [prompt=atomic_5] [policy=molmoact2] 

**Duration**: 2m 00s (timed out)

**Attempt stats**:
- chunks: 247
- rtt_ms: mean 281, p95 318, max 694
- horizon_arm_span (rad): mean 0.408, max 1.254
- state_vs_a0 at boundaries (rad): mean 0.019, max 0.129
- clip rate: 0/20748 dim-steps (0.0%)

**Command**:
```
./ikea-10/eval/run_eval_molmoact2.sh --rerun
```

**Configuration**:
- `attempt_timeout_s`: 120.0
- `attempts`: 3
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `results_dir`: /home/andon/yam-tests/ikea-10/eval/results/molmoact2
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 16:58:15 -- failure  (repl attempt #1)

**Instruction**: 'Turn the wooden leg clockwise five more times to tighten it partway.'

**Notes**: [ikea-10 task 5/10: LACK / side table] [prompt=atomic_4] [policy=molmoact2] 

**Duration**: 0m 18s

**Attempt stats**:
- chunks: 38
- rtt_ms: mean 295, p95 315, max 779
- horizon_arm_span (rad): mean 0.309, max 0.734
- state_vs_a0 at boundaries (rad): mean 0.056, max 0.137
- clip rate: 17/3108 dim-steps (0.5%)

**Command**:
```
./ikea-10/eval/run_eval_molmoact2.sh --rerun
```

**Configuration**:
- `attempt_timeout_s`: 120.0
- `attempts`: 3
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `results_dir`: /home/andon/yam-tests/ikea-10/eval/results/molmoact2
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 17:18:24 -- failure  (repl attempt #1)

**Instruction**: 'Pick up the black square and put it on the missing corner.'

**Notes**: [ikea-10 task 6/10: LÄMPLIG / stainless steel trivet] [prompt=atomic_1] [policy=molmoact2] 

**Duration**: 2m 00s (timed out)

**Attempt stats**:
- chunks: 247
- rtt_ms: mean 282, p95 318, max 770
- horizon_arm_span (rad): mean 0.353, max 1.358
- state_vs_a0 at boundaries (rad): mean 0.038, max 0.201
- clip rate: 39/20748 dim-steps (0.2%)

**Command**:
```
./ikea-10/eval/run_eval_molmoact2.sh --rerun
```

**Configuration**:
- `attempt_timeout_s`: 120.0
- `attempts`: 3
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `results_dir`: /home/andon/yam-tests/ikea-10/eval/results/molmoact2
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 17:21:02 -- failure  (repl attempt #1)

**Instruction**: 'press on the upper right black square until tight'

**Notes**: [policy=molmoact2] 

**Duration**: 1m 04s

**Attempt stats**:
- chunks: 133
- rtt_ms: mean 280, p95 318, max 786
- horizon_arm_span (rad): mean 0.189, max 1.221
- state_vs_a0 at boundaries (rad): mean 0.023, max 0.169
- clip rate: 6/11102 dim-steps (0.1%)

**Command**:
```
./eval-yam/scripts/run_repl_molmoact2.sh --rerun
```

**Configuration**:
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `rerun_save`: /home/andon/yam-tests/eval-yam/logs/rrd/2026-05-25_171911_molmoact2.rrd
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 17:28:54 -- failure  (repl attempt #1)

**Instruction**: 'Lift up the flap.'

**Notes**: [ikea-10 task 7/10: LÅNESPELARE / foldable laptop support] [prompt=atomic_1] [policy=molmoact2] 

**Duration**: 1m 37s

**Attempt stats**:
- chunks: 200
- rtt_ms: mean 286, p95 317, max 759
- horizon_arm_span (rad): mean 0.192, max 0.843
- state_vs_a0 at boundaries (rad): mean 0.036, max 0.214
- clip rate: 56/16716 dim-steps (0.3%)

**Command**:
```
./ikea-10/eval/run_eval_molmoact2.sh --rerun
```

**Configuration**:
- `attempt_timeout_s`: 120.0
- `attempts`: 3
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `results_dir`: /home/andon/yam-tests/ikea-10/eval/results/molmoact2
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-25 17:43:04 -- failure  (repl attempt #1)

**Instruction**: 'Flip the stand upside down.'

**Notes**: [ikea-10 task 7/10: LÅNESPELARE / foldable laptop support] [prompt=atomic_3] [policy=molmoact2] 

**Duration**: 2m 00s (timed out)

**Attempt stats**:
- chunks: 248
- rtt_ms: mean 280, p95 317, max 773
- horizon_arm_span (rad): mean 0.256, max 0.909
- state_vs_a0 at boundaries (rad): mean 0.043, max 0.208
- clip rate: 141/20832 dim-steps (0.7%)

**Command**:
```
./ikea-10/eval/run_eval_molmoact2.sh --rerun
```

**Configuration**:
- `attempt_timeout_s`: 120.0
- `attempts`: 3
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `journal_path`: /home/andon/yam-tests/molmoact2-setup/journal.md
- `left_cam_serial`: 427622271914
- `left_can`: can1
- `left_gripper`: linear_4310
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `ramp_duration_s`: 5.0
- `rerun`: True
- `results_dir`: /home/andon/yam-tests/ikea-10/eval/results/molmoact2
- `right_cam_serial`: 352122272708
- `right_can`: can0
- `right_gripper`: linear_4310
- `server_dtype`: torch.bfloat16
- `server_host`: 127.0.0.1
- `server_meta`: "{'status': 'ok', 'repo_id': 'allenai/MolmoAct2-BimanualYAM', 'norm_tag': 'yam_dual_molmoact2', 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'num_cameras': 3, 'state_dim': 14}"
- `server_norm_tag`: yam_dual_molmoact2
- `server_repo_id`: allenai/MolmoAct2-BimanualYAM
- `server_url`: http://127.0.0.1:8202/act
- `timeout_s`: 15.0
- `top_cam_serial`: 349622072241
- `train_fps`: 30.0
- `warmup_timeout_s`: 60.0


---
## 2026-05-26 15:21:07 -- success

**Notes**: [policy=molmoact2] eval=bimanual_easy_bench_8  

**Duration**: 5m 22s

**Command**:
```
scripts/run_eval.py --policy molmoact2 --eval bimanual_easy_bench_8 --samples 1 --dry-run
```

**Configuration**:
- `attempts`: 1
- `cam_fps`: 30
- `cam_height`: 240
- `cam_width`: 424
- `config_dir`: /home/andon/yam-vla-tests/configs/policy
- `dry_run`: True
- `eval_name`: bimanual_easy_bench_8
- `evals_dir`: /home/andon/yam-vla-tests/evals
- `gripper_step`: 0.15
- `horizon_stride`: 6
- `inference_mode`: sync
- `journal_path`: /home/andon/yam-vla-tests/journal.md
- `max_chunks`: 200
- `max_step_rad`: 0.15
- `num_steps`: 10
- `policy`: molmoact2
- `timeout_s`: 15.0
- `train_fps`: 30.0

