
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

