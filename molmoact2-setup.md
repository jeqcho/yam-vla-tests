# MolmoAct2 on Bimanual YAM — Setup Report

**Working directory:** `/home/andon/yam-tests/molmoact2-setup/`
**Target task:** `"first pick up the left orange cube and put it in the box, then pick up the right orange cube and put it in the box"`

## Bottom line

- ✅ **MolmoAct2 inference server is RUNNING on :8202** (tmux session `server`). Synthetic smoke test posted 3 rounds of fake frames; the model returned `(30, 14)` finite action tensors.
- ✅ Model weights `allenai/MolmoAct2-BimanualYAM` (21 GB on disk) cached at `molmoact2-setup/hf-cache/`
- ✅ Launcher scripts: `scripts/run_server.sh`, `scripts/run_client.sh`
- ✅ Client bridge `scripts/yam_client.py` — 2 YAM follower arms + 3 RealSense cameras, polls the server, applies per-tick-clipped joint commands
- ✅ Utilities: `scripts/preflight.py`, `scripts/list_cams.py`, `scripts/capture_frames.py`, `scripts/smoke_test_server.py`
- ✅ Blackwell sm_120 / RTX 5090: server runs cleanly on torch 2.8.0+cu128 (the Ai2-pinned cu121 wheels failed with `no kernel image is available`).
- ✅ The legacy bf16 patches in `host_server_yam.py` log `needle not found` — that's expected on the YAM revision (fixed upstream); bf16 inference works regardless.

### Smoke-test stdout (from `scripts/smoke_test_server.py`, T+75m)

```
posting 3 round(s) of synthetic frames to http://127.0.0.1:8202/act
  round 0: actions shape=(30, 14), |a-s|_max(first step)=0.945, server dt=791 ms, rtt=794 ms
  round 1: actions shape=(30, 14), |a-s|_max(first step)=0.918, server dt=346 ms, rtt=349 ms
  round 2: actions shape=(30, 14), |a-s|_max(first step)=0.742, server dt=331 ms, rtt=334 ms
smoke test PASS
```

Action horizon is **30 steps × 14-D**, matching `norm_stats.json["yam_dual_molmoact2"]["action_horizon"] = 30`. Steady-state server inference ~330 ms; with the default `--horizon-stride 6` and `--train-fps 30`, the effective server-query rate is ~1.9 Hz while arm commands flow at 30 Hz between queries.

## You still need to do on return

1. **Plug in the second YAM arm.** `can1` had zero packets ever — only one arm was online when I probed. The bimanual policy is 14-D and won't work unimanual.
2. **Plug in 3 RealSense cameras** (D435 overhead + 2× D405) — none detected on USB.
3. **Install librealsense udev rules.** The `pyrealsense2` pip wheel does NOT install them; without them, non-root `rs.context().query_devices()` returns 0 even with cameras plugged in. Drop `99-realsense-libusb.rules` from `IntelRealSense/librealsense` into `/etc/udev/rules.d/`, then `sudo udevadm control --reload-rules && sudo udevadm trigger`.
4. **Verify gripper variant per arm** — pictures sent earlier (`linear_4310` / `linear_3507` / `crank_4310` / `flexible_4310`); pass via `--left-gripper`/`--right-gripper`.
5. **`gh auth login`** — no GitHub auth means I couldn't push the `molmoact2-setup/` repo. Locally committed only.
6. **Decide which CAN bus is left vs right** — see "State + left/right convention" below.

## Hardware survey (snapshot, T+0)

| Thing | State |
|---|---|
| GPU | NVIDIA RTX 5090, 32 GB VRAM, driver 595.71.05 (CUDA 13.2) |
| nvcc | 12.0.140 (not used at runtime; torch wheels ship their own CUDA libs) |
| `can0` | UP @ 1 Mbit/s, parent `usb 3-9.1`, RX=2 / TX=2 packets (from earlier init attempts) |
| `can1` | UP @ 1 Mbit/s, parent `usb 3-9.3`, **RX=0 / TX=0 packets** — no arm responding |
| CAN persistence | `/etc/udev/rules.d/flow_base.rules` already installed — `can*` auto-up at 1 Mbit/s on plug-in. Survives reboot. |
| RealSense cameras | **0 detected on USB.** No udev rules installed (see "still need to do" #3) |
| i2rt SDK | installed in `/home/andon/yam-tests/i2rt/.venv` (python 3.11) |
| mujoco | installed, sim YAM works end-to-end (7 DoFs read, gravity-comp OK) |
| Added to i2rt venv | `pyrealsense2 2.57.7`, `json-numpy 2.1.1`, `requests` |
| `tmux` session `server` | inference server running, holds ~22 GB VRAM for the life of the session |

Blackwell note: built against cu128 wheels (torch 2.8.0). No PTX-JIT fallback needed; bf16 matmul + full inference verified.

## Directory layout

```
/home/andon/yam-tests/
├── i2rt/                              # upstream i2rt-robotics/i2rt (DO NOT push)
│   └── .venv/                         # i2rt SDK + mujoco + pyrealsense2 + json-numpy + requests
├── molmoact2-setup/                   # NEW — workspace (local git, no remote yet)
│   ├── .venv/                         # torch 2.8.0+cu128, transformers 4.57.x, einops, fastapi
│   ├── molmoact2/                     # clone of allenai/molmoact2 (main)
│   ├── hf-cache/                      # MolmoAct2-BimanualYAM weights (21 GB)
│   ├── scripts/
│   │   ├── run_server.sh              # launches the FastAPI server (port 8202)
│   │   ├── run_client.sh              # launches the YAM client with the orange-cube instruction
│   │   ├── yam_client.py              # bimanual client loop
│   │   ├── preflight.py               # cams + CAN + arms + server pre-flight
│   │   ├── list_cams.py               # enumerate RealSense serials
│   │   ├── capture_frames.py          # snap one frame per camera to PNG
│   │   └── smoke_test_server.py       # post synthetic frames, verify (30, 14) actions
│   ├── logs/{install.log,server.log}  # install + server tmux output
│   ├── pyproject.toml                 # torch 2.8.0+cu128, transformers 4.57.x, einops, etc
│   ├── uv.lock
│   ├── HANDOFF.md                     # one-page bring-up checklist
│   └── .python-version                # 3.11
└── reports/
    └── molmoact2-setup.md             # ← you are here
```

## How to run the orange-cube task

The server is already up (`tmux attach -t server`). Steps below assume both arms + 3 cameras are connected and udev rules are installed.

```bash
# 0. From a fresh shell — confirm the server is alive
curl http://127.0.0.1:8202/act
# Expect: {"status":"ok","repo_id":"allenai/MolmoAct2-BimanualYAM",
#          "norm_tag":"yam_dual_molmoact2","num_cameras":3,"state_dim":14, ...}

# If it isn't running:
cd /home/andon/yam-tests/molmoact2-setup
tmux kill-session -t server 2>/dev/null
tmux new -d -s server "./scripts/run_server.sh 2>&1 | tee logs/server.log"
# Wait until logs/server.log shows "Listening on 0.0.0.0:8202" (~15 sec).

# 1. Enumerate cameras (need serials for the client)
/home/andon/yam-tests/i2rt/.venv/bin/python scripts/list_cams.py
# Note the three serial numbers — physically inspect to confirm which is top/left/right.

# 2. Pre-flight check. WARNING: this initialises both arms, which triggers
#    auto-gripper-calibration on linear_4310 / linear_3507 / flexible_4310 —
#    they will fully open and close. Clear the jaws first.
/home/andon/yam-tests/i2rt/.venv/bin/python scripts/preflight.py \
    --left-can can0 --right-can can1 \
    --left-gripper linear_4310 --right-gripper linear_4310

# 3. Dry-run the policy — NO arm movement
./scripts/run_client.sh \
    --left-can can0 --right-can can1 \
    --left-gripper linear_4310 --right-gripper linear_4310 \
    --top-cam-serial   XXXX \
    --left-cam-serial  YYYY \
    --right-cam-serial ZZZZ \
    --train-fps 30 --horizon-stride 6 \
    --max-step-rad 0.05 --gripper-step 0.05 \
    --dry-run

# 4. If actions look sane, drop --dry-run.
```

### Server lifecycle

- Holds ~22 GB VRAM for the life of the tmux session (no auto-release).
- Doesn't need sudo.
- Restart: `tmux kill-session -t server && tmux new -d -s server "./scripts/run_server.sh 2>&1 | tee logs/server.log"`
- `HF_HOME=/home/andon/yam-tests/molmoact2-setup/hf-cache` is set inside `run_server.sh`. If you ever invoke `huggingface-cli` / `snapshot_download` directly without that env, weights will re-download to `~/.cache/huggingface`. Either always go through `run_server.sh`, or persist `HF_HOME` in `~/.bashrc`.

## Wire format

From `examples/yam/host_server_yam.py` and `norm_stats.json`:

**POST `/act` request (json_numpy):**

| Field | Shape/Type | Notes |
|---|---|---|
| `top_cam` | `(H, W, 3) uint8 RGB` | overhead D435; order matters |
| `left_cam` | `(H, W, 3) uint8 RGB` | D405 close-up of left arm |
| `right_cam` | `(H, W, 3) uint8 RGB` | D405 close-up of right arm |
| `instruction` | `str` | the natural-language task |
| `state` | `(14,) float32` | `[left_q0..q5, left_grip, right_q0..q5, right_grip]` |
| `num_steps` | int (opt, 10) | flow-matching denoising steps |
| `enable_cuda_graph` | bool (opt) | per-request override |

**Response:** `{"actions": (30, 14) float32, "dt_ms": float}` — action horizon hardcoded at 30 per `norm_stats.json["yam_dual_molmoact2"]["action_horizon"]`.

`norm_tag = "yam_dual_molmoact2"` — drives action de-normalization.

## State + left/right convention

Per `norm_stats.json["yam_dual_molmoact2"]`:

```
state[0..5]   left arm joints (q0..q5)   radians   range: roughly the YAM joint limits
state[6]      left gripper                [0, 1]   normalized (0 = closed, 1 = open)
state[7..12]  right arm joints (q0..q5)  radians
state[13]     right gripper               [0, 1]   normalized
```

The gripper is **[0, 1] normalized**, not radians. The i2rt SDK's `MotorChainRobot._motor_state_to_joint_state` (motor_chain_robot.py:457) normalizes the gripper to [0, 1] using `gripper_limits`. For `crank_4310`, limits are pre-set (`[0.0, -2.7]`). For `linear_4310` / `linear_3507` / `flexible_4310`, `gripper_limits` start `None` with `needs_calibration: True`, so the SDK runs `detect_gripper_limits()` at init — which drives the gripper to both end-stops to learn its range — and then normalizes thereafter.

**Implication:** the first call to `preflight.py` or `run_client.sh` will physically open and close each gripper. Clear the jaws before running.

`get_joint_pos()` returns `(7,)` per arm with the gripper already normalized — so `read_state()` in `yam_client.py` produces a `(14,)` vector matching the policy's expected layout directly. No manual normalization needed.

**Left vs right.** "Left" and "right" are from the operator's POV looking at the workspace. Verify by running with `--dry-run`, lifting one arm out of view, and confirming which `left_cam` / `right_cam` it appears in. If swapped, exchange `--left-can` / `--right-can` (and the matching camera serials).

## Client safety

`scripts/yam_client.py` defends against runaway commands:
- Per-tick joint delta capped at `--max-step-rad` (default **0.05 rad ≈ 2.9°/tick**)
- Gripper delta capped at `--gripper-step` (default **0.05**, normalized units)
- `--dry-run` prints actions without commanding
- `SIGINT` (Ctrl+C) stops the loop and exits; arms hold their last position — **kill power if the pose isn't safe**

Why the per-tick clip matters: the smoke test showed first-step deltas of 0.7–0.9 rad on synthetic input. Without the clip, the arm would jerk hard at every server query. The policy expects you to play out the horizon (30 steps) at training cadence, not slam to step 0.

## Open questions for you

1. **Gripper variant per arm.** Pictures sent earlier. Factory default is usually `linear_4310`.
2. **Camera mounting.** D435 overhead, two D405s positioned per Ai2's reference design (`assets/m.png` in the molmoact2 repo).
3. **Workspace.** Two orange cubes and a box positioned within each arm's reach (~700 mm each).
4. **Push remote.** Local-only until `gh auth login`, then `gh repo create yam-molmoact2 --source=/home/andon/yam-tests/molmoact2-setup --private --push`.

## Progress log

(Newest first.)

- T+90m — Subagent review applied. Fixed gripper-units error ([0,1] not radians), removed fluff, added udev/CAN/server-lifecycle notes, corrected stride math.
- T+75m — **Server up, smoke test PASS.** Tmux session `server`. Returned (30, 14) actions, server dt ~330 ms.
- T+60m — HF download complete (21 GB on disk). `einops` missing dep → added.
- T+50m — Disk pressure noted (peaked at 99%); resolved after dedup.
- T+45m — Client + launchers committed locally.
- T+30m — Wrote `yam_client.py` with safety caps.
- T+25m — Added `pyrealsense2`, `json-numpy`, `requests` to i2rt venv. Sim YAM smoke test passed.
- T+15m — Started `uv sync` + `hf download` in tmux `mact2`.
- T+10m — Cloned `allenai/molmoact2`. `pyproject.toml` not in main → wrote one. Initially targeted cu121 (Ai2's validated stack); failed on Blackwell sm_120 → bumped to cu128.
- T+5m — Confirmed RTX 5090, CAN up, RealSense absent.
- T+0  — Workspace + report scaffolded.
