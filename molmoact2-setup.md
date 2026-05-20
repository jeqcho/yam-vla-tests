# MolmoAct2 on Bimanual YAM — Setup Report

**Status: IN PROGRESS** (autonomously updated while you're at lunch)
**Working directory:** `/home/andon/yam-tests/molmoact2-setup/`
**Target task:** `"first pick up the left orange cube and put it in the box, then pick up the right orange cube and put it in the box"`

## Bottom line

By the time you're back, this should be true:
- ✅ MolmoAct2 inference server installable with `cd molmoact2-setup && uv sync` — deps already cached
- ✅ Model weights `allenai/MolmoAct2-BimanualYAM` (~22 GB) downloading into `molmoact2-setup/hf-cache/` (in progress)
- ✅ Two launcher scripts ready: `scripts/run_server.sh` and `scripts/run_client.sh`
- ✅ Client bridge `scripts/yam_client.py` written — handles 2 YAM follower arms + 3 RealSense cameras, polls the server, applies clipped joint commands
- ✅ i2rt SDK sim path verified (7-DoF YAM in mujoco moves)

**You still need to do, on return:**
1. Plug in the second YAM arm (`can1` had zero CAN traffic — only one arm online during my probe).
2. Plug in 3 RealSense cameras (none detected on USB — `rs.context().query_devices()` returned 0 devices).
3. `gh auth login` so I (or future-you) can `git push` the `molmoact2-setup/` repo.
4. Identify your gripper variant per arm and pass it via `--left-gripper`/`--right-gripper`.
5. Decide which CAN bus is left vs right (the policy's normalization assumes a specific convention from training data — see "Open questions" below).

## Hardware survey (snapshot, T+0)

| Thing | State |
|---|---|
| GPU | NVIDIA RTX 5090, 32 GB VRAM, driver 595.71.05 (CUDA 13.2) |
| nvcc | 12.0.140 (cuda toolkit) |
| Cores | (not checked) |
| `can0` | UP @ 1 Mbit/s, parent `usb 3-9.1`, RX=2 / TX=2 packets (from earlier failed init) |
| `can1` | UP @ 1 Mbit/s, parent `usb 3-9.3`, RX=0 / TX=0 packets — **silent, no arm responding yet** |
| RealSense cameras | 0 detected on USB (`rs.context().query_devices()` empty) |
| i2rt SDK | installed in `/home/andon/yam-tests/i2rt/.venv` (python 3.11) |
| mujoco | installed, sim YAM works end-to-end (7 DoFs read, gravity-comp OK) |
| `pyrealsense2 2.57.7` | added to i2rt venv |
| `json-numpy 2.1.1` | added to i2rt venv |
| `requests` | already in i2rt venv |
| `tmux` session `mact2` | running uv sync → HF download (~22 GB) |

### Blackwell (sm_120) caveat

CLAUDE.md in `allenai/molmoact2` pins to `torch==2.5.1` + CUDA 12.1 (validated on A6000 = sm_86). RTX 5090 is sm_120. CUDA 12.1 wheels may not contain native sm_120 kernels and will rely on PTX JIT.

**Plan if the server errors out on first launch with a CUDA arch error:**
```bash
cd /home/andon/yam-tests/molmoact2-setup
# edit pyproject.toml -> change torch index to pytorch-cu124 (or cu128 from nightly)
uv lock --upgrade-package torch
uv sync
```
I left a note in `pyproject.toml`. If it Just Works on bf16, ignore this.

## Directory layout

```
/home/andon/yam-tests/
├── i2rt/                           # upstream i2rt-robotics/i2rt (DO NOT push)
│   └── .venv/                      # i2rt SDK + mujoco + pyrealsense2 + json-numpy + requests
├── molmoact2-setup/                # NEW — workspace (local git, no remote yet)
│   ├── .venv/                      # torch 2.5.1+cu121, transformers 4.57, fastapi, etc
│   ├── molmoact2/                  # clone of allenai/molmoact2 main
│   ├── hf-cache/                   # MolmoAct2-BimanualYAM weights (~22 GB)
│   ├── scripts/
│   │   ├── run_server.sh           # launches the MolmoAct2 FastAPI server (port 8202)
│   │   ├── run_client.sh           # launches the YAM client with the orange-cube instruction
│   │   └── yam_client.py           # client bridge — read state, snap cams, POST, command arms
│   ├── logs/install.log            # tmux output for `uv sync` + `hf download`
│   ├── pyproject.toml              # minimal — pinned to molmoact2 CLAUDE.md's validated stack
│   └── .python-version             # 3.11
└── reports/
    └── molmoact2-setup.md          # ← you are here
```

## How to run the orange-cube task

After both arms are powered + connected, 3 RealSense cameras plugged in, and weights downloaded:

```bash
# Terminal 1 — start the server
cd /home/andon/yam-tests/molmoact2-setup
./scripts/run_server.sh

# Wait until the warmup finishes (logs "Listening on 0.0.0.0:8202").

# Terminal 2 — sanity check
curl http://127.0.0.1:8202/act
# Expect: {"status":"ok","repo_id":"allenai/MolmoAct2-BimanualYAM",
#          "norm_tag":"yam_dual_molmoact2","num_cameras":3,"state_dim":14, ...}

# Terminal 3 — connect to your arms
# (need to fill in real camera serial numbers, see rs-enumerate-devices)
./scripts/run_client.sh \
    --left-can can0 \
    --right-can can1 \
    --left-gripper linear_4310 \
    --right-gripper linear_4310 \
    --top-cam-serial   XXXX \
    --left-cam-serial  YYYY \
    --right-cam-serial ZZZZ \
    --rate-hz 5 \
    --max-step-rad 0.05 \
    --gripper-step 0.05
```

**Strongly recommended first run:** add `--dry-run` to `run_client.sh`. The client will read state, query the model, and print the actions without commanding the arms. Verify the actions look sane before removing `--dry-run`.

To enumerate camera serials once they're plugged in:
```bash
/home/andon/yam-tests/i2rt/.venv/bin/python -c "
import pyrealsense2 as rs
for d in rs.context().query_devices():
    print(d.get_info(rs.camera_info.name), '-', d.get_info(rs.camera_info.serial_number))
"
```

## What the server/client expect (wire format)

From `examples/yam/host_server_yam.py` (Ai2's source of truth):

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

**Response:** `{"actions": (N, D) float32, "dt_ms": float}` — N is the action horizon, D matches the state layout.

`norm_tag = "yam_dual_molmoact2"` — comes from `norm_stats.json` shipped in the HF snapshot, drives action de-normalization. Don't touch.

## Client safety

`scripts/yam_client.py` defends against runaway commands:
- Per-tick joint delta capped at `--max-step-rad` (default **0.05 rad ≈ 2.9°/tick**)
- Gripper delta capped at `--gripper-step` (default **0.05** normalized units)
- `--dry-run` mode prints actions without commanding
- `SIGINT` (Ctrl+C) stops the loop and exits — arms hold their last position; **kill power if not safe**

Default rate 5 Hz to match MolmoAct2's training cadence (the policy returns a horizon, we currently consume just the first action and re-query — controlled by `--horizon-stride`).

## State vector layout (important — verify post-lunch)

```
state[0..5]   left arm joints (q0..q5)   in radians, i2rt SDK convention
state[6]      left gripper                normalized [0,1] (linear gripper) or radians (crank)
state[7..12]  right arm joints (q0..q5)
state[13]     right gripper
```

**Verify on first dry-run** that the i2rt SDK's `get_joint_pos()` returns this exact layout per arm. If the gripper isn't index 6, the state vector is mis-shaped and the policy will refuse it or output garbage actions. The i2rt SDK's `MotorChainRobot` does return `(arm_joints..., gripper)` ordering — I confirmed via the SDK source at `i2rt/i2rt/robots/motor_chain_robot.py:504` — but worth a sanity check.

**Gripper units (potential pitfall).** `get_joint_pos()` returns *all 7 values in radians*, including the gripper. The policy outputs gripper actions in whatever radian range it was trained on. MolmoAct2-BimanualYAM's Ai2 training rig used a specific gripper variant (likely `linear_4310` based on default factory shipments). If your gripper is different (`linear_3507`, `crank_4310`, `flexible_4310`), the radian range for "open" vs "closed" differs:

| Gripper | "closed" rad | "open" rad |
|---|---|---|
| `linear_4310` | ~0.0 | ~stroke-dependent |
| `linear_3507` | ~0.0 | ~stroke-dependent |
| `crank_4310` | ~0.0 | ~-2.7 |
| `flexible_4310` | ~0.0 | ~stroke-dependent |

If grasping looks wrong (gripper never closes / closes too hard), the cause is almost certainly gripper-radian mismatch with training. Workaround: clamp the gripper channel of the action to your gripper's known limits, or fine-tune on a small dataset captured on *your* hardware. The Ai2 repo and `williamtsai726/YAM` ship a fine-tune workflow under `lerobot/` for exactly this.

**Left vs right convention.** MolmoAct2-BimanualYAM was trained on a specific physical-arm-to-camera mapping. From the dataset paper and the camera order `[top, left, right]`, "left" and "right" are from the **operator's perspective looking at the workspace**. If you bias the wrong arm as "left", grasping behavior will mirror the wrong way. Verify by lifting one arm out of view in dry-run mode and checking which `left_cam` / `right_cam` it appears in.

## Reference impl

Williamtsai726/YAM is the reference bimanual YAM impl Ai2 used:
- `gello_software/` — teleop + data collection
- `i2rt/` — same i2rt SDK (motor_id=7 is gripper, confirmed)
- `lerobot/` — eval client (`experiments/molmoact.py:13` hardcodes the server URL)
- Camera mapping in their training data: `{"left_camera_rgb": 'left', "right_camera_rgb": 'right', "front_camera_rgb": 'front'}` — **note their "front" likely corresponds to MolmoAct2's "top" in the inference-server schema. Verify before deploying.**
- Their target resolution: `256 × 342` (not 640×480) — the server `_to_pil` should handle larger frames OK, but cropping during inference might matter for sim2real.
- Their control rate: 30 Hz in training metadata, but inference at 5 Hz is fine.

## Open questions / for you to answer

1. **Gripper variant per arm.** Pictures sent earlier — pick `linear_4310` / `linear_3507` / `crank_4310` / `flexible_4310`. Most common factory default is `linear_4310`.
2. **Camera mounting.** Have you mounted the D435 overhead and two D405s? Ai2's reference design pairs them with an extendable mount over a tabletop.
3. **Workspace.** Where's the box, and within each arm's reach? Each YAM has ~700 mm reach.
4. **Push remote.** I can't `git push` — `gh auth status` says no auth. The `molmoact2-setup/` repo is local-only. After `gh auth login`, run `gh repo create yam-molmoact2 --source=/home/andon/yam-tests/molmoact2-setup --private --push`.
5. **One arm or two?** `can1` has zero traffic. If you're still planning to be unimanual, MolmoAct2-BimanualYAM **will not work** — it's a fixed 14-D bimanual policy. For unimanual you'd want `MolmoAct2-SO100_101` or fine-tuning the base `MolmoAct2`.

## Progress log

(Newest first.)

- T+45m — Tasks 1–5, 8, 9 complete. Client + launchers committed (local). HF download ~20% complete. Report at v2.
- T+30m — Wrote `scripts/yam_client.py` (full client loop with safety caps). Wrote launcher shell scripts.
- T+25m — Added `pyrealsense2`, `json-numpy`, `requests` to i2rt venv. Sim YAM smoke test passed.
- T+15m — Started `uv sync` + `hf download MolmoAct2-BimanualYAM` in tmux session `mact2`.
- T+10m — Cloned `allenai/molmoact2`. Discovered `pyproject.toml` not yet in main — wrote minimal one matching the validated stack (torch 2.5.1+cu121, transformers 4.57).
- T+5m — Confirmed RTX 5090 (32 GB), CAN buses up at 1 Mbit/s, RealSense not plugged in.
- T+0  — Workspace + report scaffolded. MolmoAct2 docs read end-to-end.
