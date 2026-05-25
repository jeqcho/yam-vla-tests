# DreamZero on Bimanual YAM — Setup Report

**Working directory:** `/home/andon/yam-tests/dreamzero exploration/`
**Reference setup:** `/home/andon/yam-tests/molmoact2-setup/`
**Target task:** "place all snack packets into the box and close the lid" (one of the per-episode language annotations in `allenai/01122025-box-01`)

## Bottom line

- ✅ **`./scripts/run_inference.sh droid`** — one-command DreamZero-DROID inference cold-start + smoke test on Modal H100:2 via `modal.forward()` Tunnel. URL emerges from the script in ~10 min cold-start, then 3 synthetic rounds confirm `(24, 8)` actions.
- ✅ **Modal Tunnels (`modal.forward(port)`) chosen** per the [Physical Intelligence × Modal blog post](https://modal.com/blog/physical-intelligence-runs-real-time-remote-inference-for-robotic-control-on-modal). Their custom QUIC/UDP hole-punched portal isn't a public API; Tunnels are the right turnkey option, and DreamZero's ~3 s inference makes network latency irrelevant anyway.
- ✅ **YAM data prep job** at `modal/prepare_yam_data.py` handles the Ai2 v3 → DreamZero v2 conversion: splits sharded parquets into per-episode parquets, ffmpeg-cuts shard mp4s into per-episode clips using the per-camera timestamps in `meta/episodes/`, injects `annotation.task` from `meta/tasks_annotated.parquet`, runs the upstream `convert_lerobot_to_gear.py` for stats + relative_stats, and patches `modality.json` to match the DreamZero YAM YAML key names.
- ✅ **Fine-tune launcher** at `modal/dreamzero_finetune.py` chains data-prep → train: mounts the prepared volume, symlinks the AgiBot base into place, runs upstream `yam_training.sh` with bumped per-step batch on H100:4.
- ✅ **Inspection helper** `./scripts/inspect_prep.sh` pulls the prep deliverables (report + modality.json + sample frames + stats) locally for the user to eyeball before greenlighting the costly fine-tune.

## Things to note (data-prep gotchas I had to handle)

1. **Ai2 ships LeRobot v3.0, DreamZero expects v2.** v3 is sharded by file with multiple episodes per parquet/mp4; v2 is one file per episode. The prep job does the full conversion (parquet split + ffmpeg cut).
2. **Per-camera asymmetric video sharding.** `top` has 3 shards, `left`/`right` have 2 — the size cap is per-camera. The splitter uses the per-camera `videos/observation.images.<cam>/{chunk,file}_index` columns from `meta/episodes/chunk-NNN/file-NNN.parquet` to find the right shard per episode, then cuts using `from_timestamp`/`to_timestamp` (precise, no frame-count math).
3. **Task strings are per-episode, not per-task-index.** Ai2 keys them in `meta/tasks_annotated.parquet` by `episode_index` (not by `task_index` as the LeRobot v3 spec implies). The prep job loads them and injects `annotation.task` per row.
4. **Video key name mismatch.** Ai2 uses `observation.images.{top,left,right}`; DreamZero YAM YAML expects `video.{top,left,right}_camera-images-rgb`. The prep job creates the v2 video dirs with the `_camera-images-rgb` suffix and post-edits `meta/modality.json` to match.

## Two-flavor design

| Flavor | Source | Modal app | Status |
|---|---|---|---|
| **"Vanilla"** | `GEAR-Dreams/DreamZero-DROID` | `dreamzero-droid` | Public, runnable today via `./scripts/run_inference.sh droid`. Single-arm Franka schema. Can't drive bimanual YAM physically; verifies the model loads + distributed inference works + the tunnel speaks `wss://`. |
| **"YAM-tuned"** | (will be) `<your-org>/DreamZero-YAM-v1` | `dreamzero-yam` | No public checkpoint exists. Fine-tune via `./scripts/run_finetune_modal.sh yam_box_smoke dz-yam-v1 100000` once prep is greenlit. ~$150–$200 on H100:4 over ~12 hr. |

## How to run the orange-cube task (right command for each stage)

### 0 — Test vanilla inference end-to-end

```bash
cd "/home/andon/yam-tests/dreamzero exploration"
./scripts/run_inference.sh droid
```

Cold start ~10 min. Smoke test runs automatically. Server tears down on exit (pass `--keep` to leave it up).

### 1 — Inspect prepared YAM data

```bash
./scripts/inspect_prep.sh yam_box_smoke
cat hf-cache/prep_yam_box_smoke/yam_box_smoke_prep_report.md
```

The report includes the pre-fine-tune checklist (all `[x]` means green to train), modality.json keys + slice ranges, sample state vectors, sample task strings, and one PNG per camera from `sample_frames/`.

### 2 — Smoke fine-tune (200 steps, ~$10)

```bash
./scripts/run_finetune_modal.sh yam_box_smoke dz-yam-smoke 200
```

This validates: AgiBot+Wan2.1+umt5-xxl downloads, dataloader picks up the v2 layout, transforms apply, the LoRA step fires, and checkpoints write to the `dreamzero-finetune-out` volume. If it survives 200 steps without OOM/dtype errors, the full run will too.

### 3 — Full fine-tune (100k steps, ~$150–$200, ~12 hr)

```bash
./scripts/run_finetune_modal.sh yam_box_smoke dz-yam-v1 100000
```

Outputs to `dreamzero-finetune-out:/dz-yam-v1/`. Pull:

```bash
modal volume get -r dreamzero-finetune-out dz-yam-v1 ./hf-cache/
```

### 4 — Serve the trained checkpoint

```bash
# After uploading hf-cache/dz-yam-v1/ to a HF repo (or wiring repo_id to a Modal volume mount):
YAM_REPO_ID=<your-org>/DreamZero-YAM-v1 ./scripts/run_inference.sh yam
```

### 5 — Drive the real bimanual YAM

```bash
/home/andon/yam-tests/i2rt/.venv/bin/python "/home/andon/yam-tests/dreamzero exploration/scripts/dreamzero_yam_client.py" \
    --url wss://<…>-dreamzero-yam-serve.modal.run/ \
    --left-can can0 --right-can can1 \
    --left-gripper linear_4310 --right-gripper linear_4310 \
    --top-cam-serial X --left-cam-serial Y --right-cam-serial Z \
    --instruction "place all snack packets into the box and close the lid" \
    --train-fps 30 --horizon-stride 6 \
    --max-step-rad 0.05 --gripper-step 0.05 \
    --dry-run
```

`--dry-run` is the default; pass `--no-dry-run` once you've confirmed actions look sane and the workspace is clear.

## Why Modal Tunnels and not the QUIC portal

From the PI blog:

> For latency-sensitive services, the solution is **Modal Tunnels**, which expose live TCP ports on a running Modal container directly to the public internet, with automatic TLS termination and a secure, randomly assigned URL.
>
> [Then] Pi worked with Modal to build a more specialized transport: a QUIC-based portal running over UDP with automatic NAT traversal.

`modal.forward(port)` is the *public* API for Tunnels — exactly what the blog describes as the first step. The QUIC portal (Rust, UDP hole-punching, ~10–15 ms cloud overhead) is described as "a deeper collaboration" between PI and Modal — there's no public toggleable API for it, no library, no env var. For our use case (DreamZero ~3 s per inference), the difference is below the noise floor anyway. Tunnels are the right call.

The wire format that flows through the tunnel is unchanged: msgpack-numpy over WebSocket, talking to dreamzero's `socket_test_optimized_AR.py` (DROID schema) or to a future YAM-tuned variant.

## Wire protocol (WebSocket + msgpack-numpy)

DreamZero is **WebSocket + msgpack-numpy** (vs MolmoAct2's HTTP + json-numpy). Server sends a `PolicyServerConfig` dict on connect; clients then loop send-obs / recv-action.

**DROID schema** (from `eval_utils/policy_server.py:36-56`, served by `socket_test_optimized_AR.py`):

| Field | Shape/Type | Notes |
|---|---|---|
| `observation/exterior_image_0_left`  | `(H,W,3)` uint8 RGB | default 180×320 |
| `observation/exterior_image_1_left`  | `(H,W,3)` uint8 RGB | |
| `observation/wrist_image_left`        | `(H,W,3)` uint8 RGB | |
| `observation/joint_position`          | `(7,)` float32 | |
| `observation/gripper_position`        | `(1,)` float32 | |
| `observation/cartesian_position`      | `(6,)` float32 | optional |
| `prompt`                               | `str` | |
| `session_id`                          | `str` | UUID, used to reset frame buffer |
| `endpoint`                             | `"infer"` or `"reset"` | required |

Response: `(N, 8)` float32 — `[q1..q7, gripper]`, default N=24.

**YAM schema** (when a YAM-finetuned checkpoint exists):

| Field | Shape/Type | Notes |
|---|---|---|
| `video.top_camera-images-rgb`         | `(1,H,W,3)` uint8 | |
| `video.left_camera-images-rgb`        | `(1,H,W,3)` uint8 | |
| `video.right_camera-images-rgb`       | `(1,H,W,3)` uint8 | |
| `state.left_joint_pos`                | `(1,6)` float64 | |
| `state.left_gripper_pos`              | `(1,1)` float64 | `[0,1]` normalized |
| `state.right_joint_pos`               | `(1,6)` float64 | |
| `state.right_gripper_pos`             | `(1,1)` float64 | `[0,1]` normalized |
| `annotation.task`                     | `str` | NB: key name differs from DROID |
| `endpoint`                             | `"infer"` or `"reset"` | required |

Response: dict with `action.{left,right}_{joint,gripper}_pos` per key (chunked at horizon 24).

## YAM modality slicing (what the converter sets)

The Ai2 `observation.state` and `action` columns are flat `float32 (14,)` with the named slots:
`[left_joint_0.pos … left_joint_5.pos, left_gripper.pos, right_joint_0.pos … right_joint_5.pos, right_gripper.pos]`.

So the index slicing the prep job passes to `convert_lerobot_to_gear.py`:

```
--state-keys  '{"left_joint_pos":[0,6],"left_gripper_pos":[6,7],
                "right_joint_pos":[7,13],"right_gripper_pos":[13,14]}'
--action-keys '{"left_joint_pos":[0,6],"left_gripper_pos":[6,7],
                "right_joint_pos":[7,13],"right_gripper_pos":[13,14]}'
--relative-action-keys  left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos
--task-key  annotation.task
```

These slices line up with the `modality_config_yam` block in `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml:267-291` — no YAML edits needed.

## Modal apps and volumes

| App | Purpose | Lifetime |
|---|---|---|
| `dreamzero-droid` | DreamZero-DROID inference server | per `modal serve` invocation |
| `dreamzero-yam`   | DreamZero-YAM inference (needs YAM_REPO_ID set) | per `modal serve` invocation |
| `dreamzero-yam-data-prep` | v3→v2 conversion + GEAR meta | one-shot per `modal run` |
| `dreamzero-yam-finetune` | yam_training.sh launcher | one-shot |

| Volume | Contents |
|---|---|
| `dreamzero-hf-cache` | DROID + YAM model snapshots (persists across deploys) |
| `dreamzero-ckpts` | AgiBot base + Wan2.1-I2V + umt5-xxl for fine-tuning |
| `dreamzero-yam-data` | `raw/<repo>/` Ai2 v3 sources + `prepared/<tag>/` v2 GEAR-formatted output + `<tag>_prep_report.md` |
| `dreamzero-finetune-out` | Fine-tune checkpoints |

Stop an app: `modal app stop <name>`. List: `modal app list`.

## Directory layout

```
/home/andon/yam-tests/
├── dreamzero exploration/                  # workspace
│   ├── dreamzero/                          # clone of dreamzero0/dreamzero (untracked)
│   ├── modal/
│   │   ├── dreamzero_server.py             # H100:2 WebSocket via modal.forward()
│   │   ├── dreamzero_finetune.py           # H100:4 yam_training.sh wrapper
│   │   └── prepare_yam_data.py             # v3→v2 + GEAR meta (Ai2 BimanualYAM)
│   ├── scripts/
│   │   ├── run_inference.sh                # ★ one-command e2e inference test
│   │   ├── run_finetune_modal.sh           # finetune launcher
│   │   ├── inspect_prep.sh                 # pull prep artifacts locally
│   │   ├── smoke_test_remote.py            # synthetic WebSocket smoke test
│   │   └── dreamzero_yam_client.py         # i2rt + RealSense hardware client
│   ├── pyproject.toml                      # local-only deps
│   └── README.md / HANDOFF.md / journal.md
├── molmoact2-setup/                        # reference setup (unchanged)
└── reports/
    ├── molmoact2-setup.md
    └── dreamzero-setup.md                  # ← you are here
```

## Progress log

(Newest first.)

- T+5h — `modal.forward()` Tunnels switch in place; one-command `run_inference.sh` works; data prep handles v3→v2 split with per-camera timestamp cuts; per-episode `annotation.task` injected from Ai2's `tasks_annotated.parquet`.
- T+4h — Hit v3 vs v2 layout mismatch (Ai2 ships v3 sharded; DreamZero expects v2 per-episode). Rewrote splitter to use `meta/episodes/chunk-NNN/file-NNN.parquet` for per-cam video shard mapping + timestamps.
- T+3h — Hit per-camera asymmetric video sharding (top has 3 shards, left/right have 2). Naive file-NNN↔file-NNN assumption broke; switched to per-episode metadata.
- T+2h — Hit per-episode task layout: tasks live in `meta/tasks_annotated.parquet` indexed by `episode_index` (not `task_index`). Switched the loader.
- T+1h — Hit `assert DREAMZERO_DIR.is_dir()` at module-import time inside Modal container. Moved to entrypoint.
- T+0  — Initial scaffold (Modal-based inference + finetune + smoke test + bimanual client).
