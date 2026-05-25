# Handoff: GR00T N1.7 on Bimanual YAM

Long-form context lives at `reports/grootn1.7-setup.md` (still to write). This
file is the bring-up checklist.

## Step 0 — one-time HF access (DO THIS FIRST)

The GR00T N1.7 processor pulls preprocessing config from the
**gated** repo `nvidia/Cosmos-Reason2-2B`. Without access, every server start
will die with `GatedRepoError: 401`. Fix:

1. Visit https://huggingface.co/nvidia/Cosmos-Reason2-2B and click
   "Agree and access repository" (approval is near-instant — no human review).
2. Authenticate locally:
   ```bash
   hf auth login
   # paste a read token from https://huggingface.co/settings/tokens
   ```
3. Verify with the helper:
   ```bash
   cd "/home/andon/yam-tests/grootn1.7 exploration"
   "Isaac-GR00T/.venv/bin/python" scripts/check_hf_access.py
   ```
   Should print three OK lines. If any fails, the helper tells you what to do.
4. For Modal finetuning, the secret `hf-token-jeqcho` in the `andon-labs`
   workspace already exists. Just confirm it has key `HF_TOKEN` set.

## What's in the box

- `Isaac-GR00T/` — clone of `nvidia/Isaac-GR00T` main. Has its own `.venv` with
  torch 2.7.1+cu128, transformers 4.57.3, the gr00t package installed editable.
- `scripts/` — YAM-bimanual glue:
  - `yam_client.py` — talks to the GR00T policy server over ZeroMQ + msgpack.
    Reads 14-D YAM joint state, posts 3 RGB streams + language, gets back a
    16-step action chunk, applies it with per-tick safety clipping.
    Uses the i2rt venv at `/home/andon/yam-tests/i2rt/.venv` (already has
    pyrealsense2 + i2rt SDK; we added pyzmq + msgpack-numpy).
  - `run_server.sh` — starts the GR00T server on port 5556 with the YAM
    modality config. Auto-detects a local finetune at
    `hf-cache/checkpoints/yam-latest`; falls back to `nvidia/GR00T-N1.7-3B`
    (base model — useful for plumbing, but no YAM-specific weights).
  - `run_server_vanilla.sh` — starts the vanilla base model with the closest
    pretrain embodiment tag (`XDOF`) on port 5555. Useful as an A/B baseline
    against a YAM finetune. Caveat: XDOF was not trained on YAM 14-D bimanual
    layout, so the action numbers won't actually drive the arms; this is for
    "does the plumbing work end-to-end" testing only.
  - `run_client.sh` — wraps yam_client.py with sensible defaults.
  - `yam_config.py` — registers the YAM bimanual modality (NEW_EMBODIMENT
    tag, 4 modality keys: left_arm/left_gripper/right_arm/right_gripper, 16-step
    action horizon, 3 cameras).
  - `yam_modality.json` — meta/modality.json template for YAM lerobot
    datasets (state/action splits + video key mapping).
  - `smoke_test_server.py` — posts 3 rounds of synthetic frames and checks
    the action shape.
  - `preflight.py`, `list_cams.py`, `capture_frames.py` — hardware checks.
- `hf-cache/` — local HF cache (currently has `nvidia/GR00T-N1.7-3B`, ~6.5 GB).
  Created by `run_server.sh`. After finetuning, place YAM checkpoints under
  `hf-cache/checkpoints/yam-latest/`.
- `modal/finetune_yam.py` — single-GPU Modal job that finetunes the base
  model on one bimanual YAM lerobot dataset from
  `allenai/molmoact2-bimanualyam-dataset`. See `modal/README.md`.

## Bring the system online (both arms + 3 cameras plugged in)

```bash
cd "/home/andon/yam-tests/grootn1.7 exploration"

# 0a. Confirm cameras enumerate (need 3 RealSense devices)
/home/andon/yam-tests/i2rt/.venv/bin/python scripts/list_cams.py

# 0b. Pre-flight: cameras + CAN buses + arms init.
#     WARNING: arm init runs gripper auto-calibration — clear the jaws.
/home/andon/yam-tests/i2rt/.venv/bin/python scripts/preflight.py \
    --left-can can0 --right-can can1 \
    --left-gripper linear_4310 --right-gripper linear_4310 \
    --skip-server     # server isn't up yet

# 1. Start the inference server (Terminal A). Two options:
./scripts/run_server.sh
#    Uses the YAM finetune if it exists at hf-cache/checkpoints/yam-latest,
#    else falls back to the base model with NEW_EMBODIMENT (untrained).
#    Listens on tcp://0.0.0.0:5556. Wait for "Server is ready and listening".

# OR for a vanilla A/B baseline:
./scripts/run_server_vanilla.sh
#    Same base model but with --embodiment-tag XDOF (one of the pretrain
#    tags). Listens on :5555.

# 2. Smoke-test the server with synthetic frames (Terminal B)
"/home/andon/yam-tests/grootn1.7 exploration/Isaac-GR00T/.venv/bin/python" \
    scripts/smoke_test_server.py --port 5556
# Expect 3 rounds, each printing horizon=16 and 4 modality keys.

# 3. Eyeball a dry-run on the real arms
./scripts/run_client.sh \
    --left-can can0 --right-can can1 \
    --left-gripper linear_4310 --right-gripper linear_4310 \
    --top-cam-serial   <T> \
    --left-cam-serial  <L> \
    --right-cam-serial <R> \
    --server-port 5556 \
    --dry-run

# 4. If actions look sane (deltas << 1 rad from current state), drop --dry-run.
```

## Vanilla vs. finetuned — what to expect

|                          | Vanilla `nvidia/GR00T-N1.7-3B`                                                                          | YAM-finetuned                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Embodiment tag           | `XDOF` (closest pretrain tag — `--run_server_vanilla.sh`)                                              | `NEW_EMBODIMENT` (matches `scripts/yam_config.py`)                                                                         |
| State/action layout      | XDOF's generic layout — **not** YAM 14-D, so actions don't line up with arm joints                      | YAM 14-D `[left_q0..5, left_grip, right_q0..5, right_grip]` matches the i2rt SDK directly                                  |
| What it can do           | Verify the plumbing — frames reach the model, the model returns numbers — but those numbers are junk on YAM | Reach for / pick up / place orange cubes IF the finetune dataset covered the task. Quality depends on how much data + steps |
| Checkpoint location      | Downloaded automatically on first server start (`hf-cache/hub/...`)                                     | `hf-cache/checkpoints/yam-latest/` (drop in after `modal volume get`)                                                       |

A YAM checkpoint **doesn't exist publicly yet** — NVIDIA validated GR00T N1.7
on YAM in their blog post but didn't ship a finetuned checkpoint (only DROID /
LIBERO / SimplerEnv-Bridge / SimplerEnv-Fractal). You'll need to run the Modal
finetune (`modal/README.md`) to produce one.

## How the wire format differs from MolmoAct2

| Concern                | MolmoAct2 server (`molmoact2-setup/`)          | GR00T N1.7 server (here)                                                       |
| ---------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------ |
| Transport              | HTTP POST + `json_numpy`, port 8202            | ZeroMQ REQ/REP + `msgpack-numpy`, port 5556                                    |
| Observation layout     | Flat `top_cam`/`left_cam`/`right_cam`/`state`  | Nested `video`/`state`/`language` with explicit (B=1, T=1) dims up front      |
| State keys             | One `(14,)` array                              | Four arrays: `left_arm(6)`, `left_gripper(1)`, `right_arm(6)`, `right_gripper(1)` |
| Action horizon         | 30 × 14                                        | 16 × per-key (4 keys → reconstructed 14-D)                                    |
| Action representation  | absolute joint positions                       | left/right arm RELATIVE deltas from current state; grippers ABSOLUTE          |

Per-tick safety clipping in `yam_client.py` is identical to the MolmoAct2
client. Move-to-ready is a no-op for GR00T (no fixed training-mean pose).

## Currently NOT done (because I can't)

- ❌ **End-to-end smoke test on the actual model** — blocked on HF auth for
  `nvidia/Cosmos-Reason2-2B`. The modality config registers cleanly and
  imports work, but a real `Gr00tPolicy(...)` instantiation will fail until
  Step 0 above is done. Run `scripts/smoke_test_server.py` after the user
  logs in.
- ❌ **Finetune actually launched on Modal.** Script is written and CLI parses;
  the user must `modal run modal/finetune_yam.py ...` (will cost ~$5 for a
  small validation run).
- ❌ **Hardware-side identification of which CAN bus is left vs right.** Same
  unknowns as the MolmoAct2 handoff — `scripts/identify_arms.py` doesn't
  exist here yet; either copy from `molmoact2-setup/scripts/identify_arms.py`
  or just use the dry-run + wiggle method.
- ❌ **Push to a remote.** No `gh auth`. Local-only.

## If the server crashes at startup

- `GatedRepoError: 401 ... Cosmos-Reason2-2B` → Step 0 not done. Either
  `hf auth login` or set `HF_TOKEN` in the environment.
- `Embodiment tag 'NEW_EMBODIMENT' is not supported by this checkpoint` →
  you forgot `--modality-config-path scripts/yam_config.py`. The base model
  doesn't bake in NEW_EMBODIMENT; the config has to be loaded as a side
  effect of importing yam_config.py.
- `CUDA error: no kernel image is available` → torch wheel is too old for
  Blackwell sm_120. The Isaac-GR00T `pyproject.toml` already pins
  cu128 wheels via `[tool.uv.sources]`, so this shouldn't happen — but if it
  does, redo `uv sync` from `Isaac-GR00T/`.

## File tree (new files only)

```
grootn1.7 exploration/
├── HANDOFF.md                       # this file
├── journal.md
├── yam_setup_config.json            # site-local hardware defaults
├── Isaac-GR00T/                     # nvidia/Isaac-GR00T clone (with .venv)
├── hf-cache/                        # local HF cache (base model lives here)
├── modal/
│   ├── finetune_yam.py              # Modal: finetune on a YAM lerobot dataset
│   └── README.md
└── scripts/
    ├── run_server.sh                # starts GR00T policy server :5556
    ├── run_server_vanilla.sh        # starts vanilla base model :5555 (XDOF tag)
    ├── run_client.sh                # wraps yam_client.py
    ├── yam_client.py                # bimanual YAM client over zmq
    ├── yam_config.py                # NEW_EMBODIMENT YAM modality registration
    ├── yam_modality.json            # meta/modality.json for finetuning datasets
    ├── smoke_test_server.py         # synthetic-frame end-to-end check
    ├── preflight.py                 # cams + CAN + arms + server health
    ├── list_cams.py                 # enumerate RealSense devices
    └── capture_frames.py            # snap one frame per camera to PNG
```
