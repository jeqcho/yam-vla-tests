# GR00T N1.7 / Bimanual YAM — research journal

`scripts/yam_client.py` appends a markdown entry here at end-of-run (status,
notes, command, configuration). The first entry below is the scaffold setup;
later entries are auto-generated.

---
## 2026-05-21 11:50:00 -- setup

**Purpose**: Initial scaffold of GR00T N1.7 on bimanual YAM, mirroring
the MolmoAct2 setup pattern. Goal: be one command away from running both
the vanilla base model and an eventual YAM-finetuned checkpoint against
the bimanual YAM hardware.

**What got built**:
- `Isaac-GR00T/` clone of nvidia/Isaac-GR00T@main, with `.venv` (torch
  2.7.1+cu128, transformers 4.57.3, gr00t pkg editable). bf16 matmul on
  the RTX 5090 (sm_120) verified.
- `nvidia/GR00T-N1.7-3B` base model downloaded to `hf-cache/` (6.5 GB).
- YAM modality config (`scripts/yam_config.py`) registers a
  NEW_EMBODIMENT with 14-D state split as [left_arm(6) + left_gripper(1)
  + right_arm(6) + right_gripper(1)], 3 RGB streams, 16-step action
  horizon, RELATIVE arm + ABSOLUTE gripper action types. Verified the
  config loads cleanly and shows up in MODALITY_CONFIGS at runtime.
- Server-side wrapper (`scripts/run_server.sh`) launches NVIDIA's
  `gr00t/eval/run_gr00t_server.py` on port 5556 with our modality config,
  auto-detecting a local finetune at `hf-cache/checkpoints/yam-latest/`
  and falling back to the base model otherwise.
- Vanilla baseline (`scripts/run_server_vanilla.sh`) on port 5555 with
  `--embodiment-tag XDOF` for A/B comparison.
- Client (`scripts/yam_client.py`) — talks the GR00T msgpack-numpy + zmq
  wire protocol via a slim in-script PolicyClient (avoids importing the
  full gr00t package into the i2rt venv). 14-D YAM state -> nested GR00T
  observation, action chunk decoded back to 14-D, per-tick safety
  clipping (max_step_rad=0.15 by default), same arm-init order as the
  MolmoAct2 client (cameras before arms to dodge the USB
  enumeration storm), same SDK lock fix patch, same journal prompt.
- `scripts/smoke_test_server.py` for synthetic-frame end-to-end check.
- `scripts/preflight.py`, `list_cams.py`, `capture_frames.py` for
  hardware bring-up.
- Modal training script (`modal/finetune_yam.py`) — single-GPU H100,
  pulls one of the `allenai/*-block-*` or `allenai/*-box-*` YAM lerobot
  v3 datasets (~30-60 min of data each, 14-D state/action matches our
  modality exactly), drops yam_modality.json into the dataset's meta/,
  runs launch_finetune.py with --embodiment-tag NEW_EMBODIMENT, persists
  the checkpoint into a Modal Volume. Uses the existing
  `hf-token-jeqcho` secret in the andon-labs workspace.

**What did NOT get done**:
- Live smoke test of the GR00T server. Loading the processor requires
  agreeing to the gated `nvidia/Cosmos-Reason2-2B` license on HF, which
  needs the user's interactive consent. The error surfaces as a clear
  `GatedRepoError: 401` on first server start; HANDOFF.md step 0 walks
  through the fix.
- The Modal finetune itself — script is ready, the user must run it.

**Key constraints found**:
- The base GR00T-N1.7-3B does NOT ship a YAM embodiment tag — only
  DROID/LIBERO/SimplerEnv-{Bridge,Fractal} are publicly finetuned. NVIDIA
  validated YAM in their blog post but didn't release that checkpoint.
- The base model's processor pulls `nvidia/Cosmos-Reason2-2B`
  preprocessor config at load time. That repo is gated; once-per-user
  click-through agreement is required.
- N1.7 uses RELATIVE EEF / relative-joint actions across embodiments.
  Our YAM config follows this — left/right arm joints are RELATIVE
  deltas from current state, grippers are ABSOLUTE.
- The 720 hours of AllenAI bimanual YAM data on HF
  (`allenai/molmoact2-bimanualyam-dataset` collection) is the obvious
  finetuning data — each subset is a 14-D state/action lerobot v3
  dataset with 3 cameras (top/left/right), matching our modality
  config exactly modulo the camera-key naming
  (we map `left_wrist` -> `observation.images.left`).

**Open questions for whoever picks this up**:
- Run the Modal finetune end-to-end on at least one block subset
  (allenai/29112025-block-01 or similar — 54 episodes, ~76k frames)
  and confirm the resulting checkpoint loads in run_server.sh and
  produces sensible actions in the smoke test.
- Decide how aggressive to be with multi-subset training. The collection
  has ~80 subsets totaling ~720 hours; a real YAM-tuned model probably
  wants 20-30k steps across a mix of block + box + charging subsets.
- Side-by-side eval vs the MolmoAct2-BimanualYAM model on the same
  hardware would be the obvious downstream comparison once both
  finetunes are in hand.

**Status**: scaffold complete, end-to-end blocked on HF auth.
