# DreamZero exploration — journal

## 2026-05-21 — initial scaffolding (Claude)

**Goal:** mirror the `molmoact2-setup/` structure for DreamZero (WAM, 14B
video-diffusion-backed VLA from NVIDIA GEAR Lab), with both a "vanilla" and a
"YAM-tuned" path on a bimanual YAM rig.

**Key constraints discovered during research:**

1. **DreamZero requires ≥2 GPUs distributed.** `socket_test_optimized_AR.py`
   hard-launches via `torch.distributed.run --nproc_per_node 2`. The 14B BF16
   weights are ~28 GB. Local RTX 5090 (1× 32 GB) cannot run it.
2. **No public YAM-finetuned checkpoint exists.** GEAR-Dreams HF org has
   `DreamZero-DROID` (single-arm Franka, runnable) and `DreamZero-AgiBot` (LoRA
   base, not deployable). The paper's "30 min of YAM play data" result is
   reproducible but the resulting weights were not released.
3. **Protocol is WebSocket + msgpack-numpy** (different from MolmoAct2's HTTP+
   json-numpy). PolicyServerConfig is sent on connect; clients then loop obs
   → action.
4. **YAM modality keys** (from `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`):
   - cameras: `video.{top,left,right}_camera-images-rgb`
   - state: `state.{left,right}_{joint,gripper}_pos` (6, 1, 6, 1)
   - action: same with `action.` prefix
   - prompt: `annotation.task` (not `prompt`)

**Decisions:**

- **Modal-only inference path.** All inference runs on Modal H100:2 (or
  GB200 if user opts in). Exposed via `@modal.web_server(port=8000)` which
  proxies WebSocket upgrades through the public HTTPS gateway.
- **Two flavors via `MODEL` env**: `droid` (public, runnable today) and `yam`
  (placeholder repo id, blocks until a checkpoint exists).
- **Fine-tune launcher prepared but not auto-invoked.** Full 100k-step run
  is ~$200 on H100:4 and requires a properly-shaped YAM LeRobot dataset that
  doesn't exist publicly. User must trigger explicitly.
- **Hardware client defaults to `--dry-run`** — same safety posture as
  `molmoact2-setup/scripts/yam_client.py`.

**Files added:**

```
modal/dreamzero_server.py            # H100:2 WebSocket server
modal/dreamzero_finetune.py          # H100:4 yam_training.sh wrapper
scripts/smoke_test_remote.py         # synthetic-frame WebSocket smoke test
scripts/dreamzero_yam_client.py      # i2rt+RealSense hardware client
scripts/run_modal_server.sh
scripts/run_finetune_modal.sh
pyproject.toml                       # modal + websockets + msgpack-numpy + openpi-client
README.md, HANDOFF.md
```

Workspace + full report at `/home/andon/yam-tests/reports/dreamzero-setup.md`.

**Next session:**

- `modal serve modal/dreamzero_server.py` once to validate cold start. ~15 min
  for image build + 23 GB HF download (cached afterward).
- If user provides a YAM dataset HF id, smoke-run the fine-tune at
  `--max-steps 200` (~$10) before committing to the full 100k.
