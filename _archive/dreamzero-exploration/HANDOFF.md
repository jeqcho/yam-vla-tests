# DreamZero exploration — HANDOFF

**Working dir:** `/home/andon/yam-tests/dreamzero exploration/`
**Sibling reference:** `molmoact2-setup/`
**Inference path:** Modal Tunnels (`modal.forward()`) per the [Physical Intelligence × Modal blog post](https://modal.com/blog/physical-intelligence-runs-real-time-remote-inference-for-robotic-control-on-modal). The QUIC-portal that PI co-built isn't a public API; Tunnels are.
**Cost so far:** $0 (Modal containers haven't held GPUs for any meaningful wallclock; the prep job is CPU-only and ~$0.20).

## ONE COMMAND to test inference

```bash
cd "/home/andon/yam-tests/dreamzero exploration"
./scripts/run_inference.sh droid
```

This:
1. Brings up `GEAR-Dreams/DreamZero-DROID` on Modal H100:2 (~10 min cold start; ~$10/hr).
2. Polls the log for the `wss://` URL printed by `modal.forward(8000)`.
3. Runs `scripts/smoke_test_remote.py` against it (3 rounds of synthetic frames).
4. Prints ✅ if the model returned a finite `(N, 8)` action tensor.
5. Tears the server back down on exit (pass `--keep` to leave it up).

That's the "vanilla DreamZero" half of the user brief, runnable end-to-end.

## YAM fine-tune (in progress as of HANDOFF)

Data prep is running on Modal right now:

```bash
tail -F logs/prep_smoke.log
```

It's converting `allenai/01122025-box-01` (45 episodes, 105K frames, ~58 min of
bimanual teleop) from **LeRobot v3 sharded** (Ai2's layout) to **LeRobot v2
per-episode** (DreamZero's expected layout). The prep job:
- splits 7 shard parquets → 45 per-episode parquets (fast)
- ffmpeg-cuts 21 shard mp4s (3 cams × 7 shards) → 135 per-episode mp4s (slow,
  ~30–45 min on a Modal CPU container)
- injects `annotation.task` per row from `meta/tasks_annotated.parquet`
  (which Ai2 keys by `episode_index`)
- runs upstream `scripts/data/convert_lerobot_to_gear.py` to compute stats
  and emit `modality.json` / `embodiment.json` / `relative_stats_dreamzero.json`
- writes a deliverable report at `prepared/<tag>_prep_report.md`

When it finishes (notify via `modal app list`), pull the deliverables:

```bash
./scripts/inspect_prep.sh yam_box_smoke
```

That dumps `hf-cache/prep_yam_box_smoke/{prep_report.md, modality.json,
embodiment.json, stats.json, sample_frames/}` for visual inspection.

### What to eyeball in the report

```
cat hf-cache/prep_yam_box_smoke/yam_box_smoke_prep_report.md
```

Specifically:
- **Episodes count > 0** (paranoia after the v3→v2 bug — Stage 2 should report 45 eps total)
- **First-5-episode lengths** — should be in 1500–4000 range (50–130 s clips at 30 fps)
- **Sample task strings** — should look like real instructions, e.g.
  `'Place all snack packets into the box and close the lid.'`
- **`modality.state`** has all 4 keys `{left,right}_{joint,gripper}_pos`
  with correct index slices: `[0,6] [6,7] [7,13] [13,14]`
- **`modality.video`** keys end in `_camera-images-rgb` (so the YAM YAML resolves)
- **Pre-fine-tune checklist** all `[x]`
- **Sample frames** under `sample_frames/observation.images.{top,left,right}_camera-images-rgb_first_frame.png` — verify the cube/box scene is recognizable

### Kicking off the fine-tune

If the prep report looks clean:

```bash
# Smoke (~$10, validates training pipeline e2e, no useful weights):
./scripts/run_finetune_modal.sh yam_box_smoke dz-yam-smoke 200

# Full (~$150–$200 on H100:4 over ~12 hr; matches the paper):
./scripts/run_finetune_modal.sh yam_box_smoke dz-yam-v1 100000
```

Outputs land on the `dreamzero-finetune-out` Modal volume. Pull with
`modal volume get -r dreamzero-finetune-out <run-name> ./hf-cache/`.

To then serve the trained checkpoint:
1. Upload `./hf-cache/<run-name>/` to a private HF repo, say `your/DreamZero-YAM-v1`.
2. `YAM_REPO_ID=your/DreamZero-YAM-v1 ./scripts/run_inference.sh yam`.
3. Then drive a real bimanual YAM:
   ```
   /home/andon/yam-tests/i2rt/.venv/bin/python scripts/dreamzero_yam_client.py \
       --url <wss-from-step-2> \
       --left-can can0 --right-can can1 \
       --left-gripper linear_4310 --right-gripper linear_4310 \
       --top-cam-serial X --left-cam-serial Y --right-cam-serial Z \
       --instruction "place all snack packets into the box and close the lid" \
       --dry-run
   ```

## Why Tunnels not @modal.web_server

Per the [Physical Intelligence × Modal blog](https://modal.com/blog/physical-intelligence-runs-real-time-remote-inference-for-robotic-control-on-modal):

> Modal Tunnels expose live TCP ports on a running Modal container directly to the public internet, with automatic TLS termination and a secure, randomly assigned URL.

`modal.forward(8000)` is the public-API path PI used as their first hop. Their further QUIC/UDP/NAT-hole-punched portal (~10–15 ms cloud overhead) is described as a deeper co-build with Modal and isn't a turnkey feature anyone can adopt today.

For DreamZero, per-inference is ~3 s on H100 — network latency is far in the noise. Tunnels are the right call: no HTTP routing layer, just `wss://` straight to the WebSocket server.

## Modal apps and volumes

| App | Purpose | Lifetime |
|---|---|---|
| `dreamzero-droid` | DreamZero-DROID inference server | per `modal serve` invocation; ephemeral |
| `dreamzero-yam` | DreamZero-YAM inference (needs YAM_REPO_ID set) | per `modal serve` invocation |
| `dreamzero-yam-data-prep` | v3→v2 dataset conversion + GEAR meta | one-shot per `modal run` |
| `dreamzero-yam-finetune` | yam_training.sh launcher | one-shot |

| Volume | What lives there |
|---|---|
| `dreamzero-hf-cache` | DROID + YAM model snapshots (persists across deploys) |
| `dreamzero-ckpts` | AgiBot base + Wan2.1-I2V + umt5-xxl for fine-tuning |
| `dreamzero-yam-data` | `raw/<repo>/` Ai2 v3 sources + `prepared/<tag>/` v2 GEAR-formatted output + `<tag>_prep_report.md` |
| `dreamzero-finetune-out` | Fine-tune checkpoints |

Stop any deployed app: `modal app stop <name>`. List: `modal app list`.

## Outstanding decisions for the user

1. **Greenlight the full 100k fine-tune.** Will spend ~$150-$200 on H100:4 over ~12 hr. The 200-step smoke run is queued/coming so you can see the pipeline turns over before that commit.
2. **Single dataset vs multi-dataset stitch.** Right now we're using `allenai/01122025-box-01` (~58 min teleop, 45 eps). Could stitch in more "box" subsets (`01122025-box-02`, `02122025-box-01`, etc.) for a bigger pretrain. The prep job supports `--hf-repos a,b,c` already.
3. **YAM camera ordering.** Ai2's videos are named `right/left/top` — I've already aligned the modality.json keys to `top_camera-images-rgb/left_camera-images-rgb/right_camera-images-rgb` (the order DreamZero YAML expects). Verify on first inference that "left" and "right" really do correspond to operator POV.
4. **Dispose of Modal apps when done** — `modal app stop dreamzero-droid` etc. — to avoid idle charges if you forget to Ctrl-C the local supervisor.
