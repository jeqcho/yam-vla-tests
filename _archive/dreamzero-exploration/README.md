# DreamZero on bimanual YAM — setup

Bring up [DreamZero](https://github.com/dreamzero0/dreamzero) on Modal H100s
and drive a local bimanual YAM against it. Inference is remote because the 14B
WAM is multi-GPU; the YAM workstation has a single RTX 5090.

Transport: **`modal.forward()` Tunnels** ([per the Physical Intelligence / Modal
blog post](https://modal.com/blog/physical-intelligence-runs-real-time-remote-inference-for-robotic-control-on-modal)).
PI's custom QUIC portal isn't a public Modal feature; Tunnels are the next-best
public path.

## One command to test inference

```bash
./scripts/run_inference.sh droid
```

That:
1. Cold-starts `GEAR-Dreams/DreamZero-DROID` on Modal H100:2 (~10 min first time).
2. Waits for the `modal.forward()` tunnel URL.
3. Hits it with 3 synthetic-frame inference rounds via `scripts/smoke_test_remote.py`.
4. Prints ✅ and tears down (pass `--keep` to leave the server up for further pokes).

## Two flavors

| Flavor | Checkpoint | Status |
|---|---|---|
| **Vanilla** | [`GEAR-Dreams/DreamZero-DROID`](https://huggingface.co/GEAR-Dreams/DreamZero-DROID) | Public, runnable today. Single-arm Franka schema. Probe-only on a bimanual YAM. |
| **YAM-finetuned** | (will be) `<your-org>/DreamZero-YAM-v1` | No public checkpoint exists. Fine-tune via `./scripts/run_finetune_modal.sh`. |

## YAM fine-tune workflow

```bash
# 1. Prep an Ai2 BimanualYAM subset (v3 → v2 layout + GEAR meta).  ~30–45 min, ~$0.20.
modal run modal/prepare_yam_data.py::prepare --hf-repo allenai/01122025-box-01 --tag yam_box_smoke

# 2. Pull the deliverables and eyeball them.
./scripts/inspect_prep.sh yam_box_smoke
cat hf-cache/prep_yam_box_smoke/yam_box_smoke_prep_report.md

# 3. Smoke fine-tune to validate the training pipeline (200 steps, ~$10).
./scripts/run_finetune_modal.sh yam_box_smoke dz-yam-smoke 200

# 4. Full fine-tune (100k steps, ~$150-$200, ~12 hr on H100:4).
./scripts/run_finetune_modal.sh yam_box_smoke dz-yam-v1 100000

# 5. Pull the checkpoint and upload to HF.
modal volume get -r dreamzero-finetune-out dz-yam-v1 ./hf-cache/

# 6. Serve it.
YAM_REPO_ID=<your-org>/DreamZero-YAM-v1 ./scripts/run_inference.sh yam
```

## Drive a real bimanual YAM (after a YAM checkpoint exists)

From the YAM workstation:

```bash
/home/andon/yam-tests/i2rt/.venv/bin/python scripts/dreamzero_yam_client.py \
    --url wss://<...>-dreamzero-yam-serve.modal.run/ \
    --left-can can0 --right-can can1 \
    --left-gripper linear_4310 --right-gripper linear_4310 \
    --top-cam-serial X --left-cam-serial Y --right-cam-serial Z \
    --instruction "place all snack packets into the box and close the lid" \
    --train-fps 30 --horizon-stride 6 \
    --max-step-rad 0.05 --gripper-step 0.05 \
    --dry-run
```

Defaults to `--dry-run`; pass `--no-dry-run` once the action chunks look sane.

## Layout

```
dreamzero exploration/
├── dreamzero/                       # cloned upstream (untracked)
├── modal/
│   ├── dreamzero_server.py          # H100:2 WebSocket server via modal.forward()
│   ├── dreamzero_finetune.py        # H100:4 yam_training.sh wrapper
│   └── prepare_yam_data.py          # v3→v2 + GEAR-meta data prep job
├── scripts/
│   ├── run_inference.sh             # ★ one-command inference test
│   ├── run_finetune_modal.sh        # fine-tune launcher
│   ├── inspect_prep.sh              # pull prep artifacts down for inspection
│   ├── smoke_test_remote.py         # synthetic-frame WebSocket smoke test
│   └── dreamzero_yam_client.py      # i2rt+RealSense hardware client
├── pyproject.toml                   # local-only deps (modal, websockets, msgpack)
├── README.md, HANDOFF.md
```
