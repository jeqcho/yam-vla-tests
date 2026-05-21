# DreamZero on bimanual YAM — setup

This folder lets you bring up [DreamZero](https://github.com/dreamzero0/dreamzero)
on **remote H100s via Modal** and drive a local bimanual YAM rig against it.
Inference is not local because DreamZero is a 14B WAM that needs ≥2 GPUs
distributed; the YAM workstation has a single RTX 5090.

Two flavors:

| Flavor | Checkpoint | State today |
|---|---|---|
| **Vanilla** | [`GEAR-Dreams/DreamZero-DROID`](https://huggingface.co/GEAR-Dreams/DreamZero-DROID) | ✅ public; runnable today via Modal. Single-arm Franka schema — won't drive a bimanual YAM physically, but you can verify end-to-end inference, latency, and action stats. |
| **YAM-finetuned** | `GEAR-Dreams/DreamZero-YAM-bimanual` (placeholder) | ❌ no public checkpoint. The paper's "30 minutes of YAM play data" result is reproducible via [`scripts/train/yam_training.sh`](dreamzero/scripts/train/yam_training.sh); we ship a Modal launcher for that fine-tune. |

## One-command smoke test (vanilla DROID)

```bash
# Terminal A — bring up the Modal server (prints a wss:// URL in the banner)
./scripts/run_modal_server.sh droid

# Terminal B — synthetic-frame round-trip against that URL
uv sync
uv run python scripts/smoke_test_remote.py \
    --url wss://<your-workspace>--dreamzero-droid-serve.modal.run \
    --schema droid --rounds 5
```

Expected output: per-round `action shape=(N, 8)` with finite values and an RTT
on the order of 3 s on H100 (per the model card).

## Drive a real bimanual YAM (requires a YAM-finetuned checkpoint)

Once you have a DreamZero checkpoint finetuned on bimanual YAM:

```bash
# 1. Edit modal/dreamzero_server.py:CHECKPOINTS["yam"] to point at it,
#    or pass it via env: YAM_REPO_ID=org/your-checkpoint
./scripts/run_modal_server.sh yam

# 2. From the YAM workstation (with the i2rt venv as interpreter):
/home/andon/yam-tests/i2rt/.venv/bin/python scripts/dreamzero_yam_client.py \
    --url wss://<…>-dreamzero-yam-serve.modal.run \
    --left-can can0 --right-can can1 \
    --left-gripper linear_4310 --right-gripper linear_4310 \
    --top-cam-serial <S1> --left-cam-serial <S2> --right-cam-serial <S3> \
    --instruction "pick up the orange cube on the left and put it in the box" \
    --dry-run         # mandatory until you've inspected the actions
```

## Fine-tune YAM-bimanual from DreamZero-AgiBot

```bash
# Stage data + checkpoints + run yam_training.sh on 4×H100. Budget ~$150 for
# a full 100k-step run; pass a small --max-steps for a smoke run first.
./scripts/run_finetune_modal.sh <hf-dataset-id> dreamzero_yam_run1 200
```

See `REPORT_dreamzero_setup.md` for the modality schema the dataset must
expose and how to crosswalk Ai2's `MolmoAct2-BimanualYAM` data into it.

## Layout

```
dreamzero exploration/
├── dreamzero/                # cloned dreamzero0/dreamzero (untracked)
├── modal/
│   ├── dreamzero_server.py   # WebSocket policy server on Modal H100s
│   └── dreamzero_finetune.py # yam_training.sh wrapper on Modal H100s
├── scripts/
│   ├── run_modal_server.sh
│   ├── run_finetune_modal.sh
│   ├── smoke_test_remote.py
│   └── dreamzero_yam_client.py
├── pyproject.toml            # local venv: msgpack/websockets/modal/i2rt-compat
├── .python-version           # 3.11
└── README.md, HANDOFF.md
```
