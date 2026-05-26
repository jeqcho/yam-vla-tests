# Modal finetune for bimanual YAM GR00T N1.7

`finetune_yam.py` launches a GPU-backed Modal job that finetunes
`nvidia/GR00T-N1.7-3B` on a bimanual YAM dataset (LeRobot v3 format) from
[`allenai/molmoact2-bimanualyam-dataset`](https://huggingface.co/collections/allenai/molmoact2-bimanualyam-dataset).

## One-time setup

1. Get HF access to the gated `nvidia/Cosmos-Reason2-2B` VLM backbone:
   - Visit https://huggingface.co/nvidia/Cosmos-Reason2-2B and click
     "Agree and access repository" (approval is near-instant).
2. Make sure your HF token is wired into a Modal Secret with key `HF_TOKEN`.
   - The `andon-labs` workspace already has `hf-token-jeqcho` (default).
   - If you need to use a different secret name, set `HF_SECRET_NAME` in your
     env before invoking, e.g. `HF_SECRET_NAME=hf-token-dk1 modal run ...`.
   - To make a brand-new secret:
     ```bash
     modal secret create my-hf-secret HF_TOKEN=hf_xxx
     ```

## Launch a finetune

```bash
cd "/home/andon/yam-tests/grootn1.7 exploration"

# Small validation run: ~2k steps on one ~30-min YAM block dataset.
# Expect ~90 min on H100, ~$3-5.
modal run modal/finetune_yam.py \
    --dataset-repo-id allenai/29112025-block-01 \
    --max-steps 2000 \
    --global-batch-size 32 \
    --output-subdir yam-latest
```

## Retrieve the checkpoint

```bash
modal volume get gr00t-yam-checkpoints /yam-latest \
    "/home/andon/yam-tests/grootn1.7 exploration/hf-cache/checkpoints/yam-latest"
```

Then `./scripts/run_server.sh` will auto-detect and load it.

## Scaling to a real run

The single-subset 2k-step run is a smoke test for the pipeline, not a
production-quality YAM policy. For a real finetune:

- Use multiple subsets (mix `block-*` + `box-*` + `charging-*` collections
  from the allenai bimanual YAM collection).
- Bump `--max-steps` to 20k-50k.
- Use 8x H100 (set `gpu="H100:8"` in `finetune_yam.py` and pass
  `--num-gpus 8` to launch_finetune.py). Cost is ~$30/hr.
- Track with W&B (set `WANDB_API_KEY` as a second Modal secret).
