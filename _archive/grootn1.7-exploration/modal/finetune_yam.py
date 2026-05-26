"""Modal job: finetune GR00T N1.7-3B on a bimanual YAM lerobot dataset.

Best-practice setup:
    - Train + val datasets pulled separately (different recording days from the
      AllenAI bimanual YAM collection). Val never enters the trainer; we use it
      ONLY for post-training open-loop MSE to pick the best checkpoint.
    - Checkpoints saved at --save-steps intervals; --save-total-limit caps
      disk use. After training we score every saved checkpoint by mean MSE on
      a small held-out trajectory set and symlink the winner to /final/.
    - WANDB streams loss / lr / step throughout (project: finetune-gr00t-yam).
    - Persistent Modal Volumes for both the HF cache (so re-runs skip the
      6 GB base model download) and checkpoint output.

Usage:

    cd "/home/andon/yam-tests/grootn1.7 exploration"
    modal run modal/finetune_yam.py \
        --train-dataset-repo-id allenai/29112025-block-01 \
        --val-dataset-repo-id   allenai/29112025-block-02 \
        --max-steps 4000 \
        --global-batch-size 32

Pull the best checkpoint locally afterwards:

    modal volume get gr00t-yam-checkpoints /yam-latest/best \
        "hf-cache/checkpoints/yam-latest"

Then `./scripts/run_server.sh` will auto-detect it.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "gr00t-yam-finetune"
VOLUME_NAME = "gr00t-yam-checkpoints"
CACHE_VOLUME = "gr00t-yam-cache"

GR00T_REPO = "https://github.com/NVIDIA/Isaac-GR00T.git"
GR00T_BRANCH = "main"

YAM_CONFIG_LOCAL = Path(__file__).parent.parent / "scripts" / "yam_config.py"
YAM_MODALITY_LOCAL = Path(__file__).parent.parent / "scripts" / "yam_modality.json"


# ---------------------------------------------------------------------------
# Image build
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "git-lfs",
        "ffmpeg",
        "libaio-dev",
        "build-essential",
        "curl",
    )
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -s /root/.local/bin/uv /usr/local/bin/uv",
    )
    .run_commands(
        f"git lfs install && "
        f"git clone --recurse-submodules --branch {GR00T_BRANCH} {GR00T_REPO} /opt/Isaac-GR00T",
        "cd /opt/Isaac-GR00T && uv sync",
        "cd /opt/Isaac-GR00T && uv pip install -e .",
        "cd /opt/Isaac-GR00T && uv pip install hf-transfer",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
        "WANDB_DISABLE_GIT": "true",
    })
    .add_local_file(str(YAM_CONFIG_LOCAL),    "/opt/yam_config.py",     copy=True)
    .add_local_file(str(YAM_MODALITY_LOCAL),  "/opt/yam_modality.json", copy=True)
)

app = modal.App(APP_NAME, image=image)

checkpoints_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
cache_vol       = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100",
    timeout=60 * 60 * 8,          # 8 hours (plenty for 4-8k steps + post-eval)
    volumes={
        "/checkpoints": checkpoints_vol,
        "/root/.cache/huggingface": cache_vol,
    },
    secrets=[
        # `hf-token-jeqcho` already lives in the andon-labs workspace. It
        # exposes either HF_TOKEN or HF_API_TOKEN depending on how it was
        # populated; we read both below.
        modal.Secret.from_name("hf-token-jeqcho"),
        # `wandb` (eliasaronson, shared) is the existing workspace-wide secret.
        # If you want per-user W&B, switch to `wandb-dk1` or create your own.
        modal.Secret.from_name("wandb"),
    ],
)
def finetune(
    train_dataset_repo_id: str = "allenai/29112025-block-01",
    val_dataset_repo_id: str   = "allenai/29112025-block-02",
    base_model_path: str       = "nvidia/GR00T-N1.7-3B",
    max_steps: int             = 4000,
    save_steps: int            = 500,
    save_total_limit: int      = 8,
    global_batch_size: int     = 32,
    dataloader_num_workers: int = 4,
    learning_rate: float       = 1e-4,
    output_subdir: str         = "yam-latest",
    wandb_project: str         = "finetune-gr00t-yam",
    val_traj_ids: tuple[int, ...] = (0, 1, 2, 3),
    val_steps_per_traj: int    = 120,
    val_action_horizon: int    = 16,
) -> dict:
    """Single-GPU finetune + post-training MSE-based best-checkpoint pick."""
    import json
    import shutil
    from huggingface_hub import snapshot_download

    # ---- HF + W&B auth ----
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF token missing in Modal secret. Expected HF_TOKEN or HF_API_TOKEN."
        )
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    if not os.environ.get("WANDB_API_KEY"):
        print("[finetune] WARN: WANDB_API_KEY not in env; --use-wandb will fail open")

    # ---- 1. Download train + val datasets ----
    def _prep_dataset(repo_id: str) -> str:
        local_dir = f"/root/dataset/{repo_id.replace('/', '__')}"
        print(f"[finetune] Downloading {repo_id} -> {local_dir}")
        path = snapshot_download(
            repo_id=repo_id, repo_type="dataset",
            local_dir=local_dir, token=hf_token,
        )
        # Drop our meta/modality.json so the gr00t loader knows the YAM schema.
        meta = Path(path) / "meta"
        meta.mkdir(exist_ok=True)
        shutil.copy("/opt/yam_modality.json", meta / "modality.json")
        print(f"[finetune]   wrote {meta/'modality.json'}")
        return path

    train_path = _prep_dataset(train_dataset_repo_id)
    val_path   = _prep_dataset(val_dataset_repo_id)

    # ---- 2. Pre-download base model so the trainer doesn't time out ----
    print(f"[finetune] Pre-downloading {base_model_path}")
    snapshot_download(repo_id=base_model_path, token=hf_token)

    # ---- 3. Launch finetuner ----
    output_dir = f"/checkpoints/{output_subdir}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run_name = f"yam_{Path(train_path).name}_{max_steps}steps"
    cmd = [
        "uv", "run", "python",
        "/opt/Isaac-GR00T/gr00t/experiment/launch_finetune.py",
        "--base-model-path",         base_model_path,
        "--dataset-path",            train_path,
        "--modality-config-path",    "/opt/yam_config.py",
        "--embodiment-tag",          "NEW_EMBODIMENT",
        "--num-gpus",                "1",
        "--output-dir",              output_dir,
        "--max-steps",               str(max_steps),
        "--save-steps",              str(save_steps),
        "--save-total-limit",        str(save_total_limit),
        "--global-batch-size",       str(global_batch_size),
        "--dataloader-num-workers",  str(dataloader_num_workers),
        "--learning-rate",           str(learning_rate),
        "--use-wandb",
        "--wandb-project",           wandb_project,
        "--experiment-name",         run_name,
    ]
    print("[finetune] launch_finetune.py:")
    print("           " + " ".join(cmd))
    subprocess.run(cmd, cwd="/opt/Isaac-GR00T", check=True)
    checkpoints_vol.commit()

    # ---- 4. Post-training open-loop eval on val set across every checkpoint ----
    print("\n[finetune] Picking best checkpoint by open-loop MSE on val set...")
    ckpts = sorted(
        (p for p in Path(output_dir).glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not ckpts:
        raise RuntimeError(f"no checkpoints under {output_dir}")
    print(f"[finetune]   found {len(ckpts)} checkpoints: "
          + ", ".join(c.name for c in ckpts))

    # Programmatic eval — load Gr00tPolicy ONCE per checkpoint (not once per
    # traj_id) and call evaluate_single_trajectory from the upstream module.
    import sys
    sys.path.insert(0, "/opt")
    sys.path.insert(0, "/opt/Isaac-GR00T")
    import yam_config  # registers NEW_EMBODIMENT modality  # noqa: F401
    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.open_loop_eval import evaluate_single_trajectory
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    embodiment = EmbodimentTag.resolve("NEW_EMBODIMENT")

    eval_results = []
    for ckpt in ckpts:
        print(f"\n[finetune] === eval {ckpt.name} ===")
        try:
            policy = Gr00tPolicy(
                embodiment_tag=embodiment,
                model_path=str(ckpt),
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except Exception as e:
            print(f"[finetune]   load FAIL: {type(e).__name__}: {e}")
            continue
        dataset_loader = LeRobotEpisodeLoader(
            dataset_path=val_path,
            modality_configs=policy.get_modality_config(),
            video_backend="torchcodec",
            video_backend_kwargs=None,
        )
        per_traj_mse = []
        for traj_id in val_traj_ids:
            if traj_id >= len(dataset_loader):
                print(f"[finetune]   traj {traj_id}: OOR (dataset has {len(dataset_loader)})")
                continue
            try:
                mse, mae = evaluate_single_trajectory(
                    policy, dataset_loader, traj_id, embodiment,
                    modality_keys=None,
                    steps=val_steps_per_traj,
                    action_horizon=val_action_horizon,
                    save_plot_path=str(ckpt / f"val_traj_{traj_id}.png"),
                )
            except Exception as e:
                print(f"[finetune]   traj {traj_id}: FAILED {type(e).__name__}: {e}")
                continue
            per_traj_mse.append(float(mse))
            print(f"[finetune]   traj {traj_id}: MSE={mse:.5f} MAE={mae:.5f}")

        # Free GPU before loading the next checkpoint.
        del policy
        torch.cuda.empty_cache()

        if not per_traj_mse:
            print(f"[finetune] {ckpt.name}: NO val MSE collected, skipping")
            continue
        mean_mse = sum(per_traj_mse) / len(per_traj_mse)
        eval_results.append({"checkpoint": ckpt.name, "mean_mse": mean_mse,
                             "per_traj_mse": per_traj_mse})
        print(f"[finetune] {ckpt.name}: mean_mse={mean_mse:.5f}")

    if not eval_results:
        raise RuntimeError("no checkpoint produced usable val MSE")

    eval_results.sort(key=lambda r: r["mean_mse"])
    best = eval_results[0]
    print(f"\n[finetune] BEST checkpoint: {best['checkpoint']} "
          f"(mean_mse={best['mean_mse']:.5f})")

    # Write a summary file + symlink the winner as /<output_subdir>/best.
    summary_path = Path(output_dir) / "val_summary.json"
    summary_path.write_text(json.dumps({
        "train_dataset": train_dataset_repo_id,
        "val_dataset": val_dataset_repo_id,
        "val_traj_ids": list(val_traj_ids),
        "results": eval_results,
        "best": best,
    }, indent=2))
    print(f"[finetune] wrote {summary_path}")

    best_link = Path(output_dir) / "best"
    if best_link.is_symlink() or best_link.exists():
        if best_link.is_dir() and not best_link.is_symlink():
            shutil.rmtree(best_link)
        else:
            best_link.unlink()
    best_link.symlink_to(best["checkpoint"])
    print(f"[finetune] {best_link} -> {best['checkpoint']}")

    checkpoints_vol.commit()
    return {
        "output_dir": output_dir,
        "best_checkpoint": best["checkpoint"],
        "mean_mse": best["mean_mse"],
        "all_results": eval_results,
    }


@app.local_entrypoint()
def main(
    train_dataset_repo_id: str = "allenai/29112025-block-01",
    val_dataset_repo_id: str   = "allenai/29112025-block-02",
    max_steps: int             = 4000,
    save_steps: int            = 500,
    global_batch_size: int     = 32,
    learning_rate: float       = 1e-4,
    output_subdir: str         = "yam-latest",
    wandb_project: str         = "finetune-gr00t-yam",
) -> None:
    result = finetune.remote(
        train_dataset_repo_id=train_dataset_repo_id,
        val_dataset_repo_id=val_dataset_repo_id,
        max_steps=max_steps,
        save_steps=save_steps,
        global_batch_size=global_batch_size,
        learning_rate=learning_rate,
        output_subdir=output_subdir,
        wandb_project=wandb_project,
    )
    print()
    print("=" * 70)
    print(f"DONE — best checkpoint: {result['best_checkpoint']}")
    print(f"       mean val MSE:    {result['mean_mse']:.5f}")
    print()
    print("Pull the best checkpoint locally with:")
    print(
        f"  modal volume get {VOLUME_NAME} /{output_subdir}/best "
        f'"hf-cache/checkpoints/{output_subdir}"'
    )
    print()
    print("All per-checkpoint MSE scores:")
    for r in result["all_results"]:
        print(f"  {r['checkpoint']:25s} mean_mse={r['mean_mse']:.5f}")
    print()
    print("Then ./scripts/run_server.sh will auto-detect it.")
