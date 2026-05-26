"""Modal app: fine-tune DreamZero-AgiBot on a YAM dataset prepared by
`modal/prepare_yam_data.py`.

Pipeline:
  prepare_yam_data.py  →  dreamzero-yam-data volume  →  this fine-tune
                          (uses meta/modality.json + tag=yam from the prep)

The training script expects:
  1. A LeRobot v2 YAM tree with `meta/{modality,embodiment,stats,
     relative_stats_dreamzero,tasks,episodes}.*`. The prep job builds this.
  2. Wan2.1-I2V-14B-480P base weights (~30 GB).
  3. umt5-xxl tokenizer (~10 GB).
  4. DreamZero-AgiBot LoRA-base (~45 GB).

Run with:
    # Smoke (~200 steps, ~$10, validates the full pipeline e2e):
    modal run modal/dreamzero_finetune.py::run \\
        --prepared-tag yam_box_smoke --run-name dz-yam-smoke --max-steps 200

    # Full (~100k steps, ~$200, ~12 hr on H100:4):
    modal run modal/dreamzero_finetune.py::run \\
        --prepared-tag yam_box_full --run-name dz-yam-v1 --max-steps 100000

Note on "30 minutes of play data"
    The DreamZero paper claim is that 30 min of teleop data (~55 trajectories)
    suffices for the post-train. You still need to *do* the LoRA SFT pass
    (default 100k steps) — the "30 minutes" refers to dataset size, not
    wallclock.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# -----------------------------------------------------------------------------
# Volumes
# -----------------------------------------------------------------------------

# Cache for HF model downloads (DreamZero-AgiBot, Wan2.1-I2V-14B-480P, umt5-xxl).
ckpt_volume = modal.Volume.from_name("dreamzero-ckpts", create_if_missing=True)
# Prepared YAM dataset(s); populated by modal/prepare_yam_data.py.
data_volume = modal.Volume.from_name("dreamzero-yam-data", create_if_missing=True)
# Outputs: training checkpoints land here and persist across runs.
output_volume = modal.Volume.from_name("dreamzero-finetune-out", create_if_missing=True)

# -----------------------------------------------------------------------------
# Image  (same recipe as dreamzero_server.py; identical so it caches)
# -----------------------------------------------------------------------------

DREAMZERO_DIR = Path(__file__).parent.parent / "dreamzero"
# Skip the assert at import time; container imports this module too.

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "ninja-build")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/ckpts/hf_home",
            "PYTHONUNBUFFERED": "1",
            "HYDRA_FULL_ERROR": "1",
            "ATTENTION_BACKEND": "TE",
            "WANDB_DIR": "/out/wandb",
        }
    )
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        extra_index_url="https://download.pytorch.org/whl/cu129",
    )
    .pip_install("packaging", "ninja", "wheel", "setuptools>=68")
    .run_commands(
        "MAX_JOBS=8 pip install --no-build-isolation flash-attn==2.7.4.post1 || "
        "MAX_JOBS=8 pip install --no-build-isolation flash-attn",
    )
    .add_local_dir(str(DREAMZERO_DIR), "/opt/dreamzero", copy=True)
    .run_commands(
        "cd /opt/dreamzero && pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129",
    )
)

app = modal.App("dreamzero-yam-finetune", image=image)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _hf_download(repo_id: str, dest: str, repo_type: str = "model") -> None:
    if os.path.isdir(dest) and os.listdir(dest):
        print(f"  (cached) {repo_id}", flush=True)
        return
    os.makedirs(dest, exist_ok=True)
    print(f"  → fetching {repo_id} into {dest}", flush=True)
    subprocess.check_call(
        ["hf", "download", repo_id, "--repo-type", repo_type, "--local-dir", dest]
    )


@app.function(
    timeout=60 * 60 * 4,
    volumes={"/ckpts": ckpt_volume},
    cpu=8,
)
def stage_model_checkpoints() -> None:
    """Pull AgiBot base + Wan2.1 + umt5-xxl onto the ckpts volume (idempotent)."""
    print("Staging model checkpoints into /ckpts ...", flush=True)
    _hf_download("GEAR-Dreams/DreamZero-AgiBot", "/ckpts/DreamZero-AgiBot")
    _hf_download("Wan-AI/Wan2.1-I2V-14B-480P", "/ckpts/Wan2.1-I2V-14B-480P")
    _hf_download("google/umt5-xxl", "/ckpts/umt5-xxl")
    print("Staging done. Committing volume…", flush=True)
    ckpt_volume.commit()


@app.function(
    gpu=os.environ.get("FT_GPU", "H100:4"),
    timeout=60 * 60 * 36,
    volumes={
        "/ckpts": ckpt_volume,
        "/data": data_volume,
        "/out": output_volume,
    },
    secrets=(
        [modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])]
        if os.environ.get("MODAL_USE_WANDB", "0") == "1"
        else []
    ),
)
def train(prepared_tag: str, run_name: str, max_steps: int = 100_000) -> str:
    """Run scripts/train/yam_training.sh against the prepared YAM dataset."""
    data_root = f"/data/prepared/{prepared_tag}"
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"No prepared dataset at {data_root}. "
            f"Run `modal run modal/prepare_yam_data.py::prepare --hf-repo ... "
            f"--tag {prepared_tag}` first."
        )
    if not os.path.isfile(f"{data_root}/meta/embodiment.json"):
        raise RuntimeError(f"{data_root}/meta/embodiment.json missing — prep failed?")

    output_dir = f"/out/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "YAM_DATA_ROOT": data_root,
        "OUTPUT_DIR": output_dir,
        "WAN_CKPT_DIR": "/ckpts/Wan2.1-I2V-14B-480P",
        "TOKENIZER_DIR": "/ckpts/umt5-xxl",
        # NUM_GPUS auto-detects via nvidia-smi inside the script.
    })

    # yam_training.sh hard-codes `pretrained_model_path=./checkpoints/DreamZero-AgiBot`,
    # so symlink the staged copy into place.
    os.makedirs("/opt/dreamzero/checkpoints", exist_ok=True)
    link = "/opt/dreamzero/checkpoints/DreamZero-AgiBot"
    if not os.path.islink(link) and not os.path.exists(link):
        os.symlink("/ckpts/DreamZero-AgiBot", link)

    # Allow caller to shorten the run for smoke tests.
    cmd = ["bash", "/opt/dreamzero/scripts/train/yam_training.sh"]
    if max_steps != 100_000:
        with open("/opt/dreamzero/scripts/train/yam_training.sh") as f:
            script = f.read()
        script = script.replace("max_steps=100000", f"max_steps={max_steps}")
        # Also tighten save_steps so a smoke run actually emits a checkpoint.
        if max_steps < 1000:
            script = script.replace("save_steps=10000", f"save_steps={max(50, max_steps // 2)}")
        patched = "/tmp/yam_training_patched.sh"
        with open(patched, "w") as f:
            f.write(script)
        os.chmod(patched, 0o755)
        cmd = ["bash", patched]

    print(f"Launching: {' '.join(cmd)}", flush=True)
    print(f"YAM_DATA_ROOT={data_root}", flush=True)
    print(f"OUTPUT_DIR={output_dir}", flush=True)
    rc = subprocess.call(cmd, env=env, cwd="/opt/dreamzero")
    output_volume.commit()
    if rc != 0:
        raise RuntimeError(f"yam_training.sh exited with rc={rc}")
    return output_dir


@app.local_entrypoint()
def run(
    prepared_tag: str = "yam_box_smoke",
    run_name: str = "dz-yam-smoke",
    max_steps: int = 100_000,
    skip_model_stage: bool = False,
):
    if not DREAMZERO_DIR.is_dir():
        sys.exit(f"DreamZero repo missing at {DREAMZERO_DIR}; clone it first.")
    if not skip_model_stage:
        stage_model_checkpoints.remote()
    out = train.remote(prepared_tag, run_name, max_steps=max_steps)
    print(f"\nFine-tune output is on the dreamzero-finetune-out volume at: {out}")
    print(f"Pull it with:  modal volume get -r dreamzero-finetune-out {run_name} ./hf-cache/")
