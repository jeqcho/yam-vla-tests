"""Modal app: fine-tune DreamZero-AgiBot on a bimanual YAM dataset.

DreamZero's `scripts/train/yam_training.sh` does exactly this:
    torchrun --nproc_per_node $NUM_GPUS --standalone \
        groot/vla/experiment/experiment.py data=dreamzero/yam_relative …

The training script expects:
  1. YAM dataset in LeRobot v2 layout (parquet + mp4) with `meta/modality.json`
     declaring 14-D state and action across `left_joint_pos/left_gripper_pos/
     right_joint_pos/right_gripper_pos` and `meta/embodiment.json` with
     `embodiment_tag: yam`.
  2. Wan2.1-I2V-14B-480P base weights (~30 GB).
  3. umt5-xxl tokenizer (~10 GB).
  4. DreamZero-AgiBot LoRA-base (~45 GB).

This Modal app stages all of those onto a persistent Volume and then runs the
yam_training.sh script. It is NOT auto-invoked — call `modal run
modal/dreamzero_finetune.py::run` explicitly. Cost on 4×H100 at 100k steps is
on the order of $200; the README says "30 minutes of play data" is enough,
which is data volume, not training time.

Required arguments:
    --dataset-hf-id <repo_id>
        HF dataset id whose root is a LeRobot v2 YAM tree. Set to an Ai2
        MolmoAct2-BimanualYAM dataset id once we've crosswalked its
        `modality.json` to DreamZero's expected schema, OR to a private
        repo containing your own teleop.

    --run-name <name>
        Becomes the OUTPUT_DIR suffix; also the WandB run name if WANDB_API_KEY
        is set as a Modal secret.

Optional:
    --num-gpus 4         GPUs to allocate (H100s). YAM_training.sh uses
                         per_device_batch=4; 4×H100 gives global batch 16.
    --max-steps 100000   Default matches the paper.
    --skip-data-stage    Reuse previously-staged data on the volume.

Note on "30 minutes of play data"
    The DreamZero paper claim is that 30 min of teleop data (~55 trajectories)
    suffices for the post-train. You still need to *do* the full LoRA SFT pass
    (default 100k steps) — the "30 minutes" refers only to the size of the
    target-embodiment dataset.
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

# Cache: HF downloads (DreamZero-AgiBot, Wan2.1-I2V-14B-480P, umt5-xxl, dataset).
ckpt_volume = modal.Volume.from_name("dreamzero-ckpts", create_if_missing=True)
# Outputs: training checkpoints land here and persist across runs.
output_volume = modal.Volume.from_name("dreamzero-finetune-out", create_if_missing=True)

# -----------------------------------------------------------------------------
# Image  (same recipe as dreamzero_server.py; identical so it caches)
# -----------------------------------------------------------------------------

DREAMZERO_DIR = Path(__file__).parent.parent / "dreamzero"
assert DREAMZERO_DIR.is_dir(), (
    f"Expected dreamzero repo at {DREAMZERO_DIR}. "
    f"Run `git clone https://github.com/dreamzero0/dreamzero ./dreamzero`."
)

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
def stage_checkpoints_and_data(dataset_hf_id: str) -> None:
    """Pull AgiBot base, Wan2.1, umt5-xxl, and the user dataset onto the volume."""
    print(f"Staging into /ckpts (dataset={dataset_hf_id}) ...", flush=True)
    _hf_download("GEAR-Dreams/DreamZero-AgiBot", "/ckpts/DreamZero-AgiBot")
    _hf_download("Wan-AI/Wan2.1-I2V-14B-480P", "/ckpts/Wan2.1-I2V-14B-480P")
    _hf_download("google/umt5-xxl", "/ckpts/umt5-xxl")
    _hf_download(dataset_hf_id, "/ckpts/yam_dataset", repo_type="dataset")
    print("Staging done. Committing volume…", flush=True)
    ckpt_volume.commit()


@app.function(
    gpu=os.environ.get("FT_GPU", "H100:4"),
    timeout=60 * 60 * 36,
    volumes={"/ckpts": ckpt_volume, "/out": output_volume},
    secrets=(
        [modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])]
        if os.environ.get("MODAL_USE_WANDB", "0") == "1"
        else []
    ),
)
def train(run_name: str, max_steps: int = 100_000) -> str:
    """Run scripts/train/yam_training.sh against the staged checkpoints."""
    output_dir = f"/out/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "YAM_DATA_ROOT": "/ckpts/yam_dataset",
            "OUTPUT_DIR": output_dir,
            "WAN_CKPT_DIR": "/ckpts/Wan2.1-I2V-14B-480P",
            "TOKENIZER_DIR": "/ckpts/umt5-xxl",
            # NUM_GPUS auto-detects via nvidia-smi.
        }
    )

    # The script hard-codes `pretrained_model_path=./checkpoints/DreamZero-AgiBot`,
    # so we symlink. Cleaner than editing the script for the small win.
    os.makedirs("/opt/dreamzero/checkpoints", exist_ok=True)
    link = "/opt/dreamzero/checkpoints/DreamZero-AgiBot"
    if not os.path.islink(link) and not os.path.exists(link):
        os.symlink("/ckpts/DreamZero-AgiBot", link)

    # Allow caller to shorten the run for smoke tests.
    cmd = ["bash", "/opt/dreamzero/scripts/train/yam_training.sh"]
    if max_steps != 100_000:
        # yam_training.sh sets max_steps=100000 inline; sed the override.
        with open("/opt/dreamzero/scripts/train/yam_training.sh") as f:
            script = f.read()
        script = script.replace("max_steps=100000", f"max_steps={max_steps}")
        patched = "/tmp/yam_training_patched.sh"
        with open(patched, "w") as f:
            f.write(script)
        os.chmod(patched, 0o755)
        cmd = ["bash", patched]

    print(f"Launching: {' '.join(cmd)}", flush=True)
    print(f"OUTPUT_DIR={output_dir}", flush=True)
    rc = subprocess.call(cmd, env=env, cwd="/opt/dreamzero")
    output_volume.commit()
    if rc != 0:
        raise RuntimeError(f"yam_training.sh exited with rc={rc}")
    return output_dir


@app.local_entrypoint()
def run(
    dataset_hf_id: str = "REQUIRED",
    run_name: str = "dreamzero_yam_lora_run1",
    max_steps: int = 100_000,
    skip_data_stage: bool = False,
):
    if dataset_hf_id == "REQUIRED":
        sys.exit(
            "ERROR: pass --dataset-hf-id <repo>. The repo must be a LeRobot v2\n"
            "YAM tree with meta/modality.json declaring left_joint_pos/...\n"
            "See REPORT_dreamzero_setup.md for the modality schema."
        )
    if not skip_data_stage:
        stage_checkpoints_and_data.remote(dataset_hf_id)
    out = train.remote(run_name, max_steps=max_steps)
    print(f"\nFine-tune output is on the dreamzero-finetune-out volume at: {out}")
    print(f"Copy it down with:  modal volume get dreamzero-finetune-out {run_name} ./hf-cache/")
