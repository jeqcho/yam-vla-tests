"""Modal app: serve DreamZero (14B WAM) over WebSocket on remote H100 GPUs.

Why Modal?
    DreamZero's `socket_test_optimized_AR.py` uses `torchrun --nproc_per_node 2`
    + NCCL for distributed inference and the 14B BF16 weights are ~28 GB. The
    local host (single RTX 5090 / 32 GB) cannot run this. Modal gives us a
    multi-GPU box on demand with a public WebSocket URL.

What this serves
    Two flavors selectable via `MODEL` env at deploy time:

      MODEL=droid   →  GEAR-Dreams/DreamZero-DROID
                       (single-arm Franka, 2 ext cams + 1 wrist, 7+1 action)
                       This is the *public* DreamZero policy and what the
                       paper's "vanilla" results are reported on.

      MODEL=yam     →  GEAR-Dreams/DreamZero-YAM-bimanual
                       PLACEHOLDER repo id. No public checkpoint exists yet;
                       fill this in after `dreamzero_finetune.py` produces one
                       (or after Ai2/NVIDIA publishes one).

Wire format
    See `eval_utils/policy_server.py` in dreamzero/. The client receives a
    `PolicyServerConfig` dict over msgpack on connect, then sends obs frames
    keyed by `observation/...`, gets back `action` arrays. We expose the
    WebSocket on port 8000 via `modal.web_server`, which gives us a public
    HTTPS/WSS URL of the form:
        https://<workspace>--dreamzero-droid-policy-serve.modal.run

Usage
    # Bring it up (returns a URL; tail logs with `modal logs ...`):
    modal serve modal/dreamzero_server.py
    # …or detach and keep it running:
    modal deploy modal/dreamzero_server.py

    # Local client:
    uv run python scripts/smoke_test_remote.py --url wss://<the-url>/

Cost note
    H100:2 is ~$10/hr on Modal. Keep `container_idle_timeout` short and stop
    the deployment (`modal app stop dreamzero-droid`) when not testing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# -----------------------------------------------------------------------------
# Knobs
# -----------------------------------------------------------------------------

MODEL = os.environ.get("MODEL", "droid").lower()
# `droid` = the public DreamZero-DROID checkpoint, runnable today.
# `yam`   = a future bimanual-YAM-finetuned checkpoint; will 404 today.
CHECKPOINTS = {
    "droid": "GEAR-Dreams/DreamZero-DROID",
    "yam": os.environ.get("YAM_REPO_ID", "GEAR-Dreams/DreamZero-YAM-bimanual"),
}
REPO_ID = CHECKPOINTS[MODEL]
N_GPUS = int(os.environ.get("N_GPUS", "2"))  # DreamZero requires >=2
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")  # H100 or B200 (GB200) per README
PORT = 8000
APP_NAME = f"dreamzero-{MODEL}"

# -----------------------------------------------------------------------------
# Image build
#
# We follow dreamzero/README.md exactly: CUDA-12.9 PyTorch wheels, flash-attn,
# install dreamzero in editable mode. We do NOT install the GB200-only
# transformer_engine path here; that's commented out below and only worth
# enabling if we ever deploy on B200.
# -----------------------------------------------------------------------------

DREAMZERO_DIR = Path(__file__).parent.parent / "dreamzero"
assert DREAMZERO_DIR.is_dir(), (
    f"Expected the dreamzero repo at {DREAMZERO_DIR}. "
    f"Run `git clone https://github.com/dreamzero0/dreamzero ./dreamzero` first."
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
            "HF_HOME": "/root/.cache/huggingface",
            "PYTHONUNBUFFERED": "1",
            # dreamzero/socket_test_optimized_AR.py reads these:
            "ATTENTION_BACKEND": "TE",
        }
    )
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        extra_index_url="https://download.pytorch.org/whl/cu129",
    )
    # flash-attn needs torch already installed; --no-build-isolation per README
    .run_commands(
        "MAX_JOBS=8 pip install --no-build-isolation flash-attn==2.7.4.post1 || "
        "MAX_JOBS=8 pip install --no-build-isolation flash-attn",
    )
    # Copy the cloned dreamzero source into the image and editable-install it.
    .add_local_dir(str(DREAMZERO_DIR), "/opt/dreamzero", copy=True)
    .run_commands(
        "cd /opt/dreamzero && pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129",
    )
)

# Persistent HF cache so we don't redownload weights between deploys.
hf_cache = modal.Volume.from_name("dreamzero-hf-cache", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=60 * 60 * 6,
    # Cold start is ~10 minutes (HF download + flash-attn import + warmup).
    # Keep one container warm-ish to amortize that across smoke tests.
    scaledown_window=600,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])]
    if os.environ.get("MODAL_REQUIRE_HF_SECRET") == "1"
    else [],
)
@modal.concurrent(max_inputs=1)
@modal.web_server(port=PORT, startup_timeout=60 * 25)
def serve():
    """Launch DreamZero's WebSocket policy server on `0.0.0.0:PORT`.

    `modal.web_server` waits for the port to start accepting connections, then
    forwards a public HTTPS URL to it. WebSocket upgrades (`wss://`) pass
    through unchanged.
    """
    ckpt_dir = f"/root/.cache/huggingface/dreamzero_{MODEL}"
    if not os.path.isdir(ckpt_dir) or not os.listdir(ckpt_dir):
        print(f"Downloading {REPO_ID} → {ckpt_dir}", flush=True)
        # Use `hf` CLI (newer name for huggingface-cli).
        subprocess.check_call(
            [
                "hf",
                "download",
                REPO_ID,
                "--repo-type",
                "model",
                "--local-dir",
                ckpt_dir,
            ]
        )
    else:
        print(f"Reusing cached checkpoint at {ckpt_dir}", flush=True)

    # Launch dreamzero's torchrun-based server. Daemonize so this function
    # returns control to `@modal.web_server`, which then probes :8000.
    cmd = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={N_GPUS}",
        "/opt/dreamzero/socket_test_optimized_AR.py",
        f"--port={PORT}",
        f"--model-path={ckpt_dir}",
        "--enable-dit-cache",
    ]
    print("Launching:", " ".join(cmd), flush=True)
    # The server prints heartbeats to stdout; we don't wait().
    subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=os.environ.copy())


@app.local_entrypoint()
def main():
    """`modal run modal/dreamzero_server.py` prints the public URL and exits."""
    print(f"Deploy: modal deploy modal/dreamzero_server.py  → URL appears at the end of stdout.")
    print(f"Serve+stream: modal serve modal/dreamzero_server.py  → URL appears in the banner.")
    print(f"MODEL={MODEL}  REPO_ID={REPO_ID}  GPU={GPU_TYPE}:{N_GPUS}")
