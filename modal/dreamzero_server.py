"""Modal app: serve DreamZero (14B WAM) over WebSocket on remote H100 GPUs,
exposed via a **`modal.forward()` Tunnel** rather than `@modal.web_server`.

Why Tunnels (per https://modal.com/blog/physical-intelligence-runs-real-time-remote-inference-for-robotic-control-on-modal)
---------------------------------------------------------------------------------------------------------------------
Physical Intelligence's writeup compares two ways to reach a container:

  1. `modal.forward(port)` — direct TCP forward, TLS terminated at Modal's edge.
     Public API, ~few ms TLS handshake, then raw bytes. Good for low-overhead
     long-lived sockets (which is what DreamZero's WebSocket server is).

  2. A custom QUIC-over-UDP NAT-hole-punched portal that they co-built with
     Modal for ~10–15 ms cloud overhead. Not a publicly-available knob; PI
     describes it as a deeper collaboration. We can't adopt it today.

For DreamZero, per-inference is ~3 s on H100 — network latency is in the
noise. Tunnels are the right call: simpler than `@modal.web_server`, no HTTP
routing layer, and the `tunnel.url` is a `wss://` endpoint the client speaks
to directly.

Why Modal at all
----------------
Local host is a single RTX 5090 (32 GB). DreamZero's `socket_test_optimized_AR.py`
hard-launches via `torchrun --nproc_per_node 2` (NCCL). The 14B BF16 weights
alone are ~28 GB. Two H100s on Modal is the cheapest box that just works.

Quickstart
----------
    # Bring it up — runs until you Ctrl-C; URL appears in stdout banner.
    modal serve modal/dreamzero_server.py

    # …or detach:
    modal deploy modal/dreamzero_server.py

    # Local smoke test (URL is printed by the function, NOT in modal's banner —
    # `modal logs dreamzero-droid` to see it on a deployed app):
    uv run python scripts/smoke_test_remote.py \\
        --url wss://<the-url-from-stdout>/

Cost
----
H100:2 ≈ $10/hr while running. `scaledown_window=600` releases the box
~10 min after the last request. `modal app stop dreamzero-droid` to kill it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

MODEL = os.environ.get("MODEL", "droid").lower()
CHECKPOINTS = {
    "droid": "GEAR-Dreams/DreamZero-DROID",  # public; runnable today
    # YAM-finetuned: populated by env once dreamzero_finetune.py uploads one.
    "yam": os.environ.get("YAM_REPO_ID", "GEAR-Dreams/DreamZero-YAM-bimanual"),
}
REPO_ID = CHECKPOINTS[MODEL]
N_GPUS = int(os.environ.get("N_GPUS", "2"))
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")
PORT = 8000
APP_NAME = f"dreamzero-{MODEL}"

DREAMZERO_DIR = Path(__file__).parent.parent / "dreamzero"
# Don't assert at module top — the module is also imported inside Modal
# containers, where the local-clone path doesn't exist. The local_entrypoint
# checks instead, before any .remote() call.

# ---------------------------------------------------------------------------
# Image — CUDA 12.9 + torch 2.8 + flash-attn + editable dreamzero install.
# Matches dreamzero/pyproject.toml and README exactly.
# ---------------------------------------------------------------------------

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
            "ATTENTION_BACKEND": "TE",  # read by socket_test_optimized_AR.py
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

hf_cache = modal.Volume.from_name("dreamzero-hf-cache", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=60 * 60 * 6,
    scaledown_window=600,
    # Tunnels need this so the container survives the WebSocket lifetime.
    max_containers=1,
)
def serve() -> None:
    """Launch DreamZero's WebSocket policy server and expose it via a Modal Tunnel.

    The HTTPS tunnel terminates TLS at Modal's edge and forwards to localhost:PORT
    inside the container. WebSocket upgrades pass through unchanged, so a client
    connects with `wss://<tunnel.url-host>/`.
    """
    ckpt_dir = f"/root/.cache/huggingface/dreamzero_{MODEL}"
    if not os.path.isdir(ckpt_dir) or not os.listdir(ckpt_dir):
        print(f"Downloading {REPO_ID} → {ckpt_dir}", flush=True)
        subprocess.check_call([
            "hf", "download", REPO_ID, "--repo-type", "model",
            "--local-dir", ckpt_dir,
        ])
    else:
        print(f"Reusing cached checkpoint at {ckpt_dir}", flush=True)

    cmd = [
        "python", "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={N_GPUS}",
        "/opt/dreamzero/socket_test_optimized_AR.py",
        f"--port={PORT}",
        f"--model-path={ckpt_dir}",
        "--enable-dit-cache",
    ]
    print("Launching:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=os.environ.copy())

    # Wait for the server to bind :PORT before forwarding the tunnel.
    print(f"Waiting for server to bind 0.0.0.0:{PORT} ...", flush=True)
    import socket as _socket
    deadline = time.time() + 60 * 25  # 25 min cold start budget
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with rc={proc.returncode}")
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                break
        time.sleep(2)
    else:
        proc.kill()
        raise TimeoutError("server did not bind :8000 within 25 min")

    # `modal.forward(port)` returns a context-managed tunnel. The HTTPS form
    # gives us a stable wss:// URL — the WebSocket handshake passes through.
    with modal.forward(PORT) as tunnel:
        host = tunnel.url.replace("https://", "")
        print("\n" + "=" * 70, flush=True)
        print(f"  DreamZero server is up. Connect with:", flush=True)
        print(f"    wss://{host}/", flush=True)
        print(f"  (URL form for browsers/curl:  {tunnel.url})", flush=True)
        print("=" * 70 + "\n", flush=True)

        # Hold the function open for the lifetime of the server process.
        # When `serve` returns, the tunnel + container shut down.
        rc = proc.wait()
        raise SystemExit(rc)


@app.local_entrypoint()
def main():
    if not DREAMZERO_DIR.is_dir():
        sys.exit(f"DreamZero repo missing at {DREAMZERO_DIR}; clone it first.")
    print(f"MODEL={MODEL}  REPO_ID={REPO_ID}  GPU={GPU_TYPE}:{N_GPUS}")
    print("Run with:   modal serve modal/dreamzero_server.py")
    print("…or detach: modal deploy modal/dreamzero_server.py")
    print("URL appears in the function's stdout — not the modal banner.")
