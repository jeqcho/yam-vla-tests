#!/usr/bin/env bash
# GR00T N1.7 — bimanual YAM finetune server.
# Wire: ZeroMQ REQ/REP + msgpack-numpy on :5556.
# Model: jeqcho/gr00t-n17-yam-bimanual (bf16 safetensors).
# Backbone: Isaac-GR00T's gr00t/eval/run_gr00t_server.py inside its own uv venv.
#
# Prereqs (one-time):
#   1. Clone + uv-sync Isaac-GR00T (15 GB venv). Reuse the existing
#      grootn1.7-exploration clone if available; override via
#      GR00T_DIR=/abs/path.
#   2. ./scripts/download_checkpoints.sh gr00t-n17
#   3. HF auth for the gated `nvidia/Cosmos-Reason2-2B` (Cosmos processor).
#      `hf auth login` once + accept EULA on huggingface.co.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# Reuse the existing Isaac-GR00T clone (still under _archive/ post-refactor).
DEFAULT_GR00T_DIR="$REPO_ROOT/_archive/grootn1.7-exploration/Isaac-GR00T"
GR00T_DIR="${GR00T_DIR:-$DEFAULT_GR00T_DIR}"
if [[ ! -d "$GR00T_DIR" ]]; then
    echo "Isaac-GR00T clone not found at $GR00T_DIR" >&2
    echo "  Either clone+sync it there, or set GR00T_DIR=/abs/path" >&2
    exit 1
fi
if [[ ! -x "$GR00T_DIR/.venv/bin/python" ]]; then
    echo "Isaac-GR00T venv missing at $GR00T_DIR/.venv/" >&2
    echo "  Run: cd \"$GR00T_DIR\" && uv sync --all-extras" >&2
    exit 1
fi

CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/hf-cache/checkpoints/jeqcho_gr00t-n17-yam-bimanual}"
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Checkpoint dir not found: $CKPT_DIR" >&2
    echo "  Run: $REPO_ROOT/scripts/download_checkpoints.sh gr00t-n17" >&2
    exit 1
fi

PORT="${PORT:-5556}"
DEVICE="${DEVICE:-cuda:0}"

# Cosmos blobs are cached in the original grootn1.7-exploration HF cache.
# Detect; force offline mode if found, so transformers doesn't 401 on the
# gated repo even without a fresh `hf auth login`.
DEFAULT_HF_HOME="$REPO_ROOT/_archive/grootn1.7-exploration/hf-cache"
COSMOS_CACHED=0
if [[ -d "$DEFAULT_HF_HOME/models--nvidia--Cosmos-Reason2-2B" ]] \
|| [[ -d "$DEFAULT_HF_HOME/hub/models--nvidia--Cosmos-Reason2-2B" ]]; then
    COSMOS_CACHED=1
fi
export HF_HOME="${HF_HOME:-$DEFAULT_HF_HOME}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

if [[ "$COSMOS_CACHED" == "1" && -z "${HF_HUB_OFFLINE:-}" ]]; then
    echo "[servers/gr00t-n17] Cosmos blobs cached at $HF_HOME -- forcing HF_HUB_OFFLINE=1"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

cd "$GR00T_DIR"

echo "[servers/gr00t-n17] model_path: $CKPT_DIR"
echo "[servers/gr00t-n17] port:       $PORT"
echo "[servers/gr00t-n17] device:     $DEVICE"
echo "[servers/gr00t-n17] modality:   $HERE/yam_config.py"

# offline_shim.py bypasses transformers' HF API ping for the gated
# Cosmos backbone when HF_HUB_OFFLINE=1. Imported as a side-effect
# before run_gr00t_server.py runs so the patch applies first.
exec uv run python -c "
import sys, runpy
sys.path.insert(0, '$HERE')
import offline_shim  # noqa: F401
sys.argv = [
    'run_gr00t_server.py',
    '--model-path', '$CKPT_DIR',
    '--embodiment-tag', 'NEW_EMBODIMENT',
    '--modality-config-path', '$HERE/yam_config.py',
    '--device', '$DEVICE',
    '--host', '0.0.0.0',
    '--port', '$PORT',
] + sys.argv[1:]
runpy.run_path('gr00t/eval/run_gr00t_server.py', run_name='__main__')
" "$@"
