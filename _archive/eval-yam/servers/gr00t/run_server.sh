#!/usr/bin/env bash
# Launch the GR00T-N1.7 inference server for the bimanual YAM checkpoint
# `jeqcho/gr00t-n17-yam-bimanual` on port 5556 (ZMQ + msgpack-numpy).
#
# Server: Isaac-GR00T's gr00t/eval/run_gr00t_server.py
# Model:  jeqcho/gr00t-n17-yam-bimanual (HF transformers safetensors, bf16)
# Wire:   ZeroMQ REQ/REP + msgpack-numpy, tcp://0.0.0.0:5556
#
# Prereqs (one-time):
#   1. Isaac-GR00T must be cloned + uv-synced with its own .venv. The
#      existing clone in `grootn1.7 exploration/Isaac-GR00T` is reused
#      verbatim -- 15 GB venv, no point duplicating.
#      To use a different clone, set GR00T_DIR=/path/to/Isaac-GR00T.
#   2. The checkpoint must be present at $CKPT_DIR (defaults to
#      hf-cache/jeqcho_gr00t-n17-yam-bimanual). Run
#      ../../download_checkpoints.sh first.
#   3. HF auth for the gated `nvidia/Cosmos-Reason2-2B` repo (GR00T processor
#      pulls config from there). See grootn1.7-exploration/HANDOFF.md Step 0.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_YAM_DIR="$(cd "$HERE/../.." && pwd)"

# Reuse the existing gr00t exploration's Isaac-GR00T + uv venv.
GR00T_DIR="${GR00T_DIR:-$(cd "$EVAL_YAM_DIR/../grootn1.7 exploration/Isaac-GR00T" && pwd)}"
if [[ ! -d "$GR00T_DIR" ]]; then
    echo "Isaac-GR00T not found at $GR00T_DIR" >&2
    echo "  Set GR00T_DIR=/path/to/Isaac-GR00T, or clone+uv-sync the repo." >&2
    exit 1
fi
if [[ ! -x "$GR00T_DIR/.venv/bin/python" ]]; then
    echo "Isaac-GR00T venv not found at $GR00T_DIR/.venv/" >&2
    echo "  Run: cd \"$GR00T_DIR\" && uv sync --all-extras" >&2
    exit 1
fi

CKPT_DIR="${CKPT_DIR:-$EVAL_YAM_DIR/hf-cache/checkpoints/jeqcho_gr00t-n17-yam-bimanual}"
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Checkpoint dir not found: $CKPT_DIR" >&2
    echo "  Download it with: $EVAL_YAM_DIR/download_checkpoints.sh gr00t-n17" >&2
    exit 1
fi

PORT="${PORT:-5556}"
DEVICE="${DEVICE:-cuda:0}"

# Default HF cache to the existing grootn1.7-exploration cache because it
# already contains the gated `nvidia/Cosmos-Reason2-2B` blobs (~4.6 GB,
# pulled when that setup did its `hf auth login` round). This means the
# gr00t-n17 server starts cleanly even without a fresh `hf auth login`
# on this machine. Override with HF_HOME=/some/other/path if you'd rather
# keep caches separated.
#
# Note: --model-path is absolute -> the gr00t-n17 finetune weights are
# loaded directly from disk regardless of HF_HOME; only the backbone
# (Cosmos-Reason2-2B) is fetched via HF.
DEFAULT_HF_HOME="$EVAL_YAM_DIR/../grootn1.7 exploration/hf-cache"
COSMOS_CACHED=0
if [[ -d "$DEFAULT_HF_HOME/models--nvidia--Cosmos-Reason2-2B" ]]; then
    COSMOS_CACHED=1
else
    DEFAULT_HF_HOME="$EVAL_YAM_DIR/hf-cache"
    # Check the new location too.
    if [[ -d "$DEFAULT_HF_HOME/hub/models--nvidia--Cosmos-Reason2-2B" ]] || \
       [[ -d "$DEFAULT_HF_HOME/models--nvidia--Cosmos-Reason2-2B" ]]; then
        COSMOS_CACHED=1
    fi
fi
export HF_HOME="${HF_HOME:-$DEFAULT_HF_HOME}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# CRITICAL: even when the Cosmos-Reason2-2B blobs are cached locally,
# transformers.cached_file() still pings the HF API to verify access,
# which fails with 401 on a gated repo unless logged in. Setting
# HF_HUB_OFFLINE=1 forces transformers to use the local cache without an
# API check. We only force offline mode when we've confirmed the blobs
# are present, so a fresh box with `hf auth login` done still resolves
# remote downloads.
if [[ "$COSMOS_CACHED" == "1" && -z "${HF_HUB_OFFLINE:-}" ]]; then
    echo "[run_server_gr00t] cosmos blobs cached at $HF_HOME -- forcing HF_HUB_OFFLINE=1"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

cd "$GR00T_DIR"

echo "[run_server_gr00t] model_path: $CKPT_DIR"
echo "[run_server_gr00t] port:       $PORT"
echo "[run_server_gr00t] device:     $DEVICE"
echo "[run_server_gr00t] modality config: $HERE/yam_config.py"

# Make the offline_shim.py importable as a side-effect via PYTHONSTARTUP.
# This bypasses transformers' is_base_mistral() HF API call when running
# offline with cached weights -- see offline_shim.py for the full story.
if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
    export PYTHONSTARTUP="$HERE/offline_shim.py"
    echo "[run_server_gr00t] offline mode: PYTHONSTARTUP=$PYTHONSTARTUP"
fi

# uv run launches python with PYTHONSTARTUP honored. But run_gr00t_server.py
# is invoked as a module entry point, and PYTHONSTARTUP only fires for
# interactive shells. Instead, prepend a `python -c` import of the shim
# before exec'ing into the server -- guarantees the patch is applied
# before transformers imports huggingface_hub.
exec uv run python -c "
import sys, runpy
sys.path.insert(0, '$HERE')
import offline_shim  # noqa: F401 -- side-effect: monkey-patches model_info if HF_HUB_OFFLINE=1
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
