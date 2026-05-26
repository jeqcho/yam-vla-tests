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

# Resolution order for the Isaac-GR00T clone (the 15 GB venv plus the
# upstream repo). The consolidation typically skips heavy clones, so
# the actual install usually still lives at the original location. Note
# that original folder name contains a SPACE.
_GR00T_CANDIDATES=(
    "${GR00T_DIR:-}"
    "$REPO_ROOT/_archive/grootn1.7-exploration/Isaac-GR00T"
    "$HOME/yam-tests/grootn1.7 exploration/Isaac-GR00T"
)
GR00T_DIR=""
for cand in "${_GR00T_CANDIDATES[@]}"; do
    [[ -z "$cand" ]] && continue
    if [[ -d "$cand" && -x "$cand/.venv/bin/python" ]]; then
        GR00T_DIR="$cand"
        break
    fi
done
if [[ -z "$GR00T_DIR" ]]; then
    echo "Could not find a valid Isaac-GR00T clone (with .venv/)." >&2
    echo "  Tried:" >&2
    for cand in "${_GR00T_CANDIDATES[@]}"; do
        [[ -n "$cand" ]] && echo "    $cand" >&2
    done
    echo "  Either clone+sync it, or set GR00T_DIR=/abs/path." >&2
    exit 1
fi
echo "[servers/gr00t-n17] GR00T_DIR=$GR00T_DIR" >&2

# Same fallback chain for the checkpoint: the canonical hf-cache lives
# next to whichever Isaac-GR00T install was active when the checkpoint
# was downloaded.
_CKPT_CANDIDATES=(
    "${CKPT_DIR:-}"
    "$REPO_ROOT/hf-cache/checkpoints/jeqcho_gr00t-n17-yam-bimanual"
    "$HOME/yam-tests/eval-yam/hf-cache/checkpoints/jeqcho_gr00t-n17-yam-bimanual"
    "$HOME/yam-tests/grootn1.7 exploration/hf-cache/checkpoints/jeqcho_gr00t-n17-yam-bimanual"
)
CKPT_DIR=""
for cand in "${_CKPT_CANDIDATES[@]}"; do
    [[ -z "$cand" ]] && continue
    if [[ -d "$cand" ]]; then
        CKPT_DIR="$cand"
        break
    fi
done
if [[ -z "$CKPT_DIR" ]]; then
    echo "Could not find the gr00t-n17 checkpoint directory." >&2
    echo "  Tried:" >&2
    for cand in "${_CKPT_CANDIDATES[@]}"; do
        [[ -n "$cand" ]] && echo "    $cand" >&2
    done
    echo "  Run: $REPO_ROOT/scripts/download_checkpoints.sh gr00t-n17" >&2
    exit 1
fi
echo "[servers/gr00t-n17] CKPT_DIR=$CKPT_DIR" >&2

PORT="${PORT:-5556}"
DEVICE="${DEVICE:-cuda:0}"

# Find the HF cache holding the Cosmos backbone (gated repo — needed
# at policy-load time). Same fallback chain as the Isaac-GR00T clone.
_HFCACHE_CANDIDATES=(
    "${HF_HOME:-}"
    "$REPO_ROOT/_archive/grootn1.7-exploration/hf-cache"
    "$HOME/yam-tests/grootn1.7 exploration/hf-cache"
)
DEFAULT_HF_HOME=""
for cand in "${_HFCACHE_CANDIDATES[@]}"; do
    [[ -z "$cand" ]] && continue
    if [[ -d "$cand" ]]; then
        DEFAULT_HF_HOME="$cand"
        break
    fi
done
DEFAULT_HF_HOME="${DEFAULT_HF_HOME:-$REPO_ROOT/_archive/grootn1.7-exploration/hf-cache}"

COSMOS_CACHED=0
if [[ -d "$DEFAULT_HF_HOME/models--nvidia--Cosmos-Reason2-2B" ]] \
|| [[ -d "$DEFAULT_HF_HOME/hub/models--nvidia--Cosmos-Reason2-2B" ]]; then
    COSMOS_CACHED=1
fi
export HF_HOME="${HF_HOME:-$DEFAULT_HF_HOME}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"
echo "[servers/gr00t-n17] HF_HOME=$HF_HOME" >&2

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
