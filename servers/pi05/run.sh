#!/usr/bin/env bash
# π₀.₅ — bimanual YAM finetune server.
# Wire: WebSocket + msgpack via openpi on :8000.
# Model: jeqcho/pi05-yam-bimanual (Orbax JAX).
# Backbone: openpi's scripts/serve_policy.py inside its own uv venv.
#
# Prereqs (one-time):
#   1. Clone + uv-sync openpi (~10 GB: JAX[cuda12], torch 2.7.1, etc.).
#      Set OPENPI_DIR=/abs/path/to/openpi (default: jeqcho's local
#      training fork at ~/Documents/openpi-agilex if available).
#   2. ./scripts/download_checkpoints.sh pi05
#   3. servers/pi05/register_yam_pi05.py must match the TrainConfig the
#      checkpoint was actually trained with. The template here uses
#      adapt_to_pi=False and use_delta_joint_actions=False (verified by
#      end-to-end bring-up). See the file's docstring for rationale.
#
# VRAM: ~14 GB bf16. CANNOT run concurrently with MolmoAct2 on a single 5090.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# Default: jeqcho's local training fork. Override via OPENPI_DIR=...
OPENPI_DIR="${OPENPI_DIR:-/home/andon/Documents/openpi-agilex}"
if [[ ! -d "$OPENPI_DIR" ]]; then
    echo "openpi clone not found at $OPENPI_DIR" >&2
    echo "  git clone https://github.com/Physical-Intelligence/openpi.git \"$OPENPI_DIR\"" >&2
    echo "  cd \"$OPENPI_DIR\" && uv sync" >&2
    echo "  Then set OPENPI_DIR=$OPENPI_DIR" >&2
    exit 1
fi
if [[ ! -x "$OPENPI_DIR/.venv/bin/python" ]]; then
    echo "openpi venv missing at $OPENPI_DIR/.venv/" >&2
    echo "  Run: cd \"$OPENPI_DIR\" && uv sync" >&2
    exit 1
fi

# Pi-0.5 checkpoint fallback chain. The consolidation typically skips
# the multi-GB hf-cache, so the checkpoint usually still lives at the
# original eval-yam location.
_CKPT_CANDIDATES=(
    "${CKPT_DIR:-}"
    "$REPO_ROOT/hf-cache/checkpoints/jeqcho_pi05-yam-bimanual"
    "$HOME/yam-tests/eval-yam/hf-cache/checkpoints/jeqcho_pi05-yam-bimanual"
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
    echo "Could not find the pi-0.5 checkpoint directory." >&2
    echo "  Tried:" >&2
    for cand in "${_CKPT_CANDIDATES[@]}"; do
        [[ -n "$cand" ]] && echo "    $cand" >&2
    done
    echo "  Run: $REPO_ROOT/scripts/download_checkpoints.sh pi05" >&2
    exit 1
fi
echo "[servers/pi05] CKPT_DIR=$CKPT_DIR" >&2

PORT="${PORT:-8000}"

# HF_HOME follows the same fallback as the checkpoint (sibling dirs).
_HFHOME_CANDIDATES=(
    "${HF_HOME:-}"
    "$REPO_ROOT/hf-cache"
    "$HOME/yam-tests/eval-yam/hf-cache"
)
_RESOLVED_HF=""
for cand in "${_HFHOME_CANDIDATES[@]}"; do
    [[ -z "$cand" ]] && continue
    if [[ -d "$cand" ]]; then
        _RESOLVED_HF="$cand"
        break
    fi
done
export HF_HOME="${_RESOLVED_HF:-$REPO_ROOT/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

export PYTHONPATH="$HERE:${PYTHONPATH:-}"

cd "$OPENPI_DIR"

echo "[servers/pi05] openpi:    $OPENPI_DIR"
echo "[servers/pi05] ckpt_dir:  $CKPT_DIR"
echo "[servers/pi05] port:      $PORT"
echo "[servers/pi05] registering yam_pi05 config..."

# Register yam_pi05 BEFORE tyro parses serve_policy's flags.
exec uv run python -c "
import sys
sys.path.insert(0, '$HERE')
import register_yam_pi05  # side-effect: appends yam_pi05 to _CONFIGS
import runpy
sys.argv = [
    'serve_policy.py',
    '--port', '$PORT',
    'policy:checkpoint',
    '--policy.config', 'yam_pi05',
    '--policy.dir', '$CKPT_DIR',
] + sys.argv[1:]
runpy.run_path('scripts/serve_policy.py', run_name='__main__')
" "$@"
