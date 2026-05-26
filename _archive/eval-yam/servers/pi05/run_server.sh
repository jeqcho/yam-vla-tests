#!/usr/bin/env bash
# Launch the Pi-0.5 inference server for `jeqcho/pi05-yam-bimanual` on
# port 8000 (WebSocket + msgpack via openpi).
#
# Server: openpi/scripts/serve_policy.py
# Model:  jeqcho/pi05-yam-bimanual (Orbax JAX checkpoint, action_horizon=16)
# Wire:   WebSocket + msgpack-numpy (openpi vendored codec), :8000
#
# Prereqs (one-time -- this script does NOT install them):
#   1. Clone openpi:
#        git clone https://github.com/Physical-Intelligence/openpi.git
#        cd openpi && uv sync   # ~10 GB: JAX[cuda12], torch 2.7.1, etc.
#      (If you have a fork with the upstream yam_pi05 config, prefer that.)
#      Set OPENPI_DIR=/path/to/openpi (default: ../openpi alongside this dir).
#
#   2. Download the checkpoint into hf-cache/:
#        ../../download_checkpoints.sh pi05
#      Set CKPT_DIR=/abs/path/to/pi05-yam-bimanual to override.
#
#   3. Make sure register_yam_pi05.py here matches the EXACT TrainConfig used
#      to train the checkpoint (image keys, adapt_to_pi, etc.). See the
#      docstring at the top of register_yam_pi05.py.
#
# Server requires ~14 GB VRAM in bf16. Sharing the 5090 with the MolmoAct2
# server is not possible (32 GB total, MolmoAct2 takes ~21 GB).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_YAM_DIR="$(cd "$HERE/../.." && pwd)"

# Default to the existing openpi-agilex clone (already uv-synced, has JAX
# 0.5.3 on CUDA, python 3.12). Override with OPENPI_DIR=/elsewhere -- e.g.
# point at jeqcho's training fork once it lands on this box, so the real
# yam_pi05 TrainConfig is in _CONFIGS without needing register_yam_pi05.py.
OPENPI_DIR="${OPENPI_DIR:-/home/andon/Documents/openpi-agilex}"
if [[ ! -d "$OPENPI_DIR" ]]; then
    echo "openpi clone not found at $OPENPI_DIR" >&2
    echo "  git clone https://github.com/Physical-Intelligence/openpi.git \"$OPENPI_DIR\"" >&2
    echo "  cd \"$OPENPI_DIR\" && uv sync" >&2
    echo "Or set OPENPI_DIR=/path/to/openpi." >&2
    exit 1
fi
if [[ ! -x "$OPENPI_DIR/.venv/bin/python" ]]; then
    echo "openpi venv not found at $OPENPI_DIR/.venv/" >&2
    echo "  cd \"$OPENPI_DIR\" && uv sync" >&2
    exit 1
fi

CKPT_DIR="${CKPT_DIR:-$EVAL_YAM_DIR/hf-cache/checkpoints/jeqcho_pi05-yam-bimanual}"
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Checkpoint dir not found: $CKPT_DIR" >&2
    echo "  Download with: $EVAL_YAM_DIR/download_checkpoints.sh pi05" >&2
    exit 1
fi

PORT="${PORT:-8000}"

# Local HF cache.
export HF_HOME="${HF_HOME:-$EVAL_YAM_DIR/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# Make sure register_yam_pi05.py is importable so the side-effect runs.
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

cd "$OPENPI_DIR"

echo "[run_server_pi05] openpi:    $OPENPI_DIR"
echo "[run_server_pi05] ckpt_dir:  $CKPT_DIR"
echo "[run_server_pi05] port:      $PORT"
echo "[run_server_pi05] registering yam_pi05 config..."

# Register the yam_pi05 config BEFORE serve_policy parses its config flag.
# We do this as a small wrapper python invocation that imports the shim,
# then re-execs into serve_policy with the same args. Using -m so import
# resolves through PYTHONPATH.
exec uv run python -c "
import sys
sys.path.insert(0, '$HERE')
import register_yam_pi05  # side-effect: appends yam_pi05 to _CONFIGS
import runpy
# tyro CLI structure: top-level flags (--port, --env, ...) precede the
# subcommand 'policy:checkpoint', then subcommand-scoped flags follow.
# Mixing them produces \`Unrecognized options\` errors.
sys.argv = [
    'serve_policy.py',
    '--port', '$PORT',
    'policy:checkpoint',
    '--policy.config', 'yam_pi05',
    '--policy.dir', '$CKPT_DIR',
] + sys.argv[1:]
runpy.run_path('scripts/serve_policy.py', run_name='__main__')
" "$@"
