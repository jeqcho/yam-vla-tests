#!/usr/bin/env bash
# Start the MolmoAct2-BimanualYAM RTC inference server on port 8203.
# Sister script to scripts/run_server.sh (which starts the non-RTC :8202
# server). The two servers can run simultaneously; they share GPU but each
# loads its own copy of the model weights -- expect ~26 GB of VRAM if you
# run both at once.

set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-$SETUP_DIR/hf-cache}"

# Use the same venv as the main :8202 server -- both need torch + transformers.
# The lerobot fork is installed there alongside (see requirements.txt).
VENV="$SETUP_DIR/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    echo "venv not found at $VENV. Bootstrap with:" >&2
    echo "  cd $SETUP_DIR && uv sync" >&2
    exit 2
fi

# Verify lerobot is importable; warn the user with the install command if not.
if ! "$VENV/bin/python" -c "import lerobot" 2>/dev/null; then
    cat >&2 <<EOF
ERROR: lerobot (Ai2 fork, molmoact2-policy branch) not found in:
  $VENV

Install with:
  VIRTUAL_ENV=$VENV uv pip install -r $SCRIPT_DIR/requirements.txt

Or directly:
  VIRTUAL_ENV=$VENV uv pip install \\
    'lerobot @ git+https://github.com/allenai/lerobot.git@molmoact2-policy'
EOF
    exit 2
fi

exec "$VENV/bin/python" "$SCRIPT_DIR/host_server_rtc.py" \
    --host 0.0.0.0 \
    --port 8203 \
    --dtype bfloat16 \
    "$@"
