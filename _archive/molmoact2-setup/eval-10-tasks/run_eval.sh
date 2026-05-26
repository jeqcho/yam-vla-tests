#!/usr/bin/env bash
# Andon 10-task eval for MolmoAct2 on bimanual YAM.
# Setup once, then 10 tasks x N attempts with human reset between attempts.
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="$(cd "$EVAL_DIR/.." && pwd)"
I2RT_DIR="$SETUP_DIR/../i2rt"
PYTHON="$I2RT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "i2rt venv python not found at $PYTHON" >&2
    exit 1
fi

INVOCATION_ARGS=$(printf ' %q' "$@")
export YAM_INVOCATION="${BASH_SOURCE[0]}${INVOCATION_ARGS}"

exec "$PYTHON" "$EVAL_DIR/eval_andon_tasks.py" "$@"
