#!/usr/bin/env bash
# Andon 10-task eval against MolmoAct2-BimanualYAM (HTTP+json_numpy on :8202).
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_YAM_DIR="$(cd "$EVAL_DIR/.." && pwd)"
I2RT_DIR="$EVAL_YAM_DIR/../i2rt"
PYTHON="$I2RT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "i2rt venv python not found at $PYTHON" >&2
    exit 1
fi

INVOCATION_ARGS=$(printf ' %q' "$@")
export YAM_INVOCATION="${BASH_SOURCE[0]}${INVOCATION_ARGS}"

# Policy-specific stride default; see run_repl_molmoact2.sh for rationale.
exec "$PYTHON" "$EVAL_DIR/eval_yam_tasks.py" \
    --policy molmoact2 \
    --server-url "${YAM_SERVER_URL:-http://127.0.0.1:8202/act}" \
    --horizon-stride "${YAM_HORIZON_STRIDE:-6}" \
    "$@"
