#!/usr/bin/env bash
# IKEA-10 furniture eval against MolmoAct2-BimanualYAM (HTTP+json_numpy :8202).
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IKEA_DIR="$(cd "$EVAL_DIR/.." && pwd)"
I2RT_DIR="$IKEA_DIR/../i2rt"
PYTHON="$I2RT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "i2rt venv python not found at $PYTHON" >&2
    exit 1
fi

INVOCATION_ARGS=$(printf ' %q' "$@")
export YAM_INVOCATION="${BASH_SOURCE[0]}${INVOCATION_ARGS}"

# Same per-policy stride default as eval-yam's molmoact2 wrapper.
exec "$PYTHON" "$EVAL_DIR/eval_ikea_tasks.py" \
    --policy molmoact2 \
    --server-url "${YAM_SERVER_URL:-http://127.0.0.1:8202/act}" \
    --horizon-stride "${YAM_HORIZON_STRIDE:-6}" \
    "$@"
