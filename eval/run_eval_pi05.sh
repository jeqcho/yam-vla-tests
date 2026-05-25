#!/usr/bin/env bash
# IKEA-10 furniture eval against Pi-0.5 yam_pi05 (WebSocket+msgpack :8000).
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

exec "$PYTHON" "$EVAL_DIR/eval_ikea_tasks.py" \
    --policy pi05 \
    --server-host "${YAM_SERVER_HOST:-127.0.0.1}" \
    --server-port "${YAM_SERVER_PORT:-8000}" \
    --horizon-stride "${YAM_HORIZON_STRIDE:-8}" \
    "$@"
