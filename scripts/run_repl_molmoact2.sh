#!/usr/bin/env bash
# Interactive REPL against the MolmoAct2-BimanualYAM server (HTTP+json_numpy).
# Server: run_server_molmoact2.sh on :8202.
set -euo pipefail

EVAL_YAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I2RT_DIR="$EVAL_YAM_DIR/../i2rt"
PYTHON="$I2RT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "i2rt venv python not found at $PYTHON" >&2
    exit 1
fi

INVOCATION_ARGS=$(printf ' %q' "$@")
export YAM_INVOCATION="${BASH_SOURCE[0]}${INVOCATION_ARGS}"

# Policy-specific stride default (=20% of MolmoAct2's 30-step chunk =
# ~200 ms open-loop window at 30 Hz). User-passed --horizon-stride wins
# via argparse last-occurrence-wins behavior. Override via env:
#   YAM_HORIZON_STRIDE=10 ./run_repl_molmoact2.sh
exec "$PYTHON" "$EVAL_YAM_DIR/scripts/repl_yam.py" \
    --policy molmoact2 \
    --server-url "${YAM_SERVER_URL:-http://127.0.0.1:8202/act}" \
    --horizon-stride "${YAM_HORIZON_STRIDE:-6}" \
    "$@"
