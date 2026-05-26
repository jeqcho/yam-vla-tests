#!/usr/bin/env bash
# Interactive REPL against the Pi-0.5 yam_pi05 server (WebSocket+msgpack).
# Server: run_server_pi05.sh on :8000.
#
# NOTE: requires `openpi-client` installed in the i2rt venv (one-time):
#   VIRTUAL_ENV=/home/andon/yam-tests/i2rt/.venv uv pip install \
#     'openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client'
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

# Policy-specific stride default (=16% of pi05's 50-step chunk =
# ~267 ms open-loop window at 30 Hz). pi05 has the longest horizon of
# the three policies, so a smaller % stride still gives a longer
# absolute motion window. User-passed --horizon-stride wins via
# argparse last-occurrence-wins. Env override:
#   YAM_HORIZON_STRIDE=12 ./run_repl_pi05.sh
exec "$PYTHON" "$EVAL_YAM_DIR/scripts/repl_yam.py" \
    --policy pi05 \
    --server-host "${YAM_SERVER_HOST:-127.0.0.1}" \
    --server-port "${YAM_SERVER_PORT:-8000}" \
    --horizon-stride "${YAM_HORIZON_STRIDE:-8}" \
    "$@"
