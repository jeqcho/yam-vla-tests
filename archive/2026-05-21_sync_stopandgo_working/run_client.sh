#!/usr/bin/env bash
# Start the bimanual-YAM client that talks to the MolmoAct2 server.
# Uses the i2rt venv since it has the i2rt SDK installed.
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I2RT_DIR="$SETUP_DIR/../i2rt"
PYTHON="$I2RT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "i2rt venv python not found at $PYTHON" >&2
    echo "Run 'cd $I2RT_DIR && uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .' first." >&2
    exit 1
fi

# Preserve the user's original shell invocation so yam_client.py can record
# it in the research journal. Without this, the journal would show the
# python-level argv ("scripts/yam_client.py --left-can ..."), which is
# correct but harder to recognize. Use printf %q to keep quoting reproducible.
INVOCATION_ARGS=$(printf ' %q' "$@")
export YAM_INVOCATION="${BASH_SOURCE[0]}${INVOCATION_ARGS}"

exec "$PYTHON" "$SETUP_DIR/scripts/yam_client.py" \
    --instruction "first pick up the left orange cube and put it in the box, then pick up the right orange cube and put it in the box" \
    "$@"
