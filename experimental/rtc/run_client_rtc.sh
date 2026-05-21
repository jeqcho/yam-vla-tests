#!/usr/bin/env bash
# Start the bimanual-YAM RTC client that talks to host_server_rtc.py on :8203.
# Sister to scripts/run_client.sh. Uses the i2rt venv since the client needs
# the i2rt SDK to drive the arms.
#
# The defaults below preserve the SAME instruction and SAME camera serials
# the legacy client uses so an A/B benchmark requires only swapping which
# runner you invoke.
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
I2RT_DIR="$SETUP_DIR/../i2rt"
PYTHON="$I2RT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "i2rt venv python not found at $PYTHON" >&2
    echo "Bootstrap with: cd $I2RT_DIR && uv venv --python 3.11 && uv pip install -e ." >&2
    exit 1
fi

# Preserve the original shell invocation so the journal entry shows the
# exact CLI the user typed (not the python-level argv).
INVOCATION_ARGS=$(printf ' %q' "$@")
export YAM_INVOCATION="${BASH_SOURCE[0]}${INVOCATION_ARGS}"

exec "$PYTHON" "$SCRIPT_DIR/yam_client_rtc.py" \
    --server-url "http://127.0.0.1:8203/act" \
    --instruction "first pick up the left orange cube and put it in the box, then pick up the right orange cube and put it in the box" \
    "$@"
