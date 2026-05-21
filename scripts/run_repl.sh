#!/usr/bin/env bash
# Interactive task REPL for bimanual YAM + MolmoAct2.
# Setup once, then loop: type instruction, run, press enter to reset, journal.
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I2RT_DIR="$SETUP_DIR/../i2rt"
PYTHON="$I2RT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "i2rt venv python not found at $PYTHON" >&2
    exit 1
fi

# Preserve invocation for the journal.
INVOCATION_ARGS=$(printf ' %q' "$@")
export YAM_INVOCATION="${BASH_SOURCE[0]}${INVOCATION_ARGS}"

exec "$PYTHON" "$SETUP_DIR/scripts/yam_repl.py" "$@"
