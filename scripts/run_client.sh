#!/usr/bin/env bash
# run_client.sh — drive yam_client.py with the i2rt venv (which has the i2rt
# SDK + pyrealsense2 + zmq/msgpack-numpy installed).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

export YAM_INVOCATION="$0 $*"

# Default to the same orange-cube instruction used by the MolmoAct2 client.
INSTRUCTION_DEFAULT="first pick up the left orange cube and put it in the box, then pick up the right orange cube and put it in the box"

# Bridge --instruction defaults if the user didn't pass one.
has_instruction=0
for arg in "$@"; do
    if [[ "$arg" == "--instruction" || "$arg" == "--instruction="* ]]; then
        has_instruction=1; break
    fi
done

if (( has_instruction == 0 )); then
    set -- "$@" --instruction "$INSTRUCTION_DEFAULT"
fi

exec /home/andon/yam-tests/i2rt/.venv/bin/python \
    "$HERE/yam_client.py" "$@"
