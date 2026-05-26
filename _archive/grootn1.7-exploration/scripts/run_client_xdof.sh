#!/usr/bin/env bash
# run_client_xdof.sh — drive yam_client_xdof.py with the i2rt venv.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

export YAM_INVOCATION="$0 $*"

INSTRUCTION_DEFAULT="Pick up an orange cube and place it in the box."

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
    "$HERE/yam_client_xdof.py" "$@"
