#!/usr/bin/env bash
# Top-level dispatcher: bring up a VLA inference server.
#
# Usage:
#   ./scripts/run_server.sh <policy> [args...]
#
#   policy  one of:  molmoact2  gr00t-n17  pi05
#
# Delegates to servers/<policy>/run.sh. Extra args after the policy name
# are forwarded.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

if [[ $# -lt 1 ]]; then
    cat >&2 <<EOF
Usage: $(basename "$0") <policy> [args...]

policies: molmoact2  gr00t-n17  pi05

Each delegates to servers/<policy>/run.sh. Server-side prereqs and env
vars are documented in those scripts.
EOF
    exit 2
fi

policy="$1"; shift
runner="$REPO_ROOT/servers/$policy/run.sh"
if [[ ! -x "$runner" ]]; then
    echo "Unknown or non-executable policy '$policy'" >&2
    echo "Expected: $runner" >&2
    echo "Known: $(ls -1 "$REPO_ROOT/servers" | tr '\n' ' ')" >&2
    exit 2
fi

exec "$runner" "$@"
