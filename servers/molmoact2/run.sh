#!/usr/bin/env bash
# MolmoAct2-BimanualYAM server. HTTP + json_numpy on :8202.
# Delegates to the legacy molmoact2-setup tree (still under _archive/)
# because the server code + uv-venv + bf16 patches all live there.
#
# Override the legacy dir via MOLMOACT2_DIR=/path/to/molmoact2-setup.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

MOLMOACT2_DIR="${MOLMOACT2_DIR:-$REPO_ROOT/_archive/molmoact2-setup}"
if [[ ! -x "$MOLMOACT2_DIR/scripts/run_server.sh" ]]; then
    echo "molmoact2-setup not found at $MOLMOACT2_DIR" >&2
    echo "  set MOLMOACT2_DIR=/abs/path/to/molmoact2-setup" >&2
    exit 1
fi

exec "$MOLMOACT2_DIR/scripts/run_server.sh" "$@"
