#!/usr/bin/env bash
# MolmoAct2-BimanualYAM server. HTTP + json_numpy on :8202.
#
# Resolution order for the upstream molmoact2-setup tree (which has the
# nested `molmoact2/` clone, the dedicated uv venv with bf16 patches,
# and the 41 GB hf-cache):
#
#   1. $MOLMOACT2_DIR if set
#   2. $REPO_ROOT/_archive/molmoact2-setup     (post-consolidation, if
#      the molmoact2/ subdir was actually copied — usually NOT, the
#      consolidation skips the upstream clone + hf-cache)
#   3. $HOME/yam-tests/molmoact2-setup         (pre-consolidation home)
#   4. error out with a clear message
#
# The first directory that has both scripts/run_server.sh AND the
# nested molmoact2/ clone wins.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

_is_valid_setup() {
    [[ -x "$1/scripts/run_server.sh" && -d "$1/molmoact2" ]]
}

if [[ -n "${MOLMOACT2_DIR:-}" ]]; then
    : # already explicitly set; trust it
elif _is_valid_setup "$REPO_ROOT/_archive/molmoact2-setup"; then
    MOLMOACT2_DIR="$REPO_ROOT/_archive/molmoact2-setup"
elif _is_valid_setup "$HOME/yam-tests/molmoact2-setup"; then
    MOLMOACT2_DIR="$HOME/yam-tests/molmoact2-setup"
else
    echo "Could not find a valid molmoact2-setup directory." >&2
    echo "  Tried: $REPO_ROOT/_archive/molmoact2-setup" >&2
    echo "         $HOME/yam-tests/molmoact2-setup" >&2
    echo "  A valid setup has both:" >&2
    echo "    - scripts/run_server.sh         (the launch script)" >&2
    echo "    - molmoact2/                    (the upstream Ai2 clone)" >&2
    echo "  Set MOLMOACT2_DIR=/abs/path to override." >&2
    exit 1
fi

if ! _is_valid_setup "$MOLMOACT2_DIR"; then
    echo "MOLMOACT2_DIR=$MOLMOACT2_DIR is missing required pieces:" >&2
    [[ -x "$MOLMOACT2_DIR/scripts/run_server.sh" ]] || \
        echo "  - missing scripts/run_server.sh" >&2
    [[ -d "$MOLMOACT2_DIR/molmoact2" ]] || \
        echo "  - missing molmoact2/ (the upstream Ai2 clone)" >&2
    exit 1
fi

echo "[run_server.sh] using MOLMOACT2_DIR=$MOLMOACT2_DIR" >&2
exec "$MOLMOACT2_DIR/scripts/run_server.sh" "$@"
