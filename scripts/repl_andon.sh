#!/usr/bin/env bash
# One-liner launcher for the YAM+MolmoAct2 interactive REPL with this
# workstation's hardware config baked in. Run server first via
# ./scripts/run_server.sh, then this in another terminal.
set -euo pipefail
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$SETUP_DIR/scripts/run_repl.sh" \
    --left-can can0 --right-can can1 \
    --left-gripper linear_4310 --right-gripper linear_4310 \
    --top-cam-serial 349622072241 \
    --left-cam-serial 427622271914 \
    --right-cam-serial 352122272708 \
    --cam-width 640 --cam-height 360 \
    --horizon-stride 10 \
    --rerun --rerun-save /tmp/yam_repl.rrd \
    "$@"
