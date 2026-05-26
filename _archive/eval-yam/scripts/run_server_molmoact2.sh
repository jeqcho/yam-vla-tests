#!/usr/bin/env bash
# Launch the MolmoAct2-BimanualYAM inference server.
# Server: molmoact2-setup/molmoact2/examples/yam/host_server_yam.py on port 8202.
# Model:  allenai/MolmoAct2-BimanualYAM (HF transformers, bf16)
# Wire:   HTTP POST /act + json_numpy, http://0.0.0.0:8202/act
#
# Implementation note: just delegates to molmoact2-setup's run_server.sh so
# there is exactly ONE place where the MolmoAct2 server is launched.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_YAM_DIR="$(cd "$HERE/.." && pwd)"
MOLMOACT_DIR="$EVAL_YAM_DIR/../molmoact2-setup"

if [[ ! -x "$MOLMOACT_DIR/scripts/run_server.sh" ]]; then
    echo "molmoact2-setup/scripts/run_server.sh not found at $MOLMOACT_DIR" >&2
    exit 1
fi

exec "$MOLMOACT_DIR/scripts/run_server.sh" "$@"
