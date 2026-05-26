#!/usr/bin/env bash
# Start the MolmoAct2-BimanualYAM inference server.
# Defaults match the values used by examples/yam/host_server_yam.py.
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SETUP_DIR/molmoact2"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-$SETUP_DIR/hf-cache}"

# Use the setup-dir venv (where torch + transformers are pinned).
exec uv run --project "$SETUP_DIR" python examples/yam/host_server_yam.py \
    --host 0.0.0.0 \
    --port 8202 \
    --dtype bfloat16 \
    --cuda-graph \
    "$@"
