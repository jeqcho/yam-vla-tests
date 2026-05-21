#!/usr/bin/env bash
# Bring up the DreamZero WebSocket server on Modal.
#
# Usage:
#   ./scripts/run_modal_server.sh droid         # public DROID checkpoint (default)
#   ./scripts/run_modal_server.sh yam           # YAM-finetuned (requires a checkpoint!)
#
# `modal serve` streams logs to your terminal until you Ctrl-C; it tears the
# container down on exit. Use `modal deploy` instead to keep it running.
#
# The URL appears in the banner output. Copy it into smoke_test_remote.py:
#   uv run python scripts/smoke_test_remote.py --url wss://<…>/ --schema droid
set -euo pipefail

MODEL="${1:-droid}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

if [ ! -d dreamzero ]; then
  echo "Cloning dreamzero repo (needed for Modal image build)…"
  git clone --depth 1 https://github.com/dreamzero0/dreamzero.git dreamzero
fi

MODEL="$MODEL" exec modal serve modal/dreamzero_server.py
