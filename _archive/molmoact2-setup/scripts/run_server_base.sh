#!/usr/bin/env bash
# Start the host_server_yam.py against the *vanilla* MolmoAct2 foundation
# checkpoint instead of the fine-tuned BimanualYAM. Wire schema is identical
# (same norm_tag "yam_dual_molmoact2", same state_dim=14, same 3-camera order)
# because vanilla MolmoAct2's norm_stats.json ships metadata for all 7
# embodiments including yam_dual_molmoact2.
#
# Vanilla MolmoAct2 is a "post-trained foundation checkpoint, intended for
# fine-tuning" — out-of-box deployment performance will be worse than the
# BimanualYAM checkpoint, but the server starts cleanly and accepts the same
# /act requests so it's worth comparing on identical scenes.
#
# Prereq: download the weights once (~22 GB):
#   cd /home/andon/yam-tests/molmoact2-setup
#   export HF_HUB_ENABLE_HF_TRANSFER=1
#   export HF_HOME="$PWD/hf-cache"
#   uv run hf download allenai/MolmoAct2
#
# Then kill the BimanualYAM server (only one server can hold the GPU at a
# time on a 32 GB card) and start this one — defaults to port 8302 so you
# can distinguish them in logs and curl probes.
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SETUP_DIR/molmoact2"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-$SETUP_DIR/hf-cache}"

exec uv run --project "$SETUP_DIR" python examples/yam/host_server_yam.py \
    --host 0.0.0.0 \
    --port 8302 \
    --repo-id allenai/MolmoAct2 \
    --dtype bfloat16 \
    --cuda-graph \
    "$@"
