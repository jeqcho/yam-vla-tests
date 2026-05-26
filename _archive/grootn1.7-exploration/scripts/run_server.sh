#!/usr/bin/env bash
# run_server.sh — launch the GR00T N1.7 inference server for bimanual YAM.
#
# Defaults to the YAM-finetuned checkpoint under hf-cache/checkpoints/yam if it
# exists; falls back to the base model with --embodiment-tag NEW_EMBODIMENT (no
# YAM weights, just useful for plumbing tests). Override with --model-path.
#
# Server speaks the GR00T PolicyServer wire protocol over ZeroMQ on port 5556
# (port 5555 is the upstream default — using 5556 lets you keep MolmoAct2 on
# 8202 and a separate stock-default GR00T server on 5555 if you want).
#
# Tasks performed:
#   - sets HF_HOME and TRANSFORMERS_CACHE to the local hf-cache/ so downloaded
#     weights don't blow out ~/.cache/huggingface
#   - registers the bimanual YAM modality config (scripts/yam_config.py)
#   - launches gr00t/eval/run_gr00t_server.py
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
GR00T_DIR="$REPO_ROOT/Isaac-GR00T"

# Local HF cache so weights live in the project tree, not ~/.cache/huggingface.
export HF_HOME="$REPO_ROOT/hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# Default to YAM finetune dir if present; else base model. CLI args win.
DEFAULT_FINETUNE_DIR="$REPO_ROOT/hf-cache/checkpoints/yam-latest"
if [[ -d "$DEFAULT_FINETUNE_DIR" ]]; then
    DEFAULT_MODEL_PATH="$DEFAULT_FINETUNE_DIR"
    DEFAULT_TAG="NEW_EMBODIMENT"
    echo "[run_server] Found local finetune at $DEFAULT_FINETUNE_DIR — using it."
else
    DEFAULT_MODEL_PATH="nvidia/GR00T-N1.7-3B"
    DEFAULT_TAG="NEW_EMBODIMENT"
    echo "[run_server] No local finetune; falling back to base model nvidia/GR00T-N1.7-3B."
    echo "[run_server] WARNING: base model has no YAM-specific weights. Inference will run,"
    echo "[run_server]          but actions won't be meaningful until you finetune."
fi

cd "$GR00T_DIR"

exec uv run python gr00t/eval/run_gr00t_server.py \
    --model-path "${MODEL_PATH:-$DEFAULT_MODEL_PATH}" \
    --embodiment-tag "${EMBODIMENT_TAG:-$DEFAULT_TAG}" \
    --modality-config-path "$HERE/yam_config.py" \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port "${PORT:-5556}" \
    "$@"
