#!/usr/bin/env bash
# run_server_vanilla.sh — launch the *vanilla* GR00T N1.7 base model with the
# closest pretrain embodiment tag (XDOF). Useful as a side-by-side baseline
# vs. a YAM-finetuned checkpoint.
#
# IMPORTANT: the base GR00T-N1.7-3B has no YAM-specific embodiment head. Using
# XDOF runs the generic "X-DOF relative-EEF + relative-joint" head, which was
# pretrained on a different action layout and dim. You will get well-formed
# numeric output, but the actions are unlikely to make sense on YAM hardware
# without finetuning. This is here so you can prove the plumbing works and
# compare A/B against your finetune.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
GR00T_DIR="$REPO_ROOT/Isaac-GR00T"

export HF_HOME="$REPO_ROOT/hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

cd "$GR00T_DIR"

# Port 5555 is the GR00T default — using it here so vanilla and the YAM
# finetune (5556) can coexist if you have enough VRAM (you almost certainly
# won't on a single RTX 5090; each holds ~7 GB bf16 + activations).
exec uv run python gr00t/eval/run_gr00t_server.py \
    --model-path "${MODEL_PATH:-nvidia/GR00T-N1.7-3B}" \
    --embodiment-tag "${EMBODIMENT_TAG:-XDOF}" \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port "${PORT:-5555}" \
    "$@"
