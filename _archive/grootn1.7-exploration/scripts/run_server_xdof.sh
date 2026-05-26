#!/usr/bin/env bash
# run_server_xdof.sh — ZERO-SHOT GR00T N1.7-3B server using the XDOF embodiment.
#
# The released base checkpoint was pretrained on AllenAI bimanual YAM data
# under the `xdof_relative_eef_relative_joint` tag (see the model's
# experiment_cfg/conf.yaml inside hf-cache/), so we can run inference on YAM
# without ANY finetuning. The XDOF modality is already baked into the
# checkpoint's processor_config.json — we do NOT pass --modality-config-path
# (which would over-register and shadow it).
#
# Listens on tcp://0.0.0.0:5557 (5555 = vanilla XDOF-no-config baseline,
# 5556 = post-finetune NEW_EMBODIMENT path). Three distinct ports so you can
# A/B them without restarting.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
GR00T_DIR="$REPO_ROOT/Isaac-GR00T"

export HF_HOME="$REPO_ROOT/hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# Load HF token from .env so we can pull the gated Cosmos-Reason2-2B
# preprocessor config (see HANDOFF.md Step 0). Falls back to whatever is
# already in the environment if .env is absent.
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
    : "${HF_TOKEN:=${HF_API_TOKEN:-}}"
    export HF_TOKEN
fi

cd "$GR00T_DIR"

exec uv run python gr00t/eval/run_gr00t_server.py \
    --model-path "${MODEL_PATH:-nvidia/GR00T-N1.7-3B}" \
    --embodiment-tag "${EMBODIMENT_TAG:-xdof_relative_eef_relative_joint}" \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port "${PORT:-5557}" \
    "$@"
