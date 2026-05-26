#!/usr/bin/env bash
# Kick off DreamZero YAM-bimanual fine-tuning on Modal.
#
# Usage:
#   ./scripts/run_finetune_modal.sh <hf-dataset-id> [run-name] [max-steps]
#
# Prerequisites:
#   1. A LeRobot v2 YAM bimanual dataset on HuggingFace with the DreamZero
#      modality schema. See REPORT_dreamzero_setup.md → "Preparing YAM data".
#   2. `modal token set` already configured (this account holds the credits).
#   3. Optional: `modal secret create wandb WANDB_API_KEY=...` for logging.
#
# Cost: ~$10–$15/hr on 4×H100. 100k steps ≈ 8–14 hr depending on data volume.
# Override --max-steps for a short smoke run (e.g. 200 steps validates the
# pipeline end-to-end for ~$5).
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <hf-dataset-id> [run-name] [max-steps]" >&2
  exit 1
fi
DATASET="$1"
RUN_NAME="${2:-dreamzero_yam_lora_$(date +%Y%m%d_%H%M)}"
MAX_STEPS="${3:-100000}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
if [ ! -d dreamzero ]; then
  echo "Cloning dreamzero repo (needed for Modal image build)…"
  git clone --depth 1 https://github.com/dreamzero0/dreamzero.git dreamzero
fi

echo "Dataset:   $DATASET"
echo "Run name:  $RUN_NAME"
echo "Max steps: $MAX_STEPS"
echo

exec modal run modal/dreamzero_finetune.py::run \
    --dataset-hf-id "$DATASET" \
    --run-name "$RUN_NAME" \
    --max-steps "$MAX_STEPS"
