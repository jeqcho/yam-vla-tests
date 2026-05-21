#!/usr/bin/env bash
# Pull the data-prep deliverables from the Modal volume into ./hf-cache/
# so the user can eyeball them locally before greenlighting the fine-tune.
#
# Usage:
#   ./scripts/inspect_prep.sh                  # tag=yam_box_smoke (default)
#   ./scripts/inspect_prep.sh yam_box_full     # other tag
set -euo pipefail

TAG="${1:-yam_box_smoke}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/../hf-cache/prep_${TAG}"
mkdir -p "$DEST"
cd "$DEST"

echo "Pulling artifacts for tag=${TAG} into $DEST ..."
modal volume get --force dreamzero-yam-data "prepared/${TAG}_prep_report.md" .  || true
modal volume get --force -r dreamzero-yam-data "prepared/sample_frames" .       || true
modal volume get --force dreamzero-yam-data "prepared/${TAG}/meta/modality.json" .   || true
modal volume get --force dreamzero-yam-data "prepared/${TAG}/meta/embodiment.json" . || true
modal volume get --force dreamzero-yam-data "prepared/${TAG}/meta/stats.json" .      || true

echo
echo "Pulled:"
ls -la "$DEST" 2>/dev/null

echo
echo "=========================================================="
echo "Inspect these before greenlighting the fine-tune:"
echo
echo "  cat $DEST/${TAG}_prep_report.md"
echo "  jq '{state:.state,action:.action,video:.video,annotation:.annotation}' $DEST/modality.json"
echo "  cat $DEST/embodiment.json"
echo "  ls $DEST/sample_frames/"
echo
echo "If the checklist in the report shows all [x], run:"
echo "  ./scripts/run_finetune_modal.sh ${TAG} dz-yam-smoke 200    # ~\$10 smoke run"
echo "=========================================================="
