#!/usr/bin/env bash
# Download VLA checkpoints for eval-yam into hf-cache/checkpoints/<name>/.
#
# Usage:
#   ./download_checkpoints.sh                  # all 3 (~50 GB)
#   ./download_checkpoints.sh molmoact2        # just one
#   ./download_checkpoints.sh pi05 gr00t-n17   # two
#
# Repos:
#   molmoact2  ->  allenai/MolmoAct2-BimanualYAM       (~21 GB,  bf16 transformers)
#   pi05       ->  jeqcho/pi05-yam-bimanual            (~12 GB,  Orbax JAX)
#   gr00t-n17  ->  jeqcho/gr00t-n17-yam-bimanual       (~6 GB,   bf16 transformers)
#
# Each download lands in eval-yam/hf-cache/checkpoints/<repo_under_dirs>/.
# The matching run_server_*.sh scripts default CKPT_DIR to these paths.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$HERE/hf-cache"
CKPT_DIR="$CACHE_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

# Prefer hf_transfer for big checkpoints.
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="$CACHE_DIR"

# Find a python with `hf` CLI. The molmoact2-setup venv has huggingface_hub
# installed; reuse it. Fall back to PATH.
HF_BIN=""
for cand in \
    "$HERE/../molmoact2-setup/.venv/bin/hf" \
    "$HERE/../i2rt/.venv/bin/hf" \
    "$(command -v hf 2>/dev/null || true)" \
    "$(command -v huggingface-cli 2>/dev/null || true)"; do
    if [[ -n "$cand" && -x "$cand" ]]; then HF_BIN="$cand"; break; fi
done
if [[ -z "$HF_BIN" ]]; then
    echo "No 'hf' or 'huggingface-cli' CLI on PATH. Install with:" >&2
    echo "  VIRTUAL_ENV=\"$HERE/../i2rt/.venv\" uv pip install 'huggingface_hub[cli,hf_transfer]'" >&2
    exit 1
fi
echo "[download] using HF CLI at $HF_BIN"

declare -A REPOS=(
    [molmoact2]="allenai/MolmoAct2-BimanualYAM"
    [pi05]="jeqcho/pi05-yam-bimanual"
    [gr00t-n17]="jeqcho/gr00t-n17-yam-bimanual"
)

# Local subdir name from repo id (slash -> underscore).
_local_dir_for() {
    local repo="$1"
    echo "$CKPT_DIR/${repo//\//_}"
}

_download() {
    local key="$1"
    local repo="${REPOS[$key]:-}"
    if [[ -z "$repo" ]]; then
        echo "Unknown checkpoint key '$key' (known: ${!REPOS[*]})" >&2
        return 2
    fi
    local dst
    dst="$(_local_dir_for "$repo")"
    echo "[download] $key  ->  $repo  ->  $dst"
    mkdir -p "$dst"
    # hf-style positional: `hf download <repo> --local-dir <dst>`. CLI is
    # idempotent -- re-runs verify cached files.
    #
    # pi05's HF repo also ships ``train_state/`` (16 GB Orbax optimizer
    # state) which is only useful for resuming training. serve_policy.py
    # loads only ``params/`` + ``assets/`` for inference. Exclude it.
    local -a extra_args=()
    if [[ "$key" == "pi05" ]]; then
        extra_args+=(--exclude "train_state/*")
    fi
    "$HF_BIN" download "$repo" --local-dir "$dst" "${extra_args[@]}"
}

if [[ $# -eq 0 ]]; then
    keys=(molmoact2 pi05 gr00t-n17)
else
    keys=("$@")
fi

for k in "${keys[@]}"; do
    _download "$k"
done

echo
echo "[download] done. Checkpoints land in:"
for k in "${keys[@]}"; do
    echo "  $k  ->  $(_local_dir_for "${REPOS[$k]}")"
done
