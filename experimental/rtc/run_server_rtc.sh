#!/usr/bin/env bash
# Start the MolmoAct2-BimanualYAM RTC inference server on port 8203.
# Sister script to scripts/run_server.sh (which starts the non-RTC :8202
# server). The two servers can run simultaneously; they share GPU but each
# loads its own copy of the model weights -- expect ~26 GB of VRAM if you
# run both at once.

set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-$SETUP_DIR/hf-cache}"

# The RTC server uses a SEPARATE venv (.venv-rtc) on Python 3.12 because the
# lerobot fork (molmoact2-policy branch) requires >=3.12 while the main
# server's .venv is pinned to 3.11. The two servers can run side-by-side --
# each loads its own copy of the model weights (~13 GB VRAM each at bf16).
VENV="$SETUP_DIR/.venv-rtc"
if [ ! -x "$VENV/bin/python" ]; then
    cat >&2 <<EOF
ERROR: RTC venv not found at $VENV.

Bootstrap with:
  cd $SETUP_DIR && uv venv --python 3.12 .venv-rtc
  VIRTUAL_ENV=$VENV uv pip install torch torchvision \\
      --index-url https://download.pytorch.org/whl/cu128
  VIRTUAL_ENV=$VENV uv pip install \\
      transformers fastapi 'uvicorn[standard]' json-numpy \\
      huggingface_hub hf-transfer pillow numpy accelerate \\
      safetensors einops requests
  VIRTUAL_ENV=$VENV uv pip install --no-deps \\
      'lerobot @ git+https://github.com/allenai/lerobot.git@molmoact2-policy'
EOF
    exit 2
fi
if ! "$VENV/bin/python" -c "import lerobot" 2>/dev/null; then
    echo "ERROR: lerobot not installed in $VENV. See bootstrap instructions above." >&2
    exit 2
fi

exec "$VENV/bin/python" "$SCRIPT_DIR/host_server_rtc.py" \
    --host 0.0.0.0 \
    --port 8203 \
    --dtype bfloat16 \
    "$@"
