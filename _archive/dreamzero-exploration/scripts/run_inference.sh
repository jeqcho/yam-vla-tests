#!/usr/bin/env bash
# One-command end-to-end DreamZero inference test.
#
# Brings up the Modal server (or attaches to a running one), waits for the
# WebSocket tunnel URL, then runs a synthetic-frame smoke test. Exits 0 if
# DreamZero returned a finite action tensor.
#
# Usage:
#   ./scripts/run_inference.sh                 # MODEL=droid (default, runnable today)
#   ./scripts/run_inference.sh yam             # MODEL=yam (needs a fine-tuned ckpt)
#   ./scripts/run_inference.sh droid --keep    # leave the server running after smoke test
#
# The Modal server is deployed via `modal serve` in this terminal — you'll see
# image build progress, then "DreamZero server is up. Connect with: wss://…".
# Ctrl-C this script to stop the server (which also stops the Modal billing).
#
# Cost: H100:2 ≈ $10/hr while up. Cold start ≈ 10-15 min (image build + 23 GB
# checkpoint download — cached on the dreamzero-hf-cache volume after first run).
set -euo pipefail

MODEL="${1:-droid}"
shift || true
KEEP=0
for arg in "$@"; do
  [ "$arg" = "--keep" ] && KEEP=1
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

if [ ! -d dreamzero ]; then
  echo "Cloning dreamzero repo (needed for Modal image build) ..."
  git clone --depth 1 https://github.com/dreamzero0/dreamzero.git dreamzero
fi

LOG="logs/server_${MODEL}.log"
mkdir -p logs
: > "$LOG"

echo "============================================================"
echo "  Bringing up DreamZero-${MODEL} on Modal H100:2"
echo "  Live log:  tail -F $LOG"
echo "============================================================"

# Start `modal serve` in the background. It streams container stdout to its own
# stdout, which we redirect to $LOG.
MODEL="$MODEL" nohup modal serve modal/dreamzero_server.py > "$LOG" 2>&1 &
SERVER_PID=$!
echo "modal serve pid=$SERVER_PID"

cleanup() {
  if [ "$KEEP" = "1" ]; then
    echo
    echo "--keep set — leaving server running. Stop with:"
    echo "    kill $SERVER_PID    # local supervisor"
    echo "    modal app stop dreamzero-${MODEL}"
    return
  fi
  echo
  echo "Stopping local supervisor pid=$SERVER_PID ..."
  kill "$SERVER_PID" 2>/dev/null || true
  # The Modal app stops automatically when modal serve exits.
}
trap cleanup EXIT

# Poll the log for the URL banner printed by our serve() function. Cold start
# can take 20 min on first deploy (image build + HF download).
echo
echo "Waiting for tunnel URL (up to 30 min for cold start) ..."
URL=""
for i in $(seq 1 1800); do
  if grep -q "DreamZero server is up" "$LOG" 2>/dev/null; then
    URL="$(grep -oE 'wss://[A-Za-z0-9._-]+/' "$LOG" | head -1)"
    [ -n "$URL" ] && break
  fi
  if grep -qE "Error|Traceback|exited early|TimeoutError" "$LOG" 2>/dev/null; then
    echo "!! Server failed to come up. Tail:"
    tail -30 "$LOG"
    exit 1
  fi
  sleep 1
done

if [ -z "$URL" ]; then
  echo "!! No URL after 30 min. Tail:"
  tail -30 "$LOG"
  exit 1
fi

echo
echo "============================================================"
echo "  SERVER UP at: $URL"
echo "============================================================"
echo
echo "Running smoke test (3 rounds of synthetic frames) ..."
uv run python scripts/smoke_test_remote.py --url "$URL" --schema "$MODEL" --rounds 3
RC=$?

echo
if [ "$RC" = "0" ]; then
  echo "============================================================"
  echo "  ✅  DreamZero-${MODEL} inference VERIFIED end-to-end."
  echo "============================================================"
  echo
  echo "To poke it more from your own client:"
  echo "    uv run python scripts/smoke_test_remote.py --url $URL --schema $MODEL"
  echo
else
  echo "!! Smoke test failed (rc=$RC). Server log tail:"
  tail -20 "$LOG"
fi

if [ "$KEEP" = "0" ]; then
  echo "Tearing down server (pass --keep to leave it up)."
fi
exit "$RC"
