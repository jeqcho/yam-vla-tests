#!/usr/bin/env bash
# Print the wss:// URL of a running DreamZero server (from the local log).
# Useful when run_inference.sh was launched with --keep and you want to grab
# the URL again later.
set -euo pipefail
MODEL="${1:-droid}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/../logs/server_${MODEL}.log"
if [ ! -f "$LOG" ]; then
  echo "No log at $LOG. Have you run ./scripts/run_inference.sh ${MODEL}?" >&2
  exit 1
fi
URL="$(grep -oE 'wss://[A-Za-z0-9._-]+/' "$LOG" | head -1)"
if [ -z "$URL" ]; then
  echo "No wss:// URL in $LOG yet. The server may still be cold-starting." >&2
  exit 1
fi
echo "$URL"
