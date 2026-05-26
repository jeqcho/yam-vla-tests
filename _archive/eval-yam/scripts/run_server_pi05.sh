#!/usr/bin/env bash
# Top-level wrapper: launches the pi05 server.
# Real implementation lives in eval-yam/servers/pi05/run_server.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/../servers/pi05/run_server.sh" "$@"
