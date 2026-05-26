#!/usr/bin/env bash
# Top-level wrapper: launches the gr00t-n17 server.
# Real implementation lives in eval-yam/servers/gr00t/run_server.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/../servers/gr00t/run_server.sh" "$@"
