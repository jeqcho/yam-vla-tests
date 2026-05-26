"""Verify HF auth is wired up for GR00T N1.7. Run BEFORE starting the server.

Checks:
    1. An HF token is configured locally (HF_TOKEN env var or `hf auth login`).
    2. The token has been granted access to nvidia/Cosmos-Reason2-2B.
    3. The base model nvidia/GR00T-N1.7-3B is fetchable.

Exits 0 if everything is green, non-zero with a clear error otherwise.
"""
from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi


def main() -> int:
    api = HfApi()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        try:
            token = api.token
        except Exception:
            token = None

    if not token:
        print("FAIL: no HF token found.")
        print("Fix: hf auth login    (or set HF_TOKEN in your env)")
        return 1

    who = api.whoami(token=token).get("name", "?")
    print(f"OK: authenticated as {who}")

    # Cosmos-Reason2-2B is gated. A 401 here means the user hasn't agreed to the
    # license. A 404 would mean the repo moved.
    try:
        api.model_info("nvidia/Cosmos-Reason2-2B", token=token)
        print("OK: nvidia/Cosmos-Reason2-2B accessible (gated, you've agreed to the license)")
    except Exception as e:
        msg = str(e)
        if "401" in msg or "Gated" in msg or "Access" in msg:
            print("FAIL: nvidia/Cosmos-Reason2-2B is gated and your token lacks access.")
            print("Fix: visit https://huggingface.co/nvidia/Cosmos-Reason2-2B and click")
            print("     'Agree and access repository' (approval is near-instant).")
            return 2
        print(f"FAIL: error checking nvidia/Cosmos-Reason2-2B -- {e}")
        return 2

    try:
        api.model_info("nvidia/GR00T-N1.7-3B", token=token)
        print("OK: nvidia/GR00T-N1.7-3B accessible")
    except Exception as e:
        print(f"FAIL: nvidia/GR00T-N1.7-3B not accessible -- {e}")
        return 3

    print()
    print("All checks passed. You're clear to start the server with ./scripts/run_server.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
