"""Synthetic end-to-end smoke test: POST a fake request to the MolmoAct2 server
and verify the action tensor comes back with the right shape and finite values.

Use this BEFORE plugging in real cameras/arms. Requires the server to be
running:

    # Terminal A
    ./scripts/run_server.sh

    # Terminal B
    /home/andon/yam-tests/molmoact2-setup/.venv/bin/python scripts/smoke_test_server.py
"""
from __future__ import annotations

import argparse
import sys
import time

import json_numpy
import numpy as np
import requests

json_numpy.patch()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server-url", default="http://127.0.0.1:8202/act")
    p.add_argument("--instruction", default="first pick up the left orange cube and put it in the box, then pick up the right orange cube and put it in the box")
    p.add_argument("--height", type=int, default=180)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()

    # 1. Health check.
    try:
        r = requests.get(args.server_url, timeout=5.0)
        r.raise_for_status()
        print("health:", r.json())
    except Exception as e:
        print(f"health FAIL: {e}", file=sys.stderr)
        return 1

    # 2. Synthetic input: structured grayscale gradient frames + zero state.
    yy, xx = np.mgrid[0:args.height, 0:args.width].astype(np.float32)
    grad = (xx / args.width * 255).astype(np.uint8)
    base_img = np.stack([grad, grad, grad], axis=-1)
    top, left, right = base_img.copy(), base_img.copy(), base_img.copy()
    # Stamp a colored corner so the three frames are visibly different.
    top[-20:, -40:, 0] = 255   # red blob on top
    left[-20:, -40:, 1] = 255  # green on left
    right[-20:, -40:, 2] = 255 # blue on right

    state = np.zeros(14, dtype=np.float32)

    print(f"\nposting {args.rounds} round(s) of synthetic frames to {args.server_url}")
    for i in range(args.rounds):
        payload = {
            "top_cam": top,
            "left_cam": left,
            "right_cam": right,
            "instruction": args.instruction,
            "state": state,
            "num_steps": args.num_steps,
            "timestamp": time.time(),
        }
        body = json_numpy.dumps(payload)
        t0 = time.perf_counter()
        r = requests.post(args.server_url, data=body,
                         headers={"Content-Type": "application/json"},
                         timeout=60.0)
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            print(f"  round {i}: HTTP {r.status_code} -- {r.text[:300]}", file=sys.stderr)
            return 1
        out = json_numpy.loads(r.text)
        actions = np.asarray(out["actions"], dtype=np.float32)
        server_dt_ms = float(out.get("dt_ms", 0.0))
        if actions.ndim != 2 or actions.shape[1] != 14:
            print(f"  round {i}: unexpected action shape {actions.shape}", file=sys.stderr)
            return 1
        if not np.isfinite(actions).all():
            print(f"  round {i}: non-finite actions in response", file=sys.stderr)
            return 1
        # Sanity: 1st step delta should be small (model expects continuous control).
        first_delta = float(np.max(np.abs(actions[0] - state)))
        print(f"  round {i}: actions shape={actions.shape}, "
              f"|a-s|_max(first step)={first_delta:.3f}, "
              f"server dt={server_dt_ms:.0f} ms, rtt={rtt_ms:.0f} ms")

    print("\nsmoke test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
