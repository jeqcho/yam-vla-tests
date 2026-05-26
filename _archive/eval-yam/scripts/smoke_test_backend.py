"""Headless smoke test for a running VLA server.

Sends a single synthetic observation through the matching Backend and
checks the response is the expected shape. Surfaces wire-format bugs in
seconds without needing the arms or cameras.

Usage:
  /home/andon/yam-tests/i2rt/.venv/bin/python eval-yam/scripts/smoke_test_backend.py \
      --policy molmoact2

  ... --policy pi05      --server-host 127.0.0.1 --server-port 8000
  ... --policy gr00t-n17 --server-host 127.0.0.1 --server-port 5556

Exit 0 = OK, exit 2 = wire-format / shape failure.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

import yam_backends  # noqa: E402


def _build_backend(args) -> yam_backends.Backend:
    if args.policy == "molmoact2":
        return yam_backends.MolmoActHTTPBackend(server_url=args.server_url)
    if args.policy == "pi05":
        return yam_backends.Pi05WebsocketBackend(host=args.server_host,
                                                  port=args.server_port)
    if args.policy == "gr00t-n17":
        return yam_backends.Gr00tZmqBackend(host=args.server_host,
                                             port=args.server_port)
    raise ValueError(f"Unknown --policy {args.policy!r}")


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke-test a VLA server")
    p.add_argument("--policy", required=True,
                   choices=["molmoact2", "pi05", "gr00t-n17"])
    p.add_argument("--server-url", default=None,
                   help="(molmoact2) full HTTP URL, default http://127.0.0.1:8202/act")
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=None)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--width",  type=int, default=424)
    p.add_argument("--instruction", default="warmup smoke test")
    p.add_argument("--timeout-s", type=float, default=60.0)
    p.add_argument("-n", "--n-calls", type=int, default=2,
                   help="number of inference calls to do (default 2)")
    args = p.parse_args()

    pd = {
        "molmoact2": {"server_url": "http://127.0.0.1:8202/act"},
        "pi05":      {"server_port": 8000},
        "gr00t-n17": {"server_port": 5556},
    }[args.policy]
    if args.policy == "molmoact2":
        args.server_url = args.server_url or pd["server_url"]
    else:
        args.server_port = args.server_port or pd["server_port"]

    print(f"[smoke] policy={args.policy}", flush=True)
    backend = _build_backend(args)

    print(f"[smoke] health check ...", flush=True)
    try:
        meta = backend.health_check(timeout_s=5.0)
        print(f"[smoke] health OK: {meta}", flush=True)
    except Exception as e:
        print(f"[smoke] FAIL health: {type(e).__name__}: {e}", flush=True)
        return 2

    H, W = args.height, args.width
    top   = np.full((H, W, 3), 32, dtype=np.uint8)
    left  = np.full((H, W, 3), 64, dtype=np.uint8)
    right = np.full((H, W, 3), 96, dtype=np.uint8)
    # Use a state that looks vaguely in-distribution (not zeros) so the model
    # doesn't crash on extreme normalization. Roughly a "ready" pose:
    # arm joints near 0, grippers ~0.5.
    state = np.array(
        [0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.5,    # left arm + gripper
         0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.5],   # right arm + gripper
        dtype=np.float32,
    )

    ok = True
    for i in range(args.n_calls):
        try:
            t0 = time.perf_counter()
            actions, rtt_ms = backend.predict(
                top, left, right, state,
                instruction=args.instruction,
                num_steps=10,
                timeout_s=args.timeout_s,
            )
        except Exception as e:
            print(f"[smoke] FAIL predict #{i+1}: {type(e).__name__}: {e}",
                  flush=True)
            return 2
        elapsed = (time.perf_counter() - t0) * 1000
        if actions.ndim != 2 or actions.shape[1] != 14:
            print(f"[smoke] FAIL predict #{i+1}: expected (N, 14), got {actions.shape}",
                  flush=True)
            return 2
        if not np.isfinite(actions).all():
            print(f"[smoke] FAIL predict #{i+1}: actions contain non-finite values",
                  flush=True)
            return 2
        print(f"[smoke] predict #{i+1}: actions shape={actions.shape} dtype={actions.dtype} "
              f"rtt_ms={rtt_ms:.0f} elapsed_ms={elapsed:.0f}", flush=True)
        print(f"[smoke]   a[0]  = {np.array2string(actions[0],  precision=3)}",
              flush=True)
        print(f"[smoke]   a[-1] = {np.array2string(actions[-1], precision=3)}",
              flush=True)
        # Sanity: actions shouldn't be wildly out of joint range.
        max_abs = float(np.max(np.abs(actions)))
        if max_abs > 10.0:
            print(f"[smoke] WARN: max |action| = {max_abs:.2f} -- looks out of range "
                  f"(expected radians, typically |q| < 3)", flush=True)

    print(f"[smoke] OK -- all {args.n_calls} calls succeeded", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
