"""Synthetic smoke test for the GR00T N1.7-3B XDOF YAM server.

Posts 3 rounds of fake frames + state in the XDOF wire format and verifies
the server returns (40, D) action arrays for every modality key.

Usage:
    "Isaac-GR00T/.venv/bin/python" scripts/smoke_test_server_xdof.py \
        --host 127.0.0.1 --port 5557
"""
from __future__ import annotations

import argparse
import sys
import time

import msgpack_numpy as mnp
import numpy as np
import zmq


def _call(socket: zmq.Socket, endpoint: str, data: dict | None = None,
          requires_input: bool = True):
    req: dict = {"endpoint": endpoint}
    if requires_input:
        req["data"] = data
    socket.send(mnp.packb(req))
    msg = socket.recv()
    resp = mnp.unpackb(msg, raw=False)
    if isinstance(resp, dict) and "error" in resp:
        raise RuntimeError(f"server error: {resp['error']}")
    return resp


def build_fake_xdof_obs(H: int, W: int, instruction: str) -> dict:
    """One (B=1, T=2 for video, T=1 for state) observation in XDOF format."""
    def _video() -> np.ndarray:
        # (B=1, T=2, H, W, 3)
        return (np.random.rand(1, 2, H, W, 3) * 255).astype(np.uint8)

    return {
        "video": {
            "top_camera-images-rgb_320_240":   _video(),
            "left_camera-images-rgb_320_240":  _video(),
            "right_camera-images-rgb_320_240": _video(),
        },
        "state": {
            # The base model's statistics.json has EMPTY (dim=0) stats for
            # left_wrist_eef / right_wrist_eef under XDOF (because the YAM
            # subset of the training mixture didn't carry EEF state). Send
            # empty arrays to match. The model's EEF action head is also
            # empty; we'll ignore those outputs on the action side.
            "left_wrist_eef":    np.zeros((1, 1, 0), dtype=np.float32),
            "right_wrist_eef":   np.zeros((1, 1, 0), dtype=np.float32),
            "left_gripper_pos":  np.array([[[0.5]]], dtype=np.float32),
            "right_gripper_pos": np.array([[[0.5]]], dtype=np.float32),
            "left_joint_pos":    np.zeros((1, 1, 6), dtype=np.float32),
            "right_joint_pos":   np.zeros((1, 1, 6), dtype=np.float32),
        },
        "language": {
            "annotation.task": [[instruction]],
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5557)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--image-height", type=int, default=240)
    p.add_argument("--image-width", type=int, default=320)
    p.add_argument("--instruction", default="Move the blocks to spell AI2")
    p.add_argument("--timeout-ms", type=int, default=120000)
    args = p.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    sock.connect(f"tcp://{args.host}:{args.port}")

    print(f"ping -> {_call(sock, 'ping', requires_input=False)}")

    print(f"posting {args.rounds} round(s) to tcp://{args.host}:{args.port}")
    H, W = args.image_height, args.image_width
    horizon = None
    for r in range(args.rounds):
        obs = build_fake_xdof_obs(H, W, args.instruction)
        t0 = time.perf_counter()
        resp = _call(sock, "get_action", {"observation": obs, "options": None})
        dt_ms = (time.perf_counter() - t0) * 1000.0
        action, info = resp
        any_key = next(iter(action.keys()))
        arr = np.asarray(action[any_key])
        if arr.ndim != 3:
            print(f"  round {r}: FAIL — {any_key}.ndim={arr.ndim}, expected 3")
            return 1
        horizon = arr.shape[1] if horizon is None else horizon
        # Check finiteness across every key.
        for k, v in action.items():
            a = np.asarray(v, dtype=np.float32)
            if not np.all(np.isfinite(a)):
                print(f"  round {r}: FAIL — non-finite values in action[{k}]")
                return 1
        shape_summary = " | ".join(
            f"{k}={tuple(np.asarray(action[k]).shape)}" for k in sorted(action.keys())
        )
        print(f"  round {r}: rtt={dt_ms:.0f} ms  horizon={arr.shape[1]}  {shape_summary}")

    print("smoke test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
