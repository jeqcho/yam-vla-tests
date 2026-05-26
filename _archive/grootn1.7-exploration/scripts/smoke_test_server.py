"""Synthetic smoke test for the GR00T N1.7 YAM PolicyServer.

Posts 3 rounds of fake frames + state via the ZeroMQ wire protocol and verifies
the server returns finite (horizon, D) action arrays for every modality key.

Usage (from inside the i2rt venv OR Isaac-GR00T venv — both have zmq+msgpack):

    .../python scripts/smoke_test_server.py --host 127.0.0.1 --port 5556
"""
from __future__ import annotations

import argparse
import sys
import time

import msgpack_numpy as mnp
import numpy as np
import zmq


def _call(socket: zmq.Socket, endpoint: str, data: dict | None = None,
          requires_input: bool = True) -> object:
    req: dict = {"endpoint": endpoint}
    if requires_input:
        req["data"] = data
    socket.send(mnp.packb(req))
    msg = socket.recv()
    resp = mnp.unpackb(msg, raw=False)
    if isinstance(resp, dict) and "error" in resp:
        raise RuntimeError(f"server error: {resp['error']}")
    return resp


def build_fake_obs(H: int, W: int, instruction: str) -> dict:
    """Build one (B=1, T=1) observation with synthetic frames + zero state."""
    def _img() -> np.ndarray:
        return (np.random.rand(1, 1, H, W, 3) * 255).astype(np.uint8)

    return {
        "video": {
            "top": _img(),
            "left_wrist": _img(),
            "right_wrist": _img(),
        },
        "state": {
            "left_arm":      np.zeros((1, 1, 6), dtype=np.float32),
            "left_gripper":  np.zeros((1, 1, 1), dtype=np.float32),
            "right_arm":     np.zeros((1, 1, 6), dtype=np.float32),
            "right_gripper": np.zeros((1, 1, 1), dtype=np.float32),
        },
        "language": {
            "annotation.human.task_description": [[instruction]],
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--image-height", type=int, default=240)
    p.add_argument("--image-width", type=int, default=424)
    p.add_argument("--instruction", default="pick up the orange cube and put it in the box")
    p.add_argument("--timeout-ms", type=int, default=60000)
    args = p.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    sock.connect(f"tcp://{args.host}:{args.port}")

    pong = _call(sock, "ping", requires_input=False)
    print(f"ping -> {pong}")

    print(f"posting {args.rounds} round(s) of synthetic frames to "
          f"tcp://{args.host}:{args.port}")
    H, W = args.image_height, args.image_width
    horizon = None
    for r in range(args.rounds):
        obs = build_fake_obs(H, W, args.instruction)
        t0 = time.perf_counter()
        resp = _call(sock, "get_action", {"observation": obs, "options": None})
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # Server returns [action_dict, info_dict]
        action, info = resp
        # Pick any modality key to read horizon.
        any_key = next(iter(action.keys()))
        arr = np.asarray(action[any_key])
        if arr.ndim != 3:
            print(f"  round {r}: FAIL — action[{any_key}].ndim={arr.ndim}, expected 3 (B,T,D)")
            return 1
        if horizon is None:
            horizon = arr.shape[1]
        if not np.all(np.isfinite(np.concatenate([
            np.asarray(action[k], dtype=np.float32).reshape(-1) for k in action
        ]))):
            print(f"  round {r}: FAIL — non-finite values in action")
            return 1
        shape_summary = " | ".join(
            f"{k}={tuple(np.asarray(action[k]).shape)}" for k in action
        )
        print(f"  round {r}: rtt={dt_ms:.0f} ms  horizon={arr.shape[1]}  {shape_summary}")

    print("smoke test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
