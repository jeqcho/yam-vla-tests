"""Smoke-test a remote DreamZero WebSocket policy server.

Posts a few rounds of synthetic frames + zero state and confirms the server
returns a finite action chunk. Pass:

    uv run python scripts/smoke_test_remote.py \\
        --url wss://<workspace>--dreamzero-droid-serve.modal.run \\
        --schema droid \\
        --rounds 5

Schema notes
------------
The DreamZero `policy_server.py` sends a `PolicyServerConfig` dict on connect.
We print it; that's authoritative for what fields the server actually wants.

`--schema droid`  matches GEAR-Dreams/DreamZero-DROID:
    observation/exterior_image_0_left, observation/exterior_image_1_left,
    observation/wrist_image_left  (each H×W×3 uint8, default 180×320)
    observation/joint_position (7,) float, observation/gripper_position (1,) float
    prompt (str), session_id (str)
    action: (N, 8)  — 7 joints + 1 gripper, default N=24 (action_horizon)

`--schema yam`  matches the (not-yet-released) bimanual checkpoint with the
modality keys from `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`:
    video.top_camera-images-rgb, video.left_camera-images-rgb,
    video.right_camera-images-rgb  (each H×W×3 uint8 or (T,H,W,3))
    state.left_joint_pos (6,), state.left_gripper_pos (1,),
    state.right_joint_pos (6,), state.right_gripper_pos (1,)
    annotation.task (str)
    action.{left,right}_{joint,gripper}_pos  per-key

The YAM path is currently a stub — `dreamzero_server.py MODEL=yam` will 404 on
the HF download until a checkpoint exists.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid

import msgpack
import msgpack_numpy
import numpy as np

try:
    import websockets
except ImportError:  # pragma: no cover
    print("pip/uv: missing `websockets`. Run `uv sync` in this folder.", file=sys.stderr)
    raise

msgpack_numpy.patch()


def _droid_obs(h: int, w: int, prompt: str, session_id: str) -> dict:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return {
        "observation/exterior_image_0_left": frame.copy(),
        "observation/exterior_image_1_left": frame.copy(),
        "observation/wrist_image_left": frame.copy(),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/cartesian_position": np.zeros(6, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": prompt,
        "session_id": session_id,
        "endpoint": "infer",
    }


def _yam_obs(h: int, w: int, prompt: str) -> dict:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return {
        "video.top_camera-images-rgb": frame[None, ...].copy(),    # (1, H, W, 3)
        "video.left_camera-images-rgb": frame[None, ...].copy(),
        "video.right_camera-images-rgb": frame[None, ...].copy(),
        "state.left_joint_pos": np.zeros((1, 6), dtype=np.float64),
        "state.left_gripper_pos": np.zeros((1, 1), dtype=np.float64),
        "state.right_joint_pos": np.zeros((1, 6), dtype=np.float64),
        "state.right_gripper_pos": np.zeros((1, 1), dtype=np.float64),
        "annotation.task": prompt,
        "endpoint": "infer",
    }


async def run(args: argparse.Namespace) -> int:
    print(f"Connecting to {args.url} (schema={args.schema}) ...")
    # ping_interval=None: dreamzero's server doesn't respond to control frames.
    async with websockets.connect(args.url, ping_interval=None, max_size=None) as ws:
        # First frame is server's PolicyServerConfig (or metadata for the
        # optimized AR server). Print it; it's authoritative.
        cfg_bytes = await ws.recv()
        try:
            cfg = msgpack_numpy.unpackb(cfg_bytes)
        except Exception:
            cfg = {"_raw": repr(cfg_bytes)[:200]}
        print("server config / metadata:", cfg)

        h, w = args.height, args.width
        session_id = uuid.uuid4().hex
        builder = _droid_obs if args.schema == "droid" else _yam_obs

        for round_i in range(args.rounds):
            if args.schema == "droid":
                obs = builder(h, w, args.prompt, session_id)
            else:
                obs = builder(h, w, args.prompt)

            packer = msgpack_numpy.Packer()
            t0 = time.perf_counter()
            await ws.send(packer.pack(obs))
            raw = await ws.recv()
            rtt_ms = (time.perf_counter() - t0) * 1000.0

            try:
                act = msgpack_numpy.unpackb(raw)
            except Exception:
                print(f"  round {round_i}: server returned non-msgpack payload "
                      f"({len(raw)} bytes). Probably an error traceback:")
                print(raw[:2000])
                return 1

            if isinstance(act, np.ndarray):
                shape = act.shape
                finite = np.isfinite(act).all()
                amax = float(np.abs(act).max())
                print(f"  round {round_i}: action shape={shape}, finite={finite}, "
                      f"|a|_max={amax:.3f}, rtt={rtt_ms:.0f} ms")
            elif isinstance(act, dict):
                summary = {
                    k: getattr(v, "shape", "?") for k, v in act.items()
                    if isinstance(v, np.ndarray)
                }
                print(f"  round {round_i}: dict keys={list(act.keys())[:8]} "
                      f"shapes={summary} rtt={rtt_ms:.0f} ms")
            else:
                print(f"  round {round_i}: unexpected payload type {type(act)}: {act!r}")
                return 1

    print("smoke test PASS")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--url", required=True,
                   help="wss://<workspace>--dreamzero-droid-serve.modal.run "
                        "(get from `modal serve` output)")
    p.add_argument("--schema", choices=["droid", "yam"], default="droid")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--height", type=int, default=180,
                   help="Synthetic frame H — DROID server resizes anyway, but match "
                        "the server-config image_resolution to avoid client-side resize")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--prompt", default="pick up the blue cube and place it in the bin")
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
