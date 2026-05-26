"""Drive a bimanual YAM rig from a DreamZero policy hosted on Modal.

This is the analog of `molmoact2-setup/scripts/yam_client.py`, rewritten for:
  * WebSocket + msgpack-numpy (not HTTP + json_numpy)
  * the *bimanual YAM* DreamZero schema — left/right joints + grippers, 3 cams
    keyed as top/left/right per `groot/vla/configs/data/dreamzero/...yam_relative`.

REQUIRES a DreamZero checkpoint fine-tuned on YAM-bimanual. As of this writing
no such checkpoint is public; only DreamZero-DROID (single 7-DOF arm) is
released. Talking to the DROID server with bimanual state is undefined
behaviour; this script refuses unless `--allow-droid-server` is passed (and
even then defaults to dry-run).

Safety
------
Same defenses as `yam_client.py`:
  * per-tick joint delta clipped to `--max-step-rad` (default 0.05 rad ≈ 2.9°)
  * gripper delta clipped to `--gripper-step` (default 0.05, normalized)
  * `--dry-run` (default!) prints actions without commanding the arms
  * SIGINT stops the loop; arms hold the last commanded pose

Run
---
    /home/andon/yam-tests/i2rt/.venv/bin/python scripts/dreamzero_yam_client.py \\
        --url wss://<workspace>--dreamzero-yam-serve.modal.run \\
        --left-can can0 --right-can can1 \\
        --left-gripper linear_4310 --right-gripper linear_4310 \\
        --top-cam-serial XXXX --left-cam-serial YYYY --right-cam-serial ZZZZ \\
        --instruction "pick up the orange cube on the left and put it in the box" \\
        --dry-run                       # mandatory until you've verified actions
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import msgpack
import msgpack_numpy
import numpy as np

try:
    import websockets
except ImportError:
    print("pip/uv: missing `websockets`. Run `uv sync` in the dreamzero exploration folder.",
          file=sys.stderr)
    raise

# i2rt is provided by /home/andon/yam-tests/i2rt/.venv; run this script with
# that interpreter on the YAM workstation.
try:
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.robot import Robot
    from i2rt.robots.utils import ArmType, GripperType
    HAVE_I2RT = True
except ImportError:
    HAVE_I2RT = False

msgpack_numpy.patch()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dreamzero.yam.client")


# ----------------------------------------------------------------------------
# Hardware glue — minimal versions of the routines from yam_client.py
# ----------------------------------------------------------------------------

@dataclass
class ArmConfig:
    can: str
    gripper_type: str  # one of {linear_4310, linear_3507, crank_4310, flexible_4310}


def _init_arm(cfg: ArmConfig) -> Optional["Robot"]:
    if not HAVE_I2RT:
        return None
    gt = {
        "linear_4310": GripperType.LINEAR_4310,
        "linear_3507": GripperType.LINEAR_3507,
        "crank_4310":  GripperType.CRANK_4310,
        "flexible_4310": GripperType.FLEXIBLE_4310,
    }[cfg.gripper_type]
    return get_yam_robot(channel=cfg.can, gripper_type=gt, arm_type=ArmType.LEADER)


def _read_state(left: "Robot", right: "Robot") -> dict:
    """DreamZero YAM modality schema, all (1, D) float64."""
    lj = np.asarray(left.get_joint_pos(), dtype=np.float64).reshape(-1)
    rj = np.asarray(right.get_joint_pos(), dtype=np.float64).reshape(-1)
    # i2rt returns (7,) per arm with gripper at index 6, already normalized [0,1].
    return {
        "state.left_joint_pos":    lj[:6].reshape(1, 6),
        "state.left_gripper_pos":  lj[6:7].reshape(1, 1),
        "state.right_joint_pos":   rj[:6].reshape(1, 6),
        "state.right_gripper_pos": rj[6:7].reshape(1, 1),
    }


def _capture_frames(pipelines: dict) -> dict:
    """Read one RGB frame from each RealSense pipeline. Returns the DreamZero
    bimanual video keys with (1, H, W, 3) uint8."""
    import pyrealsense2 as rs
    out = {}
    for name, pipe in pipelines.items():
        frames = pipe.wait_for_frames(timeout_ms=200)
        c = frames.get_color_frame()
        if not c:
            raise RuntimeError(f"camera {name} returned no color frame")
        rgb = np.asanyarray(c.get_data())  # already RGB after configure_pipeline()
        out[name] = rgb[None, ...].astype(np.uint8)
    return out


def _open_pipelines(serials: dict, width: int, height: int):
    import pyrealsense2 as rs
    pipes = {}
    for key, serial in serials.items():
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
        pipe = rs.pipeline()
        pipe.start(cfg)
        pipes[key] = pipe
    return pipes


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------

async def _send_obs(ws, packer: msgpack_numpy.Packer, obs: dict) -> dict:
    await ws.send(packer.pack(obs))
    raw = await ws.recv()
    return msgpack_numpy.unpackb(raw)


def _stack_action(act_dict: dict) -> np.ndarray:
    """Flatten DreamZero YAM action dict into (N, 14) [lj×6, lg×1, rj×6, rg×1]."""
    def _arr(k):
        v = act_dict[k]
        if v.ndim == 1:
            v = v[None, :]
        return v.astype(np.float32, copy=False)
    lj = _arr("action.left_joint_pos")
    lg = _arr("action.left_gripper_pos")
    rj = _arr("action.right_joint_pos")
    rg = _arr("action.right_gripper_pos")
    n = min(lj.shape[0], lg.shape[0], rj.shape[0], rg.shape[0])
    return np.concatenate([lj[:n], lg[:n], rj[:n], rg[:n]], axis=-1)  # (n, 14)


def _clip(cmd: np.ndarray, cur: np.ndarray, max_dq: float, max_dg: float) -> np.ndarray:
    """Per-tick joint+gripper clipping. cmd, cur are (14,)."""
    j_idx = np.r_[0:6, 7:13]
    g_idx = np.r_[6, 13]
    out = cmd.copy()
    out[j_idx] = cur[j_idx] + np.clip(cmd[j_idx] - cur[j_idx], -max_dq, max_dq)
    out[g_idx] = cur[g_idx] + np.clip(cmd[g_idx] - cur[g_idx], -max_dg, max_dg)
    return out


async def run(args: argparse.Namespace) -> int:
    if "droid" in args.url.lower() and not args.allow_droid_server:
        log.error(
            "Server URL contains 'droid' but --allow-droid-server was not set. "
            "DreamZero-DROID expects single-arm Franka schema; sending bimanual "
            "YAM state will produce undefined behaviour. Aborting."
        )
        return 2

    # Hardware bring-up (skipped under --no-hardware).
    if args.no_hardware:
        log.warning("--no-hardware: using zero state and synthetic frames")
        left = right = None
        pipes = None
    else:
        if not HAVE_I2RT:
            log.error("i2rt SDK not importable. Use /home/andon/yam-tests/i2rt/.venv/bin/python.")
            return 3
        left = _init_arm(ArmConfig(args.left_can, args.left_gripper))
        right = _init_arm(ArmConfig(args.right_can, args.right_gripper))
        pipes = _open_pipelines(
            {
                "video.top_camera-images-rgb":   args.top_cam_serial,
                "video.left_camera-images-rgb":  args.left_cam_serial,
                "video.right_camera-images-rgb": args.right_cam_serial,
            },
            width=args.cam_width, height=args.cam_height,
        )

    stop = asyncio.Event()
    def _on_sigint(*_):
        log.info("SIGINT — stopping after current action chunk")
        stop.set()
    signal.signal(signal.SIGINT, _on_sigint)

    session_id = uuid.uuid4().hex
    log.info("session_id=%s", session_id)
    log.info("connecting to %s", args.url)

    async with websockets.connect(args.url, ping_interval=None, max_size=None) as ws:
        cfg_bytes = await ws.recv()
        try:
            cfg = msgpack_numpy.unpackb(cfg_bytes)
            log.info("server config: %s", cfg)
        except Exception:
            log.warning("server sent non-msgpack first frame (%d bytes); ignoring", len(cfg_bytes))

        packer = msgpack_numpy.Packer()
        tick_dt = 1.0 / args.train_fps

        chunk_i = 0
        while not stop.is_set():
            t_chunk = time.perf_counter()

            # Build obs
            if pipes is not None:
                images = _capture_frames(pipes)
            else:
                blank = np.zeros((1, args.cam_height, args.cam_width, 3), dtype=np.uint8)
                images = {
                    "video.top_camera-images-rgb":   blank,
                    "video.left_camera-images-rgb":  blank,
                    "video.right_camera-images-rgb": blank,
                }
            if left is not None:
                st = _read_state(left, right)
            else:
                st = {
                    "state.left_joint_pos":    np.zeros((1, 6), dtype=np.float64),
                    "state.left_gripper_pos":  np.zeros((1, 1), dtype=np.float64),
                    "state.right_joint_pos":   np.zeros((1, 6), dtype=np.float64),
                    "state.right_gripper_pos": np.zeros((1, 1), dtype=np.float64),
                }
            obs = {
                **images, **st,
                "annotation.task": args.instruction,
                "session_id": session_id,
                "endpoint": "infer",
            }

            t0 = time.perf_counter()
            act = await _send_obs(ws, packer, obs)
            dt_infer = time.perf_counter() - t0

            if not isinstance(act, dict):
                log.error("server returned non-dict action (%s); aborting: %r",
                          type(act), act if not isinstance(act, bytes) else act[:200])
                return 4

            chunk = _stack_action(act)  # (N, 14)
            log.info("chunk %d: shape=%s |a|_max=%.3f dt_infer=%.0f ms",
                     chunk_i, chunk.shape, float(np.abs(chunk).max()), dt_infer * 1000)

            # Roll the action horizon out at train_fps with per-tick clipping.
            for step_i in range(0, chunk.shape[0], args.horizon_stride):
                if stop.is_set():
                    break
                cmd = chunk[step_i]
                if left is not None:
                    cur = np.concatenate([left.get_joint_pos(), right.get_joint_pos()])
                    cmd = _clip(cmd, cur, args.max_step_rad, args.gripper_step)
                if args.dry_run:
                    log.info("  step %d cmd[:6]=%s", step_i, np.array2string(cmd[:6], precision=3))
                else:
                    left.command_joint_pos(np.r_[cmd[0:6], cmd[6]])
                    right.command_joint_pos(np.r_[cmd[7:13], cmd[13]])
                t_next = t_chunk + (step_i + args.horizon_stride) * tick_dt
                sleep_s = t_next - time.perf_counter()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
            chunk_i += 1
    return 0


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--url", required=True, help="wss://… of the DreamZero YAM server")
    p.add_argument("--instruction", required=True)
    # Hardware
    p.add_argument("--no-hardware", action="store_true",
                   help="Skip i2rt + RealSense bring-up; use zero state + black frames")
    p.add_argument("--left-can", default="can0")
    p.add_argument("--right-can", default="can1")
    p.add_argument("--left-gripper", default="linear_4310",
                   choices=["linear_4310", "linear_3507", "crank_4310", "flexible_4310"])
    p.add_argument("--right-gripper", default="linear_4310",
                   choices=["linear_4310", "linear_3507", "crank_4310", "flexible_4310"])
    p.add_argument("--top-cam-serial", default="")
    p.add_argument("--left-cam-serial", default="")
    p.add_argument("--right-cam-serial", default="")
    p.add_argument("--cam-width", type=int, default=320)
    p.add_argument("--cam-height", type=int, default=180)
    # Control
    p.add_argument("--train-fps", type=float, default=30.0)
    p.add_argument("--horizon-stride", type=int, default=6,
                   help="Stride through the (N=24) action horizon between server queries")
    p.add_argument("--max-step-rad", type=float, default=0.05)
    p.add_argument("--gripper-step", type=float, default=0.05)
    # Safety
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="Default: print actions, do NOT command arms. Pass --no-dry-run to drive.")
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    p.add_argument("--allow-droid-server", action="store_true",
                   help="Permit talking to a DROID server with bimanual state (DEBUG ONLY)")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse())))
