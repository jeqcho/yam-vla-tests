"""Ramp both YAM arms from their current pose to MolmoAct2-BimanualYAM's
training-mean pose, then exit.

The point: action_stats.mean from norm_stats.json is the centroid of the
training distribution. Starting inference from a pose far from this centroid
(e.g. both arms parallel to the table with most joints near 0) reliably
produces hedged near-identity actions. Putting the arms here first should
get the policy into a regime where it has confident behavior to follow.

Reads the target pose straight from the model's norm_stats.json. Linearly
interpolates from current state to target over --duration-s seconds at --hz.
Leaves grippers where they are (does not actuate them).

Usage:
    /home/andon/yam-tests/i2rt/.venv/bin/python \\
        /home/andon/yam-tests/molmoact2-setup/scripts/move_to_ready.py \\
        --left-can can0 --right-can can1 \\
        --left-gripper linear_4310 --right-gripper linear_4310

Safety:
  - Both arms init in position-holding mode (kp != 0) so they don't sag.
  - 3-second countdown before motion starts.
  - Ctrl-C during the ramp stops the loop and leaves the arms at the
    last commanded interpolation step, then exits.
  - Default 5s duration with the largest joint travel (~1.4 rad) gives a
    peak joint velocity ~0.28 rad/s -- conservative.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time

import numpy as np

# i2rt
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType


NORM_STATS_PATH = (
    "/home/andon/yam-tests/molmoact2-setup/hf-cache/hub/"
    "models--allenai--MolmoAct2-BimanualYAM/snapshots/"
    "28e56c0fa4cb8598bfc2261e45499b3cc77763d4/norm_stats.json"
)
NORM_TAG = "yam_dual_molmoact2"


def _trace(msg: str) -> None:
    print(f"[move_to_ready] {msg}", flush=True)


def load_target_pose() -> np.ndarray:
    """Returns the 14-dim training-mean action vector."""
    with open(NORM_STATS_PATH) as f:
        d = json.load(f)
    mean = d["metadata_by_tag"][NORM_TAG]["action_stats"]["mean"]
    return np.asarray(mean, dtype=np.float32)


def init_arm(channel: str, gripper: str):
    return get_yam_robot(
        channel=channel,
        arm_type=ArmType.from_string_name("yam"),
        gripper_type=GripperType.from_string_name(gripper),
        zero_gravity_mode=False,  # active position-holding from init
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stderr)

    p = argparse.ArgumentParser()
    p.add_argument("--left-can",  default="can0")
    p.add_argument("--right-can", default="can1")
    p.add_argument("--left-gripper",  default="linear_4310")
    p.add_argument("--right-gripper", default="linear_4310")
    p.add_argument("--duration-s", type=float, default=5.0,
                   help="seconds to ramp from current pose to ready pose")
    p.add_argument("--hz", type=float, default=30.0,
                   help="interpolation update rate")
    p.add_argument("--countdown-s", type=float, default=3.0,
                   help="seconds to wait before motion starts (Ctrl-C window)")
    p.add_argument("--max-joint-delta", type=float, default=2.0,
                   help="abort if any joint must travel more than this many radians "
                        "(safety against bad norm_stats parsing)")
    args = p.parse_args()

    target = load_target_pose()
    _trace(f"target pose (norm_stats mean) = {np.array2string(target, precision=3)}")
    if target.shape != (14,):
        _trace(f"FATAL: expected 14-dim target, got shape {target.shape}")
        return 2

    _trace(f"initializing arms on {args.left_can} / {args.right_can}...")
    left  = init_arm(args.left_can,  args.left_gripper)
    right = init_arm(args.right_can, args.right_gripper)
    time.sleep(0.5)  # let SDK background thread stabilize

    q_l = np.asarray(left.get_joint_pos(),  dtype=np.float32)
    q_r = np.asarray(right.get_joint_pos(), dtype=np.float32)
    start = np.concatenate([q_l, q_r])
    _trace(f"start state                  = {np.array2string(start, precision=3)}")

    # Keep grippers where they are -- target is "ready arm pose", not "ready gripper".
    goal = target.copy()
    goal[6]  = start[6]
    goal[13] = start[13]
    _trace(f"goal pose (grippers held)    = {np.array2string(goal, precision=3)}")

    delta = goal - start
    max_delta = float(np.max(np.abs(delta)))
    _trace(f"max per-joint delta: {max_delta:.3f} rad ({np.degrees(max_delta):.1f} deg)")
    if max_delta > args.max_joint_delta:
        _trace(f"REFUSING: max delta {max_delta:.3f} > --max-joint-delta {args.max_joint_delta}")
        try: left.close()
        except Exception: pass
        try: right.close()
        except Exception: pass
        os._exit(3)

    n_steps = max(1, int(args.duration_s * args.hz))
    dt = 1.0 / args.hz
    peak_vel = max_delta / args.duration_s
    _trace(f"plan: {n_steps} steps over {args.duration_s:.1f}s @ {args.hz:.0f} Hz  "
           f"(peak vel ~{peak_vel:.2f} rad/s = {np.degrees(peak_vel):.0f} deg/s)")

    # Countdown so the human has time to abort if they don't like the look of it.
    for i in range(int(args.countdown_s), 0, -1):
        _trace(f"  starting motion in {i}s... (Ctrl-C to abort)")
        time.sleep(1.0)

    stop = {"abort": False}
    def _sigint(_sig, _frame):
        stop["abort"] = True
        _trace("SIGINT -- stopping at next step.")
    signal.signal(signal.SIGINT, _sigint)

    _trace("ramping...")
    t0 = time.perf_counter()
    for i in range(1, n_steps + 1):
        if stop["abort"]:
            _trace(f"aborted at step {i}/{n_steps}")
            break
        alpha = i / n_steps  # 0 < alpha <= 1
        cmd = start + alpha * delta
        left.command_joint_pos(cmd[:7].astype(np.float32))
        right.command_joint_pos(cmd[7:].astype(np.float32))
        time.sleep(dt)
    elapsed = time.perf_counter() - t0
    _trace(f"ramp done in {elapsed:.2f}s")

    _trace("holding final pose for 2s so PD can settle...")
    time.sleep(2.0)

    q_l = np.asarray(left.get_joint_pos(),  dtype=np.float32)
    q_r = np.asarray(right.get_joint_pos(), dtype=np.float32)
    final = np.concatenate([q_l, q_r])
    _trace(f"final state                  = {np.array2string(final, precision=3)}")
    _trace(f"position error vs goal       = {np.array2string(final - goal, precision=3)}")

    try: left.close()
    except Exception: pass
    try: right.close()
    except Exception: pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
