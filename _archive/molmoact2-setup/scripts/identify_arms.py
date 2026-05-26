"""Identify which physical arm is on which CAN bus, without moving anything.

Initializes both arms with auto-calibration DISABLED (so grippers don't open
/close), drops both into zero-torque mode (so motors don't hold position),
then prints joint angles for both arms continuously. Physically move one arm
by hand and see which CAN bus shows joint deltas.

    /home/andon/yam-tests/i2rt/.venv/bin/python scripts/identify_arms.py \\
        --can-a can0 --can-b can1

The motor-controlled joints are 6-DoF arm + 1-DoF gripper. With zero-torque
mode the arm will sag under gravity unless you hold it — so support each
arm while you wiggle a joint to test.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def init(channel: str, gripper: str) -> "object":
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType
    # gripper_limits_override skips auto-calibration so the gripper does not move at init.
    robot = get_yam_robot(
        channel=channel,
        arm_type=ArmType.from_string_name("yam"),
        gripper_type=GripperType.from_string_name(gripper),
        zero_gravity_mode=True,
        gripper_limits_override=np.array([0.0, 1.0]),
    )
    robot.zero_torque_mode()
    return robot


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--can-a", default="can0")
    p.add_argument("--can-b", default="can1")
    p.add_argument("--gripper", default="linear_4310")
    p.add_argument("--rate-hz", type=float, default=5.0)
    args = p.parse_args()

    print(f"Initializing arm A on {args.can_a} and arm B on {args.can_b} "
          f"(gripper={args.gripper}, no auto-cal, zero-torque)...")
    print("WARNING: arms will sag under gravity. Support each one before you let go.")
    a = init(args.can_a, args.gripper)
    b = init(args.can_b, args.gripper)
    print("\nReady. Now physically move one arm and watch which row's joints change.\n"
          "Ctrl+C to exit.\n")

    last_a = np.asarray(a.get_joint_pos(), dtype=np.float32)
    last_b = np.asarray(b.get_joint_pos(), dtype=np.float32)
    dt = 1.0 / args.rate_hz

    try:
        while True:
            qa = np.asarray(a.get_joint_pos(), dtype=np.float32)
            qb = np.asarray(b.get_joint_pos(), dtype=np.float32)
            da = qa - last_a
            db = qb - last_b
            print(
                f"A({args.can_a}): {np.array2string(qa, precision=2, suppress_small=True)}  "
                f"|d|={np.max(np.abs(da)):.3f}     "
                f"B({args.can_b}): {np.array2string(qb, precision=2, suppress_small=True)}  "
                f"|d|={np.max(np.abs(db)):.3f}",
                flush=True,
            )
            last_a, last_b = qa, qb
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nexit")
        return 0


if __name__ == "__main__":
    sys.exit(main())
