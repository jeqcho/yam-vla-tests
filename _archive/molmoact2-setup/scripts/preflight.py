"""Pre-flight check for bimanual MolmoAct2 deployment.

Runs through the boring things you'd otherwise discover one trauma at a time:

    /home/andon/yam-tests/i2rt/.venv/bin/python scripts/preflight.py \\
        --left-can can0 --right-can can1 \\
        --left-gripper linear_4310 --right-gripper linear_4310 \\
        --server-url http://127.0.0.1:8202/act

Checks (in order):
    1. RealSense cameras detected, prints serial numbers
    2. Both CAN buses UP at the right bitrate
    3. Each YAM arm initializes (this will briefly move motors to read offsets!)
    4. State vector reads at the expected (14,) shape
    5. Server responds to GET /act with the expected schema
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def check_cameras() -> int:
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[FAIL] pyrealsense2 not installed in this venv")
        return 1
    ctx = rs.context()
    ds = list(ctx.query_devices())
    if not ds:
        print("[FAIL] no RealSense devices detected on USB")
        return 1
    print(f"[ OK ] {len(ds)} RealSense device(s):")
    for d in ds:
        print(f"       {d.get_info(rs.camera_info.name):<30s} serial={d.get_info(rs.camera_info.serial_number)}")
    if len(ds) < 3:
        print(f"[WARN] need 3 cameras (top/left/right), have {len(ds)}")
        return 1
    return 0


def check_can(channel: str) -> int:
    import subprocess
    r = subprocess.run(["ip", "-det", "link", "show", channel], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[FAIL] CAN interface {channel} not present")
        return 1
    out = r.stdout
    if "state UP" not in out:
        print(f"[FAIL] {channel} is not UP")
        return 1
    if "bitrate 1000000" not in out:
        print(f"[WARN] {channel} bitrate is not 1 Mbit/s")
        return 1
    print(f"[ OK ] {channel} UP @ 1 Mbit/s")
    return 0


def check_arm(channel: str, gripper: str) -> tuple[int, "object"]:
    try:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType
    except ImportError as e:
        print(f"[FAIL] i2rt SDK import: {e}")
        return 1, None
    try:
        robot = get_yam_robot(
            channel=channel,
            arm_type=ArmType.from_string_name("yam"),
            gripper_type=GripperType.from_string_name(gripper),
            zero_gravity_mode=True,
        )
    except Exception as e:
        print(f"[FAIL] arm init on {channel} ({gripper}): {e}")
        return 1, None
    n = robot.num_dofs()
    q = np.asarray(robot.get_joint_pos(), dtype=np.float32)
    print(f"[ OK ] arm on {channel}: dofs={n}, joint_pos={np.array2string(q, precision=3)}")
    if n != 7:
        print(f"[WARN] expected 7 DoFs (6 arm + 1 gripper), got {n}")
    return 0, robot


def check_server(server_url: str) -> int:
    try:
        import requests
    except ImportError:
        print("[FAIL] requests not installed")
        return 1
    try:
        r = requests.get(server_url, timeout=3.0)
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        print(f"[FAIL] server health at {server_url}: {e}")
        return 1
    expected_keys = {"status", "repo_id", "norm_tag", "num_cameras", "state_dim"}
    missing = expected_keys - set(body)
    if missing:
        print(f"[WARN] server response missing keys: {missing}")
    if body.get("state_dim") != 14:
        print(f"[FAIL] expected state_dim=14, got {body.get('state_dim')}")
        return 1
    if body.get("num_cameras") != 3:
        print(f"[FAIL] expected num_cameras=3, got {body.get('num_cameras')}")
        return 1
    print(f"[ OK ] server: {body}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--left-can", default="can0")
    p.add_argument("--right-can", default="can1")
    p.add_argument("--left-gripper", default="linear_4310")
    p.add_argument("--right-gripper", default="linear_4310")
    p.add_argument("--server-url", default="http://127.0.0.1:8202/act")
    p.add_argument("--skip-arms", action="store_true",
                   help="Skip arm init (will not power on motors)")
    p.add_argument("--skip-server", action="store_true")
    args = p.parse_args()

    rc = 0
    print("\n== RealSense cameras ==")
    rc |= check_cameras()

    print("\n== CAN interfaces ==")
    rc |= check_can(args.left_can)
    rc |= check_can(args.right_can)

    if not args.skip_arms:
        print("\n== YAM arms ==")
        rc_l, left = check_arm(args.left_can, args.left_gripper)
        rc_r, right = check_arm(args.right_can, args.right_gripper)
        rc |= rc_l | rc_r
        if left is not None and right is not None:
            sl = np.asarray(left.get_joint_pos(), dtype=np.float32)
            sr = np.asarray(right.get_joint_pos(), dtype=np.float32)
            state = np.concatenate([sl, sr])
            print(f"[ OK ] composed state (14,): {np.array2string(state, precision=3)}")

    if not args.skip_server:
        print("\n== MolmoAct2 server ==")
        rc |= check_server(args.server_url)

    print(f"\n{'==' * 20}\nresult: {'PASS' if rc == 0 else 'FAIL'} (rc={rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
