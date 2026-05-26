"""Pre-flight check for the bimanual YAM + GR00T N1.7 inference setup.

Verifies (in order):
    1. RealSense cameras enumerate (skipped if --no-cams)
    2. CAN buses are up at 1 Mbit/s
    3. Both arms initialize (this triggers gripper auto-cal — clear the jaws!)
    4. (optional) The GR00T policy server responds to ping

Run with the i2rt venv:
    /home/andon/yam-tests/i2rt/.venv/bin/python scripts/preflight.py \\
        --left-can can0 --right-can can1 \\
        --left-gripper linear_4310 --right-gripper linear_4310

WARNING: arm init runs gripper auto-calibration on linear_4310 / linear_3507 /
flexible_4310. The gripper will open and close. Clear the jaws first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Optional

import numpy as np


def check_cameras() -> bool:
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[preflight] pyrealsense2 not installed in this venv")
        return False
    ctx = rs.context()
    devs = list(ctx.query_devices())
    if not devs:
        print("[preflight] FAIL: no RealSense cameras detected")
        return False
    print(f"[preflight] OK: {len(devs)} RealSense device(s) detected:")
    for d in devs:
        print(f"           {d.get_info(rs.camera_info.name)} "
              f"serial={d.get_info(rs.camera_info.serial_number)} "
              f"usb={d.get_info(rs.camera_info.usb_type_descriptor)}")
    return True


def check_can(channel: str) -> bool:
    try:
        out = subprocess.check_output(
            ["ip", "-details", "link", "show", channel],
            stderr=subprocess.STDOUT, timeout=5
        ).decode()
    except subprocess.CalledProcessError as e:
        print(f"[preflight] FAIL: {channel} not found ({e.output.decode().strip()})")
        return False
    if "UP" not in out.split("\n")[0]:
        print(f"[preflight] FAIL: {channel} is not UP")
        return False
    if "1000000" not in out:
        print(f"[preflight] WARN: {channel} bitrate is not 1 Mbit/s — proceeding anyway")
    print(f"[preflight] OK: {channel} is UP")
    return True


def check_arm(can_channel: str, gripper: str) -> bool:
    try:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType
    except ImportError as e:
        print(f"[preflight] FAIL: i2rt not importable in this venv: {e}")
        return False
    try:
        print(f"[preflight] Initializing arm on {can_channel} (gripper={gripper})...")
        arm = get_yam_robot(
            channel=can_channel,
            arm_type=ArmType.from_string_name("yam"),
            gripper_type=GripperType.from_string_name(gripper),
            zero_gravity_mode=False,
        )
        q = np.asarray(arm.get_joint_pos(), dtype=np.float32)
        print(f"[preflight] OK: {can_channel} arm initialized, q={np.array2string(q, precision=3)}")
        try:
            arm.close()
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"[preflight] FAIL: {can_channel} arm init failed: {e}")
        return False


def check_server(host: str, port: int, timeout_s: float = 3.0) -> bool:
    try:
        import msgpack_numpy as mnp
        import zmq
    except ImportError as e:
        print(f"[preflight] WARN: zmq/msgpack-numpy missing in this venv: {e}")
        return False
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
    sock.connect(f"tcp://{host}:{port}")
    try:
        sock.send(mnp.packb({"endpoint": "ping"}))
        msg = sock.recv()
        resp = mnp.unpackb(msg, raw=False)
        print(f"[preflight] OK: server ping at {host}:{port} -> {resp}")
        return True
    except Exception as e:
        print(f"[preflight] WARN: server not reachable at {host}:{port}: {e}")
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--left-can", default="can0")
    p.add_argument("--right-can", default="can1")
    p.add_argument("--left-gripper", default="linear_4310")
    p.add_argument("--right-gripper", default="linear_4310")
    p.add_argument("--no-cams", action="store_true")
    p.add_argument("--no-arms", action="store_true")
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=5556)
    p.add_argument("--skip-server", action="store_true")
    args = p.parse_args()

    ok = True
    if not args.no_cams:
        ok &= check_cameras()
    ok &= check_can(args.left_can)
    ok &= check_can(args.right_can)
    if not args.no_arms:
        ok &= check_arm(args.left_can, args.left_gripper)
        ok &= check_arm(args.right_can, args.right_gripper)
    if not args.skip_server:
        check_server(args.server_host, args.server_port)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
