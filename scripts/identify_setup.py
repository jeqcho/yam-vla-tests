"""Interactive setup-identification script.

Identifies:
  1. Which CAN channel is the LEFT arm vs RIGHT arm
     (moves one arm at a time, you tell it which moved)
  2. Which RealSense serial is the LEFT camera vs RIGHT camera
     (you place a bright orange object on one side; the script uses HSV
     pixel counting to figure out which camera sees it most)
  3. Which V4L2 device is the TOP (webcam) camera

Saves all of this to yam_setup_config.json next to this script's parent dir.
Subsequent inference runs can read that JSON instead of you typing the flags.

Run once after assembling the rig (or if you suspect something changed):

    /home/andon/yam-tests/i2rt/.venv/bin/python \\
        /home/andon/yam-tests/molmoact2-setup/scripts/identify_setup.py

Always-on safety: each arm is captured and ramped back to its startup pose
before close. SDK lock fix is auto-applied via yam_client import.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import glob
from pathlib import Path

import numpy as np

# Importing yam_client triggers install_sdk_lock_fix at module load.
sys.path.insert(0, "/home/andon/yam-tests/molmoact2-setup/scripts")
import yam_client  # noqa: F401  -- side effect

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType


CONFIG_PATH = Path("/home/andon/yam-tests/molmoact2-setup/yam_setup_config.json")
WRIST_ROLL_J = 5     # 0-indexed; spins end-effector in place, safest joint to wiggle
WIGGLE_AMP = 0.3     # rad ~ 17 deg
WIGGLE_CYCLES = 3
WIGGLE_HZ = 0.7
CMD_HZ = 30.0


# ---------------- arm identification ----------------

def init_arm_no_gripcal(channel: str, gripper: str):
    """Init arm but skip gripper auto-cal (faster, no gripper motion)."""
    return get_yam_robot(
        channel=channel,
        arm_type=ArmType.from_string_name("yam"),
        gripper_type=GripperType.from_string_name(gripper),
        zero_gravity_mode=False,
        gripper_limits_override=np.array([0.0, 1.0]),  # skip cal
    )


def wiggle_wrist(robot, amp: float = WIGGLE_AMP, cycles: int = WIGGLE_CYCLES,
                 freq_hz: float = WIGGLE_HZ) -> None:
    startup = np.asarray(robot.get_joint_pos(), dtype=np.float32)
    cmd = startup.copy()
    dt = 1.0 / CMD_HZ
    duration_s = cycles / freq_hz
    n = int(duration_s * CMD_HZ)
    omega = 2.0 * np.pi * freq_hz
    t0 = time.perf_counter()
    next_tick = t0
    for _ in range(n):
        t = time.perf_counter() - t0
        cmd[WRIST_ROLL_J] = startup[WRIST_ROLL_J] + amp * np.sin(omega * t)
        robot.command_joint_pos(cmd.astype(np.float32))
        next_tick += dt
        s = next_tick - time.perf_counter()
        if s > 0:
            time.sleep(s)
    # ramp back to start
    ramp_n = int(1.0 * CMD_HZ)
    cur = np.asarray(robot.get_joint_pos(), dtype=np.float32)
    for i in range(1, ramp_n + 1):
        alpha = i / ramp_n
        c = cur + alpha * (startup - cur)
        robot.command_joint_pos(c.astype(np.float32))
        time.sleep(dt)
    time.sleep(0.3)


def ask_lr(prompt: str) -> str:
    while True:
        ans = input(prompt).strip().lower()
        if ans.startswith('l'):
            return 'l'
        if ans.startswith('r'):
            return 'r'
        print("Please answer L or R.")


def identify_arms(can_a: str, can_b: str, gripper: str) -> dict:
    print(f"\n=== Arm identification ===")
    print(f"  channel A = {can_a}")
    print(f"  channel B = {can_b}")
    print(f"\nInitializing both arms (skipping gripper auto-cal for speed)...", flush=True)
    arm_a = init_arm_no_gripcal(can_a, gripper)
    arm_b = init_arm_no_gripcal(can_b, gripper)
    time.sleep(0.5)
    print("Both arms held in position-holding mode.")

    input(f"\nI will wiggle the wrist-roll of the arm on {can_a}. "
          f"Watch the arms. Press Enter to start...")
    wiggle_wrist(arm_a)

    ans = ask_lr(f"\nWhich physical arm just wiggled its wrist? [L]eft or [R]ight: ")
    if ans == 'l':
        mapping = {"left_can": can_a, "right_can": can_b}
    else:
        mapping = {"left_can": can_b, "right_can": can_a}
    print(f"\nRecorded: left={mapping['left_can']}  right={mapping['right_can']}")

    # Safety: close both arms cleanly (each arm holds its startup pose; close()
    # zeros torques but the pose is the user's chosen startup, which they
    # accepted as droppable when they began the script).
    try: arm_a.close()
    except Exception: pass
    try: arm_b.close()
    except Exception: pass
    return mapping


# ---------------- camera identification ----------------

def list_realsense() -> list[str]:
    import pyrealsense2 as rs
    ctx = rs.context()
    return [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]


def list_v4l2_uvc() -> list[str]:
    """Return /dev/videoN paths that are UVC webcams (not RealSense)."""
    out = []
    seen_devices = set()
    for d in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            name = open(os.path.join(d, "name")).read().strip().lower()
        except Exception:
            continue
        if "realsense" in name:
            continue
        node = "/dev/" + os.path.basename(d)
        # Many webcams expose video0 and video1 (capture + metadata); we want
        # the first per unique device. Use sysfs symlink to dedupe.
        try:
            dev_path = os.path.realpath(os.path.join(d, "device"))
        except Exception:
            dev_path = d
        if dev_path in seen_devices:
            continue
        seen_devices.add(dev_path)
        out.append(node)
    return out


def grab_realsense_frame(serial: str, w: int = 640, h: int = 480, fps: int = 30) -> np.ndarray:
    import pyrealsense2 as rs
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
    pipe = rs.pipeline()
    pipe.start(cfg)
    # warmup
    for _ in range(5):
        try: pipe.wait_for_frames(timeout_ms=2000)
        except Exception: pass
    frames = pipe.wait_for_frames(timeout_ms=2000)
    color = frames.get_color_frame()
    img = np.asanyarray(color.get_data())
    pipe.stop()
    return img


def grab_v4l2_frame(dev: str, w: int = 640, h: int = 480, fps: int = 30) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(5):
        cap.read()
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read from {dev}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def count_orange_pixels(img_rgb: np.ndarray) -> int:
    """Count pixels matching a "bright orange" HSV signature."""
    import cv2
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # Orange: H 5-25 (out of 180), high saturation, high value
    mask = cv2.inRange(hsv, (5, 120, 120), (25, 255, 255))
    return int(np.count_nonzero(mask))


def identify_cameras() -> dict:
    print(f"\n=== Camera identification ===")
    rs_serials = list_realsense()
    uvc_devs = list_v4l2_uvc()
    print(f"  Found RealSense serials: {rs_serials}")
    print(f"  Found UVC (webcam) devices: {uvc_devs}")

    if len(rs_serials) != 2:
        print(f"WARNING: expected 2 RealSense cameras, found {len(rs_serials)}.")
        if len(rs_serials) < 2:
            raise RuntimeError("Cannot identify left/right without 2 RealSense cameras.")

    if not uvc_devs:
        print("WARNING: no UVC webcam found; top camera will be left unset.")
        top_dev = None
    else:
        if len(uvc_devs) > 1:
            print(f"  Multiple UVC devices; using first: {uvc_devs[0]}")
        top_dev = uvc_devs[0]

    # Place orange cube on LEFT, identify left D405.
    print("\nPut the bright orange cube on the LEFT side of the workspace.")
    print("Make sure it's clearly visible to whichever camera covers the left arm.")
    input("Press Enter when ready...")

    counts = {}
    for serial in rs_serials:
        try:
            img = grab_realsense_frame(serial)
            n = count_orange_pixels(img)
            counts[serial] = n
            print(f"  {serial}: {n} bright-orange pixels")
        except Exception as e:
            print(f"  {serial}: capture failed: {e}")
            counts[serial] = -1

    if max(counts.values()) <= 0:
        print("ERROR: no orange detected by either camera. Aborting camera ID.")
        return {"top_cam_v4l2": top_dev, "left_cam_serial": None, "right_cam_serial": None}

    left_serial = max(counts, key=counts.get)
    right_serial = next(s for s in rs_serials if s != left_serial)
    print(f"\n  Identified LEFT camera (more orange) : {left_serial}")
    print(f"  Identified RIGHT camera (less orange): {right_serial}")

    # Optional sanity check: cube on right.
    ans = input("\nMove cube to the RIGHT side and press Enter to verify "
                "(or type 'skip' to skip verification): ").strip().lower()
    if not ans.startswith('s'):
        counts2 = {}
        for serial in rs_serials:
            try:
                img = grab_realsense_frame(serial)
                n = count_orange_pixels(img)
                counts2[serial] = n
                print(f"  {serial}: {n} bright-orange pixels")
            except Exception as e:
                print(f"  {serial}: capture failed: {e}")
                counts2[serial] = -1
        if max(counts2.values()) > 0:
            verify_right_serial = max(counts2, key=counts2.get)
            if verify_right_serial == right_serial:
                print(f"  Verified: {right_serial} is RIGHT (more orange on second test).")
            else:
                print(f"  WARNING: cube-on-right test pointed at {verify_right_serial}, "
                      f"but first test said {right_serial} is right.")
                print(f"  Two tests disagree; check that the cube was clearly on one side only.")

    return {
        "top_cam_v4l2": top_dev,
        "left_cam_serial": left_serial,
        "right_cam_serial": right_serial,
    }


# ---------------- main ----------------

def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--can-a", default="can0")
    p.add_argument("--can-b", default="can1")
    p.add_argument("--gripper", default="linear_4310",
                   help="gripper type (both arms assumed the same)")
    p.add_argument("--skip-arms", action="store_true",
                   help="skip arm identification (only do cameras)")
    p.add_argument("--skip-cameras", action="store_true",
                   help="skip camera identification (only do arms)")
    p.add_argument("--out", default=str(CONFIG_PATH))
    args = p.parse_args()

    print("This will identify your CAN/arm and camera mappings and save them "
          f"to:\n  {args.out}\n")

    config = {}
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text())
            print(f"Existing config found:\n  {json.dumps(config, indent=2)}")
        except Exception:
            print("Existing config exists but could not be parsed; will overwrite.")

    if not args.skip_arms:
        config.update(identify_arms(args.can_a, args.can_b, args.gripper))
    if not args.skip_cameras:
        config.update(identify_cameras())

    config["gripper"] = args.gripper

    out_path = Path(args.out)
    out_path.write_text(json.dumps(config, indent=2) + "\n")

    print(f"\n=== Saved config ===\n  {out_path}\n")
    print(json.dumps(config, indent=2))
    print(f"\nUse these flags in run_client.sh:")
    flags = []
    if config.get("left_can") and config.get("right_can"):
        flags.append(f"--left-can {config['left_can']} --right-can {config['right_can']}")
    if config.get("gripper"):
        flags.append(f"--left-gripper {config['gripper']} --right-gripper {config['gripper']}")
    if config.get("top_cam_v4l2"):
        flags.append(f"--top-cam-v4l2 {config['top_cam_v4l2']}")
    if config.get("left_cam_serial"):
        flags.append(f"--left-cam-serial {config['left_cam_serial']}")
    if config.get("right_cam_serial"):
        flags.append(f"--right-cam-serial {config['right_cam_serial']}")
    for f in flags:
        print(f"  {f}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
