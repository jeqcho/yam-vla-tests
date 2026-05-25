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
import logging
import os
import signal
import sys
import time
import glob
from pathlib import Path

# Silence i2rt + python-can chatter BEFORE importing the SDK, otherwise
# interactive prompts get buried under per-step INFO logs and CAN-close
# tracebacks.
_NOISY_LOGGERS = ("root", "i2rt", "i2rt.robots.utils", "i2rt.robots.get_robot",
                  "i2rt.robots.motor_chain_robot", "can",
                  "can.interfaces.socketcan", "can.interfaces.socketcan.socketcan")
logging.basicConfig(level=logging.ERROR, force=True)
for _n in _NOISY_LOGGERS:
    logging.getLogger(_n).setLevel(logging.ERROR)

import numpy as np

# Importing yam_client triggers install_sdk_lock_fix at module load.
sys.path.insert(0, "/home/andon/yam-tests/molmoact2-setup/scripts")
import yam_client  # noqa: F401  -- side effect

# Re-silence after yam_client may have re-configured logging.
logging.getLogger().setLevel(logging.ERROR)
for _n in _NOISY_LOGGERS:
    logging.getLogger(_n).setLevel(logging.ERROR)

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType


def _quiet_close(robot) -> None:
    """Close a robot and swallow shutdown-time CAN-thread spam.

    The background CAN thread races the socket teardown: as the fd goes
    to -1, in-flight bus.send() calls raise ValueError and the thread
    prints a multi-line traceback to stderr. Redirect stderr to /dev/null
    for the close + a short grace window so the next prompt stays readable.
    """
    null_f = open(os.devnull, "w")
    old_err_fd = os.dup(2)
    old_stderr = sys.stderr
    try:
        os.dup2(null_f.fileno(), 2)
        sys.stderr = null_f
        try:
            robot.close()
        except Exception:
            pass
        time.sleep(0.6)  # give background threads time to die quietly
    finally:
        sys.stderr = old_stderr
        os.dup2(old_err_fd, 2)
        os.close(old_err_fd)
        null_f.close()


CONFIG_PATH = Path("/home/andon/yam-tests/molmoact2-setup/yam_setup_config.json")
WRIST_ROLL_J = 5     # 0-indexed; spins end-effector in place, safest joint to wiggle
WIGGLE_AMP = 0.3     # rad ~ 17 deg
WIGGLE_CYCLES = 2    # was 3 at 0.7 Hz (~4.3 s) -- 2 at 1.0 Hz = 2 s, plenty
WIGGLE_HZ = 1.0      #   to see which arm moved
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

    # Safety: close both arms cleanly. _quiet_close suppresses the post-close
    # CAN tracebacks so the next prompt stays readable.
    _quiet_close(arm_a)
    _quiet_close(arm_b)
    print("\n[arms closed]\n", flush=True)
    return mapping


# ---------------- camera identification ----------------

def list_realsense() -> list[tuple[str, str]]:
    """Return [(serial, model_name), ...] for all RealSense devices.

    Model name comes from rs.camera_info.name, e.g. 'Intel RealSense D435'.
    This rig uses D435 (wide FOV) for the top/context camera and D405s for
    the wrist cameras; identify_cameras warns if that pairing isn't met.
    """
    import pyrealsense2 as rs
    ctx = rs.context()
    out = []
    for dev in ctx.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        try:
            model = dev.get_info(rs.camera_info.name)
        except Exception:
            model = "(unknown)"
        out.append((serial, model))
    return out


def list_v4l2_uvc() -> list[str]:
    """Return stable /dev/v4l/by-id/... paths for UVC webcams (non-RealSense).

    We deliberately return by-id symlinks rather than /dev/videoN, because
    v4l2 device numbering can shift between sessions (USB enumeration order
    is not stable when RealSense and webcams share root hubs). by-id paths
    are derived from the USB vendor/product string and are stable across
    re-enumeration.

    Only the first (index0) entry per camera is returned -- it's the capture
    interface; index1 is typically a metadata stream we don't want.
    """
    out = []
    seen_devices = set()
    byid_dir = "/dev/v4l/by-id"
    if os.path.isdir(byid_dir):
        for link in sorted(os.listdir(byid_dir)):
            if "realsense" in link.lower():
                continue
            if "-video-index0" not in link:
                continue
            full = os.path.join(byid_dir, link)
            try:
                target = os.path.realpath(full)  # e.g. /dev/video12
            except Exception:
                continue
            if target in seen_devices:
                continue
            seen_devices.add(target)
            out.append(full)
        if out:
            return out

    # Fallback: scan /sys/class/video4linux/ and use /dev/videoN paths.
    for d in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            name = open(os.path.join(d, "name")).read().strip().lower()
        except Exception:
            continue
        if "realsense" in name:
            continue
        node = "/dev/" + os.path.basename(d)
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
    rs_devices = list_realsense()                       # [(serial, model), ...]
    rs_serials = [s for s, _ in rs_devices]
    rs_model_by_serial = dict(rs_devices)
    uvc_devs = list_v4l2_uvc()
    print(f"  Found RealSense devices:")
    for serial, model in rs_devices:
        print(f"    {serial}  ({model})")
    print(f"  Found UVC (webcam) devices: {uvc_devs}")

    # Partition by model: this rig uses D435 (wide FOV) as the top/context
    # camera and D405 as wrist cameras. Picking top by model is far more
    # robust than triangulating it from the orange-cube test, which the user
    # has already been bitten by once.
    def _is(model: str, substr: str) -> bool:
        return substr in (model or "")

    d435s = [s for s, m in rs_devices if _is(m, "D435")]
    d405s = [s for s, m in rs_devices if _is(m, "D405")]

    # Top camera assignment (model-based).
    top_serial: Optional[str] = None
    top_v4l2: Optional[str] = None
    if d435s:
        top_serial = d435s[0]
        if len(d435s) > 1:
            print(f"  Note: {len(d435s)} D435s present; using {top_serial} for top.")
        print(f"  TOP camera (D435, by model): {top_serial}")
    elif uvc_devs:
        top_v4l2 = uvc_devs[0]
        if len(uvc_devs) > 1:
            print(f"  Multiple UVC devices; using first: {top_v4l2}.")
        print(f"  TOP camera (UVC fallback, no D435 found): {top_v4l2}")
    else:
        print("  WARNING: no D435 and no UVC webcam; top camera unset.")

    # Left/right candidates: the D405s. (If <2 D405s, fall back to any
    # remaining RealSense so the orange test can still run.)
    if len(d405s) >= 2:
        wrist_candidates = d405s
        if len(d405s) > 2:
            print(f"  Note: {len(d405s)} D405s present; only two are wrist "
                  f"cameras. The orange test will pick the two with the most "
                  f"orange; any extra D405 is unused.")
    else:
        print(f"  WARNING: expected 2 D405 wrist cameras, found {len(d405s)}. "
              f"Falling back to any non-top RealSense for wrist identification.")
        wrist_candidates = [s for s in rs_serials if s != top_serial]
        if len(wrist_candidates) < 2:
            raise RuntimeError(
                f"Need >=2 RealSense cameras for left/right identification; "
                f"after assigning top={top_serial}, only {len(wrist_candidates)} "
                f"remain."
            )

    def _orange(serial):
        try:
            return count_orange_pixels(grab_realsense_frame(serial))
        except Exception as e:
            print(f"  {serial}: capture failed: {e}")
            return -1

    # Test 1: cube on LEFT -- LEFT wrist cam sees the most orange.
    print("\nPut the bright orange cube on the LEFT side of the workspace.")
    print("Make sure it's clearly visible to the LEFT wrist camera.")
    input("Press Enter when ready...")
    counts_l = {s: _orange(s) for s in wrist_candidates}
    for s, n in counts_l.items():
        print(f"  {s}: {n} bright-orange pixels")
    if max(counts_l.values()) <= 0:
        print("ERROR: no orange detected by any wrist camera. Aborting camera ID.")
        return {"left_cam_serial": None, "right_cam_serial": None,
                "top_cam_serial": top_serial, "top_cam_v4l2": top_v4l2}
    left_serial = max(counts_l, key=counts_l.get)

    # Test 2: cube on RIGHT (verification when only 2 wrist candidates;
    # required disambiguation when there are extras).
    needs_disambiguation = len(wrist_candidates) > 2
    if needs_disambiguation:
        print(f"\nNow put the cube on the RIGHT side. Required: {len(wrist_candidates)} "
              f"wrist candidates -- need the right-cube test to pick which one is RIGHT.")
        input("Press Enter when ready...")
        ran_right = True
    else:
        ans = input("\nMove cube to the RIGHT side and press Enter to verify "
                    "(or 'skip' to skip verification): ").strip().lower()
        ran_right = not ans.startswith('s')

    if ran_right:
        counts_r = {s: _orange(s) for s in wrist_candidates}
        for s, n in counts_r.items():
            print(f"  {s}: {n} bright-orange pixels")
        rest_r = {s: c for s, c in counts_r.items() if s != left_serial}
        if not rest_r or max(rest_r.values()) <= 0:
            if needs_disambiguation:
                print("ERROR: no orange on right; can't disambiguate among "
                      "extra D405s. Aborting camera ID.")
                return {"left_cam_serial": left_serial, "right_cam_serial": None,
                        "top_cam_serial": top_serial, "top_cam_v4l2": top_v4l2}
            print("  No orange on right; falling back to elimination.")
            right_serial = next(s for s in wrist_candidates if s != left_serial)
        else:
            right_serial = max(rest_r, key=rest_r.get)
            if counts_r.get(left_serial, 0) > counts_r.get(right_serial, 0):
                print(f"  WARNING: identified-left {left_serial} saw more "
                      f"orange in the right-cube test than identified-right "
                      f"{right_serial}. Double-check the cube placement.")
    else:
        right_serial = next(s for s in wrist_candidates if s != left_serial)

    left_model  = rs_model_by_serial.get(left_serial,  "(unknown)")
    right_model = rs_model_by_serial.get(right_serial, "(unknown)")
    top_model   = rs_model_by_serial.get(top_serial,   "(unknown)") if top_serial else None
    print(f"\n  LEFT  arm camera : {left_serial}  ({left_model})")
    print(f"  RIGHT arm camera : {right_serial}  ({right_model})")
    if top_serial:
        print(f"  TOP    camera    : {top_serial}  ({top_model})")
    elif top_v4l2:
        print(f"  TOP    camera    : {top_v4l2}  (V4L2)")

    if not _is(left_model,  "D405"): print(
        f"  WARNING: LEFT  cam is {left_model}, expected D405 (wrist).")
    if not _is(right_model, "D405"): print(
        f"  WARNING: RIGHT cam is {right_model}, expected D405 (wrist).")
    if top_serial and not _is(top_model, "D435"): print(
        f"  WARNING: TOP   cam is {top_model}, expected D435 (context).")

    return {
        "left_cam_serial":  left_serial,
        "right_cam_serial": right_serial,
        "top_cam_serial":   top_serial,
        "top_cam_v4l2":     top_v4l2,
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

    old_config: dict = {}
    if CONFIG_PATH.exists():
        try:
            old_config = json.loads(CONFIG_PATH.read_text())
            print(f"Existing config found:\n  {json.dumps(old_config, indent=2)}")
        except Exception:
            print("Existing config exists but could not be parsed; will overwrite.")

    config = dict(old_config)
    if not args.skip_arms:
        config.update(identify_arms(args.can_a, args.can_b, args.gripper))
    if not args.skip_cameras:
        config.update(identify_cameras())

    config["gripper"] = args.gripper

    out_path = Path(args.out)
    out_path.write_text(json.dumps(config, indent=2) + "\n")

    print(f"\n=== Saved config ===\n  {out_path}\n")
    print(json.dumps(config, indent=2))

    print("\n=== Diff vs previous config ===")
    keys = sorted(set(old_config) | set(config))
    changes = 0
    for k in keys:
        ov, nv = old_config.get(k, "<unset>"), config.get(k, "<unset>")
        if ov != nv:
            print(f"  {k}: {ov}  ->  {nv}")
            changes += 1
    if not changes:
        print("  (no changes)")
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
