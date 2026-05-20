"""Bimanual YAM client for the MolmoAct2-BimanualYAM inference server.

Wire-format reference: examples/yam/host_server_yam.py in allenai/molmoact2.
Run with the i2rt venv:

    /home/andon/yam-tests/i2rt/.venv/bin/python scripts/yam_client.py \\
        --left-can can0 --right-can can1 \\
        --left-gripper linear_4310 --right-gripper linear_4310 \\
        --top-cam-serial AAAA --left-cam-serial BBBB --right-cam-serial CCCC \\
        --server-url http://127.0.0.1:8202/act \\
        --instruction "first pick up the left orange cube and put it in the box, then pick up the right orange cube and put it in the box" \\
        --train-fps 30 --horizon-stride 6 --max-step-rad 0.05 --gripper-step 0.05

Safety: every command is clipped to within --max-step-rad of the current state
per arm joint, and the gripper is clipped to --gripper-step per step. Ctrl+C
stops the loop and exits; the arms hold their last commanded position — kill
power if that pose isn't safe.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import json_numpy
import numpy as np
import requests

# i2rt imports — provided by the i2rt venv (/home/andon/yam-tests/i2rt/.venv).
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmType, GripperType

json_numpy.patch()

# Make stdout unbuffered so we can actually see where things hang.
# i2rt's logger may have already called basicConfig at import time; force-add
# our own StreamHandler so our messages always appear with timestamps.
import os
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s | %(message)s"))
_handler.setLevel(logging.INFO)
# Avoid duplicate handlers if the module gets reloaded.
if not any(getattr(h, "_yam_client_handler", False) for h in _root.handlers):
    _handler._yam_client_handler = True
    _root.addHandler(_handler)

log = logging.getLogger("yam.client")


def trace(msg: str) -> None:
    """Always-flushed marker print so we see where the script is in real time."""
    print(f"[TRACE] {msg}", flush=True)


# Default per-step caps (radians for joints, normalized for gripper).
# Tuned conservatively — increase only after the policy looks safe.
DEFAULT_MAX_STEP_RAD = 0.05
DEFAULT_GRIPPER_STEP = 0.05
DEFAULT_TRAIN_FPS = 30.0   # the policy's training cadence — controls inner-loop pace
DEFAULT_HORIZON_STRIDE = 6 # play this many steps from each (30, 14) horizon before re-querying
STATE_DIM = 14   # per-arm 7-D × 2
ARM_DOFS = 7     # 6 arm joints + 1 gripper


class CameraStream:
    """Base camera interface — start, grab one HxWx3 uint8 RGB frame, stop."""

    def __init__(self, name: str, width: int = 640, height: int = 480, fps: int = 30):
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps

    def start(self) -> None: raise NotImplementedError
    def grab(self) -> np.ndarray: raise NotImplementedError
    def stop(self) -> None: raise NotImplementedError


class RealSenseStream(CameraStream):
    """RealSense color stream via librealsense."""

    def __init__(self, serial: str, name: str, **kw):
        super().__init__(name, **kw)
        self.serial = serial
        self.pipeline = None

    def start(self) -> None:
        import pyrealsense2 as rs
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.pipeline = rs.pipeline()
        self.pipeline.start(cfg)
        # D405 sometimes takes >1s to produce its first frame after start().
        # Drop a few warmup frames with a generous timeout so the first real
        # grab() doesn't time out (cf. scripts/capture_frames.py).
        for _ in range(5):
            try:
                self.pipeline.wait_for_frames(timeout_ms=2000)
            except Exception:
                pass
        log.info("camera %s (RealSense %s) started @ %dx%d/%d Hz", self.name, self.serial,
                 self.width, self.height, self.fps)

    def grab(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"camera {self.name} ({self.serial}) produced no color frame")
        return np.asanyarray(color.get_data())

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()


class V4L2Stream(CameraStream):
    """Generic UVC / V4L2 webcam via OpenCV. Used for non-RealSense cameras."""

    def __init__(self, device: str, name: str, **kw):
        super().__init__(name, **kw)
        self.device = device
        self.cap = None

    def start(self) -> None:
        import cv2
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open {self.device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Discard a few frames so AE settles.
        for _ in range(5):
            self.cap.read()
        log.info("camera %s (V4L2 %s) started @ %dx%d/%d Hz", self.name, self.device,
                 self.width, self.height, self.fps)

    def grab(self) -> np.ndarray:
        import cv2
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"camera {self.name} ({self.device}) produced no frame")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()


def make_camera(name: str, serial: Optional[str], v4l2_device: Optional[str],
                width: int, height: int, fps: int) -> CameraStream:
    """Build the right camera backend based on which CLI flag was set."""
    if serial and v4l2_device:
        raise ValueError(f"{name}: pass exactly one of --{name}-cam-serial / --{name}-cam-v4l2")
    if not serial and not v4l2_device:
        raise ValueError(f"{name}: must pass --{name}-cam-serial or --{name}-cam-v4l2")
    if serial:
        return RealSenseStream(serial, name, width=width, height=height, fps=fps)
    return V4L2Stream(v4l2_device, name, width=width, height=height, fps=fps)


def init_arm(can_channel: str, gripper: str, ee_mass: Optional[float] = None) -> Robot:  # noqa: D401
    trace(f"init_arm({can_channel}, {gripper}): entering get_yam_robot...")
    """Create a YAM follower robot in position-holding mode (kp != 0).

    NOTE: this deliberately does NOT use the SDK's zero_gravity_mode. That mode
    sets kp=0 and relies on gravity feedforward only; if gravity comp is even
    slightly mis-tuned the arm drifts under gravity. We want the arm to actively
    hold whatever pose it has when the script starts. command_joint_pos() then
    drives it from there.
    """
    arm_type = ArmType.from_string_name("yam")
    gripper_type = GripperType.from_string_name(gripper)
    trace(f"init_arm({can_channel}): calling get_yam_robot (may take ~3-5s incl gripper auto-cal)")
    robot = get_yam_robot(
        channel=can_channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
        zero_gravity_mode=False,
        ee_mass=ee_mass,
    )
    trace(f"init_arm({can_channel}): get_yam_robot returned, reading joint pos")
    q0 = np.asarray(robot.get_joint_pos(), dtype=np.float32)
    trace(f"init_arm({can_channel}): joint_pos={np.array2string(q0, precision=3)}, commanding hold")
    robot.command_joint_pos(q0)
    trace(f"init_arm({can_channel}): done")
    return robot


def read_state(left: Robot, right: Robot) -> np.ndarray:
    """Compose the 14-D state vector: [left_q6+grip, right_q6+grip]."""
    s_l = np.asarray(left.get_joint_pos(), dtype=np.float32).reshape(-1)
    s_r = np.asarray(right.get_joint_pos(), dtype=np.float32).reshape(-1)
    if s_l.shape != (ARM_DOFS,) or s_r.shape != (ARM_DOFS,):
        raise RuntimeError(
            f"expected ({ARM_DOFS},) per arm, got left={s_l.shape}, right={s_r.shape}"
        )
    return np.concatenate([s_l, s_r], axis=0).astype(np.float32)


def safe_command(
    left: Robot,
    right: Robot,
    current_state: np.ndarray,
    desired_action: np.ndarray,
    max_step_rad: float,
    gripper_step: float,
) -> np.ndarray:
    """Clip the desired action so each joint moves at most max_step_rad from
    the current state in this tick. Returns the actually applied command.
    """
    if desired_action.shape != (STATE_DIM,):
        raise ValueError(f"action shape {desired_action.shape} != ({STATE_DIM},)")
    delta = desired_action - current_state
    # Per-arm caps: indices 0..5 + 7..12 are arm joints, 6 + 13 are grippers.
    caps = np.full(STATE_DIM, max_step_rad, dtype=np.float32)
    caps[6] = gripper_step
    caps[13] = gripper_step
    clipped_delta = np.clip(delta, -caps, caps)
    cmd = (current_state + clipped_delta).astype(np.float32)
    left.command_joint_pos(cmd[:ARM_DOFS])
    right.command_joint_pos(cmd[ARM_DOFS:])
    return cmd


def post_actions(
    server_url: str,
    top: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    state: np.ndarray,
    instruction: str,
    num_steps: int,
    timeout_s: float,
) -> tuple[np.ndarray, float]:
    """Round-trip one /act call. Returns (actions[N, D], dt_ms)."""
    payload = {
        "top_cam": top,
        "left_cam": left_img,
        "right_cam": right_img,
        "instruction": instruction,
        "state": state,
        "num_steps": num_steps,
        "timestamp": time.time(),
    }
    body = json_numpy.dumps(payload)
    t0 = time.perf_counter()
    resp = requests.post(server_url, data=body, headers={"Content-Type": "application/json"},
                         timeout=timeout_s)
    resp.raise_for_status()
    out = json_numpy.loads(resp.text)
    if "actions" not in out:
        raise RuntimeError(f"server response missing 'actions': keys={list(out.keys())}")
    actions = np.asarray(out["actions"], dtype=np.float32)
    server_dt_ms = float(out.get("dt_ms", 0.0))
    rtt_ms = (time.perf_counter() - t0) * 1000.0
    log.debug("server dt=%.1f ms, rtt=%.1f ms, actions shape=%s",
              server_dt_ms, rtt_ms, actions.shape)
    return actions, rtt_ms


def main() -> None:
    p = argparse.ArgumentParser(description="MolmoAct2-BimanualYAM client")
    p.add_argument("--left-can", default="can0", help="CAN channel for the LEFT arm")
    p.add_argument("--right-can", default="can1", help="CAN channel for the RIGHT arm")
    p.add_argument("--left-gripper", default="linear_4310",
                   choices=["crank_4310", "linear_3507", "linear_4310", "flexible_4310"],
                   help="Gripper type on the left arm")
    p.add_argument("--right-gripper", default="linear_4310",
                   choices=["crank_4310", "linear_3507", "linear_4310", "flexible_4310"],
                   help="Gripper type on the right arm")
    # Per-camera: pass exactly one of --<slot>-cam-serial (RealSense) or --<slot>-cam-v4l2 (UVC webcam, e.g. /dev/video0)
    p.add_argument("--top-cam-serial",   default=None, help="RealSense serial for overhead (top) camera")
    p.add_argument("--top-cam-v4l2",     default=None, help="V4L2 device path for overhead (top) camera, e.g. /dev/video0")
    p.add_argument("--left-cam-serial",  default=None, help="RealSense serial for left-arm camera")
    p.add_argument("--left-cam-v4l2",    default=None, help="V4L2 device path for left-arm camera")
    p.add_argument("--right-cam-serial", default=None, help="RealSense serial for right-arm camera")
    p.add_argument("--right-cam-v4l2",   default=None, help="V4L2 device path for right-arm camera")
    # Bandwidth-tunable camera config. Defaults sized for two D405s on USB 2.0
    # (~9.2 MB/s each at 424x240 RGB8 / 30 fps -- 18.4 MB/s total, fits the
    # ~40 MB/s practical ceiling of USB 2.0 with headroom for CAN + webcam).
    # Bump these if the cameras land on a USB 3.0 controller.
    p.add_argument("--cam-width",  type=int, default=424)
    p.add_argument("--cam-height", type=int, default=240)
    p.add_argument("--cam-fps",    type=int, default=30)
    p.add_argument("--server-url", default="http://127.0.0.1:8202/act",
                   help="MolmoAct2 server /act endpoint")
    p.add_argument("--instruction", required=True,
                   help="Natural-language task; e.g. 'first pick up the left orange cube and put it in the box, then pick up the right orange cube and put it in the box'")
    p.add_argument("--train-fps", type=float, default=DEFAULT_TRAIN_FPS,
                   help="Policy training cadence — inner loop sleeps 1/train_fps between commands")
    p.add_argument("--num-steps", type=int, default=10,
                   help="Flow-matching steps (server-side)")
    p.add_argument("--max-step-rad", type=float, default=DEFAULT_MAX_STEP_RAD,
                   help="Per-joint per-tick clip (rad)")
    p.add_argument("--gripper-step", type=float, default=DEFAULT_GRIPPER_STEP,
                   help="Gripper per-tick clip (normalized units)")
    p.add_argument("--horizon-stride", type=int, default=DEFAULT_HORIZON_STRIDE,
                   help="Apply this many steps from each returned horizon before re-querying. "
                        "With train_fps=30 and stride=6, server is queried 5 Hz.")
    p.add_argument("--timeout-s", type=float, default=15.0,
                   help="HTTP timeout per /act call (steady state). The first call after the server "
                        "comes up may take >5s because the model re-captures CUDA graphs for the "
                        "client's specific image shape; the script's explicit warmup uses a longer "
                        "timeout of its own.")
    p.add_argument("--warmup-timeout-s", type=float, default=60.0,
                   help="HTTP timeout for the one-shot warmup call. Bump if the server is loading.")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't command the arms; print actions only")
    args = p.parse_args()

    # Health-check the server first so we fail fast.
    health_url = args.server_url.rstrip("/").rsplit("/", 1)[0] + "/act" if args.server_url.endswith("/act") else args.server_url
    try:
        r = requests.get(health_url, timeout=3.0)
        r.raise_for_status()
        log.info("server health: %s", r.json())
    except Exception as e:
        log.error("server health check failed at %s: %s", health_url, e)
        sys.exit(2)

    # Init arms first (will fail loud if CAN/hardware is wrong).
    left: Optional[Robot] = None
    right: Optional[Robot] = None
    trace("about to init LEFT arm")
    left = init_arm(args.left_can, args.left_gripper)
    trace("about to init RIGHT arm")
    right = init_arm(args.right_can, args.right_gripper)
    trace("both arms initialized")

    # Cameras — each slot can be RealSense or V4L2 independently. Resolution
    # applies to all three; the MolmoAct2 image processor tiles adaptively so
    # any reasonable size works (training was at 256x342).
    cam_kw = dict(width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
    trace(f"building cameras at {args.cam_width}x{args.cam_height}/{args.cam_fps}fps")
    top   = make_camera("top",   args.top_cam_serial,   args.top_cam_v4l2,   **cam_kw)
    cam_l = make_camera("left",  args.left_cam_serial,  args.left_cam_v4l2,  **cam_kw)
    cam_r = make_camera("right", args.right_cam_serial, args.right_cam_v4l2, **cam_kw)
    for c in (top, cam_l, cam_r):
        trace(f"starting camera {c.name}")
        c.start()
        trace(f"camera {c.name} started")

    # Use Python's default SIGINT behavior (raises KeyboardInterrupt at the
    # next interpreter checkpoint) rather than a custom handler that sets a
    # flag. The custom-handler approach can leave non-daemon SDK threads alive
    # after main() returns and the process won't exit; KeyboardInterrupt
    # unwinds the stack faster and we force-exit at the bottom of finally.
    stop_flag = {"stop": False}  # kept for backward compat with intra-loop checks

    inner_dt = 1.0 / args.train_fps
    ideal_query_hz = args.train_fps / max(1, args.horizon_stride)
    log.info("Entering control loop: train_fps=%.1f Hz, stride=%d "
             "(ideal re-query ~%.1f Hz; actual is lower by ~server dt_ms), instruction=%r",
             args.train_fps, args.horizon_stride, ideal_query_hz, args.instruction)
    log.info("Per-tick caps: arm=%.3f rad, gripper=%.3f", args.max_step_rad, args.gripper_step)

    # Warmup the server with one /act call at the actual image shape so it
    # captures CUDA graphs once with a generous timeout, before the real
    # closed-loop control begins.
    try:
        state = read_state(left, right)
        log.info("Warming up server with a one-shot call at the real image shape "
                 "(timeout=%.0fs)...", args.warmup_timeout_s)
        _wu_actions, _wu_rtt = post_actions(
            args.server_url, top.grab(), cam_l.grab(), cam_r.grab(), state,
            args.instruction, args.num_steps, args.warmup_timeout_s,
        )
        log.info("Server warmup OK (rtt=%.0f ms, actions shape=%s)",
                 _wu_rtt, _wu_actions.shape)
    except Exception as e:
        log.error("Server warmup failed: %s. Continuing anyway.", e)

    try:
        while not stop_flag["stop"]:
            state = read_state(left, right)
            top_img = top.grab()
            left_img = cam_l.grab()
            right_img = cam_r.grab()

            actions, rtt_ms = post_actions(
                args.server_url, top_img, left_img, right_img, state,
                args.instruction, args.num_steps, args.timeout_s,
            )

            # Per-query diagnostic: what is the model actually asking for?
            # action[0] is what we'll command at this tick. The delta from
            # current state tells us how much motion the model wants. The
            # range across the horizon tells us whether the model expects
            # any motion within this batch at all.
            a0 = actions[0]
            delta0 = a0 - state
            horizon_range = (actions.max(axis=0) - actions.min(axis=0))
            arm_delta_max = float(np.max(np.abs(delta0[:6]) ) if delta0.size >= 6 else 0.0)
            arm_delta_max = max(arm_delta_max, float(np.max(np.abs(delta0[7:13]))))
            horizon_arm_span = float(max(np.max(horizon_range[:6]),
                                          np.max(horizon_range[7:13])))
            log.info(
                "/act rtt=%dms  |a0-state|_max(arm)=%.3f rad  horizon span(arm)=%.3f rad  "
                "L_grip=%.2f->%.2f  R_grip=%.2f->%.2f",
                rtt_ms, arm_delta_max, horizon_arm_span,
                state[6], a0[6], state[13], a0[13],
            )

            stride = max(1, args.horizon_stride)
            n_to_play = min(stride, actions.shape[0])
            for i in range(n_to_play):
                if stop_flag["stop"]:
                    break
                step_start = time.perf_counter()
                desired = actions[i].astype(np.float32)
                if args.dry_run:
                    log.info("dry-run action[%d]: %s", i,
                             np.array2string(desired, precision=3))
                else:
                    state = read_state(left, right)
                    safe_command(left, right, state, desired,
                                 args.max_step_rad, args.gripper_step)
                # Pace inner loop at the policy's training cadence.
                sleep_left = inner_dt - (time.perf_counter() - step_start)
                if sleep_left > 0:
                    time.sleep(sleep_left)
                elif sleep_left < -0.050:
                    # Only log severe overruns (>50ms = >1.5x the target tick).
                    # Small overruns are USB-contention noise, harmless at our
                    # arm command rates -- the motors lerp between sparser
                    # position targets just fine.
                    log.warning("inner step overrun by %.1f ms (target %.1f ms)",
                                -sleep_left * 1000.0, inner_dt * 1000.0)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt -- shutting down")
    finally:
        # During cleanup, a second Ctrl-C should hard-kill immediately rather
        # than escape from cv2.cap.release() or any other blocking call.
        def _hard_exit(_sig, _frame):
            os._exit(130)  # 128 + SIGINT
        try:
            signal.signal(signal.SIGINT, _hard_exit)
        except Exception:
            pass

        log.info("Stopping cameras")
        for c in (top, cam_l, cam_r):
            try:
                c.stop()
            except BaseException as e:
                log.warning("camera %s stop failed: %s", c.name, e)
        # Stop the i2rt SDK's per-arm background control threads. Without
        # this, the process won't exit even after main() returns -- the
        # control threads are non-daemon and Python waits on them forever.
        log.info("Closing arm SDKs")
        for arm in (left, right):
            if arm is None:
                continue
            try:
                arm.close()
            except BaseException as e:
                log.warning("arm.close() failed: %s", e)
        log.info("Arms left in their last commanded position -- kill power if not safe.")
        # Force-exit: any thread that didn't honor close() (e.g. waiting on a
        # CAN read in a C extension) won't keep the process alive.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
