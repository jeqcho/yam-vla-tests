"""Hardware glue for the bimanual YAM rig: cameras, arms, state, motion.

This module owns the i2rt SDK contact + RealSense / V4L2 camera streams.
Everything else (safety, journal, observability) is in sibling modules
so users can import just what they need.

Read order if you're new:
    install_sdk_lock_fix  -- one-time i2rt patch, called from init_arm
    load_setup_config     -- read yam_setup_config.json
    CameraStream + subclasses + make_camera   -- camera abstraction
    init_arm / read_state -- arm lifecycle + state read
    ramp_to_pose          -- linear interp for safe move-to-ready / return-on-exit
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from yam_vla.core.observation import ARM_DOF, STATE_DIM

log = logging.getLogger("yam_vla.hardware")


# ---------------------------------------------------------------------------
# Module-wide defaults that the control loop reads
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_FPS: float = 30.0      # inner-loop cadence (policy training freq)
DEFAULT_HORIZON_STRIDE: int = 6      # actions to play before re-querying
DEFAULT_CAM_WIDTH: int = 424         # USB-2-friendly D405 default
DEFAULT_CAM_HEIGHT: int = 240
DEFAULT_CAM_FPS: int = 30

DEFAULT_SETUP_CONFIG_PATH: str = str(
    Path(__file__).resolve().parents[3] / "yam_setup_config.json"
)


# ---------------------------------------------------------------------------
# i2rt SDK lock fix
# ---------------------------------------------------------------------------
# Original problem: dm_driver's control loop holds command_lock through the
# full 7-motor CAN round-trip (~3 ms). The other SDK thread also needs
# command_lock to push our target positions; Linux's mutex isn't fair under
# sustained contention -> 100s of ms starvation -> visible burst motion
# when the lock frees and a stale target gets pushed.
#
# Patched loop: hold command_lock only for a microsecond list-copy and do
# CAN I/O on the local copy. Validated with test_sdk_lock_fix.py — p99
# acquire drops from ~400 ms to <0.1 ms, set_commands throughput ~10x.

_SDK_LOCK_FIX_INSTALLED = False


def install_sdk_lock_fix() -> None:
    """Idempotently patch i2rt's dm_driver control loop. Safe to call multiple times."""
    global _SDK_LOCK_FIX_INSTALLED
    if _SDK_LOCK_FIX_INSTALLED:
        return

    import logging as _logging
    import time as _t
    from i2rt.motor_drivers import dm_driver as _dm  # heavy; deferred to call time
    EXPECTED = _dm.EXPECTED_CONTROL_PERIOD

    def _patched(self) -> None:
        last_step_time = _t.time()
        step_time_exceed_count = 0
        step_time_sum = 0.0
        step_time_count = 0
        report_start_time = _t.time()
        with self._rate_recorder:
            while self.running:
                try:
                    curr_time = _t.time()
                    step_time = curr_time - last_step_time
                    last_step_time = curr_time
                    step_time_sum += step_time
                    step_time_count += 1
                    if step_time > EXPECTED:
                        step_time_exceed_count += 1
                    if step_time_exceed_count > 0 and curr_time - report_start_time >= self._report_interval:
                        mean_step_time = step_time_sum / step_time_count if step_time_count > 0 else 0.0
                        _logging.info(
                            f"[PATCHED {self} {self._report_interval}s Report] "
                            f"step_time > {EXPECTED}s: {step_time_exceed_count} times, "
                            f"mean step_time: {mean_step_time:.6f} s"
                        )
                        step_time_exceed_count = 0
                        step_time_sum = 0.0
                        step_time_count = 0
                        report_start_time = curr_time

                    # THE FIX: brief snapshot, then CAN outside the lock.
                    with self.command_lock:
                        local_commands = list(self.commands)
                    try:
                        motor_feedback = self._set_commands(local_commands)
                    except RuntimeError as e:
                        if "Motor error detected" in str(e):
                            _logging.warning(f"Motor error in control loop, attempting recovery: {e}")
                            recovered = self._try_recover_motors()
                            if recovered:
                                _logging.warning("Motor recovery successful, continuing control loop")
                                continue
                            else:
                                self.running = False
                                raise
                        raise
                    errors = np.array([motor_feedback[i].error_code != "0x1"
                                       for i in range(len(motor_feedback))])
                    if np.any(errors):
                        _logging.warning(f"Motor errors detected in feedback: {errors}")
                        recovered = self._try_recover_motors(motor_feedback)
                        if recovered:
                            _logging.warning("Motor recovery successful, continuing control loop")
                            continue
                        self.running = False
                        _logging.error(f"motor errors: {errors}")
                        raise Exception(
                            "motors have unrecoverable errors after recovery attempts, stopping control loop"
                        )
                    with self.state_lock:
                        self.state = motor_feedback
                        self._update_absolute_positions(motor_feedback)
                    if self.same_bus_device_driver is not None:
                        _t.sleep(0.001)
                        with self.same_bus_device_lock:
                            self.same_bus_device_states = self.same_bus_device_driver.read_states()
                    _t.sleep(0)
                    self._rate_recorder.track()
                except Exception as e:
                    # close() races our control loop: it closes the CAN socket
                    # (fd -> -1) before the loop notices self.running flipped
                    # to False, so the in-flight send() raises. That's not an
                    # error worth shouting about. Quiet exit when shutdown is
                    # already underway; loud only for real mid-run failures.
                    if not self.running:
                        return
                    _logging.warning(f"DM Error in PATCHED control loop: {e}")
                    self.running = False
                    raise e

    _dm.DMChainCanInterface._set_torques_and_update_state = _patched
    _SDK_LOCK_FIX_INSTALLED = True
    log.info("i2rt SDK lock fix installed")


# ---------------------------------------------------------------------------
# Setup config (yam_setup_config.json)
# ---------------------------------------------------------------------------

def load_setup_config(path: Optional[str] = None) -> dict:
    """Read the per-machine hardware defaults from yam_setup_config.json.

    Returned dict is used as argparse defaults. Schema (all optional):
      left_can, right_can       CAN channels
      gripper                   gripper type, applied to both arms
      top_cam_serial            RealSense serial for top cam (wins over v4l2)
      top_cam_v4l2              V4L2 device path for a UVC top camera
      left_cam_serial           RealSense serial for left-wrist cam
      right_cam_serial          RealSense serial for right-wrist cam
    """
    if path is None:
        path = DEFAULT_SETUP_CONFIG_PATH
    try:
        with open(path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.warning("failed to read %s: %s -- using built-in defaults", path, e)
        return {}
    if cfg:
        log.info("Loaded setup config from %s", path)
    return cfg


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

class CameraStream:
    """Base camera interface: start, grab one HxWx3 uint8 RGB frame, stop."""

    def __init__(self, name: str, width: int, height: int, fps: int):
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps

    def start(self) -> None: raise NotImplementedError
    def grab(self) -> np.ndarray: raise NotImplementedError
    def stop(self) -> None: raise NotImplementedError


class RealSenseStream(CameraStream):
    """Intel RealSense color stream via librealsense."""

    def __init__(self, serial: str, name: str, width: int, height: int, fps: int):
        super().__init__(name, width, height, fps)
        self.serial = serial
        self.pipeline = None

    def start(self) -> None:
        import pyrealsense2 as rs
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.pipeline = rs.pipeline()
        self.pipeline.start(cfg)
        # Wait for warmup frames -- D435 on USB 2.0 can take 10+ s.
        budget_s = 20.0
        deadline = time.monotonic() + budget_s
        got = 0
        while got < 3 and time.monotonic() < deadline:
            try:
                self.pipeline.wait_for_frames(timeout_ms=2000)
                got += 1
            except Exception:
                pass
        if got == 0:
            raise RuntimeError(
                f"camera {self.name} (RealSense {self.serial}) produced no "
                f"frames within {budget_s:.0f}s -- check USB port + cable."
            )
        log.info("camera %s (RealSense %s) started @ %dx%d/%d Hz (warmup %d)",
                 self.name, self.serial, self.width, self.height, self.fps, got)

    def grab(self) -> np.ndarray:
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        except RuntimeError:
            log.warning("camera %s (%s): grab timeout, retrying once", self.name, self.serial)
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"camera {self.name} produced no color frame")
        return np.asanyarray(color.get_data())

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError as e:
                log.warning("camera %s stop noop: %s", self.name, e)


class V4L2Stream(CameraStream):
    """Generic UVC/V4L2 webcam via OpenCV."""

    def __init__(self, device: str, name: str, width: int, height: int, fps: int):
        super().__init__(name, width, height, fps)
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
        for _ in range(5):
            self.cap.read()  # AE settle
        log.info("camera %s (V4L2 %s) started @ %dx%d/%d Hz",
                 self.name, self.device, self.width, self.height, self.fps)

    def grab(self) -> np.ndarray:
        import cv2
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"camera {self.name} produced no frame")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()


def make_camera(
    name: str,
    serial: Optional[str] = None,
    v4l2_device: Optional[str] = None,
    *,
    width: int = DEFAULT_CAM_WIDTH,
    height: int = DEFAULT_CAM_HEIGHT,
    fps: int = DEFAULT_CAM_FPS,
) -> CameraStream:
    """Build the right CameraStream subclass.

    If both serial and v4l2_device are provided, serial wins -- a leftover
    v4l2_device in setup config from a previous hardware gen doesn't block
    a CLI --<slot>-cam-serial override.
    """
    if not serial and not v4l2_device:
        raise ValueError(f"{name}: must pass serial or v4l2_device")
    if serial:
        if v4l2_device:
            log.warning("%s: both serial and v4l2 set; using serial", name)
        return RealSenseStream(serial, name, width=width, height=height, fps=fps)
    return V4L2Stream(v4l2_device, name, width=width, height=height, fps=fps)


# ---------------------------------------------------------------------------
# Arms (i2rt YAM follower)
# ---------------------------------------------------------------------------

def init_arm(can_channel: str, gripper: str, ee_mass: Optional[float] = None):
    """Create a YAM follower robot in position-holding mode.

    Position-holding (kp != 0) rather than zero_gravity_mode: gravity comp can
    be slightly mis-tuned and the arm drifts; position-hold actively keeps
    whatever pose it has at script start.
    """
    install_sdk_lock_fix()
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType

    arm_type = ArmType.from_string_name("yam")
    gripper_type = GripperType.from_string_name(gripper)
    log.info("init_arm(%s, %s): get_yam_robot (gripper auto-cal ~3-5s)",
             can_channel, gripper)
    robot = get_yam_robot(
        channel=can_channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
        zero_gravity_mode=False,
        ee_mass=ee_mass,
    )
    q0 = np.asarray(robot.get_joint_pos(), dtype=np.float32)
    log.info("init_arm(%s): pose=%s, commanding hold",
             can_channel, np.array2string(q0, precision=3))
    robot.command_joint_pos(q0)
    return robot


def read_state(left, right) -> np.ndarray:
    """Compose the 14-D YAM state: [left_q(6), left_grip, right_q(6), right_grip]."""
    s_l = np.asarray(left.get_joint_pos(), dtype=np.float32).reshape(-1)
    s_r = np.asarray(right.get_joint_pos(), dtype=np.float32).reshape(-1)
    if s_l.shape != (ARM_DOF + 1,) or s_r.shape != (ARM_DOF + 1,):
        raise RuntimeError(
            f"expected ({ARM_DOF + 1},) per arm, got left={s_l.shape}, right={s_r.shape}"
        )
    return np.concatenate([s_l, s_r], axis=0).astype(np.float32)


def ramp_to_pose(
    left, right, target_14d: np.ndarray,
    *,
    duration_s: float = 5.0,
    hz: float = 30.0,
    abort_flag: Optional[dict] = None,
    label: str = "ramp",
) -> None:
    """Linearly interpolate both arms from their current pose to `target_14d`.

    `abort_flag['abort'] = True` causes the loop to stop at the next step.
    On abort, arms are left at the last commanded interpolation point (which
    they will hold as long as the SDK control threads are alive).
    """
    q_l = np.asarray(left.get_joint_pos(),  dtype=np.float32)
    q_r = np.asarray(right.get_joint_pos(), dtype=np.float32)
    start = np.concatenate([q_l, q_r])
    goal = np.asarray(target_14d, dtype=np.float32).copy()
    if goal.shape != (STATE_DIM,):
        raise ValueError(f"target_14d must be ({STATE_DIM},), got {goal.shape}")
    delta = goal - start
    max_d = float(np.max(np.abs(delta)))
    log.info("[%s] max per-joint delta = %.3f rad (%.1f deg), %.1fs ramp",
             label, max_d, np.degrees(max_d), duration_s)
    n_steps = max(1, int(duration_s * hz))
    dt = 1.0 / hz
    for i in range(1, n_steps + 1):
        if abort_flag is not None and abort_flag.get("abort"):
            log.warning("[%s] aborted at step %d/%d", label, i, n_steps)
            return
        alpha = i / n_steps
        cmd = start + alpha * delta
        left.command_joint_pos(cmd[:ARM_DOF + 1].astype(np.float32))
        right.command_joint_pos(cmd[ARM_DOF + 1:].astype(np.float32))
        time.sleep(dt)
    time.sleep(0.5)  # PD settle


__all__ = [
    # constants
    "DEFAULT_TRAIN_FPS", "DEFAULT_HORIZON_STRIDE",
    "DEFAULT_CAM_WIDTH", "DEFAULT_CAM_HEIGHT", "DEFAULT_CAM_FPS",
    "DEFAULT_SETUP_CONFIG_PATH",
    # SDK + config
    "install_sdk_lock_fix", "load_setup_config",
    # cameras
    "CameraStream", "RealSenseStream", "V4L2Stream", "make_camera",
    # arms
    "init_arm", "read_state", "ramp_to_pose",
]
