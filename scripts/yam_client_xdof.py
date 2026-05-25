"""ZERO-SHOT bimanual YAM client for GR00T N1.7-3B (base model).

GR00T N1.7-3B was pretrained on AllenAI bimanual YAM data under the
`xdof_relative_eef_relative_joint` embodiment tag — verified by reading
the base checkpoint's `experiment_cfg/conf.yaml`:

    datasets:
      - /mnt/aws-lfs-02/.../xdof.yam_v7_all_merged_global_task_exclude_bad_subtasks
        embodiment_tag: xdof_relative_eef_relative_joint
      - /mnt/aws-lfs-02/.../xdof.yam_v7_subtask_only_merged_global_task
        embodiment_tag: xdof_relative_eef_relative_joint_subtask

So we can run the base model directly on YAM. The catch: the XDOF embodiment
schema is more elaborate than the simple 14-D state I built for the finetune
path. This client implements the full XDOF wire format.

Schema (from base model processor_config.json):
    video keys: top_camera-images-rgb_320_240,
                left_camera-images-rgb_320_240,
                right_camera-images-rgb_320_240
        delta_indices: [-30, 0]   -> two frames per call, 1 sec apart
        resolution:    320 x 240 (must match exactly)
    state keys:
        left_wrist_eef        (9,)    XYZ + 6D rotation, computed via FK
        right_wrist_eef       (9,)    XYZ + 6D rotation, computed via FK
        left_gripper_pos      (1,)    normalized [0, 1]
        right_gripper_pos     (1,)    normalized [0, 1]
        left_joint_pos        (6,)    raw joint radians
        right_joint_pos       (6,)    raw joint radians
        delta_indices: [0]   -> current only
    action keys: same 6 names, 40-step horizon
        action_configs:
            left/right_wrist_eef:    RELATIVE EEF, format=XYZ_ROT6D
            left/right_gripper_pos:  ABSOLUTE NON_EEF
            left/right_joint_pos:    RELATIVE NON_EEF
        Gr00tPolicy.decode_action does the relative->absolute conversion
        on the server side, so what we receive is already absolute.
    language key: annotation.task

We send all 6 state keys + 3 cameras + language. We DECODE the action chunk
using ONLY left_joint_pos / right_joint_pos + gripper_pos — the EEF action
outputs are ignored (joint-space is simpler to apply; no need for IK).

Run with the i2rt venv (has pyrealsense2 + i2rt SDK + pyzmq + msgpack-numpy):

    /home/andon/yam-tests/i2rt/.venv/bin/python scripts/yam_client_xdof.py \
        --left-can can0 --right-can can1 \
        --left-gripper linear_4310 --right-gripper linear_4310 \
        --top-cam-serial AAAA --left-cam-serial BBBB --right-cam-serial CCCC \
        --server-host 127.0.0.1 --server-port 5557 \
        --instruction "Move the blocks to spell AI2" \
        --dry-run

Safety: identical to scripts/yam_client.py — per-tick joint delta capped at
--max-step-rad, per-tick gripper cap at --gripper-step, return-to-startup
ramp on exit. Cameras initialized before arms to dodge the USB enumeration
storm vs CAN watchdog race.
"""
from __future__ import annotations

import argparse
import collections
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np

# i2rt imports
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from i2rt.robots.kinematics import Kinematics


# =============================================================================
# SDK lock fix — copied verbatim from yam_client.py / molmoact2's client.
# =============================================================================

def install_sdk_lock_fix() -> None:
    import logging as _logging
    import time as _t
    from i2rt.motor_drivers import dm_driver as _dm
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
                    if (
                        step_time_exceed_count > 0
                        and curr_time - report_start_time >= self._report_interval
                    ):
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
                    with self.command_lock:
                        local_commands = list(self.commands)
                    try:
                        motor_feedback = self._set_commands(local_commands)
                    except RuntimeError as e:
                        if "Motor error detected" in str(e):
                            recovered = self._try_recover_motors()
                            if recovered:
                                continue
                            self.running = False
                            raise
                        raise
                    errors = np.array([motor_feedback[i].error_code != "0x1"
                                       for i in range(len(motor_feedback))])
                    if np.any(errors):
                        recovered = self._try_recover_motors(motor_feedback)
                        if recovered:
                            continue
                        self.running = False
                        raise Exception("motors have unrecoverable errors")
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
                    print(f"DM Error in PATCHED control loop: {e}")
                    self.running = False
                    raise e

    _dm.DMChainCanInterface._set_torques_and_update_state = _patched


install_sdk_lock_fix()

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s | %(message)s"))
if not any(getattr(h, "_xdof_client_handler", False) for h in _root.handlers):
    _handler._xdof_client_handler = True
    _root.addHandler(_handler)
log = logging.getLogger("yam.xdof.client")


def trace(msg: str) -> None:
    print(f"[TRACE] {msg}", flush=True)


# =============================================================================
# Slim PolicyClient — zmq + msgpack-numpy, compatible with gr00t PolicyServer.
# =============================================================================

import msgpack_numpy as _mnp
import zmq as _zmq


class GrootPolicyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5557, timeout_ms: int = 60000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._ctx = _zmq.Context()
        self._init_socket()

    def _init_socket(self) -> None:
        self.socket = self._ctx.socket(_zmq.REQ)
        self.socket.setsockopt(_zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(_zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _call(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        req: dict = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        try:
            self.socket.send(_mnp.packb(req))
            msg = self.socket.recv()
        except _zmq.error.Again:
            self._init_socket()
            raise
        resp = _mnp.unpackb(msg, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"server error: {resp['error']}")
        return resp

    def ping(self) -> dict:
        return self._call("ping", requires_input=False)

    def get_modality_config(self) -> dict:
        return self._call("get_modality_config", requires_input=False)

    def get_action(self, observation: dict, options: dict | None = None):
        resp = self._call("get_action", {"observation": observation, "options": options})
        action, info = resp
        return action, info


# =============================================================================
# Hardware glue — cameras + arms + safety command.
# =============================================================================

DEFAULT_MAX_STEP_RAD = 0.15
DEFAULT_GRIPPER_STEP = 0.15
DEFAULT_TRAIN_FPS = 30.0
DEFAULT_HORIZON_STRIDE = 8     # play 8 of 40 steps before re-query (~3.75 Hz at 30 fps)
STATE_DIM = 14
ARM_DOFS = 7                   # 6 arm joints + 1 gripper

# XDOF expects 320x240 input. Cameras can stream higher and we resize.
TRAIN_IMG_W = 320
TRAIN_IMG_H = 240
HISTORY_FRAMES_AGO = 30        # delta_indices = [-30, 0] -> need a frame from 30 ticks ago

# Subdir of the YAM joint vector that's the arm vs gripper. read_state() returns
# (7,) per arm: [q0..q5, gripper_normalized].


class CameraStream:
    def __init__(self, name: str, width: int = 320, height: int = 240, fps: int = 30):
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps

    def start(self) -> None: raise NotImplementedError
    def grab(self) -> np.ndarray: raise NotImplementedError
    def stop(self) -> None: raise NotImplementedError


class RealSenseStream(CameraStream):
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
        for _ in range(5):
            try:
                self.pipeline.wait_for_frames(timeout_ms=2000)
            except Exception:
                pass
        log.info("camera %s (RealSense %s) @ %dx%d/%d Hz", self.name, self.serial,
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
        for _ in range(5):
            self.cap.read()
        log.info("camera %s (V4L2 %s) @ %dx%d/%d Hz", self.name, self.device,
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
    if serial and v4l2_device:
        raise ValueError(f"{name}: pass exactly one of --{name}-cam-serial / --{name}-cam-v4l2")
    if not serial and not v4l2_device:
        raise ValueError(f"{name}: must pass --{name}-cam-serial or --{name}-cam-v4l2")
    if serial:
        return RealSenseStream(serial, name, width=width, height=height, fps=fps)
    return V4L2Stream(v4l2_device, name, width=width, height=height, fps=fps)


def resize_for_model(img: np.ndarray) -> np.ndarray:
    """Cheap centered resize to the model's expected (240, 320, 3) input.

    Cameras may be running at a higher resolution; we downsize here so the model
    sees the same shape it was trained on. Uses cv2 INTER_AREA which is fine for
    downscale.
    """
    import cv2
    if img.shape[0] == TRAIN_IMG_H and img.shape[1] == TRAIN_IMG_W:
        return img
    return cv2.resize(img, (TRAIN_IMG_W, TRAIN_IMG_H), interpolation=cv2.INTER_AREA)


def init_arm(can_channel: str, gripper: str, ee_mass: Optional[float] = None) -> Robot:
    trace(f"init_arm({can_channel}, {gripper})")
    arm_type = ArmType.from_string_name("yam")
    gripper_type = GripperType.from_string_name(gripper)
    robot = get_yam_robot(
        channel=can_channel, arm_type=arm_type, gripper_type=gripper_type,
        zero_gravity_mode=False, ee_mass=ee_mass,
    )
    q0 = np.asarray(robot.get_joint_pos(), dtype=np.float32)
    robot.command_joint_pos(q0)
    trace(f"init_arm({can_channel}): joint_pos={np.array2string(q0, precision=3)}")
    return robot


def read_state(left: Robot, right: Robot) -> np.ndarray:
    """14-D: [left_q0..5, left_grip, right_q0..5, right_grip]."""
    s_l = np.asarray(left.get_joint_pos(), dtype=np.float32).reshape(-1)
    s_r = np.asarray(right.get_joint_pos(), dtype=np.float32).reshape(-1)
    return np.concatenate([s_l, s_r], axis=0).astype(np.float32)


def safe_command(left: Robot, right: Robot,
                 current_state: np.ndarray, desired_action: np.ndarray,
                 max_step_rad: float, gripper_step: float) -> tuple[np.ndarray, int]:
    if desired_action.shape != (STATE_DIM,):
        raise ValueError(f"action shape {desired_action.shape} != ({STATE_DIM},)")
    delta = desired_action - current_state
    caps = np.full(STATE_DIM, max_step_rad if max_step_rad > 0 else np.inf, dtype=np.float32)
    caps[6]  = gripper_step if gripper_step > 0 else np.inf
    caps[13] = gripper_step if gripper_step > 0 else np.inf
    clipped_delta = np.clip(delta, -caps, caps)
    n_clipped = int(np.sum(clipped_delta != delta))
    cmd = (current_state + clipped_delta).astype(np.float32)
    left.command_joint_pos(cmd[:ARM_DOFS])
    right.command_joint_pos(cmd[ARM_DOFS:])
    return cmd, n_clipped


def ramp_to_pose(left: Robot, right: Robot, target_14d: np.ndarray,
                 duration_s: float = 5.0, hz: float = 30.0,
                 abort_flag: dict | None = None, label: str = "ramp") -> None:
    q_l = np.asarray(left.get_joint_pos(), dtype=np.float32)
    q_r = np.asarray(right.get_joint_pos(), dtype=np.float32)
    start = np.concatenate([q_l, q_r])
    goal = np.asarray(target_14d, dtype=np.float32).copy()
    delta = goal - start
    max_d = float(np.max(np.abs(delta)))
    log.info("[%s] start=%s", label, np.array2string(start, precision=3))
    log.info("[%s] goal =%s  max_delta=%.3f rad", label,
             np.array2string(goal, precision=3), max_d)
    n_steps = max(1, int(duration_s * hz))
    dt = 1.0 / hz
    for i in range(1, n_steps + 1):
        if abort_flag is not None and abort_flag.get("abort"):
            log.warning("[%s] aborted at step %d/%d", label, i, n_steps)
            return
        alpha = i / n_steps
        cmd = start + alpha * delta
        left.command_joint_pos(cmd[:7].astype(np.float32))
        right.command_joint_pos(cmd[7:].astype(np.float32))
        time.sleep(dt)
    time.sleep(0.5)
    log.info("[%s] done", label)


# =============================================================================
# XDOF wire-format helpers.
# =============================================================================

class YamFK:
    """Wraps i2rt's mink-based FK for one arm. Returns (xyz, rot6d) per call."""

    def __init__(self, gripper: str):
        xml_path = combine_arm_and_gripper_xml(
            ArmType.YAM, GripperType.from_string_name(gripper)
        )
        self.kin = Kinematics(xml_path, "grasp_site")
        # MuJoCo model has 6 arm joints + 2 gripper joints = 8 qpos slots.
        # We only know arm joints + a normalized gripper; pad gripper joints
        # with zeros since FK at the grasp_site is gripper-invariant to within
        # mm (gripper articulation moves the fingers, not the wrist).
        self._n_qpos = 8
        # Sanity-check qpos size by trying one FK call.
        try:
            self.kin.fk(np.zeros(self._n_qpos))
        except ValueError as e:
            # Fall back to 6 (no-gripper model).
            self._n_qpos = 6
            self.kin.fk(np.zeros(self._n_qpos))

    def xyz_rot6d(self, q6: np.ndarray) -> np.ndarray:
        """Compute (9,) xyz+rot6d for one arm given (6,) joint angles."""
        if self._n_qpos == 8:
            q = np.concatenate([q6, np.zeros(2)])
        else:
            q = q6
        T = self.kin.fk(q)
        xyz = T[0:3, 3]
        rot6d = T[0:2, 0:3].flatten()
        return np.concatenate([xyz, rot6d]).astype(np.float32)


class FrameHistoryBuffer:
    """Rolling per-camera buffer that exposes the frame from `n_steps_ago` ticks.

    We push every tick and ask for the frame N pushes back. If the buffer doesn't
    have N entries yet, returns the oldest frame we have.
    """

    def __init__(self, n_steps_ago: int):
        self.n = n_steps_ago
        # Need at least n_steps_ago + 1 slots; round up a bit.
        self.buf: collections.deque = collections.deque(maxlen=n_steps_ago + 1)

    def push(self, frame: np.ndarray) -> None:
        self.buf.append(frame)

    def past(self) -> np.ndarray:
        # If we have fewer than n+1 frames, return the oldest we have (which on
        # the first tick is the same as current — effectively no history yet).
        if len(self.buf) <= 1:
            return self.buf[0]
        # buf[0] is the oldest; buf[-1] is the newest.
        return self.buf[0]


def build_observation(
    top_now: np.ndarray, top_past: np.ndarray,
    left_now: np.ndarray, left_past: np.ndarray,
    right_now: np.ndarray, right_past: np.ndarray,
    state_14d: np.ndarray,
    left_eef9: np.ndarray, right_eef9: np.ndarray,
    instruction: str,
) -> dict:
    """Pack XDOF-format observation with (B=1, T=2) video and (B=1, T=1) state."""

    def _video(past: np.ndarray, now: np.ndarray) -> np.ndarray:
        # Output shape (B=1, T=2, H, W, 3)
        return np.stack([past, now], axis=0)[np.newaxis, ...].astype(np.uint8)

    def _state(v: np.ndarray) -> np.ndarray:
        return v[np.newaxis, np.newaxis, ...].astype(np.float32)

    obs = {
        "video": {
            "top_camera-images-rgb_320_240":   _video(top_past, top_now),
            "left_camera-images-rgb_320_240":  _video(left_past, left_now),
            "right_camera-images-rgb_320_240": _video(right_past, right_now),
        },
        "state": {
            "left_wrist_eef":    _state(left_eef9),
            "right_wrist_eef":   _state(right_eef9),
            "left_gripper_pos":  _state(state_14d[6:7]),
            "right_gripper_pos": _state(state_14d[13:14]),
            "left_joint_pos":    _state(state_14d[0:6]),
            "right_joint_pos":   _state(state_14d[7:13]),
        },
        "language": {
            "annotation.task": [[instruction]],
        },
    }
    return obs


def decode_actions_to_yam14(action_chunk: dict, horizon: int) -> np.ndarray:
    """Convert XDOF action dict into a (horizon, 14) YAM-layout array.

    Layout: [left_q0..5, left_grip, right_q0..5, right_grip].

    The server has ALREADY converted RELATIVE -> ABSOLUTE for joint and
    EEF actions (see gr00t/data/state_action/state_action_processor.py:455).
    Gripper actions are ABSOLUTE in the XDOF config. So all four streams we
    consume are absolute target positions.

    We deliberately ignore action.left_wrist_eef and action.right_wrist_eef
    (relative EEF targets would need IK to apply on YAM hardware). Joint-space
    is simpler and the model produces both.
    """
    lj = np.asarray(action_chunk["left_joint_pos"], dtype=np.float32)[0]    # (T, 6)
    rj = np.asarray(action_chunk["right_joint_pos"], dtype=np.float32)[0]   # (T, 6)
    lg = np.asarray(action_chunk["left_gripper_pos"], dtype=np.float32)[0]  # (T, 1)
    rg = np.asarray(action_chunk["right_gripper_pos"], dtype=np.float32)[0] # (T, 1)
    if not (lj.shape[0] == rj.shape[0] == lg.shape[0] == rg.shape[0]):
        raise RuntimeError(
            f"action horizon mismatch: lj={lj.shape}, rj={rj.shape}, "
            f"lg={lg.shape}, rg={rg.shape}"
        )
    T = lj.shape[0]
    if horizon != -1 and T != horizon:
        log.warning("server returned action horizon %d, expected %d", T, horizon)
    actions = np.concatenate([lj, lg, rj, rg], axis=-1)  # (T, 14)
    if actions.shape[-1] != STATE_DIM:
        raise RuntimeError(f"decoded action dim {actions.shape[-1]} != {STATE_DIM}")
    return actions


# =============================================================================
# Journal — same shape as scripts/yam_client.py.
# =============================================================================

DEFAULT_JOURNAL_PATH = "/home/andon/yam-tests/grootn1.7 exploration/journal.md"


def _journal_format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _journal_invocation() -> str:
    inv = os.environ.get("YAM_INVOCATION")
    if inv:
        return inv
    return " ".join(sys.argv)


def _journal_format_args(args) -> str:
    lines = []
    for k, v in sorted(vars(args).items()):
        if v is None or v is False:
            continue
        sv = repr(v) if isinstance(v, str) and len(v) > 120 else str(v)
        lines.append(f"- `{k}`: {sv}")
    return "\n".join(lines) if lines else "_(none)_"


def prompt_journal_entry(start_time_s: float, args) -> Optional[dict]:
    if not sys.stdin.isatty():
        return None
    if getattr(args, "no_journal", False):
        return None
    duration_s = time.time() - start_time_s
    print("\n" + "=" * 70, flush=True)
    print("Research journal -- record this run?", flush=True)
    print("=" * 70, flush=True)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Duration:    {_journal_format_duration(duration_s)}", flush=True)
    print("\nHow did the run go?  [s] success  [f] failure  [u] unclear  [enter] skip", flush=True)
    try:
        choice = input("> ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return None
    if not choice:
        return None
    status = {"s": "success", "f": "failure", "u": "unclear"}.get(choice[:1])
    if status is None:
        return None
    try:
        notes = input("\nWhat happened? (one line, optional)\n> ").strip()
        purpose = input("\nPurpose of this run? (optional)\n> ").strip()
    except (KeyboardInterrupt, EOFError):
        notes = locals().get("notes", "")
        purpose = ""
    return {
        "status": status, "notes": notes, "purpose": purpose,
        "duration_s": duration_s,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_journal_entry(path: str, entry: dict, args, invocation: str) -> None:
    md = ["", "---", f"## {entry['timestamp']} -- {entry['status']} (zero-shot XDOF)", ""]
    if entry.get("purpose"):
        md.extend([f"**Purpose**: {entry['purpose']}", ""])
    if entry.get("notes"):
        md.extend([f"**Notes**: {entry['notes']}", ""])
    md.extend([f"**Duration**: {_journal_format_duration(entry['duration_s'])}", ""])
    md.extend(["**Command**:", "```", invocation, "```", ""])
    md.extend(["**Configuration**:", _journal_format_args(args), ""])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[journal] wrote {entry['status']} entry to {path}", flush=True)


# =============================================================================
# Main loop.
# =============================================================================

def main() -> None:
    journal_start_s = time.time()
    journal_invocation = _journal_invocation()

    p = argparse.ArgumentParser(description="GR00T N1.7 ZERO-SHOT bimanual-YAM client (XDOF tag)")
    p.add_argument("--left-can", default="can0")
    p.add_argument("--right-can", default="can1")
    p.add_argument("--left-gripper", default="linear_4310",
                   choices=["crank_4310", "linear_3507", "linear_4310", "flexible_4310"])
    p.add_argument("--right-gripper", default="linear_4310",
                   choices=["crank_4310", "linear_3507", "linear_4310", "flexible_4310"])
    p.add_argument("--top-cam-serial",   default=None)
    p.add_argument("--top-cam-v4l2",     default=None)
    p.add_argument("--left-cam-serial",  default=None)
    p.add_argument("--left-cam-v4l2",    default=None)
    p.add_argument("--right-cam-serial", default=None)
    p.add_argument("--right-cam-v4l2",   default=None)
    p.add_argument("--cam-width",  type=int, default=320,
                   help="Camera capture width (resized to 320 for the model)")
    p.add_argument("--cam-height", type=int, default=240,
                   help="Camera capture height (resized to 240 for the model)")
    p.add_argument("--cam-fps",    type=int, default=30)
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=5557,
                   help="ZeroMQ port of the XDOF GR00T server (run_server_xdof.sh)")
    p.add_argument("--server-timeout-ms", type=int, default=60000)
    p.add_argument("--instruction", required=True,
                   help="Natural-language task. Goes into annotation.task.")
    p.add_argument("--train-fps", type=float, default=DEFAULT_TRAIN_FPS,
                   help="Inner-loop pacing — controls how often we tick a command "
                        "and how the frame-history buffer indexes.")
    p.add_argument("--max-step-rad", type=float, default=DEFAULT_MAX_STEP_RAD)
    p.add_argument("--gripper-step", type=float, default=DEFAULT_GRIPPER_STEP)
    p.add_argument("--horizon-stride", type=int, default=DEFAULT_HORIZON_STRIDE,
                   help="Play this many of the 40 returned action steps before re-query. "
                        "stride=8 with train_fps=30 -> server is queried ~3.75 Hz.")
    p.add_argument("--action-horizon", type=int, default=40)
    p.add_argument("--dump-frames", default=None,
                   help="Save first {top,left,right} frames as PNGs and exit.")
    p.add_argument("--no-return-on-exit", action="store_true",
                   help="DANGEROUS: skip return-to-startup ramp.")
    p.add_argument("--ramp-duration-s", type=float, default=5.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Don't command the arms; print actions only.")
    p.add_argument("--no-journal", action="store_true")
    p.add_argument("--journal-path", default=DEFAULT_JOURNAL_PATH)
    args = p.parse_args()

    if args.max_step_rad <= 0 and args.gripper_step <= 0:
        log.warning("=" * 70)
        log.warning("ALL CLIPPING DISABLED — arms will track raw policy output.")
        log.warning("=" * 70)

    # 1. Server.
    client = GrootPolicyClient(host=args.server_host, port=args.server_port,
                               timeout_ms=args.server_timeout_ms)
    try:
        pong = client.ping()
        log.info("server ping: %s", pong)
    except Exception as e:
        log.error("server ping failed at %s:%d -- %s", args.server_host, args.server_port, e)
        sys.exit(2)

    # Confirm the server's modality matches what we expect for XDOF.
    try:
        mod = client.get_modality_config()
        expected = {"top_camera-images-rgb_320_240",
                    "left_camera-images-rgb_320_240",
                    "right_camera-images-rgb_320_240"}
        got_video = set(mod.get("video").modality_keys) if hasattr(mod.get("video"), "modality_keys") else set()
        if expected != got_video:
            log.warning("server video keys mismatch! expected %s, got %s",
                        sorted(expected), sorted(got_video))
        else:
            log.info("server video keys match XDOF schema")
    except Exception as e:
        log.warning("could not fetch server modality config: %s", e)

    # 2. FK objects (one per arm — gripper variants can differ).
    fk_left = YamFK(args.left_gripper)
    fk_right = YamFK(args.right_gripper)

    # 3. Cameras BEFORE arms.
    left: Optional[Robot] = None
    right: Optional[Robot] = None
    top = cam_l = cam_r = None
    try:
        cam_kw = dict(width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
        top   = make_camera("top",   args.top_cam_serial,   args.top_cam_v4l2,   **cam_kw)
        cam_l = make_camera("left",  args.left_cam_serial,  args.left_cam_v4l2,  **cam_kw)
        cam_r = make_camera("right", args.right_cam_serial, args.right_cam_v4l2, **cam_kw)
        for c in (top, cam_l, cam_r):
            c.start()
        for _ in range(3):
            top.grab(); cam_l.grab(); cam_r.grab()
    except Exception:
        for c in (top, cam_l, cam_r):
            if c is not None:
                try: c.stop()
                except Exception: pass
        raise

    # 4. Arms.
    left = init_arm(args.left_can, args.left_gripper)
    right = init_arm(args.right_can, args.right_gripper)
    startup_pose = np.concatenate([
        np.asarray(left.get_joint_pos(),  dtype=np.float32),
        np.asarray(right.get_joint_pos(), dtype=np.float32),
    ])
    log.info("startup pose: %s", np.array2string(startup_pose, precision=3))

    # 5. Frame history buffers (one per camera) — capacity is fixed at
    # HISTORY_FRAMES_AGO + 1 so we can always serve "frame from 30 ticks ago".
    hist_top   = FrameHistoryBuffer(HISTORY_FRAMES_AGO)
    hist_left  = FrameHistoryBuffer(HISTORY_FRAMES_AGO)
    hist_right = FrameHistoryBuffer(HISTORY_FRAMES_AGO)

    stop_flag = {"stop": False}
    inner_dt = 1.0 / args.train_fps
    ideal_query_hz = args.train_fps / max(1, args.horizon_stride)
    log.info("control loop: train_fps=%.1f Hz, stride=%d (re-query ~%.1f Hz)",
             args.train_fps, args.horizon_stride, ideal_query_hz)
    log.info("per-tick caps: arm=%.3f rad, gripper=%.3f", args.max_step_rad, args.gripper_step)
    log.info("instruction: %r", args.instruction)

    # 6. Warmup — populate frame buffer and do one /get_action.
    log.info("Warming up server (this may take 5-30s as it captures CUDA graphs)...")
    try:
        state = read_state(left, right)
        for _ in range(2):  # two ticks so past != current after first call
            top_img = resize_for_model(top.grab())
            left_img = resize_for_model(cam_l.grab())
            right_img = resize_for_model(cam_r.grab())
            hist_top.push(top_img); hist_left.push(left_img); hist_right.push(right_img)
        eef_l = fk_left.xyz_rot6d(state[0:6])
        eef_r = fk_right.xyz_rot6d(state[7:13])
        obs = build_observation(
            hist_top.buf[-1], hist_top.past(),
            hist_left.buf[-1], hist_left.past(),
            hist_right.buf[-1], hist_right.past(),
            state, eef_l, eef_r, args.instruction,
        )
        t0 = time.perf_counter()
        action_chunk, info = client.get_action(obs)
        warm_rtt = (time.perf_counter() - t0) * 1000.0
        sample = decode_actions_to_yam14(action_chunk, args.action_horizon)
        log.info("warmup OK: rtt=%.0f ms, action=(horizon=%d, 14)", warm_rtt, sample.shape[0])
    except Exception as e:
        log.error("warmup FAILED: %s. continuing anyway.", e)

    loop_t0 = time.perf_counter()

    try:
        while not stop_flag["stop"]:
            state = read_state(left, right)

            # Update frame history every tick.
            top_now   = resize_for_model(top.grab())
            left_now  = resize_for_model(cam_l.grab())
            right_now = resize_for_model(cam_r.grab())
            hist_top.push(top_now)
            hist_left.push(left_now)
            hist_right.push(right_now)

            if args.dump_frames:
                import cv2
                os.makedirs(args.dump_frames, exist_ok=True)
                for name, img in [("top", top_now), ("left", left_now), ("right", right_now)]:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out_path = os.path.join(args.dump_frames, f"{name}.png")
                    cv2.imwrite(out_path, bgr)
                    log.info("dumped %s (%dx%d) to %s", name, img.shape[1], img.shape[0], out_path)
                log.info("dump-frames mode -- exiting")
                sys.stdout.flush()
                os._exit(0)

            eef_l = fk_left.xyz_rot6d(state[0:6])
            eef_r = fk_right.xyz_rot6d(state[7:13])
            obs = build_observation(
                top_now,  hist_top.past(),
                left_now, hist_left.past(),
                right_now, hist_right.past(),
                state, eef_l, eef_r, args.instruction,
            )

            t0 = time.perf_counter()
            action_chunk, info = client.get_action(obs)
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            actions = decode_actions_to_yam14(action_chunk, args.action_horizon)

            # Diagnostic: how far does each step ask the arms to move from state?
            def _arm_delta_max(a_idx: int) -> float:
                d = actions[a_idx] - state
                return float(max(np.max(np.abs(d[:6])), np.max(np.abs(d[7:13]))))
            h = actions.shape[0]
            a0_d  = _arm_delta_max(0)
            a_mid = _arm_delta_max(h // 2)
            a_end = _arm_delta_max(h - 1)
            horizon_arm_span = float(np.max(
                actions.max(axis=0)[np.r_[:6, 7:13]] - actions.min(axis=0)[np.r_[:6, 7:13]]
            ))
            log.info(
                "/get_action rtt=%dms  |a[i]-state|_max @ 0/mid/end: %.3f/%.3f/%.3f rad  "
                "span=%.3f  L_grip[0,-1]=%.2f,%.2f  R_grip[0,-1]=%.2f,%.2f",
                rtt_ms, a0_d, a_mid, a_end, horizon_arm_span,
                actions[0][6], actions[-1][6], actions[0][13], actions[-1][13],
            )

            stride = max(1, args.horizon_stride)
            n_to_play = min(stride, actions.shape[0])
            clipped = steps = 0
            for i in range(n_to_play):
                if stop_flag["stop"]: break
                step_start = time.perf_counter()
                desired = actions[i].astype(np.float32)
                if args.dry_run:
                    log.info("dry-run action[%d]: %s", i, np.array2string(desired, precision=3))
                else:
                    state = read_state(left, right)
                    _, n_c = safe_command(left, right, state, desired,
                                          args.max_step_rad, args.gripper_step)
                    clipped += n_c; steps += 1
                sleep_left = inner_dt - (time.perf_counter() - step_start)
                if sleep_left > 0:
                    time.sleep(sleep_left)
                elif sleep_left < -0.050:
                    log.warning("inner step overrun by %.1f ms",
                                -sleep_left * 1000.0)
            if steps > 0 and (args.max_step_rad > 0 or args.gripper_step > 0) and clipped > 0:
                log.info("clip: %d/%d dim-steps clipped",
                         clipped, STATE_DIM * steps)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt -- shutting down")
    finally:
        try:
            entry = prompt_journal_entry(journal_start_s, args)
            if entry is not None:
                write_journal_entry(args.journal_path, entry, args, journal_invocation)
        except Exception as e:
            log.warning("journal step failed: %s", e)

        abort = {"abort": False, "ctrlc_count": 0}
        def _cleanup_sigint(_sig, _frame):
            abort["ctrlc_count"] += 1
            if abort["ctrlc_count"] == 1:
                log.warning("Ctrl-C in cleanup: aborting return-ramp. Ctrl-C again to hard-exit.")
                abort["abort"] = True
            else:
                os._exit(130)
        try: signal.signal(signal.SIGINT, _cleanup_sigint)
        except Exception: pass

        if left is not None and right is not None and not args.no_return_on_exit:
            try:
                log.info("Returning arms to startup pose (%.1fs)...", args.ramp_duration_s)
                ramp_to_pose(left, right, startup_pose,
                             duration_s=args.ramp_duration_s,
                             abort_flag=abort, label="return-on-exit")
            except BaseException as e:
                log.warning("return ramp failed: %s", e)
        elif args.no_return_on_exit:
            log.warning("--no-return-on-exit: arms will drop")

        for c in (top, cam_l, cam_r):
            try: c.stop()
            except BaseException as e: log.warning("camera stop failed: %s", e)
        for arm in (left, right):
            if arm is None: continue
            try: arm.close()
            except BaseException as e: log.warning("arm.close() failed: %s", e)
        log.info("done")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
