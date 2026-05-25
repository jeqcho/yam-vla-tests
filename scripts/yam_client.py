"""Bimanual YAM client for the GR00T N1.7 PolicyServer.

Wire-format reference: gr00t/policy/server_client.py in NVIDIA/Isaac-GR00T.
Talks msgpack-numpy over a ZeroMQ REQ socket (default port 5556).

Run with the i2rt venv (which has the i2rt SDK, pyrealsense2, and we've added
pyzmq + msgpack-numpy to it earlier):

    /home/andon/yam-tests/i2rt/.venv/bin/python scripts/yam_client.py \
        --left-can can0 --right-can can1 \
        --left-gripper linear_4310 --right-gripper linear_4310 \
        --top-cam-serial AAAA --left-cam-serial BBBB --right-cam-serial CCCC \
        --server-host 127.0.0.1 --server-port 5556 \
        --instruction "first pick up the left orange cube and put it in the box, then pick up the right orange cube and put it in the box" \
        --train-fps 30 --horizon-stride 4 --max-step-rad 0.05 --gripper-step 0.05

Safety: every command is clipped to within --max-step-rad of the current state
per arm joint, and the gripper to --gripper-step per step. Ctrl+C stops the
loop; the arms hold their last commanded position — kill power if that pose
isn't safe.

Schema mapping (GR00T modality_keys -> YAM 14-D state):

    obs["state"]["left_arm"]      shape (B=1, T=1, 6)  joints 0..5
    obs["state"]["left_gripper"]  shape (B=1, T=1, 1)  normalized [0, 1]
    obs["state"]["right_arm"]     shape (B=1, T=1, 6)
    obs["state"]["right_gripper"] shape (B=1, T=1, 1)
    obs["video"]["top"]           shape (B=1, T=1, H, W, 3) uint8 RGB
    obs["video"]["left_wrist"]    shape (B=1, T=1, H, W, 3) uint8 RGB
    obs["video"]["right_wrist"]   shape (B=1, T=1, H, W, 3) uint8 RGB
    obs["language"]["annotation.human.task_description"]  list[list[str]]  (B, 1)

    actions["left_arm"]      shape (1, 16, 6)
    actions["left_gripper"]  shape (1, 16, 1)
    actions["right_arm"]     shape (1, 16, 6)
    actions["right_gripper"] shape (1, 16, 1)

Decoded back to the 14-D YAM layout: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)].
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np

# i2rt imports — provided by /home/andon/yam-tests/i2rt/.venv. Don't import the
# upstream gr00t package here; we only need the zmq+msgpack wire protocol
# (PolicyClient), and we re-implement a slim client below so this script works
# from an environment that does NOT have the heavyweight gr00t deps installed.
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmType, GripperType


# =============================================================================
# SDK lock fix copied verbatim from the MolmoAct2 client. The motor SDK has a
# starvation bug; patching the control loop drops set_commands p99 from ~400 ms
# to <0.1 ms. See molmoact2-setup/REPORT_yam_sdk_lock_investigation.md for the
# full investigation.
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
                            _logging.warning(f"Motor error in control loop, attempting recovery: {e}")
                            recovered = self._try_recover_motors()
                            if recovered:
                                _logging.warning("Motor recovery successful, continuing control loop")
                                continue
                            else:
                                self.running = False
                                raise
                        raise
                    errors = np.array([motor_feedback[i].error_code != "0x1" for i in range(len(motor_feedback))])
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
_handler.setLevel(logging.INFO)
if not any(getattr(h, "_yam_client_handler", False) for h in _root.handlers):
    _handler._yam_client_handler = True
    _root.addHandler(_handler)
log = logging.getLogger("yam.gr00t.client")


def trace(msg: str) -> None:
    print(f"[TRACE] {msg}", flush=True)


# =============================================================================
# Slim PolicyClient — talks the same wire protocol as gr00t/policy/server_client.py
# without pulling in the full gr00t package (which has heavy CUDA deps).
# =============================================================================

import msgpack_numpy as _mnp
import zmq as _zmq


class GrootPolicyClient:
    """Minimal ZeroMQ REQ client compatible with gr00t.policy.server_client.PolicyServer.

    Implements only the endpoints we need: ping, get_action.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5556, timeout_ms: int = 30000):
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

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        resp = self._call("get_action", {"observation": observation, "options": options})
        # Server returns (action_dict, info_dict)
        action, info = resp
        return action, info


# =============================================================================
# Camera + arm code copied from molmoact2-setup/scripts/yam_client.py.
# Identical hardware, identical safety needs.
# =============================================================================

DEFAULT_MAX_STEP_RAD = 0.15
DEFAULT_GRIPPER_STEP = 0.15
DEFAULT_TRAIN_FPS = 30.0
DEFAULT_HORIZON_STRIDE = 4   # GR00T default action_horizon=16; play 4 then re-query (~7.5 Hz at 30 fps)
STATE_DIM = 14
ARM_DOFS = 7


class CameraStream:
    def __init__(self, name: str, width: int = 424, height: int = 240, fps: int = 30):
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
        log.info("camera %s (RealSense %s) started @ %dx%d/%d Hz", self.name, self.serial, self.width, self.height, self.fps)

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
        log.info("camera %s (V4L2 %s) started @ %dx%d/%d Hz", self.name, self.device, self.width, self.height, self.fps)

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


def init_arm(can_channel: str, gripper: str, ee_mass: Optional[float] = None) -> Robot:
    trace(f"init_arm({can_channel}, {gripper}): entering get_yam_robot...")
    arm_type = ArmType.from_string_name("yam")
    gripper_type = GripperType.from_string_name(gripper)
    robot = get_yam_robot(
        channel=can_channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
        zero_gravity_mode=False,
        ee_mass=ee_mass,
    )
    q0 = np.asarray(robot.get_joint_pos(), dtype=np.float32)
    robot.command_joint_pos(q0)
    trace(f"init_arm({can_channel}): joint_pos={np.array2string(q0, precision=3)}, holding")
    return robot


def read_state(left: Robot, right: Robot) -> np.ndarray:
    """14-D state vector: [left_q0..5, left_grip, right_q0..5, right_grip]."""
    s_l = np.asarray(left.get_joint_pos(), dtype=np.float32).reshape(-1)
    s_r = np.asarray(right.get_joint_pos(), dtype=np.float32).reshape(-1)
    if s_l.shape != (ARM_DOFS,) or s_r.shape != (ARM_DOFS,):
        raise RuntimeError(f"expected ({ARM_DOFS},) per arm, got left={s_l.shape}, right={s_r.shape}")
    return np.concatenate([s_l, s_r], axis=0).astype(np.float32)


def safe_command(
    left: Robot, right: Robot,
    current_state: np.ndarray, desired_action: np.ndarray,
    max_step_rad: float, gripper_step: float,
) -> tuple[np.ndarray, int]:
    if desired_action.shape != (STATE_DIM,):
        raise ValueError(f"action shape {desired_action.shape} != ({STATE_DIM},)")
    delta = desired_action - current_state
    caps = np.full(STATE_DIM, max_step_rad if max_step_rad > 0 else np.inf, dtype=np.float32)
    caps[6] = gripper_step if gripper_step > 0 else np.inf
    caps[13] = gripper_step if gripper_step > 0 else np.inf
    clipped_delta = np.clip(delta, -caps, caps)
    n_clipped = int(np.sum(clipped_delta != delta))
    cmd = (current_state + clipped_delta).astype(np.float32)
    left.command_joint_pos(cmd[:ARM_DOFS])
    right.command_joint_pos(cmd[ARM_DOFS:])
    return cmd, n_clipped


# =============================================================================
# GR00T wire format helpers.
# =============================================================================

def build_observation(
    top_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    state_14d: np.ndarray,
    instruction: str,
) -> dict:
    """Pack a single observation in GR00T (B=1, T=1) format.

    All arrays carry an explicit batch axis (1,) and time axis (1,) up front.
    """
    def _add_bt(arr: np.ndarray) -> np.ndarray:
        return arr[np.newaxis, np.newaxis, ...]

    obs = {
        "video": {
            "top": _add_bt(top_img.astype(np.uint8)),
            "left_wrist": _add_bt(left_img.astype(np.uint8)),
            "right_wrist": _add_bt(right_img.astype(np.uint8)),
        },
        "state": {
            "left_arm": _add_bt(state_14d[0:6].astype(np.float32)),
            "left_gripper": _add_bt(state_14d[6:7].astype(np.float32)),
            "right_arm": _add_bt(state_14d[7:13].astype(np.float32)),
            "right_gripper": _add_bt(state_14d[13:14].astype(np.float32)),
        },
        "language": {
            "annotation.human.task_description": [[instruction]],
        },
    }
    return obs


def decode_actions(action_chunk: dict, horizon: int) -> np.ndarray:
    """Convert GR00T action dict back into a (horizon, 14) numpy array in YAM order.

    Returned axis 1 layout: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)].
    """
    la = np.asarray(action_chunk["left_arm"], dtype=np.float32)        # (B, T, 6)
    lg = np.asarray(action_chunk["left_gripper"], dtype=np.float32)    # (B, T, 1)
    ra = np.asarray(action_chunk["right_arm"], dtype=np.float32)       # (B, T, 6)
    rg = np.asarray(action_chunk["right_gripper"], dtype=np.float32)   # (B, T, 1)
    # Strip batch.
    la, lg, ra, rg = la[0], lg[0], ra[0], rg[0]
    if not (la.shape[0] == lg.shape[0] == ra.shape[0] == rg.shape[0]):
        raise RuntimeError(
            f"action horizon mismatch: left_arm={la.shape}, left_gripper={lg.shape}, "
            f"right_arm={ra.shape}, right_gripper={rg.shape}"
        )
    T = la.shape[0]
    if horizon != -1 and T != horizon:
        log.warning("server returned action horizon %d, expected %d", T, horizon)
    actions = np.concatenate([la, lg, ra, rg], axis=-1)  # (T, 14)
    if actions.shape[-1] != STATE_DIM:
        raise RuntimeError(f"decoded action dim {actions.shape[-1]} != {STATE_DIM}")
    return actions


def ramp_to_pose(
    left: Robot, right: Robot, target_14d: np.ndarray,
    duration_s: float = 5.0, hz: float = 30.0,
    abort_flag: dict | None = None,
    label: str = "ramp",
) -> None:
    q_l = np.asarray(left.get_joint_pos(), dtype=np.float32)
    q_r = np.asarray(right.get_joint_pos(), dtype=np.float32)
    start = np.concatenate([q_l, q_r])
    goal = np.asarray(target_14d, dtype=np.float32).copy()
    delta = goal - start
    max_d = float(np.max(np.abs(delta)))
    log.info("[%s] start=%s", label, np.array2string(start, precision=3))
    log.info("[%s] goal =%s", label, np.array2string(goal, precision=3))
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
        left.command_joint_pos(cmd[:7].astype(np.float32))
        right.command_joint_pos(cmd[7:].astype(np.float32))
        time.sleep(dt)
    time.sleep(0.5)
    log.info("[%s] done", label)


# =============================================================================
# Journal — same shape as MolmoAct2 client; writes to grootn1.7's journal.md.
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
    print("\nHow did the run go?", flush=True)
    print("  [s] success  [f] failure  [u] unclear  [enter] skip", flush=True)
    try:
        choice = input("> ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return None
    if not choice:
        return None
    status_map = {"s": "success", "f": "failure", "u": "unclear"}
    status = status_map.get(choice[:1])
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
    md = ["", "---", f"## {entry['timestamp']} -- {entry['status']}", ""]
    if entry.get("purpose"):
        md.append(f"**Purpose**: {entry['purpose']}")
        md.append("")
    if entry.get("notes"):
        md.append(f"**Notes**: {entry['notes']}")
        md.append("")
    md.append(f"**Duration**: {_journal_format_duration(entry['duration_s'])}")
    md.append("")
    md.append("**Command**:")
    md.append("```")
    md.append(invocation)
    md.append("```")
    md.append("")
    md.append("**Configuration**:")
    md.append(_journal_format_args(args))
    md.append("")
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

    p = argparse.ArgumentParser(description="GR00T N1.7 bimanual-YAM client")
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
    p.add_argument("--cam-width",  type=int, default=424)
    p.add_argument("--cam-height", type=int, default=240)
    p.add_argument("--cam-fps",    type=int, default=30)
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=5556,
                   help="ZeroMQ port the GR00T server is listening on")
    p.add_argument("--server-timeout-ms", type=int, default=30000)
    p.add_argument("--instruction", required=True,
                   help="Natural-language task")
    p.add_argument("--train-fps", type=float, default=DEFAULT_TRAIN_FPS)
    p.add_argument("--max-step-rad", type=float, default=DEFAULT_MAX_STEP_RAD)
    p.add_argument("--gripper-step", type=float, default=DEFAULT_GRIPPER_STEP)
    p.add_argument("--horizon-stride", type=int, default=DEFAULT_HORIZON_STRIDE,
                   help="Play this many steps from each (16, 14) horizon before re-querying. "
                        "With train_fps=30 and stride=4, server is queried ~7.5 Hz.")
    p.add_argument("--action-horizon", type=int, default=16,
                   help="Expected horizon. Used only for validation; the server tells us "
                        "what it actually returns.")
    p.add_argument("--dump-frames", default=None,
                   help="Save first {top,left,right} frames as PNGs into this dir and exit.")
    p.add_argument("--move-to-ready", action="store_true",
                   help="(no-op for GR00T — no fixed training-mean pose available; left in "
                        "the CLI for symmetry with the MolmoAct2 client)")
    p.add_argument("--ramp-duration-s", type=float, default=5.0)
    p.add_argument("--no-return-on-exit", action="store_true",
                   help="DANGEROUS: skip return-to-startup ramp at exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't command the arms; print actions only.")
    p.add_argument("--no-journal", action="store_true")
    p.add_argument("--journal-path", default=DEFAULT_JOURNAL_PATH)
    args = p.parse_args()

    if args.max_step_rad <= 0 and args.gripper_step <= 0:
        log.warning("=" * 70)
        log.warning("ALL CLIPPING DISABLED — arms will track raw policy output.")
        log.warning("=" * 70)

    # 1. Connect to server and ping.
    client = GrootPolicyClient(host=args.server_host, port=args.server_port,
                               timeout_ms=args.server_timeout_ms)
    try:
        pong = client.ping()
        log.info("server ping: %s", pong)
    except Exception as e:
        log.error("server ping failed at %s:%d -- %s", args.server_host, args.server_port, e)
        sys.exit(2)

    # 2. Cameras BEFORE arms (USB enumeration storm vs CAN watchdog).
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

    # 3. Arms.
    left = init_arm(args.left_can, args.left_gripper)
    right = init_arm(args.right_can, args.right_gripper)
    startup_pose = np.concatenate([
        np.asarray(left.get_joint_pos(),  dtype=np.float32),
        np.asarray(right.get_joint_pos(), dtype=np.float32),
    ])
    log.info("startup pose: %s", np.array2string(startup_pose, precision=3))

    stop_flag = {"stop": False}
    inner_dt = 1.0 / args.train_fps
    ideal_query_hz = args.train_fps / max(1, args.horizon_stride)
    log.info("control loop: train_fps=%.1f Hz, stride=%d (ideal re-query ~%.1f Hz)",
             args.train_fps, args.horizon_stride, ideal_query_hz)
    log.info("per-tick caps: arm=%.3f rad, gripper=%.3f", args.max_step_rad, args.gripper_step)

    # 4. Warmup.
    try:
        state = read_state(left, right)
        obs = build_observation(top.grab(), cam_l.grab(), cam_r.grab(),
                                state, args.instruction)
        t0 = time.perf_counter()
        action_chunk, info = client.get_action(obs)
        warm_rtt_ms = (time.perf_counter() - t0) * 1000.0
        sample_action = decode_actions(action_chunk, args.action_horizon)
        log.info("warmup OK: rtt=%.0f ms, action shape=(horizon=%d, 14)",
                 warm_rtt_ms, sample_action.shape[0])
    except Exception as e:
        log.error("server warmup failed: %s -- proceeding anyway", e)

    loop_t0 = time.perf_counter()

    try:
        while not stop_flag["stop"]:
            state = read_state(left, right)
            top_img = top.grab()
            left_img = cam_l.grab()
            right_img = cam_r.grab()

            if args.dump_frames:
                import cv2
                os.makedirs(args.dump_frames, exist_ok=True)
                for name, img in [("top", top_img), ("left", left_img), ("right", right_img)]:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out_path = os.path.join(args.dump_frames, f"{name}.png")
                    cv2.imwrite(out_path, bgr)
                    log.info("dumped %s (%dx%d) to %s", name, img.shape[1], img.shape[0], out_path)
                log.info("dump-frames mode -- exiting")
                sys.stdout.flush()
                os._exit(0)

            obs = build_observation(top_img, left_img, right_img, state, args.instruction)
            t0 = time.perf_counter()
            action_chunk, info = client.get_action(obs)
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            actions = decode_actions(action_chunk, args.action_horizon)

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
                "/get_action rtt=%dms  arm |a[i]-state|_max @ 0/mid/end: %.3f/%.3f/%.3f rad  "
                "horizon_span=%.3f rad  L_grip[0,-1]=%.2f,%.2f  R_grip[0,-1]=%.2f,%.2f",
                rtt_ms, a0_d, a_mid, a_end, horizon_arm_span,
                actions[0][6], actions[-1][6], actions[0][13], actions[-1][13],
            )

            stride = max(1, args.horizon_stride)
            n_to_play = min(stride, actions.shape[0])
            clipped_this_query = 0
            steps_this_query = 0
            for i in range(n_to_play):
                if stop_flag["stop"]:
                    break
                step_start = time.perf_counter()
                desired = actions[i].astype(np.float32)
                if args.dry_run:
                    log.info("dry-run action[%d]: %s", i, np.array2string(desired, precision=3))
                else:
                    state = read_state(left, right)
                    _, n_clipped = safe_command(left, right, state, desired,
                                                args.max_step_rad, args.gripper_step)
                    clipped_this_query += n_clipped
                    steps_this_query += 1
                sleep_left = inner_dt - (time.perf_counter() - step_start)
                if sleep_left > 0:
                    time.sleep(sleep_left)
                elif sleep_left < -0.050:
                    log.warning("inner step overrun by %.1f ms (target %.1f ms)",
                                -sleep_left * 1000.0, inner_dt * 1000.0)
            if steps_this_query > 0 and (args.max_step_rad > 0 or args.gripper_step > 0):
                max_possible = STATE_DIM * steps_this_query
                pct = 100.0 * clipped_this_query / max_possible
                if clipped_this_query > 0:
                    log.info("clip: %d/%d dim-steps clipped (%.1f%%)",
                             clipped_this_query, max_possible, pct)
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
        try:
            signal.signal(signal.SIGINT, _cleanup_sigint)
        except Exception:
            pass

        if left is not None and right is not None and not args.no_return_on_exit:
            try:
                log.info("Returning arms to startup pose (%.1fs ramp)...", args.ramp_duration_s)
                ramp_to_pose(left, right, startup_pose,
                             duration_s=args.ramp_duration_s,
                             abort_flag=abort, label="return-on-exit")
            except BaseException as e:
                log.warning("return ramp failed: %s. ARMS MAY DROP.", e)
        elif args.no_return_on_exit:
            log.warning("--no-return-on-exit: skipping return ramp. ARMS WILL DROP.")

        for c in (top, cam_l, cam_r):
            try: c.stop()
            except BaseException as e: log.warning("camera %s stop failed: %s", c.name, e)
        for arm in (left, right):
            if arm is None: continue
            try: arm.close()
            except BaseException as e: log.warning("arm.close() failed: %s", e)
        log.info("Arms returned to startup pose and motors disabled.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
