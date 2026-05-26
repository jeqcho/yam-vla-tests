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
import threading
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


def install_sdk_lock_fix() -> None:
    """Replace i2rt's dm_driver control loop with a version that doesn't hold
    command_lock during CAN I/O.

    The shipped loop (dm_driver.py:529) holds self.command_lock through the
    full 7-motor CAN round-trip (~3 ms). The OTHER SDK thread,
    motor_chain_robot._server_thread, also needs command_lock to push our
    target positions, and Linux's mutex isn't fair under sustained contention
    -- it gets starved for hundreds of ms. While starved, no new commands
    reach the motors; the SDK keeps streaming the last target at 300 Hz; the
    arm holds. Then the lock frees, the now-stale target gets pushed, the
    motor PD jumps -> visible burst motion.

    Patched loop holds command_lock only for a microsecond list-copy and does
    CAN I/O on the local copy. Acquire p99 drops from ~400 ms to <0.1 ms,
    set_commands throughput improves ~10x. Validated with test_sdk_lock_fix.py.

    Call once at process startup, BEFORE any DMChainCanInterface is created.
    """
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


# Apply the SDK lock fix immediately at import time so it lands before any
# DMChainCanInterface is constructed. See install_sdk_lock_fix() docstring.
install_sdk_lock_fix()

json_numpy.patch()

# Make stdout unbuffered so we can actually see where things hang.
# i2rt's logger may have already called basicConfig at import time; force-add
# our own StreamHandler so our messages always appear with timestamps.
import os
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

_root = logging.getLogger()
# Root stays at WARNING so i2rt + python-can periodic INFO reports (Grav Comp
# Control Frequency, DMChainCanInterface Total rate, PATCHED step_time
# reports) don't spam the REPL prompt. Real errors (WARNING+) still surface.
_root.setLevel(logging.WARNING)

# i2rt or python-can calls logging.basicConfig() at import time, which adds a
# default StreamHandler to root with the bare "INFO:root:..." format. Our
# handler would then sit alongside it and every record would print twice.
# Strip pre-existing handlers before installing ours.
if not any(getattr(h, "_yam_client_handler", False) for h in _root.handlers):
    for _h in list(_root.handlers):
        _root.removeHandler(_h)
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s | %(message)s"))
    _handler.setLevel(logging.INFO)
    _handler._yam_client_handler = True
    _root.addHandler(_handler)

# Our own logger keeps INFO so /act diagnostics, [boundary] lines, and the
# REPL's per-attempt status messages still show.
log = logging.getLogger("yam.client")
log.setLevel(logging.INFO)

# Explicitly cap the chattier third-party loggers. They keep WARNING+ so
# motor errors etc. still come through.
for _name in ("i2rt", "can",
              "can.interfaces.socketcan",
              "can.interfaces.socketcan.socketcan"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def trace(msg: str) -> None:
    """Always-flushed marker print so we see where the script is in real time."""
    print(f"[TRACE] {msg}", flush=True)


# --- Optional Rerun observability ----------------------------------------
# Holds the rerun module when --rerun is enabled, else None. Lazy-imported in
# main() so the import cost (~half a second) is only paid when requested.
_rr = None


def _rr_log_observation(t_s: float, top_img, left_img, right_img, state) -> None:
    """Log one observation (3 camera frames + 14-dim joint state) to Rerun.

    No-op if --rerun wasn't passed. Uses a monotonic 'time' timeline (seconds
    since process start) so the viewer scrubs cleanly. Joint state is split
    into left/right arm groups with one scalar entity per joint -- the viewer
    auto-stacks them into a plot.
    """
    if _rr is None:
        return
    _rr.set_time("time", duration=t_s)
    _rr.log("cam/top",   _rr.Image(top_img))
    _rr.log("cam/left",  _rr.Image(left_img))
    _rr.log("cam/right", _rr.Image(right_img))
    for i in range(6):
        _rr.log(f"state/left/j{i}",  _rr.Scalars(float(state[i])))
        _rr.log(f"state/right/j{i}", _rr.Scalars(float(state[i + 7])))
    _rr.log("state/left/gripper",  _rr.Scalars(float(state[6])))
    _rr.log("state/right/gripper", _rr.Scalars(float(state[13])))


def _rr_log_inference(t_s: float, actions, executed_idx: int, rtt_ms: float,
                      horizon_arm_span: float) -> None:
    """Log per-query inference outputs: rtt, horizon span, executed action.

    `executed_idx` is the index within `actions` we're about to send to the
    arms; we plot its 14 joint values so you can see action vs state on the
    same timeline.
    """
    if _rr is None:
        return
    _rr.set_time("time", duration=t_s)
    _rr.log("metrics/rtt_ms",           _rr.Scalars(float(rtt_ms)))
    _rr.log("metrics/horizon_arm_span", _rr.Scalars(float(horizon_arm_span)))
    a = actions[executed_idx]
    for i in range(6):
        _rr.log(f"action/left/j{i}",  _rr.Scalars(float(a[i])))
        _rr.log(f"action/right/j{i}", _rr.Scalars(float(a[i + 7])))
    _rr.log("action/left/gripper",  _rr.Scalars(float(a[6])))
    _rr.log("action/right/gripper", _rr.Scalars(float(a[13])))


# --- Research journal ----------------------------------------------------
# At end of every run, prompt the user for a one-line status report and
# append a structured markdown entry to journal.md. See prompt_journal_entry
# and write_journal_entry below. Set --no-journal to skip, or override the
# path with --journal-path.
from datetime import datetime

DEFAULT_JOURNAL_PATH = "/home/andon/yam-tests/molmoact2-setup/journal.md"
DEFAULT_SETUP_CONFIG_PATH = "/home/andon/yam-tests/molmoact2-setup/yam_setup_config.json"


def load_saved_config(path: str = DEFAULT_SETUP_CONFIG_PATH) -> dict:
    """Read yam_setup_config.json if present, else return {}.

    Returned dict is used as argparse defaults in main() so running
    identify_setup.py is the only step after hardware swaps -- CLI flags
    still override. Schema (all optional):
      left_can, right_can   : CAN channels
      gripper               : gripper type, applied to both arms
      top_cam_serial        : RealSense serial for top cam (wins over
                              top_cam_v4l2 if both set)
      top_cam_v4l2          : V4L2 device path for a UVC top camera
      left_cam_serial       : RealSense serial for left-arm camera
      right_cam_serial      : RealSense serial for right-arm camera
    """
    import json as _json
    try:
        with open(path) as f:
            cfg = _json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.warning("failed to read %s: %s -- using built-in defaults", path, e)
        return {}
    if cfg:
        log.info("Loaded saved setup config from %s", path)
    return cfg


def _journal_format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _journal_invocation() -> str:
    """Return the user's original shell invocation if run_client.sh exported
    YAM_INVOCATION, else fall back to sys.argv (the python-level call).
    """
    inv = os.environ.get("YAM_INVOCATION")
    if inv:
        return inv
    return " ".join(sys.argv)


def _journal_format_args(args) -> str:
    """Render argparse Namespace as a markdown bullet list. Skips defaults
    that are None/False to keep entries readable.
    """
    lines = []
    for k, v in sorted(vars(args).items()):
        if v is None or v is False:
            continue
        # Truncate long strings (instruction can be huge).
        sv = repr(v) if isinstance(v, str) and len(v) > 120 else str(v)
        lines.append(f"- `{k}`: {sv}")
    return "\n".join(lines) if lines else "_(none)_"


def prompt_journal_entry(start_time_s: float, args) -> Optional[dict]:
    """Interactively ask the user how the run went.

    Returns a dict with status/notes/purpose/duration/timestamp, or None if
    the user skipped (or stdin isn't a TTY -- so CI/piped runs are safe).
    """
    if not sys.stdin.isatty():
        print("[journal] stdin is not a TTY, skipping journal prompt", flush=True)
        return None
    if getattr(args, "no_journal", False):
        return None

    duration_s = time.time() - start_time_s
    print("\n" + "=" * 70, flush=True)
    print("Research journal -- record this run?", flush=True)
    print("=" * 70, flush=True)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Duration:    {_journal_format_duration(duration_s)}", flush=True)
    print("", flush=True)
    print("How did the run go?", flush=True)
    print("  [s] success  -- task completed as intended", flush=True)
    print("  [f] failure  -- task did not complete or had clear problems", flush=True)
    print("  [u] unclear  -- partial / mixed / hard to say", flush=True)
    print("  [enter or 'skip']  don't record this run", flush=True)
    sys.stdout.flush()
    try:
        choice = input("> ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n[journal] skipped", flush=True)
        return None
    if not choice or choice.startswith("skip"):
        print("[journal] skipped", flush=True)
        return None
    status_map = {"s": "success", "f": "failure", "u": "unclear"}
    status = status_map.get(choice[:1])
    if status is None:
        print(f"[journal] unrecognized status {choice!r}, skipping", flush=True)
        return None

    try:
        notes = input("\nWhat happened? (one line, optional)\n> ").strip()
        purpose = input("\nPurpose of this run? (optional, what were you testing)\n> ").strip()
    except (KeyboardInterrupt, EOFError):
        notes = locals().get("notes", "")
        purpose = ""
        print("\n[journal] partial entry recorded", flush=True)

    return {
        "status": status,
        "notes": notes,
        "purpose": purpose,
        "duration_s": duration_s,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_journal_entry(path: str, entry: dict, args, invocation: str) -> None:
    """Append a single markdown entry to the journal."""
    md = []
    md.append("")
    md.append("---")
    md.append(f"## {entry['timestamp']} -- {entry['status']}")
    md.append("")
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


# Default per-step caps (radians for joints, normalized for gripper).
# 0.15 rad/step at 30 Hz = 4.5 rad/s (~260 deg/s) joint velocity ceiling -- well
# above any speed the policy should naturally produce in-distribution, but still
# bounded enough that a single bad action chunk can't slam an arm. Pass
# --max-step-rad 0 to disable the clip entirely (raw model output, no safety
# net beyond i2rt's own 400 ms motor timeout).
DEFAULT_MAX_STEP_RAD = 0.15
DEFAULT_GRIPPER_STEP = 0.15
DEFAULT_TRAIN_FPS = 30.0   # the policy's training cadence — controls inner-loop pace
DEFAULT_HORIZON_STRIDE = 6 # play this many steps from each (30, 14) horizon before re-querying
STATE_DIM = 14   # per-arm 7-D × 2
ARM_DOFS = 7     # 6 arm joints + 1 gripper

# Path to the model's norm_stats.json. action_stats.mean is the centroid of
# the training distribution -- a good "ready" pose to start inference from.
NORM_STATS_PATH = (
    "/home/andon/yam-tests/molmoact2-setup/hf-cache/hub/"
    "models--allenai--MolmoAct2-BimanualYAM/snapshots/"
    "28e56c0fa4cb8598bfc2261e45499b3cc77763d4/norm_stats.json"
)
NORM_TAG = "yam_dual_molmoact2"


def load_training_mean_pose() -> np.ndarray:
    """Return the 14-D centroid of the training action distribution."""
    import json as _json
    with open(NORM_STATS_PATH) as f:
        d = _json.load(f)
    mean = d["metadata_by_tag"][NORM_TAG]["action_stats"]["mean"]
    return np.asarray(mean, dtype=np.float32)


def ramp_to_pose(
    left, right, target_14d: np.ndarray,
    duration_s: float = 5.0, hz: float = 30.0,
    abort_flag: dict | None = None,
    label: str = "ramp",
) -> None:
    """Linearly interpolate both arms from their current pose to target_14d.
    abort_flag['abort'] = True causes the loop to stop at the next step (the arms
    are left at the last commanded interpolation point -- they will NOT fall as
    long as the SDK control threads are still running and commanding that pose).
    """
    q_l = np.asarray(left.get_joint_pos(),  dtype=np.float32)
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
            log.warning("[%s] aborted at step %d/%d -- arms held at intermediate pose",
                        label, i, n_steps)
            return
        alpha = i / n_steps
        cmd = start + alpha * delta
        left.command_joint_pos(cmd[:7].astype(np.float32))
        right.command_joint_pos(cmd[7:].astype(np.float32))
        time.sleep(dt)
    # Hold briefly so PD settles.
    time.sleep(0.5)
    log.info("[%s] done", label)


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
        # Wait until we've actually received some warmup frames, not just
        # spent a fixed number of timeouts pretending to. D405s usually
        # produce a first frame within ~1 s; a D435 on USB 2.0 has been
        # observed to take 10+ s -- if we bail too early start() returns
        # "successfully" but the next grab() raises.
        budget_s = 20.0
        target_frames = 3
        deadline = time.monotonic() + budget_s
        got = 0
        while got < target_frames and time.monotonic() < deadline:
            try:
                self.pipeline.wait_for_frames(timeout_ms=2000)
                got += 1
            except Exception:
                pass
        if got == 0:
            raise RuntimeError(
                f"camera {self.name} (RealSense {self.serial}) produced no "
                f"frames within {budget_s:.0f}s of pipeline start -- check "
                f"USB port (D435 needs USB 3) and cable."
            )
        log.info("camera %s (RealSense %s) started @ %dx%d/%d Hz "
                 "(warmup %d frames)",
                 self.name, self.serial, self.width, self.height, self.fps, got)

    def grab(self) -> np.ndarray:
        # One retry on a transient "Frame didn't arrive within 2000" -- those
        # have been observed right after USB re-enumeration. A second waiter
        # almost always succeeds. The retry uses the same pipeline; if it's
        # genuinely stuck, the second attempt will time out too and we raise.
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        except RuntimeError:
            log.warning("camera %s (%s): grab timeout, retrying once",
                        self.name, self.serial)
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"camera {self.name} ({self.serial}) produced no color frame")
        return np.asanyarray(color.get_data())

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError as e:
                # "stop() cannot be called before start()" means the pipeline
                # never streamed (probably a failed start). Not worth raising
                # during teardown.
                log.warning("camera %s stop noop: %s", self.name, e)


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
    """Build the right camera backend.

    If both serial and v4l2_device are set, serial wins -- this lets a CLI
    `--<slot>-cam-serial X` override a `top_cam_v4l2` left in the saved
    setup config from a previous hardware generation without having to
    explicitly clear the v4l2 entry. Re-run identify_setup.py to clean up.
    """
    if not serial and not v4l2_device:
        raise ValueError(f"{name}: must pass --{name}-cam-serial or --{name}-cam-v4l2")
    if serial:
        if v4l2_device:
            log.warning("%s: both --%s-cam-serial and --%s-cam-v4l2 set; "
                        "using serial. Re-run identify_setup.py to update config.",
                        name, name, name)
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
) -> tuple[np.ndarray, int]:
    """Clip the desired action so each joint moves at most max_step_rad from
    the current state in this tick.

    max_step_rad <= 0 disables the joint clip (gripper clip is also disabled
    iff gripper_step <= 0 by the same mechanism, so passing 0 to both yields
    pass-through behavior -- the policy's raw output goes straight to the
    motors). i2rt's own 400 ms motor timeout is the only remaining safety.

    Returns (cmd_actually_sent, n_clipped_dims) so callers can tally how
    often the cap fires per query.
    """
    if desired_action.shape != (STATE_DIM,):
        raise ValueError(f"action shape {desired_action.shape} != ({STATE_DIM},)")
    delta = desired_action - current_state
    # Per-arm caps: indices 0..5 + 7..12 are arm joints, 6 + 13 are grippers.
    # A non-positive cap means "no cap on that dimension" -- use +inf so the
    # clip is a no-op there.
    caps = np.full(STATE_DIM,
                   max_step_rad if max_step_rad > 0 else np.inf,
                   dtype=np.float32)
    caps[6]  = gripper_step if gripper_step > 0 else np.inf
    caps[13] = gripper_step if gripper_step > 0 else np.inf
    clipped_delta = np.clip(delta, -caps, caps)
    n_clipped = int(np.sum(clipped_delta != delta))
    cmd = (current_state + clipped_delta).astype(np.float32)
    left.command_joint_pos(cmd[:ARM_DOFS])
    right.command_joint_pos(cmd[ARM_DOFS:])
    return cmd, n_clipped


class AsyncInferenceFetcher:
    """Overlap inference with execution.

    Usage:
        fetcher = AsyncInferenceFetcher(...)
        fetcher.kick_off(state, top_img, left_img, right_img)   # non-blocking
        # ... execute current chunk, do other work ...
        actions, rtt_ms = fetcher.wait_for_result()              # blocks if not done

    Stores the in-flight POST in a background thread. Re-raises any HTTP /
    parsing errors when .wait_for_result() is called. Single-slot: a new
    kick_off() while one is in flight will raise.
    """
    def __init__(self, server_url: str, instruction: str, num_steps: int, timeout_s: float):
        self._url = server_url
        self._instr = instruction
        self._num_steps = num_steps
        self._timeout_s = timeout_s
        self._thread: Optional[threading.Thread] = None
        # Result is set by the worker thread. Read with thread.join() first.
        # Sentinel value None means 'not yet completed'.
        self._result: Optional[tuple] = None
        self._error: Optional[str] = None

    def kick_off(self, state: np.ndarray, top_img: np.ndarray,
                 left_img: np.ndarray, right_img: np.ndarray) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("AsyncInferenceFetcher: previous request still in flight")
        # Snapshot the inputs so the caller can mutate / re-grab without racing.
        s = np.ascontiguousarray(state, dtype=np.float32).copy()
        t = np.ascontiguousarray(top_img).copy()
        l = np.ascontiguousarray(left_img).copy()
        r = np.ascontiguousarray(right_img).copy()
        self._result = None
        self._error = None

        def _worker():
            try:
                actions, rtt_ms = post_actions(
                    self._url, t, l, r, s, self._instr, self._num_steps, self._timeout_s
                )
                self._result = (actions, rtt_ms)
            except BaseException as e:  # noqa: BLE001
                self._error = repr(e)
        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def wait_for_result(self) -> tuple:
        if self._thread is None:
            raise RuntimeError("AsyncInferenceFetcher.wait_for_result with no in-flight request")
        self._thread.join()
        self._thread = None
        if self._error is not None:
            raise RuntimeError(f"async /act failed: {self._error}")
        return self._result


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
    # Capture wall-clock start of the run for the research journal duration.
    journal_start_s = time.time()
    journal_invocation = _journal_invocation()

    # Defaults are sourced from yam_setup_config.json (populated by
    # identify_setup.py) so that after a hardware swap, re-running
    # identify_setup.py is the only step needed -- no script edits.
    _cfg = load_saved_config()
    _gripper_default = _cfg.get("gripper", "linear_4310")
    p = argparse.ArgumentParser(description="MolmoAct2-BimanualYAM client")
    p.add_argument("--left-can",  default=_cfg.get("left_can",  "can0"),
                   help="CAN channel for the LEFT arm")
    p.add_argument("--right-can", default=_cfg.get("right_can", "can1"),
                   help="CAN channel for the RIGHT arm")
    p.add_argument("--left-gripper",  default=_gripper_default,
                   choices=["crank_4310", "linear_3507", "linear_4310", "flexible_4310"],
                   help="Gripper type on the left arm")
    p.add_argument("--right-gripper", default=_gripper_default,
                   choices=["crank_4310", "linear_3507", "linear_4310", "flexible_4310"],
                   help="Gripper type on the right arm")
    # Per-camera: pass exactly one of --<slot>-cam-serial (RealSense) or --<slot>-cam-v4l2 (UVC webcam, e.g. /dev/video0)
    p.add_argument("--top-cam-serial",   default=_cfg.get("top_cam_serial"),
                   help="RealSense serial for overhead (top) camera")
    p.add_argument("--top-cam-v4l2",     default=_cfg.get("top_cam_v4l2"),
                   help="V4L2 device path for overhead (top) camera, e.g. /dev/video0")
    p.add_argument("--left-cam-serial",  default=_cfg.get("left_cam_serial"),
                   help="RealSense serial for left-arm camera")
    p.add_argument("--left-cam-v4l2",    default=_cfg.get("left_cam_v4l2"),
                   help="V4L2 device path for left-arm camera")
    p.add_argument("--right-cam-serial", default=_cfg.get("right_cam_serial"),
                   help="RealSense serial for right-arm camera")
    p.add_argument("--right-cam-v4l2",   default=_cfg.get("right_cam_v4l2"),
                   help="V4L2 device path for right-arm camera")
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
                   help="Per-arm-joint per-tick clip (rad). At 30 Hz, 0.15 caps "
                        "joint velocity at ~4.5 rad/s. Pass 0 to disable the clip "
                        "entirely (raw policy output goes to motors).")
    p.add_argument("--gripper-step", type=float, default=DEFAULT_GRIPPER_STEP,
                   help="Gripper per-tick clip (normalized units). Pass 0 to disable.")
    p.add_argument("--dump-frames",  default=None,
                   help="If set, save the first {top,left,right} frame the client sends to "
                        "the server into this directory as PNGs, then exit. Useful for "
                        "visually verifying the model is seeing what we think it is.")
    p.add_argument("--move-to-ready", action="store_true",
                   help="Before inference, linearly ramp both arms from their startup pose "
                        "to the MolmoAct2 training-distribution centroid (~shoulder 79°, elbow 70°). "
                        "Without this the model often hedges flat near-identity actions.")
    p.add_argument("--ramp-duration-s", type=float, default=5.0,
                   help="seconds for move-to-ready (and return-to-start on exit) ramps")
    p.add_argument("--no-return-on-exit", action="store_true",
                   help="DANGEROUS: skip the return-to-startup-pose ramp at exit. "
                        "If your startup pose was upright/stowed, you NEED the return ramp -- "
                        "without it the arms drop when the SDK disables motors on close().")
    p.add_argument("--horizon-stride", type=int, default=DEFAULT_HORIZON_STRIDE,
                   help="Apply this many steps from each returned horizon before re-querying. "
                        "With train_fps=30 and stride=6, server is queried 5 Hz.")
    p.add_argument("--inference-mode", default="sync",
                   choices=["sync", "async-naive", "async-time-aligned"],
                   help="sync (default): POST blocks the inner loop -- arm holds for "
                        "~RTT ms between chunks, stop-and-go motion. "
                        "async-naive: kick off next POST when current chunk starts "
                        "executing, apply a[0..stride-1] of each new chunk -- expected "
                        "to have a backward jump at every chunk boundary (Phase 2 test). "
                        "async-time-aligned: same overlap, but apply a[K..K+stride-1] "
                        "where K=stride to compensate for inference latency -- the "
                        "smooth replacement for sync (Phase 3).")
    p.add_argument("--timeout-s", type=float, default=15.0,
                   help="HTTP timeout per /act call (steady state). The first call after the server "
                        "comes up may take >5s because the model re-captures CUDA graphs for the "
                        "client's specific image shape; the script's explicit warmup uses a longer "
                        "timeout of its own.")
    p.add_argument("--warmup-timeout-s", type=float, default=60.0,
                   help="HTTP timeout for the one-shot warmup call. Bump if the server is loading.")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't command the arms; print actions only")
    p.add_argument("--rerun", action="store_true",
                   help="Stream observations (3 cam frames + 14-dim joint state) and "
                        "per-query actions/RTT to a Rerun viewer. By default spawns "
                        "the viewer locally; use --rerun-connect to point at a remote.")
    p.add_argument("--rerun-connect", default=None, metavar="HOST:PORT",
                   help="Connect to an existing rerun viewer at HOST:PORT instead of "
                        "spawning one. Example: 127.0.0.1:9876")
    p.add_argument("--rerun-save", default=None, metavar="PATH",
                   help="Also save the rerun recording to a .rrd file. Even if the "
                        "live viewer lags, the file lets you replay the full session "
                        "later with `rerun PATH`. Implies --rerun.")
    p.add_argument("--no-journal", action="store_true",
                   help="Skip the end-of-run prompt that asks how the run went. "
                        "Default behavior is to ask and append a markdown entry to "
                        "the journal file.")
    p.add_argument("--journal-path", default=DEFAULT_JOURNAL_PATH,
                   help=f"Path to the research journal (markdown, appended). "
                        f"Default: {DEFAULT_JOURNAL_PATH}")
    args = p.parse_args()

    # Loud-warn the user if they've disabled the per-step clip. Six months from
    # now we want this to be impossible to miss in the scrollback.
    if args.max_step_rad <= 0 and args.gripper_step <= 0:
        log.warning("=" * 70)
        log.warning("--max-step-rad=0 AND --gripper-step=0: PER-STEP CLIPPING DISABLED")
        log.warning("Arms will track raw policy output. The only remaining safety")
        log.warning("is i2rt's 400 ms motor timeout. If the model produces a bad")
        log.warning("action chunk, the arms WILL execute it.")
        log.warning("=" * 70)
    elif args.max_step_rad <= 0:
        log.warning("--max-step-rad=0: arm-joint clipping disabled (grippers still clipped at %.3f)",
                    args.gripper_step)
    elif args.gripper_step <= 0:
        log.warning("--gripper-step=0: gripper clipping disabled (arms still clipped at %.3f rad)",
                    args.max_step_rad)

    # Initialize Rerun viewer if requested. Done before arms init so any setup
    # failures (missing display, port already in use) happen before motors turn on.
    rerun_requested = args.rerun or (args.rerun_save is not None)
    if rerun_requested:
        try:
            import rerun as rr
            global _rr
            _rr = rr
            # Spawn whatever rerun is first on PATH. The user has reported
            # that the older system-wide viewer (e.g. 0.26.x) renders video
            # playback better than the SDK-matching venv viewer (0.32.x) on
            # this machine, so we don't force the venv binary. A version
            # skew may trigger a warning at startup -- that's expected.
            rr.init("yam_inference", spawn=(args.rerun_connect is None))
            if args.rerun_connect:
                host, _, port = args.rerun_connect.partition(":")
                rr.connect_grpc(f"rerun+http://{host}:{port}/proxy")
                log.info("Rerun: connected to viewer at %s", args.rerun_connect)
            else:
                log.info("Rerun: spawned local viewer")
            if args.rerun_save:
                rr.save(args.rerun_save)
                log.info("Rerun: also saving recording to %s", args.rerun_save)
        except ImportError:
            log.error("--rerun requested but rerun-sdk not installed in this venv. "
                      "Install with: VIRTUAL_ENV=/home/andon/yam-tests/i2rt/.venv "
                      "uv pip install rerun-sdk")
            sys.exit(2)
        except Exception as e:
            log.error("Rerun init failed: %s. Continuing without it.", e)
            _rr = None

    # Health-check the server first so we fail fast.
    health_url = args.server_url.rstrip("/").rsplit("/", 1)[0] + "/act" if args.server_url.endswith("/act") else args.server_url
    try:
        r = requests.get(health_url, timeout=3.0)
        r.raise_for_status()
        log.info("server health: %s", r.json())
    except Exception as e:
        log.error("server health check failed at %s: %s", health_url, e)
        sys.exit(2)

    # Cameras BEFORE arms. The D405 USB-3 enumeration storm and the CAN
    # adapters share the same Intel xHCI controller (PCI 0000:80:14.0, IRQ
    # 138). When pyrealsense2 starts a D405 pipeline AFTER motors are
    # already taking commands at 30 Hz, the burst of USB control transfers
    # delays gs_usb URBs long enough to trip the DM motor watchdog
    # ("loss communication on motor X"). Initializing cameras first means
    # the USB storm is over by the time the motor control thread spins up,
    # so there is nothing to time out.
    left: Optional[Robot] = None
    right: Optional[Robot] = None
    top = cam_l = cam_r = None
    try:
        cam_kw = dict(width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
        trace(f"building cameras at {args.cam_width}x{args.cam_height}/{args.cam_fps}fps")
        top   = make_camera("top",   args.top_cam_serial,   args.top_cam_v4l2,   **cam_kw)
        cam_l = make_camera("left",  args.left_cam_serial,  args.left_cam_v4l2,  **cam_kw)
        cam_r = make_camera("right", args.right_cam_serial, args.right_cam_v4l2, **cam_kw)
        for c in (top, cam_l, cam_r):
            trace(f"starting camera {c.name}")
            c.start()
            trace(f"camera {c.name} started")
        # Settle: grab a couple of frames so the auto-exposure has converged
        # and the pipelines are fully past their startup transients before
        # we let the motor control thread come online. Tolerant of misses:
        # start() already proved each cam can produce frames; the control
        # loop grabs have their own timeout/retry.
        for _ in range(3):
            for c in (top, cam_l, cam_r):
                try: c.grab()
                except Exception as e: log.warning("settle: %s.grab() failed: %s", c.name, e)
        trace("cameras streaming, USB quiet -- safe to init arms")
    except Exception:
        # If camera setup failed, close anything we partially opened then
        # re-raise. We have not started motor threads yet so arms aren't
        # at risk.
        for c in (top, cam_l, cam_r):
            if c is not None:
                try: c.stop()
                except Exception: pass
        raise

    trace("about to init LEFT arm")
    left = init_arm(args.left_can, args.left_gripper)
    trace("about to init RIGHT arm")
    right = init_arm(args.right_can, args.right_gripper)
    trace("both arms initialized")

    # SAFETY: capture the user's startup pose RIGHT NOW. The DM motors need
    # continuous position commands to stay up; close() zeroes torques and the
    # arms drop. Before exit we will ramp the arms back to this startup pose,
    # whatever they chose it to be (presumably a stable rest pose).
    startup_pose = np.concatenate([
        np.asarray(left.get_joint_pos(),  dtype=np.float32),
        np.asarray(right.get_joint_pos(), dtype=np.float32),
    ])
    log.info("Captured startup pose for return-on-exit: %s",
             np.array2string(startup_pose, precision=3))

    # Optional: move arms to MolmoAct2 training-mean pose so the model has
    # in-distribution proprioception to ground on. Keeps grippers at startup.
    if args.move_to_ready:
        target = load_training_mean_pose()
        target[6]  = startup_pose[6]
        target[13] = startup_pose[13]
        log.info("--move-to-ready: ramping arms to training-mean pose (5s)...")
        ramp_to_pose(left, right, target, duration_s=args.ramp_duration_s,
                     label="move-to-ready")

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

    # Wall-clock origin for the rerun timeline. We use a monotonic clock so
    # the viewer's time axis is stable even if the system clock jumps.
    loop_t0 = time.perf_counter()

    # Chunk-boundary telemetry (Phase 1 of async refactor): track the last raw
    # action of the previous chunk so we can log the discontinuity at every
    # chunk transition. In sync mode the discontinuity should be small (arm
    # holds during POST, model's a[0] of new chunk approx equals state approx
    # equals previous chunk's a[stride-1]). In naive async it will spike. In
    # time-aligned async it should return to the sync baseline.
    last_chunk_tail: Optional[np.ndarray] = None
    boundary_idx = 0

    try:
        if args.inference_mode != "sync":
            # =====================================================================
            # ASYNC INFERENCE PATH (Phase 2: async-naive; Phase 3: async-time-aligned)
            # ---------------------------------------------------------------------
            # Overlap inference and execution by kicking off the NEXT POST in a
            # background thread the moment the current chunk begins executing.
            # If RTT < stride*dt the next chunk arrives before execution ends,
            # so there is no idle gap between chunks (no stop-and-go).
            #
            # Action slicing differs by mode:
            #   async-naive:        apply actions[0..stride-1]
            #   async-time-aligned: apply actions[stride..2*stride-1]
            #
            # async-naive is the deliberately-broken Phase 2 control case: each
            # chunk's a[0] was planned for the state we sent ~RTT ms ago, but
            # the arm has now moved ~stride*dt of actual motion past that state,
            # so applying a[0] commands a backward jump. The per-step clip
            # catches it but motion is jerky.
            #
            # async-time-aligned is Phase 3: since the chunk was generated
            # ~stride*dt ago and we're about to apply it now, the model's
            # prediction for "the action at relative time stride*dt" is a[stride],
            # which approximately equals the next intended pose. No jump.
            # =====================================================================
            chunk_start_idx = args.horizon_stride if args.inference_mode == "async-time-aligned" else 0
            log.info("ASYNC mode = %s, chunk_start_idx = %d (apply a[%d..%d])",
                     args.inference_mode, chunk_start_idx,
                     chunk_start_idx, chunk_start_idx + args.horizon_stride - 1)

            fetcher = AsyncInferenceFetcher(args.server_url, args.instruction,
                                            args.num_steps, args.timeout_s)

            # Bootstrap: do a synchronous first chunk so we have something to
            # execute on iteration 1.
            log.info("Async bootstrap: fetching initial chunk synchronously...")
            state = read_state(left, right)
            top_img = top.grab()
            left_img = cam_l.grab()
            right_img = cam_r.grab()
            _rr_log_observation(time.perf_counter() - loop_t0,
                                top_img, left_img, right_img, state)
            fetcher.kick_off(state, top_img, left_img, right_img)
            actions, rtt_ms = fetcher.wait_for_result()
            log.info("Async bootstrap OK (rtt=%.0f ms, actions shape=%s)",
                     rtt_ms, actions.shape)

            while not stop_flag["stop"]:
                # Sample state + cams for the NEXT POST (which we're about to
                # kick off). 'next_state' is what we'll log boundary diagnostics
                # against -- it represents the arm's actual position at the
                # moment we start executing the current chunk.
                next_state = read_state(left, right)
                next_top = top.grab()
                next_left = cam_l.grab()
                next_right = cam_r.grab()
                _rr_log_observation(time.perf_counter() - loop_t0,
                                    next_top, next_left, next_right, next_state)

                # Kick off the next /act in the background -- runs while we
                # execute the current chunk on the arms.
                fetcher.kick_off(next_state, next_top, next_left, next_right)

                # Per-query diagnostic on the CURRENT actions (which we are
                # about to apply). Use next_state, the arm's actual position
                # right now, so the |a[i]-state| numbers tell us about the
                # action we're about to send.
                def _arm_delta_max_async(a_idx: int) -> float:
                    d = actions[a_idx] - next_state
                    return float(max(np.max(np.abs(d[:6])), np.max(np.abs(d[7:13]))))
                a0_d  = _arm_delta_max_async(0)
                a5_d  = _arm_delta_max_async(min(5, actions.shape[0]-1))
                a10_d = _arm_delta_max_async(min(10, actions.shape[0]-1))
                a19_d = _arm_delta_max_async(min(19, actions.shape[0]-1))
                a29_d = _arm_delta_max_async(actions.shape[0]-1)
                horizon_range = (actions.max(axis=0) - actions.min(axis=0))
                horizon_arm_span = float(max(np.max(horizon_range[:6]),
                                              np.max(horizon_range[7:13])))
                log.info(
                    "/act rtt=%dms  arm |a[i]-state|_max @ i=0/5/10/19/29: %.3f/%.3f/%.3f/%.3f/%.3f rad  "
                    "horizon_span=%.3f rad  L_grip[0,29]=%.2f,%.2f  R_grip[0,29]=%.2f,%.2f",
                    rtt_ms, a0_d, a5_d, a10_d, a19_d, a29_d, horizon_arm_span,
                    actions[0][6],  actions[-1][6],
                    actions[0][13], actions[-1][13],
                )

                stride = max(1, args.horizon_stride)
                n_to_play = min(stride, actions.shape[0] - chunk_start_idx)
                if n_to_play <= 0:
                    log.warning("Async: chunk too short (shape=%s, chunk_start_idx=%d)",
                                actions.shape, chunk_start_idx)
                    break
                _rr_log_inference(time.perf_counter() - loop_t0, actions,
                                  executed_idx=chunk_start_idx, rtt_ms=rtt_ms,
                                  horizon_arm_span=horizon_arm_span)

                # Boundary diagnostic. In async mode the "first applied action"
                # is a[chunk_start_idx], not a[0]. Measure how it relates to
                # the arm's current actual position. Big disagreement = jump.
                if last_chunk_tail is not None:
                    arm_idx = np.r_[0:6, 7:13]
                    a_first = actions[chunk_start_idx]
                    state_vs_first_arm = float(np.max(np.abs(a_first[arm_idx] - next_state[arm_idx])))
                    tail_vs_first_arm = float(np.max(np.abs(a_first[arm_idx] - last_chunk_tail[arm_idx])))
                    state_vs_first_grip_l = float(abs(a_first[6]  - next_state[6]))
                    state_vs_first_grip_r = float(abs(a_first[13] - next_state[13]))
                    boundary_idx += 1
                    log.info(
                        "[boundary] #%d  apply_idx=%d  state_vs_a[K](arm)=%.3f rad  "
                        "tail_vs_a[K](arm)=%.3f rad  "
                        "state_vs_a[K](grip L,R)=%.2f,%.2f",
                        boundary_idx, chunk_start_idx, state_vs_first_arm, tail_vs_first_arm,
                        state_vs_first_grip_l, state_vs_first_grip_r,
                    )

                # Execute actions[chunk_start_idx : chunk_start_idx+n_to_play]
                clipped_this_query = 0
                steps_this_query = 0
                for i in range(n_to_play):
                    if stop_flag["stop"]:
                        break
                    step_start = time.perf_counter()
                    action_idx = chunk_start_idx + i
                    desired = actions[action_idx].astype(np.float32)
                    if args.dry_run:
                        log.info("dry-run action[%d]: %s", action_idx,
                                 np.array2string(desired, precision=3))
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
                        log.info("clip: %d/%d dim-steps clipped (%.1f%%) "
                                 "[--max-step-rad=%.3f --gripper-step=%.3f]",
                                 clipped_this_query, max_possible, pct,
                                 args.max_step_rad, args.gripper_step)

                # Stash the last raw action we applied so the next boundary
                # check can compute the discontinuity.
                last_chunk_tail = actions[chunk_start_idx + n_to_play - 1].astype(np.float32).copy()

                # Wait for the in-flight chunk (the one we kicked off at the
                # top of this iteration). If RTT < stride*dt this returns
                # immediately; otherwise we wait the residual latency.
                actions, rtt_ms = fetcher.wait_for_result()

        else:
            # =====================================================================
            # SYNC INFERENCE PATH (default, the original behavior)
            # =====================================================================
            while not stop_flag["stop"]:
                state = read_state(left, right)
                top_img = top.grab()
                left_img = cam_l.grab()
                right_img = cam_r.grab()
                _rr_log_observation(time.perf_counter() - loop_t0,
                                    top_img, left_img, right_img, state)

                # One-shot frame-dump for visual debugging. Run with --dump-frames /tmp/foo
                # then inspect /tmp/foo/top.png / left.png / right.png to see exactly what
                # the model is being shown. NOTE: don't re-import os here; it's already at
                # module level. A local `import os` inside this block would shadow it and
                # break the finally's os._exit(0) call when dump-frames is NOT set.
                if args.dump_frames:
                    import cv2
                    os.makedirs(args.dump_frames, exist_ok=True)
                    for name, img in [("top", top_img), ("left", left_img), ("right", right_img)]:
                        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        out_path = os.path.join(args.dump_frames, f"{name}.png")
                        cv2.imwrite(out_path, bgr)
                        log.info("dumped %s (%dx%d) to %s", name, img.shape[1], img.shape[0], out_path)
                    log.info("dump-frames mode -- exiting before any inference.")
                    sys.stdout.flush()
                    os._exit(0)

                actions, rtt_ms = post_actions(
                    args.server_url, top_img, left_img, right_img, state,
                    args.instruction, args.num_steps, args.timeout_s,
                )

                # Per-query diagnostic: what is the model actually asking for?
                # We log:
                #   - |action[i]-state|_max for i in {0, 5, 10, 19, 29} (arm joints only)
                #     -> shows WHERE in the horizon the model wants to move
                #   - horizon span(arm) -> max-min across all 30 actions
                # If |a[29]-state| is large but |a[0..19]-state| is small, the model
                # plans motion AFTER the stride cutoff and we never execute it.
                def _arm_delta_max(a_idx: int) -> float:
                    d = actions[a_idx] - state
                    return float(max(np.max(np.abs(d[:6])), np.max(np.abs(d[7:13]))))
                a0_d  = _arm_delta_max(0)
                a5_d  = _arm_delta_max(min(5,  actions.shape[0]-1))
                a10_d = _arm_delta_max(min(10, actions.shape[0]-1))
                a19_d = _arm_delta_max(min(19, actions.shape[0]-1))
                a29_d = _arm_delta_max(actions.shape[0]-1)
                horizon_range = (actions.max(axis=0) - actions.min(axis=0))
                horizon_arm_span = float(max(np.max(horizon_range[:6]),
                                              np.max(horizon_range[7:13])))
                log.info(
                    "/act rtt=%dms  arm |a[i]-state|_max @ i=0/5/10/19/29: %.3f/%.3f/%.3f/%.3f/%.3f rad  "
                    "horizon_span=%.3f rad  L_grip[0,29]=%.2f,%.2f  R_grip[0,29]=%.2f,%.2f",
                    rtt_ms, a0_d, a5_d, a10_d, a19_d, a29_d, horizon_arm_span,
                    actions[0][6],  actions[-1][6],
                    actions[0][13], actions[-1][13],
                )

                stride = max(1, args.horizon_stride)
                n_to_play = min(stride, actions.shape[0])
                _rr_log_inference(time.perf_counter() - loop_t0, actions,
                                  executed_idx=0, rtt_ms=rtt_ms,
                                  horizon_arm_span=horizon_arm_span)

                # Chunk-boundary telemetry. Two quantities:
                #   state_vs_a0: |new_chunk.a[0] - current_arm_state|, max over 12
                #                arm joints. Tells us how much the arm would
                #                "jump" if we naively apply a[0] right now.
                #   tail_vs_a0:  |new_chunk.a[0] - prev_chunk.a[stride-1]|, max
                #                over 12 arm joints. Tells us how big the
                #                discontinuity is between the model's last
                #                command and the model's next command.
                # In sync, state_vs_a0 should be small (arm held during POST).
                # In naive async, state_vs_a0 will spike because the arm moved
                # during POST but the model planned from a stale state.
                if last_chunk_tail is not None:
                    arm_idx = np.r_[0:6, 7:13]
                    a0 = actions[0]
                    state_vs_a0_arm = float(np.max(np.abs(a0[arm_idx] - state[arm_idx])))
                    tail_vs_a0_arm = float(np.max(np.abs(a0[arm_idx] - last_chunk_tail[arm_idx])))
                    state_vs_a0_grip_l = float(abs(a0[6]  - state[6]))
                    state_vs_a0_grip_r = float(abs(a0[13] - state[13]))
                    boundary_idx += 1
                    log.info(
                        "[boundary] #%d  state_vs_a0(arm)=%.3f rad  "
                        "tail_vs_a0(arm)=%.3f rad  "
                        "state_vs_a0(grip L,R)=%.2f,%.2f",
                        boundary_idx, state_vs_a0_arm, tail_vs_a0_arm,
                        state_vs_a0_grip_l, state_vs_a0_grip_r,
                    )

                # Count joints clipped across this stride. Useful for tuning
                # --max-step-rad: if "clipped" is consistently >0 you're capping
                # legitimate motion; if it stays 0, your cap is loose enough.
                clipped_this_query = 0
                steps_this_query = 0
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
                        _, n_clipped = safe_command(left, right, state, desired,
                                                    args.max_step_rad, args.gripper_step)
                        clipped_this_query += n_clipped
                        steps_this_query += 1
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
                # Per-query clip telemetry: total clipped dims across this chunk.
                # 14 dims × steps_this_query is the max possible. Logging here so
                # it sits next to the /act diagnostics in the stdout stream.
                if steps_this_query > 0 and (args.max_step_rad > 0 or args.gripper_step > 0):
                    max_possible = STATE_DIM * steps_this_query
                    pct = 100.0 * clipped_this_query / max_possible
                    if clipped_this_query > 0:
                        log.info("clip: %d/%d dim-steps clipped (%.1f%%) "
                                 "[--max-step-rad=%.3f --gripper-step=%.3f]",
                                 clipped_this_query, max_possible, pct,
                                 args.max_step_rad, args.gripper_step)

                # Stash the last raw action we sent so the next iteration's
                # boundary log can compute the discontinuity. Use the raw action
                # (pre-clip) since we're measuring what the MODEL is producing,
                # not what the safety clip allowed through.
                if n_to_play > 0:
                    last_chunk_tail = actions[n_to_play - 1].astype(np.float32).copy()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt -- shutting down")
    finally:
        # Research journal goes FIRST so a Ctrl-C during cleanup can't skip
        # it. Motors are still in position-hold mode from the last commanded
        # action -- they will hold (not drift) for the seconds the user
        # spends answering. We have NOT yet installed the cleanup SIGINT
        # handler, so SIGINT here uses Python's default behavior and raises
        # KeyboardInterrupt, which prompt_journal_entry catches as 'skip'.
        try:
            entry = prompt_journal_entry(journal_start_s, args)
            if entry is not None:
                write_journal_entry(args.journal_path, entry, args, journal_invocation)
        except Exception as e:
            log.warning("journal step failed: %s", e)

        # SAFETY: Before disabling motors we ramp arms back to startup_pose
        # so they end up where the user knows they can be safely de-powered.
        # close() zeros torques -> arms fall under gravity -> ARMS DROP.
        # The ONLY way to exit safely is to first reach a pose where the
        # arms naturally rest. We ramp back to whatever pose they were in
        # when the script started -- the user picked that pose; it's safe.
        #
        # Ctrl-C handling during cleanup:
        #   - 1st Ctrl-C: abort the return ramp, arms drop, traceback warns.
        #   - 2nd Ctrl-C: hard-exit immediately.
        # Single dict shared between SIGINT handler and ramp_to_pose. The ramp
        # checks abort["abort"] each step -- mutations through this reference
        # propagate live.
        abort = {"abort": False, "ctrlc_count": 0}
        def _cleanup_sigint(_sig, _frame):
            abort["ctrlc_count"] += 1
            if abort["ctrlc_count"] == 1:
                log.warning("Ctrl-C in cleanup: aborting return-ramp. ARMS WILL DROP. "
                            "Ctrl-C again to hard-exit.")
                abort["abort"] = True
            else:
                os._exit(130)
        try:
            signal.signal(signal.SIGINT, _cleanup_sigint)
        except Exception:
            pass

        # Return the arms to startup_pose BEFORE closing them. This is the
        # critical safety step. While this is running the SDK control threads
        # are still alive commanding position, so the arms hold.
        if left is not None and right is not None and 'startup_pose' in locals() \
                and not args.no_return_on_exit:
            try:
                log.info("Returning arms to startup pose (%.1fs ramp) before disable...",
                         args.ramp_duration_s)
                ramp_to_pose(left, right, startup_pose,
                             duration_s=args.ramp_duration_s,
                             abort_flag=abort,
                             label="return-on-exit")
                if abort["abort"]:
                    log.warning("return ramp was aborted -- arms may be mid-trajectory")
            except BaseException as e:
                log.warning("return-to-startup ramp failed: %s. ARMS MAY DROP.", e)
        elif args.no_return_on_exit:
            log.warning("--no-return-on-exit set: skipping return ramp. ARMS WILL DROP "
                        "if they are not in a pose that rests stably.")

        log.info("Stopping cameras")
        for c in (top, cam_l, cam_r):
            try:
                c.stop()
            except BaseException as e:
                log.warning("camera %s stop failed: %s", c.name, e)

        # NOW close the arm SDKs. close() zeros torque + closes CAN socket.
        # Arms will lose holding torque after this. They MUST be in a pose
        # where that's acceptable -- ramp above should have put them there.
        log.info("Closing arm SDKs (motors will lose holding torque now)")
        for arm in (left, right):
            if arm is None:
                continue
            try:
                arm.close()
            except BaseException as e:
                log.warning("arm.close() failed: %s", e)
        log.info("Arms returned to startup pose and motors disabled.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
