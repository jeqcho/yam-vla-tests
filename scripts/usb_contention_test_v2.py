"""v2 -- diagnose CAN comm errors by counting them under different camera loads.

The previous version measured the wrong thing: get_joint_pos() returns a
cached state from a Python lock, so we were timing memory accesses (~ns),
not CAN round-trips. Actual CAN traffic happens in the SDK's background
control thread.

This version:
  1. Initializes BOTH arms via the SDK -- realistic CAN load (~300 polls/s
     per bus across 7 motors). Arms in zero_torque mode so they don't move.
  2. Captures the SDK's log output via a custom logging handler that
     counts "loss communication" errors per second.
  3. Runs four 20-second windows -- idle, webcam, D405, both -- and
     reports comm-error rate per window.

If the error rate goes up substantially when cameras stream, USB load is
implicated. If it's the same in all conditions, the errors come from
elsewhere (motor firmware, polling-rate-vs-DM-response-time mismatch,
kernel CAN driver).

Run:
    /home/andon/yam-tests/i2rt/.venv/bin/python /home/andon/yam-tests/molmoact2-setup/scripts/usb_contention_test_v2.py \\
        --left-can can0 --right-can can1 \\
        --left-gripper linear_4310 --right-gripper linear_4310 \\
        --webcam-dev /dev/video0 \\
        --d405-left-serial  352122272708 \\
        --d405-right-serial 427622271914
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np


# Matches the SDK's error format:
# "motor id: N, error: loss communication at yam_real and channel socketcan channel 'canX'"
COMM_ERR_RE = re.compile(
    r"motor id:\s*(\d+),\s*error:\s*loss communication.*channel '(can\d+)'"
)


class ErrorCounter(logging.Handler):
    """Logging handler that increments per-channel comm-error counts.

    Plain class -- not a dataclass -- because logging.Handler.__init__ puts
    `self` in a weakset that requires hashability, and @dataclass kills the
    default identity __hash__.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.counts: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        m = COMM_ERR_RE.search(record.getMessage())
        if m:
            motor_id = m.group(1)
            channel = m.group(2)
            key = f"{channel}:motor{motor_id}"
            self.counts[key] = self.counts.get(key, 0) + 1


@contextmanager
def webcam_streaming(dev_path: str, w: int, h: int, fps: int) -> Iterator[None]:
    import cv2
    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"can't open {dev_path}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(5):
        cap.read()
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            cap.read()
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set(); t.join(timeout=1.0); cap.release()


@contextmanager
def d405_streaming(serial: str, w: int, h: int, fps: int) -> Iterator[None]:
    import pyrealsense2 as rs
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
    pipe = rs.pipeline()
    pipe.start(cfg)
    for _ in range(5):
        try: pipe.wait_for_frames(timeout_ms=2000)
        except Exception: pass

    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try: pipe.wait_for_frames(timeout_ms=200)
            except Exception: pass
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set(); t.join(timeout=1.0); pipe.stop()


def measure_window(counter: ErrorCounter, duration_s: float, label: str) -> dict:
    """Snapshot the counter before/after the window. Returns errors/sec by channel."""
    before = dict(counter.counts)
    t0 = time.perf_counter()
    time.sleep(duration_s)
    actual = time.perf_counter() - t0
    after = dict(counter.counts)
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)}
    total = sum(delta.values())
    return {"label": label, "duration_s": actual, "total": total,
            "rate_per_s": total / actual, "by_motor": delta}


def report(window: dict) -> None:
    print(f"  {window['label']:<22s}  duration={window['duration_s']:.1f}s   "
          f"errors={window['total']:>3d}   rate={window['rate_per_s']:.2f}/s   "
          f"breakdown={dict(sorted([(k,v) for k,v in window['by_motor'].items() if v>0])) or '{} (clean)'}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--left-can",  default="can0")
    p.add_argument("--right-can", default="can1")
    p.add_argument("--left-gripper",  default="linear_4310")
    p.add_argument("--right-gripper", default="linear_4310")
    p.add_argument("--webcam-dev", default="/dev/video0")
    p.add_argument("--d405-left-serial", required=True)
    p.add_argument("--d405-right-serial", required=True)
    p.add_argument("--width", type=int, default=424)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--window-s", type=float, default=20.0,
                   help="seconds per condition")
    args = p.parse_args()

    # Install the error counter on the root logger BEFORE the SDK runs.
    counter = ErrorCounter()
    logging.getLogger().addHandler(counter)
    logging.getLogger().setLevel(logging.INFO)

    print("Initializing both arms (zero_torque, no gripper cal, no motion)...", flush=True)
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType

    def init(channel, gripper):
        return get_yam_robot(
            channel=channel,
            arm_type=ArmType.from_string_name("yam"),
            gripper_type=GripperType.from_string_name(gripper),
            zero_gravity_mode=False,
            gripper_limits_override=np.array([0.0, 1.0]),
        )

    left  = init(args.left_can,  args.left_gripper)
    right = init(args.right_can, args.right_gripper)
    left.zero_torque_mode()
    right.zero_torque_mode()

    print(f"\nSDK control threads now polling motors on {args.left_can} and {args.right_can}.")
    print(f"Counting 'loss communication' errors over {args.window_s:.0f}s windows.\n", flush=True)
    print("  condition              window stats")
    print(f"  {'-'*22}   {'-'*72}")

    # Let SDK threads warm up; clear early init noise.
    time.sleep(2.0)
    counter.counts.clear()

    results = []
    results.append(measure_window(counter, args.window_s, "A) idle (baseline)"))
    report(results[-1])

    with webcam_streaming(args.webcam_dev, args.width, args.height, args.fps):
        time.sleep(0.5)
        results.append(measure_window(counter, args.window_s, "B) webcam only"))
    report(results[-1])

    with d405_streaming(args.d405_left_serial, args.width, args.height, args.fps):
        time.sleep(0.5)
        results.append(measure_window(counter, args.window_s, "C) D405 #1 only"))
    report(results[-1])

    with d405_streaming(args.d405_left_serial,  args.width, args.height, args.fps), \
         d405_streaming(args.d405_right_serial, args.width, args.height, args.fps):
        time.sleep(0.5)
        results.append(measure_window(counter, args.window_s, "D) 2x D405"))
    report(results[-1])

    with webcam_streaming(args.webcam_dev, args.width, args.height, args.fps), \
         d405_streaming(args.d405_left_serial,  args.width, args.height, args.fps), \
         d405_streaming(args.d405_right_serial, args.width, args.height, args.fps):
        time.sleep(0.5)
        results.append(measure_window(counter, args.window_s, "E) webcam + 2x D405"))
    report(results[-1])

    # Verdict
    print("\n  --- verdict ---")
    baseline = results[0]["rate_per_s"]
    worst = max(r["rate_per_s"] for r in results[1:])
    worst_label = next(r["label"] for r in results[1:] if r["rate_per_s"] == worst)
    print(f"  baseline (idle):                {baseline:.2f} errors/s")
    print(f"  worst load:    {worst_label:<24s} {worst:.2f} errors/s  ({worst/max(baseline,0.001):.1f}x baseline)")
    if baseline > 0.05 and worst < 2 * baseline:
        print("  -> Comm errors happen at the same rate regardless of camera load.")
        print("     USB contention is NOT the dominant cause.")
        print("     Likely cause: DM motor firmware not always responding within the SDK's")
        print("     ~3ms polling deadline (SDK polls at ~300 Hz). This is intermittent and")
        print("     tolerable -- the SDK retries internally and the control loop continues.")
    elif worst > 3 * baseline:
        print("  -> Comm errors increase substantially with camera load.")
        print("     USB contention IS a factor. Try moving cameras to USB 3.0 ports.")
    else:
        print("  -> Modest increase with camera load. USB contention contributes but isn't")
        print("     the only factor.")

    try: left.close()
    except Exception: pass
    try: right.close()
    except Exception: pass
    return 0


if __name__ == "__main__":
    import os
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
