"""Measure CAN read latency under increasing USB load.

Hypothesis: the CAN comm errors and gravity-comp control-loop slowdowns are
caused by USB-controller contention -- everything is on one USB 2.0 host
controller (one Genesys Logic hub: 3-9). When the cameras start streaming,
the xHCI interrupt budget gets eaten and CAN frames get delayed.

This script proves or disproves that by initializing ONE arm in
zero_torque_mode (motors disabled, only reads), then timing 1000+
get_joint_pos() calls under each of these conditions:

  A) idle           -- no cameras streaming, baseline
  B) webcam only    -- V4L2 UVC stream on /dev/video0
  C) D405 only      -- RealSense 424x240/30fps RGB
  D) webcam + D405  -- the run_client.py loadout

Run:
    /home/andon/yam-tests/i2rt/.venv/bin/python /home/andon/yam-tests/molmoact2-setup/scripts/usb_contention_test.py \\
        --can can0 --gripper linear_4310 \\
        --webcam-dev /dev/video0 \\
        --d405-serial 352122272708

The arm should NOT move during this test (zero_torque_mode + gripper cal
skipped). Support the elbow just in case.
"""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np


def percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def measure_can_latency(robot, duration_s: float = 6.0) -> np.ndarray:
    """Tight-loop get_joint_pos() for duration_s; return latencies in ms."""
    latencies = []
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        _ = robot.get_joint_pos()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return np.asarray(latencies, dtype=np.float64)


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
    import threading
    stop = threading.Event()

    def reader_loop():
        while not stop.is_set():
            cap.read()
    t = threading.Thread(target=reader_loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=1.0)
        cap.release()


@contextmanager
def d405_streaming(serial: str, w: int, h: int, fps: int) -> Iterator[None]:
    import pyrealsense2 as rs
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
    pipe = rs.pipeline()
    pipe.start(cfg)
    # Warmup
    for _ in range(5):
        try: pipe.wait_for_frames(timeout_ms=2000)
        except Exception: pass

    import threading
    stop = threading.Event()

    def reader_loop():
        while not stop.is_set():
            try:
                pipe.wait_for_frames(timeout_ms=200)
            except Exception:
                pass
    t = threading.Thread(target=reader_loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=1.0)
        pipe.stop()


def report(name: str, lat: np.ndarray) -> None:
    n = len(lat)
    print(
        f"  {name:<22s}  n={n:>5}  "
        f"mean={lat.mean():>6.2f}ms  p50={percentile(lat,50):>6.2f}  "
        f"p90={percentile(lat,90):>6.2f}  p99={percentile(lat,99):>6.2f}  "
        f"max={lat.max():>7.2f}ms  "
        f"slow(>10ms)={(lat>10).sum():>4}/{n}",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--can", default="can0")
    p.add_argument("--gripper", default="linear_4310")
    p.add_argument("--webcam-dev", default="/dev/video0")
    p.add_argument("--d405-serial", required=True)
    p.add_argument("--width", type=int, default=424)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration", type=float, default=6.0)
    args = p.parse_args()

    print(f"\nInitializing arm on {args.can} (zero_torque, no gripper cal)...", flush=True)
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType
    robot = get_yam_robot(
        channel=args.can,
        arm_type=ArmType.from_string_name("yam"),
        gripper_type=GripperType.from_string_name(args.gripper),
        zero_gravity_mode=False,
        gripper_limits_override=np.array([0.0, 1.0]),  # skip cal
    )
    robot.zero_torque_mode()  # no holding torque -- support arm by hand
    # Let SDK control thread settle
    time.sleep(0.5)

    print(f"\nMeasuring CAN read latency for {args.duration}s per condition.\n", flush=True)
    print(f"  conditions:                  stats over the duration")
    print(f"  {'-'*22}   {'-'*72}")

    # A: idle
    lat_idle = measure_can_latency(robot, args.duration)
    report("A) idle (baseline)", lat_idle)

    # B: webcam only
    with webcam_streaming(args.webcam_dev, args.width, args.height, args.fps):
        time.sleep(0.5)  # let stream stabilize
        lat_webcam = measure_can_latency(robot, args.duration)
    report("B) webcam only", lat_webcam)

    # C: D405 only
    with d405_streaming(args.d405_serial, args.width, args.height, args.fps):
        time.sleep(0.5)
        lat_d405 = measure_can_latency(robot, args.duration)
    report("C) D405 only", lat_d405)

    # D: webcam + D405
    with webcam_streaming(args.webcam_dev, args.width, args.height, args.fps), \
         d405_streaming(args.d405_serial, args.width, args.height, args.fps):
        time.sleep(0.5)
        lat_both = measure_can_latency(robot, args.duration)
    report("D) webcam + D405", lat_both)

    # Verdict
    print("\n  --- verdict ---")
    baseline_p99 = percentile(lat_idle, 99)
    worst_p99 = percentile(lat_both, 99)
    baseline_max = lat_idle.max()
    worst_max = lat_both.max()
    print(f"  p99 latency:   idle={baseline_p99:.2f} ms   -> with both cams={worst_p99:.2f} ms   "
          f"({worst_p99/max(baseline_p99,0.01):.1f}x)")
    print(f"  max latency:   idle={baseline_max:.2f} ms   -> with both cams={worst_max:.2f} ms   "
          f"({worst_max/max(baseline_max,0.01):.1f}x)")
    print(f"  slow reads (>10ms):   idle={(lat_idle>10).sum()}/{len(lat_idle)}   "
          f"-> with both cams={(lat_both>10).sum()}/{len(lat_both)}")

    if worst_p99 > 3 * baseline_p99 or worst_max > 10 * baseline_max:
        print("\n  CONFIRMED: cameras streaming significantly worsen CAN read latency.")
        print("  Fix: move USB devices to the USB 3.0 ports on bus 002 or bus 004.")
        rc = 1
    elif worst_p99 > 1.5 * baseline_p99:
        print("\n  MILD CONTENTION: cameras streaming slow CAN reads modestly but not dramatically.")
        rc = 0
    else:
        print("\n  NOT CONFIRMED: cameras don't significantly impact CAN latency.")
        print("  The control-loop slowness has another cause -- look elsewhere.")
        rc = 0

    try:
        robot.close()
    except Exception:
        pass
    return rc


if __name__ == "__main__":
    import os
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
