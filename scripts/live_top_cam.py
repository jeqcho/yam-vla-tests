"""Live-stream the top camera into a Rerun viewer.

Standalone tool -- doesn't touch the arms or model. Useful for:
  - Comparing today's top-cam framing to a saved Friday .rrd recording
  - A/B testing UVC vs RealSense top cam without restarting the REPL
  - Sanity-checking camera mounting before bringing arms online

Run via the i2rt venv:

    /home/andon/yam-tests/i2rt/.venv/bin/python \\
        /home/andon/yam-tests/eval-yam/scripts/live_top_cam.py

Defaults to whatever ``top_cam_*`` is set in yam_setup_config.json. Override:
    --serial 349622072241          # RealSense by serial
    --v4l2 /dev/video12            # UVC webcam by device path
    --width 424 --height 240 --fps 30
    --no-spawn --connect HOST:PORT # if a viewer is already running

Ctrl-C to quit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_CFG = "/home/andon/yam-tests/molmoact2-setup/yam_setup_config.json"


def _load_cfg():
    try:
        return json.loads(Path(DEFAULT_CFG).read_text())
    except Exception:
        return {}


def _open_realsense(serial: str, w: int, h: int, fps: int):
    import pyrealsense2 as rs
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
    pipe = rs.pipeline()
    pipe.start(cfg)
    # Wait for first frames so AE settles.
    for _ in range(5):
        try: pipe.wait_for_frames(timeout_ms=2000)
        except Exception: pass
    print(f"[live] RealSense {serial} @ {w}x{h}/{fps}fps OK", flush=True)
    def grab():
        frames = pipe.wait_for_frames(timeout_ms=2000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("no color frame")
        return np.asanyarray(color.get_data())
    def stop():
        pipe.stop()
    return grab, stop


def _open_v4l2(device: str, w: int, h: int, fps: int):
    import cv2
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(5): cap.read()
    print(f"[live] V4L2 {device} @ {w}x{h}/{fps}fps OK", flush=True)
    def grab():
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"no frame from {device}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    def stop():
        cap.release()
    return grab, stop


def main():
    p = argparse.ArgumentParser(description="Live top-cam Rerun stream")
    p.add_argument("--serial", default=None, help="RealSense serial. Default: yam_setup_config.json top_cam_serial")
    p.add_argument("--v4l2",   default=None, help="V4L2 device path. Default: yam_setup_config.json top_cam_v4l2")
    p.add_argument("--width",  type=int, default=424)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--fps",    type=int, default=30)
    p.add_argument("--no-spawn", action="store_true", help="don't spawn a viewer (use --connect)")
    p.add_argument("--connect", default=None, metavar="HOST:PORT", help="connect to existing viewer instead of spawning")
    args = p.parse_args()

    cfg = _load_cfg()
    if not args.serial and not args.v4l2:
        args.serial = cfg.get("top_cam_serial")
        args.v4l2   = cfg.get("top_cam_v4l2")

    if args.serial:
        grab, stop = _open_realsense(args.serial, args.width, args.height, args.fps)
        label = f"realsense:{args.serial}"
    elif args.v4l2:
        grab, stop = _open_v4l2(args.v4l2, args.width, args.height, args.fps)
        label = f"v4l2:{args.v4l2}"
    else:
        print("no top camera configured -- pass --serial or --v4l2", file=sys.stderr)
        sys.exit(2)

    import rerun as rr
    spawn = not (args.no_spawn or args.connect)
    rr.init("live_top_cam", spawn=spawn)
    if args.connect:
        host, _, port = args.connect.partition(":")
        rr.connect_grpc(f"rerun+http://{host}:{port}/proxy")
        print(f"[live] streaming to viewer at {args.connect}", flush=True)
    else:
        print("[live] spawned local Rerun viewer", flush=True)

    rr.log("info", rr.TextDocument(f"Live top cam stream -- {label}\n"
                                   f"{args.width}x{args.height} @ {args.fps} fps"))
    t0 = time.perf_counter()
    n = 0
    try:
        while True:
            img = grab()
            rr.set_time("time", duration=time.perf_counter() - t0)
            rr.log("cam/top", rr.Image(img))
            n += 1
            if n % args.fps == 0:
                print(f"[live] {n} frames", flush=True)
    except KeyboardInterrupt:
        print(f"\n[live] stopping after {n} frames", flush=True)
    finally:
        stop()


if __name__ == "__main__":
    main()
