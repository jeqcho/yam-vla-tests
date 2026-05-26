"""Capture one frame from each declared camera and save as PNGs.

Useful for sanity-checking camera mounts before running the policy. Specifies
which camera goes into which slot (top / left_wrist / right_wrist) the same
way the client does.

Run with the i2rt venv:
    /home/andon/yam-tests/i2rt/.venv/bin/python scripts/capture_frames.py \\
        --out /tmp/yam_frames \\
        --top-cam-serial AAAA --left-cam-serial BBBB --right-cam-serial CCCC
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np


def _make_stream(name: str, serial: Optional[str], v4l2: Optional[str],
                 width: int, height: int, fps: int):
    if serial and v4l2:
        raise ValueError(f"{name}: pass exactly one of --{name}-cam-serial / --{name}-cam-v4l2")
    if not serial and not v4l2:
        return None
    if serial:
        import pyrealsense2 as rs
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        pipeline = rs.pipeline()
        pipeline.start(cfg)
        for _ in range(5):
            try: pipeline.wait_for_frames(timeout_ms=2000)
            except Exception: pass
        def _grab():
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            return np.asanyarray(color.get_data())
        def _stop():
            pipeline.stop()
        return _grab, _stop
    else:
        import cv2
        cap = cv2.VideoCapture(v4l2, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(5):
            cap.read()
        def _grab():
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"camera {name} ({v4l2}) produced no frame")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        def _stop():
            cap.release()
        return _grab, _stop


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output directory for PNGs.")
    p.add_argument("--top-cam-serial",   default=None)
    p.add_argument("--top-cam-v4l2",     default=None)
    p.add_argument("--left-cam-serial",  default=None)
    p.add_argument("--left-cam-v4l2",    default=None)
    p.add_argument("--right-cam-serial", default=None)
    p.add_argument("--right-cam-v4l2",   default=None)
    p.add_argument("--width",  type=int, default=424)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--fps",    type=int, default=30)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import cv2
    streams = []
    for name, serial, v4l2 in [
        ("top",   args.top_cam_serial,   args.top_cam_v4l2),
        ("left",  args.left_cam_serial,  args.left_cam_v4l2),
        ("right", args.right_cam_serial, args.right_cam_v4l2),
    ]:
        s = _make_stream(name, serial, v4l2, args.width, args.height, args.fps)
        if s is None:
            print(f"[capture] {name}: skipped (no --{name}-cam-serial / --{name}-cam-v4l2)")
            continue
        grab, stop = s
        try:
            img = grab()
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(args.out, f"{name}.png")
            cv2.imwrite(out_path, bgr)
            print(f"[capture] {name}: saved {img.shape[1]}x{img.shape[0]} -> {out_path}")
        finally:
            stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
