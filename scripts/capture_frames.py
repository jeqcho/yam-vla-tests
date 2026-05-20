"""Snap one frame from each RealSense camera and save to disk for review.

Useful when verifying camera placement (top/left/right) before running the
policy. Saves to `frames/{name}_{serial}.png`.

    /home/andon/yam-tests/i2rt/.venv/bin/python scripts/capture_frames.py \\
        --top-cam-serial   AAAA \\
        --left-cam-serial  BBBB \\
        --right-cam-serial CCCC \\
        --outdir frames
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pyrealsense2 as rs
from PIL import Image


def grab_one(serial: str, name: str, w: int, h: int, fps: int) -> np.ndarray:
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
    pipe = rs.pipeline()
    pipe.start(cfg)
    try:
        # Discard the first ~5 frames so AE settles.
        for _ in range(5):
            pipe.wait_for_frames(timeout_ms=2000)
        frames = pipe.wait_for_frames(timeout_ms=2000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"{name} ({serial}) produced no color frame")
        return np.asanyarray(color.get_data())
    finally:
        pipe.stop()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top-cam-serial", required=True)
    p.add_argument("--left-cam-serial", required=True)
    p.add_argument("--right-cam-serial", required=True)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--outdir", default="frames")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for name, serial in [
        ("top", args.top_cam_serial),
        ("left", args.left_cam_serial),
        ("right", args.right_cam_serial),
    ]:
        img = grab_one(serial, name, args.width, args.height, args.fps)
        path = os.path.join(args.outdir, f"{name}_{serial}.png")
        Image.fromarray(img, mode="RGB").save(path)
        print(f"  saved {path}  shape={img.shape}")


if __name__ == "__main__":
    main()
