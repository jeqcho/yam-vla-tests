"""Measure whether the synchronous /act client is starving on inference.

Hits the live MolmoAct2 server with REAL camera frames (no arms involved)
in two modes:

  sync  : POST -> wait for response -> 'execute' stride steps (sleep
          stride/30 s, since we're not driving motors here) -> POST again.
          This mirrors yam_client.py's actual loop structure.

  async : POST in a background thread; the moment we start executing the
          current stride, kick off the NEXT POST. The next chunk should
          arrive before we finish executing, masking the inference cost.

Compares effective replan rate, mean cycle time, and idle ratio at several
stride values. Output is a markdown table.

CRITICAL: This script does NOT initialize the arms or import the i2rt SDK.
It exists purely to characterize HTTP + inference behavior independently of
the arm control loop.

Usage:

    /home/andon/yam-tests/i2rt/.venv/bin/python \\
        /home/andon/yam-tests/molmoact2-setup/scripts/diagnose_sync_overhead.py \\
        --strides 4,6,10,15,30 --n-cycles 25
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import json_numpy

json_numpy.patch()

CONFIG_PATH = Path("/home/andon/yam-tests/molmoact2-setup/yam_setup_config.json")
DEFAULT_INSTRUCTION = "pick up the orange cube"


# ---------- camera capture (read-only, no arms) ----------

class _RSStream:
    def __init__(self, serial: str, w: int, h: int, fps: int) -> None:
        import pyrealsense2 as rs
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        self.pipe = rs.pipeline()
        self.pipe.start(cfg)
        for _ in range(5):
            try: self.pipe.wait_for_frames(timeout_ms=2000)
            except Exception: pass

    def grab(self) -> np.ndarray:
        frames = self.pipe.wait_for_frames(timeout_ms=2000)
        return np.asanyarray(frames.get_color_frame().get_data())

    def stop(self) -> None:
        try: self.pipe.stop()
        except Exception: pass


class _V4L2Stream:
    def __init__(self, dev: str, w: int, h: int, fps: int) -> None:
        import cv2
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(5):
            self.cap.read()
        self._cv2 = cv2

    def grab(self) -> np.ndarray:
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("V4L2 grab failed")
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        try: self.cap.release()
        except Exception: pass


# ---------- /act helpers ----------

def _post(server_url: str, top, left, right, state, instruction,
          num_steps: int, timeout_s: float) -> float:
    """POST one /act request and return server dt_ms.
    Discards the actions -- we only care about timing.
    """
    payload = {
        "top_cam":     top,
        "left_cam":    left,
        "right_cam":   right,
        "instruction": instruction,
        "state":       state,
        "num_steps":   num_steps,
    }
    body = json_numpy.dumps(payload)
    t0 = time.perf_counter()
    resp = requests.post(server_url, data=body,
                         headers={"Content-Type": "application/json"},
                         timeout=timeout_s)
    resp.raise_for_status()
    out = json_numpy.loads(resp.text)
    rtt_ms = (time.perf_counter() - t0) * 1000.0
    server_ms = float(out.get("dt_ms", 0.0))
    return rtt_ms, server_ms


# ---------- experiments ----------

def run_sync(server_url: str, cams, instruction: str, num_steps: int,
             stride: int, n_cycles: int, train_fps: float = 30.0,
             timeout_s: float = 30.0) -> dict:
    """Mirror yam_client's loop: read state -> grab cams -> POST -> wait ->
    'execute' stride steps (sleep) -> loop."""
    top_c, left_c, right_c = cams
    state = np.zeros(14, dtype=np.float32)
    step_dt = 1.0 / train_fps
    stride_dur_s = stride * step_dt

    cycle_starts = []
    rtts = []
    server_dts = []
    cycle_times = []
    wall_t0 = time.perf_counter()
    for i in range(n_cycles):
        cycle_t0 = time.perf_counter()
        cycle_starts.append(cycle_t0 - wall_t0)
        # 1. read state + grab cams (negligible)
        top  = top_c.grab()
        left = left_c.grab()
        rght = right_c.grab()
        # 2. POST and BLOCK
        rtt_ms, server_ms = _post(server_url, top, left, rght, state,
                                  instruction, num_steps, timeout_s)
        rtts.append(rtt_ms)
        server_dts.append(server_ms)
        # 3. 'execute' stride steps -- in real client this drives motors at
        #    30 Hz; here we just sleep the equivalent wall time.
        time.sleep(stride_dur_s)
        cycle_times.append((time.perf_counter() - cycle_t0) * 1000.0)
    return {
        "mode": "sync", "stride": stride, "n_cycles": n_cycles,
        "rtts": rtts, "server_dts": server_dts, "cycle_times": cycle_times,
        "cycle_starts": cycle_starts,
        "stride_dur_s": stride_dur_s, "train_fps": train_fps,
    }


def run_async(server_url: str, cams, instruction: str, num_steps: int,
              stride: int, n_cycles: int, train_fps: float = 30.0,
              timeout_s: float = 30.0) -> dict:
    """Overlap inference with execution: while 'executing' chunk N, the
    POST for chunk N+1 is in flight on a background thread. If RTT <
    stride_dur_s, the next chunk is ready before execution finishes and
    there's no idle wait.
    """
    top_c, left_c, right_c = cams
    state = np.zeros(14, dtype=np.float32)
    step_dt = 1.0 / train_fps
    stride_dur_s = stride * step_dt

    cycle_starts = []
    rtts = []
    server_dts = []
    cycle_times = []
    inflight_wait_ms = []  # ms spent waiting on the in-flight POST after stride

    pending = {"thread": None, "result": None, "post_t0": 0.0}

    def _kick_off_post():
        top  = top_c.grab()
        left = left_c.grab()
        rght = right_c.grab()
        pending["post_t0"] = time.perf_counter()
        pending["result"] = None

        def _worker():
            try:
                rtt_ms, srv = _post(server_url, top, left, rght, state,
                                    instruction, num_steps, timeout_s)
                pending["result"] = (rtt_ms, srv)
            except Exception as e:
                pending["result"] = ("ERR", str(e))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        pending["thread"] = t

    # Prime: first chunk is fetched synchronously (no chunk yet to execute).
    _kick_off_post()
    pending["thread"].join()
    first = pending["result"]
    if isinstance(first[0], str):
        raise RuntimeError(f"first /act failed: {first[1]}")
    rtts.append(first[0]); server_dts.append(first[1])

    wall_t0 = time.perf_counter()
    for i in range(n_cycles):
        cycle_t0 = time.perf_counter()
        cycle_starts.append(cycle_t0 - wall_t0)
        # Kick off the NEXT POST in a background thread BEFORE we start
        # executing the current chunk. Real client would do the same.
        _kick_off_post()
        # 'Execute' the current chunk (sleep).
        time.sleep(stride_dur_s)
        # Wait for the in-flight POST to finish (if not already).
        wait_start = time.perf_counter()
        pending["thread"].join()
        wait_ms = (time.perf_counter() - wait_start) * 1000.0
        inflight_wait_ms.append(wait_ms)
        res = pending["result"]
        if isinstance(res[0], str):
            raise RuntimeError(f"/act failed mid-stream: {res[1]}")
        rtts.append(res[0]); server_dts.append(res[1])
        cycle_times.append((time.perf_counter() - cycle_t0) * 1000.0)
    return {
        "mode": "async", "stride": stride, "n_cycles": n_cycles,
        "rtts": rtts[1:], "server_dts": server_dts[1:],
        "cycle_times": cycle_times, "cycle_starts": cycle_starts,
        "inflight_wait_ms": inflight_wait_ms,
        "stride_dur_s": stride_dur_s, "train_fps": train_fps,
    }


# ---------- reporting ----------

def _summary(rs: dict) -> dict:
    rtts = rs["rtts"]; cycles = rs["cycle_times"]
    server = rs["server_dts"]
    # effective replan rate = N replans / total wall time
    total_wall_ms = sum(cycles)
    n = len(cycles)
    eff_hz = (n / (total_wall_ms / 1000.0)) if total_wall_ms > 0 else 0.0
    ideal_hz = rs["train_fps"] / rs["stride"]
    # idle fraction = (cycle_time - stride_dur_ms) / cycle_time, capped at 0
    stride_dur_ms = rs["stride_dur_s"] * 1000.0
    idle_fracs = [max(0.0, (c - stride_dur_ms) / c) for c in cycles]
    out = {
        "mode": rs["mode"], "stride": rs["stride"], "n": n,
        "rtt_mean_ms": statistics.mean(rtts),
        "rtt_p50_ms":  statistics.median(rtts),
        "rtt_p95_ms":  sorted(rtts)[int(0.95 * (len(rtts) - 1))] if rtts else 0.0,
        "server_mean_ms": statistics.mean(server),
        "cycle_mean_ms":  statistics.mean(cycles),
        "stride_dur_ms":  stride_dur_ms,
        "effective_replan_hz": eff_hz,
        "ideal_replan_hz":     ideal_hz,
        "ideal_vs_actual_pct": 100.0 * eff_hz / ideal_hz if ideal_hz > 0 else 0.0,
        "idle_frac_mean":  statistics.mean(idle_fracs),
    }
    if "inflight_wait_ms" in rs:
        out["inflight_wait_mean_ms"] = statistics.mean(rs["inflight_wait_ms"])
    return out


def _print_table(rows: list[dict]) -> None:
    print()
    print("| mode  | stride | n  | RTT mean | server mean | cycle mean | stride dur | effective Hz | ideal Hz | actual/ideal | idle frac |")
    print("|-------|--------|----|----------|-------------|------------|------------|--------------|----------|--------------|-----------|")
    for r in rows:
        print(f"| {r['mode']:<5s} | {r['stride']:6d} | {r['n']:2d} | "
              f"{r['rtt_mean_ms']:7.1f} ms | {r['server_mean_ms']:8.1f} ms | "
              f"{r['cycle_mean_ms']:7.1f} ms | {r['stride_dur_ms']:7.1f} ms | "
              f"{r['effective_replan_hz']:9.2f} Hz | {r['ideal_replan_hz']:5.2f} Hz | "
              f"{r['ideal_vs_actual_pct']:9.0f} % | {r['idle_frac_mean']:7.1%} |")
    print()


# ---------- main ----------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server-url", default="http://127.0.0.1:8202/act")
    p.add_argument("--strides", default="4,6,10,15,30",
                   help="comma-separated stride values to sweep")
    p.add_argument("--n-cycles", type=int, default=20,
                   help="cycles per (mode, stride) cell")
    p.add_argument("--num-steps", type=int, default=10,
                   help="flow-matching steps in /act request")
    p.add_argument("--cam-width",  type=int, default=640,
                   help="camera resolution (matches training: 640x360)")
    p.add_argument("--cam-height", type=int, default=360)
    p.add_argument("--cam-fps",    type=int, default=30)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--config", default=str(CONFIG_PATH),
                   help="path to yam_setup_config.json (for cam mapping)")
    p.add_argument("--skip-warmup", action="store_true",
                   help="skip the explicit warmup /act (don't skip on a fresh server)")
    p.add_argument("--mode", choices=["both", "sync", "async"], default="both")
    args = p.parse_args()

    # Load camera mapping.
    cfg = json.loads(Path(args.config).read_text())
    top_v4l2 = cfg.get("top_cam_v4l2")
    left_serial  = cfg.get("left_cam_serial")
    right_serial = cfg.get("right_cam_serial")
    assert top_v4l2 and left_serial and right_serial, \
        "config must have top_cam_v4l2, left_cam_serial, right_cam_serial"

    # Health-check the server.
    r = requests.get(args.server_url, timeout=5.0)
    r.raise_for_status()
    print(f"server: {r.json()}")
    print(f"camera: {args.cam_width}x{args.cam_height} @ {args.cam_fps} fps")
    print(f"strides: {args.strides}  n_cycles per cell: {args.n_cycles}")
    print()

    # Open cameras (read-only, no arms).
    print("opening cameras ...", flush=True)
    top_c   = _V4L2Stream(top_v4l2,    args.cam_width, args.cam_height, args.cam_fps)
    left_c  = _RSStream(left_serial,   args.cam_width, args.cam_height, args.cam_fps)
    right_c = _RSStream(right_serial,  args.cam_width, args.cam_height, args.cam_fps)
    cams = (top_c, left_c, right_c)

    try:
        # Warmup at this exact image shape so CUDA graph is captured before
        # we start measuring. First call at a new shape can take seconds.
        if not args.skip_warmup:
            print("warmup /act at "
                  f"{args.cam_height}x{args.cam_width} ...", flush=True)
            t0 = time.perf_counter()
            for k in range(3):
                _post(args.server_url, top_c.grab(), left_c.grab(), right_c.grab(),
                      np.zeros(14, dtype=np.float32), args.instruction,
                      args.num_steps, timeout_s=60.0)
            print(f"  ... warmup done ({(time.perf_counter()-t0)*1000:.0f} ms total)")

        strides = [int(s) for s in args.strides.split(",")]
        rows: list[dict] = []
        for stride in strides:
            if args.mode in ("sync", "both"):
                print(f"running sync stride={stride} ... ", end="", flush=True)
                rs = run_sync(args.server_url, cams, args.instruction,
                              args.num_steps, stride, args.n_cycles)
                s = _summary(rs); rows.append(s)
                print(f"effective={s['effective_replan_hz']:.2f} Hz")
            if args.mode in ("async", "both"):
                print(f"running async stride={stride} ... ", end="", flush=True)
                ra = run_async(args.server_url, cams, args.instruction,
                               args.num_steps, stride, args.n_cycles)
                s = _summary(ra); rows.append(s)
                print(f"effective={s['effective_replan_hz']:.2f} Hz")

        _print_table(rows)

        # Verdict.
        if args.mode == "both":
            print("Interpretation:")
            print("- If async/sync 'effective Hz' values are similar -> sync is NOT the")
            print("  bottleneck (the loop is already efficient, no refactor needed).")
            print("- If async is meaningfully faster (esp. at small strides) -> the")
            print("  current sync loop is bottlenecked on inference latency. Worth")
            print("  considering an async refactor in yam_client.py.")
            print("- 'idle frac' is the share of the cycle spent waiting on the")
            print("  server (sync) or on the in-flight POST (async). Lower is better.")
    finally:
        for c in cams:
            try: c.stop()
            except Exception: pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
