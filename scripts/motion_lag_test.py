"""Quantify the "arm moves in bursts" phenomenon.

We drive both arms with a smooth sinusoidal open-loop command on one joint
each (shoulder pitch j2), sample commanded vs actual at high rate, and
emit:

  - lag_test.csv   per-tick: t, cmd_left, q_left, cmd_right, q_right
  - lag_test.txt   summary: per-arm tracking lag, max stall duration,
                   jerk RMS, % of samples where state didn't change

If the motion is smooth, command-actual lag should be a small constant
(~ kp/kd settling time, ~50 ms), and "stall %" should be near zero. If
the SDK control thread freezes for chunks, stall % climbs and you'll see
flat plateaus in q where cmd keeps moving.

Usage:
    /home/andon/yam-tests/i2rt/.venv/bin/python \\
        /home/andon/yam-tests/molmoact2-setup/scripts/motion_lag_test.py \\
        --left-can can0 --right-can can1 \\
        --left-gripper linear_4310 --right-gripper linear_4310 \\
        --duration-s 8

ALWAYS-ON SAFETY (matches yam_client.py):
  - Capture startup_pose at init.
  - Before exit, ramp both arms back to startup_pose.
  - Then close(). Arms will NOT drop.

Joint 2 (shoulder pitch) is what we sweep because that's the joint with
the largest gravity moment and where you noticed the burstiness most.
Amplitude is small (0.3 rad ~ 17 deg) and slow (0.5 Hz). Both arms move
in sync so any cross-arm asymmetry shows up as a phase difference.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from typing import Any

import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType


SHOULDER_J = 1   # index into the 7-D arm state vector (joints 1..6 + gripper)
DEFAULT_AMPLITUDE = 0.3   # rad
DEFAULT_FREQ = 0.5        # Hz, full sinusoid cycle


def init_arm(channel: str, gripper: str):
    return get_yam_robot(
        channel=channel,
        arm_type=ArmType.from_string_name("yam"),
        gripper_type=GripperType.from_string_name(gripper),
        zero_gravity_mode=False,
    )


def ramp_to(left, right, target_14d: np.ndarray, duration_s: float = 4.0, hz: float = 30.0,
            abort_flag: dict | None = None, label: str = "ramp"):
    """Linear ramp current -> target."""
    q_l = np.asarray(left.get_joint_pos(),  dtype=np.float32)
    q_r = np.asarray(right.get_joint_pos(), dtype=np.float32)
    start = np.concatenate([q_l, q_r])
    goal = np.asarray(target_14d, dtype=np.float32).copy()
    print(f"[{label}] start = {np.array2string(start, precision=3)}", flush=True)
    print(f"[{label}] goal  = {np.array2string(goal,  precision=3)}", flush=True)
    n = max(1, int(duration_s * hz))
    dt = 1.0 / hz
    for i in range(1, n + 1):
        if abort_flag and abort_flag.get("abort"):
            print(f"[{label}] aborted at {i}/{n}", flush=True)
            return
        alpha = i / n
        cmd = start + alpha * (goal - start)
        left.command_joint_pos(cmd[:7].astype(np.float32))
        right.command_joint_pos(cmd[7:].astype(np.float32))
        time.sleep(dt)
    time.sleep(0.5)
    print(f"[{label}] done", flush=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stderr)

    p = argparse.ArgumentParser()
    p.add_argument("--left-can",  default="can0")
    p.add_argument("--right-can", default="can1")
    p.add_argument("--left-gripper",  default="linear_4310")
    p.add_argument("--right-gripper", default="linear_4310")
    p.add_argument("--duration-s", type=float, default=8.0,
                   help="how long to drive the sinusoid (s)")
    p.add_argument("--cmd-hz", type=float, default=50.0,
                   help="rate at which to send position commands AND sample state")
    p.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE,
                   help="sinusoid amplitude on shoulder pitch (rad)")
    p.add_argument("--freq", type=float, default=DEFAULT_FREQ,
                   help="sinusoid frequency (Hz)")
    p.add_argument("--bias-shoulder", type=float, default=1.0,
                   help="center the sinusoid around this shoulder angle (rad). "
                        "1.0 puts the arm in a near-training-mean pose so the "
                        "shoulder has decent gravity load when sweeping.")
    p.add_argument("--out", default="lag_test.csv",
                   help="CSV output path (also writes .txt summary alongside)")
    args = p.parse_args()

    print(f"\nInitializing arms on {args.left_can} / {args.right_can}...", flush=True)
    left  = init_arm(args.left_can,  args.left_gripper)
    right = init_arm(args.right_can, args.right_gripper)
    time.sleep(0.5)

    # Startup pose for safe return.
    q_l = np.asarray(left.get_joint_pos(),  dtype=np.float32)
    q_r = np.asarray(right.get_joint_pos(), dtype=np.float32)
    startup_pose = np.concatenate([q_l, q_r])
    print(f"Startup pose (for safe exit) = {np.array2string(startup_pose, precision=3)}", flush=True)

    # Build "test pose" = startup but with shoulder lifted to args.bias_shoulder.
    # That puts joint 2 at a meaningful angle where gravity actually loads it,
    # so any "burstiness" caused by the PD loop dropping out is visible.
    test_pose = startup_pose.copy()
    test_pose[SHOULDER_J]     = args.bias_shoulder  # left shoulder
    test_pose[7 + SHOULDER_J] = args.bias_shoulder  # right shoulder
    # Move to test pose slowly so we don't already start in burst territory.
    print("Ramping to test pose (shoulder bias)...", flush=True)
    ramp_to(left, right, test_pose, duration_s=4.0, hz=30.0, label="ramp-to-test")

    # SIGINT during the data-collection loop should abort cleanly so we can
    # still ramp back to startup.
    stop = {"abort": False}
    def _sigint(_sig, _frame):
        stop["abort"] = True
        print("[SIGINT] stopping data collection at next sample", flush=True)
    signal.signal(signal.SIGINT, _sigint)

    n_samples = int(args.duration_s * args.cmd_hz)
    dt = 1.0 / args.cmd_hz
    omega = 2.0 * np.pi * args.freq
    rows: list[list[float]] = []
    print(f"\nDriving sinusoid: amplitude={args.amplitude:.2f} rad, freq={args.freq:.2f} Hz, "
          f"bias={args.bias_shoulder:.2f} rad, {n_samples} samples @ {args.cmd_hz:.0f} Hz", flush=True)
    print("  t_s   cmd_L_j2   q_L_j2   cmd_R_j2   q_R_j2", flush=True)

    cmd_l = test_pose[:7].copy()
    cmd_r = test_pose[7:].copy()
    t0 = time.perf_counter()
    next_tick = t0
    for i in range(n_samples):
        if stop["abort"]:
            print(f"aborted at sample {i}", flush=True)
            break
        # Phase of the sinusoid.
        t = time.perf_counter() - t0
        offset = args.amplitude * np.sin(omega * t)
        cmd_l[SHOULDER_J] = args.bias_shoulder + offset
        cmd_r[SHOULDER_J] = args.bias_shoulder + offset
        # Send commands.
        send_t = time.perf_counter()
        left.command_joint_pos(cmd_l.astype(np.float32))
        right.command_joint_pos(cmd_r.astype(np.float32))
        # Immediately sample state.
        q_l = np.asarray(left.get_joint_pos(),  dtype=np.float32)
        q_r = np.asarray(right.get_joint_pos(), dtype=np.float32)
        sample_t = time.perf_counter()
        rows.append([
            float(t),
            float(cmd_l[SHOULDER_J]), float(q_l[SHOULDER_J]),
            float(cmd_r[SHOULDER_J]), float(q_r[SHOULDER_J]),
            float(sample_t - send_t),     # latency: command send -> state read
        ])
        # Print every ~0.5s so we see something live.
        if i % max(1, int(args.cmd_hz / 2)) == 0:
            print(f"  {t:5.2f}  {cmd_l[SHOULDER_J]:+.3f}  {q_l[SHOULDER_J]:+.3f}  "
                  f"{cmd_r[SHOULDER_J]:+.3f}  {q_r[SHOULDER_J]:+.3f}", flush=True)
        # Pace to cmd_hz.
        next_tick += dt
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)

    # Write CSV.
    out_path = os.path.abspath(args.out)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "cmd_L_j2", "q_L_j2", "cmd_R_j2", "q_R_j2", "cmd_to_read_s"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} samples to {out_path}", flush=True)

    # ---------------- summary stats ----------------
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape[0] > 5:
        t  = arr[:, 0]
        c_l, q_l_ = arr[:, 1], arr[:, 2]
        c_r, q_r_ = arr[:, 3], arr[:, 4]
        lat = arr[:, 5]

        def analyze(name: str, c: np.ndarray, q: np.ndarray) -> dict[str, Any]:
            # Stall detection: q didn't change since previous sample, but cmd did.
            dq = np.diff(q)
            dc = np.diff(c)
            cmd_changing = np.abs(dc) > 1e-4
            q_flat = np.abs(dq) < 1e-4
            stalls = cmd_changing & q_flat
            # Find longest contiguous stall.
            longest = 0
            cur = 0
            for s in stalls:
                if s:
                    cur += 1
                    longest = max(longest, cur)
                else:
                    cur = 0
            stall_pct = float(np.mean(stalls)) * 100.0 if len(stalls) else 0.0
            longest_ms = longest * (1000.0 / args.cmd_hz)
            # Tracking lag via cross-correlation (peak-to-peak).
            c_mid = c - c.mean()
            q_mid = q - q.mean()
            # Search lag up to 0.5 s.
            max_lag = int(0.5 * args.cmd_hz)
            best, best_lag = -np.inf, 0
            for lag in range(0, max_lag):
                if lag >= len(c):
                    break
                xc = np.dot(c_mid[: len(c) - lag], q_mid[lag:])
                if xc > best:
                    best, best_lag = xc, lag
            lag_ms = best_lag * (1000.0 / args.cmd_hz)
            # Jerk (RMS of 3rd diff of q).
            dt_s = 1.0 / args.cmd_hz
            if len(q) > 4:
                d3 = np.diff(q, n=3) / (dt_s ** 3)
                jerk_rms = float(np.sqrt(np.mean(d3 * d3)))
            else:
                jerk_rms = float("nan")
            return {
                "stall_pct": stall_pct,
                "longest_stall_ms": longest_ms,
                "lag_ms": lag_ms,
                "jerk_rms": jerk_rms,
            }

        L = analyze("LEFT",  c_l, q_l_)
        R = analyze("RIGHT", c_r, q_r_)
        lat_p50 = float(np.percentile(lat, 50)) * 1000
        lat_p99 = float(np.percentile(lat, 99)) * 1000
        lat_max = float(lat.max()) * 1000

        summary = (
            f"\n=== motion_lag_test summary ===\n"
            f"samples: {len(rows)}  duration: {t[-1]:.2f}s  target rate: {args.cmd_hz:.0f}Hz\n"
            f"sinusoid: amp={args.amplitude:.2f} rad  freq={args.freq:.2f} Hz  "
            f"bias={args.bias_shoulder:.2f} rad on joint {SHOULDER_J + 1}\n"
            f"\n"
            f"per-arm tracking metrics (smaller = better):\n"
            f"  arm    stall%   longest_stall   tracking_lag   jerk_rms\n"
            f"  LEFT   {L['stall_pct']:5.1f}%  {L['longest_stall_ms']:8.1f} ms  "
            f"{L['lag_ms']:8.1f} ms  {L['jerk_rms']:.2e}\n"
            f"  RIGHT  {R['stall_pct']:5.1f}%  {R['longest_stall_ms']:8.1f} ms  "
            f"{R['lag_ms']:8.1f} ms  {R['jerk_rms']:.2e}\n"
            f"\n"
            f"command->read latency (single-thread):\n"
            f"  p50={lat_p50:.2f} ms  p99={lat_p99:.2f} ms  max={lat_max:.2f} ms\n"
            f"\n"
            f"interpretation:\n"
            f"  - stall%% > 5 or longest_stall > 100ms = PD loop visibly freezing\n"
            f"  - lag should be ~ kp/kd settling, typically 20-80 ms\n"
            f"  - large asymmetry between LEFT and RIGHT = per-CAN-bus issue\n"
            f"  - large lat_max (>50ms) = main thread itself is being stalled\n"
        )
        print(summary)
        with open(out_path.replace(".csv", ".txt"), "w") as f:
            f.write(summary)
        print(f"summary also written to {out_path.replace('.csv', '.txt')}", flush=True)

    # SAFETY: ramp back to startup before close.
    abort = {"abort": False, "n": 0}
    def _cleanup_sigint(_sig, _frame):
        abort["n"] += 1
        if abort["n"] == 1:
            print("[cleanup] Ctrl-C: aborting return ramp. ARMS WILL DROP.", flush=True)
            abort["abort"] = True
        else:
            os._exit(130)
    signal.signal(signal.SIGINT, _cleanup_sigint)

    print("Returning arms to startup pose (5s ramp) before disable...", flush=True)
    try:
        ramp_to(left, right, startup_pose, duration_s=5.0, hz=30.0,
                abort_flag=abort, label="return-on-exit")
    except BaseException as e:
        print(f"WARNING: return ramp failed: {e} -- arms may drop on close.", flush=True)

    try: left.close()
    except Exception: pass
    try: right.close()
    except Exception: pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
