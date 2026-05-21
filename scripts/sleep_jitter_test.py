"""Separate OS-scheduling jitter from _state_lock contention.

Lag investigation so far:
  - get_joint_pos blocks up to 764ms occasionally
  - CAN bus is healthy (300Hz per motor, max 5ms global gap)
  - so the lag must be Python-level: either lock contention with the SDK
    control thread (which holds _state_lock during update()) or our thread
    being starved by the OS

This script isolates the OS-scheduling component. With the SDK control
thread running (one arm initialized in position-holding mode), the main
thread does ONLY time.sleep(target_dt) in a loop -- no get_joint_pos,
no command_joint_pos, no lock acquires. We measure the actual time each
sleep took.

If sleep durations show 100+ms outliers -> our thread is being starved
  by the OS / by the GIL while the SDK thread runs. This is not a lock
  problem and can't be fixed by changing how we acquire _state_lock.

If sleep durations are tight (close to target_dt) -> our thread IS being
  scheduled fine; the 764ms blocks in get_joint_pos must come from lock
  contention with the SDK control thread specifically.

Run alongside the always-on safety ramp (initialize arm, ramp to bias
pose first so the SDK is under representative load).

Usage:
    /home/andon/yam-tests/i2rt/.venv/bin/python \\
        /home/andon/yam-tests/molmoact2-setup/scripts/sleep_jitter_test.py \\
        --can can0 --gripper linear_4310 --duration-s 8
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType


def init_arm(channel: str, gripper: str):
    return get_yam_robot(
        channel=channel,
        arm_type=ArmType.from_string_name("yam"),
        gripper_type=GripperType.from_string_name(gripper),
        zero_gravity_mode=False,
    )


def ramp_to(robot, goal_7d: np.ndarray, duration_s: float = 4.0, hz: float = 30.0):
    start = np.asarray(robot.get_joint_pos(), dtype=np.float32)
    goal = np.asarray(goal_7d, dtype=np.float32)
    n = max(1, int(duration_s * hz))
    dt = 1.0 / hz
    for i in range(1, n + 1):
        alpha = i / n
        cmd = start + alpha * (goal - start)
        robot.command_joint_pos(cmd)
        time.sleep(dt)
    time.sleep(0.3)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--can", default="can0")
    p.add_argument("--gripper", default="linear_4310")
    p.add_argument("--duration-s", type=float, default=8.0)
    p.add_argument("--sleep-ms", type=float, default=20.0,
                   help="target time.sleep duration per iteration (ms)")
    p.add_argument("--bias-shoulder", type=float, default=1.0,
                   help="ramp shoulder here so SDK is under representative load")
    args = p.parse_args()

    print(f"Init arm on {args.can}...", flush=True)
    robot = init_arm(args.can, args.gripper)
    time.sleep(0.5)

    startup = np.asarray(robot.get_joint_pos(), dtype=np.float32)
    print(f"Startup pose = {np.array2string(startup, precision=3)}", flush=True)

    # Ramp to a loaded pose so the SDK has work to do (gravity comp etc).
    test_pose = startup.copy()
    test_pose[1] = args.bias_shoulder
    print("Ramping to test pose...", flush=True)
    ramp_to(robot, test_pose, duration_s=4.0)

    target = args.sleep_ms / 1000.0
    n = max(10, int(args.duration_s / target))

    print(f"\nPure-sleep test: {n} iterations of time.sleep({target*1000:.0f}ms)", flush=True)
    print(f"NO get_joint_pos, NO command_joint_pos -- only time.sleep.", flush=True)
    print(f"SDK control thread is running concurrently and updating motors.\n", flush=True)

    durations = []
    overruns = 0
    t0 = time.perf_counter()
    for i in range(n):
        t_a = time.perf_counter()
        time.sleep(target)
        t_b = time.perf_counter()
        dur = t_b - t_a
        durations.append(dur)
        if dur > target * 1.5:
            overruns += 1
        if i % max(1, n // 20) == 0:
            print(f"  iter {i:>4d}: target={target*1000:.0f}ms  actual={dur*1000:6.2f}ms",
                  flush=True)

    arr = np.asarray(durations) * 1000.0  # ms
    print(f"\n=== sleep_jitter_test summary ===")
    print(f"  target: {target*1000:.1f} ms, iterations: {n}, total {time.perf_counter()-t0:.2f}s")
    print(f"  duration stats (ms):")
    print(f"    p50 ={np.percentile(arr, 50):7.2f}")
    print(f"    p90 ={np.percentile(arr, 90):7.2f}")
    print(f"    p99 ={np.percentile(arr, 99):7.2f}")
    print(f"    max ={arr.max():7.2f}")
    print(f"    samples >{target*1000*1.5:.0f}ms: {(arr > target*1000*1.5).sum()}/{n}")
    print(f"    samples >100ms: {(arr > 100.0).sum()}/{n}")
    print(f"    samples >200ms: {(arr > 200.0).sum()}/{n}")
    print()
    print("Interpretation:")
    if arr.max() < 50.0:
        print("  -> SLEEPS ARE TIGHT. The OS is scheduling our thread fine.")
        print("     The 764ms blocks in get_joint_pos must come from LOCK CONTENTION")
        print("     with the SDK control thread holding _state_lock.")
    elif arr.max() < 150.0:
        print("  -> SLEEPS ARE MILDLY JITTERY (50-150ms outliers). Some OS scheduling")
        print("     pressure but not enough to fully account for the 764ms blocks.")
        print("     Likely BOTH OS jitter AND lock contention contributing.")
    else:
        print("  -> SLEEPS HAVE BIG OUTLIERS (>150ms). Our thread is being STARVED")
        print("     by the OS / by the GIL during SDK control-thread activity.")
        print("     The get_joint_pos lag isn't (primarily) a lock problem.")

    # Always-on safety: ramp back to startup.
    abort = {"abort": False, "n": 0}
    def _cleanup_sigint(_sig, _frame):
        abort["n"] += 1
        if abort["n"] == 1:
            print("[cleanup] Ctrl-C -- aborting return ramp. ARMS WILL DROP.", flush=True)
            abort["abort"] = True
        else:
            os._exit(130)
    signal.signal(signal.SIGINT, _cleanup_sigint)

    print("\nReturning to startup pose before disable...", flush=True)
    try:
        ramp_to(robot, startup, duration_s=5.0)
    except BaseException as e:
        print(f"WARNING return ramp failed: {e}. Arms may drop.", flush=True)
    try: robot.close()
    except Exception: pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
