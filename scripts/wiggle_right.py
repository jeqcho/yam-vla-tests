"""Quick smoke test: wiggle just the right arm (can1).

Sweeps wrist-roll (joint 6) ±0.3 rad for 3 cycles at 0.7 Hz. Wrist roll
is the safest joint to wiggle -- it just spins the end-effector in place,
no swept volume change.

Auto-applies the SDK lock fix by importing yam_client, then ramps back to
the startup pose on exit so the arm doesn't drop.

Usage:
    /home/andon/yam-tests/i2rt/.venv/bin/python \\
        /home/andon/yam-tests/molmoact2-setup/scripts/wiggle_right.py
"""
from __future__ import annotations

import os
import signal
import sys
import time

import numpy as np

# Importing yam_client triggers install_sdk_lock_fix() at module load time.
sys.path.insert(0, "/home/andon/yam-tests/molmoact2-setup/scripts")
import yam_client  # noqa: F401  -- side-effect import

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType


WRIST_ROLL = 5         # joint index (j6 in 1-indexed)
AMPLITUDE  = 0.3       # rad, ~17 degrees of spin
FREQ_HZ    = 0.7       # full sinusoid cycle frequency
DURATION_S = 4.3       # ~3 cycles at 0.7 Hz
CMD_HZ     = 30.0


def main() -> int:
    print("Initializing right arm on can1 (position-holding)...", flush=True)
    robot = get_yam_robot(
        channel="can1",
        arm_type=ArmType.from_string_name("yam"),
        gripper_type=GripperType.from_string_name("linear_4310"),
        zero_gravity_mode=False,
    )
    time.sleep(0.5)

    startup = np.asarray(robot.get_joint_pos(), dtype=np.float32)
    print(f"Startup pose: {np.array2string(startup, precision=3)}", flush=True)
    print(f"Wiggling joint {WRIST_ROLL + 1} (wrist roll) by +-{AMPLITUDE:.2f} rad "
          f"@ {FREQ_HZ:.2f} Hz for {DURATION_S:.1f}s...", flush=True)

    omega = 2.0 * np.pi * FREQ_HZ
    dt = 1.0 / CMD_HZ
    n_steps = int(DURATION_S * CMD_HZ)
    cmd = startup.copy()
    t0 = time.perf_counter()
    next_tick = t0
    for i in range(n_steps):
        t = time.perf_counter() - t0
        cmd[WRIST_ROLL] = startup[WRIST_ROLL] + AMPLITUDE * np.sin(omega * t)
        robot.command_joint_pos(cmd.astype(np.float32))
        next_tick += dt
        s = next_tick - time.perf_counter()
        if s > 0:
            time.sleep(s)

    # SAFETY: ramp back to startup before close.
    abort = {"abort": False, "n": 0}
    def _cleanup(_s, _f):
        abort["n"] += 1
        if abort["n"] == 1:
            print("[cleanup] Ctrl-C -- aborting return ramp. ARM WILL DROP.", flush=True)
            abort["abort"] = True
        else:
            os._exit(130)
    signal.signal(signal.SIGINT, _cleanup)

    print("Returning to startup pose (2s ramp)...", flush=True)
    current = np.asarray(robot.get_joint_pos(), dtype=np.float32)
    n_ramp = int(2.0 * CMD_HZ)
    for i in range(1, n_ramp + 1):
        if abort["abort"]:
            break
        alpha = i / n_ramp
        c = current + alpha * (startup - current)
        robot.command_joint_pos(c.astype(np.float32))
        time.sleep(dt)
    time.sleep(0.5)

    try: robot.close()
    except Exception: pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
