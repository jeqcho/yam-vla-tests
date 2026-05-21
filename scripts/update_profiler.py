"""Profile motor_chain_robot.update() to find which step occasionally hangs.

CONFIRMED so far:
  - main thread fine
  - CAN traffic fine
  - motor_chain_robot._server_thread occasionally takes hundreds of ms to
    multi-second stretches per update() call, holding _state_lock and
    preventing new commands from reaching dm_driver

WHAT IS UPDATE() DOING DURING THOSE HANGS?

Candidates inside `with self._state_lock:`:
  (a) self._compute_gravity_compensation(self._joint_state)  -- MuJoCo
  (b) gripper-force-limiter computation block
  (c) self._update_joint_state(motor_torques, joint_commands)
        - encoder info collection
        - motor_chain.set_commands (lock + read_states, should be fast)
        - self._motor_state_to_joint_state
        - self._check_current_qpos_in_joint_limits
        - optional self._joint_state_saver

This script monkey-patches motor_chain_robot.MotorChainRobot.update() to
time each section. Whenever a section takes >100ms it logs the breakdown,
along with the running max for each section. After the run it prints a
per-section histogram.

Run with the lag-test driving the arm. The profiler captures from BOTH the
SDK's _server_thread (the one doing the hangs) and from any other code
that triggers update().

Usage:
    /home/andon/yam-tests/i2rt/.venv/bin/python \\
        /home/andon/yam-tests/molmoact2-setup/scripts/update_profiler.py \\
        --can can0 --gripper linear_4310 --duration-s 12
"""
from __future__ import annotations

import argparse
import copy
import os
import signal
import sys
import time
import threading

import numpy as np


def install_profiler(robot_module):
    """Monkey-patch MotorChainRobot.update() with per-section timing."""
    orig_update = robot_module.MotorChainRobot.update

    # Per-section accumulators.
    stats = {
        "lock_command": [],
        "lock_state_acquire": [],
        "grav_comp": [],
        "torque_calc": [],
        "gripper_limiter": [],
        "update_joint_state": [],
        "total": [],
    }
    slow_events: list[dict] = []
    SLOW_THRESHOLD_MS = 50.0

    def update_timed(self):
        t0 = time.perf_counter()

        # Section: acquire _command_lock + deepcopy
        ta = time.perf_counter()
        with self._command_lock:
            joint_commands = copy.deepcopy(self._commands)
        tb = time.perf_counter()
        stats["lock_command"].append(tb - ta)

        # Section: acquire _state_lock
        ta = time.perf_counter()
        self._state_lock.acquire()
        tb = time.perf_counter()
        stats["lock_state_acquire"].append(tb - ta)

        try:
            # Section: grav comp
            ta = time.perf_counter()
            g = self._compute_gravity_compensation(self._joint_state)
            tb = time.perf_counter()
            stats["grav_comp"].append(tb - ta)
            grav_dt = tb - ta

            # Section: torque calc + clip + gripper limiter
            ta = time.perf_counter()
            friction_comp = self._coulomb_friction * np.sign(self._joint_state.vel)
            motor_torques = joint_commands.torques + g * self.gravity_comp_factor + friction_comp
            motor_torques = np.clip(motor_torques, -self._clip_motor_torque, self._clip_motor_torque)
            self._last_motor_torques = motor_torques.copy()
            tb = time.perf_counter()
            stats["torque_calc"].append(tb - ta)
            tcalc_dt = tb - ta

            # Section: gripper force limiter
            ta = time.perf_counter()
            if self._gripper_index is not None:
                if self._limit_gripper_force > 0 and self._joint_state is not None:
                    gripper_state = {
                        "target_qpos": joint_commands.pos[self._gripper_index],
                        "current_qpos": self.remapper.to_robot_joint_pos_space(
                            self._joint_state.pos)[self._gripper_index],
                        "current_qvel": self._joint_state.vel[self._gripper_index],
                        "current_eff": self._joint_state.eff[self._gripper_index],
                        "current_normalized_qpos": self._joint_state.pos[self._gripper_index],
                        "target_normalized_qpos": self.remapper.to_command_joint_pos_space(
                            joint_commands.pos)[self._gripper_index],
                        "last_command_qpos": self._last_gripper_command_qpos,
                    }
                    joint_commands.pos[self._gripper_index] = self._gripper_force_limiter.update(gripper_state)
                joint_commands.pos[self._gripper_index] = np.clip(
                    joint_commands.pos[self._gripper_index],
                    min(self._gripper_limits),
                    max(self._gripper_limits),
                )
                self._last_gripper_command_qpos = joint_commands.pos[self._gripper_index]
            tb = time.perf_counter()
            stats["gripper_limiter"].append(tb - ta)
            grip_dt = tb - ta

            # Section: _update_joint_state (motor I/O setup + state read + limit check)
            ta = time.perf_counter()
            self._update_joint_state(motor_torques, joint_commands)
            tb = time.perf_counter()
            stats["update_joint_state"].append(tb - ta)
            uj_dt = tb - ta
        finally:
            self._state_lock.release()

        total_dt = time.perf_counter() - t0
        stats["total"].append(total_dt)

        # If total took longer than threshold, log the breakdown.
        if total_dt * 1000 > SLOW_THRESHOLD_MS:
            ev = {
                "t": time.perf_counter(),
                "total_ms": total_dt * 1000,
                "grav_comp_ms": grav_dt * 1000,
                "torque_calc_ms": tcalc_dt * 1000,
                "gripper_limiter_ms": grip_dt * 1000,
                "update_joint_state_ms": uj_dt * 1000,
            }
            slow_events.append(ev)
            print(f"[SLOW update()] total={ev['total_ms']:7.1f}ms  "
                  f"grav={ev['grav_comp_ms']:6.1f}  "
                  f"tcalc={ev['torque_calc_ms']:5.1f}  "
                  f"grip={ev['gripper_limiter_ms']:5.1f}  "
                  f"uj={ev['update_joint_state_ms']:7.1f}", flush=True)

    robot_module.MotorChainRobot.update = update_timed
    return stats, slow_events


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--can", default="can0")
    p.add_argument("--gripper", default="linear_4310")
    p.add_argument("--duration-s", type=float, default=12.0)
    p.add_argument("--cmd-hz", type=float, default=50.0)
    p.add_argument("--amplitude", type=float, default=0.3)
    p.add_argument("--freq", type=float, default=0.5)
    p.add_argument("--bias-shoulder", type=float, default=1.0)
    args = p.parse_args()

    # Patch BEFORE we instantiate the robot.
    from i2rt.robots import motor_chain_robot as mcr
    stats, slow_events = install_profiler(mcr)
    print(f"installed update() profiler (threshold {50}ms)", flush=True)

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType

    print(f"Init arm on {args.can}...", flush=True)
    robot = get_yam_robot(
        channel=args.can,
        arm_type=ArmType.from_string_name("yam"),
        gripper_type=GripperType.from_string_name(args.gripper),
        zero_gravity_mode=False,
    )
    time.sleep(0.5)
    startup = np.asarray(robot.get_joint_pos(), dtype=np.float32)
    print(f"Startup = {np.array2string(startup, precision=3)}", flush=True)

    # Ramp to test pose.
    test_pose = startup.copy()
    test_pose[1] = args.bias_shoulder
    print("Ramping to test pose...", flush=True)
    n_ramp = 120
    for i in range(1, n_ramp + 1):
        alpha = i / n_ramp
        cmd = startup + alpha * (test_pose - startup)
        robot.command_joint_pos(cmd)
        time.sleep(1.0 / 30.0)
    time.sleep(0.3)

    # Drive sinusoid for duration_s while profiler logs slow updates.
    print(f"\nDriving sinusoid for {args.duration_s:.1f}s at {args.cmd_hz:.0f} Hz...\n", flush=True)
    omega = 2.0 * np.pi * args.freq
    n_samples = int(args.duration_s * args.cmd_hz)
    dt_target = 1.0 / args.cmd_hz
    cmd = test_pose.copy()
    t0 = time.perf_counter()
    next_tick = t0
    for i in range(n_samples):
        t = time.perf_counter() - t0
        cmd[1] = args.bias_shoulder + args.amplitude * np.sin(omega * t)
        robot.command_joint_pos(cmd.astype(np.float32))
        # Don't read state here -- we want our main thread quiet so profiler
        # captures the SDK control thread cleanly.
        next_tick += dt_target
        s = next_tick - time.perf_counter()
        if s > 0:
            time.sleep(s)

    # Summary.
    print(f"\n=== update() per-section latency over the run ===")
    print(f"  {'section':<22s}  {'count':>6s}  {'p50':>7s}  {'p90':>7s}  "
          f"{'p99':>7s}  {'max':>8s}  {'>50ms':>6s}  {'>500ms':>7s}")
    for key in ("total", "lock_command", "lock_state_acquire", "grav_comp",
                "torque_calc", "gripper_limiter", "update_joint_state"):
        arr = np.asarray(stats[key]) * 1000.0
        if len(arr) == 0:
            continue
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, 90))
        p99 = float(np.percentile(arr, 99))
        mx = float(arr.max())
        n_slow = int((arr > 50).sum())
        n_super = int((arr > 500).sum())
        print(f"  {key:<22s}  {len(arr):>6d}  {p50:>7.2f}  {p90:>7.2f}  "
              f"{p99:>7.2f}  {mx:>8.2f}  {n_slow:>6d}  {n_super:>7d}")

    print(f"\n  total slow update() events (>50ms): {len(slow_events)}")
    if slow_events:
        # Sort by total_ms descending and print top 5.
        worst = sorted(slow_events, key=lambda e: -e["total_ms"])[:5]
        print(f"  worst 5 events:")
        for e in worst:
            print(f"    total={e['total_ms']:8.1f}ms  grav={e['grav_comp_ms']:7.1f}  "
                  f"tcalc={e['torque_calc_ms']:6.1f}  grip={e['gripper_limiter_ms']:6.1f}  "
                  f"uj={e['update_joint_state_ms']:8.1f}")

    # SAFETY ramp back.
    abort = {"abort": False, "n": 0}
    def _cleanup(_s, _f):
        abort["n"] += 1
        if abort["n"] == 1:
            print("[cleanup] Ctrl-C -- aborting return. ARMS WILL DROP.", flush=True)
            abort["abort"] = True
        else:
            os._exit(130)
    signal.signal(signal.SIGINT, _cleanup)

    print("\nReturning to startup...", flush=True)
    cur = np.asarray(robot.get_joint_pos(), dtype=np.float32)
    for i in range(1, 151):
        if abort["abort"]:
            break
        alpha = i / 150
        cmd = cur + alpha * (startup - cur)
        robot.command_joint_pos(cmd)
        time.sleep(1.0 / 30.0)

    try: robot.close()
    except Exception: pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
