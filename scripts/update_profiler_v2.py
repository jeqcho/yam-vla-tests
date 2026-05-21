"""v2: drill into _update_joint_state to find the exact slow line.

v1 showed 66 slow update() events all attributed entirely to
_update_joint_state. v2 monkey-patches that method too, timing each of
its sub-sections separately:

  1. encoder check  (has_gripper_encoder branch)
  2. motor_chain.set_commands(...)  -- may block on dm_driver's command_lock
  3. _motor_state_to_joint_state
  4. _check_current_qpos_in_joint_limits  -- possible logging
  5. _joint_state_saver  (usually None)

Whenever the total _update_joint_state call takes >50ms, prints the
breakdown. Final summary shows per-section percentiles.

The most likely culprit: motor_chain.set_commands() acquires the
DMSingleMotorCanInterface.command_lock, the SAME lock that dm_driver's
Thread-1 holds during its entire CAN cycle (~3ms when clean, but can be
~150ms+ when a motor needs silent retries). Lock contention with that
high-frequency thread can starve our update() call.
"""
from __future__ import annotations

import argparse
import copy
import os
import signal
import sys
import time

import numpy as np


def install_deep_profiler(robot_module):
    """Monkey-patch _update_joint_state with per-section timing."""
    orig = robot_module.MotorChainRobot._update_joint_state

    stats = {
        "encoder_check": [],
        "set_commands": [],
        "motor_state_to_joint_state": [],
        "check_qpos_limits": [],
        "joint_state_saver": [],
        "total_uj": [],
    }
    slow_events: list[dict] = []

    def _uj_timed(self, motor_torques, joint_commands, encoder_infos=None):
        t0 = time.perf_counter()

        # Section 1: encoder check
        ta = time.perf_counter()
        if (
            hasattr(self.motor_chain, "get_same_bus_device_states")
            and callable(self.motor_chain.get_same_bus_device_states)
            and self.motor_chain.same_bus_device_driver is not None
        ):
            has_gripper_encoder = True
            encoder_infos = self.motor_chain.get_same_bus_device_states()
        else:
            has_gripper_encoder = False
            encoder_infos = encoder_infos
        tb = time.perf_counter()
        stats["encoder_check"].append(tb - ta)
        enc_dt = tb - ta

        # Section 2: motor_chain.set_commands (LIKELY CULPRIT)
        ta = time.perf_counter()
        motor_state = self.motor_chain.set_commands(
            motor_torques,
            pos=joint_commands.pos,
            vel=joint_commands.vel,
            kp=joint_commands.kp,
            kd=joint_commands.kd,
        )
        tb = time.perf_counter()
        stats["set_commands"].append(tb - ta)
        sc_dt = tb - ta

        # Section 3: motor_state_to_joint_state
        ta = time.perf_counter()
        self._joint_state = self._motor_state_to_joint_state(motor_state)
        tb = time.perf_counter()
        stats["motor_state_to_joint_state"].append(tb - ta)
        mtj_dt = tb - ta

        # Section 4: check qpos limits
        ta = time.perf_counter()
        self._check_current_qpos_in_joint_limits()
        tb = time.perf_counter()
        stats["check_qpos_limits"].append(tb - ta)
        chk_dt = tb - ta

        # Section 5: joint state saver
        ta = time.perf_counter()
        if self._joint_state_saver is not None:
            # Mirror the original logic.
            assert not (has_gripper_encoder and self._gripper_index is not None)
            ee_pos = ee_vel = ee_eff = None
            if has_gripper_encoder:
                ee_pos = np.array([info.position for info in encoder_infos])
                ee_vel = np.array([info.velocity for info in encoder_infos])
            elif self._gripper_index is not None:
                ee_pos = self._joint_state.pos[self._gripper_index]
                ee_vel = self._joint_state.vel[self._gripper_index]
                ee_eff = self._joint_state.eff[self._gripper_index]
            if self._gripper_index is None:
                pos = self._joint_state.pos
                vel = self._joint_state.vel
                eff = self._joint_state.eff
            else:
                pos = self._joint_state.pos[: self._gripper_index]
                vel = self._joint_state.vel[: self._gripper_index]
                eff = self._joint_state.eff[: self._gripper_index]
            self._joint_state_saver.add(
                timestamp=self._joint_state.timestamp,
                pos=pos, vel=vel, eff=eff,
                ee_pos=ee_pos, ee_vel=ee_vel,
            )
        tb = time.perf_counter()
        stats["joint_state_saver"].append(tb - ta)
        sav_dt = tb - ta

        total_dt = time.perf_counter() - t0
        stats["total_uj"].append(total_dt)
        if total_dt * 1000 > 50:
            ev = dict(
                total_ms=total_dt*1000,
                enc_ms=enc_dt*1000,
                set_cmd_ms=sc_dt*1000,
                m2j_ms=mtj_dt*1000,
                chk_ms=chk_dt*1000,
                sav_ms=sav_dt*1000,
            )
            slow_events.append(ev)
            print(f"[SLOW _uj] total={ev['total_ms']:7.1f}ms  "
                  f"enc={ev['enc_ms']:5.1f}  set_cmd={ev['set_cmd_ms']:7.1f}  "
                  f"m2j={ev['m2j_ms']:5.1f}  chk={ev['chk_ms']:5.1f}  "
                  f"sav={ev['sav_ms']:5.1f}", flush=True)

    robot_module.MotorChainRobot._update_joint_state = _uj_timed
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

    from i2rt.robots import motor_chain_robot as mcr
    stats, slow_events = install_deep_profiler(mcr)
    print(f"installed _update_joint_state profiler (threshold 50ms)", flush=True)

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

    print(f"\nDriving sinusoid {args.duration_s:.1f}s at {args.cmd_hz:.0f} Hz...\n",
          flush=True)
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
        next_tick += dt_target
        s = next_tick - time.perf_counter()
        if s > 0:
            time.sleep(s)

    # Summary.
    print(f"\n=== _update_joint_state per-section latency ===")
    print(f"  {'section':<30s}  {'count':>6s}  {'p50':>7s}  {'p90':>7s}  "
          f"{'p99':>7s}  {'max':>8s}  {'>50ms':>6s}  {'>500ms':>7s}")
    for key in ("total_uj", "encoder_check", "set_commands",
                "motor_state_to_joint_state", "check_qpos_limits",
                "joint_state_saver"):
        arr = np.asarray(stats[key]) * 1000.0
        if len(arr) == 0:
            continue
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, 90))
        p99 = float(np.percentile(arr, 99))
        mx = float(arr.max())
        n_slow = int((arr > 50).sum())
        n_super = int((arr > 500).sum())
        print(f"  {key:<30s}  {len(arr):>6d}  {p50:>7.2f}  {p90:>7.2f}  "
              f"{p99:>7.2f}  {mx:>8.2f}  {n_slow:>6d}  {n_super:>7d}")

    print(f"\n  total slow _uj events (>50ms): {len(slow_events)}")
    if slow_events:
        worst = sorted(slow_events, key=lambda e: -e["total_ms"])[:5]
        print(f"  worst 5 events:")
        for e in worst:
            print(f"    total={e['total_ms']:7.1f}  enc={e['enc_ms']:5.1f}  "
                  f"set_cmd={e['set_cmd_ms']:7.1f}  m2j={e['m2j_ms']:5.1f}  "
                  f"chk={e['chk_ms']:5.1f}  sav={e['sav_ms']:5.1f}")

    # Safe return.
    abort = {"abort": False, "n": 0}
    def _cleanup(_s, _f):
        abort["n"] += 1
        if abort["n"] == 1:
            print("[cleanup] Ctrl-C -- arms WILL drop", flush=True)
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
        c = cur + alpha * (startup - cur)
        robot.command_joint_pos(c)
        time.sleep(1.0/30.0)

    try: robot.close()
    except Exception: pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
