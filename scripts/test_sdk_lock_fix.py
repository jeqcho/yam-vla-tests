"""Test whether holding command_lock only during a brief snapshot (instead of
during all of CAN I/O) eliminates the mutex-starvation slow events.

Replaces dm_driver.DMChainCanInterface._set_torques_and_update_state with a
modified version that, inside the command_lock block, ONLY copies the
current commands to a local variable. CAN I/O and error handling are moved
OUTSIDE the lock. This drops command_lock hold time from ~3 ms (a full
CAN round-trip across 7 motors) to a few microseconds (a list copy).

If the lag is genuinely caused by mutex starvation between Thread-1 and
motor_chain_robot._server_thread, this patch should make the slow events
disappear. If they persist, the SDK architecture isn't the (only) cause
and we need to look elsewhere.

Same instrumentation as lock_acquire_profiler.py so we can directly
compare numbers.

Usage:
    /home/andon/yam-tests/i2rt/.venv/bin/python \\
        /home/andon/yam-tests/molmoact2-setup/scripts/test_sdk_lock_fix.py \\
        --can can0 --gripper linear_4310 --duration-s 12
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

import numpy as np


def install_loop_patch(dm_module):
    """Replace _set_torques_and_update_state with a version that releases
    command_lock before doing CAN I/O.

    The original (paraphrased):
        with self.command_lock:
            motor_feedback = self._set_commands(self.commands)  # CAN I/O
            ... error handling ...
        with self.state_lock:
            self.state = motor_feedback
            ...

    The patched version:
        with self.command_lock:
            local_commands = list(self.commands)   # microsecond snapshot
        # lock released here
        motor_feedback = self._set_commands(local_commands)   # CAN OUTSIDE lock
        ... error handling (also outside lock) ...
        with self.state_lock:
            self.state = motor_feedback
            ...
    """
    Chain = dm_module.DMChainCanInterface
    EXPECTED_CONTROL_PERIOD = dm_module.EXPECTED_CONTROL_PERIOD

    def patched(self) -> None:
        last_step_time = time.time()
        step_time_exceed_count = 0
        step_time_sum = 0.0
        step_time_count = 0
        report_start_time = time.time()
        with self._rate_recorder:
            while self.running:
                try:
                    curr_time = time.time()
                    step_time = curr_time - last_step_time
                    last_step_time = curr_time
                    step_time_sum += step_time
                    step_time_count += 1
                    if step_time > EXPECTED_CONTROL_PERIOD:
                        step_time_exceed_count += 1
                    if step_time_exceed_count > 0 and curr_time - report_start_time >= self._report_interval:
                        mean_step_time = step_time_sum / step_time_count if step_time_count > 0 else 0.0
                        logging.info(
                            f"[PATCHED {self} {self._report_interval}s Report] "
                            f"step_time > {EXPECTED_CONTROL_PERIOD}s: {step_time_exceed_count} times, "
                            f"mean step_time: {mean_step_time:.6f} s"
                        )
                        step_time_exceed_count = 0
                        step_time_sum = 0.0
                        step_time_count = 0
                        report_start_time = curr_time

                    # --- THE FIX ---
                    # Briefly grab lock to snapshot commands, then release.
                    with self.command_lock:
                        local_commands = list(self.commands)
                    # Lock released. CAN I/O happens unlocked.
                    try:
                        motor_feedback = self._set_commands(local_commands)
                    except RuntimeError as e:
                        if "Motor error detected" in str(e):
                            logging.warning(f"Motor error in control loop, attempting recovery: {e}")
                            recovered = self._try_recover_motors()
                            if recovered:
                                logging.warning("Motor recovery successful, continuing control loop")
                                continue
                            else:
                                self.running = False
                                raise
                        raise
                    errors = np.array([motor_feedback[i].error_code != "0x1"
                                       for i in range(len(motor_feedback))])
                    if np.any(errors):
                        logging.warning(f"Motor errors detected in feedback: {errors}")
                        recovered = self._try_recover_motors(motor_feedback)
                        if recovered:
                            logging.warning("Motor recovery successful, continuing control loop")
                            continue
                        self.running = False
                        logging.error(f"motor errors: {errors}")
                        raise Exception(
                            "motors have unrecoverable errors after recovery attempts, stopping control loop"
                        )

                    with self.state_lock:
                        self.state = motor_feedback
                        self._update_absolute_positions(motor_feedback)
                    if self.same_bus_device_driver is not None:
                        time.sleep(0.001)
                        with self.same_bus_device_lock:
                            self.same_bus_device_states = self.same_bus_device_driver.read_states()
                    time.sleep(0)
                    self._rate_recorder.track()
                except Exception as e:
                    print(f"DM Error in PATCHED control loop: {e}")
                    self.running = False
                    raise e

    Chain._set_torques_and_update_state = patched
    print("[fix] installed PATCHED _set_torques_and_update_state "
          "(command_lock held only for snapshot, not CAN)", flush=True)


def install_acquire_profiler(dm_module):
    """Same set_commands timing as lock_acquire_profiler.py."""
    Chain = dm_module.DMChainCanInterface
    MotorCmd = dm_module.MotorCmd
    stats = {"build_cmd": [], "lock_acquire": [], "lock_held_assign": [],
             "read_states": [], "total": []}
    slow = []
    THRESH_MS = 50.0

    def patched(self, torques, pos=None, vel=None, kp=None, kd=None, get_state=True):
        t0 = time.perf_counter()
        command = []
        for idx in range(len(self.motor_list)):
            command.append(MotorCmd(
                torque=torques[idx],
                pos=pos[idx] if pos is not None else 0.0,
                vel=vel[idx] if vel is not None else 0.0,
                kp=kp[idx] if kp is not None else 0.0,
                kd=kd[idx] if kd is not None else 0.0,
            ))
        t_built = time.perf_counter()
        stats["build_cmd"].append(t_built - t0)
        t_a = time.perf_counter()
        self.command_lock.acquire()
        t_b = time.perf_counter()
        stats["lock_acquire"].append(t_b - t_a)
        try:
            self.commands = command
        finally:
            self.command_lock.release()
        t_release = time.perf_counter()
        stats["lock_held_assign"].append(t_release - t_b)
        result = None
        if get_state:
            result = self.read_states(torques=torques)
        t_end = time.perf_counter()
        stats["read_states"].append(t_end - t_release)
        total_dt = t_end - t0
        stats["total"].append(total_dt)
        if total_dt * 1000 > THRESH_MS:
            ev = dict(total_ms=total_dt*1000,
                      build_ms=(t_built-t0)*1000,
                      acquire_ms=(t_b-t_a)*1000,
                      assign_ms=(t_release-t_b)*1000,
                      read_ms=(t_end-t_release)*1000)
            slow.append(ev)
            print(f"[SLOW] total={ev['total_ms']:7.1f}ms  "
                  f"ACQUIRE={ev['acquire_ms']:7.1f}  "
                  f"read={ev['read_ms']:.2f}", flush=True)
        return result

    Chain.set_commands = patched
    return stats, slow


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--can", default="can0")
    p.add_argument("--gripper", default="linear_4310")
    p.add_argument("--duration-s", type=float, default=12.0)
    p.add_argument("--cmd-hz", type=float, default=50.0)
    p.add_argument("--amplitude", type=float, default=0.3)
    p.add_argument("--freq", type=float, default=0.5)
    p.add_argument("--bias-shoulder", type=float, default=1.0)
    p.add_argument("--bias-elbow", type=float, default=1.0)
    p.add_argument("--bias-wrist-pitch", type=float, default=-0.5)
    p.add_argument("--no-fix", action="store_true",
                   help="DON'T install the loop patch (run as control)")
    args = p.parse_args()

    from i2rt.motor_drivers import dm_driver as dm

    # ALWAYS install the acquire profiler so we get the same numbers as
    # lock_acquire_profiler.py for direct comparison.
    stats, slow = install_acquire_profiler(dm)
    print("[fix] installed set_commands timing profiler", flush=True)

    if not args.no_fix:
        install_loop_patch(dm)
    else:
        print("[fix] --no-fix: leaving SDK loop unpatched (control run)", flush=True)

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

    test_pose = startup.copy()
    test_pose[1] = args.bias_shoulder
    test_pose[2] = args.bias_elbow
    test_pose[3] = args.bias_wrist_pitch
    print(f"Ramping to test pose...", flush=True)
    for i in range(1, 121):
        alpha = i / 120
        cmd = startup + alpha * (test_pose - startup)
        robot.command_joint_pos(cmd)
        time.sleep(1.0 / 30.0)
    time.sleep(0.3)

    print(f"\nDriving sinusoid {args.duration_s:.1f}s at {args.cmd_hz:.0f} Hz "
          f"({'WITH FIX' if not args.no_fix else 'NO FIX (control)'})\n", flush=True)
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

    print(f"\n=== set_commands latency ({'WITH FIX' if not args.no_fix else 'NO FIX'}) ===")
    print(f"  {'section':<22s}  {'count':>6s}  {'p50':>7s}  {'p90':>7s}  "
          f"{'p99':>7s}  {'max':>8s}  {'>50ms':>6s}  {'>500ms':>7s}")
    for key in ("total", "lock_acquire", "lock_held_assign", "read_states"):
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
    print(f"\n  slow events: {len(slow)}")

    # Safe return.
    abort = {"abort": False, "n": 0}
    def _cleanup(_s, _f):
        abort["n"] += 1
        if abort["n"] == 1:
            print("[cleanup] Ctrl-C -- ARMS WILL DROP", flush=True)
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
