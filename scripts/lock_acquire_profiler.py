"""Time the mutex acquire specifically, inside DMSingleMotorCanInterface.set_commands.

Hypothesis under test:
  When motor_chain_robot calls motor_chain.set_commands(...), the slow part
  is *waiting to acquire DMSingleMotorCanInterface.command_lock* because
  dm_driver.Thread-1 (the CAN talker) holds that same lock continuously at
  ~300 Hz.

If true, we should see:
  - lock_acquire_ms approximately equal to the total set_commands_ms
  - command_assign_ms ~ 0
  - read_states_ms ~ small but possibly non-zero (it acquires state_lock,
    a DIFFERENT lock, also touched by Thread-1)

If lock_acquire is fast but something else is slow -> the lock-starvation
story is wrong and we look elsewhere.

Run a 12-second sinusoid. Prints any set_commands call >50ms with the
breakdown live. Final summary shows percentiles per section.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import numpy as np


def install_profiler(dm_module):
    """Monkey-patch DMSingleMotorCanInterface.set_commands with sub-timings.

    We replicate the original implementation, separately timing each step.
    """
    SingleIface = dm_module.DMSingleMotorCanInterface
    MotorCmd = dm_module.MotorCmd

    stats = {
        "build_cmd": [],
        "lock_acquire": [],
        "lock_held_assign": [],
        "read_states": [],
        "total": [],
    }
    slow: list[dict] = []
    THRESH_MS = 50.0

    orig = SingleIface.set_commands

    def patched(self, torques, pos=None, vel=None, kp=None, kd=None, get_state=True):
        t0 = time.perf_counter()

        # Section 1: build the command list (Python work, should be fast).
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

        # Section 2: acquire the mutex. THIS is where the lock-starvation
        # theory says the time goes.
        t_a = time.perf_counter()
        self.command_lock.acquire()
        t_b = time.perf_counter()
        stats["lock_acquire"].append(t_b - t_a)
        acquire_dt = t_b - t_a

        # Section 3: while holding the lock, do the assignment, then release.
        # The body here should be instant.
        try:
            self.commands = command
        finally:
            self.command_lock.release()
        t_release = time.perf_counter()
        stats["lock_held_assign"].append(t_release - t_b)
        assign_dt = t_release - t_b

        # Section 4: read_states (acquires the OTHER lock, state_lock).
        result = None
        if get_state:
            result = self.read_states(torques=torques)
        t_end = time.perf_counter()
        stats["read_states"].append(t_end - t_release)
        read_dt = t_end - t_release

        total_dt = t_end - t0
        stats["total"].append(total_dt)
        if total_dt * 1000 > THRESH_MS:
            ev = {
                "total_ms": total_dt * 1000,
                "build_ms": (t_built - t0) * 1000,
                "acquire_ms": acquire_dt * 1000,
                "assign_ms": assign_dt * 1000,
                "read_ms": read_dt * 1000,
            }
            slow.append(ev)
            print(f"[SLOW set_cmd] total={ev['total_ms']:7.1f}ms  "
                  f"build={ev['build_ms']:5.2f}  "
                  f"ACQUIRE={ev['acquire_ms']:7.1f}  "
                  f"assign={ev['assign_ms']:6.3f}  "
                  f"read={ev['read_ms']:5.2f}", flush=True)
        return result

    SingleIface.set_commands = patched
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
    args = p.parse_args()

    from i2rt.motor_drivers import dm_driver as dm
    stats, slow = install_profiler(dm)
    print("installed DMSingleMotorCanInterface.set_commands profiler "
          "(threshold 50ms; ACQUIRE column = lock wait time)", flush=True)

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

    # Safe loaded pose.
    test_pose = startup.copy()
    test_pose[1] = args.bias_shoulder
    test_pose[2] = args.bias_elbow
    test_pose[3] = args.bias_wrist_pitch
    print(f"Ramping to test pose: shoulder={args.bias_shoulder:.2f} "
          f"elbow={args.bias_elbow:.2f} wrist_pitch={args.bias_wrist_pitch:.2f}",
          flush=True)
    for i in range(1, 121):
        alpha = i / 120
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

    print(f"\n=== DMSingleMotorCanInterface.set_commands per-section latency ===")
    print(f"  {'section':<22s}  {'count':>6s}  {'p50':>7s}  {'p90':>7s}  "
          f"{'p99':>7s}  {'max':>8s}  {'>50ms':>6s}  {'>500ms':>7s}")
    for key in ("total", "build_cmd", "lock_acquire", "lock_held_assign", "read_states"):
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

    if slow:
        print(f"\n  slow events: {len(slow)}.  worst 5:")
        for e in sorted(slow, key=lambda x: -x["total_ms"])[:5]:
            print(f"    total={e['total_ms']:7.1f}  build={e['build_ms']:5.2f}  "
                  f"ACQUIRE={e['acquire_ms']:7.1f}  assign={e['assign_ms']:6.3f}  "
                  f"read={e['read_ms']:5.2f}")

    # Verdict
    if len(stats["lock_acquire"]) > 0 and len(stats["total"]) > 0:
        acq_arr = np.asarray(stats["lock_acquire"]) * 1000.0
        tot_arr = np.asarray(stats["total"]) * 1000.0
        # Fraction of total spent in acquire on the WORST events.
        slow_mask = tot_arr > 50
        if slow_mask.sum() > 0:
            frac = float((acq_arr[slow_mask] / tot_arr[slow_mask]).mean())
            print(f"\n  on slow events, ACQUIRE took {frac*100:.0f}% of total set_cmd time.")
            if frac > 0.85:
                print("  -> CONFIRMED: the lock acquire IS the bottleneck. Mutex starvation")
                print("     between motor_chain_robot's thread and dm_driver's Thread-1.")
            elif frac > 0.5:
                print("  -> MIXED: acquire is a big chunk but not everything. Read_states may")
                print("     be contributing (which would mean state_lock is also contended).")
            else:
                print("  -> NOT CONFIRMED: acquire is fast. The slow time must be in read_states")
                print("     or elsewhere. Lock-starvation theory wrong; we need to look deeper.")

    # Safe return
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
