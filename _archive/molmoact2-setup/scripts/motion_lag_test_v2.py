"""v2 -- isolate WHERE the 525ms stalls come from.

v1 showed:
  - both arms stall in lockstep at 9.8% (not per-bus asymmetry)
  - main thread blocks for up to 666 ms occasionally (p99=545 ms)
  - 60 ms longest stall in measured joint state

v1 conflated several things. This v2:
  1. Times `command_joint_pos` and `get_joint_pos` SEPARATELY (per call,
     per arm) so we know which of the four operations actually blocks.
  2. Installs a logging.Handler that captures SDK 'loss communication'
     and 'CAN Error' lines with a timestamp -- we cross-reference these
     with the latency spikes to test whether each spike == a CAN retry.
  3. Supports --passive: only sample state, send no commands.
     If stalls still happen in passive mode -> the SDK background thread
     itself is the source, not our command activity.
  4. Supports --single-arm: skip the right arm entirely.
     If stalls disappear -> running two arms simultaneously is the trigger.
  5. Writes a full timeline CSV per sample.

Usage:
    /home/andon/yam-tests/i2rt/.venv/bin/python \\
        /home/andon/yam-tests/molmoact2-setup/scripts/motion_lag_test_v2.py \\
        --left-can can0 --right-can can1 \\
        --left-gripper linear_4310 --right-gripper linear_4310 \\
        --duration-s 8

Try also:
  --passive                          (no commands sent)
  --single-arm                       (only can0)
  --no-bias-shoulder                 (skip the pre-test ramp; test from startup)

Always-on safety: capture startup_pose, ramp back before close().
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import signal
import sys
import time
from typing import Optional

import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType


SHOULDER_J = 1   # joint index for shoulder pitch
COMM_ERR_RE = re.compile(
    r"motor id:\s*(\d+),\s*error:\s*loss communication.*channel '(can\d+)'"
)
CAN_ERR_RE = re.compile(r"CAN Error.*motor (\d+)")


class SDKErrorTimestamper(logging.Handler):
    """Capture SDK error lines with the timestamp at which they occurred."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.events: list[tuple[float, str, str, int]] = []  # (t_perf, kind, channel, motor_id)
        self.t0 = time.perf_counter()

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        t = time.perf_counter() - self.t0
        m1 = COMM_ERR_RE.search(msg)
        if m1:
            self.events.append((t, "loss_comm", m1.group(2), int(m1.group(1))))
            return
        m2 = CAN_ERR_RE.search(msg)
        if m2:
            # Couldn't parse channel; tag with "?"
            self.events.append((t, "can_err", "?", int(m2.group(1))))


def init_arm(channel: str, gripper: str):
    return get_yam_robot(
        channel=channel,
        arm_type=ArmType.from_string_name("yam"),
        gripper_type=GripperType.from_string_name(gripper),
        zero_gravity_mode=False,
    )


def ramp_to(arms: list, target_14d: np.ndarray, duration_s: float, hz: float = 30.0,
            abort_flag: dict | None = None, label: str = "ramp"):
    """Linear ramp current -> target. arms is [left] or [left, right]."""
    if len(arms) == 1:
        q = np.asarray(arms[0].get_joint_pos(), dtype=np.float32)
        start = np.concatenate([q, np.zeros(7, dtype=np.float32)])  # pad
    else:
        q_l = np.asarray(arms[0].get_joint_pos(), dtype=np.float32)
        q_r = np.asarray(arms[1].get_joint_pos(), dtype=np.float32)
        start = np.concatenate([q_l, q_r])
    goal = np.asarray(target_14d, dtype=np.float32).copy()
    print(f"[{label}] start = {np.array2string(start, precision=3)}", flush=True)
    print(f"[{label}] goal  = {np.array2string(goal,  precision=3)}", flush=True)
    n = max(1, int(duration_s * hz))
    dt = 1.0 / hz
    for i in range(1, n + 1):
        if abort_flag and abort_flag.get("abort"):
            print(f"[{label}] aborted {i}/{n}", flush=True)
            return
        alpha = i / n
        cmd = start + alpha * (goal - start)
        arms[0].command_joint_pos(cmd[:7].astype(np.float32))
        if len(arms) == 2:
            arms[1].command_joint_pos(cmd[7:].astype(np.float32))
        time.sleep(dt)
    time.sleep(0.3)
    print(f"[{label}] done", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--left-can",  default="can0")
    p.add_argument("--right-can", default="can1")
    p.add_argument("--left-gripper",  default="linear_4310")
    p.add_argument("--right-gripper", default="linear_4310")
    p.add_argument("--duration-s", type=float, default=8.0)
    p.add_argument("--cmd-hz", type=float, default=50.0)
    p.add_argument("--amplitude", type=float, default=0.3)
    p.add_argument("--freq", type=float, default=0.5)
    p.add_argument("--bias-shoulder", type=float, default=1.0,
                   help="ramp shoulder here before the test (matches v1)")
    p.add_argument("--no-bias-shoulder", action="store_true",
                   help="don't ramp to a test pose; sample from startup pose")
    p.add_argument("--passive", action="store_true",
                   help="don't send commands; only sample get_joint_pos at cmd-hz")
    p.add_argument("--single-arm", action="store_true",
                   help="only initialize/test the left arm; skip right entirely")
    p.add_argument("--out", default="lag_test_v2.csv")
    args = p.parse_args()

    # Install SDK error timestamper -- must be BEFORE arms init so we catch
    # init-time errors too.
    sdk_err = SDKErrorTimestamper()
    logging.getLogger().addHandler(sdk_err)
    logging.getLogger().setLevel(logging.INFO)

    # Also tee a basic stream handler for visibility (without duplicating).
    if not any(isinstance(h, logging.StreamHandler) and h is not sdk_err
               for h in logging.getLogger().handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(sh)

    print("\nMode:", "PASSIVE" if args.passive else "ACTIVE (send commands)",
          "|", "SINGLE-ARM" if args.single_arm else "DUAL-ARM", flush=True)
    print(f"Initializing arm(s)...", flush=True)
    left  = init_arm(args.left_can,  args.left_gripper)
    right = None if args.single_arm else init_arm(args.right_can, args.right_gripper)
    arms = [left] if right is None else [left, right]
    time.sleep(0.5)

    # Startup pose.
    q_l = np.asarray(left.get_joint_pos(), dtype=np.float32)
    if right is not None:
        q_r = np.asarray(right.get_joint_pos(), dtype=np.float32)
        startup_pose = np.concatenate([q_l, q_r])
    else:
        startup_pose = np.concatenate([q_l, np.zeros(7, dtype=np.float32)])
    print(f"Startup pose = {np.array2string(startup_pose, precision=3)}", flush=True)

    # Pre-test bias ramp (matches v1 default so results are comparable).
    if not args.no_bias_shoulder and not args.passive:
        test_pose = startup_pose.copy()
        test_pose[SHOULDER_J] = args.bias_shoulder
        if right is not None:
            test_pose[7 + SHOULDER_J] = args.bias_shoulder
        ramp_to(arms, test_pose, duration_s=4.0, hz=30.0, label="ramp-to-test")

    # Reset timestamper t0 right before the measurement loop so events align.
    sdk_err.t0 = time.perf_counter()

    # Data collection.
    n_samples = int(args.duration_s * args.cmd_hz)
    dt = 1.0 / args.cmd_hz
    omega = 2.0 * np.pi * args.freq
    rows: list[list[float]] = []

    stop = {"abort": False}
    def _sigint(_sig, _frame):
        stop["abort"] = True
        print("[SIGINT] stopping measurement", flush=True)
    signal.signal(signal.SIGINT, _sigint)

    print(f"\nMeasurement loop: {n_samples} samples @ {args.cmd_hz:.0f} Hz "
          f"(passive={args.passive}, single_arm={args.single_arm})\n", flush=True)

    cmd_l = (startup_pose[:7]).copy()
    cmd_r = (startup_pose[7:]).copy() if right is not None else None
    if not args.no_bias_shoulder and not args.passive:
        cmd_l[SHOULDER_J] = args.bias_shoulder
        if cmd_r is not None:
            cmd_r[SHOULDER_J] = args.bias_shoulder

    t0 = time.perf_counter()
    next_tick = t0
    for i in range(n_samples):
        if stop["abort"]:
            print(f"aborted at sample {i}", flush=True)
            break
        t = time.perf_counter() - t0
        # Update commanded sinusoid offset (only used in active mode).
        offset = args.amplitude * np.sin(omega * t)

        # Time each call independently.
        if not args.passive:
            cmd_l[SHOULDER_J] = (args.bias_shoulder if not args.no_bias_shoulder
                                 else startup_pose[SHOULDER_J]) + offset
            t_cmd_l_start = time.perf_counter()
            left.command_joint_pos(cmd_l.astype(np.float32))
            t_cmd_l = time.perf_counter() - t_cmd_l_start
        else:
            t_cmd_l = 0.0

        if not args.passive and right is not None:
            cmd_r[SHOULDER_J] = (args.bias_shoulder if not args.no_bias_shoulder
                                 else startup_pose[7 + SHOULDER_J]) + offset
            t_cmd_r_start = time.perf_counter()
            right.command_joint_pos(cmd_r.astype(np.float32))
            t_cmd_r = time.perf_counter() - t_cmd_r_start
        else:
            t_cmd_r = 0.0

        t_read_l_start = time.perf_counter()
        q_l = np.asarray(left.get_joint_pos(), dtype=np.float32)
        t_read_l = time.perf_counter() - t_read_l_start

        if right is not None:
            t_read_r_start = time.perf_counter()
            q_r = np.asarray(right.get_joint_pos(), dtype=np.float32)
            t_read_r = time.perf_counter() - t_read_r_start
        else:
            t_read_r = 0.0
            q_r = np.zeros(7, dtype=np.float32)

        cmd_l_j2 = (cmd_l[SHOULDER_J] if not args.passive else float("nan"))
        cmd_r_j2 = (cmd_r[SHOULDER_J] if cmd_r is not None and not args.passive
                    else float("nan"))
        rows.append([
            float(t),
            float(cmd_l_j2), float(q_l[SHOULDER_J]),
            float(cmd_r_j2), float(q_r[SHOULDER_J]),
            float(t_cmd_l), float(t_cmd_r), float(t_read_l), float(t_read_r),
        ])

        if i % max(1, int(args.cmd_hz)) == 0:
            print(f"  t={t:5.2f}s  q_L={q_l[SHOULDER_J]:+.3f}  q_R={q_r[SHOULDER_J]:+.3f}  "
                  f"cmd_L_t={t_cmd_l*1000:5.1f}ms  cmd_R_t={t_cmd_r*1000:5.1f}ms  "
                  f"read_L_t={t_read_l*1000:5.1f}ms  read_R_t={t_read_r*1000:5.1f}ms",
                  flush=True)

        next_tick += dt
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)

    # Write CSV with full per-sample timings.
    out_path = os.path.abspath(args.out)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s",
                    "cmd_L_j2", "q_L_j2",
                    "cmd_R_j2", "q_R_j2",
                    "cmd_L_dt_s", "cmd_R_dt_s",
                    "read_L_dt_s", "read_R_dt_s"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} samples to {out_path}", flush=True)

    # Also write SDK events.
    ev_path = out_path.replace(".csv", "_sdk_events.csv")
    with open(ev_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "kind", "channel", "motor_id"])
        for ev in sdk_err.events:
            w.writerow(ev)
    print(f"Wrote {len(sdk_err.events)} SDK error events to {ev_path}", flush=True)

    # ---------------- analysis ----------------
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape[0] > 5:
        cmd_l_dt = arr[:, 5]
        cmd_r_dt = arr[:, 6]
        read_l_dt = arr[:, 7]
        read_r_dt = arr[:, 8]

        def stats(name: str, x: np.ndarray) -> None:
            p50 = float(np.percentile(x, 50)) * 1000
            p90 = float(np.percentile(x, 90)) * 1000
            p99 = float(np.percentile(x, 99)) * 1000
            mx  = float(x.max()) * 1000
            n_long = int((x > 0.020).sum())  # >20ms
            n_huge = int((x > 0.200).sum())  # >200ms
            print(f"  {name:<14s}  p50={p50:6.2f}  p90={p90:6.2f}  "
                  f"p99={p99:6.2f}  max={mx:7.2f}  >20ms={n_long}  >200ms={n_huge}")

        print("\n=== per-call latency (ms) ===")
        print(f"  {'call':<14s}  {'p50':>6}  {'p90':>6}  {'p99':>6}  {'max':>7}  >20ms  >200ms")
        if not args.passive:
            stats("cmd_L  (left)",  cmd_l_dt)
            if right is not None:
                stats("cmd_R  (right)", cmd_r_dt)
        stats("read_L (left)",  read_l_dt)
        if right is not None:
            stats("read_R (right)", read_r_dt)

        # Correlation: do cmd and read spikes hit on the same samples?
        # Compute fraction of samples where BOTH read_L and read_R are slow.
        if right is not None:
            big_l = read_l_dt > 0.020
            big_r = read_r_dt > 0.020
            if big_l.sum() and big_r.sum():
                joint = (big_l & big_r).sum()
                print(f"\n  read>20ms samples:  L={big_l.sum()}  R={big_r.sum()}  "
                      f"both={joint}  (joint/L={joint/max(big_l.sum(),1):.0%}, "
                      f"joint/R={joint/max(big_r.sum(),1):.0%})")
                if joint / max(big_l.sum(), 1) > 0.7:
                    print("  -> SIMULTANEOUS: left and right read spikes co-occur.")
                    print("     Likely a global resource block (USB hub IRQ, "
                          "shared SDK state, OS scheduling).")
                else:
                    print("  -> INDEPENDENT: each bus stalls on its own.")
                    print("     Likely per-bus issues (CAN dropouts, USB device-level).")

        # SDK error events vs read spikes: temporal correlation.
        if sdk_err.events:
            ev_times = np.asarray([e[0] for e in sdk_err.events])
            big_t = arr[arr[:, 7] > 0.020, 0]
            if len(big_t):
                # For each error event, find nearest read>20ms sample
                near = []
                for et in ev_times:
                    diffs = np.abs(big_t - et)
                    near.append(float(diffs.min()) if len(diffs) else float("nan"))
                near = np.array(near)
                within_100ms = int((near < 0.1).sum())
                print(f"\n  SDK error events: {len(sdk_err.events)}  "
                      f"({within_100ms}/{len(sdk_err.events)} within 100ms of a "
                      f"read>20ms sample)")
                if within_100ms / max(len(sdk_err.events), 1) > 0.5:
                    print("  -> ERROR CORRELATED: latency spikes coincide with "
                          "SDK CAN errors. The retry loop is the burst trigger.")
                else:
                    print("  -> NOT CORRELATED: spikes happen WITHOUT explicit "
                          "SDK errors. Look at lower-level (USB, syscall, lock).")
        else:
            print(f"\n  SDK error events: 0 in {arr[-1,0]:.1f}s "
                  f"-- spikes are NOT from SDK-visible CAN retries.")

    # SAFETY: ramp back.
    abort = {"abort": False, "n": 0}
    def _cleanup_sigint(_sig, _frame):
        abort["n"] += 1
        if abort["n"] == 1:
            print("[cleanup] Ctrl-C: aborting return-ramp. ARMS WILL DROP.", flush=True)
            abort["abort"] = True
        else:
            os._exit(130)
    signal.signal(signal.SIGINT, _cleanup_sigint)

    print("\nReturning to startup before disable...", flush=True)
    try:
        ramp_to(arms, startup_pose, duration_s=5.0, hz=30.0,
                abort_flag=abort, label="return")
    except BaseException as e:
        print(f"WARNING: return ramp failed: {e}. Arms may drop.", flush=True)

    try: left.close()
    except Exception: pass
    if right is not None:
        try: right.close()
        except Exception: pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
