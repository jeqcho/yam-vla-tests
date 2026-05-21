"""v2: capture CAN payloads (not just timing), decode commanded position out
of each motor-command frame, and detect 'stale command runs.'

v1 showed CAN traffic is steady at 300 Hz per motor. But that only proves
frames are *flowing* on the wire -- it doesn't say whether they carry
*new* values or the same old commanded position repeated. If our main
thread is blocked, the SDK's background control thread keeps sending the
last commanded position at 300 Hz -- on the wire that looks identical to
healthy traffic.

This script records each command frame's first 16 bits (DM motor MIT mode:
bytes 0-1 are the 16-bit raw target position), then computes:
  - run-length encoding per motor: stretches where the commanded position
    didn't change between consecutive command frames
  - longest stale run per motor in milliseconds
  - distribution of stale-run lengths

If we see long stale runs (e.g. >50 ms of identical commands followed
by a sudden jump in raw value), that's the visible-arm-burst signature
explained: the SDK is faithfully streaming what we last commanded, while
our main thread was blocked from updating it.

Run alongside motion_lag_test_v2 or yam_client.py as before.

Usage:
    /home/andon/yam-tests/i2rt/.venv/bin/python \\
        /home/andon/yam-tests/molmoact2-setup/scripts/can_frame_timing_v2.py \\
        --channel can0 --duration-s 25
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import can
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--channel", default="can0")
    p.add_argument("--duration-s", type=float, default=25.0)
    p.add_argument("--out", default=None,
                   help="CSV of per-frame records; default /tmp/can_payload_<channel>.csv")
    args = p.parse_args()
    out_path = args.out or f"/tmp/can_payload_{args.channel}.csv"

    print(f"Listening passively on {args.channel} for {args.duration_s}s (payload capture)...",
          flush=True)
    print(f"(start motion_lag_test_v2.py now in another terminal)", flush=True)

    bus = can.interface.Bus(channel=args.channel, interface="socketcan")
    # frames: list of (t_s, arb_id, dlc, payload_bytes)
    frames: list[tuple[float, int, int, bytes]] = []
    t0 = time.perf_counter()
    deadline = t0 + args.duration_s
    while time.perf_counter() < deadline:
        msg = bus.recv(timeout=0.5)
        if msg is None:
            continue
        t = time.perf_counter() - t0
        frames.append((t, msg.arbitration_id, msg.dlc, bytes(msg.data)))
    bus.shutdown()

    print(f"\nCaptured {len(frames)} frames in {args.duration_s:.1f}s "
          f"(~{len(frames)/args.duration_s:.0f} fr/s)", flush=True)

    # Write raw CSV.
    with open(out_path, "w") as f:
        f.write("t_s,arb_id_hex,dlc,p0,p1,p2,p3,p4,p5,p6,p7\n")
        for t, aid, dlc, pl in frames:
            pl_pad = pl + b"\x00" * (8 - len(pl))
            f.write(f"{t:.6f},0x{aid:X},{dlc}," +
                    ",".join(str(b) for b in pl_pad[:8]) + "\n")
    print(f"Wrote {out_path}", flush=True)

    # Stale-run analysis on COMMAND frames only (arb_id 1..7).
    # In DM motor MIT-mode command frames, bytes 0-1 are the 16-bit raw
    # target position (the SDK packs floats into this range).
    print("\n=== stale-command-run analysis (per motor) ===")
    print("  Stretches of consecutive command frames carrying the SAME raw target.")
    print("  Long runs = the SDK was streaming a stale commanded position because")
    print("  our main thread was blocked from updating it.\n")
    print(f"  {'motor_id':>8s}  {'n_cmds':>7s}  {'unique':>7s}  "
          f"{'mean_run':>9s}  {'p99_run':>9s}  {'max_run':>9s}  "
          f"{'max_run_ms':>11s}  {'runs>10ms':>10s}")

    # Group command frames by motor id (arb_id == motor_id for 1..7).
    by_motor: dict[int, list[tuple[float, int]]] = {}  # motor_id -> [(t, pos16)]
    for t, aid, _, pl in frames:
        if 1 <= aid <= 7 and len(pl) >= 2:
            pos16 = (pl[0] << 8) | pl[1]
            by_motor.setdefault(aid, []).append((t, pos16))

    summary_lines = []
    for mid in sorted(by_motor.keys()):
        seq = by_motor[mid]
        if len(seq) < 2:
            continue
        # Run-length encode the position values.
        runs_n = []           # run length in samples
        runs_t = []           # run length in seconds
        i = 0
        while i < len(seq):
            j = i
            while j + 1 < len(seq) and seq[j+1][1] == seq[i][1]:
                j += 1
            n_cmds = j - i + 1
            duration_s = seq[j][0] - seq[i][0]
            runs_n.append(n_cmds)
            runs_t.append(duration_s)
            i = j + 1
        runs_n_arr = np.asarray(runs_n)
        runs_t_arr = np.asarray(runs_t) * 1000.0  # ms
        n_total = len(seq)
        n_unique = len(set(pos for _, pos in seq))
        mean_run = float(runs_n_arr.mean())
        p99_run = float(np.percentile(runs_n_arr, 99))
        max_run = int(runs_n_arr.max())
        max_run_ms = float(runs_t_arr.max())
        runs_over_10ms = int((runs_t_arr > 10.0).sum())
        print(f"  {mid:>8d}  {n_total:>7d}  {n_unique:>7d}  "
              f"{mean_run:>9.2f}  {p99_run:>9.2f}  {max_run:>9d}  "
              f"{max_run_ms:>11.1f}  {runs_over_10ms:>10d}")
        summary_lines.append((mid, max_run_ms, runs_over_10ms))

    # Verdict.
    print("\n  --- verdict ---")
    if not summary_lines:
        print("  No command frames seen on this bus. Re-run with the test driving the arm.")
    else:
        worst_ms = max(s[1] for s in summary_lines)
        worst_mid = next(s[0] for s in summary_lines if s[1] == worst_ms)
        total_long_runs = sum(s[2] for s in summary_lines)
        print(f"  worst stale run: motor {worst_mid}, {worst_ms:.1f} ms of identical commands")
        print(f"  total stale runs >10ms across all motors: {total_long_runs}")
        if worst_ms > 100:
            print("  -> CONFIRMED: long stale runs (>100ms). The SDK was sending stale")
            print("     commands for that long. Our main thread was blocked from updating.")
            print("     This matches the visible arm bursts: stand still, then snap.")
        elif worst_ms > 50:
            print("  -> MODERATE: some stale runs in the 50-100ms range. Mild jitter,")
            print("     should be visible but not severe.")
        else:
            print("  -> CLEAN: no significant stale-command stretches. The SDK is sending")
            print("     fresh commands at the expected rate. If you still saw lag, it's")
            print("     not from our command path being stalled.")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
