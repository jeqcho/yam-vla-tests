"""Watch CAN frames at the kernel/socketcan level while the SDK runs.

The lag-test isolation revealed: the SDK silently retries dropped CAN
responses (each retry is ~11ms; up to 15 retries per motor; 7 motors
per arm) while holding _state_lock. The 0 logged 'loss communication'
events mean the dropped frames eventually succeed -- they're being
silently retried, not failing outright.

This script answers: ARE THE FRAMES PHYSICALLY ARRIVING LATE ON THE
WIRE, or arriving on time but missed by the SDK's recv() window?

It opens a raw passive socketcan listener on the same channel the SDK
is using (no commands -- pure observation) and times the inter-arrival
gaps between consecutive frames. If the SDK is meanwhile commanding
the arm, we'll see request/response pairs and any gaps between them.

Run alongside another process driving the arm (e.g. yam_client.py or
motion_lag_test_v2.py). This one only watches.

Usage:
    /home/andon/yam-tests/i2rt/.venv/bin/python \\
        /home/andon/yam-tests/molmoact2-setup/scripts/can_frame_timing.py \\
        --channel can0 --duration-s 15
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import can


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--channel", default="can0")
    p.add_argument("--duration-s", type=float, default=15.0)
    p.add_argument("--out", default=None,
                   help="optional CSV path; default /tmp/can_frame_timing_<channel>.csv")
    args = p.parse_args()
    out_path = args.out or f"/tmp/can_frame_timing_{args.channel}.csv"

    print(f"Listening passively on {args.channel} for {args.duration_s}s...", flush=True)
    print(f"(start your other process now -- yam_client.py or motion_lag_test_v2.py)", flush=True)

    bus = can.interface.Bus(channel=args.channel, bustype="socketcan")
    frames: list[tuple[float, int, int, bool]] = []  # (t, arbitration_id, dlc, extended)
    t0 = time.perf_counter()
    deadline = t0 + args.duration_s
    while time.perf_counter() < deadline:
        msg = bus.recv(timeout=0.5)
        if msg is None:
            continue
        t = time.perf_counter() - t0
        frames.append((t, msg.arbitration_id, msg.dlc, msg.is_extended_id))

    bus.shutdown()
    print(f"\nCaptured {len(frames)} frames in {args.duration_s:.1f}s "
          f"(~{len(frames)/args.duration_s:.0f} fr/s)", flush=True)

    # Write CSV.
    with open(out_path, "w") as f:
        f.write("t_s,arb_id_hex,dlc,extended\n")
        for t, aid, dlc, ext in frames:
            f.write(f"{t:.6f},0x{aid:X},{dlc},{int(ext)}\n")
    print(f"Wrote {out_path}", flush=True)

    # Per-ID inter-arrival timing.
    if len(frames) > 2:
        import collections
        by_id: dict[int, list[float]] = collections.defaultdict(list)
        for t, aid, _, _ in frames:
            by_id[aid].append(t)
        print(f"\n=== inter-arrival timing per arbitration_id ===")
        print(f"  {'arb_id':>8s}  {'n':>5s}  {'mean_ms':>8s}  {'p99_ms':>8s}  {'max_ms':>8s}  {'gaps>20ms':>10s}")
        for aid in sorted(by_id.keys()):
            ts = sorted(by_id[aid])
            if len(ts) < 2:
                continue
            gaps = [(ts[i+1] - ts[i]) * 1000.0 for i in range(len(ts)-1)]
            import numpy as np
            arr = np.asarray(gaps)
            mean = float(arr.mean())
            p99 = float(np.percentile(arr, 99))
            mx = float(arr.max())
            n_long = int((arr > 20.0).sum())
            print(f"  0x{aid:>6X}  {len(ts):>5d}  {mean:>8.2f}  {p99:>8.2f}  {mx:>8.2f}  {n_long:>10d}")

        # Global gaps (between consecutive frames of any kind) -- shows the
        # bus going "quiet" for periods of time, which is consistent with
        # the SDK in its retry loop NOT sending anything either.
        ts_all = sorted([t for t, _, _, _ in frames])
        gaps_all = [(ts_all[i+1] - ts_all[i]) * 1000.0 for i in range(len(ts_all)-1)]
        import numpy as np
        arr_all = np.asarray(gaps_all)
        print(f"\n  global inter-frame gap on {args.channel}: "
              f"mean={arr_all.mean():.2f}ms  p99={np.percentile(arr_all,99):.2f}ms  "
              f"max={arr_all.max():.2f}ms  gaps>50ms={(arr_all>50.0).sum()}")
        if arr_all.max() > 100:
            big_idx = int(np.argmax(arr_all))
            print(f"  longest gap = {arr_all[big_idx]:.1f}ms  at t={ts_all[big_idx]:.3f}s "
                  f"(between frame#{big_idx} and #{big_idx+1})")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
