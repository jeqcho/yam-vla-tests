#!/usr/bin/env python3
"""Interactive policy REPL — type instructions, watch arms run.

Usage:
    ./scripts/run_repl.py --policy <name> [hardware overrides...]

Concrete examples:
    # MolmoAct2 with default hardware (cameras + cans from yam_setup_config.json)
    ./scripts/run_repl.py --policy molmoact2

    # GR00T-N1.7 in dry-run (no arm motion, just print actions)
    ./scripts/run_repl.py --policy gr00t-n17 --dry-run

    # Pi-0.5 with Rerun streaming on
    ./scripts/run_repl.py --policy pi05 --rerun

What it does:
    1. Bring up cameras + arms (same as run_eval.py)
    2. Ramp arms to the policy's canonical ready pose (in-distribution
       joint configuration; preserves current gripper opening)
    3. Show a prompt loop:
         > pick up the orange cube
         [running... press → or Enter to stop]
         ...arms move...
         > stack two blocks
         [running...]
         > /quit
    4. Each instruction runs ONE attempt via the same core.run_attempt
       loop the eval harness uses -- you get the same safety, telemetry,
       and Rerun .rrd recording for free.

Built for prompt iteration: smoke-test a fresh finetune, find which
phrasing works, debug a misbehaving model. Use run_eval.py for batch
runs with CSV scoring.

Special commands inside the REPL:
    /quit  /q     exit cleanly (return arms to startup pose first)
    /help         show this list
    /info         print server info + last attempt stats
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Make src/ importable.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np

from yam_vla.core import (
    AttemptKnobs, PolicyConfig, RerunRecorder,
    capture_invocation, init_arm, load_setup_config, make_camera,
    prompt_journal_entry, ramp_to_pose, run_attempt, write_journal_entry,
    DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT, DEFAULT_CAM_FPS,
    DEFAULT_GRIPPER_STEP, DEFAULT_HORIZON_STRIDE,
    DEFAULT_JOURNAL_PATH, DEFAULT_MAX_STEP_RAD, DEFAULT_TRAIN_FPS,
)


# Operator-UX helpers (→ / Enter early-stop, log silencing) live in
# yam_vla.core.keyboard so the eval harness and REPL stay in lockstep.
from yam_vla.core.keyboard import AdvanceWatcher, silence_root_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_repl.py",
        description="Interactive REPL for any registered VLA policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--policy", required=True,
                   help="policy name; matches configs/policy/<name>.yaml")
    p.add_argument("--config-dir", default=str(_REPO / "configs" / "policy"))
    p.add_argument("--max-chunks", type=int, default=200)
    p.add_argument("--horizon-stride", type=int, default=None,
                   help="overrides the per-policy YAML default")
    p.add_argument("--train-fps", type=float, default=DEFAULT_TRAIN_FPS)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--timeout-s", type=float, default=15.0)
    p.add_argument("--inference-mode", default="sync",
                   choices=["sync", "async-naive", "async-time-aligned"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-step-rad", type=float, default=DEFAULT_MAX_STEP_RAD)
    p.add_argument("--gripper-step", type=float, default=DEFAULT_GRIPPER_STEP)
    p.add_argument("--no-return-on-exit", action="store_true")

    # hardware overrides
    p.add_argument("--left-can",  default=None)
    p.add_argument("--right-can", default=None)
    p.add_argument("--gripper",   default=None)
    p.add_argument("--top-cam-serial",   default=None)
    p.add_argument("--top-cam-v4l2",     default=None)
    p.add_argument("--left-cam-serial",  default=None)
    p.add_argument("--left-cam-v4l2",    default=None)
    p.add_argument("--right-cam-serial", default=None)
    p.add_argument("--right-cam-v4l2",   default=None)
    p.add_argument("--cam-width",  type=int, default=DEFAULT_CAM_WIDTH)
    p.add_argument("--cam-height", type=int, default=DEFAULT_CAM_HEIGHT)
    p.add_argument("--cam-fps",    type=int, default=DEFAULT_CAM_FPS)

    # observability
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--rerun-save", default=None)
    p.add_argument("--rerun-connect", default=None)
    p.add_argument("--no-journal", action="store_true")
    p.add_argument("--journal-path", default=DEFAULT_JOURNAL_PATH)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s")
    # Cap i2rt's noisy root-logger INFO output (Grav Comp Frequency / Total
    # rate heartbeats) so it can't garble the operator's typed instructions.
    silence_root_logger()
    log = logging.getLogger("yam_vla.repl")

    args = build_parser().parse_args()

    policy_yaml = Path(args.config_dir) / f"{args.policy}.yaml"
    if not policy_yaml.exists():
        sys.exit(f"policy config not found: {policy_yaml}")

    cfg = PolicyConfig.from_path(policy_yaml)
    if args.horizon_stride is None:
        args.horizon_stride = int(cfg.control.get("horizon_stride_default", 6))
    # Canonical "ready pose" the arms ramp to once after init. See the
    # policy YAML's control.ready_pose for derivation. Optional -- if
    # absent, arms simply hold their post-init pose (legacy behavior).
    ready_pose_cfg = cfg.control.get("ready_pose")
    ready_pose_ramp_s = float(cfg.control.get("ready_pose_ramp_duration_s", 5.0))
    policy = cfg.build()
    setup_cfg = load_setup_config()

    info = policy.info(timeout_s=5.0)
    print(f"\n[repl] policy : {policy.name} ({info.model_id or '?'})", flush=True)
    print(f"[repl] horizon hint: {info.action_horizon_hint}, stride: {args.horizon_stride}",
          flush=True)

    invocation = capture_invocation()
    session_start_s = time.time()

    # Hardware bring-up
    rerun = RerunRecorder(
        enabled=args.rerun, save_path=args.rerun_save,
        connect=args.rerun_connect, app_id=f"yam_vla_repl_{policy.name}",
    )
    top = cam_l = cam_r = None
    left = right = None
    startup_pose: Optional[np.ndarray] = None
    last_stats = None

    try:
        cam_kw = dict(width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
        top = make_camera("top",
                          args.top_cam_serial or setup_cfg.get("top_cam_serial"),
                          args.top_cam_v4l2   or setup_cfg.get("top_cam_v4l2"), **cam_kw)
        cam_l = make_camera("left",
                            args.left_cam_serial or setup_cfg.get("left_cam_serial"),
                            args.left_cam_v4l2   or setup_cfg.get("left_cam_v4l2"), **cam_kw)
        cam_r = make_camera("right",
                            args.right_cam_serial or setup_cfg.get("right_cam_serial"),
                            args.right_cam_v4l2   or setup_cfg.get("right_cam_v4l2"), **cam_kw)
        for c in (top, cam_l, cam_r):
            c.start()
        for _ in range(3):
            for c in (top, cam_l, cam_r):
                try: c.grab()
                except Exception: pass

        gripper = args.gripper or setup_cfg.get("gripper", "linear_4310")
        left  = init_arm(args.left_can  or setup_cfg.get("left_can",  "can0"), gripper)
        right = init_arm(args.right_can or setup_cfg.get("right_can", "can1"), gripper)
        startup_pose = np.concatenate([
            np.asarray(left.get_joint_pos(),  dtype=np.float32),
            np.asarray(right.get_joint_pos(), dtype=np.float32),
        ])

        # Ramp arms to the policy's canonical ready pose so every typed
        # instruction starts from an in-distribution joint configuration
        # (the centroid of training-action means). Gripper indices are
        # preserved from current state so we don't slam-close on what's
        # held. No ramp if the policy YAML doesn't declare one.
        if ready_pose_cfg is not None:
            rp = np.asarray(ready_pose_cfg, dtype=np.float32)
            if rp.shape != (14,):
                log.warning("ready_pose has shape %s, expected (14,); skipping ramp",
                            rp.shape)
            else:
                rp[6]  = startup_pose[6]
                rp[13] = startup_pose[13]
                log.info("Ramping to canonical ready pose (%.1fs)...", ready_pose_ramp_s)
                ramp_to_pose(left, right, rp,
                             duration_s=ready_pose_ramp_s,
                             label="initial move-to-ready")

        # Prompt loop
        print("\n" + "=" * 70, flush=True)
        print(f"yam_vla REPL  --  policy={policy.name}", flush=True)
        print("=" * 70, flush=True)
        print("Type an instruction (or /help, /info, /quit). Enter while", flush=True)
        print("an attempt is running ends that attempt early.", flush=True)
        print("=" * 70 + "\n", flush=True)

        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in {"/quit", "/q", "/exit"}:
                break
            if line == "/help":
                print("  /quit   -- exit cleanly (returns arms to startup)", flush=True)
                print("  /info   -- show server info + last attempt stats", flush=True)
                print("  /help   -- this", flush=True)
                print("  <text>  -- send <text> as instruction; run one attempt", flush=True)
                continue
            if line == "/info":
                print(f"  policy: {policy.name}", flush=True)
                print(f"  model:  {info.model_id}", flush=True)
                print(f"  transport: {info.transport}", flush=True)
                if last_stats:
                    print(f"  last attempt: chunks={last_stats.chunks} "
                          f"duration={last_stats.duration_s:.1f}s "
                          f"rtt_mean={last_stats.rtt_ms_mean:.0f}ms "
                          f"clip_rate={last_stats.clip_rate:.3f}", flush=True)
                continue

            # Run an attempt
            knobs = AttemptKnobs(
                instruction=line,
                max_chunks=args.max_chunks,
                train_fps=args.train_fps,
                horizon_stride=args.horizon_stride,
                max_step_rad=args.max_step_rad,
                gripper_step=args.gripper_step,
                timeout_s=args.timeout_s,
                inference_mode=args.inference_mode,
                dry_run=args.dry_run,
                policy_opts={"num_steps": args.num_steps},
            )
            watcher = AdvanceWatcher()
            watcher.start()
            print(f"[attempt] running. press → or Enter to stop early.", flush=True)
            try:
                last_stats = run_attempt(
                    policy=policy, knobs=knobs,
                    top_cam=top, left_cam=cam_l, right_cam=cam_r,
                    left_arm=left, right_arm=right,
                    rerun=rerun,
                    stop=watcher.predicate(),
                )
            finally:
                # Restore cooked stdin BEFORE the next `input("> ")` so
                # the operator's typed characters aren't eaten by the
                # background raw-mode watcher.
                watcher.stop()
            print(f"[attempt done] chunks={last_stats.chunks} "
                  f"duration={last_stats.duration_s:.1f}s "
                  f"rtt_mean={last_stats.rtt_ms_mean:.0f}ms "
                  f"rtt_max={last_stats.rtt_ms_max:.0f}ms "
                  f"clip_rate={last_stats.clip_rate:.3f}", flush=True)

    except KeyboardInterrupt:
        log.info("[repl] KeyboardInterrupt")
    finally:
        # Journal first
        try:
            entry = prompt_journal_entry(session_start_s, args)
            if entry is not None:
                entry["notes"] = f"[repl policy={policy.name}] " + entry.get("notes", "")
                write_journal_entry(args.journal_path, entry, args, invocation)
        except Exception as e:
            log.warning("journal step failed: %s", e)

        # Return-to-startup ramp
        abort = {"abort": False, "ctrlc_count": 0}
        def _cleanup_sigint(_sig, _frame):
            abort["ctrlc_count"] += 1
            if abort["ctrlc_count"] == 1:
                log.warning("Ctrl-C in cleanup: aborting return ramp. ARMS WILL DROP.")
                abort["abort"] = True
            else:
                sys.exit(130)
        try: signal.signal(signal.SIGINT, _cleanup_sigint)
        except Exception: pass

        if left is not None and right is not None and startup_pose is not None \
                and not args.no_return_on_exit:
            try:
                log.info("Returning arms to startup pose (5.0s)...")
                ramp_to_pose(left, right, startup_pose,
                             duration_s=5.0, abort_flag=abort,
                             label="return-on-exit")
            except BaseException as e:
                log.warning("return ramp failed: %s", e)

        for c in (top, cam_l, cam_r):
            if c is not None:
                try: c.stop()
                except Exception as e: log.warning("camera stop: %s", e)
        for arm in (left, right):
            if arm is not None:
                try: arm.close()
                except Exception as e: log.warning("arm.close: %s", e)
        try: policy.close()
        except Exception as e: log.warning("policy.close: %s", e)

        print("[repl] done.", flush=True)


if __name__ == "__main__":
    main()
