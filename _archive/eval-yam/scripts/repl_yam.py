"""Multi-backend interactive REPL for bimanual YAM.

Same UX as molmoact2-setup/scripts/yam_repl.py -- type instruction, press
enter, evaluate, journal, repeat -- but speaks to one of three VLA servers
behind a backend abstraction:

  --policy molmoact2   HTTP+json_numpy  (allenai/MolmoAct2-BimanualYAM)
  --policy pi05        WebSocket+msgpack (jeqcho/pi05-yam-bimanual via openpi)
  --policy gr00t-n17   ZMQ+msgpack       (jeqcho/gr00t-n17-yam-bimanual via Isaac-GR00T)

The hardware path (cameras, arms, safety clipping, async fetcher, journal,
boundary diagnostics, return-on-exit ramp) is identical to molmoact2-setup's
REPL -- imported wholesale. Only the inference call site is replaced via
yam_backends.install_backend().

Run via the i2rt venv:

    /home/andon/yam-tests/i2rt/.venv/bin/python eval-yam/scripts/repl_yam.py \
        --policy molmoact2 \
        --server-url http://127.0.0.1:8202/act

The per-policy wrapper scripts (run_repl_*.sh) hide the --policy flag and
the policy-specific server connection flags.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from typing import Optional

# yam_client (in molmoact2-setup) applies install_sdk_lock_fix() at import,
# which MUST happen before any DMChainCanInterface is created. So import it
# before any other i2rt touch.
_HERE = os.path.dirname(os.path.abspath(__file__))
_MOLMOACT_SCRIPTS = os.path.normpath(
    os.path.join(_HERE, "..", "..", "molmoact2-setup", "scripts")
)
if not os.path.isdir(_MOLMOACT_SCRIPTS):
    raise RuntimeError(
        f"expected molmoact2-setup at {_MOLMOACT_SCRIPTS} -- this script "
        "imports yam_client/yam_repl from there as libraries"
    )
sys.path.insert(0, _MOLMOACT_SCRIPTS)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

import yam_client as yc  # noqa: E402  -- applies SDK lock fix at import
from yam_client import (  # noqa: E402
    DEFAULT_GRIPPER_STEP,
    DEFAULT_HORIZON_STRIDE,
    DEFAULT_JOURNAL_PATH,
    DEFAULT_MAX_STEP_RAD,
    DEFAULT_TRAIN_FPS,
    init_arm,
    load_saved_config,
    load_training_mean_pose,
    log,
    make_camera,
    ramp_to_pose,
    read_state,
    trace,
)
import yam_repl  # noqa: E402  -- for run_one_attempt, write_attempt_entry, prompts
from yam_repl import (  # noqa: E402
    confirm_ready,
    prompt_attempt_outcome,
    prompt_instruction,
    run_one_attempt,
    write_attempt_entry,
)

import yam_backends  # noqa: E402  -- our backends


# ---------------------------------------------------------------------------
# Helpers: journal-based instruction recall + interactive knob tuning
# ---------------------------------------------------------------------------

def _last_instruction_from_journal(path: str) -> Optional[str]:
    """Return the most-recent **Instruction** value in the journal, or None.

    Lets the REPL pre-fill the instruction prompt with the last thing run
    in any past session -- so re-running the same task on a new day is
    just press-enter at the prompt.
    """
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except (FileNotFoundError, OSError):
        return None
    import re
    matches = re.findall(r"\*\*Instruction\*\*:\s*'(.+?)'", text)
    return matches[-1] if matches else None


# The runtime knobs an operator might want to tune mid-session. Each entry:
# (args attr name, parser/'bool', one-line description). The parser is
# applied to the new value string before setattr.
_TUNABLE_KNOBS = [
    ("horizon_stride",  int,   "steps of each action chunk to play before re-query"),
    ("max_step_rad",    float, "per-tick joint cap (rad). 0 = disabled"),
    ("gripper_step",    float, "per-tick gripper cap (normalized). 0 = disabled"),
    ("train_fps",       float, "inner-loop tick rate (Hz)"),
    ("num_steps",       int,   "flow-matching steps (molmoact2 only)"),
    ("timeout_s",       float, "per-/act HTTP/WS/ZMQ timeout (s)"),
    ("ramp_duration_s", float, "ramp duration for ready/reset/return (s)"),
    ("dry_run",         "bool","True = print actions, do NOT command arms"),
]


def prompt_tune_knobs(args) -> None:
    """Show the current runtime knobs; let the operator edit any of them
    in place before this attempt. Mutates `args` so the new values:

      - apply IMMEDIATELY to the about-to-run attempt (the inner loop
        reads args.horizon_stride etc. at every iteration);
      - PERSIST to subsequent attempts in the same session;
      - get RECORDED in the journal (Configuration block dumps args).

    Press enter at the menu to skip and continue. 'q' to skip THIS
    instruction. Ctrl-C bubbles up like elsewhere in the REPL.
    """
    while True:
        print("", flush=True)
        print("tunable knobs  (num to edit, enter to continue, 'q' to skip this instruction):",
              flush=True)
        for i, (name, _parser, desc) in enumerate(_TUNABLE_KNOBS, 1):
            val = getattr(args, name)
            print(f"  {i}. {name:<18} = {val!r:<10}   {desc}", flush=True)
        sys.stdout.flush()
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not choice:
            return
        if choice in ("q", "quit", "skip"):
            raise _SkipInstruction()
        try:
            idx = int(choice) - 1
            name, parser, _desc = _TUNABLE_KNOBS[idx]
        except (ValueError, IndexError):
            print(f"  ?? unknown {choice!r}; enter 1-{len(_TUNABLE_KNOBS)} "
                  f"or blank to continue", flush=True)
            continue
        current = getattr(args, name)
        try:
            new_raw = input(f"  new {name} (curr {current!r}, blank=keep): ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not new_raw:
            continue
        try:
            if parser == "bool":
                new_val = new_raw.lower() in ("1", "true", "t", "yes", "y", "on")
            else:
                new_val = parser(new_raw)
        except ValueError as e:
            print(f"  invalid {name}: {e}", flush=True)
            continue
        setattr(args, name, new_val)
        print(f"  -> {name} = {new_val!r}", flush=True)


class _SkipInstruction(Exception):
    """Internal sentinel: operator typed 'q' in the knob-tune menu."""


# ---------------------------------------------------------------------------
# Per-policy argparse defaults
# ---------------------------------------------------------------------------

def _policy_defaults(policy: str) -> dict:
    """Server-connection defaults per policy."""
    return {
        "molmoact2": {"server_url": "http://127.0.0.1:8202/act"},
        "pi05":      {"server_host": "127.0.0.1", "server_port": 8000},
        "gr00t-n17": {"server_host": "127.0.0.1", "server_port": 5556},
    }[policy]


def _build_backend(args) -> yam_backends.Backend:
    if args.policy == "molmoact2":
        return yam_backends.MolmoActHTTPBackend(server_url=args.server_url)
    if args.policy == "pi05":
        return yam_backends.Pi05WebsocketBackend(host=args.server_host,
                                                  port=args.server_port)
    if args.policy == "gr00t-n17":
        return yam_backends.Gr00tZmqBackend(host=args.server_host,
                                             port=args.server_port)
    raise ValueError(f"Unknown --policy {args.policy!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _cfg = load_saved_config()
    _gripper_default = _cfg.get("gripper", "linear_4310")

    p = argparse.ArgumentParser(
        description="Multi-backend REPL for bimanual YAM "
                    "(MolmoAct2 / Pi-0.5 / GR00T-N1.7)"
    )
    # Policy selector. Required so a wrapper script must set it -- avoids
    # accidentally hitting an unexpected backend.
    p.add_argument("--policy", required=True,
                   choices=["molmoact2", "pi05", "gr00t-n17"],
                   help="Which VLA server to talk to.")
    # Server connection -- meaning depends on --policy. We accept both
    # --server-url (HTTP backends) and --server-host/--server-port (ZMQ/WS).
    # The wrapper scripts set the right one.
    p.add_argument("--server-url", default=None,
                   help="(--policy molmoact2) Full HTTP endpoint URL.")
    p.add_argument("--server-host", default="127.0.0.1",
                   help="(--policy pi05 | gr00t-n17) Server hostname.")
    p.add_argument("--server-port", type=int, default=None,
                   help="(--policy pi05 | gr00t-n17) Server port.")
    # Hardware (mirrors yam_repl flags).
    p.add_argument("--left-can",  default=_cfg.get("left_can",  "can0"))
    p.add_argument("--right-can", default=_cfg.get("right_can", "can1"))
    p.add_argument("--left-gripper",  default=_gripper_default,
                   choices=["crank_4310", "linear_3507", "linear_4310", "flexible_4310"])
    p.add_argument("--right-gripper", default=_gripper_default,
                   choices=["crank_4310", "linear_3507", "linear_4310", "flexible_4310"])
    p.add_argument("--top-cam-serial",   default=_cfg.get("top_cam_serial"))
    p.add_argument("--top-cam-v4l2",     default=_cfg.get("top_cam_v4l2"))
    p.add_argument("--left-cam-serial",  default=_cfg.get("left_cam_serial"))
    p.add_argument("--left-cam-v4l2",    default=_cfg.get("left_cam_v4l2"))
    p.add_argument("--right-cam-serial", default=_cfg.get("right_cam_serial"))
    p.add_argument("--right-cam-v4l2",   default=_cfg.get("right_cam_v4l2"))
    p.add_argument("--cam-width",  type=int, default=424)
    p.add_argument("--cam-height", type=int, default=240)
    p.add_argument("--cam-fps",    type=int, default=30)
    # Inference / policy execution.
    p.add_argument("--timeout-s",        type=float, default=15.0)
    p.add_argument("--warmup-timeout-s", type=float, default=60.0)
    p.add_argument("--num-steps", type=int, default=10,
                   help="Flow-matching steps (used by MolmoAct2 only; "
                        "ignored by pi05/gr00t-n17 servers).")
    p.add_argument("--train-fps",      type=float, default=DEFAULT_TRAIN_FPS)
    p.add_argument("--horizon-stride", type=int,   default=DEFAULT_HORIZON_STRIDE)
    p.add_argument("--max-step-rad",   type=float, default=DEFAULT_MAX_STEP_RAD)
    p.add_argument("--gripper-step",   type=float, default=DEFAULT_GRIPPER_STEP)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ramp-duration-s", type=float, default=5.0)
    p.add_argument("--no-return-on-exit", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--rerun-connect", default=None, metavar="HOST:PORT")
    p.add_argument("--rerun-save",    default=None, metavar="PATH")
    p.add_argument("--journal-path", default=DEFAULT_JOURNAL_PATH)
    args = p.parse_args()

    # Apply per-policy server defaults so wrapper scripts only need to
    # override what they want.
    pd = _policy_defaults(args.policy)
    if args.policy == "molmoact2":
        if args.server_url is None:
            args.server_url = pd["server_url"]
    else:
        if args.server_port is None:
            args.server_port = pd["server_port"]

    # Make a server-url string for argparse args dumps / journal headers,
    # even for non-HTTP backends -- so the journal entry reads sensibly.
    if args.policy != "molmoact2" and args.server_url is None:
        proto = "ws" if args.policy == "pi05" else "tcp"
        args.server_url = f"{proto}://{args.server_host}:{args.server_port}"

    # --- Build backend and route post_actions through it -----------------
    backend = _build_backend(args)
    yam_backends.install_backend(backend)

    if args.max_step_rad <= 0 and args.gripper_step <= 0:
        log.warning("=" * 70)
        log.warning("--max-step-rad=0 AND --gripper-step=0: clipping DISABLED")
        log.warning("=" * 70)

    invocation = os.environ.get("YAM_INVOCATION") or " ".join(sys.argv)

    # Rerun (optional).
    rerun_requested = args.rerun or (args.rerun_save is not None)
    if rerun_requested:
        # AUTO-SAVE: if --rerun was passed but no --rerun-save path was
        # specified, default to writing the .rrd to eval-yam/logs/rrd/.
        # Cost: ~300-900 MB per active minute. Without this, joint
        # histories die with the viewer when the REPL exits.
        if args.rerun and not args.rerun_save:
            from datetime import datetime as _dt
            rrd_dir = os.path.join(os.path.dirname(_HERE), "logs", "rrd")
            os.makedirs(rrd_dir, exist_ok=True)
            args.rerun_save = os.path.join(
                rrd_dir,
                f"{_dt.now().strftime('%Y-%m-%d_%H%M%S')}_{args.policy}.rrd",
            )
            log.info("AUTO-SAVING Rerun recording to %s", args.rerun_save)
        try:
            import rerun as rr
            yc._rr = rr
            rr.init(f"yam_repl_{args.policy}")
            # MULTI-SINK setup. In Rerun 0.32, calling rr.save() after
            # rr.init(spawn=True) REPLACES the viewer's gRPC sink with
            # a file sink -- the live viewer goes dark while data is
            # written to disk. The correct way to have BOTH is rr.set_sinks
            # with explicit GrpcSink + FileSink. Note that set_sinks
            # "replaces existing sinks" so we must NOT use spawn=True
            # in rr.init; we call rr.spawn() to launch the viewer process
            # without connecting, then set_sinks wires everything.
            sinks = []
            if args.rerun_connect:
                host, _, port = args.rerun_connect.partition(":")
                sinks.append(rr.GrpcSink(url=f"rerun+http://{host}:{port}/proxy"))
            else:
                # Spawn the viewer process; connect=False so we don't
                # bind a sink that set_sinks would then override and
                # leave dangling.
                rr.spawn(connect=False)
                sinks.append(rr.GrpcSink())  # connects to default localhost:9876
            if args.rerun_save:
                sinks.append(rr.FileSink(args.rerun_save))
            rr.set_sinks(*sinks)
        except ImportError:
            log.error("--rerun requested but rerun-sdk not installed.")
            sys.exit(2)

    # Health check via backend (transport-aware).
    try:
        meta = backend.health_check(timeout_s=3.0)
        log.info("[%s] server health: %s", args.policy, meta)
    except Exception as e:
        log.error("[%s] server health check failed: %s", args.policy, e)
        sys.exit(2)

    # Stash the server's self-reported identity onto args so the journal's
    # _journal_format_args() dumps it into every attempt's Configuration
    # block. Critical when multiple checkpoints can be served at the same
    # port over the lifetime of a project (gr00t base vs YAM finetune vs
    # future finetunes; pi05 yam_pi05 vs other configs). Without this,
    # CSV rows recorded today are indistinguishable from CSV rows recorded
    # against a different checkpoint of the same policy class.
    #   - molmoact2 server returns repo_id / dtype / norm_tag / num_cameras
    #   - gr00t PolicyServer ping returns only {status, message}; capture
    #     transport at minimum
    #   - pi05 openpi metadata may include policy.config / policy.dir
    args.server_repo_id  = meta.get("repo_id",  "unknown")
    args.server_dtype    = meta.get("dtype",    "unknown")
    args.server_norm_tag = meta.get("norm_tag", "unknown")
    args.server_meta     = repr(meta)

    # Cameras before arms (USB-storm-vs-CAN ordering).
    top = cam_l = cam_r = None
    left = right = None
    try:
        cam_kw = dict(width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
        trace(f"building cameras at {args.cam_width}x{args.cam_height}/{args.cam_fps}fps")
        top   = make_camera("top",   args.top_cam_serial,   args.top_cam_v4l2,   **cam_kw)
        cam_l = make_camera("left",  args.left_cam_serial,  args.left_cam_v4l2,  **cam_kw)
        cam_r = make_camera("right", args.right_cam_serial, args.right_cam_v4l2, **cam_kw)
        for c in (top, cam_l, cam_r):
            c.start()
        for _ in range(3):
            for c in (top, cam_l, cam_r):
                try: c.grab()
                except Exception as e: log.warning("settle: %s.grab() failed: %s", c.name, e)
        trace("cameras streaming, USB quiet -- safe to init arms")
    except Exception:
        for c in (top, cam_l, cam_r):
            if c is not None:
                try: c.stop()
                except Exception: pass
        raise

    trace("init LEFT arm")
    left = init_arm(args.left_can, args.left_gripper)
    trace("init RIGHT arm")
    right = init_arm(args.right_can, args.right_gripper)
    trace("both arms initialized")

    startup_pose = read_state(left, right)
    log.info("Captured startup pose for return-on-exit: %s",
             np.array2string(startup_pose, precision=3))

    # NOTE: the training-mean ready pose comes from the MolmoAct2 norm_stats.
    # For pi05 / gr00t-n17 it's only an approximation, but it's still a
    # reasonable in-distribution starting pose: all three policies were
    # trained on the same AllenAI bimanual-YAM dataset family, so the joint
    # angle centroid is approximately shared. Gripper widths stay at startup.
    ready_pose = load_training_mean_pose()
    ready_pose[6]  = startup_pose[6]
    ready_pose[13] = startup_pose[13]
    log.info("Ramping to training-mean ready pose (%.1fs)...", args.ramp_duration_s)
    ramp_to_pose(left, right, ready_pose, duration_s=args.ramp_duration_s,
                 label="initial move-to-ready")

    # Server warmup at the real image shape.
    try:
        state = read_state(left, right)
        log.info("Warming up server (timeout=%.0fs)...", args.warmup_timeout_s)
        # The patched post_actions ignores server_url; we keep the same call
        # shape so warmup runs through the active backend.
        _wu_actions, _wu_rtt = yc.post_actions(
            args.server_url, top.grab(), cam_l.grab(), cam_r.grab(), state,
            "warmup", args.num_steps, args.warmup_timeout_s,
        )
        log.info("Server warmup OK (rtt=%.0f ms, actions shape=%s)",
                 _wu_rtt, _wu_actions.shape)
    except Exception as e:
        log.error("Server warmup failed: %s. Continuing anyway.", e)

    loop_t0 = time.perf_counter()
    # Pre-load the most-recent instruction across ALL past sessions so the
    # first prompt at REPL launch is "press enter to reuse last: '...'".
    # Convenient for re-running yesterday's task without retyping.
    last_instruction: Optional[str] = _last_instruction_from_journal(args.journal_path)
    if last_instruction:
        log.info("Pre-loaded last instruction from %s: %r",
                 args.journal_path, last_instruction)
    attempt_idx = 0

    print("\n" + "=" * 70, flush=True)
    print(f"YAM REPL  |  policy={args.policy}", flush=True)
    print("  - type an instruction (or press enter to reuse last)", flush=True)
    print("  - then the knob-tuning menu appears -- num to edit, enter to start", flush=True)
    print("  - press enter again to stop the attempt and reset", flush=True)
    print("  - 's' at the outcome prompt skips the journal", flush=True)
    print("  - 'quit' (or ctrl-c) at the instruction prompt to exit", flush=True)
    print("=" * 70, flush=True)

    try:
        while True:
            instruction = prompt_instruction(last_instruction)
            if instruction is None:
                break
            last_instruction = instruction

            # Knob-tuning menu: lets the operator edit per-tick caps,
            # stride, etc. before EACH attempt. Mutations persist across
            # attempts in this session and are recorded in the journal's
            # Configuration block automatically.
            try:
                prompt_tune_knobs(args)
            except _SkipInstruction:
                log.info("skipped instruction at knob-tune menu; back to prompt")
                continue
            except (EOFError, KeyboardInterrupt):
                break

            try:
                if not confirm_ready(instruction):
                    log.info("skipped this instruction; back to prompt")
                    continue
            except (EOFError, KeyboardInterrupt):
                break

            attempt_idx += 1
            try:
                stats = run_one_attempt(
                    args, left, right, top, cam_l, cam_r,
                    instruction, attempt_idx, loop_t0,
                )
            except KeyboardInterrupt:
                log.warning("Ctrl-C during attempt -- stopping and tearing down")
                raise

            log.info("Resetting arms to ready pose (%.1fs)...",
                     args.ramp_duration_s)
            ramp_to_pose(left, right, ready_pose,
                         duration_s=args.ramp_duration_s, label="reset")

            try:
                status, notes = prompt_attempt_outcome()
            except KeyboardInterrupt:
                raise
            if status is None:
                log.info("attempt #%d skipped (no journal entry)", attempt_idx)
            else:
                tagged_notes = f"[policy={args.policy}] " + (notes or "")
                write_attempt_entry(args.journal_path, attempt_idx,
                                    instruction, status, tagged_notes, stats,
                                    args, invocation)
    except KeyboardInterrupt:
        log.info("Ctrl-C -- shutting down")
    finally:
        abort = {"abort": False, "ctrlc_count": 0}
        def _cleanup_sigint(_sig, _frame):
            abort["ctrlc_count"] += 1
            if abort["ctrlc_count"] == 1:
                log.warning("Ctrl-C in cleanup: aborting return-ramp. "
                            "ARMS WILL DROP. Ctrl-C again to hard-exit.")
                abort["abort"] = True
            else:
                os._exit(130)
        try:
            signal.signal(signal.SIGINT, _cleanup_sigint)
        except Exception:
            pass

        if left is not None and right is not None and not args.no_return_on_exit:
            try:
                log.info("Returning arms to startup pose (%.1fs ramp)...",
                         args.ramp_duration_s)
                ramp_to_pose(left, right, startup_pose,
                             duration_s=args.ramp_duration_s,
                             abort_flag=abort, label="return-on-exit")
            except BaseException as e:
                log.warning("return-to-startup ramp failed: %s. ARMS MAY DROP.", e)
        elif args.no_return_on_exit:
            log.warning("--no-return-on-exit set: skipping return ramp. "
                        "ARMS WILL DROP if not in a stable rest pose.")

        log.info("Stopping cameras")
        for c in (top, cam_l, cam_r):
            if c is None: continue
            try: c.stop()
            except BaseException as e: log.warning("cam %s stop failed: %s", c.name, e)

        log.info("Closing arm SDKs")
        for arm in (left, right):
            if arm is None: continue
            try: arm.close()
            except BaseException as e: log.warning("arm.close() failed: %s", e)

        log.info("Done. Ran %d attempt(s) with policy=%s.", attempt_idx, args.policy)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
