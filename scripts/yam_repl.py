"""Interactive task REPL for bimanual YAM + MolmoAct2.

Setup once (cameras, arms, server warmup, move-to-ready), then loop:
  - Type a natural-language instruction at the prompt.
  - Press enter to start the attempt.
  - Press enter again to stop the attempt and ramp the arms back to the
    training-mean ready pose.
  - Record a journal entry (success / failure / unclear / skip).
  - Repeat.

Ctrl-C at any prompt -> teardown (ramp arms back to startup pose, close arms).

Run via the i2rt venv (same as yam_client.py):

    /home/andon/yam-tests/i2rt/.venv/bin/python scripts/yam_repl.py \
        --left-can can0 --right-can can1 \
        --left-gripper linear_4310 --right-gripper linear_4310 \
        --top-cam-v4l2 /dev/video12 \
        --left-cam-serial AAAA --right-cam-serial BBBB \
        --cam-width 640 --cam-height 360 \
        --server-url http://127.0.0.1:8202/act
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Optional

# yam_client must be importable. It applies install_sdk_lock_fix() at import,
# which has to happen BEFORE any DMChainCanInterface is created. So import it
# before we touch i2rt for anything else.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import requests  # noqa: E402

import yam_client as yc  # noqa: E402
from yam_client import (  # noqa: E402
    ARM_DOFS,
    DEFAULT_GRIPPER_STEP,
    DEFAULT_HORIZON_STRIDE,
    DEFAULT_JOURNAL_PATH,
    DEFAULT_MAX_STEP_RAD,
    DEFAULT_TRAIN_FPS,
    STATE_DIM,
    _journal_format_args,
    _journal_format_duration,
    init_arm,
    load_saved_config,
    load_training_mean_pose,
    log,
    make_camera,
    post_actions,
    ramp_to_pose,
    read_state,
    safe_command,
    trace,
)


# ---------------------------------------------------------------------------
# Per-attempt journal
# ---------------------------------------------------------------------------

def write_attempt_entry(
    path: str,
    attempt_idx: int,
    instruction: str,
    status: str,
    notes: str,
    stats: dict,
    args,
    invocation: str,
) -> None:
    """Append one markdown entry per task attempt to the journal.

    Schema is intentionally close to write_journal_entry's so the file stays
    readable as a mixed log of one-shot runs and REPL attempts.
    """
    md = []
    md.append("")
    md.append("---")
    md.append(f"## {stats['timestamp']} -- {status}  (repl attempt #{attempt_idx})")
    md.append("")
    md.append(f"**Instruction**: {instruction!r}")
    md.append("")
    if notes:
        md.append(f"**Notes**: {notes}")
        md.append("")
    md.append(f"**Duration**: {_journal_format_duration(stats['duration_s'])}"
              + (" (timed out)" if stats.get("timed_out") else ""))
    md.append("")
    md.append("**Attempt stats**:")
    md.append(f"- chunks: {stats['n_chunks']}")
    if stats["n_chunks"] > 0:
        md.append(f"- rtt_ms: mean {stats['mean_rtt_ms']:.0f}, "
                  f"p95 {stats['p95_rtt_ms']:.0f}, max {stats['max_rtt_ms']:.0f}")
        md.append(f"- horizon_arm_span (rad): mean {stats['mean_horizon_span']:.3f}, "
                  f"max {stats['max_horizon_span']:.3f}")
        md.append(f"- state_vs_a0 at boundaries (rad): mean {stats['mean_state_vs_a0']:.3f}, "
                  f"max {stats['max_state_vs_a0']:.3f}")
        if stats["max_possible_clip"] > 0:
            pct = 100.0 * stats["clipped_dim_steps"] / stats["max_possible_clip"]
            md.append(f"- clip rate: {stats['clipped_dim_steps']}/"
                      f"{stats['max_possible_clip']} dim-steps ({pct:.1f}%)")
    md.append("")
    md.append("**Command**:")
    md.append("```")
    md.append(invocation)
    md.append("```")
    md.append("")
    md.append("**Configuration**:")
    md.append(_journal_format_args(args))
    md.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[journal] wrote attempt #{attempt_idx} ({status}) to {path}", flush=True)


# ---------------------------------------------------------------------------
# Stop-on-enter helper
# ---------------------------------------------------------------------------

class EnterStopWatcher:
    """Spawn a daemon thread that polls stdin for an enter press; when seen,
    sets stop_flag['stop'] = True so the main control loop exits at its next
    check.

    Polling (select with a short timeout) instead of blocking input() means
    the watcher can be cancelled cleanly when the attempt ends for some
    OTHER reason (auto-timeout, exception, etc.). Without cancel(), a
    blocking watcher would still be waiting on stdin during the next prompt
    and would steal the operator's next keystroke -- which is what caused
    'f' to vanish at the outcome prompt and the attempt to log as 'skip'.
    """

    def __init__(self):
        self.stop_flag = {"stop": False}
        self._cancelled = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        import select
        def _wait():
            # Only poll a real TTY. If stdin is a pipe (CI/tests), bail.
            try:
                if not sys.stdin.isatty():
                    return
            except Exception:
                return
            while not self._cancelled:
                try:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
                except (OSError, ValueError):
                    return
                if self._cancelled:
                    return
                if not rlist:
                    continue
                try:
                    line = sys.stdin.readline()
                except Exception:
                    return
                if self._cancelled:
                    return
                self.stop_flag["stop"] = True
                if not line:
                    print("[stop] EOF on stdin -- stopping...", flush=True)
                else:
                    print("[stop] enter received, stopping after current chunk...",
                          flush=True)
                return
        self._thread = threading.Thread(target=_wait, daemon=True,
                                         name="enter-stop-watcher")
        self._thread.start()

    def cancel(self) -> None:
        """Tell the watcher to exit without consuming any further stdin.
        Call this from the main thread whenever the attempt ends for a
        reason other than the watcher firing (timeout, exception, etc.).
        Idempotent and safe to call after the watcher has already returned.
        """
        self._cancelled = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# One attempt
# ---------------------------------------------------------------------------

def run_one_attempt(
    args,
    left, right, top, cam_l, cam_r,
    instruction: str,
    attempt_idx: int,
    loop_t0: float,
    attempt_timeout_s: Optional[float] = None,
) -> dict:
    """Run the closed-loop control until the operator presses enter, then
    return a stats dict. Does NOT ramp the arms back (caller does that).

    attempt_timeout_s: wall-clock cap; if set (and >0), the loop ends
        automatically after that many seconds even without an enter press.
        Checked at chunk boundaries, so granularity is ~stride/train_fps
        (~0.3 s by default). Useful for unattended evals.

    Mirrors yam_client.main()'s inner loop (lines 946-1083) so behavior is
    identical to a one-shot run. The only differences:
      - stop signal comes from a stdin watcher, not a wall-clock or Ctrl-C
      - per-attempt stats are accumulated and returned
      - boundary_idx is local to this attempt
    """
    inner_dt = 1.0 / args.train_fps
    rtts: list[float] = []
    spans: list[float] = []
    state_vs_a0_arm_samples: list[float] = []
    clipped_dim_steps = 0
    max_possible_clip = 0
    n_chunks = 0

    last_chunk_tail: Optional[np.ndarray] = None
    boundary_idx = 0

    watcher = EnterStopWatcher()
    watcher.start()
    print(f"[attempt #{attempt_idx}] running -- press enter to stop", flush=True)

    attempt_start_s = time.time()

    # Wrap the inner loop in a broad except so any exception (camera grab,
    # arm I/O, etc.) is logged with a traceback before the attempt returns.
    # Without this, exceptions propagate up through main()'s try/finally
    # and the teardown's os._exit(0) eats the traceback -- the user sees
    # "running" then immediate ramp-back with no error message.
    has_timeout = attempt_timeout_s is not None and attempt_timeout_s > 0
    timed_out = False

    try:
      while not watcher.stop_flag["stop"]:
        if has_timeout and (time.time() - attempt_start_s) > attempt_timeout_s:
            log.info("attempt timeout (%.0fs) reached -- stopping",
                     attempt_timeout_s)
            timed_out = True
            break
        state = read_state(left, right)
        top_img = top.grab()
        left_img = cam_l.grab()
        right_img = cam_r.grab()
        yc._rr_log_observation(time.perf_counter() - loop_t0,
                               top_img, left_img, right_img, state)

        try:
            actions, rtt_ms = post_actions(
                args.server_url, top_img, left_img, right_img, state,
                instruction, args.num_steps, args.timeout_s,
            )
        except requests.RequestException as e:
            log.error("/act failed: %s -- stopping attempt", e)
            break

        n_chunks += 1
        rtts.append(rtt_ms)

        # Per-query diagnostic (same shape as yam_client logs).
        def _arm_delta_max(a_idx: int) -> float:
            d = actions[a_idx] - state
            return float(max(np.max(np.abs(d[:6])), np.max(np.abs(d[7:13]))))
        a0_d  = _arm_delta_max(0)
        a5_d  = _arm_delta_max(min(5,  actions.shape[0] - 1))
        a10_d = _arm_delta_max(min(10, actions.shape[0] - 1))
        a19_d = _arm_delta_max(min(19, actions.shape[0] - 1))
        a29_d = _arm_delta_max(actions.shape[0] - 1)
        horizon_range = actions.max(axis=0) - actions.min(axis=0)
        horizon_arm_span = float(max(np.max(horizon_range[:6]),
                                      np.max(horizon_range[7:13])))
        spans.append(horizon_arm_span)
        log.info(
            "/act rtt=%dms  arm |a[i]-state|_max @ i=0/5/10/19/29: "
            "%.3f/%.3f/%.3f/%.3f/%.3f rad  horizon_span=%.3f rad  "
            "L_grip[0,29]=%.2f,%.2f  R_grip[0,29]=%.2f,%.2f",
            rtt_ms, a0_d, a5_d, a10_d, a19_d, a29_d, horizon_arm_span,
            actions[0][6],  actions[-1][6],
            actions[0][13], actions[-1][13],
        )

        stride = max(1, args.horizon_stride)
        n_to_play = min(stride, actions.shape[0])
        yc._rr_log_inference(time.perf_counter() - loop_t0, actions,
                             executed_idx=0, rtt_ms=rtt_ms,
                             horizon_arm_span=horizon_arm_span)

        # Boundary telemetry.
        if last_chunk_tail is not None:
            arm_idx = np.r_[0:6, 7:13]
            a0 = actions[0]
            state_vs_a0_arm = float(np.max(np.abs(a0[arm_idx] - state[arm_idx])))
            tail_vs_a0_arm = float(np.max(np.abs(a0[arm_idx] - last_chunk_tail[arm_idx])))
            state_vs_a0_grip_l = float(abs(a0[6]  - state[6]))
            state_vs_a0_grip_r = float(abs(a0[13] - state[13]))
            state_vs_a0_arm_samples.append(state_vs_a0_arm)
            boundary_idx += 1
            log.info(
                "[boundary] #%d  state_vs_a0(arm)=%.3f rad  "
                "tail_vs_a0(arm)=%.3f rad  "
                "state_vs_a0(grip L,R)=%.2f,%.2f",
                boundary_idx, state_vs_a0_arm, tail_vs_a0_arm,
                state_vs_a0_grip_l, state_vs_a0_grip_r,
            )

        clipped_this_query = 0
        steps_this_query = 0
        for i in range(n_to_play):
            if watcher.stop_flag["stop"]:
                break
            step_start = time.perf_counter()
            desired = actions[i].astype(np.float32)
            if args.dry_run:
                log.info("dry-run action[%d]: %s", i,
                         np.array2string(desired, precision=3))
            else:
                state = read_state(left, right)
                _, n_clipped = safe_command(left, right, state, desired,
                                            args.max_step_rad, args.gripper_step)
                clipped_this_query += n_clipped
                steps_this_query += 1
            sleep_left = inner_dt - (time.perf_counter() - step_start)
            if sleep_left > 0:
                time.sleep(sleep_left)
            elif sleep_left < -0.050:
                log.warning("inner step overrun by %.1f ms (target %.1f ms)",
                            -sleep_left * 1000.0, inner_dt * 1000.0)

        if steps_this_query > 0 and (args.max_step_rad > 0 or args.gripper_step > 0):
            mp = STATE_DIM * steps_this_query
            clipped_dim_steps += clipped_this_query
            max_possible_clip += mp
            if clipped_this_query > 0:
                pct = 100.0 * clipped_this_query / mp
                log.info("clip: %d/%d dim-steps clipped (%.1f%%) "
                         "[--max-step-rad=%.3f --gripper-step=%.3f]",
                         clipped_this_query, mp, pct,
                         args.max_step_rad, args.gripper_step)

        if n_to_play > 0:
            last_chunk_tail = actions[n_to_play - 1].astype(np.float32).copy()
    except KeyboardInterrupt:
        # Let Ctrl-C continue propagating up to main()'s teardown.
        raise
    except Exception:
        # Log the traceback so a transient camera/arm hiccup doesn't vanish
        # into the os._exit(0) at the bottom of main()'s finally.
        log.exception("attempt #%d crashed inside control loop", attempt_idx)
    finally:
        # Critical: shut down the stdin watcher so it does not steal the
        # next keystroke at the outcome prompt. Idempotent if it already
        # fired (because the operator pressed enter).
        watcher.cancel()

    duration_s = time.time() - attempt_start_s
    rtts_np = np.asarray(rtts, dtype=np.float32) if rtts else np.zeros(0, dtype=np.float32)
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": duration_s,
        "n_chunks": n_chunks,
        "mean_rtt_ms": float(rtts_np.mean()) if rtts else 0.0,
        "p95_rtt_ms": float(np.percentile(rtts_np, 95)) if rtts else 0.0,
        "max_rtt_ms": float(rtts_np.max()) if rtts else 0.0,
        "mean_horizon_span": float(np.mean(spans)) if spans else 0.0,
        "max_horizon_span": float(np.max(spans)) if spans else 0.0,
        "mean_state_vs_a0": float(np.mean(state_vs_a0_arm_samples))
            if state_vs_a0_arm_samples else 0.0,
        "max_state_vs_a0": float(np.max(state_vs_a0_arm_samples))
            if state_vs_a0_arm_samples else 0.0,
        "clipped_dim_steps": clipped_dim_steps,
        "max_possible_clip": max_possible_clip,
        "timed_out": timed_out,
    }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def prompt_instruction(last: Optional[str]) -> Optional[str]:
    """Read an instruction line from the operator.

    Returns:
      str  -- the instruction to run
      None -- the operator wants to quit ('q'/'quit'/'exit', or EOF/Ctrl-C)
    """
    print("", flush=True)
    print("─" * 70, flush=True)
    if last is not None:
        print(f"instruction (enter to reuse last: {last!r})", flush=True)
    else:
        print("instruction (or 'quit' to exit)", flush=True)
    sys.stdout.flush()
    try:
        line = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if line.lower() in {"q", "quit", "exit"}:
        return None
    if not line:
        if last is None:
            print("(no previous instruction; type one or 'quit')", flush=True)
            return prompt_instruction(last)
        return last
    return line


def confirm_ready(instruction: str) -> bool:
    """Show the instruction and wait for one more enter before launching.
    Gives the operator a beat to position objects / step back.

    Returns False if the operator wants to skip this attempt.
    """
    print(f"\nabout to run: {instruction!r}", flush=True)
    print("[enter to start, 's' to skip this instruction, ctrl-c to quit]",
          flush=True)
    sys.stdout.flush()
    try:
        ans = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise
    return ans != "s"


def prompt_attempt_outcome() -> tuple[Optional[str], str]:
    """Ask the operator how the attempt went.

    Returns (status, notes). status is one of 's'/'f'/'u' mapped to
    'success'/'failure'/'unclear', or None if the user skipped (no journal).
    """
    print("", flush=True)
    print("How did it go?  [s]uccess / [f]ailure / [u]nclear / "
          "[enter=skip, no journal]", flush=True)
    sys.stdout.flush()
    try:
        choice = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None, ""
    if not choice:
        return None, ""
    status_map = {"s": "success", "f": "failure", "u": "unclear"}
    status = status_map.get(choice[:1])
    if status is None:
        print(f"[journal] unrecognized {choice!r}, skipping", flush=True)
        return None, ""
    try:
        notes = input("Notes (one line, optional)\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        notes = ""
    return status, notes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Defaults are sourced from yam_setup_config.json (populated by
    # identify_setup.py). Re-run identify_setup.py after any hardware swap.
    _cfg = load_saved_config()
    _gripper_default = _cfg.get("gripper", "linear_4310")

    p = argparse.ArgumentParser(
        description="Interactive task REPL for bimanual YAM + MolmoAct2"
    )
    # Hardware (mirrors yam_client flags so run_repl.sh is a drop-in).
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
    # Server.
    p.add_argument("--server-url", default="http://127.0.0.1:8202/act")
    p.add_argument("--timeout-s",        type=float, default=15.0)
    p.add_argument("--warmup-timeout-s", type=float, default=60.0)
    p.add_argument("--num-steps", type=int, default=10)
    # Policy execution.
    p.add_argument("--train-fps",      type=float, default=DEFAULT_TRAIN_FPS)
    p.add_argument("--horizon-stride", type=int,   default=DEFAULT_HORIZON_STRIDE)
    p.add_argument("--max-step-rad",   type=float, default=DEFAULT_MAX_STEP_RAD)
    p.add_argument("--gripper-step",   type=float, default=DEFAULT_GRIPPER_STEP)
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions instead of commanding the arms")
    # Reset behavior.
    p.add_argument("--ramp-duration-s", type=float, default=5.0,
                   help="Seconds for the move-to-ready and per-attempt reset ramps")
    p.add_argument("--no-return-on-exit", action="store_true",
                   help="DANGEROUS: skip the return-to-startup-pose ramp on exit")
    # Observability.
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--rerun-connect", default=None, metavar="HOST:PORT")
    p.add_argument("--rerun-save",    default=None, metavar="PATH")
    # Journal.
    p.add_argument("--journal-path", default=DEFAULT_JOURNAL_PATH)
    args = p.parse_args()

    # Same loud warning as yam_client when safety clips are off.
    if args.max_step_rad <= 0 and args.gripper_step <= 0:
        log.warning("=" * 70)
        log.warning("--max-step-rad=0 AND --gripper-step=0: clipping DISABLED")
        log.warning("=" * 70)

    invocation = os.environ.get("YAM_INVOCATION") or " ".join(sys.argv)

    # Rerun.
    rerun_requested = args.rerun or (args.rerun_save is not None)
    if rerun_requested:
        try:
            import rerun as rr
            yc._rr = rr
            rr.init("yam_repl", spawn=(args.rerun_connect is None))
            if args.rerun_connect:
                host, _, port = args.rerun_connect.partition(":")
                rr.connect_grpc(f"rerun+http://{host}:{port}/proxy")
            if args.rerun_save:
                rr.save(args.rerun_save)
                log.info("Rerun: saving recording to %s", args.rerun_save)
        except ImportError:
            log.error("--rerun requested but rerun-sdk not installed. Install with:"
                      " VIRTUAL_ENV=/home/andon/yam-tests/i2rt/.venv "
                      "uv pip install rerun-sdk")
            sys.exit(2)

    # Health-check.
    try:
        r = requests.get(args.server_url, timeout=3.0)
        r.raise_for_status()
        log.info("server health: %s", r.json())
    except Exception as e:
        log.error("server health check failed at %s: %s", args.server_url, e)
        sys.exit(2)

    # Cameras before arms (USB-storm-vs-CAN ordering -- see yam_client comment).
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
        # AE-settle pass: failures here are tolerated; the in-start warmup
        # already proved each camera can produce frames, and the first
        # real grab in the control loop has its own retry/timeout.
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

    # Ready pose: training-mean for the arms, keep startup gripper widths.
    ready_pose = load_training_mean_pose()
    ready_pose[6]  = startup_pose[6]
    ready_pose[13] = startup_pose[13]
    log.info("Ramping to training-mean ready pose (%.1fs)...", args.ramp_duration_s)
    ramp_to_pose(left, right, ready_pose, duration_s=args.ramp_duration_s,
                 label="initial move-to-ready")

    # Server warmup at the real image shape, captures CUDA graphs once.
    try:
        state = read_state(left, right)
        log.info("Warming up server (timeout=%.0fs)...", args.warmup_timeout_s)
        _wu_actions, _wu_rtt = post_actions(
            args.server_url, top.grab(), cam_l.grab(), cam_r.grab(), state,
            "warmup", args.num_steps, args.warmup_timeout_s,
        )
        log.info("Server warmup OK (rtt=%.0f ms, actions shape=%s)",
                 _wu_rtt, _wu_actions.shape)
    except Exception as e:
        log.error("Server warmup failed: %s. Continuing anyway.", e)

    loop_t0 = time.perf_counter()

    # ---- REPL ------------------------------------------------------------
    last_instruction: Optional[str] = None
    attempt_idx = 0

    print("\n" + "=" * 70, flush=True)
    print("YAM + MolmoAct2 REPL", flush=True)
    print("  - type an instruction and press enter", flush=True)
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

            # Reset to ready pose. ramp_to_pose ignores Ctrl-C without an
            # abort_flag, which is fine -- between attempts we always want
            # the reset to complete; Ctrl-C will take effect at the next
            # input() prompt.
            log.info("Resetting arms to ready pose (%.1fs)...",
                     args.ramp_duration_s)
            ramp_to_pose(left, right, ready_pose,
                         duration_s=args.ramp_duration_s, label="reset")

            try:
                status, notes = prompt_attempt_outcome()
            except KeyboardInterrupt:
                # Ctrl-C at outcome -> skip journal and tear down.
                raise
            if status is None:
                log.info("attempt #%d skipped (no journal entry)", attempt_idx)
            else:
                write_attempt_entry(args.journal_path, attempt_idx,
                                     instruction, status, notes, stats, args,
                                     invocation)
    except KeyboardInterrupt:
        log.info("Ctrl-C -- shutting down")
    finally:
        # Teardown: ramp arms to startup_pose, stop cameras, close arms.
        # Identical to yam_client.main()'s finally block.
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
                if abort["abort"]:
                    log.warning("return ramp was aborted -- arms may be mid-trajectory")
            except BaseException as e:
                log.warning("return-to-startup ramp failed: %s. ARMS MAY DROP.", e)
        elif args.no_return_on_exit:
            log.warning("--no-return-on-exit set: skipping return ramp. "
                        "ARMS WILL DROP if not in a stable rest pose.")

        log.info("Stopping cameras")
        for c in (top, cam_l, cam_r):
            if c is None:
                continue
            try: c.stop()
            except BaseException as e: log.warning("cam %s stop failed: %s", c.name, e)

        log.info("Closing arm SDKs")
        for arm in (left, right):
            if arm is None:
                continue
            try: arm.close()
            except BaseException as e: log.warning("arm.close() failed: %s", e)

        log.info("Done. Ran %d attempt(s).", attempt_idx)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
