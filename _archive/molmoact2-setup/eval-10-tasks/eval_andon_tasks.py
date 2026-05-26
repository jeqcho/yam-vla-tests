"""Andon 10-task evaluation harness for MolmoAct2 on bimanual YAM.

Runs each of 10 tasks N times (default 3), with a human operator resetting
the scene between attempts and pressing enter to advance.

Per-attempt flow:
  1. Show "Task X/10, attempt Y/N: <prompt>".
  2. Operator resets scene, types enter to start (or 's' to skip, 'q' to abort).
  3. Control loop runs (same plumbing as yam_repl).
  4. Operator presses enter again to stop.
  5. Arms ramp to training-mean ready pose.
  6. Operator records outcome [s]uccess / [f]ailure / [u]nclear / enter=skip.
  7. Append to journal.md AND results CSV.

Setup (cameras, arms, server warmup, ready-ramp) is done once at start, and
teardown (return-to-startup ramp, close arms) once at end. Ctrl-C at any
prompt triggers teardown.

Reference: experiments/8-ten-andon-tasks/reports/10-tasks.md in the
robotics-task-horizon repo.

Run via the i2rt venv:

    /home/andon/yam-tests/i2rt/.venv/bin/python eval-10-tasks/eval_andon_tasks.py

Or via the wrapper:

    ./eval-10-tasks/run_eval.sh
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.normpath(os.path.join(_HERE, "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import numpy as np  # noqa: E402
import requests  # noqa: E402

# yam_client first -- it applies install_sdk_lock_fix at import.
import yam_client as yc  # noqa: E402
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
    post_actions,
    ramp_to_pose,
    read_state,
    trace,
)
import yam_repl  # noqa: E402  -- reuse run_one_attempt + write_attempt_entry


# The 10 Andon tasks (verbatim from
# reference/robotics-task-horizon/experiments/8-ten-andon-tasks/reports/10-tasks.md).
# Edit a prompt here if you want to A/B different phrasings.
TASKS: list[str] = [
    "Stack two orange blocks vertically",
    "Put the orange cube into the tape roll",
    "Put the knife in the box",
    "Put the apple on the plate",
    "Stack three cups",
    "Place pen on notebook",
    "Fold this t-shirt",
    "Close the lid of the box",
    "Rotate the hex key on the screw one full turn clockwise",
    "Put the electric plug into the socket",
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def prompt_ready(task_idx: int, attempt: int, n_attempts: int,
                 instruction: str) -> str:
    """Return one of: 'go' (start attempt), 'skip' (skip this attempt),
    'quit' (abort eval).
    """
    print("\n" + "=" * 70, flush=True)
    print(f"Task {task_idx + 1}/{len(TASKS)}  |  attempt {attempt}/{n_attempts}",
          flush=True)
    print(f"  prompt: {instruction!r}", flush=True)
    print("=" * 70, flush=True)
    print("[enter to start, 's' to skip this attempt, 'q' to abort eval]",
          flush=True)
    sys.stdout.flush()
    try:
        ans = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "quit"
    if ans in {"q", "quit", "exit"}:
        return "quit"
    if ans == "s" or ans == "skip":
        return "skip"
    return "go"


def prompt_outcome() -> tuple[Optional[str], str]:
    """Returns one of:
      ("success"/"failure"/"unclear", notes)  -- log this attempt
      ("redo", "")                            -- discard, rerun same attempt
      (None, "")                              -- skip (CSV "skip" row, no journal)
    """
    print("\nHow did it go?", flush=True)
    print("  [s]uccess / [f]ailure / [u]nclear  -- log this attempt", flush=True)
    print("  [r] redo this attempt              -- discard rollout, prompt-start same attempt again", flush=True)
    print("  [enter] skip                       -- CSV 'skip' row, no journal", flush=True)
    sys.stdout.flush()
    try:
        choice = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None, ""
    if not choice:
        return None, ""
    if choice[:1] == "r":
        return "redo", ""
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
# Resume support
# ---------------------------------------------------------------------------

def _load_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_resumable_session(
    results_dir: str | os.PathLike,
    selected_tasks: list[int],
    n_attempts: int,
) -> Optional[tuple[Path, list[dict], set[tuple[int, int]], set[tuple[int, int]]]]:
    """Look for the most-recent results CSV that still has work left under
    the current --tasks / --attempts selection.

    Returns (csv_path, prior_rows, done_set, remaining_set) where:
      - done_set is the {(task_num, attempt)} pairs already in the CSV
      - remaining_set is target - done

    Target is built from the CURRENT CLI flags, so if you resume with a
    different --tasks / --attempts the script just resumes the intersection.

    Returns None if there is no eligible session (no CSVs, or all of them
    are already complete for the current selection).
    """
    rd = Path(results_dir)
    if not rd.exists():
        return None
    csvs = sorted(rd.glob("results_*.csv"), reverse=True)
    if not csvs:
        return None
    # Only consider the single most-recent CSV. If that one is already
    # complete for the current selection, do NOT fall back to older
    # sessions -- a finished eval shouldn't pop a resume prompt.
    path = csvs[0]
    try:
        rows = _load_csv_rows(path)
    except Exception:
        return None
    done: set[tuple[int, int]] = set()
    for r in rows:
        try:
            done.add((int(r["task_num"]), int(r["attempt"])))
        except (KeyError, ValueError):
            continue
    target = {(t + 1, a) for t in selected_tasks for a in range(1, n_attempts + 1)}
    remaining = target - done
    if not remaining:
        return None
    return path, rows, done, remaining


def prompt_resume_choice(path: Path, done: set, remaining: set) -> str:
    """Return 'continue', 'new', or 'quit'."""
    total = len(done) + len(remaining)
    print("\n" + "=" * 70, flush=True)
    print(f"Found previous eval session: {path.name}", flush=True)
    print(f"  {len(done)}/{total} attempts done, {len(remaining)} remaining "
          f"under current --tasks / --attempts", flush=True)
    print("=" * 70, flush=True)
    print("[c] continue  -- append to that CSV, skip already-done attempts",
          flush=True)
    print("[n] new       -- start fresh, new CSV file",     flush=True)
    print("[q] quit      -- exit without touching hardware", flush=True)
    sys.stdout.flush()
    while True:
        try:
            ans = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "quit"
        if ans in {"c", "continue"}: return "continue"
        if ans in {"n", "new"}:      return "new"
        if ans in {"q", "quit", "exit"}: return "quit"
        print("Please answer c / n / q.", flush=True)


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

class ResultsCsv:
    """Per-attempt results table. Rows are written immediately so a crash
    mid-eval doesn't lose data.
    """

    FIELDS = [
        "timestamp", "task_num", "task_name", "attempt", "status",
        "duration_s", "timed_out", "n_chunks", "mean_rtt_ms", "max_rtt_ms",
        "mean_horizon_span", "max_state_vs_a0", "clip_pct", "notes",
    ]

    def __init__(self, path: Path):
        self.path = path
        new_file = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if new_file:
            self._writer.writeheader()
            self._fh.flush()

    def add(self, task_num: int, task_name: str, attempt: int,
            status: str, notes: str, stats: dict) -> None:
        clip_pct = (100.0 * stats["clipped_dim_steps"] /
                    stats["max_possible_clip"]) if stats["max_possible_clip"] else 0.0
        self._writer.writerow({
            "timestamp": stats["timestamp"],
            "task_num":  task_num,
            "task_name": task_name,
            "attempt":   attempt,
            "status":    status,
            "duration_s":     f"{stats['duration_s']:.1f}",
            "timed_out":      "1" if stats.get("timed_out") else "0",
            "n_chunks":       stats["n_chunks"],
            "mean_rtt_ms":    f"{stats['mean_rtt_ms']:.0f}",
            "max_rtt_ms":     f"{stats['max_rtt_ms']:.0f}",
            "mean_horizon_span": f"{stats['mean_horizon_span']:.3f}",
            "max_state_vs_a0":   f"{stats['max_state_vs_a0']:.3f}",
            "clip_pct":          f"{clip_pct:.1f}",
            "notes":             notes,
        })
        self._fh.flush()

    def close(self) -> None:
        try: self._fh.close()
        except Exception: pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Andon 10-task eval for MolmoAct2 on YAM")
    _cfg = load_saved_config()
    _gripper_default = _cfg.get("gripper", "linear_4310")
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
    p.add_argument("--dry-run", action="store_true")
    # Eval-specific.
    p.add_argument("-n", "--attempts", type=int, default=3,
                   help="attempts per task")
    p.add_argument("--tasks", default=None,
                   help="comma-separated 1-based task indices to run "
                        "(default: all 10). e.g. --tasks 1,2,5")
    p.add_argument("--attempt-timeout-s", type=float, default=60.0,
                   help="wall-clock cap per attempt; the rollout auto-stops "
                        "and ramps to ready after this many seconds, even "
                        "without an enter press. Pass 0 (or negative) to "
                        "disable -- operator must press enter every time. "
                        "Default 60.")
    # Reset behavior.
    p.add_argument("--ramp-duration-s", type=float, default=5.0)
    p.add_argument("--no-return-on-exit", action="store_true")
    # Observability.
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--rerun-connect", default=None, metavar="HOST:PORT")
    p.add_argument("--rerun-save",    default=None, metavar="PATH")
    # Outputs.
    p.add_argument("--journal-path", default=DEFAULT_JOURNAL_PATH)
    p.add_argument("--results-dir",
                   default=os.path.join(_HERE, "results"),
                   help="directory for per-session CSV results")
    args = p.parse_args()

    # Task selection.
    if args.tasks:
        try:
            selected = sorted({int(x) - 1 for x in args.tasks.split(",") if x.strip()})
        except ValueError:
            print(f"--tasks must be comma-separated integers, got {args.tasks!r}",
                  file=sys.stderr)
            sys.exit(2)
        for idx in selected:
            if not 0 <= idx < len(TASKS):
                print(f"task index {idx+1} out of range 1..{len(TASKS)}",
                      file=sys.stderr)
                sys.exit(2)
    else:
        selected = list(range(len(TASKS)))

    # Resume prompt: do this BEFORE any hardware bring-up so 'quit' is cheap.
    resume_path: Optional[Path] = None
    prior_rows: list[dict] = []
    done_set: set[tuple[int, int]] = set()
    prev = find_resumable_session(args.results_dir, selected, args.attempts)
    if prev is not None:
        path, rows, done, remaining = prev
        choice = prompt_resume_choice(path, done, remaining)
        if choice == "quit":
            print("Quit before hardware setup. No changes made.", flush=True)
            sys.exit(0)
        if choice == "continue":
            resume_path = path
            prior_rows = rows
            done_set = done
            log.info("Resuming session %s (%d already done, %d remaining)",
                     resume_path.name, len(done), len(remaining))

    invocation = os.environ.get("YAM_INVOCATION") or " ".join(sys.argv)

    # Rerun (optional).
    if args.rerun or args.rerun_save:
        try:
            import rerun as rr
            yc._rr = rr
            rr.init("yam_eval_10tasks", spawn=(args.rerun_connect is None))
            if args.rerun_connect:
                host, _, port = args.rerun_connect.partition(":")
                rr.connect_grpc(f"rerun+http://{host}:{port}/proxy")
            if args.rerun_save:
                rr.save(args.rerun_save)
        except ImportError:
            log.error("--rerun requested but rerun-sdk not installed.")
            sys.exit(2)

    # Health-check server.
    try:
        r = requests.get(args.server_url, timeout=3.0); r.raise_for_status()
        log.info("server health: %s", r.json())
    except Exception as e:
        log.error("server health check failed at %s: %s", args.server_url, e)
        sys.exit(2)

    # Cameras before arms.
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
        trace("cameras streaming -- safe to init arms")
    except Exception:
        for c in (top, cam_l, cam_r):
            if c is not None:
                try: c.stop()
                except Exception: pass
        raise

    left = init_arm(args.left_can, args.left_gripper)
    right = init_arm(args.right_can, args.right_gripper)

    startup_pose = read_state(left, right)
    log.info("Captured startup pose: %s",
             np.array2string(startup_pose, precision=3))

    ready_pose = load_training_mean_pose()
    ready_pose[6]  = startup_pose[6]
    ready_pose[13] = startup_pose[13]
    log.info("Ramping to training-mean ready pose (%.1fs)...",
             args.ramp_duration_s)
    ramp_to_pose(left, right, ready_pose, duration_s=args.ramp_duration_s,
                 label="initial move-to-ready")

    # Server warmup at real image shape.
    try:
        state = read_state(left, right)
        log.info("Warming up server (timeout=%.0fs)...", args.warmup_timeout_s)
        _wu_actions, _wu_rtt = post_actions(
            args.server_url, top.grab(), cam_l.grab(), cam_r.grab(), state,
            "warmup", args.num_steps, args.warmup_timeout_s,
        )
        log.info("Server warmup OK (rtt=%.0f ms)", _wu_rtt)
    except Exception as e:
        log.error("Server warmup failed: %s. Continuing anyway.", e)

    # Results CSV: reuse the resumed path if continuing, else a fresh
    # timestamped file. ResultsCsv opens in append mode so either works.
    session_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if resume_path is not None:
        results_path = resume_path
    else:
        results_path = Path(args.results_dir) / f"results_{session_ts}.csv"
    results = ResultsCsv(results_path)
    log.info("Writing per-attempt results to %s", results_path)

    loop_t0 = time.perf_counter()
    global_attempt = 0
    tallies: dict[int, dict[str, int]] = {
        i: {"success": 0, "failure": 0, "unclear": 0, "skip": 0}
        for i in selected
    }
    # Pre-populate tallies from any prior rows we're resuming over, so the
    # end-of-session summary reflects the full session, not just this run.
    for r in prior_rows:
        try:
            t_idx = int(r["task_num"]) - 1
        except (KeyError, ValueError):
            continue
        if t_idx not in tallies:
            continue
        status = r.get("status", "")
        if status in tallies[t_idx]:
            tallies[t_idx][status] += 1
    aborted = False

    total_target = len(selected) * args.attempts
    remaining_target = total_target - len(done_set)
    print("\n" + "#" * 70, flush=True)
    print(f"# Andon 10-task eval  |  {len(selected)} task(s) x {args.attempts} attempt(s) "
          f"= {total_target} total", flush=True)
    if done_set:
        print(f"# Resuming: {len(done_set)} already done, {remaining_target} remaining",
              flush=True)
    print(f"# Session: {session_ts}", flush=True)
    print("#" * 70, flush=True)

    try:
        for task_idx in selected:
            instruction = TASKS[task_idx]
            attempt = 1
            # While loop (not for) so 'redo' can repeat the same attempt
            # without advancing the counter or burning a CSV slot.
            while attempt <= args.attempts:
                if (task_idx + 1, attempt) in done_set:
                    log.info("task %d attempt %d already in resumed CSV; skipping",
                             task_idx + 1, attempt)
                    attempt += 1
                    continue
                action = prompt_ready(task_idx, attempt, args.attempts, instruction)
                if action == "quit":
                    aborted = True
                    raise KeyboardInterrupt
                if action == "skip":
                    log.info("task %d attempt %d skipped", task_idx + 1, attempt)
                    tallies[task_idx]["skip"] += 1
                    attempt += 1
                    continue

                global_attempt += 1
                stats = yam_repl.run_one_attempt(
                    args, left, right, top, cam_l, cam_r,
                    instruction, global_attempt, loop_t0,
                    attempt_timeout_s=args.attempt_timeout_s,
                )

                log.info("Resetting arms to ready pose (%.1fs)...",
                         args.ramp_duration_s)
                ramp_to_pose(left, right, ready_pose,
                             duration_s=args.ramp_duration_s, label="reset")

                status, notes = prompt_outcome()
                if status == "redo":
                    log.info("operator requested REDO of task %d attempt %d "
                             "-- rollout discarded, no journal/CSV entry",
                             task_idx + 1, attempt)
                    # Don't advance attempt; loop re-runs the same one.
                    continue
                if status is None:
                    log.info("attempt skipped from journal")
                    tallies[task_idx]["skip"] += 1
                    results.add(task_idx + 1, instruction, attempt,
                                "skip", notes, stats)
                    attempt += 1
                else:
                    tallies[task_idx][status] += 1
                    # Tag the journal entry with the task number so the
                    # full markdown record stays readable as a mix of REPL
                    # and eval entries.
                    tagged_notes = f"[eval task {task_idx + 1}/{len(TASKS)}] " + (notes or "")
                    yam_repl.write_attempt_entry(
                        args.journal_path, global_attempt,
                        instruction, status, tagged_notes, stats, args,
                        invocation,
                    )
                    results.add(task_idx + 1, instruction, attempt,
                                status, notes, stats)
                    attempt += 1
    except KeyboardInterrupt:
        if aborted:
            log.info("Eval aborted by operator. Tearing down.")
        else:
            log.info("Ctrl-C -- shutting down")
    finally:
        # Print per-task summary.
        print("\n" + "#" * 70, flush=True)
        print("# Eval summary", flush=True)
        print("#" * 70, flush=True)
        print(f"{'#':>3}  {'task':<48} {'S':>3} {'F':>3} {'U':>3} {'Sk':>3} {'rate':>6}",
              flush=True)
        total_s = total_f = total_u = total_sk = 0
        for task_idx in selected:
            t = tallies[task_idx]
            ran = t["success"] + t["failure"] + t["unclear"]
            rate = (100.0 * t["success"] / ran) if ran else 0.0
            print(f"{task_idx+1:>3}  {TASKS[task_idx][:48]:<48} "
                  f"{t['success']:>3} {t['failure']:>3} {t['unclear']:>3} "
                  f"{t['skip']:>3} {rate:>5.0f}%",
                  flush=True)
            total_s += t["success"]; total_f += t["failure"]
            total_u += t["unclear"]; total_sk += t["skip"]
        ran_total = total_s + total_f + total_u
        rate_total = (100.0 * total_s / ran_total) if ran_total else 0.0
        print(f"{'':>3}  {'TOTAL':<48} {total_s:>3} {total_f:>3} {total_u:>3} "
              f"{total_sk:>3} {rate_total:>5.0f}%",
              flush=True)
        print(f"\nResults CSV: {results_path}", flush=True)
        print(f"Journal:     {args.journal_path}", flush=True)

        results.close()

        # Teardown (same as yam_repl).
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

        log.info("Eval done. Ran %d attempt(s).", global_attempt)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
