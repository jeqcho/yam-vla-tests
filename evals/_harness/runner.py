"""Multi-task eval session orchestrator.

`start_session(policy, eval_def, args, results_base_dir)` is the single
entry point. It owns:

    1. Hardware bring-up (cameras + arms + startup-pose capture)
    2. Operator prompts (task picker, prompt picker, per-attempt ready)
    3. Per-attempt control loop  (delegated to core.run_attempt)
    4. CSV writing                (delegated to ResultsWriter)
    5. Journal append             (delegated to core.journal)
    6. Safe shutdown              (return-to-startup ramp, motor disable)

This file is pure new-code. The legacy `_yc.main()` is no longer called,
the post_actions monkey-patches are gone. Every policy goes through
exactly the same control loop -- the only thing that changes per-policy
is the YAML config you pass to PolicyConfig.build().
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from yam_vla.core import (
    AttemptKnobs, AttemptStats, PolicyConfig, Policy, RerunRecorder,
    capture_invocation, init_arm, load_setup_config, make_camera,
    prompt_journal_entry, ramp_to_pose, run_attempt, write_journal_entry,
    DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT, DEFAULT_CAM_FPS,
    DEFAULT_HORIZON_STRIDE, DEFAULT_TRAIN_FPS,
    DEFAULT_MAX_STEP_RAD, DEFAULT_GRIPPER_STEP, DEFAULT_JOURNAL_PATH,
)

from evals._harness.results import ResultsWriter, AttemptRow
from evals._harness.tasks import EvalDefinition, EvalTask

log = logging.getLogger("yam_vla.evals.runner")


# ---------------------------------------------------------------------------
# Operator-prompt helpers (interactive flow)
# ---------------------------------------------------------------------------

def _prompt_select_tasks(eval_def: EvalDefinition) -> Optional[list[int]]:
    """Show the task list, let operator pick a subset. Returns 0-based indices or None."""
    print("\n" + "=" * 70, flush=True)
    print(f"Eval: {eval_def.name}  --  {len(eval_def.tasks)} tasks", flush=True)
    print("=" * 70, flush=True)
    for i, t in enumerate(eval_def.tasks, 1):
        label = f"{t.id}" + (f"  ({t.english})" if t.english else "")
        print(f"  {i:>2}. {label}", flush=True)
    print(f"\n[enter] = all  |  comma-sep 1-{len(eval_def.tasks)}  |  'q' to quit",
          flush=True)
    sys.stdout.flush()
    try:
        ans = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if ans in {"q", "quit", "exit"}:
        return None
    if not ans:
        return list(range(len(eval_def.tasks)))
    try:
        sel = sorted({int(x) - 1 for x in ans.split(",") if x.strip()})
    except ValueError:
        print(f"  ?? unparseable {ans!r}; defaulting to all", flush=True)
        return list(range(len(eval_def.tasks)))
    sel = [i for i in sel if 0 <= i < len(eval_def.tasks)]
    return sel or list(range(len(eval_def.tasks)))


def _prompt_select_prompt(task: EvalTask) -> Optional[tuple[str, str]]:
    """Pick which instruction to send: full or atomic_N. Returns (text, kind)."""
    if not task.atomic_actions:
        return (task.instruction, "full")

    print("", flush=True)
    print("-" * 70, flush=True)
    print(f"  {task.id}" + (f"  --  {task.english}" if task.english else ""), flush=True)
    print("-" * 70, flush=True)
    print(f"  [enter] full instruction:", flush=True)
    print(f"          {task.instruction!r}", flush=True)
    for i, a in enumerate(task.atomic_actions, 1):
        print(f"  {i}. atomic_{i}: {a!r}", flush=True)
    print(f"  s = skip task  |  q = abort eval", flush=True)
    sys.stdout.flush()
    try:
        ans = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if ans in {"q", "quit", "exit"}:
        return None
    if ans in {"s", "skip"}:
        return ("", "skip")
    if not ans:
        return (task.instruction, "full")
    try:
        idx = int(ans) - 1
        if 0 <= idx < len(task.atomic_actions):
            return (task.atomic_actions[idx], f"atomic_{idx + 1}")
    except ValueError:
        pass
    print(f"  ?? unrecognized {ans!r}; using full instruction", flush=True)
    return (task.instruction, "full")


def _prompt_ready(task: EvalTask, attempt: int, n_attempts: int,
                   prompt_text: str, prompt_kind: str, policy_name: str) -> str:
    """Show context, ask operator to start. Returns 'go' / 'skip' / 'quit'."""
    print("\n" + "=" * 70, flush=True)
    print(f"[policy={policy_name}]  attempt {attempt}/{n_attempts}  prompt={prompt_kind}",
          flush=True)
    print(f"  task: {task.id}" + (f"  ({task.english})" if task.english else ""), flush=True)
    print(f"  sending: {prompt_text!r}", flush=True)
    print("=" * 70, flush=True)
    print("[enter to start  |  's' skip attempt  |  'q' abort eval]", flush=True)
    sys.stdout.flush()
    try:
        ans = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "quit"
    if ans in {"q", "quit", "exit"}:
        return "quit"
    if ans in {"s", "skip"}:
        return "skip"
    return "go"


def _prompt_score_attempt() -> tuple[str, str]:
    """After an attempt ends, ask operator how it went. Returns (status, notes)."""
    print("\n[score] how did it go?  s=success  f=failure  u=unclear  [enter]=skip", flush=True)
    try:
        ans = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return ("skip", "")
    status_map = {"s": "success", "f": "failure", "u": "unclear"}
    status = status_map.get(ans[:1] if ans else "", "skip")
    notes = ""
    if status != "skip":
        try:
            notes = input("notes (optional)\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            notes = ""
    return status, notes


# ---------------------------------------------------------------------------
# Stop-watcher: hit Enter on stdin to break out of the running attempt
# ---------------------------------------------------------------------------

class _EnterStopWatcher:
    """Background thread that flips a flag when the operator hits Enter.

    Used as the `stop` predicate for run_attempt -- gives the operator a
    way to end the attempt the moment they see the task succeed/fail,
    instead of waiting for max_chunks.
    """
    def __init__(self):
        self._stopped = False
        self._thread = None

    def start(self) -> None:
        import threading

        def _wait():
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            self._stopped = True

        self._thread = threading.Thread(target=_wait, daemon=True)
        self._thread.start()

    @property
    def stopped(self) -> bool:
        return self._stopped

    def predicate(self):
        return lambda: self._stopped


# ---------------------------------------------------------------------------
# The main entry point
# ---------------------------------------------------------------------------

def start_session(
    policy: Policy,
    eval_def: EvalDefinition,
    args: argparse.Namespace,
    *,
    results_base_dir: str | Path,
) -> None:
    """Run one eval session against `policy` on `eval_def.tasks`.

    See module docstring for the orchestration steps.
    """
    invocation = capture_invocation()
    session_start_s = time.time()

    # Per-machine defaults from yam_setup_config.json
    setup_cfg = load_setup_config()

    results = ResultsWriter(
        base_dir=results_base_dir,
        policy=policy.name,
        eval_name=eval_def.name,
    )
    log.info("[session] CSV: %s", results.path)

    # Server handshake -- fail fast if the inference server isn't up
    info = policy.info(timeout_s=5.0)
    log.info("[session] policy: %s  model: %s  horizon_hint: %s",
             info.backend, info.model_id, info.action_horizon_hint)
    model_id = info.model_id or "unknown"

    # ---------- task picker ----------
    sel = _prompt_select_tasks(eval_def)
    if sel is None:
        log.info("aborted by operator")
        return

    # ---------- camera bring-up (BEFORE arms; see docs/handoffs/molmoact2-setup.md) ----------
    rerun = RerunRecorder(
        enabled=getattr(args, "rerun", False),
        save_path=getattr(args, "rerun_save", None),
        connect=getattr(args, "rerun_connect", None),
    )

    top = cam_l = cam_r = None
    left = right = None
    try:
        cam_kw = dict(
            width=getattr(args, "cam_width", DEFAULT_CAM_WIDTH),
            height=getattr(args, "cam_height", DEFAULT_CAM_HEIGHT),
            fps=getattr(args, "cam_fps", DEFAULT_CAM_FPS),
        )
        top = make_camera("top",
                          getattr(args, "top_cam_serial", None) or setup_cfg.get("top_cam_serial"),
                          getattr(args, "top_cam_v4l2", None)   or setup_cfg.get("top_cam_v4l2"),
                          **cam_kw)
        cam_l = make_camera("left",
                            getattr(args, "left_cam_serial", None) or setup_cfg.get("left_cam_serial"),
                            getattr(args, "left_cam_v4l2", None)   or setup_cfg.get("left_cam_v4l2"),
                            **cam_kw)
        cam_r = make_camera("right",
                            getattr(args, "right_cam_serial", None) or setup_cfg.get("right_cam_serial"),
                            getattr(args, "right_cam_v4l2", None)   or setup_cfg.get("right_cam_v4l2"),
                            **cam_kw)
        for c in (top, cam_l, cam_r):
            c.start()
        # Settle: drain a few frames so AE has converged before motor threads spin up.
        for _ in range(3):
            for c in (top, cam_l, cam_r):
                try: c.grab()
                except Exception as e: log.warning("settle: %s.grab() failed: %s", c.name, e)

        # ---------- arms ----------
        gripper = getattr(args, "gripper", None) or setup_cfg.get("gripper", "linear_4310")
        left_can  = getattr(args, "left_can",  None) or setup_cfg.get("left_can",  "can0")
        right_can = getattr(args, "right_can", None) or setup_cfg.get("right_can", "can1")
        left  = init_arm(left_can,  gripper)
        right = init_arm(right_can, gripper)

        # Capture startup pose for return-on-exit
        startup_pose = np.concatenate([
            np.asarray(left.get_joint_pos(),  dtype=np.float32),
            np.asarray(right.get_joint_pos(), dtype=np.float32),
        ])
        log.info("[session] startup pose: %s",
                 np.array2string(startup_pose, precision=3))

        # ---------- per-task / per-attempt loop ----------
        n_attempts = getattr(args, "attempts", None) or eval_def.n_attempts_default
        for ti in sel:
            task = eval_def.tasks[ti]
            picked = _prompt_select_prompt(task)
            if picked is None:
                log.info("aborted by operator")
                break
            prompt_text, prompt_kind = picked
            if prompt_kind == "skip":
                log.info("[task %s] skipped by operator", task.id)
                continue

            for attempt in range(1, n_attempts + 1):
                action = _prompt_ready(task, attempt, n_attempts,
                                        prompt_text, prompt_kind, policy.name)
                if action == "quit":
                    raise KeyboardInterrupt
                if action == "skip":
                    results.write(AttemptRow(
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                        policy=policy.name, model_id=model_id,
                        eval=eval_def.name, task_id=task.id,
                        attempt=attempt, status="skip",
                        prompt_kind=prompt_kind, prompt_text=prompt_text,
                    ))
                    continue

                # Run the attempt. Operator hits Enter to end early.
                watcher = _EnterStopWatcher()
                watcher.start()
                knobs = AttemptKnobs(
                    instruction=prompt_text,
                    max_chunks=getattr(args, "max_chunks", 200),
                    train_fps=getattr(args, "train_fps", DEFAULT_TRAIN_FPS),
                    horizon_stride=getattr(args, "horizon_stride", DEFAULT_HORIZON_STRIDE),
                    max_step_rad=getattr(args, "max_step_rad", DEFAULT_MAX_STEP_RAD),
                    gripper_step=getattr(args, "gripper_step", DEFAULT_GRIPPER_STEP),
                    timeout_s=getattr(args, "timeout_s", 15.0),
                    inference_mode=getattr(args, "inference_mode", "sync"),
                    dry_run=getattr(args, "dry_run", False),
                    policy_opts={"num_steps": getattr(args, "num_steps", 10)},
                )
                print("\n[attempt] running. press Enter to stop early.", flush=True)
                stats = run_attempt(
                    policy=policy, knobs=knobs,
                    top_cam=top, left_cam=cam_l, right_cam=cam_r,
                    left_arm=left, right_arm=right,
                    rerun=rerun,
                    stop=watcher.predicate(),
                )

                # Score
                status, notes = _prompt_score_attempt()
                results.write(AttemptRow(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    policy=policy.name, model_id=model_id,
                    eval=eval_def.name, task_id=task.id,
                    attempt=attempt,
                    status=status if status != "skip" else "incomplete",
                    duration_s=stats.duration_s,
                    chunks=stats.chunks,
                    rtt_ms_mean=stats.rtt_ms_mean,
                    rtt_ms_p95=stats.rtt_ms_p95,
                    rtt_ms_max=stats.rtt_ms_max,
                    horizon_arm_mean=stats.horizon_arm_mean,
                    clip_rate=stats.clip_rate,
                    prompt_kind=prompt_kind, prompt_text=prompt_text,
                    notes=notes,
                ))

    except KeyboardInterrupt:
        log.info("[session] KeyboardInterrupt — shutting down")
    finally:
        # Journal first so a Ctrl-C in cleanup can't kill the record.
        try:
            entry = prompt_journal_entry(session_start_s, args)
            if entry is not None:
                entry["notes"] = f"[policy={policy.name}] eval={eval_def.name}  " + entry.get("notes", "")
                write_journal_entry(
                    getattr(args, "journal_path", DEFAULT_JOURNAL_PATH),
                    entry, args, invocation,
                )
        except Exception as e:
            log.warning("journal step failed: %s", e)

        # Return arms to startup pose so close() doesn't drop them
        abort = {"abort": False, "ctrlc_count": 0}
        def _cleanup_sigint(_sig, _frame):
            abort["ctrlc_count"] += 1
            if abort["ctrlc_count"] == 1:
                log.warning("Ctrl-C in cleanup: aborting return ramp. ARMS WILL DROP.")
                abort["abort"] = True
            else:
                os._exit(130)
        try:
            signal.signal(signal.SIGINT, _cleanup_sigint)
        except Exception:
            pass

        if left is not None and right is not None and 'startup_pose' in locals() \
                and not getattr(args, "no_return_on_exit", False):
            try:
                log.info("Returning arms to startup pose (5.0s)...")
                ramp_to_pose(left, right, startup_pose,
                             duration_s=5.0,
                             abort_flag=abort,
                             label="return-on-exit")
            except BaseException as e:
                log.warning("return ramp failed: %s. ARMS MAY DROP.", e)

        # Cameras
        for c in (top, cam_l, cam_r):
            if c is not None:
                try: c.stop()
                except Exception as e: log.warning("camera stop: %s", e)

        # Arms
        for arm in (left, right):
            if arm is not None:
                try: arm.close()
                except Exception as e: log.warning("arm.close: %s", e)

        # Close the policy transport
        try:
            policy.close()
        except Exception as e:
            log.warning("policy.close: %s", e)

        log.info("[session] done. results: %s", results.path)


__all__ = ["start_session"]
