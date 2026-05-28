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

from yam_vla.core.keyboard import (
    AdvanceWatcher as _AdvanceWatcher,
    RawTerm as _RawTerm,
    read_key as _read_key,
    reset_countdown as _reset_countdown,
    wait_for_advance as _wait_for_advance,
)

from yam_vla.core import (
    AttemptKnobs, AttemptStats, PolicyConfig, Policy, RerunRecorder,
    capture_invocation, load_setup_config, make_camera,
    prompt_journal_entry, ramp_to_pose, run_attempt, write_journal_entry,
    DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT, DEFAULT_CAM_FPS,
    DEFAULT_HORIZON_STRIDE, DEFAULT_TRAIN_FPS,
    DEFAULT_MAX_STEP_RAD, DEFAULT_GRIPPER_STEP, DEFAULT_JOURNAL_PATH,
    get_backend,
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
    """Show task + iteration banner, wait for operator advance.

    Returns 'go' / 'skip' / 'quit'.  Operator advances with right-arrow
    OR Enter.
    """
    label = task.id + (f"  ({task.english})" if task.english else "")
    print("\n" + "=" * 70, flush=True)
    print(f"  TASK  : {label}", flush=True)
    print(f"  ITER  : {attempt} of {n_attempts}     [policy={policy_name}]", flush=True)
    print(f"  PROMPT: {prompt_text!r}", flush=True)
    print("=" * 70, flush=True)
    return _wait_for_advance(
        "  press → or Enter to START   |   's' skip this attempt   |   'q' abort eval"
    )


def _prompt_score_attempt(task: EvalTask, attempt: int, n_attempts: int) -> tuple[str, str]:
    """After an attempt ends, ask operator how it went.

    Status pick uses raw-mode single keypress (consistent with the start
    banner — no Enter required). Notes entry stays cooked-mode so the
    operator can type a free-form line if they want.
    Returns (status, notes).
    """
    label = task.id + (f"  ({task.english})" if task.english else "")
    print("\n" + "-" * 70, flush=True)
    print(f"  SCORE iteration {attempt} of {n_attempts}  --  {label}", flush=True)
    print("-" * 70, flush=True)
    print("  s = success   f = failure   u = unclear   r = redo   [enter or →] = skip",
          flush=True)

    # Status pick: raw single keypress. We block until the operator hits
    # one of the recognized keys, so background log spam can't accidentally
    # eat the input.
    if not sys.stdin.isatty():
        # Non-TTY (piped / CI): fall back to a single line read.
        try:
            ans = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ("skip", "")
        if ans[:1] == "r":
            return ("redo", "")
        status_map = {"s": "success", "f": "failure", "u": "unclear"}
        status = status_map.get(ans[:1] if ans else "", "skip")
    else:
        with _RawTerm():
            while True:
                key = _read_key(timeout=0.5)
                if key is None:
                    continue
                if key == "s":
                    status = "success"
                    break
                if key == "f":
                    status = "failure"
                    break
                if key == "u":
                    status = "unclear"
                    break
                if key == "r":
                    return ("redo", "")
                if key in ("enter", "right"):
                    status = "skip"
                    break
                if key == "q":
                    return ("skip", "")
        # Echo what we recorded so the operator has visual confirmation.
        print(f"  recorded: {status}", flush=True)

    notes = ""
    if status != "skip":
        try:
            notes = input("notes (optional, [enter] to skip)\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            notes = ""
    return status, notes


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

    # Resolve the active embodiment backend. `run_eval.py` stashes one on
    # args; fall back to YAM for any caller that built a Namespace by hand
    # without going through the CLI (e.g. notebooks).
    backend = getattr(args, "_backend", None) or get_backend(
        getattr(args, "embodiment", "yam")
    )

    # Per-machine defaults from the backend's setup-config file.
    setup_cfg = load_setup_config(backend.default_setup_config_path())

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

        # ---------- arms (per-embodiment backend) ----------
        backend.install_quirks()
        hw_cfg = backend.resolve_hw_cfg(args, setup_cfg)
        left  = backend.init_left(hw_cfg)
        right = backend.init_right(hw_cfg)

        # Capture startup pose for return-on-exit
        startup_pose = np.concatenate([
            np.asarray(left.get_joint_pos(),  dtype=np.float32),
            np.asarray(right.get_joint_pos(), dtype=np.float32),
        ])
        log.info("[session] startup pose: %s",
                 np.array2string(startup_pose, precision=3))

        # Resolve the canonical ready pose for this policy (from the
        # policy YAML's control.ready_pose). If present, the arms ramp
        # to it once now and back to it between every attempt -- so the
        # policy always starts from a pose its training distribution
        # has seen many times.
        #
        # Gripper indices (6 and 13) are overwritten with the startup-
        # pose gripper values so the ramp doesn't slam-close on whatever
        # is currently held.
        ready_pose_cfg = getattr(args, "ready_pose", None)
        ready_pose: Optional[np.ndarray] = None
        ready_pose_ramp_s = float(
            getattr(args, "ready_pose_ramp_duration_s", 5.0) or 5.0
        )
        if ready_pose_cfg is not None:
            rp = np.asarray(ready_pose_cfg, dtype=np.float32)
            if rp.shape != (14,):
                log.warning("ready_pose has shape %s, expected (14,); skipping ramp",
                            rp.shape)
            else:
                rp[6]  = startup_pose[6]
                rp[13] = startup_pose[13]
                ready_pose = rp
                log.info("[session] canonical ready pose (grippers preserved): %s",
                         np.array2string(ready_pose, precision=3))
                log.info("Ramping to canonical ready pose (%.1fs)...", ready_pose_ramp_s)
                ramp_to_pose(left, right, ready_pose,
                             duration_s=ready_pose_ramp_s,
                             label="initial move-to-ready")

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

            # --reset-seconds: scene-reset window between attempts of the
            # same task. Operator can let it expire OR press → to advance
            # immediately. 0 = no countdown (preserves ikea_10 / andon_10
            # behavior).  Resolved here so a per-eval YAML default can
            # apply even if --reset-seconds isn't passed on the CLI.
            cli_reset = getattr(args, "reset_seconds", None)
            if cli_reset is None:
                reset_seconds = float(eval_def.reset_seconds_default)
            else:
                reset_seconds = float(cli_reset)

            attempt = 1
            while attempt <= n_attempts:
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
                    attempt += 1
                    continue

                # Run the attempt. Operator presses → (or Enter) to end early.
                watcher = _AdvanceWatcher()
                watcher.start()
                # Camera-health watchdog: detects stale-camera within
                # ~100ms (much faster than waiting for the next grab()
                # to time out 1.5s in), exits rollout cleanly before
                # the camera-CAN-USB cascade reaches motor watchdog.
                # See yam_vla.core.hardware.CameraHealthWatcher docstring.
                from yam_vla.core.hardware import CameraHealthWatcher
                cam_watchdog = CameraHealthWatcher(
                    cameras=[top, cam_l, cam_r],
                    stale_threshold_s=float(
                        getattr(args, "camera_stale_threshold_s", 0.6)
                    ),
                )
                cam_watchdog.start()
                knobs = AttemptKnobs(
                    instruction=prompt_text,
                    max_chunks=getattr(args, "max_chunks", 200),
                    train_fps=getattr(args, "train_fps", DEFAULT_TRAIN_FPS),
                    horizon_stride=getattr(args, "horizon_stride", DEFAULT_HORIZON_STRIDE),
                    max_step_rad=getattr(args, "max_step_rad", DEFAULT_MAX_STEP_RAD),
                    gripper_step=getattr(args, "gripper_step", DEFAULT_GRIPPER_STEP),
                    timeout_s=getattr(args, "timeout_s", 15.0),
                    attempt_timeout_s=getattr(args, "attempt_timeout_s", 60.0),
                    inference_mode=getattr(args, "inference_mode", "sync"),
                    dry_run=getattr(args, "dry_run", False),
                    policy_opts={"num_steps": getattr(args, "num_steps", 10)},
                )
                print("\n[running] arms moving. press → (or Enter) to STOP.", flush=True)
                # Combined stop predicate: operator advance OR camera health alarm.
                # Either fires => rollout exits cleanly.
                def _combined_stop():
                    return watcher.stopped or cam_watchdog.stopped
                attempt_crashed = False
                attempt_error: Optional[BaseException] = None
                try:
                    stats = run_attempt(
                        policy=policy, knobs=knobs,
                        top_cam=top, left_cam=cam_l, right_cam=cam_r,
                        left_arm=left, right_arm=right,
                        rerun=rerun,
                        stop=_combined_stop,
                    )
                    # If the watchdog tripped (not operator), treat the
                    # exit as a soft crash -- camera state is uncertain,
                    # so the post-attempt ramp gate should engage.
                    if cam_watchdog.stopped:
                        attempt_crashed = True
                        attempt_error = RuntimeError(
                            f"camera watchdog tripped: {cam_watchdog.stop_reason}"
                        )
                        log.error("[camera-watchdog] %s", cam_watchdog.stop_reason)
                except KeyboardInterrupt:
                    # Operator-driven abort. Propagate to outer except.
                    raise
                except Exception as e:
                    attempt_crashed = True
                    attempt_error = e
                    log.error("attempt crashed: %s", e)
                    # Fabricate a minimal stats so downstream CSV write
                    # doesn't NameError. Status will be "crash".
                    from yam_vla.core import AttemptStats
                    stats = AttemptStats(status="crash")
                finally:
                    # Restore cooked stdin BEFORE any input() call.
                    watcher.stop()
                    cam_watchdog.stop()

                # --- DANGER WINDOW ---
                # If the attempt crashed (esp. due to camera disconnect
                # which on this rig is correlated with CAN-USB disruption),
                # the motor comms may be down and `get_joint_pos()` may
                # return stale cached values. Issuing a ramp from stale
                # cached pose to ready pose will compute a SMALL delta
                # (because cache says "you're near ready") but the arms
                # have physically dropped under gravity -- so when motor
                # comms recover the motors snap from current-fallen pose
                # to the commanded interpolated trajectory. This is
                # whiplash and broke a camera stand on 2026-05-28.
                #
                # Mitigation: do NOT auto-ramp after a crash. Warn the
                # operator instead. They visually inspect the arms,
                # decide whether to ramp (slowly), and either re-arm or
                # skip.
                if attempt_crashed:
                    print("\n" + "!" * 70, flush=True)
                    print("!  ATTEMPT CRASHED  --  hardware state may be UNSAFE", flush=True)
                    print("!", flush=True)
                    print(f"!  reason: {attempt_error}", flush=True)
                    print("!", flush=True)
                    print("!  Motors may have lost CAN comms and torque during the crash.", flush=True)
                    print("!  If arms physically dropped under gravity, issuing motion", flush=True)
                    print("!  commands now will cause them to SNAP back to commanded", flush=True)
                    print("!  position -- this is dangerous and has caused damage on", flush=True)
                    print("!  this rig before. Please VISUALLY INSPECT THE ARMS NOW.", flush=True)
                    print("!" * 70, flush=True)
                    print("\n  → / Enter  -- arms LOOK ok, ramp back slowly (10s)", flush=True)
                    print("  's'        -- arms look BAD, skip ramp, go to scoring", flush=True)
                    print("  'q'        -- abort eval entirely (you'll re-arm by hand)", flush=True)
                    ans = _wait_for_advance("")
                    if ans == "quit":
                        raise KeyboardInterrupt
                    elif ans == "go":
                        # Slow ramp (10s, 2x normal) and explicitly re-read
                        # joint pos in case it was stale. If get_joint_pos
                        # itself raises (motor comms still bad), skip ramp.
                        try:
                            cur = np.concatenate([
                                np.asarray(left.get_joint_pos(),  dtype=np.float32),
                                np.asarray(right.get_joint_pos(), dtype=np.float32),
                            ])
                            if ready_pose is not None:
                                rp_now = ready_pose.copy()
                                rp_now[6]  = cur[6]
                                rp_now[13] = cur[13]
                                slow_ramp_s = max(10.0, ready_pose_ramp_s * 2)
                                log.info("[post-crash] Slow ramp to ready pose (%.1fs)...",
                                         slow_ramp_s)
                                ramp_to_pose(left, right, rp_now,
                                             duration_s=slow_ramp_s,
                                             label="post-crash slow-ramp")
                        except Exception as e:
                            log.error("post-crash ramp aborted: %s", e)
                    # else ans == "skip": fall through to scoring with arms wherever they are
                elif ready_pose is not None:
                    # Normal path -- attempt completed cleanly. Ramp back
                    # to canonical ready pose so the next iteration starts
                    # from the training-distribution centroid and the
                    # operator can reset the scene without flopped arms.
                    try:
                        cur = np.concatenate([
                            np.asarray(left.get_joint_pos(),  dtype=np.float32),
                            np.asarray(right.get_joint_pos(), dtype=np.float32),
                        ])
                        rp_now = ready_pose.copy()
                        rp_now[6]  = cur[6]
                        rp_now[13] = cur[13]
                        log.info("Ramping back to ready pose (%.1fs)...", ready_pose_ramp_s)
                        ramp_to_pose(left, right, rp_now,
                                     duration_s=ready_pose_ramp_s,
                                     label="post-attempt ramp-to-ready")
                    except Exception as e:
                        log.warning("post-attempt ramp failed: %s", e)

                # Score (operator types s/f/u/r/<enter>)
                status, notes = _prompt_score_attempt(task, attempt, n_attempts)
                if status == "redo":
                    log.info("[task %s] iteration %d redo requested", task.id, attempt)
                    # Same iteration runs again; no CSV row written.
                    continue
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

                # Reset window before the next attempt of this task.
                # Last iteration of the task: no inter-attempt reset; the
                # next task's _prompt_select_prompt + _prompt_ready give
                # the operator their reset time.
                if attempt < n_attempts and reset_seconds > 0:
                    next_label = f"iter {attempt + 1} of {n_attempts}  ({task.id})"
                    rc = _reset_countdown(reset_seconds, label=next_label)
                    if rc == "quit":
                        raise KeyboardInterrupt
                    if rc == "skip":
                        log.info("[task %s] remaining iterations skipped by operator",
                                 task.id)
                        break

                attempt += 1

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
