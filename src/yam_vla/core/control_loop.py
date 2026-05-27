"""Per-attempt control loop: cameras + arms + policy.predict + safety + sleep.

This is what replaced `_yc.main()` in the legacy codebase. Used by both
the eval harness (which calls it per task/attempt) and the REPL (which
calls it per typed instruction).

The public API is `run_attempt(...)` -- one function that takes a fully-
constructed Policy + hardware handles + control knobs and runs one
fixed-duration or until-quit inference loop.

Reading order:
    AttemptKnobs        -- the dataclass of all tunable knobs
    AttemptStats        -- what run_attempt returns
    run_attempt         -- the function itself
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from yam_vla.core.observation import (
    LEFT_ARM, LEFT_GRIP, RIGHT_ARM, RIGHT_GRIP, STATE_DIM,
    YamObservation, ImageRole,
)
from yam_vla.core.policy import Policy, Prediction
from yam_vla.core.safety import safe_command, DEFAULT_MAX_STEP_RAD, DEFAULT_GRIPPER_STEP
from yam_vla.core.hardware import read_state, DEFAULT_TRAIN_FPS, DEFAULT_HORIZON_STRIDE
from yam_vla.core.observability import RerunRecorder
from yam_vla.core.runner import AsyncPolicyInference

log = logging.getLogger("yam_vla.control_loop")


# ---------------------------------------------------------------------------
# Public knobs (one dataclass instead of N argparse flags)
# ---------------------------------------------------------------------------

@dataclass
class AttemptKnobs:
    """All tunable knobs for one attempt. Eval and REPL both build one of these."""
    instruction:     str
    max_chunks:      int   = 200             # safety bound; ~133s @ 6-stride / 30Hz
    train_fps:       float = DEFAULT_TRAIN_FPS
    horizon_stride:  int   = DEFAULT_HORIZON_STRIDE
    max_step_rad:    float = DEFAULT_MAX_STEP_RAD
    gripper_step:    float = DEFAULT_GRIPPER_STEP
    timeout_s:       float = 15.0            # per-INFERENCE-CALL HTTP/ZMQ/WS timeout
    attempt_timeout_s: float = 60.0          # per-ATTEMPT wall-clock cap; 0 = disabled
    inference_mode:  str   = "sync"          # "sync" | "async-naive" | "async-time-aligned"
    dry_run:         bool  = False
    policy_opts:     dict  = field(default_factory=dict)  # e.g. {"num_steps": 10}


@dataclass
class AttemptStats:
    """What run_attempt returns -- the per-attempt CSV row inputs."""
    status:             str   = "incomplete"   # "quit"|"timeout"|"maxchunks"|"crash"
    chunks:             int   = 0
    duration_s:         float = 0.0
    rtt_ms_mean:        float = 0.0
    rtt_ms_p95:         float = 0.0
    rtt_ms_max:         float = 0.0
    horizon_arm_mean:   float = 0.0
    horizon_arm_max:    float = 0.0
    clip_rate:          float = 0.0
    raw_rtts_ms:        list  = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: stop predicate so callers can drive when the attempt ends
# ---------------------------------------------------------------------------

StopPredicate = Callable[[], bool]  # returns True to break the loop


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def run_attempt(
    *,
    policy: Policy,
    knobs: AttemptKnobs,
    top_cam,
    left_cam,
    right_cam,
    left_arm,
    right_arm,
    rerun: Optional[RerunRecorder] = None,
    stop: Optional[StopPredicate] = None,
) -> AttemptStats:
    """Run ONE attempt of `policy` on `knobs.instruction` against the live arms.

    Returns when:
      * `stop()` returns True (operator hit Enter / scored the attempt), OR
      * `knobs.max_chunks` reached (safety bound)
      * the user Ctrl-C's (raised as KeyboardInterrupt to the caller)

    Two inference modes:
      "sync"                -- POST blocks the inner loop, arm holds during inference
      "async-time-aligned"  -- overlap inference with execution, K=stride lookahead

    Returns an AttemptStats with timing + clip telemetry. `status` defaults
    to "incomplete" -- the caller decides "success"/"failure" via the
    journal prompt.
    """
    if rerun is None:
        rerun = RerunRecorder(enabled=False)
    if stop is None:
        stop = lambda: False

    inner_dt = 1.0 / knobs.train_fps
    stride = max(1, knobs.horizon_stride)

    stats = AttemptStats()
    rtts_ms: list[float] = []
    horizon_arm_spans: list[float] = []
    clipped_total = 0
    steps_total = 0

    loop_t0 = time.perf_counter()
    arm_idx = np.r_[LEFT_ARM, RIGHT_ARM]  # array of 12 arm-joint indices

    # ---------- async helper if requested ----------
    async_infer: Optional[AsyncPolicyInference] = None
    chunk_start_idx = 0
    if knobs.inference_mode != "sync":
        chunk_start_idx = stride if knobs.inference_mode == "async-time-aligned" else 0
        async_infer = AsyncPolicyInference(
            policy, timeout_s=knobs.timeout_s, opts=knobs.policy_opts,
        )
        # Bootstrap: synchronous first chunk
        obs = _capture_observation(top_cam, left_cam, right_cam,
                                   left_arm, right_arm, knobs.instruction)
        rerun.log_observation(time.perf_counter() - loop_t0,
                              obs.images[ImageRole.TOP],
                              obs.images[ImageRole.LEFT_WRIST],
                              obs.images[ImageRole.RIGHT_WRIST], obs.state)
        async_infer.kick_off(obs)
        pred = async_infer.wait()
        rtts_ms.append(pred.rtt_ms)
        log.info("async bootstrap OK rtt=%.0fms shape=%s",
                 pred.rtt_ms, pred.actions.shape)
    else:
        pred = None  # filled in on first iteration

    # ---------- main loop ----------
    try:
        while stats.chunks < knobs.max_chunks:
            if stop():
                stats.status = "quit"
                break
            if knobs.attempt_timeout_s > 0 and \
                    (time.perf_counter() - loop_t0) >= knobs.attempt_timeout_s:
                log.info("attempt timeout (%.0fs) reached after %d chunks; ending",
                         knobs.attempt_timeout_s, stats.chunks)
                stats.status = "timeout"
                break

            # SYNC: capture, predict (blocking), then play chunk.
            # ASYNC: next-chunk POST already in flight; capture, kick off
            #        the chunk after, play the previous chunk, then await.
            if knobs.inference_mode == "sync":
                obs = _capture_observation(top_cam, left_cam, right_cam,
                                           left_arm, right_arm, knobs.instruction)
                rerun.log_observation(time.perf_counter() - loop_t0,
                                      obs.images[ImageRole.TOP],
                                      obs.images[ImageRole.LEFT_WRIST],
                                      obs.images[ImageRole.RIGHT_WRIST], obs.state)
                pred = policy.predict(obs, timeout_s=knobs.timeout_s, **knobs.policy_opts)
                rtts_ms.append(pred.rtt_ms)

            actions = pred.actions
            horizon_range = actions.max(axis=0) - actions.min(axis=0)
            horizon_arm_span = float(np.max(horizon_range[arm_idx]))
            horizon_arm_spans.append(horizon_arm_span)
            rerun.log_inference(time.perf_counter() - loop_t0, actions,
                                executed_idx=chunk_start_idx,
                                rtt_ms=pred.rtt_ms,
                                horizon_arm_span=horizon_arm_span)

            n_to_play = min(stride, actions.shape[0] - chunk_start_idx)
            if n_to_play <= 0:
                log.warning("chunk too short shape=%s start=%d",
                            actions.shape, chunk_start_idx)
                break

            # In async mode, kick off the NEXT chunk's POST now -- it
            # runs in the background while we execute the current chunk.
            if async_infer is not None:
                next_obs = _capture_observation(top_cam, left_cam, right_cam,
                                                left_arm, right_arm, knobs.instruction)
                rerun.log_observation(time.perf_counter() - loop_t0,
                                      next_obs.images[ImageRole.TOP],
                                      next_obs.images[ImageRole.LEFT_WRIST],
                                      next_obs.images[ImageRole.RIGHT_WRIST],
                                      next_obs.state)
                async_infer.kick_off(next_obs)

            # Per-query telemetry: how far is the model commanding from current state?
            log.info("chunk #%d rtt=%dms horizon_arm_span=%.3frad n_to_play=%d",
                     stats.chunks + 1, int(pred.rtt_ms), horizon_arm_span, n_to_play)

            # Execute the chunk
            for i in range(n_to_play):
                if stop():
                    stats.status = "quit"
                    break
                step_start = time.perf_counter()
                action_idx = chunk_start_idx + i
                desired = actions[action_idx].astype(np.float32)
                if knobs.dry_run:
                    log.info("dry-run action[%d]: %s", action_idx,
                             np.array2string(desired, precision=3))
                else:
                    state = read_state(left_arm, right_arm)
                    _, n_clipped = safe_command(
                        left_arm, right_arm, state, desired,
                        max_step_rad=knobs.max_step_rad,
                        gripper_step=knobs.gripper_step,
                    )
                    clipped_total += n_clipped
                    steps_total += 1
                sleep_left = inner_dt - (time.perf_counter() - step_start)
                if sleep_left > 0:
                    time.sleep(sleep_left)
                elif sleep_left < -0.050:
                    log.warning("inner step overrun by %.1fms (target %.1fms)",
                                -sleep_left * 1000.0, inner_dt * 1000.0)

            stats.chunks += 1
            if stats.status == "quit":
                break

            # Wait for the in-flight async chunk (or, in sync, loop back)
            if async_infer is not None:
                pred = async_infer.wait()
                rtts_ms.append(pred.rtt_ms)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt -- ending attempt")
        stats.status = "quit"
    except Exception as e:
        log.error("attempt crashed: %s", e)
        stats.status = "crash"

    # ---------- summarize ----------
    stats.duration_s = time.perf_counter() - loop_t0
    stats.raw_rtts_ms = list(rtts_ms)
    if rtts_ms:
        a = np.asarray(rtts_ms)
        stats.rtt_ms_mean = float(a.mean())
        stats.rtt_ms_p95  = float(np.percentile(a, 95))
        stats.rtt_ms_max  = float(a.max())
    if horizon_arm_spans:
        a = np.asarray(horizon_arm_spans)
        stats.horizon_arm_mean = float(a.mean())
        stats.horizon_arm_max  = float(a.max())
    if steps_total > 0:
        stats.clip_rate = clipped_total / (STATE_DIM * steps_total)
    return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_observation(top_cam, left_cam, right_cam,
                          left_arm, right_arm,
                          instruction: str) -> YamObservation:
    """Snap one frame from each camera + the current arm state -> YamObservation."""
    return YamObservation(
        images={
            ImageRole.TOP:         np.ascontiguousarray(top_cam.grab()),
            ImageRole.LEFT_WRIST:  np.ascontiguousarray(left_cam.grab()),
            ImageRole.RIGHT_WRIST: np.ascontiguousarray(right_cam.grab()),
        },
        state=read_state(left_arm, right_arm),
        prompt=instruction,
    )


__all__ = ["AttemptKnobs", "AttemptStats", "run_attempt"]
