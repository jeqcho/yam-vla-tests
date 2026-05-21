"""Bimanual YAM client for the MolmoAct2-BimanualYAM RTC server.

Sister script to scripts/yam_client.py, but talks to the RTC-enabled server
at :8203 (host_server_rtc.py) and manages the RTC leftover queue per the
"Real-Time Execution of Action Chunking Flow Policies" recipe
(Black/Galliker/Levine 2025, arXiv:2506.07339):

  - Bootstrap: synchronous first /act (no leftover), produces chunk #1.
  - Steady state, every iteration:
        * Save chunk[exec_horizon : ] as `leftover` for the NEXT request.
        * Kick off the next /act with that leftover (in background thread).
        * Execute chunk[0 : exec_horizon] on the arms.
        * Wait for the next chunk to land; loop.
  - `inference_delay` is set to an EMA of the last few RTTs (in timesteps);
    this tells the policy how many timesteps of the leftover overlap with
    inference time so it can inpaint a smooth handoff.

To avoid forking the safety harness, we import the well-tested helpers from
scripts/yam_client.py: SDK lock fix, camera classes, init_arm, read_state,
safe_command, ramp_to_pose, prompt_journal_entry, write_journal_entry,
load_training_mean_pose, plus the AsyncInferenceFetcher (subclassed below to
add the RTC fields). Run with the i2rt venv:

    /home/andon/yam-tests/i2rt/.venv/bin/python experimental/rtc/yam_client_rtc.py \\
        --left-can can0 --right-can can1 \\
        --top-cam-serial AAAA --left-cam-serial BBBB --right-cam-serial CCCC \\
        --server-url http://127.0.0.1:8203/act \\
        --instruction "first pick up the left orange cube and put it in the box, then pick up the right orange cube and put it in the box" \\
        --train-fps 30 --execution-horizon 10 --max-step-rad 0.05

Safety: same per-tick clip (--max-step-rad / --gripper-step) and same
return-on-exit ramp as scripts/yam_client.py.
"""
from __future__ import annotations

# Import the safety harness from the existing client. NOTE: this triggers
# install_sdk_lock_fix() at import time -- exactly what we want. The
# json_numpy.patch() and logging setup also fire at import.
import os
import sys
from pathlib import Path

# Make scripts/ importable so we can pull in yam_client.py
_SETUP_ROOT = Path(__file__).resolve().parents[2]   # molmoact2-setup
_SCRIPTS_DIR = _SETUP_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# We must import yam_client as a module (NOT as `from yam_client import *`)
# to keep the install_sdk_lock_fix monkey-patch firing before any other
# i2rt imports. The import does its work at module level.
import yam_client as yc  # noqa: E402

# Pull in just the symbols we need. install_sdk_lock_fix has already run
# (it's called at the top of yam_client.py).
from yam_client import (  # noqa: E402
    AsyncInferenceFetcher,
    DEFAULT_GRIPPER_STEP,
    DEFAULT_HORIZON_STRIDE,
    DEFAULT_JOURNAL_PATH,
    DEFAULT_MAX_STEP_RAD,
    DEFAULT_TRAIN_FPS,
    STATE_DIM,
    init_arm,
    load_training_mean_pose,
    make_camera,
    prompt_journal_entry,
    ramp_to_pose,
    read_state,
    safe_command,
    trace,
    write_journal_entry,
    _journal_invocation,
)

import argparse  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from typing import Optional  # noqa: E402

import json_numpy  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402


log = logging.getLogger("yam.rtc.client")


# ---------------------------------------------------------------------------
# Extension of post_actions that also serializes the RTC-specific fields.
# yam_client.post_actions doesn't accept these kwargs, so we redefine here
# rather than monkey-patch upstream.
# ---------------------------------------------------------------------------
def post_actions_rtc(
    server_url: str,
    top: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    state: np.ndarray,
    instruction: str,
    num_steps: int,
    timeout_s: float,
    prev_chunk_left_over: Optional[np.ndarray],
    inference_delay: int,
    execution_horizon: int,
    max_guidance_weight: Optional[float] = None,
    schedule: Optional[str] = None,
    debug: bool = False,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, float, dict]:
    """Round-trip one RTC /act call. Returns (actions[N, D], rtt_ms, server_meta).

    server_meta is the 'rtc' field from the server response (echoes back the
    leftover_len_in / execution_horizon / inference_delay / num_steps / the
    actual max_guidance_weight + schedule used / seed so the client can
    sanity-check what the server actually applied).
    """
    payload = {
        "top_cam": top,
        "left_cam": left_img,
        "right_cam": right_img,
        "instruction": instruction,
        "state": state,
        "num_steps": num_steps,
        "execution_horizon": int(execution_horizon),
        "inference_delay": int(inference_delay),
        "timestamp": time.time(),
    }
    if prev_chunk_left_over is not None and len(prev_chunk_left_over) > 0:
        payload["prev_chunk_left_over"] = np.asarray(
            prev_chunk_left_over, dtype=np.float32
        )
    # Optional per-request RTC hyperparameter overrides. Each is omitted if
    # None; the server uses its boot-time RTCConfig defaults for any missing
    # field.
    if max_guidance_weight is not None:
        payload["max_guidance_weight"] = float(max_guidance_weight)
    if schedule is not None:
        payload["prefix_attention_schedule"] = str(schedule)
    if debug:
        payload["debug"] = True
    if seed is not None:
        payload["seed"] = int(seed)
    body = json_numpy.dumps(payload)
    t0 = time.perf_counter()
    resp = requests.post(
        server_url, data=body,
        headers={"Content-Type": "application/json"},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    out = json_numpy.loads(resp.text)
    if "actions" not in out:
        raise RuntimeError(f"server response missing 'actions': keys={list(out.keys())}")
    actions = np.asarray(out["actions"], dtype=np.float32)
    server_dt_ms = float(out.get("dt_ms", 0.0))
    server_meta = out.get("rtc", {}) if isinstance(out.get("rtc"), dict) else {}
    rtt_ms = (time.perf_counter() - t0) * 1000.0
    log.debug(
        "RTC server dt=%.1f ms, rtt=%.1f ms, shape=%s, server_meta=%s",
        server_dt_ms, rtt_ms, actions.shape, server_meta,
    )
    return actions, rtt_ms, server_meta


class AsyncRTCFetcher:
    """Like AsyncInferenceFetcher but carries the RTC kwargs.

    We don't subclass yam_client.AsyncInferenceFetcher because its kick_off()
    signature doesn't accept the extra args; better to duplicate the small
    bit of plumbing than fight the parent class.
    """

    def __init__(self, server_url: str, instruction: str, num_steps: int,
                 timeout_s: float):
        self._url = server_url
        self._instr = instruction
        self._num_steps = num_steps
        self._timeout_s = timeout_s
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[tuple] = None
        self._error: Optional[str] = None

    def kick_off(
        self,
        state: np.ndarray,
        top_img: np.ndarray,
        left_img: np.ndarray,
        right_img: np.ndarray,
        prev_chunk_left_over: Optional[np.ndarray],
        inference_delay: int,
        execution_horizon: int,
        max_guidance_weight: Optional[float] = None,
        schedule: Optional[str] = None,
        debug: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("AsyncRTCFetcher: previous request still in flight")
        # Snapshot inputs so the caller can mutate them.
        s = np.ascontiguousarray(state, dtype=np.float32).copy()
        t = np.ascontiguousarray(top_img).copy()
        l = np.ascontiguousarray(left_img).copy()
        r = np.ascontiguousarray(right_img).copy()
        pcl = None
        if prev_chunk_left_over is not None and len(prev_chunk_left_over) > 0:
            pcl = np.ascontiguousarray(prev_chunk_left_over, dtype=np.float32).copy()
        idelay = int(inference_delay)
        exh = int(execution_horizon)
        mgw = float(max_guidance_weight) if max_guidance_weight is not None else None
        sched = schedule
        dbg = bool(debug)
        seed_v = int(seed) if seed is not None else None

        self._result = None
        self._error = None

        def _worker():
            try:
                actions, rtt_ms, meta = post_actions_rtc(
                    self._url, t, l, r, s, self._instr, self._num_steps,
                    self._timeout_s, pcl, idelay, exh,
                    max_guidance_weight=mgw, schedule=sched,
                    debug=dbg, seed=seed_v,
                )
                self._result = (actions, rtt_ms, meta)
            except BaseException as e:  # noqa: BLE001
                self._error = repr(e)
        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def wait_for_result(self) -> tuple:
        if self._thread is None:
            raise RuntimeError("AsyncRTCFetcher.wait_for_result with no in-flight request")
        self._thread.join()
        self._thread = None
        if self._error is not None:
            raise RuntimeError(f"async /act failed: {self._error}")
        return self._result


def main() -> None:
    journal_start_s = time.time()
    journal_invocation = _journal_invocation()

    p = argparse.ArgumentParser(description="MolmoAct2-BimanualYAM RTC client")
    # ----- the same hardware/camera surface as yam_client.py --------------
    p.add_argument("--left-can", default="can0")
    p.add_argument("--right-can", default="can1")
    p.add_argument("--left-gripper", default="linear_4310",
                   choices=["crank_4310", "linear_3507", "linear_4310", "flexible_4310"])
    p.add_argument("--right-gripper", default="linear_4310",
                   choices=["crank_4310", "linear_3507", "linear_4310", "flexible_4310"])
    p.add_argument("--top-cam-serial", default=None)
    p.add_argument("--top-cam-v4l2", default=None)
    p.add_argument("--left-cam-serial", default=None)
    p.add_argument("--left-cam-v4l2", default=None)
    p.add_argument("--right-cam-serial", default=None)
    p.add_argument("--right-cam-v4l2", default=None)
    p.add_argument("--cam-width", type=int, default=424)
    p.add_argument("--cam-height", type=int, default=240)
    p.add_argument("--cam-fps", type=int, default=30)
    # ----- server / inference ---------------------------------------------
    p.add_argument("--server-url", default="http://127.0.0.1:8203/act",
                   help="Defaults to the RTC server (:8203). Point at :8202 only "
                        "if you want to A/B test against the non-RTC server -- the "
                        "non-RTC server will ignore the prev_chunk_left_over field.")
    p.add_argument("--instruction", required=True)
    p.add_argument("--train-fps", type=float, default=DEFAULT_TRAIN_FPS,
                   help="Policy training cadence (inner-loop pace, ticks/sec)")
    p.add_argument("--num-steps", type=int, default=10,
                   help="Flow-matching steps per chunk (server-side)")
    p.add_argument("--max-step-rad", type=float, default=DEFAULT_MAX_STEP_RAD,
                   help="Per-tick per-arm-joint clip (rad). Pass 0 to disable.")
    p.add_argument("--gripper-step", type=float, default=DEFAULT_GRIPPER_STEP,
                   help="Per-tick gripper clip. Pass 0 to disable.")
    # ----- RTC-specific ---------------------------------------------------
    p.add_argument(
        "--execution-horizon", type=int, default=DEFAULT_HORIZON_STRIDE,
        help="BOOTSTRAP value for execution_horizon, used only for the very "
             "first /act call (before we have an RTT measurement). After "
             "that, the client adapts execution_horizon to "
             "ceil(EMA(RTT) / dt), clamped to [--rtc-min-horizon, "
             "--rtc-max-horizon]. Per the RTC paper's canonical regime, "
             "execution_horizon and inference_delay are always equal in "
             "this client -- both control the wall-clock alignment of the "
             "leftover prefix.",
    )
    p.add_argument(
        "--rtc-min-horizon", type=int, default=5,
        help="Minimum value of the adaptive execution_horizon. Stops the "
             "system from replanning too aggressively if RTT briefly dips "
             "(< rtc_min_horizon * dt). Default 5 (~166 ms at 30 Hz).",
    )
    p.add_argument(
        "--rtc-max-horizon", type=int, default=20,
        help="Maximum value of the adaptive execution_horizon. Stops the "
             "system from going too far open-loop if RTT spikes "
             "(> rtc_max_horizon * dt). Default 20 (~666 ms at 30 Hz). "
             "Must be < chunk_size (=30) so the leftover has positions "
             "to inpaint over.",
    )
    p.add_argument(
        "--inference-delay-mode", default="ema-rtt",
        choices=["ema-rtt", "fixed", "zero"],
        help="How to drive the adaptive horizon. "
             "ema-rtt (default): execution_horizon = inference_delay = "
             "ceil(EMA(RTT)/dt), as the RTC paper assumes. "
             "fixed: both quantities pinned to --inference-delay-fixed. "
             "zero: inference_delay forced to 0 (DIAGNOSTIC ABLATION -- "
             "disables prefix anchoring entirely, degenerates to vanilla "
             "chunked inference; useful for separating RTC's contribution "
             "from raw async-chunking).",
    )
    p.add_argument(
        "--inference-delay-fixed", type=int, default=10,
        help="If --inference-delay-mode=fixed, use this constant for both "
             "execution_horizon and inference_delay.",
    )
    p.add_argument(
        "--inference-delay-ema-alpha", type=float, default=0.5,
        help="EMA smoothing factor for the RTT estimate (0..1, higher = "
             "more reactive to recent latency).",
    )
    p.add_argument(
        "--rtc-max-guidance-weight", type=float, default=None,
        help="Per-request override for RTCConfig.max_guidance_weight. If "
             "unset, uses the server's RTCConfig default (10.0). Higher = "
             "tighter prefix anchoring; lower = more model freedom near "
             "chunk boundaries.",
    )
    p.add_argument(
        "--rtc-schedule", default=None, choices=[None, "linear", "exp", "zeros", "ones"],
        help="Per-request override for prefix_attention_schedule. If unset, "
             "uses the server's default (LINEAR, the paper's recommendation).",
    )
    p.add_argument(
        "--rtc-debug", action="store_true",
        help="Set RTCConfig.debug=True for each request, so the server's "
             "tracker records per-step intermediate state. Useful when "
             "something looks wrong; off by default for speed.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="If set, each /act call uses a torch.Generator seeded with "
             "this value for deterministic flow-matching initial noise. "
             "Useful for debugging; in production leave unset for diverse "
             "rollouts.",
    )
    # ----- ramp / safety --------------------------------------------------
    p.add_argument("--move-to-ready", action="store_true",
                   help="Before inference, ramp arms to the MolmoAct2 training-mean pose.")
    p.add_argument("--ramp-duration-s", type=float, default=5.0)
    p.add_argument("--no-return-on-exit", action="store_true",
                   help="DANGEROUS: skip return-to-startup ramp at exit.")
    # ----- timeouts / misc ------------------------------------------------
    p.add_argument("--timeout-s", type=float, default=15.0)
    p.add_argument("--warmup-timeout-s", type=float, default=60.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Don't command the arms; print actions only.")
    p.add_argument("--no-journal", action="store_true")
    p.add_argument("--journal-path", default=DEFAULT_JOURNAL_PATH)
    args = p.parse_args()

    # Loud warning if both clips disabled.
    if args.max_step_rad <= 0 and args.gripper_step <= 0:
        log.warning("=" * 70)
        log.warning("PER-STEP CLIPPING DISABLED (both arm + gripper). Raw policy")
        log.warning("output will go straight to motors. Only i2rt's 400 ms")
        log.warning("watchdog provides any safety net.")
        log.warning("=" * 70)

    # ---- Health-check server -----------------------------------------------
    try:
        r = requests.get(args.server_url, timeout=3.0)
        r.raise_for_status()
        health = r.json()
        log.info("RTC server health: %s", health)
        # Confirm we're talking to the RTC server, not :8202. The RTC server
        # advertises an "rtc" field; the legacy server doesn't.
        if "rtc" not in health:
            log.warning(
                "server at %s does NOT advertise rtc support. If you meant to "
                "use the RTC server, check the URL (default :8203). Continuing "
                "anyway -- prev_chunk_left_over will be silently ignored.",
                args.server_url,
            )
        else:
            log.info(
                "Confirmed RTC server: exec_horizon=%s, schedule=%s, max_guidance=%s",
                health["rtc"].get("execution_horizon"),
                health["rtc"].get("schedule"),
                health["rtc"].get("max_guidance_weight"),
            )
            # The server's reported execution_horizon is just the RTCConfig
            # default; the client overrides it on every request with the
            # adaptive value from _compute_horizon_and_delay(). So a mismatch
            # at boot is expected and not a warning condition.
    except Exception as e:
        log.error("server health check failed at %s: %s", args.server_url, e)
        sys.exit(2)

    # ---- Cameras BEFORE arms (USB IRQ contention) --------------------------
    cam_kw = dict(width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
    trace(f"building cameras at {args.cam_width}x{args.cam_height}/{args.cam_fps}fps")
    top = cam_l = cam_r = None
    left = right = None
    try:
        top   = make_camera("top",   args.top_cam_serial,   args.top_cam_v4l2,   **cam_kw)
        cam_l = make_camera("left",  args.left_cam_serial,  args.left_cam_v4l2,  **cam_kw)
        cam_r = make_camera("right", args.right_cam_serial, args.right_cam_v4l2, **cam_kw)
        for c in (top, cam_l, cam_r):
            trace(f"starting camera {c.name}")
            c.start()
        for _ in range(3):
            top.grab(); cam_l.grab(); cam_r.grab()
        trace("cameras streaming, USB quiet -- safe to init arms")
    except Exception:
        for c in (top, cam_l, cam_r):
            if c is not None:
                try: c.stop()
                except Exception: pass
        raise

    trace("about to init LEFT arm")
    left = init_arm(args.left_can, args.left_gripper)
    trace("about to init RIGHT arm")
    right = init_arm(args.right_can, args.right_gripper)
    trace("both arms initialized")

    # Capture startup pose for safe return-on-exit.
    startup_pose = np.concatenate([
        np.asarray(left.get_joint_pos(), dtype=np.float32),
        np.asarray(right.get_joint_pos(), dtype=np.float32),
    ])
    log.info(
        "Captured startup pose for return-on-exit: %s",
        np.array2string(startup_pose, precision=3),
    )

    if args.move_to_ready:
        target = load_training_mean_pose()
        target[6] = startup_pose[6]
        target[13] = startup_pose[13]
        log.info("--move-to-ready: ramping arms to training-mean pose (%.1fs)...",
                 args.ramp_duration_s)
        ramp_to_pose(left, right, target, duration_s=args.ramp_duration_s,
                     label="move-to-ready")

    inner_dt = 1.0 / args.train_fps
    log.info(
        "RTC control loop: train_fps=%.1f Hz, execution_horizon=%d, "
        "inference_delay_mode=%s, instruction=%r",
        args.train_fps, args.execution_horizon, args.inference_delay_mode,
        args.instruction,
    )

    # ---- Warmup the server once at the real image shape --------------------
    try:
        state = read_state(left, right)
        log.info("Warming up server (timeout=%.0fs)...", args.warmup_timeout_s)
        _wu_actions, _wu_rtt, _wu_meta = post_actions_rtc(
            args.server_url, top.grab(), cam_l.grab(), cam_r.grab(), state,
            args.instruction, args.num_steps, args.warmup_timeout_s,
            prev_chunk_left_over=None, inference_delay=0,
            execution_horizon=args.execution_horizon,
        )
        log.info(
            "Server warmup OK (rtt=%.0f ms, actions shape=%s, server_meta=%s)",
            _wu_rtt, _wu_actions.shape, _wu_meta,
        )
    except Exception as e:
        log.error("server warmup failed: %s. Continuing anyway.", e)

    # ---- Adaptive horizon estimator -----------------------------------------
    # The RTC paper assumes execution_horizon == inference_delay in steady
    # state -- the new chunk is generated to start exactly where the
    # currently-executing chunk would be after `inference_delay` timesteps,
    # and we hand it off the moment the previous chunk reaches that point.
    # That assumption is correct iff
    #
    #     execution_horizon * dt  ==  RTT  ==  inference_delay * dt
    #
    # If they disagree, the leftover slice (= current_chunk[execution_horizon:])
    # refers to a wall-clock instant different from what chunk_N+1[0]
    # corresponds to, and the prefix attention anchors to the wrong
    # timestep. So we make both adapt to measured RTT:
    #
    #     execution_horizon = inference_delay = ceil(RTT_ema / dt)
    #
    # clamped to [rtc_min_horizon, rtc_max_horizon] so we never replan too
    # often (causing thrash) or too rarely (long open-loop windows). Both
    # ends are exposed as CLI flags below.
    rtt_ema_ms: Optional[float] = None

    def _compute_horizon_and_delay() -> tuple[int, int]:
        """Return (execution_horizon, inference_delay), always equal in this
        client. Falls back to args.execution_horizon (the bootstrap value)
        until rtt_ema_ms has been initialized.
        """
        if args.inference_delay_mode == "fixed":
            h = max(1, int(args.inference_delay_fixed))
            return h, h
        if args.inference_delay_mode == "zero":
            # Diagnostic ablation: degenerate to non-prefix behaviour.
            # inference_delay=0 means no prefix anchoring; execution_horizon
            # still controls how many actions we play per chunk.
            return max(1, int(args.execution_horizon)), 0
        # ema-rtt (default): both quantities track ceil(RTT / dt).
        if rtt_ema_ms is None:
            h = max(1, int(args.execution_horizon))
            return h, h
        dt_ms = inner_dt * 1000.0
        h = int(np.ceil(rtt_ema_ms / dt_ms))
        h = max(args.rtc_min_horizon, min(h, args.rtc_max_horizon))
        return h, h

    # ---- RTC state machine --------------------------------------------------
    # We always run async (RTC's whole point is to overlap inference with
    # execution). There's no sync mode here -- if you want sync, use the
    # legacy yam_client.
    fetcher = AsyncRTCFetcher(
        args.server_url, args.instruction, args.num_steps, args.timeout_s,
    )

    # Bootstrap: synchronous first chunk, no leftover.
    log.info("RTC bootstrap: fetching initial chunk (no leftover)...")
    bootstrap_state = read_state(left, right)
    bootstrap_top = top.grab()
    bootstrap_left = cam_l.grab()
    bootstrap_right = cam_r.grab()
    fetcher.kick_off(
        bootstrap_state, bootstrap_top, bootstrap_left, bootstrap_right,
        prev_chunk_left_over=None, inference_delay=0,
        execution_horizon=args.execution_horizon,
        max_guidance_weight=args.rtc_max_guidance_weight,
        schedule=args.rtc_schedule,
        debug=args.rtc_debug,
        seed=args.seed,
    )
    current_chunk, last_rtt_ms, last_meta = fetcher.wait_for_result()
    rtt_ema_ms = float(last_rtt_ms)
    log.info(
        "Bootstrap chunk OK: shape=%s, rtt=%.0fms, meta=%s",
        current_chunk.shape, last_rtt_ms, last_meta,
    )

    stop_flag = {"stop": False}
    boundary_idx = 0
    last_chunk_tail: Optional[np.ndarray] = None

    try:
        while not stop_flag["stop"]:
            # Adaptive horizon: both quantities derived from measured RTT.
            # execution_horizon controls BOTH where we slice the leftover
            # AND how many actions we play before re-querying. Keeping them
            # equal to inference_delay ensures wall-clock alignment.
            exec_horizon, inference_delay = _compute_horizon_and_delay()
            chunk_len = current_chunk.shape[0]
            n_to_play = min(exec_horizon, chunk_len)

            # ---- Prepare next-request inputs (NOW, before executing) ------
            # We send the NEXT chunk's leftover prefix
            # (= current_chunk[exec_horizon:]) so the policy can inpaint a
            # smooth handoff. The state is sampled at the moment of kick-off
            # (= the wall-clock instant chunk_N+1[0] represents in the
            # canonical RTC regime).
            next_state = read_state(left, right)
            next_top = top.grab()
            next_left = cam_l.grab()
            next_right = cam_r.grab()

            # Leftover = unexecuted tail of the current chunk. Shape:
            # (chunk_size - exec_horizon, action_dim). RTC processor pads
            # this back up to chunk_size with zeros internally.
            leftover = (
                current_chunk[exec_horizon:].astype(np.float32, copy=True)
                if chunk_len > exec_horizon else None
            )

            # ---- Per-query diagnostics on the CURRENT chunk ---------------
            def _arm_delta_max(a_idx: int) -> float:
                d = current_chunk[a_idx] - next_state
                return float(max(np.max(np.abs(d[:6])), np.max(np.abs(d[7:13]))))
            a0_d  = _arm_delta_max(0)
            a5_d  = _arm_delta_max(min(5, chunk_len - 1))
            a10_d = _arm_delta_max(min(10, chunk_len - 1))
            a19_d = _arm_delta_max(min(19, chunk_len - 1))
            aN_d  = _arm_delta_max(chunk_len - 1)
            horizon_range = current_chunk.max(axis=0) - current_chunk.min(axis=0)
            horizon_arm_span = float(max(
                np.max(horizon_range[:6]), np.max(horizon_range[7:13]),
            ))
            log.info(
                "/act rtt=%dms  arm |a[i]-state|_max @ 0/5/10/19/last: "
                "%.3f/%.3f/%.3f/%.3f/%.3f rad  horizon_span=%.3f rad  "
                "L_grip[0,last]=%.2f,%.2f  R_grip[0,last]=%.2f,%.2f  "
                "leftover_len_out=%d  inference_delay=%d",
                last_rtt_ms, a0_d, a5_d, a10_d, a19_d, aN_d, horizon_arm_span,
                current_chunk[0][6], current_chunk[-1][6],
                current_chunk[0][13], current_chunk[-1][13],
                0 if leftover is None else len(leftover),
                inference_delay,
            )

            # ---- Boundary telemetry --------------------------------------
            # state_vs_a0: how far the about-to-execute action is from the
            # arm's actual position right now. RTC's whole point is to keep
            # this small even when inference is slow.
            # tail_vs_a0: how far the new chunk's a[0] is from the LAST raw
            # action we sent (= last_chunk_tail). RTC should also keep this
            # small (prefix attention to leftover).
            if last_chunk_tail is not None:
                arm_idx = np.r_[0:6, 7:13]
                a0 = current_chunk[0]
                state_vs_a0_arm = float(np.max(np.abs(a0[arm_idx] - next_state[arm_idx])))
                tail_vs_a0_arm = float(np.max(np.abs(a0[arm_idx] - last_chunk_tail[arm_idx])))
                state_vs_a0_grip_l = float(abs(a0[6] - next_state[6]))
                state_vs_a0_grip_r = float(abs(a0[13] - next_state[13]))
                boundary_idx += 1
                log.info(
                    "[rtc-boundary] #%d  state_vs_a0(arm)=%.3f rad  "
                    "tail_vs_a0(arm)=%.3f rad  state_vs_a0(grip L,R)=%.2f,%.2f",
                    boundary_idx, state_vs_a0_arm, tail_vs_a0_arm,
                    state_vs_a0_grip_l, state_vs_a0_grip_r,
                )

            # ---- Kick off the next /act in the background ----------------
            try:
                fetcher.kick_off(
                    next_state, next_top, next_left, next_right,
                    prev_chunk_left_over=leftover,
                    inference_delay=inference_delay,
                    execution_horizon=exec_horizon,
                    max_guidance_weight=args.rtc_max_guidance_weight,
                    schedule=args.rtc_schedule,
                    debug=args.rtc_debug,
                    seed=args.seed,
                )
            except RuntimeError as e:
                log.error("kick_off failed: %s", e)
                break

            # ---- Execute current_chunk[0 : exec_horizon] -----------------
            clipped_this_query = 0
            steps_this_query = 0
            for i in range(n_to_play):
                if stop_flag["stop"]:
                    break
                step_start = time.perf_counter()
                desired = current_chunk[i].astype(np.float32)
                if args.dry_run:
                    log.info("dry-run action[%d]: %s", i,
                             np.array2string(desired, precision=3))
                else:
                    state = read_state(left, right)
                    _, n_clipped = safe_command(
                        left, right, state, desired,
                        args.max_step_rad, args.gripper_step,
                    )
                    clipped_this_query += n_clipped
                    steps_this_query += 1
                sleep_left = inner_dt - (time.perf_counter() - step_start)
                if sleep_left > 0:
                    time.sleep(sleep_left)
                elif sleep_left < -0.050:
                    log.warning(
                        "inner step overrun by %.1f ms (target %.1f ms)",
                        -sleep_left * 1000.0, inner_dt * 1000.0,
                    )
            if steps_this_query > 0 and (args.max_step_rad > 0 or args.gripper_step > 0):
                max_possible = STATE_DIM * steps_this_query
                if clipped_this_query > 0:
                    pct = 100.0 * clipped_this_query / max_possible
                    log.info(
                        "clip: %d/%d dim-steps clipped (%.1f%%) "
                        "[--max-step-rad=%.3f --gripper-step=%.3f]",
                        clipped_this_query, max_possible, pct,
                        args.max_step_rad, args.gripper_step,
                    )

            # Stash the last raw action we played for the next boundary check.
            last_chunk_tail = current_chunk[n_to_play - 1].astype(np.float32).copy()

            # ---- Wait for the next chunk (in-flight) ---------------------
            current_chunk, last_rtt_ms, last_meta = fetcher.wait_for_result()
            # Update the RTT EMA.
            alpha = float(args.inference_delay_ema_alpha)
            rtt_ema_ms = alpha * float(last_rtt_ms) + (1.0 - alpha) * float(rtt_ema_ms)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt -- shutting down")
    finally:
        # Journal first (Ctrl-C-safe; see scripts/yam_client.py).
        try:
            entry = prompt_journal_entry(journal_start_s, args)
            if entry is not None:
                write_journal_entry(args.journal_path, entry, args, journal_invocation)
        except Exception as e:
            log.warning("journal step failed: %s", e)

        # Safe return-to-startup ramp; same Ctrl-C-twice escape as yam_client.
        abort = {"abort": False, "ctrlc_count": 0}
        def _cleanup_sigint(_sig, _frame):
            abort["ctrlc_count"] += 1
            if abort["ctrlc_count"] == 1:
                log.warning(
                    "Ctrl-C in cleanup: aborting return-ramp. ARMS WILL DROP. "
                    "Ctrl-C again to hard-exit."
                )
                abort["abort"] = True
            else:
                os._exit(130)
        try:
            signal.signal(signal.SIGINT, _cleanup_sigint)
        except Exception:
            pass

        if (left is not None and right is not None
                and 'startup_pose' in locals()
                and not args.no_return_on_exit):
            try:
                log.info("Returning arms to startup pose (%.1fs ramp)...",
                         args.ramp_duration_s)
                ramp_to_pose(
                    left, right, startup_pose,
                    duration_s=args.ramp_duration_s,
                    abort_flag=abort,
                    label="return-on-exit",
                )
                if abort["abort"]:
                    log.warning("return ramp aborted -- arms may be mid-trajectory")
            except BaseException as e:
                log.warning("return ramp failed: %s. ARMS MAY DROP.", e)
        elif args.no_return_on_exit:
            log.warning("--no-return-on-exit set: skipping return ramp.")

        log.info("Stopping cameras")
        for c in (top, cam_l, cam_r):
            try:
                if c is not None: c.stop()
            except BaseException as e:
                log.warning("camera stop failed: %s", e)

        log.info("Closing arm SDKs (motors will lose torque now)")
        for arm in (left, right):
            if arm is None: continue
            try:
                arm.close()
            except BaseException as e:
                log.warning("arm.close() failed: %s", e)
        log.info("Done.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
