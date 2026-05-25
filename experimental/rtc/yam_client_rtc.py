"""Bimanual YAM client for the MolmoAct2-BimanualYAM RTC server.

Sister script to scripts/yam_client.py. Implements the executor side of
Real-Time Chunking (RTC) per Black/Galliker/Levine 2025, arXiv:2506.07339,
talking to host_server_rtc.py on :8203. Mirrors lerobot's ActionQueue
semantics (which the client venv can't import directly because i2rt is
pinned to Python 3.11 and lerobot wants 3.12) with a tiny ClientActionQueue.

Loop architecture (single thread, tick-driven):

    every dt:
        action = queue.pop()                 # one action per tick
        safe_command(action)                 # clipped per --max-step-rad
        if inference_in_flight and fetcher.done():
            new_chunk, rtt = fetcher.collect()
            real_delay = tick_counter - inference_start_tick
            delay_buffer.append(real_delay)               # in ticks
            queue.replace(new_chunk, real_delay)          # discards stale prefix
            ticks_since_last_inference = 0
            trigger_threshold = max(max(delay_buffer), s_min)   # paper s = max(d, s_min)
        if not inference_in_flight and ticks_since_last_inference >= trigger_threshold:
            kick_off(state, leftover=queue.get_left_over(),
                     inference_delay=max(delay_buffer))

Paper-faithful invariants:

  * queue.replace(chunk, real_delay) drops chunk[:real_delay] -- those
    positions correspond to wall-clock that already elapsed during
    inference (paper Alg. 1 Swap step; lerobot _replace_actions_queue).
  * leftover sent to the server is the literal unconsumed tail of the
    in-flight chunk (`original_queue[last_index:]`), NOT a pre-emptive
    slice. Jitter-robust.
  * execution_horizon and inference_delay are INDEPENDENT. The CLI flag
    --execution-horizon corresponds to paper's `s_min` (the user-provided
    floor on the per-cycle execution horizon, Alg. 1 line 11), NOT paper's
    `s` directly. The per-request paper_s = H - len(leftover) = (in steady
    state) real_delay_prev + s_min. inference_delay (= paper's d, predicted
    from RTT buffer) is the wire field we send per-request. lerobot's
    `execution_horizon` API arg is set to len(leftover) so the mask fade
    region ends at the actual leftover boundary (= H - paper_s).
  * inference_delay estimator uses max(delay_buffer) (paper Alg. 1
    line 18, `d = max(Q)`), where Q is a deque of past real_delay values
    measured in CONTROLLER TICKS (paper Alg 1 line 23 pushes `t`, the
    tick counter at completion -- NOT RTT in ms). Conservative under
    spikes. Bootstrap seeds the buffer with ceil(boot_rtt_ms/dt_ms).

Run with the i2rt venv:

    /home/andon/yam-tests/i2rt/.venv/bin/python experimental/rtc/yam_client_rtc.py \\
        --left-can can0 --right-can can1 \\
        --top-cam-serial AAAA --left-cam-serial BBBB --right-cam-serial CCCC \\
        --server-url http://127.0.0.1:8203/act \\
        --instruction "..." \\
        --train-fps 30 --execution-horizon 8 --max-step-rad 0.05

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
# Legacy yam_client.py sets the root logger to WARNING and only raises
# `yam.client` to INFO; without this, all our INFO/[summary]/[chunk]/
# [rtc-bound] lines get silently dropped. Match the legacy client's level.
log.setLevel(logging.INFO)


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
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Round-trip one RTC /act call. Returns
    (actions[N,D], actions_raw[N,D], rtt_ms, server_meta).

    Two arrays come back:
      * actions      -- de-normalized joint-space actions to command the arms.
      * actions_raw  -- raw NORMALIZED policy output, to be sent back as the
                        next call's `prev_chunk_left_over`. The model's RTC
                        prefix-attention expects the leftover in normalized
                        latent space (same scale as the flow-matching
                        trajectory), so we round-trip the raw form.

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
    # actions_raw: normalized form to be sent back as next leftover. If
    # absent (e.g. talking to an OLDER server build that doesn't emit it),
    # fall back to the processed form -- behavior degrades to pre-fix RTC
    # but doesn't crash.
    if "actions_raw" in out:
        actions_raw = np.asarray(out["actions_raw"], dtype=np.float32)
    else:
        log.warning(
            "server response missing 'actions_raw' -- falling back to "
            "processed actions as the leftover. RTC anchoring will be in "
            "the wrong space. Update the server to emit actions_raw."
        )
        actions_raw = actions.copy()
    server_dt_ms = float(out.get("dt_ms", 0.0))
    server_meta = out.get("rtc", {}) if isinstance(out.get("rtc"), dict) else {}
    rtt_ms = (time.perf_counter() - t0) * 1000.0
    log.debug(
        "RTC server dt=%.1f ms, rtt=%.1f ms, shape=%s, server_meta=%s",
        server_dt_ms, rtt_ms, actions.shape, server_meta,
    )
    return actions, actions_raw, rtt_ms, server_meta


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
                actions, actions_raw, rtt_ms, meta = post_actions_rtc(
                    self._url, t, l, r, s, self._instr, self._num_steps,
                    self._timeout_s, pcl, idelay, exh,
                    max_guidance_weight=mgw, schedule=sched,
                    debug=dbg, seed=seed_v,
                )
                self._result = (actions, actions_raw, rtt_ms, meta)
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

    def done(self) -> bool:
        """Non-blocking: True if a request is in-flight and finished, OR if
        no request is in-flight (i.e., wait_for_result is safe to call only
        when there IS an in-flight request, which the caller tracks
        separately via `inference_in_flight`)."""
        return self._thread is not None and not self._thread.is_alive()


class ClientActionQueue:
    """Tick-driven action queue mirroring lerobot.policies.rtc.action_queue.
    ActionQueue's RTC-enabled mode. We can't import lerobot in this venv
    (Python 3.11 vs lerobot's 3.12 requirement) so we reimplement the
    minimum needed in pure numpy.

    Two parallel queues (paper Alg. 1 + lerobot action_queue.py:175-194):

      * queue          -- POST-processed (de-normalized joint space) --
                          consumed by pop() to command the arms.
      * original_queue -- RAW (normalized latent space) -- returned by
                          get_left_over() and sent back as the next call's
                          prev_chunk_left_over. The model's RTC inpainting
                          expects the leftover in the same NORMALIZED space
                          as the flow-matching trajectory, not de-normalized
                          joint space (see lerobot _generate_actions_from_
                          inputs_with_rtc in modeling_molmoact2.py).

    Three semantics to honor verbatim:

      1. `replace(processed, original, real_delay)` drops first
         clamped_delay positions from BOTH queues. Those positions
         correspond to wall-clock time that already elapsed during
         inference (their mask anchors were to the leftover we *just
         executed*; replaying them sends the arm back in time -- this is
         the boundary-jump bug the paper exists to fix).
      2. `get_left_over()` returns `original_queue[last_index:]` -- the
         truly unconsumed tail of the in-flight chunk at the moment of
         the call.
      3. last_index advances on every pop() and resets to 0 on replace().

    Single-threaded model. The client's main loop owns this object; no
    threading lock needed.
    """

    def __init__(self) -> None:
        self.queue: Optional[np.ndarray] = None           # processed
        self.original_queue: Optional[np.ndarray] = None  # raw / normalized
        self.last_index: int = 0

    def replace(self, processed: np.ndarray, original: np.ndarray, real_delay: int) -> int:
        """Set both queues to `*[real_delay:]`. Returns the clamped
        delay actually applied (= number of leading positions dropped).
        Clamp matches lerobot _replace_actions_queue: real_delay is
        clamped to [0, min(len(processed), len(original))].
        """
        if processed is None or len(processed) == 0:
            self.queue = None
            self.original_queue = None
            self.last_index = 0
            return 0
        if original is None or len(original) == 0:
            # Fallback: use processed as both (raw round-trip won't anchor
            # correctly but the loop still runs).
            original = processed
        n = min(len(processed), len(original))
        clamped = max(0, min(int(real_delay), n))
        self.queue = np.asarray(processed[clamped:n], dtype=np.float32).copy()
        self.original_queue = np.asarray(original[clamped:n], dtype=np.float32).copy()
        self.last_index = 0
        return clamped

    def pop(self) -> Optional[np.ndarray]:
        """Pop the next processed action and advance last_index. Returns
        None if the queue is empty (caller should hold position)."""
        if self.queue is None or self.last_index >= len(self.queue):
            return None
        action = self.queue[self.last_index].copy()
        self.last_index += 1
        return action

    def get_left_over(self) -> Optional[np.ndarray]:
        """Return the currently unconsumed RAW tail (= what to send as
        prev_chunk_left_over), or None if nothing unconsumed. Mirrors
        lerobot ActionQueue.get_left_over: returns
        original_queue[last_index:]."""
        if self.original_queue is None or self.last_index >= len(self.original_queue):
            return None
        return self.original_queue[self.last_index:].copy()

    def qsize(self) -> int:
        if self.queue is None:
            return 0
        return max(0, len(self.queue) - self.last_index)

    def empty(self) -> bool:
        return self.qsize() == 0


def _predict_inference_delay_ticks(
    delay_buffer: "deque[int]", d_min: int, d_max: int,
    fixed: Optional[int],
) -> int:
    """Estimate inference_delay (in ticks) per RTC paper Alg. 1 line 18:
    d = max(Q) where Q is a fixed-size buffer of recent inference delays
    measured in controller ticks (paper Alg. 1 line 23 enqueues `t`, the
    GetAction counter when inference completes, NOT RTT in ms). For us
    `t` = real_delay = tick_counter - inference_start_tick at merge.
    Conservative: take the worst recent value so we don't under-predict
    under spikes.

    Returns ticks (an int in [d_min, d_max]). If `fixed` is provided,
    returns it (clamped to [d_min, d_max]). If the buffer is empty,
    returns d_min.
    """
    if fixed is not None:
        return max(d_min, min(int(fixed), d_max))
    if len(delay_buffer) == 0:
        return d_min
    ticks = int(max(delay_buffer))
    return max(d_min, min(ticks, d_max))


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
        "--execution-horizon", type=int, default=8,
        help="Paper's s_min (Alg. 1 line 11): how many ticks since the last "
             "merge before the client kicks off the next inference. "
             "Default 8 (~266 ms at 30 Hz). Fixed for the whole rollout. "
             "Trigger: ticks_since_last_inference >= this value. "
             "NOTE: this is NOT paper's `s` in the mask; the per-request "
             "paper_s = H - len(leftover), which in steady state equals "
             "real_delay_prev + this CLI value. Paper's hard constraint is "
             "d ≤ H - s (Sec 2), enforced per-request by clamping d to "
             "len(leftover). Paper hints s ≈ H/2 but H/4 (=8) gives more "
             "room for d-spikes without underflowing the queue.",
    )
    p.add_argument(
        "--rtc-rtt-buffer-size", type=int, default=8,
        help="Size of the recent-RTT buffer used to estimate inference_delay "
             "as max(buffer) / dt (paper Alg. 1 line 18). Bigger = more "
             "conservative under sporadic spikes; smaller = quicker to "
             "track latency improvements. Default 8 (~last 8 inferences).",
    )
    p.add_argument(
        "--rtc-inference-delay-fixed", type=int, default=None,
        help="If set, override the max-of-buffer estimator and pin "
             "inference_delay to this constant (ticks). Useful for "
             "reproducing a specific paper-aligned setup or for ablation.",
    )
    p.add_argument(
        "--rtc-zero-delay", action="store_true",
        help="DIAGNOSTIC ABLATION: force inference_delay = 0. The mask's "
             "frozen prefix region is empty, so the new chunk is anchored "
             "to the leftover ONLY via the fade region. Degenerates RTC "
             "toward vanilla chunked inference; useful for separating "
             "RTC's contribution from raw async-chunking.",
    )
    p.add_argument(
        "--rtc-min-inference-delay", type=int, default=1,
        help="Lower clamp on inference_delay. Default 1.",
    )
    p.add_argument(
        "--rtc-max-inference-delay", type=int, default=None,
        help="Upper clamp on inference_delay. Default (None) = auto-compute "
             "as (chunk_size - execution_horizon) // 2 so the per-cycle "
             "consumption s + d <= H - d (paper Sec 2 constraint). With "
             "H=30, s=8, this defaults to 11.",
    )
    p.add_argument(
        "--rtc-max-guidance-weight", type=float, default=None,
        help="Per-request override for RTCConfig.max_guidance_weight (β). "
             "If unset, uses the server's default (10.0; this matches "
             "lerobot's RTCConfig default, which the paper does not pin "
             "to a specific value). Higher β = tighter prefix anchoring; "
             "lower = more model freedom near chunk boundaries.",
    )
    p.add_argument(
        "--rtc-schedule", default=None, choices=[None, "linear", "exp", "zeros", "ones"],
        help="Per-request override for prefix_attention_schedule. If unset, "
             "uses the server's default (EXP, matching paper Eq. 5).",
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
            # boot-time default; the client overrides it per request with
            # `len(leftover)` (= mask fade end). So a mismatch at boot is
            # expected and not a warning condition.
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
        "instruction=%r", args.train_fps, args.execution_horizon, args.instruction,
    )

    # ---- Warmup the server once at the real image shape --------------------
    # The warmup also tells us H (chunk_size) from the server's response so
    # we can validate the inference_delay clamp range against it.
    chunk_size_from_server: Optional[int] = None
    try:
        state = read_state(left, right)
        log.info("Warming up server (timeout=%.0fs)...", args.warmup_timeout_s)
        _wu_actions, _wu_raw, _wu_rtt, _wu_meta = post_actions_rtc(
            args.server_url, top.grab(), cam_l.grab(), cam_r.grab(), state,
            args.instruction, args.num_steps, args.warmup_timeout_s,
            prev_chunk_left_over=None, inference_delay=0,
            execution_horizon=args.execution_horizon,
        )
        chunk_size_from_server = int(_wu_actions.shape[0])
        log.info(
            "Server warmup OK (rtt=%.0f ms, actions shape=%s, server_meta=%s)",
            _wu_rtt, _wu_actions.shape, _wu_meta,
        )
    except Exception as e:
        log.error("server warmup failed: %s. Continuing anyway.", e)

    # ---- Compute inference_delay bounds ------------------------------------
    # Paper's HARD constraint (Sec 2): d ≤ H - s. Rearranged: s ≤ H - d.
    # Per-request paper_s = H - len(leftover), so the binding form is:
    #     d ≤ len(leftover)
    # We compute this per kick-off (see _per_request_d_max below). The
    # STATIC bounds here are just safety floor/ceiling; the dynamic
    # per-request clamp does the real work. The static ceiling defaults
    # to (H - s_min) // 2 -- the stable steady-state fixed point assuming
    # d_prev ≈ d_current.
    H_chunk = chunk_size_from_server if chunk_size_from_server else 30
    auto_d_static = max(1, (H_chunk - int(args.execution_horizon)) // 2)
    d_max_static = int(args.rtc_max_inference_delay) if args.rtc_max_inference_delay is not None else auto_d_static
    d_min = max(0, int(args.rtc_min_inference_delay))
    if d_min > d_max_static:
        log.warning(
            "rtc_min_inference_delay=%d > rtc_max_inference_delay=%d; "
            "swapping the latter up", d_min, d_max_static,
        )
        d_max_static = d_min
    log.info(
        "inference_delay bounds: static=[%d, %d] ticks  per-request d_max "
        "is clamped to min(d_max_static, len(leftover))  (chunk_size H=%d, "
        "s_min=%d, fixed=%s, zero_delay=%s)",
        d_min, d_max_static, H_chunk, args.execution_horizon,
        args.rtc_inference_delay_fixed, args.rtc_zero_delay,
    )

    # ---- RTC state machine --------------------------------------------------
    fetcher = AsyncRTCFetcher(
        args.server_url, args.instruction, args.num_steps, args.timeout_s,
    )
    queue = ClientActionQueue()
    # Buffer of recent inference durations IN TICKS (paper Alg 1 line 23).
    # At each merge we push real_delay = tick_counter - inference_start_tick.
    delay_buffer: deque[int] = deque(maxlen=max(1, int(args.rtc_rtt_buffer_size)))

    def _predict_d_static() -> int:
        """d_pred for trigger-threshold computation (max(d_pred, s_min)
        per paper Alg 1's s = max(d, s_min)). Uses the STATIC d_max
        ceiling -- the per-request dynamic clamp only kicks in at
        kick_off when we know len(leftover)."""
        if args.rtc_zero_delay:
            return 0
        return _predict_inference_delay_ticks(
            delay_buffer, d_min, d_max_static,
            args.rtc_inference_delay_fixed,
        )

    def _predict_d(leftover_len: int) -> int:
        """Paper-faithful per-request d at kick_off time. Hard constraint
        d ≤ len(leftover) (equivalent to d ≤ H - paper_s where paper_s =
        H - len(leftover)). Also clamp by the static ceiling for sanity."""
        if args.rtc_zero_delay:
            return 0
        if leftover_len <= 0:
            return 0  # nothing to anchor; pure flow-matching
        d_max_dynamic = min(d_max_static, int(leftover_len))
        return _predict_inference_delay_ticks(
            delay_buffer, d_min, d_max_dynamic,
            args.rtc_inference_delay_fixed,
        )

    # Bootstrap: synchronous first chunk, no leftover, no discard.
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
    boot_chunk, boot_chunk_raw, boot_rtt_ms, boot_meta = fetcher.wait_for_result()
    # Seed the tick-delay buffer with the bootstrap RTT converted to ticks.
    # Bootstrap is synchronous so there's no `real_delay` in the main-loop
    # sense; ceil(boot_rtt_ms / dt_ms) is the closest tick-count we have.
    boot_delay_ticks = max(0, int(np.ceil(float(boot_rtt_ms) / (inner_dt * 1000.0))))
    delay_buffer.append(boot_delay_ticks)
    queue.replace(boot_chunk, boot_chunk_raw, real_delay=0)
    log.info(
        "Bootstrap chunk OK: shape=%s, raw_shape=%s, rtt=%.0fms (=%d ticks), "
        "queue_len=%d, meta=%s",
        boot_chunk.shape, boot_chunk_raw.shape, boot_rtt_ms,
        boot_delay_ticks, queue.qsize(), boot_meta,
    )

    # ---- Tick-driven control loop ------------------------------------------
    # Single thread. Every dt:
    #   * pop one action (or hold if underflow);
    #   * if a request is in flight and done -> merge with real_delay;
    #   * if no request in flight and exec horizon reached -> kick off next.
    stop_flag = {"stop": False}
    tick_counter = 0
    ticks_since_last_inference = 0
    inference_in_flight = False
    inference_start_tick = 0
    last_kickoff_d_predicted = 0
    last_action_played: Optional[np.ndarray] = None
    boundary_idx = 0
    last_rtt_ms = float(boot_rtt_ms)
    clipped_total = 0
    steps_total = 0
    underflow_ticks = 0
    last_paper_s_sent = 0  # paper_s = H - len(leftover) on the most recent kick_off
    # Per-cycle accounting (reset at every merge).
    ticks_played_cycle = 0
    ticks_held_cycle = 0
    state_at_last_merge: Optional[np.ndarray] = read_state(left, right)
    # Rolling summary state (lifetime; never reset).
    rtt_history: deque[float] = deque(maxlen=200)
    rtt_history.append(float(boot_rtt_ms))
    chunk_arm_span_history: deque[float] = deque(maxlen=200)
    cycles_completed = 0
    duty_played_total = 0
    duty_held_total = 0
    last_summary_tick = 0
    SUMMARY_EVERY_TICKS = max(1, int(round(5.0 / inner_dt)))  # ~5 s
    # Initial trigger threshold = max(d_pred_from_boot, s_min). Paper Alg 1:
    # s = max(d, s_min). Recomputed after every merge.
    trigger_threshold = max(_predict_d_static(), int(args.execution_horizon))
    log.info(
        "Initial trigger_threshold = max(d_pred=%d, s_min=%d) = %d ticks",
        _predict_d_static(), int(args.execution_horizon), trigger_threshold,
    )

    arm_idx = np.r_[0:6, 7:13]

    try:
        while not stop_flag["stop"]:
            step_start = time.perf_counter()
            tick_counter += 1
            ticks_since_last_inference += 1

            # Single CAN read per tick (state is used by the clip in
            # safe_command, by underflow-hold fallback, and by boundary
            # telemetry).
            state_now = read_state(left, right)

            # ---- Pop next action and command -----------------------------
            desired = queue.pop()
            if desired is None:
                underflow_ticks += 1
                ticks_held_cycle += 1
                duty_held_total += 1
                # Hold position: command the current state (zero motion).
                desired = state_now.copy().astype(np.float32)
                if underflow_ticks == 1 or underflow_ticks % 30 == 0:
                    log.warning(
                        "queue underflow (tick=%d, since-last-inference=%d): "
                        "holding state. Inference is slower than the "
                        "execution horizon allows.",
                        tick_counter, ticks_since_last_inference,
                    )
            else:
                ticks_played_cycle += 1
                duty_played_total += 1

            if args.dry_run:
                log.info("dry-run tick=%d action: %s", tick_counter,
                         np.array2string(desired, precision=3))
            else:
                _, n_clipped = safe_command(
                    left, right, state_now, desired,
                    args.max_step_rad, args.gripper_step,
                )
                clipped_total += n_clipped
                steps_total += 1
            last_action_played = desired.copy()

            # ---- Collect in-flight inference if it just finished ---------
            if inference_in_flight and fetcher.done():
                try:
                    new_chunk, new_chunk_raw, rtt_ms, meta = fetcher.wait_for_result()
                except Exception as e:  # noqa: BLE001
                    log.error("inference returned with error: %s", e)
                    break
                real_delay = tick_counter - inference_start_tick
                delay_buffer.append(int(real_delay))
                last_rtt_ms = float(rtt_ms)
                rtt_history.append(float(rtt_ms))
                clamped_delay = queue.replace(new_chunk, new_chunk_raw, real_delay)
                # Snapshot per-cycle counters BEFORE resetting.
                played_this_cycle = int(ticks_played_cycle)
                held_this_cycle = int(ticks_held_cycle)
                cycle_total = played_this_cycle + held_this_cycle
                duty_pct = (100.0 * played_this_cycle / cycle_total) if cycle_total > 0 else 0.0
                # State progress: how far the arm physically moved since the
                # last merge.
                if state_at_last_merge is not None:
                    state_progress_arm = float(np.max(np.abs(
                        state_now[arm_idx] - state_at_last_merge[arm_idx]
                    )))
                else:
                    state_progress_arm = 0.0
                state_at_last_merge = state_now.copy()
                # Reset cycle counters.
                ticks_played_cycle = 0
                ticks_held_cycle = 0
                ticks_since_last_inference = 0
                inference_in_flight = False
                cycles_completed += 1
                # Recompute the trigger threshold for the new cycle
                # (paper Alg 1: s = max(d, s_min)).
                trigger_threshold = max(_predict_d_static(), int(args.execution_horizon))

                # Pull server-side timing breakdown (item F).
                srv_pre = float(meta.get("dt_pre_ms", 0.0))
                srv_inf = float(meta.get("dt_inf_ms", 0.0))
                srv_post = float(meta.get("dt_post_ms", 0.0))

                # Per-cycle chunk diagnostics: arm-joint delta from CURRENT
                # state at chunk positions 0/5/10/15/29 (item C). Tells us
                # where the model wants the arm to go relative to "now".
                # NOTE: queue.queue is the post-discard array, so index 0
                # is the first action we'll play; "29" maps to the last
                # action of the original chunk.
                if queue.queue is not None and len(queue.queue) > 0:
                    qq = queue.queue
                    H_q = len(qq)
                    def _delta(i):
                        if i >= H_q: i = H_q - 1
                        return float(np.max(np.abs(qq[i][arm_idx] - state_now[arm_idx])))
                    deltas = [_delta(i) for i in (0, 5, 10, 15, H_q - 1)]
                    horizon_range = qq.max(axis=0) - qq.min(axis=0)
                    horizon_arm_span = float(max(
                        np.max(horizon_range[:6]), np.max(horizon_range[7:13]),
                    ))
                    chunk_arm_span_history.append(horizon_arm_span)
                    log.info(
                        "[chunk] #%d span=%.3f delta@0/5/10/15/last=%.3f/%.3f/%.3f/%.3f/%.3f rad",
                        cycles_completed, horizon_arm_span,
                        deltas[0], deltas[1], deltas[2], deltas[3], deltas[4],
                    )
                else:
                    horizon_arm_span = 0.0

                # Boundary line (items A + D + E folded together).
                # last_action_played = what we COMMANDED on the previous
                # tick. If we were underflowing right before merge, it's
                # = state_now (held). So `held_vs_a0` and `tail_vs_a0` are
                # the same in that case; the boundary log line shows both
                # so you can tell whether the recovery was smooth.
                if last_action_played is not None and queue.queue is not None and len(queue.queue) > 0:
                    a0_post = queue.queue[0]
                    state_vs_a0_arm = float(np.max(np.abs(a0_post[arm_idx] - state_now[arm_idx])))
                    tail_vs_a0_arm = float(np.max(np.abs(a0_post[arm_idx] - last_action_played[arm_idx])))
                    state_vs_a0_grip_l = float(abs(a0_post[6] - state_now[6]))
                    state_vs_a0_grip_r = float(abs(a0_post[13] - state_now[13]))
                    held_vs_a0_arm = state_vs_a0_arm if held_this_cycle > 0 else float('nan')
                    real_pred_delta = int(real_delay - last_kickoff_d_predicted)
                    # Absolute gripper positions: state[6,13] = current arm grippers,
                    # a0[6,13] = first commanded value, qq[-1][6,13] = end-of-chunk
                    # commanded value. So you can see "is the chunk telling the
                    # grippers to close (~0) or open (~1) right now and over the
                    # chunk?" instead of just the delta from state.
                    qq_end = queue.queue[-1]
                    boundary_idx += 1
                    log.info(
                        "[rtc-bound] #%d rtt=%.0f d_pred=%d real=%d Δ=%+d "
                        "paper_s=%d ply/hld=%d/%d duty=%.0f%% "
                        "s_vs_a0=%.3f t_vs_a0=%.3f h_vs_a0=%s "
                        "g_state(L,R)=%.2f,%.2f g_a0(L,R)=%.2f,%.2f "
                        "g_end(L,R)=%.2f,%.2f g_diff_a0=%.2f,%.2f "
                        "q=%d arm_prog=%.3f srv pre/inf/post=%.0f/%.0f/%.0f",
                        boundary_idx, rtt_ms, last_kickoff_d_predicted,
                        real_delay, real_pred_delta,
                        last_paper_s_sent, played_this_cycle, held_this_cycle,
                        duty_pct,
                        state_vs_a0_arm, tail_vs_a0_arm,
                        ("%.3f" % held_vs_a0_arm) if held_this_cycle > 0 else "n/a",
                        float(state_now[6]), float(state_now[13]),
                        float(a0_post[6]), float(a0_post[13]),
                        float(qq_end[6]), float(qq_end[13]),
                        state_vs_a0_grip_l, state_vs_a0_grip_r,
                        queue.qsize(), state_progress_arm,
                        srv_pre, srv_inf, srv_post,
                    )

            # ---- Kick off next inference if trigger threshold hit -------
            # Paper Alg 1: s = max(d, s_min). Threshold updated at every
            # merge from `trigger_threshold = max(_predict_d_static(),
            # args.execution_horizon)`.
            if (not inference_in_flight) and (ticks_since_last_inference >= trigger_threshold):
                kick_state = state_now  # reuse the read from earlier this tick
                kick_top = top.grab()
                kick_left = cam_l.grab()
                kick_right = cam_r.grab()

                # Leftover = currently unconsumed tail. lerobot's API arg
                # `execution_horizon` is actually the END of the fade region
                # in the prefix mask (NOT the paper's s). Setting it to
                # len(leftover) means the mask is:
                #   ones(d_pred), fade(d_pred -> len(leftover)),
                #   zeros(len(leftover) -> H)
                # which matches the paper's Eq. 5 with H - s_paper aligned
                # to the actual leftover boundary (paper-faithful given that
                # leftover already excludes the discarded prefix from the
                # prior merge).
                leftover_arr = queue.get_left_over()
                lerobot_exec_horizon = (
                    int(len(leftover_arr)) if leftover_arr is not None else 0
                )
                d_pred = _predict_d(lerobot_exec_horizon)
                last_kickoff_d_predicted = d_pred
                paper_s_this = H_chunk - lerobot_exec_horizon
                last_paper_s_sent = paper_s_this

                try:
                    fetcher.kick_off(
                        kick_state, kick_top, kick_left, kick_right,
                        prev_chunk_left_over=leftover_arr,
                        inference_delay=d_pred,
                        execution_horizon=lerobot_exec_horizon,
                        max_guidance_weight=args.rtc_max_guidance_weight,
                        schedule=args.rtc_schedule,
                        debug=args.rtc_debug,
                        seed=args.seed,
                    )
                except RuntimeError as e:
                    log.error("kick_off failed: %s", e)
                    break
                inference_in_flight = True
                inference_start_tick = tick_counter
                log.debug(
                    "kicked off /act: leftover_len=%d  paper_s=%d  "
                    "d_pred=%d  ticks_since_merge=%d  tick=%d",
                    lerobot_exec_horizon, paper_s_this, d_pred,
                    ticks_since_last_inference, tick_counter,
                )

            # ---- Tick rate -----------------------------------------------
            sleep_left = inner_dt - (time.perf_counter() - step_start)
            if sleep_left > 0:
                time.sleep(sleep_left)
            elif sleep_left < -0.050:
                log.warning(
                    "inner step overrun by %.1f ms (target %.1f ms)",
                    -sleep_left * 1000.0, inner_dt * 1000.0,
                )

            # ---- Rolling summary every ~5 s (item B) ---------------------
            if tick_counter - last_summary_tick >= SUMMARY_EVERY_TICKS:
                last_summary_tick = tick_counter
                elapsed_s = tick_counter * inner_dt
                # RTT percentiles over rolling window
                if len(rtt_history) > 0:
                    rtts = sorted(rtt_history)
                    n = len(rtts)
                    p50 = rtts[n // 2]
                    p95 = rtts[min(n - 1, int(0.95 * (n - 1)))]
                    rmax = rtts[-1]
                else:
                    p50 = p95 = rmax = 0.0
                # Duty cycle (lifetime)
                duty_total = duty_played_total + duty_held_total
                duty_pct_total = (100.0 * duty_played_total / duty_total) if duty_total > 0 else 0.0
                mean_span = (
                    sum(chunk_arm_span_history) / len(chunk_arm_span_history)
                    if len(chunk_arm_span_history) > 0 else 0.0
                )
                # Clip rate
                clip_pct = 0.0
                if steps_total > 0 and (args.max_step_rad > 0 or args.gripper_step > 0):
                    clip_pct = 100.0 * clipped_total / (STATE_DIM * steps_total)
                log.info(
                    "[summary] t=%.0fs cycles=%d rtt p50/p95/max=%.0f/%.0f/%.0f "
                    "mean_span=%.2f duty=%.0f%% underflows=%d clip=%.1f%% "
                    "d_pred=%d thresh=%d",
                    elapsed_s, cycles_completed, p50, p95, rmax,
                    mean_span, duty_pct_total, underflow_ticks, clip_pct,
                    _predict_d_static(), trigger_threshold,
                )

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
