"""Per-attempt inference + execution loop, agnostic to the policy backend.

The loop:
    1. read cameras + arm state -> YamObservation
    2. policy.predict(obs)       -> Prediction (N, 14) absolute
    3. safe-clip the chunk against the current state (per-tick)
    4. command arms; sleep dt; repeat until horizon_stride consumed
    5. while executing, kick off the next policy.predict() in a thread
       so transport latency overlaps with motion

This replaces what `_archive/molmoact2-setup/scripts/yam_repl.run_one_attempt`
did, but is backend-agnostic via the `Policy` ABC instead of the
hardcoded MolmoAct `post_actions` call.

Hardware bits (cameras, arms, safety_clip) are imported from the legacy
shim — see `yam_vla.core.legacy` for the rationale.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from yam_vla.core.observation import ImageRole, YamObservation
from yam_vla.core.policy import Policy, Prediction

log = logging.getLogger("yam_vla.core.runner")


# ---------------------------------------------------------------------------
# Async inference (overlap policy.predict with execution)
# ---------------------------------------------------------------------------

class AsyncPolicyInference:
    """Run `Policy.predict(obs)` in a background thread.

    Equivalent to the legacy `AsyncInferenceFetcher` but parameterized
    on the new Policy ABC, so it works with all three backends. Single
    in-flight slot — kicking off while a previous call hasn't been
    awaited raises.

    Usage:
        ap = AsyncPolicyInference(policy, timeout_s=5.0, opts={"num_steps": 10})
        ap.kick_off(obs)
        # ... execute current chunk ...
        pred = ap.wait()
    """
    def __init__(self, policy: Policy, *, timeout_s: float = 5.0,
                 opts: Optional[dict] = None):
        self.policy = policy
        self.timeout_s = timeout_s
        self.opts = opts or {}
        self._thread: Optional[threading.Thread] = None
        self._pred: Optional[Prediction] = None
        self._err: Optional[BaseException] = None

    def kick_off(self, obs: YamObservation) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(
                "AsyncPolicyInference.kick_off while previous call in flight"
            )
        self._pred = None
        self._err = None

        def _worker() -> None:
            try:
                self._pred = self.policy.predict(
                    obs, timeout_s=self.timeout_s, **self.opts,
                )
            except BaseException as e:  # noqa: BLE001
                self._err = e

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def wait(self) -> Prediction:
        if self._thread is None:
            raise RuntimeError("AsyncPolicyInference.wait with no in-flight request")
        self._thread.join()
        self._thread = None
        if self._err is not None:
            raise self._err
        assert self._pred is not None
        return self._pred


# ---------------------------------------------------------------------------
# Per-attempt result
# ---------------------------------------------------------------------------

@dataclass
class AttemptStats:
    """Summary stats for one attempt — what the eval CSV writer needs."""
    chunks:           int   = 0
    duration_s:       float = 0.0
    rtt_ms_mean:      float = 0.0
    rtt_ms_p95:       float = 0.0
    rtt_ms_max:       float = 0.0
    horizon_arm_mean: float = 0.0
    horizon_arm_max:  float = 0.0
    clip_rate:        float = 0.0          # frac of dim-steps that hit the safety clip
    raw_rtts_ms:      list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Observation builder — small helper so eval/REPL don't reinvent it
# ---------------------------------------------------------------------------

def build_observation(top_img: np.ndarray, left_img: np.ndarray,
                      right_img: np.ndarray, state: np.ndarray,
                      prompt: str) -> YamObservation:
    """Construct a YamObservation from the canonical camera order + state."""
    return YamObservation(
        images={
            ImageRole.TOP:         np.ascontiguousarray(top_img),
            ImageRole.LEFT_WRIST:  np.ascontiguousarray(left_img),
            ImageRole.RIGHT_WRIST: np.ascontiguousarray(right_img),
        },
        state=state,
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Convenience type aliases for the eval harness signatures
# ---------------------------------------------------------------------------

# A callable that reads 3 RGB frames (top, left, right) -- typically
# constructed from yam_vla.core.legacy.make_camera streams.
ReadFramesFn = Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]

# A callable that reads the current 14-D arm state.
ReadStateFn = Callable[[], np.ndarray]

# A callable that commands the arms with one (14,) action.
ExecActionFn = Callable[[np.ndarray], None]


__all__ = [
    "AsyncPolicyInference",
    "AttemptStats",
    "build_observation",
    "ReadFramesFn",
    "ReadStateFn",
    "ExecActionFn",
]
