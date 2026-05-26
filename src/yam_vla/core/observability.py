"""Optional Rerun observability: per-tick cameras + state + actions to .rrd.

Rerun is an external SDK (`pip install rerun-sdk`). All functions here
are no-ops when `--rerun` wasn't passed, so eval code can call them
unconditionally:

    from yam_vla.core.observability import RerunRecorder

    rr = RerunRecorder(enabled=args.rerun, save_path=args.rerun_save)
    rr.log_observation(t_s, top_img, left_img, right_img, state)
    rr.log_inference(t_s, actions, executed_idx, rtt_ms, horizon_arm_span)

If enabled=False, the methods short-circuit immediately.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("yam_vla.observability")


class RerunRecorder:
    """No-op-friendly wrapper around rerun-sdk.

    Construct with enabled=False and all .log_* methods are zero-cost
    early returns. Construct with enabled=True and they stream to a
    live viewer (and optionally a .rrd file).
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        save_path: Optional[str] = None,
        connect: Optional[str] = None,
        app_id: str = "yam_inference",
    ):
        self.enabled = enabled or (save_path is not None)
        self.save_path = save_path
        self.connect = connect
        self._rr = None

        if not self.enabled:
            return

        try:
            import rerun as rr
        except ImportError:
            log.error(
                "Rerun requested but rerun-sdk not installed. Install via:\n"
                "  uv pip install rerun-sdk"
            )
            self.enabled = False
            return

        try:
            rr.init(app_id, spawn=(self.connect is None))
            if self.connect:
                host, _, port = self.connect.partition(":")
                rr.connect_grpc(f"rerun+http://{host}:{port}/proxy")
                log.info("Rerun: connected to viewer at %s", self.connect)
            else:
                log.info("Rerun: spawned local viewer")
            if self.save_path:
                rr.save(self.save_path)
                log.info("Rerun: saving recording to %s", self.save_path)
            self._rr = rr
        except Exception as e:
            log.error("Rerun init failed: %s. Continuing without it.", e)
            self.enabled = False

    def log_observation(
        self,
        t_s: float,
        top_img: np.ndarray,
        left_img: np.ndarray,
        right_img: np.ndarray,
        state: np.ndarray,
    ) -> None:
        """Log 3 camera frames + 14-D state at one timestep."""
        if not self.enabled or self._rr is None:
            return
        rr = self._rr
        rr.set_time("time", duration=t_s)
        rr.log("cam/top",   rr.Image(top_img))
        rr.log("cam/left",  rr.Image(left_img))
        rr.log("cam/right", rr.Image(right_img))
        for i in range(6):
            rr.log(f"state/left/j{i}",  rr.Scalars(float(state[i])))
            rr.log(f"state/right/j{i}", rr.Scalars(float(state[i + 7])))
        rr.log("state/left/gripper",  rr.Scalars(float(state[6])))
        rr.log("state/right/gripper", rr.Scalars(float(state[13])))

    def log_inference(
        self,
        t_s: float,
        actions: np.ndarray,
        executed_idx: int,
        rtt_ms: float,
        horizon_arm_span: float,
    ) -> None:
        """Log per-query inference outputs: rtt + horizon span + executed action."""
        if not self.enabled or self._rr is None:
            return
        rr = self._rr
        rr.set_time("time", duration=t_s)
        rr.log("metrics/rtt_ms",           rr.Scalars(float(rtt_ms)))
        rr.log("metrics/horizon_arm_span", rr.Scalars(float(horizon_arm_span)))
        a = actions[executed_idx]
        for i in range(6):
            rr.log(f"action/left/j{i}",  rr.Scalars(float(a[i])))
            rr.log(f"action/right/j{i}", rr.Scalars(float(a[i + 7])))
        rr.log("action/left/gripper",  rr.Scalars(float(a[6])))
        rr.log("action/right/gripper", rr.Scalars(float(a[13])))


__all__ = ["RerunRecorder"]
