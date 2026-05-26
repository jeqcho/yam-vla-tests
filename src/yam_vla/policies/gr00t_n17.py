"""GR00T N1.7 — ZeroMQ REQ/REP + msgpack-numpy on port 5556.

Wire-format reference:
    _archive/grootn1.7-exploration/Isaac-GR00T/gr00t/policy/server_client.py

Server endpoints (RPC over msgpack):
    ping              -> {"backend": ..., "model": ..., "modality_config": ...}
    get_action        -> (action_dict, info)
    get_modality_config -> ModalityConfig (per-checkpoint)
    reset             -> ack
    kill              -> ack

Observation schema (leading (B=1, T=1) dims):
    obs["video"][<role_key>]            (1, 1, H, W, 3) uint8 RGB
    obs["state"]["left_arm"]            (1, 1, 6)       float32 rad
    obs["state"]["left_gripper"]        (1, 1, 1)       float32 [0,1]
    obs["state"]["right_arm"]           (1, 1, 6)
    obs["state"]["right_gripper"]       (1, 1, 1)
    obs["language"][<lang_key>]         [[str]]         (1, 1)

Response schema (16-step horizon):
    actions["left_arm"]      (1, 16, 6)   ABSOLUTE joint targets (rad)
    actions["left_gripper"]  (1, 16, 1)   ABSOLUTE
    actions["right_arm"]     (1, 16, 6)   ABSOLUTE
    actions["right_gripper"] (1, 16, 1)   ABSOLUTE

CRITICAL: the server already calls `StateActionProcessor.unapply_action` —
the wire payload is ALREADY absolute for arms, despite training-time
relative representation. Do NOT add the current state on top here.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from yam_vla.core import (
    ImageRole,
    Policy,
    Prediction,
    ServerInfo,
    YamObservation,
    YamStateCodec,
)

log = logging.getLogger("yam_vla.policies.gr00t_n17")

# Default role -> server video-key mapping; overridable from YAML.
DEFAULT_VIDEO_KEYS: dict[str, str] = {
    ImageRole.TOP:         "top",
    ImageRole.LEFT_WRIST:  "left",
    ImageRole.RIGHT_WRIST: "right",
}

DEFAULT_LANG_KEY = "annotation.human.task_description"


class Gr00tN17Policy(Policy):
    """ZeroMQ + msgpack-numpy client for `jeqcho/gr00t-n17-yam-bimanual`."""

    name = "gr00t-n17"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        *,
        video_keys: dict[str, str] | None = None,
        language_key: str = DEFAULT_LANG_KEY,
        zmq_timeout_ms: int = 30_000,
    ):
        self.host = host
        self.port = port
        self.video_keys = dict(video_keys or DEFAULT_VIDEO_KEYS)
        self.language_key = language_key
        self.zmq_timeout_ms = zmq_timeout_ms

        # Lazy state — set on first call so import doesn't require pyzmq.
        self._zmq: Any = None
        self._mnp: Any = None
        self._ctx: Any = None
        self._sock: Any = None

    # ----- transport plumbing -----

    def _ensure_socket(self) -> Any:
        if self._sock is not None:
            return self._sock
        import msgpack_numpy as _mnp  # noqa
        import zmq as _zmq  # noqa
        self._zmq = _zmq
        self._mnp = _mnp
        self._ctx = _zmq.Context.instance()
        self._sock = self._ctx.socket(_zmq.REQ)
        self._sock.setsockopt(_zmq.RCVTIMEO, self.zmq_timeout_ms)
        self._sock.setsockopt(_zmq.SNDTIMEO, self.zmq_timeout_ms)
        self._sock.connect(f"tcp://{self.host}:{self.port}")
        return self._sock

    def _call(self, endpoint: str, data: Optional[dict] = None,
              requires_input: bool = True) -> Any:
        sock = self._ensure_socket()
        req: dict = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        try:
            sock.send(self._mnp.packb(req))
            msg = sock.recv()
        except self._zmq.error.Again:
            # REQ socket stuck waiting for a reply that won't come — recreate.
            log.warning("zmq recv timeout on %s; recreating socket", endpoint)
            self._sock = None
            raise
        resp = self._mnp.unpackb(msg, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"gr00t server error: {resp['error']}")
        return resp

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            finally:
                self._sock = None

    # ----- lifecycle -----

    def info(self, timeout_s: float = 3.0) -> ServerInfo:
        sock = self._ensure_socket()
        old_timeout = sock.getsockopt(self._zmq.RCVTIMEO)
        sock.setsockopt(self._zmq.RCVTIMEO, int(timeout_s * 1000))
        try:
            resp = self._call("ping", requires_input=False)
        finally:
            sock.setsockopt(self._zmq.RCVTIMEO, old_timeout)
        if not isinstance(resp, dict):
            resp = {"ping": resp}
        return ServerInfo(
            backend=self.name,
            transport=f"zmq tcp://{self.host}:{self.port}",
            model_id=resp.get("model_id") or resp.get("repo_id"),
            action_horizon_hint=resp.get("action_horizon", 16),
            raw=resp,
        )

    def reset(self, options: dict | None = None) -> None:
        try:
            self._call("reset", data=options or {})
        except Exception as e:
            log.debug("gr00t reset() ignored: %s", e)

    # ----- inference -----

    @staticmethod
    def _bt1(x: np.ndarray) -> np.ndarray:
        """Add leading (B=1, T=1) dims expected by the GR00T server."""
        return np.expand_dims(np.expand_dims(np.asarray(x), 0), 0)

    def predict(
        self,
        obs: YamObservation,
        *,
        timeout_s: float = 5.0,
        **opts: Any,
    ) -> Prediction:
        del opts  # GR00T's horizon is checkpoint-baked; no per-call knobs

        # YAM state -> 4-key per-arm split (codec owns this layout).
        split = YamStateCodec.split(obs.state)

        observation = {
            "state": {
                k: self._bt1(v.astype(np.float32))
                for k, v in split.items()
            },
            "video": {
                self.video_keys[role]: self._bt1(obs.images[role])
                for role in ImageRole.ALL
            },
            "language": {
                self.language_key: [[obs.prompt]],
            },
        }

        sock = self._ensure_socket()
        old_timeout = sock.getsockopt(self._zmq.RCVTIMEO)
        sock.setsockopt(self._zmq.RCVTIMEO, int(timeout_s * 1000))
        t0 = time.perf_counter()
        try:
            action_dict, server_info = self._call(
                "get_action", {"observation": observation, "options": None},
            )
        finally:
            sock.setsockopt(self._zmq.RCVTIMEO, old_timeout)
        rtt_ms = (time.perf_counter() - t0) * 1000.0

        # Strip the (B=1, ...) leading dim from each stream; codec stitches.
        def _strip(arr: np.ndarray) -> np.ndarray:
            a = np.asarray(arr, dtype=np.float32)
            return a[0] if (a.ndim == 3 and a.shape[0] == 1) else a

        actions = YamStateCodec.stitch(
            _strip(action_dict["left_arm"]),
            _strip(action_dict["left_gripper"]),
            _strip(action_dict["right_arm"]),
            _strip(action_dict["right_gripper"]),
        )
        # All four streams are ALREADY absolute (server applied unapply_action).
        # Do NOT add state — that was the "2x dangerous motion" bug.

        return Prediction(
            actions=actions,
            rtt_ms=rtt_ms,
            horizon=actions.shape[0],
            server_info=server_info if isinstance(server_info, dict) else {"_raw": server_info},
        )
