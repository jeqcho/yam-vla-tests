"""π₀.₅ (Pi-0.5) — WebSocket + msgpack on port 8000 (openpi convention).

Wire-format reference:
    upstream openpi/src/openpi/serving/websocket_policy_server.py
    upstream openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py

Server schema (openpi WebsocketClientPolicy contract):
    on connect:  server sends one msgpack metadata frame
    per call:    client -> msgpack(obs);  server -> msgpack({"actions": ..., "server_timing": {...}})

Observation schema for `yam_pi05` (canonical openpi names — see model card):
    "state"   (14,) float32     [left_q(6), left_grip(1), right_q(6), right_grip(1)]
    "images"  {
      "base_0_rgb":        (3, H, W) uint8 RGB        <- top camera
      "left_wrist_0_rgb":  (3, H, W) uint8 RGB
      "right_wrist_0_rgb": (3, H, W) uint8 RGB
    }
    "prompt"  str

CRITICAL: CHW, not HWC. The agilex AlohaInputs fork is a passthrough
that SILENTLY zero-fills any canonical key it doesn't see. Sending
cam_high/cam_left_wrist (the human-friendly names) makes the model
run blind — no error, just bad actions. The role->canonical-key map
lives in configs/policy/pi05.yaml so future training forks can
override without code edits.

Response: model returns its padded internal shape (50, 32) with
defaults. Slice to (:, :14) here — cols 14..31 are training pad zeros.
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
    STATE_DIM,
    YamObservation,
)

log = logging.getLogger("yam_vla.policies.pi05")

# Default role -> CANONICAL openpi image key. The agilex fork's
# AlohaInputs.STANDARD_IMAGE_KEYS expects these literal strings.
DEFAULT_IMAGE_KEYS: dict[str, str] = {
    ImageRole.TOP:         "base_0_rgb",
    ImageRole.LEFT_WRIST:  "left_wrist_0_rgb",
    ImageRole.RIGHT_WRIST: "right_wrist_0_rgb",
}


class Pi05Policy(Policy):
    """openpi WebSocket client for `jeqcho/pi05-yam-bimanual`."""

    name = "pi05"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        image_keys: dict[str, str] | None = None,
        api_key: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.image_keys = dict(image_keys or DEFAULT_IMAGE_KEYS)
        self.api_key = api_key
        self._wcp_cls: Any = None
        self._client: Any = None

        # openpi-client may live in any of several venvs; defer the import.
        try:
            from openpi_client import websocket_client_policy as _wcp
            self._wcp_cls = _wcp.WebsocketClientPolicy
        except Exception:  # pragma: no cover -- env-dependent
            self._wcp_cls = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._wcp_cls is None:
            raise RuntimeError(
                "pi05 backend requires `openpi-client` installed in the client venv. Install with:\n"
                "  uv pip install 'openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git"
                "#subdirectory=packages/openpi-client'"
            )
        kw: dict[str, Any] = {}
        if self.api_key:
            kw["api_key"] = self.api_key
        self._client = self._wcp_cls(host=self.host, port=self.port, **kw)
        return self._client

    # ----- lifecycle -----

    def info(self, timeout_s: float = 3.0) -> ServerInfo:
        del timeout_s  # openpi-client doesn't expose connect-timeout via the wrapper
        client = self._ensure_client()
        meta: dict = {}
        for attr in ("get_server_metadata", "server_metadata"):
            if hasattr(client, attr):
                val = getattr(client, attr)
                meta = val() if callable(val) else (val or {})
                break
        return ServerInfo(
            backend=self.name,
            transport=f"ws://{self.host}:{self.port}",
            model_id=(meta or {}).get("model_id"),
            action_horizon_hint=(meta or {}).get("action_horizon", 16),
            raw=meta or {},
        )

    # ----- inference -----

    @staticmethod
    def _hwc_to_chw(img: np.ndarray) -> np.ndarray:
        """HWC uint8 RGB -> CHW uint8 RGB (msgpack-friendly contiguous copy)."""
        a = np.asarray(img)
        # core's YamObservation has already validated HWC uint8 by the time we
        # get here, but this keeps the function safe to use standalone too.
        return np.transpose(a, (2, 0, 1)).copy()

    def predict(
        self,
        obs: YamObservation,
        *,
        timeout_s: float = 5.0,
        **opts: Any,
    ) -> Prediction:
        del timeout_s, opts  # openpi-client's .infer() is blocking; no knobs

        client = self._ensure_client()
        payload = {
            "state":  obs.state,
            "images": {
                self.image_keys[role]: self._hwc_to_chw(obs.images[role])
                for role in ImageRole.ALL
            },
            "prompt": obs.prompt,
        }

        t0 = time.perf_counter()
        out = client.infer(payload)
        rtt_ms = (time.perf_counter() - t0) * 1000.0

        if not isinstance(out, dict) or "actions" not in out:
            keys = list(out.keys()) if isinstance(out, dict) else type(out).__name__
            raise RuntimeError(f"pi05 response missing 'actions': {keys}")

        actions = np.asarray(out["actions"], dtype=np.float32)
        # openpi-client occasionally returns (1, N, D); strip the batch dim.
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]

        # Strip the model's training pad. With defaults that's (50, 32) ->
        # slice the first 14 cols. The agilex AlohaOutputs only does this
        # when adapt_to_pi=True, which ALSO applies trossen-aloha joint
        # flips that are wrong for YAM. So we register yam_pi05 with
        # adapt_to_pi=False and strip the pad client-side here.
        if actions.ndim == 2 and actions.shape[1] > STATE_DIM:
            actions = actions[:, :STATE_DIM]

        server_info = {k: v for k, v in out.items() if k != "actions"}
        return Prediction(
            actions=actions,
            rtt_ms=rtt_ms,
            horizon=actions.shape[0],
            server_info=server_info,
        )
