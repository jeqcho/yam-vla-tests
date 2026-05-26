"""MolmoAct2 — HTTP + json_numpy on port 8202.

Wire-format reference:
    _archive/molmoact2-setup/molmoact2/examples/yam/host_server_yam.py

Server schema:
    POST /act  json_numpy body {top_cam, left_cam, right_cam,
                                state(14,), instruction, num_steps?, timestamp?}
    Reply      json_numpy body {actions: (N, 14) absolute joint targets,
                                dt_ms, ...}
    GET  /act  JSON health probe

Camera role -> wire-key map: see configs/policy/molmoact2.yaml.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from yam_vla.core import (
    ImageRole,
    Policy,
    Prediction,
    ServerInfo,
    YamObservation,
)

log = logging.getLogger("yam_vla.policies.molmoact2")

# Default role -> server key mapping. Overridable from YAML.
DEFAULT_IMAGE_KEYS: dict[str, str] = {
    ImageRole.TOP:         "top_cam",
    ImageRole.LEFT_WRIST:  "left_cam",
    ImageRole.RIGHT_WRIST: "right_cam",
}


class MolmoAct2Policy(Policy):
    """HTTP + json_numpy client for `allenai/MolmoAct2-BimanualYAM`."""

    name = "molmoact2"

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8202/act",
        *,
        image_keys: dict[str, str] | None = None,
        default_num_steps: int = 10,
    ):
        self.server_url = server_url
        self.image_keys = dict(image_keys or DEFAULT_IMAGE_KEYS)
        self.default_num_steps = default_num_steps
        self._patched = False  # json_numpy.patch() applied lazily

    def _ensure_patched(self) -> None:
        if self._patched:
            return
        import json_numpy
        json_numpy.patch()  # idempotent; monkey-patches stdlib json
        self._patched = True

    # ----- lifecycle -----

    def info(self, timeout_s: float = 3.0) -> ServerInfo:
        self._ensure_patched()
        import requests
        r = requests.get(self.server_url, timeout=timeout_s)
        r.raise_for_status()
        meta = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return ServerInfo(
            backend=self.name,
            transport=self.server_url,
            model_id=meta.get("repo_id"),
            action_horizon_hint=meta.get("action_horizon"),
            raw=meta,
        )

    # ----- inference -----

    def predict(
        self,
        obs: YamObservation,
        *,
        timeout_s: float = 5.0,
        **opts: Any,
    ) -> Prediction:
        self._ensure_patched()
        import json_numpy
        import requests

        num_steps = int(opts.get("num_steps", self.default_num_steps))

        # Build payload via the configured role -> wire-key map.
        payload: dict[str, Any] = {
            self.image_keys[ImageRole.TOP]:         obs.images[ImageRole.TOP],
            self.image_keys[ImageRole.LEFT_WRIST]:  obs.images[ImageRole.LEFT_WRIST],
            self.image_keys[ImageRole.RIGHT_WRIST]: obs.images[ImageRole.RIGHT_WRIST],
            "state":       obs.state,           # core already validates (14,) float32
            "instruction": obs.prompt,
            "num_steps":   num_steps,
            "timestamp":   time.time(),
        }

        body = json_numpy.dumps(payload)
        t0 = time.perf_counter()
        resp = requests.post(
            self.server_url, data=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout_s,
        )
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        out = json_numpy.loads(resp.text)
        if "actions" not in out:
            raise RuntimeError(
                f"molmoact2 response missing 'actions': keys={list(out.keys())}"
            )

        actions = np.asarray(out["actions"], dtype=np.float32)
        # MolmoAct returns (N, 14) absolute targets directly — no conversion needed.
        server_info = {k: v for k, v in out.items() if k != "actions"}
        return Prediction(
            actions=actions,
            rtt_ms=rtt_ms,
            horizon=actions.shape[0],
            server_info=server_info,
        )
